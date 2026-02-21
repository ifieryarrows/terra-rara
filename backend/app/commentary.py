"""
AI Commentary Generator using OpenRouter API.
Generates human-readable market analysis from FinBERT + XGBoost results.
"""

import logging
from typing import Optional
from datetime import datetime

from .settings import get_settings
from .openrouter_client import OpenRouterError, create_chat_completion

logger = logging.getLogger(__name__)


def _extract_chat_message_content(data: dict) -> str:
    """Extract text content from OpenRouter chat completion response."""
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts).strip()
    return ""


def _build_commentary_template_fallback(
    current_price: float,
    predicted_price: float,
    predicted_return: float,
    sentiment_index: float,
    sentiment_label: str,
    top_influencers: list[dict],
    news_count: int,
) -> str:
    """Deterministic fallback commentary used when LLM is unavailable."""
    direction = "upside" if predicted_return >= 0 else "downside"
    top_driver_names = [inf.get("feature", "unknown_driver") for inf in top_influencers[:3]]
    while len(top_driver_names) < 3:
        top_driver_names.append("unknown_driver")

    return "\n".join([
        "Risks:",
        f"1. Model indicates {direction} uncertainty around the next-day move ({predicted_return * 100:.2f}%).",
        f"2. Sentiment regime is {sentiment_label} with score {sentiment_index:.3f}, which can reverse quickly.",
        f"3. News sample size ({news_count}) may be insufficient for stable short-horizon inference.",
        "Opportunities:",
        f"1. Predicted price path implies a move from ${current_price:.4f} to ${predicted_price:.4f}.",
        f"2. Feature signal concentration around `{top_driver_names[0]}` can support tactical monitoring.",
        f"3. Secondary drivers `{top_driver_names[1]}` and `{top_driver_names[2]}` provide confirmation checkpoints.",
        f"Summary: Current model inputs suggest a cautious {direction} bias with elevated uncertainty.",
        "Bias warning: This view is model-driven and sensitive to news mix, data latency, and feature drift.",
        "This is NOT financial advice.",
    ])


async def determine_ai_stance(commentary: str) -> str:
    """
    Have the AI analyze its own commentary to determine market stance.
    
    Args:
        commentary: The generated commentary text
        
    Returns:
        BULLISH, NEUTRAL, or BEARISH
    """
    settings = get_settings()
    
    if not commentary:
        return "NEUTRAL"
    
    # First try API-based stance detection
    if settings.openrouter_api_key:
        prompt = f"""Analyze the following market commentary and determine the overall market stance.
Respond with ONLY one word: BULLISH, NEUTRAL, or BEARISH.

Commentary:
{commentary}

Your response (one word only):"""
        
        try:
            data = await create_chat_completion(
                api_key=settings.openrouter_api_key,
                model=settings.resolved_commentary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,
                timeout_seconds=30.0,
                max_retries=settings.openrouter_max_retries,
                rpm=settings.openrouter_rpm,
                fallback_models=settings.openrouter_fallback_models_list,
            )
            stance = _extract_chat_message_content(data).upper()

            # Validate response
            if stance in ["BULLISH", "NEUTRAL", "BEARISH"]:
                logger.info(f"AI stance determined: {stance}")
                return stance
            logger.warning(f"Invalid AI stance response: '{stance}', using keyword fallback")
        except OpenRouterError as e:
            logger.warning(f"AI stance detection failed via OpenRouter: {e}, using keyword fallback")
        except Exception as e:
            logger.warning(f"AI stance detection failed: {e}, using keyword fallback")
    
    # Fallback: keyword-based stance detection
    return _detect_stance_from_keywords(commentary)


def _detect_stance_from_keywords(text: str) -> str:
    """
    Detect market stance from commentary text using keyword matching.
    
    Simple heuristic based on bullish/bearish keyword counts.
    """
    text_lower = text.lower()
    
    bullish_keywords = [
        "bullish", "upside", "upward", "positive", "gain", "rise", "rising",
        "higher", "growth", "optimistic", "rally", "surge", "strength"
    ]
    bearish_keywords = [
        "bearish", "downside", "downward", "negative", "decline", "fall", "falling",
        "lower", "weakness", "pessimistic", "drop", "slump", "pressure"
    ]
    
    bullish_count = sum(1 for kw in bullish_keywords if kw in text_lower)
    bearish_count = sum(1 for kw in bearish_keywords if kw in text_lower)
    
    if bullish_count > bearish_count + 1:
        stance = "BULLISH"
    elif bearish_count > bullish_count + 1:
        stance = "BEARISH"
    else:
        stance = "NEUTRAL"
    
    logger.info(f"Keyword stance detection: bullish={bullish_count}, bearish={bearish_count} -> {stance}")
    return stance

async def generate_commentary(
    current_price: float,
    predicted_price: float,
    predicted_return: float,
    sentiment_index: float,
    sentiment_label: str,
    top_influencers: list[dict],
    news_count: int = 0,
) -> Optional[str]:
    """
    Generate AI commentary using OpenRouter API.
    
    Args:
        current_price: Current copper price
        predicted_price: Model's predicted price for tomorrow
        predicted_return: Expected return percentage
        sentiment_index: Current sentiment score (-1 to 1)
        sentiment_label: Bullish/Bearish/Neutral
        top_influencers: List of top feature influencers
        news_count: Number of news articles analyzed
        
    Returns:
        AI-generated commentary or None if failed
    """
    settings = get_settings()
    
    fallback_commentary = _build_commentary_template_fallback(
        current_price=current_price,
        predicted_price=predicted_price,
        predicted_return=predicted_return,
        sentiment_index=sentiment_index,
        sentiment_label=sentiment_label,
        top_influencers=top_influencers,
        news_count=news_count,
    )

    if not settings.openrouter_api_key:
        logger.warning("OpenRouter API key not configured, using template commentary fallback")
        return fallback_commentary
    
    # Build the prompt
    influencers_text = "\n".join([
        f"  - {inf.get('feature', 'Unknown')}: {inf.get('importance', 0)*100:.1f}%"
        for inf in top_influencers[:5]
    ])
    
    change_direction = "increase" if predicted_return > 0 else "decrease"
    
    prompt = f"""Using ONLY the data below, produce an investor-facing copper market note.

DATA PROVIDED:
- Current Price: ${current_price:.4f}
- Model-Predicted Return: {change_direction} of {abs(predicted_return*100):.2f}%
- Predicted Price: ${predicted_price:.4f}
- Market Sentiment: {sentiment_label} (Score: {sentiment_index:.3f}, range -1 to 1)
- News Volume: {news_count} articles analyzed
- Top Drivers/Influencers (XGBoost feature importance):
{influencers_text}

Write 3-5 short paragraphs (200-260 words). Use the exact numeric inputs as given. If signals disagree, explicitly note the tension and what would resolve it. End with: This is NOT financial advice."""

    # Comprehensive system prompt with strategic context
    system_prompt = """You are a commodity and metals market analyst specializing in copper. Convert ONLY the user-provided inputs (e.g., current price, model-predicted price/return, sentiment score/label, news volume, and any listed drivers or influencers) into a clear, investor-facing market note. Do not introduce external facts, headlines, macro events, supply disruptions, policy actions, inventories, or company news unless the user explicitly provided them. If something is not provided, do not assume it - state the uncertainty and explain what additional information would matter.

Core principles:
- Data-grounded: every directional claim must trace back to an explicit input (price/return, sentiment, news count, named drivers).
- Disciplined uncertainty: treat all forecasts as model-based indications, not certainties; avoid words like "will," "guaranteed," or "certain."
- No trading instructions: do not tell the reader to buy/sell/short/enter/exit; avoid leverage/risk-on language.
- Plain, professional English: concise, readable, non-sensational; no Turkish text anywhere in the response.

Strategic context (apply ONLY when directly triggered by the user's inputs/news/drivers; otherwise ignore):
- M&A signals: if the user references rising free cash flow, bid rumors, or acquisition chatter around BHP, Rio Tinto, or Glencore, interpret it as potentially supportive for mid-tier copper targets (examples: FIL, LUN, IVN) consistent with "buying is cheaper than building," but keep claims conditional.
- Safe-jurisdiction premium: if the user references disruptions or political risk in Chile/Peru, frame it as relatively supportive for Australia/Canada-based miners (e.g., BHP, TECK) versus higher-risk jurisdictions, without overstating.
- China factor: if the user references strength in Zijin Mining (2899.HK), interpret it as possible evidence of Chinese stockpiling and describe it as a lagging, confirmation-type signal for HG=F - not an immediate trigger.
- Grade decline: if the user references ore grade decline, frame it as conditionally supportive for leaching/processing technology leaders (e.g., Freeport/FCX), keeping the logic modest.

Conflict handling:
- If signals disagree (e.g., bullish sentiment but negative predicted return), explicitly note the tension, outline 1-2 plausible reconciliations, and state what would resolve it (e.g., next-day price action relative to key levels, change in sentiment, or updated news flow - only in general terms).

Output requirements:
- Write 3-5 short paragraphs, aiming for a more detailed but still readable note (about 200-260 words unless the user specifies otherwise).
- Use the exact numeric inputs as given; do not recalculate or "correct" them.
- If referencing real-world current events beyond the provided inputs, you must include credible citations with links; otherwise do not cite and do not imply you looked up anything.
- End with this exact line on its own: This is NOT financial advice."""

    try:
        data = await create_chat_completion(
            api_key=settings.openrouter_api_key,
            model=settings.resolved_commentary_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=700,
            temperature=0.6,
            timeout_seconds=30.0,
            max_retries=settings.openrouter_max_retries,
            rpm=settings.openrouter_rpm,
            fallback_models=settings.openrouter_fallback_models_list,
            referer="https://copper-mind.vercel.app",
            title="CopperMind AI Analysis",
        )
        commentary = _extract_chat_message_content(data)
        if commentary:
            logger.info(f"AI commentary generated successfully ({len(commentary)} chars)")
            return commentary.strip()

        logger.warning("Empty response from OpenRouter, using template commentary fallback")
        return fallback_commentary
    except OpenRouterError as e:
        logger.warning("OpenRouter commentary failed: %s. Using template fallback.", e)
        return fallback_commentary
    except Exception as e:
        logger.error(f"Failed to generate AI commentary: {e}")
        return fallback_commentary


def save_commentary_to_db(
    session,
    symbol: str,
    commentary: str,
    current_price: float,
    predicted_price: float,
    predicted_return: float,
    sentiment_label: str,
    ai_stance: str = "NEUTRAL",
) -> None:
    """
    Save generated commentary to database (upsert).
    Called after pipeline completion.
    """
    from .models import AICommentary
    
    settings = get_settings()
    
    # Try to find existing record
    existing = session.query(AICommentary).filter(AICommentary.symbol == symbol).first()
    
    if existing:
        # Update existing
        existing.commentary = commentary
        existing.current_price = current_price
        existing.predicted_price = predicted_price
        existing.predicted_return = predicted_return
        existing.sentiment_label = sentiment_label
        existing.ai_stance = ai_stance
        existing.generated_at = datetime.utcnow()
        existing.model_name = settings.resolved_commentary_model
        logger.info(f"Updated AI commentary for {symbol} (stance: {ai_stance})")
    else:
        # Create new
        new_commentary = AICommentary(
            symbol=symbol,
            commentary=commentary,
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            sentiment_label=sentiment_label,
            ai_stance=ai_stance,
            model_name=settings.resolved_commentary_model,
        )
        session.add(new_commentary)
        logger.info(f"Created new AI commentary for {symbol} (stance: {ai_stance})")
    
    session.commit()


def get_commentary_from_db(session, symbol: str) -> Optional[dict]:
    """
    Get stored commentary from database.
    Returns dict with commentary and metadata, or None if not found.
    """
    from .models import AICommentary
    
    record = session.query(AICommentary).filter(AICommentary.symbol == symbol).first()
    
    if record:
        return {
            "commentary": record.commentary,
            "generated_at": record.generated_at.isoformat() if record.generated_at else None,
            "current_price": record.current_price,
            "predicted_price": record.predicted_price,
            "predicted_return": record.predicted_return,
            "sentiment_label": record.sentiment_label,
            "ai_stance": record.ai_stance or "NEUTRAL",
            "model_name": record.model_name,
        }
    
    return None


async def generate_and_save_commentary(
    session,
    symbol: str,
    current_price: float,
    predicted_price: float,
    predicted_return: float,
    sentiment_index: float,
    sentiment_label: str,
    top_influencers: list[dict],
    news_count: int = 0,
) -> Optional[str]:
    """
    Generate commentary and save to database.
    Called after pipeline completion.
    """
    commentary = await generate_commentary(
        current_price=current_price,
        predicted_price=predicted_price,
        predicted_return=predicted_return,
        sentiment_index=sentiment_index,
        sentiment_label=sentiment_label,
        top_influencers=top_influencers,
        news_count=news_count,
    )
    
    if commentary:
        # Determine AI stance from the commentary
        ai_stance = await determine_ai_stance(commentary)
        
        save_commentary_to_db(
            session=session,
            symbol=symbol,
            commentary=commentary,
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            sentiment_label=sentiment_label,
            ai_stance=ai_stance,
        )
    
    return commentary
