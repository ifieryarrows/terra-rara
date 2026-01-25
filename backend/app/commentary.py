"""
AI Commentary Generator using OpenRouter API.
Generates human-readable market analysis from FinBERT + XGBoost results.
"""

import logging

# Suppress httpx request logging to prevent API keys in URLs from appearing in logs
logging.getLogger("httpx").setLevel(logging.WARNING)

import httpx
from typing import Optional
from datetime import datetime

from .settings import get_settings

logger = logging.getLogger(__name__)


async def determine_ai_stance(commentary: str) -> str:
    """
    Have the AI analyze its own commentary to determine market stance.
    
    Args:
        commentary: The generated commentary text
        
    Returns:
        BULLISH, NEUTRAL, or BEARISH
    """
    settings = get_settings()
    
    if not settings.openrouter_api_key or not commentary:
        return "NEUTRAL"
    
    prompt = f"""Analyze the following market commentary and determine the overall market stance.
Respond with ONLY one word: BULLISH, NEUTRAL, or BEARISH.

Commentary:
{commentary}

Your response (one word only):"""
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.openrouter_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0.1,
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                stance = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
                
                # Validate response
                if stance in ["BULLISH", "NEUTRAL", "BEARISH"]:
                    logger.info(f"AI stance determined: {stance}")
                    return stance
                else:
                    logger.warning(f"Invalid AI stance response: {stance}, defaulting to NEUTRAL")
                    return "NEUTRAL"
            else:
                logger.error(f"AI stance API error: {response.status_code}")
                return "NEUTRAL"
                
    except Exception as e:
        logger.error(f"AI stance detection failed: {e}")
        return "NEUTRAL"

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
    
    if not settings.openrouter_api_key:
        logger.warning("OpenRouter API key not configured, skipping commentary")
        return None
    
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
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://copper-mind.vercel.app",
                    "X-Title": "CopperMind AI Analysis",
                },
                json={
                    "model": settings.openrouter_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "max_tokens": 700,
                    "temperature": 0.6,
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                commentary = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if commentary:
                    logger.info(f"AI commentary generated successfully ({len(commentary)} chars)")
                    return commentary.strip()
                else:
                    logger.warning("Empty response from OpenRouter")
                    return None
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"Failed to generate AI commentary: {e}")
        return None


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
        existing.model_name = settings.openrouter_model
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
            model_name=settings.openrouter_model,
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
