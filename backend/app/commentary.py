"""
AI Commentary Generator using OpenRouter API.
Generates human-readable market analysis from FinBERT + XGBoost results.
"""

import httpx
import logging
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
    
    prompt = f"""You are a commodity market analyst writing for investors. Using ONLY the data below, produce a human-readable copper market commentary that is clear, measured, and non-sensational. Do not invent extra facts, macro events, or drivers beyond what is provided.

DATA
- Current Price: ${current_price:.4f}
- Tomorrow’s Model Outlook: {change_direction} of {abs(predicted_return*100):.2f}%
- Market Sentiment: {sentiment_label} (Score: {sentiment_index:.3f}, range -1 to 1)
- News Analyzed: {news_count} articles
- Top Influencing Factors (XGBoost importance, top 5):
{influencers_text}

WRITING REQUIREMENTS
- Length: 150–200 words total.
- Structure: 3 paragraphs, no headings, no bullet points, no markdown.
- Language: Professional, plain English; avoid technical ML jargon.
- **Do NOT use emojis.**
- **Do NOT mention specific future price targets (e.g., "$4.25"). Only discuss percentage moves and direction.**
- Explicitly reference 2–4 of the listed influencing factors by name and connect them to the outlook without overclaiming.
- Keep uncertainty: present the prediction as model-based and conditional; avoid certainty words like “will” or “guaranteed”; do not give trading instructions.

PARAGRAPH GUIDE
1) Overall outlook and what the model implies for the next session.
2) Key drivers: sentiment + news volume + the most important factors (tie them to the expected move).
3) Risks and short-term wrap-up: concise forecast framing and one monitoring takeaway.

End with this exact line on its own: This is NOT financial advice."""

    # Strategic system prompt with 2026 context
    system_prompt = """You are a commodity and metals market analyst focused on copper. Your job is to turn the user's provided inputs (prices/returns, sentiment, news volume, and any listed drivers) into a clear, investor-facing narrative that is grounded in the data. Do not invent facts, events, or catalysts. If a detail is not provided, do not assume it; instead, speak conditionally and describe what would matter if it occurs.

Write with disciplined uncertainty: treat forecasts as model-based indications, not certainties. Avoid trading instructions and avoid promises. Keep the tone professional, plain-language, and non-sensational.

Integrate the following strategic context as conditional interpretation rules only when relevant to the user's inputs or explicitly mentioned news/drivers; do not introduce these topics unless they connect to provided information. If multiple signals point in different directions, present the tension and explain what would resolve it.

Strategic context for 2026: M&A signals matter. If evidence of rising free cash flow, bid rumors, or acquisition chatter appears around majors such as BHP, Rio Tinto, or Glencore, interpret it as a supportive backdrop for mid-tier copper targets, with examples including FIL, LUN, and IVN, consistent with the theme that "buying is cheaper than building." Safe-jurisdiction premium can appear when Chile/Peru disruptions are in the news; when that happens, frame it as relatively supportive for Australia/Canada-based miners such as BHP and TECK, without overstating. China factor: if Zijin Mining (2899.HK) strength is referenced, treat it as potential evidence of China stockpiling copper and describe it as a lagging buy-type signal for HG=F rather than an immediate trigger. Grade decline: if "Tenör Düşüşü" or "Grade Decline" is mentioned, frame it as supportive for leaching and technology leaders such as Freeport (FCX), again conditionally.

When you cite specific companies, tickers, or the meaning of the above signals, keep claims modest and explain the logic. If the user requests factual assertions that depend on real-world current events, include credible citations with links; otherwise, stick to reasoning from the supplied data and clearly label any assumptions. Always end with: This is NOT financial advice."""

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

