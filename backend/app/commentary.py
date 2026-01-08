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
    change_emoji = "📈" if predicted_return > 0 else "📉"
    
    prompt = f"""You are a commodity market analyst writing for investors. Using ONLY the data below, produce a human-readable copper market commentary that is clear, measured, and non-sensational. Do not invent extra facts, macro events, or drivers beyond what is provided.

DATA
- Current Price: ${current_price:.4f}
- Tomorrow’s Model Prediction: ${predicted_price:.4f} ({change_emoji} {abs(predicted_return*100):.2f}% {change_direction})
- Market Sentiment: {sentiment_label} (Score: {sentiment_index:.3f}, range -1 to 1)
- News Analyzed: {news_count} articles
- Top Influencing Factors (XGBoost importance, top 5):
{influencers_text}

WRITING REQUIREMENTS
- Length: 150–200 words total.
- Structure: 3–4 paragraphs, no headings, no bullet points, no markdown.
- Language: simple, plain English; avoid technical ML jargon (do not mention “FinBERT”, “XGBoost”, “features”, “SHAP”, “probabilities”, or “hyperparameters”).
- Use the 🎯 emoji 2–4 times to highlight key points within sentences (not as a list).
- Mention the predicted direction and percent move once, exactly as given (do not recalculate).
- Explicitly reference 2–4 of the listed influencing factors by name and connect them to the outlook in a reasonable way without overclaiming.
- Keep uncertainty: present the prediction as model-based and conditional; avoid certainty words like “will” or “guaranteed”; do not give trading instructions.

PARAGRAPH GUIDE
1) Overall outlook and what the model implies for the next session.
2) Key drivers: sentiment + news volume + the most important factors (tie them to the expected move).
3) Risks/what could invalidate the view and what to watch next (can be combined with paragraph 4 if you write 3 paragraphs).
4) Short-term wrap-up: concise forecast framing and one monitoring takeaway.

End with this exact line on its own: This is NOT financial advice."""

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
                            "content": "You are a commodity market analyst. You write measured, evidence-based commentary avoiding sensationalism."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7,
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
        existing.generated_at = datetime.utcnow()
        existing.model_name = settings.openrouter_model
        logger.info(f"Updated AI commentary for {symbol}")
    else:
        # Create new
        new_commentary = AICommentary(
            symbol=symbol,
            commentary=commentary,
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            sentiment_label=sentiment_label,
            model_name=settings.openrouter_model,
        )
        session.add(new_commentary)
        logger.info(f"Created new AI commentary for {symbol}")
    
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
        save_commentary_to_db(
            session=session,
            symbol=symbol,
            commentary=commentary,
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            sentiment_label=sentiment_label,
        )
    
    return commentary

