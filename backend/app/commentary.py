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
    
    prompt = f"""You are a financial analyst assistant. Analyze the following copper market data and write a short, clear commentary for investors.

## Current Data:
- **Current Price:** ${current_price:.4f}
- **Tomorrow's Prediction:** ${predicted_price:.4f} ({change_emoji} {abs(predicted_return*100):.2f}% {change_direction})
- **Market Sentiment:** {sentiment_label} (Score: {sentiment_index:.3f})
- **News Analyzed:** {news_count} articles

## Top Influencing Factors (XGBoost Model):
{influencers_text}

## Instructions:
1. Write 3-4 paragraphs (150-200 words total)
2. Use simple, clear language; avoid overly technical jargon
3. In the first paragraph, summarize the general outlook
4. In the second paragraph, explain the key driving factors
5. In the final paragraph, state the short-term forecast
6. Use the 🎯 emoji to highlight key points
7. Add a disclaimer: "This is NOT financial advice."

Write your commentary:"""

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
                            "content": "You are an expert commodity market analyst. You provide concise and insightful analysis of copper prices."
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


# Cache for commentary (simple in-memory)
_commentary_cache: dict = {
    "commentary": None,
    "generated_at": None,
    "expires_at": None,
}


async def get_cached_commentary(
    current_price: float,
    predicted_price: float,
    predicted_return: float,
    sentiment_index: float,
    sentiment_label: str,
    top_influencers: list[dict],
    news_count: int = 0,
    ttl_minutes: int = 60,
) -> Optional[str]:
    """
    Get cached commentary or generate new one if expired.
    """
    global _commentary_cache
    
    now = datetime.now()
    
    # Check if cache is valid
    if (
        _commentary_cache["commentary"] 
        and _commentary_cache["expires_at"] 
        and now < _commentary_cache["expires_at"]
    ):
        logger.debug("Returning cached AI commentary")
        return _commentary_cache["commentary"]
    
    # Generate new commentary
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
        from datetime import timedelta
        _commentary_cache = {
            "commentary": commentary,
            "generated_at": now,
            "expires_at": now + timedelta(minutes=ttl_minutes),
        }
    
    return commentary
