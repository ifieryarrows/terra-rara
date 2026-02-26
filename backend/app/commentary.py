"""
AI Commentary Generator using OpenRouter API.
Generates market commentary and stance in a single structured LLM call.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from .openrouter_client import OpenRouterError, create_chat_completion
from .settings import get_settings

logger = logging.getLogger(__name__)

VALID_STANCES = {"BULLISH", "NEUTRAL", "BEARISH"}

COMMENTARY_RESPONSE_FORMAT = {
    "type": "json_object",
}


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


def _clean_json_content(content: str) -> str:
    """Normalize model text into parseable JSON content."""
    normalized = content.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()

    if normalized.startswith("json"):
        normalized = normalized[4:].strip()

    if not normalized.startswith("{"):
        first = normalized.find("{")
        last = normalized.rfind("}")
        if first != -1 and last != -1 and last > first:
            normalized = normalized[first : last + 1]

    return normalized


def _normalize_stance(value: str) -> str:
    stance = str(value or "").strip().upper()
    if stance not in VALID_STANCES:
        raise ValueError(f"Invalid stance: {value!r}")
    return stance


def _deterministic_stance_from_inputs(predicted_return: float, sentiment_index: float) -> str:
    combined = float(predicted_return) + float(sentiment_index)
    if combined > 0:
        return "BULLISH"
    if combined < 0:
        return "BEARISH"
    return "NEUTRAL"


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

    return "\n".join(
        [
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
        ]
    )


def _detect_stance_from_keywords(text: str) -> str:
    """Fallback stance detector from commentary keywords."""
    text_lower = (text or "").lower()

    bullish_keywords = [
        "bullish",
        "upside",
        "upward",
        "positive",
        "gain",
        "rise",
        "rising",
        "higher",
        "growth",
        "optimistic",
        "rally",
        "surge",
        "strength",
    ]
    bearish_keywords = [
        "bearish",
        "downside",
        "downward",
        "negative",
        "decline",
        "fall",
        "falling",
        "lower",
        "weakness",
        "pessimistic",
        "drop",
        "slump",
        "pressure",
    ]

    bullish_count = sum(1 for kw in bullish_keywords if kw in text_lower)
    bearish_count = sum(1 for kw in bearish_keywords if kw in text_lower)

    if bullish_count > bearish_count + 1:
        stance = "BULLISH"
    elif bearish_count > bullish_count + 1:
        stance = "BEARISH"
    else:
        stance = "NEUTRAL"

    logger.info(
        "Keyword stance detection: bullish=%s, bearish=%s -> %s",
        bullish_count,
        bearish_count,
        stance,
    )
    return stance


async def determine_ai_stance(commentary: str) -> str:
    """
    Backward-compatible stance helper.
    Dedicated stance LLM call is disabled; this fallback is deterministic and local.
    """
    if not commentary:
        return "NEUTRAL"
    return _detect_stance_from_keywords(commentary)


def _parse_commentary_payload(content: str) -> tuple[str, str]:
    payload = json.loads(_clean_json_content(content))
    if not isinstance(payload, dict):
        raise ValueError("Commentary payload must be a JSON object")

    stance = _normalize_stance(payload.get("stance", ""))
    commentary = str(payload.get("commentary", "")).strip()
    if not commentary:
        raise ValueError("Commentary text is empty")

    if "This is NOT financial advice." not in commentary:
        commentary = f"{commentary}\nThis is NOT financial advice."
    return stance, commentary


async def _generate_commentary_and_stance(
    *,
    current_price: float,
    predicted_price: float,
    predicted_return: float,
    sentiment_index: float,
    sentiment_label: str,
    top_influencers: list[dict],
    news_count: int,
) -> tuple[str, str]:
    settings = get_settings()
    deterministic_stance = _deterministic_stance_from_inputs(predicted_return, sentiment_index)
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
        return fallback_commentary, deterministic_stance

    influencers_text = "\n".join(
        [
            f"- {inf.get('feature', 'Unknown')}: {inf.get('importance', 0) * 100:.1f}%"
            for inf in top_influencers[:5]
        ]
    )

    user_prompt = f"""Generate commentary and stance using only the provided data.
Return strict JSON with keys: stance, commentary.

Rules:
- stance must be one of: BULLISH, BEARISH, NEUTRAL
- commentary must include exactly:
  1) 3 risk bullets
  2) 3 opportunity bullets
  3) 1 summary sentence
  4) 1 bias warning sentence
  5) final line: This is NOT financial advice.

Data:
- Current Price: {current_price:.4f}
- Predicted Price: {predicted_price:.4f}
- Predicted Return: {predicted_return:.6f}
- Sentiment Index: {sentiment_index:.6f}
- Sentiment Label: {sentiment_label}
- News Count: {news_count}
- Top Influencers:
{influencers_text}
"""

    base_request_kwargs = {
        "api_key": settings.openrouter_api_key,
        "model": settings.resolved_commentary_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a copper market analyst. "
                    "Use only provided inputs. Return concise, structured output."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2500,
        "temperature": 0.0,
        "timeout_seconds": 60.0,
        "max_retries": settings.openrouter_max_retries,
        "rpm": settings.openrouter_rpm,
        "fallback_models": settings.openrouter_fallback_models_list,
        "referer": "https://copper-mind.vercel.app",
        "title": "CopperMind Commentary",
    }

    async def _request_commentary() -> str:
        kwargs = dict(base_request_kwargs)
        kwargs["response_format"] = COMMENTARY_RESPONSE_FORMAT
        data = await create_chat_completion(**kwargs)
        content = _extract_chat_message_content(data)
        if not content:
            raise ValueError("Empty OpenRouter response content")
        return content

    async def _repair_commentary(malformed_content: str) -> str:
        repair_prompt = (
            "Fix this malformed output into valid JSON object with keys stance and commentary. "
            "Do not change meaning. Output JSON only.\n\n"
            f"{malformed_content}"
        )
        repair_data = await create_chat_completion(
            api_key=settings.openrouter_api_key,
            model=settings.resolved_commentary_model,
            messages=[
                {
                    "role": "system",
                    "content": "You repair JSON only. Output valid JSON and nothing else.",
                },
                {"role": "user", "content": repair_prompt},
            ],
            max_tokens=2500,
            temperature=0.0,
            timeout_seconds=60.0,
            max_retries=settings.openrouter_max_retries,
            rpm=settings.openrouter_rpm,
            fallback_models=settings.openrouter_fallback_models_list,
            referer="https://copper-mind.vercel.app",
            title="CopperMind Commentary JSON Repair",
        )
        repaired = _extract_chat_message_content(repair_data)
        if not repaired:
            raise ValueError("Empty commentary repair response")
        return repaired

    try:
        content = await _request_commentary()

        try:
            stance, commentary = _parse_commentary_payload(content)
            logger.info("AI commentary generated successfully (%s chars)", len(commentary))
            return commentary, stance
        except Exception as parse_exc:
            logger.warning("Commentary JSON parse failed, attempting repair: %s", parse_exc)
            repaired = await _repair_commentary(content)
            stance, commentary = _parse_commentary_payload(repaired)
            logger.info("AI commentary generated via JSON repair (%s chars)", len(commentary))
            return commentary, stance
    except Exception as exc:
        logger.warning("Commentary generation failed, using deterministic fallback: %s", exc)
        return fallback_commentary, deterministic_stance


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
    Generate AI commentary text.
    """
    commentary, _stance = await _generate_commentary_and_stance(
        current_price=current_price,
        predicted_price=predicted_price,
        predicted_return=predicted_return,
        sentiment_index=sentiment_index,
        sentiment_label=sentiment_label,
        top_influencers=top_influencers,
        news_count=news_count,
    )
    return commentary


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
    existing = session.query(AICommentary).filter(AICommentary.symbol == symbol).first()

    if existing:
        existing.commentary = commentary
        existing.current_price = current_price
        existing.predicted_price = predicted_price
        existing.predicted_return = predicted_return
        existing.sentiment_label = sentiment_label
        existing.ai_stance = ai_stance
        existing.generated_at = datetime.now(timezone.utc)
        existing.model_name = settings.resolved_commentary_model
        logger.info("Updated AI commentary for %s (stance: %s)", symbol, ai_stance)
    else:
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
        logger.info("Created new AI commentary for %s (stance: %s)", symbol, ai_stance)

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
    commentary, ai_stance = await _generate_commentary_and_stance(
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
            ai_stance=ai_stance,
        )

    return commentary
