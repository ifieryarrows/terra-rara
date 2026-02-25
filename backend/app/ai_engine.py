"""
AI Engine: Hybrid LLM + FinBERT sentiment scoring + XGBoost training.

Sentiment Analysis:
    Direction: OpenRouter LLM (BULLISH/BEARISH/NEUTRAL)
    Intensity: FinBERT probabilities for each article
    Reliability: strict JSON + repair + deterministic fallback

Usage:
    python -m app.ai_engine --run-all --target-symbol HG=F
    python -m app.ai_engine --score-only
    python -m app.ai_engine --refresh-sentiment
    python -m app.ai_engine --train-only --target-symbol HG=F
"""

import argparse
import json
import logging
import os
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.db import SessionLocal, init_db
from app.models import (
    NewsArticle,
    NewsSentiment,
    DailySentiment,
    PriceBar,
    NewsProcessed,
    NewsRaw,
    NewsSentimentV2,
    DailySentimentV2,
)
from app.settings import get_settings
from app.features import build_feature_matrix, get_feature_descriptions
from app.lock import pipeline_lock
from app.async_bridge import run_async_from_sync
from app.openrouter_client import OpenRouterError, OpenRouterRateLimitError, create_chat_completion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_FINBERT_OUTPUT_LOGGED = False
_FINBERT_MISSING_LABELS_WARNED = False

HYBRID_SCORING_VERSION = "hybrid_v2"
HYBRID_FALLBACK_429_MODEL_NAME = "hybrid_fallback_429"
HYBRID_FALLBACK_PARSE_MODEL_NAME = "hybrid_fallback_parse"
LLM_LABELS = {"BULLISH", "BEARISH", "NEUTRAL"}
LLM_SCORING_MAX_TOKENS_PRIMARY = 2000
LLM_SCORING_MAX_TOKENS_RETRY = 6000
LLM_V2_LABEL_THRESHOLD = 0.15
LLM_V2_EVENT_TYPES = {
    "supply_disruption",
    "supply_expansion",
    "demand_increase",
    "demand_decrease",
    "inventory_draw",
    "inventory_build",
    "policy_support",
    "policy_drag",
    "macro_usd_up",
    "macro_usd_down",
    "cost_push",
    "mixed_unclear",
    "non_copper",
}
LLM_V2_EVENT_SIGN = {
    "supply_disruption": 1,
    "inventory_draw": 1,
    "demand_increase": 1,
    "policy_support": 1,
    "macro_usd_down": 1,
    "cost_push": 1,
    "supply_expansion": -1,
    "inventory_build": -1,
    "demand_decrease": -1,
    "policy_drag": -1,
    "macro_usd_up": -1,
    "mixed_unclear": 0,
    "non_copper": 0,
}
LLM_V2_EVENT_STRENGTH = {
    "supply_disruption": 1.0,
    "inventory_draw": 0.9,
    "demand_increase": 0.95,
    "policy_support": 0.8,
    "macro_usd_down": 0.7,
    "cost_push": 0.75,
    "supply_expansion": 1.0,
    "inventory_build": 0.9,
    "demand_decrease": 0.95,
    "policy_drag": 0.8,
    "macro_usd_up": 0.7,
    "mixed_unclear": 0.25,
    "non_copper": 0.0,
}
LLM_V2_SYSTEM_PROMPT = """You are a Senior Copper Futures Analyst focused on COMEX HG=F front-month contract.
Your job is to estimate 1-5 trading day copper price impact from each article.

Core principle:
Classify by expected HG=F price reaction, NOT by whether the news is "good" or "bad" for the economy/company.

Output requirements:
Return ONLY a JSON array. One object per input id.
Each object must contain exactly:
- id (integer)
- label ("BULLISH" | "BEARISH" | "NEUTRAL")
- impact_score (number, -1.00 to 1.00, two decimals)
- confidence (number, 0.00 to 1.00, two decimals)
- relevance (number, 0.00 to 1.00, two decimals)
- event_type (one of: supply_disruption, supply_expansion, demand_increase, demand_decrease, inventory_draw, inventory_build, policy_support, policy_drag, macro_usd_up, macro_usd_down, cost_push, mixed_unclear, non_copper)
- reasoning (single line, <= 160 chars)

Copper-specific reasoning rules:
1) Supply tightening is typically BULLISH for copper price.
2) Supply expansion is typically BEARISH.
3) Demand increase is typically BULLISH.
4) Demand decrease is typically BEARISH.
5) USD stronger is usually BEARISH for dollar-denominated copper; USD weaker is usually BULLISH.
6) If article is not materially related to copper supply/demand/pricing, use non_copper + NEUTRAL with low relevance/confidence.
7) Use NEUTRAL only when net effect is truly mixed/unclear within 1-5 day horizon.

Label mapping:
- impact_score >= 0.15 => BULLISH
- impact_score <= -0.15 => BEARISH
- otherwise => NEUTRAL
"""
LLM_SCORING_RESPONSE_FORMAT_V2 = {
    "type": "json_object",
}
SCORING_V2_VERSION = "commodity_v2"


# =============================================================================
# FinBERT Sentiment Scoring
# =============================================================================


def _neutral_finbert_score() -> dict:
    """Neutral fallback score used when FinBERT output is invalid or unavailable."""
    return {
        "prob_positive": 0.33,
        "prob_neutral": 0.34,
        "prob_negative": 0.33,
        "score": 0.0,
    }


def _normalize_finbert_output(raw_output: Any) -> list[dict]:
    """
    Normalize FinBERT output into a flat ``list[dict]``.

    Supported raw formats:
    - list[dict]
    - list[list[dict]]
    - dict
    - JSON string of any of the above
    """
    output = raw_output
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ValueError("FinBERT output is not valid JSON") from exc

    if isinstance(output, dict):
        output = [output]

    if not isinstance(output, list):
        raise TypeError(f"Unsupported FinBERT output type: {type(output).__name__}")

    normalized: list[dict] = []
    for item in output:
        if isinstance(item, dict):
            normalized.append(item)
            continue

        if isinstance(item, list):
            normalized.extend([child for child in item if isinstance(child, dict)])
            continue

        logger.debug("Skipping unsupported FinBERT output item type: %s", type(item).__name__)

    return normalized


def _log_finbert_output_once(raw_output: Any) -> None:
    """Log one representative FinBERT output shape for debugging parser mismatches."""
    global _FINBERT_OUTPUT_LOGGED
    if _FINBERT_OUTPUT_LOGGED:
        return

    first_item = raw_output
    if isinstance(raw_output, list) and raw_output:
        first_item = raw_output[0]

    preview = repr(first_item)
    if len(preview) > 220:
        preview = f"{preview[:220]}..."

    logger.info(
        "FinBERT output sample: type=%s first_item_type=%s first_item=%s",
        type(raw_output).__name__,
        type(first_item).__name__,
        preview,
    )
    _FINBERT_OUTPUT_LOGGED = True

@lru_cache(maxsize=1)
def get_finbert_pipeline():
    """
    Load FinBERT model pipeline.
    Lazy loading to avoid import overhead when not needed.
    """
    settings = get_settings()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", settings.tokenizers_parallelism)

    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    
    model_name = "ProsusAI/finbert"
    
    logger.info(f"Loading FinBERT model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        max_length=512,
        truncation=True
    )
    
    logger.info("FinBERT model loaded")
    return pipe


def score_text_with_finbert(
    pipe,
    text: str
) -> dict:
    """
    Score a single text with FinBERT.
    
    Returns:
        Dict with prob_positive, prob_neutral, prob_negative, score
    """
    if not text or len(text.strip()) < 10:
        return _neutral_finbert_score()
    
    # Truncate long text
    text = text[:1000]
    
    try:
        try:
            raw_output = pipe(text, top_k=None)
        except TypeError:
            # Older transformers pipeline signature may not support top_k argument.
            raw_output = pipe(text)
        _log_finbert_output_once(raw_output)
        results = _normalize_finbert_output(raw_output)
        if not results:
            logger.warning("FinBERT output normalized to empty list, using neutral fallback")
            return _neutral_finbert_score()

        probs: dict[str, float] = {}
        for row in results:
            label = row.get("label")
            score = row.get("score")
            if not isinstance(label, str):
                continue
            try:
                probs[label.lower()] = float(score)
            except (TypeError, ValueError):
                continue

        required_labels = {"positive", "neutral", "negative"}
        if not required_labels.issubset(probs):
            global _FINBERT_MISSING_LABELS_WARNED
            if not _FINBERT_MISSING_LABELS_WARNED:
                logger.warning(
                    "FinBERT output missing labels. found=%s expected=%s (further repeats suppressed)",
                    sorted(probs.keys()),
                    sorted(required_labels),
                )
                _FINBERT_MISSING_LABELS_WARNED = True
            return _neutral_finbert_score()

        prob_pos = probs["positive"]
        prob_neu = probs["neutral"]
        prob_neg = probs["negative"]
        
        # Derived score: positive - negative (range: -1 to 1)
        score = prob_pos - prob_neg
        
        return {
            "prob_positive": prob_pos,
            "prob_neutral": prob_neu,
            "prob_negative": prob_neg,
            "score": score
        }
        
    except Exception as e:
        logger.warning(f"FinBERT scoring error: {e}")
        return _neutral_finbert_score()


# =============================================================================
# LLM Sentiment Scoring (Primary - OpenRouter)
# =============================================================================

# Copper-specific system prompt for LLM direction classification.
LLM_SENTIMENT_SYSTEM_PROMPT = """You are a neutral copper market classifier.
Task: For each input article, classify immediate 1-5 day HG=F price impact direction.
Allowed labels:
- BULLISH
- BEARISH
- NEUTRAL

Rules:
- Use only provided title/description text.
- Return one output item per input id.
- Keep reasoning one line, no newline/tab characters.
- Do not add extra keys.
- If uncertain or non-copper-relevant, use NEUTRAL with low confidence.
"""


LLM_SCORING_RESPONSE_FORMAT = {
    "type": "json_object",
}

LLM_SCORING_PROVIDER_OPTIONS = {"require_parameters": True}


class LLMStructuredOutputError(RuntimeError):
    """Raised when LLM structured output remains invalid after repair attempts."""


def _hybrid_model_name(llm_model: str) -> str:
    """Stable model identifier for hybrid scoring rows."""
    return f"hybrid({llm_model}+ProsusAI/finbert)"


def _sanitize_reasoning_text(value: Any) -> str:
    """Normalize reasoning to a single line without tabs/newlines."""
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def _neutral_llm_result(
    *,
    article_id: int,
    llm_model: str,
    reason: str,
    model_name_override: Optional[str] = None,
) -> dict:
    return {
        "id": int(article_id),
        "label": "NEUTRAL",
        "llm_confidence": 0.0,
        "llm_reasoning": _sanitize_reasoning_text(reason),
        "llm_model": llm_model,
        "model_name": model_name_override or _hybrid_model_name(llm_model),
        "json_repair_used": False,
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    """Classify OpenRouter 429 errors for deterministic neutral fallback semantics."""
    if isinstance(exc, OpenRouterRateLimitError):
        return True
    if isinstance(exc, OpenRouterError) and exc.status_code == 429:
        return True
    message = str(exc).lower()
    return "429" in message and "rate" in message


def _build_hybrid_reasoning_payload(
    *,
    label: str,
    llm_confidence: float,
    finbert_strength: float,
    finbert_polarity: float,
    llm_reasoning: str,
    llm_model: str,
    soft_neutral_applied: bool = False,
) -> str:
    payload = {
        "label": label,
        "llm_confidence": round(max(0.0, min(1.0, llm_confidence)), 4),
        "finbert_strength": round(max(0.0, min(1.0, finbert_strength)), 4),
        "finbert_polarity": round(max(-1.0, min(1.0, finbert_polarity)), 4),
        "llm_reasoning": _sanitize_reasoning_text(llm_reasoning),
        "llm_model": llm_model,
        "soft_neutral_applied": bool(soft_neutral_applied),
        "scoring_version": HYBRID_SCORING_VERSION,
    }
    return json.dumps(payload, ensure_ascii=True)


def _compute_hybrid_score(
    *,
    label: str,
    llm_confidence: float,
    finbert_strength: float,
    finbert_polarity: Optional[float] = None,
    non_neutral_boost: float = 1.35,
    soft_neutral_polarity_threshold: float = 0.12,
    soft_neutral_max_mag: float = 0.25,
    soft_neutral_scale: float = 0.8,
    return_metadata: bool = False,
) -> float | tuple[float, bool]:
    """Compute final hybrid impact score in [-1, 1] with boosted non-neutral and soft-neutral rules."""
    normalized_label = str(label).upper().strip()
    if normalized_label not in LLM_LABELS:
        normalized_label = "NEUTRAL"

    confidence = max(0.0, min(1.0, float(llm_confidence)))
    strength = max(0.0, min(1.0, float(finbert_strength)))
    polarity_value = float(finbert_polarity) if finbert_polarity is not None else 0.0
    polarity = max(-1.0, min(1.0, polarity_value))
    soft_neutral_applied = False

    if normalized_label == "NEUTRAL":
        abs_polarity = abs(polarity)
        if abs_polarity < max(0.0, float(soft_neutral_polarity_threshold)):
            final_score = 0.0
        else:
            neutral_core = (0.6 * abs_polarity) + (0.4 * strength)
            neutral_mag = min(
                max(0.0, float(soft_neutral_max_mag)),
                neutral_core * max(0.0, float(soft_neutral_scale)),
            )
            sign = 1.0 if polarity > 0 else -1.0
            final_score = sign * neutral_mag
            soft_neutral_applied = True

        if return_metadata:
            return final_score, soft_neutral_applied
        return final_score

    sign = 1.0 if normalized_label == "BULLISH" else -1.0
    base_mag = max(0.0, min(1.0, (0.7 * confidence) + (0.3 * strength)))
    boosted_mag = min(1.0, base_mag * max(0.0, float(non_neutral_boost)))
    final_score = sign * boosted_mag
    if return_metadata:
        return final_score, soft_neutral_applied
    return final_score


def _extract_chat_message_content(data: dict[str, Any]) -> str:
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


def _extract_finish_reason(data: dict[str, Any]) -> str:
    """Extract finish reason from first choice for empty-content diagnostics."""
    reason = data.get("choices", [{}])[0].get("finish_reason", "")
    if not isinstance(reason, str):
        return ""
    return reason.strip().lower()


def _clean_json_content(content: str) -> str:
    """
    Normalize model text into parseable JSON content.

    Handles common LLM output quirks:
    - Markdown fenced code blocks (```json ... ```)
    - Wrapped objects like {"results": [...]} or {"scores": [...]}
    - Thinking/reasoning preamble before JSON
    - Raw JSON arrays
    """
    normalized = content.strip()

    # Strip markdown code fences
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()

    if normalized.startswith("json"):
        normalized = normalized[4:].strip()

    # Already a JSON array — return as-is
    if normalized.startswith("["):
        return normalized

    # Wrapped object: {"results": [...], ...} or {"scores": [...], ...}
    if normalized.startswith("{"):
        try:
            import json as _json
            obj = _json.loads(normalized)
            if isinstance(obj, dict):
                # Find the first list value
                for v in obj.values():
                    if isinstance(v, list):
                        return _json.dumps(v)
                # Single object — wrap in array
                return _json.dumps([obj])
        except Exception:
            pass

    # Preamble text before JSON — find the array
    first = normalized.find("[")
    last = normalized.rfind("]")
    if first != -1 and last != -1 and last > first:
        return normalized[first:last + 1]

    # Last resort — try to find an object
    first_obj = normalized.find("{")
    last_obj = normalized.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        return normalized[first_obj:last_obj + 1]

    return normalized


def _validate_llm_results(
    *,
    raw_results: Any,
    expected_ids: list[int],
    model_name: str,
    json_repair_used: bool = False,
) -> list[dict]:
    """Validate/normalize LLM output for direction+confidence labels."""
    if not isinstance(raw_results, list):
        raise ValueError(f"Structured result must be a list, got {type(raw_results).__name__}")

    results_by_id: dict[int, dict] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            raise ValueError(f"Structured result item must be object, got {type(item).__name__}")
        if "id" not in item:
            raise ValueError("Structured result missing required field: id")

        article_id = int(item["id"])
        if article_id in results_by_id:
            raise ValueError(f"Duplicate article id in structured output: {article_id}")

        raw_label = item.get("label")
        raw_confidence = item.get("confidence")
        raw_score = item.get("score")

        score_value: Optional[float] = None
        if raw_score is not None:
            score_value = max(-1.0, min(1.0, float(raw_score)))

        if raw_label is not None:
            label = str(raw_label).upper().strip()
            if label not in LLM_LABELS:
                raise ValueError(f"Unsupported label in structured output: {label}")
        elif score_value is not None:
            if score_value > 0.05:
                label = "BULLISH"
            elif score_value < -0.05:
                label = "BEARISH"
            else:
                label = "NEUTRAL"
        else:
            raise ValueError("Structured result missing label/score fields")

        if raw_confidence is not None:
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        elif score_value is not None:
            confidence = abs(score_value)
        else:
            confidence = 0.0 if label == "NEUTRAL" else 0.5

        reasoning = _sanitize_reasoning_text(item.get("reasoning", ""))
        results_by_id[article_id] = {
            "id": article_id,
            "label": label,
            "llm_confidence": confidence,
            "llm_reasoning": reasoning,
            "llm_model": model_name,
            "model_name": _hybrid_model_name(model_name),
            "json_repair_used": json_repair_used,
        }

    expected = set(expected_ids)
    got = set(results_by_id.keys())
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise ValueError(f"Structured result ID mismatch. missing={missing} extra={extra}")

    return [results_by_id[article_id] for article_id in expected_ids]


async def score_batch_with_llm(
    articles: list[dict],
) -> list[dict]:
    """
    Classify a batch with LLM direction + confidence using strict JSON schema.
    """
    settings = get_settings()

    if not settings.openrouter_api_key:
        raise RuntimeError("OpenRouter API key not configured")

    normalized_articles = [
        {
            "id": int(article["id"]),
            "title": str(article.get("title") or ""),
            "description": str(article.get("description") or "")[:600],
        }
        for article in articles
    ]
    expected_ids = [item["id"] for item in normalized_articles]
    user_prompt = (
        "Classify each article into BULLISH, BEARISH, or NEUTRAL for short-term HG=F price impact.\n"
        "Return one output object per input id and keep reasoning single-line.\n\n"
        f"Input articles JSON:\n{json.dumps(normalized_articles, ensure_ascii=True)}"
    )
    model_name = settings.resolved_scoring_model

    async def _request_scoring(*, max_tokens: int) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "api_key": settings.openrouter_api_key,
            "model": model_name,
            "messages": [
                {"role": "system", "content": LLM_SENTIMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "timeout_seconds": 60.0,
            "max_retries": settings.openrouter_max_retries,
            "rpm": settings.openrouter_rpm,
            "fallback_models": settings.openrouter_fallback_models_list,
            "referer": "https://copper-mind.vercel.app",
            "title": "CopperMind Sentiment Analysis",
            "response_format": LLM_SCORING_RESPONSE_FORMAT,
            "extra_payload": {"reasoning": {"exclude": True}},
        }
        return await create_chat_completion(**request_kwargs)

    async def _repair_json_response(malformed_content: str) -> str:
        repair_prompt = (
            "Convert the following malformed model output into valid JSON WITHOUT changing meaning.\n"
            f"Expected ids: {expected_ids}\n"
            "Output only JSON array and keep keys: id,label,confidence,reasoning.\n\n"
            f"MALFORMED:\n{malformed_content}"
        )
        repair_data = await create_chat_completion(
            api_key=settings.openrouter_api_key,
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You fix JSON formatting only. Return valid JSON. "
                        "No markdown and no extra text."
                    ),
                },
                {"role": "user", "content": repair_prompt},
            ],
            max_tokens=2200,
            temperature=0.0,
            timeout_seconds=60.0,
            max_retries=settings.openrouter_max_retries,
            rpm=settings.openrouter_rpm,
            fallback_models=settings.openrouter_fallback_models_list,
            referer="https://copper-mind.vercel.app",
            title="CopperMind JSON Repair",
            extra_payload={"reasoning": {"exclude": True}},
        )
        repaired_content = _extract_chat_message_content(repair_data)
        if not repaired_content:
            raise LLMStructuredOutputError("JSON repair call returned empty content")
        return repaired_content

    def _parse_and_validate(content: str, *, repair_used: bool) -> list[dict]:
        raw_results = json.loads(_clean_json_content(content))
        return _validate_llm_results(
            raw_results=raw_results,
            expected_ids=expected_ids,
            model_name=model_name,
            json_repair_used=repair_used,
        )

    data = await _request_scoring(max_tokens=LLM_SCORING_MAX_TOKENS_PRIMARY)

    content = _extract_chat_message_content(data)
    if not content:
        finish_reason = _extract_finish_reason(data)
        if finish_reason == "length":
            logger.warning(
                "LLM response ended with finish_reason=length and empty content; "
                "retrying with max_tokens=%s",
                LLM_SCORING_MAX_TOKENS_RETRY,
            )
            data = await _request_scoring(max_tokens=LLM_SCORING_MAX_TOKENS_RETRY)
            content = _extract_chat_message_content(data)
        if not content:
            raise OpenRouterError(
                f"Empty response content from LLM scoring (finish_reason={finish_reason or 'unknown'})"
            )

    try:
        return _parse_and_validate(content, repair_used=False)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("LLM structured parse failed, attempting JSON repair: %s", exc)
        try:
            repaired_content = await _repair_json_response(content)
            return _parse_and_validate(repaired_content, repair_used=True)
        except Exception as repair_exc:
            raise LLMStructuredOutputError(
                f"LLM structured output invalid after repair: {repair_exc}"
            ) from repair_exc


def score_batch_with_finbert(articles: list) -> list[dict]:
    """
    Score articles with FinBERT to provide sentiment intensity probabilities.
    
    Args:
        articles: List of NewsArticle ORM objects
        
    Returns:
        List of dicts with FinBERT probabilities
    """
    pipe = get_finbert_pipeline()
    results = []
    
    for article in articles:
        text = f"{article.title} {article.description or ''}"
        scores = score_text_with_finbert(pipe, text)
        finbert_strength = abs(scores["prob_positive"] - scores["prob_negative"])
        
        results.append({
            "id": article.id,
            "score": scores["score"],
            "prob_positive": scores["prob_positive"],
            "prob_neutral": scores["prob_neutral"],
            "prob_negative": scores["prob_negative"],
            "finbert_strength": finbert_strength,
        })
    
    return results


def _clip(value: float, lower: float, upper: float) -> float:
    """Clamp numeric value."""
    return max(lower, min(upper, float(value)))


def _label_from_impact_score(impact_score: float) -> str:
    """Map impact score to discrete label."""
    if impact_score >= LLM_V2_LABEL_THRESHOLD:
        return "BULLISH"
    if impact_score <= -LLM_V2_LABEL_THRESHOLD:
        return "BEARISH"
    return "NEUTRAL"


def _sign(value: float, eps: float = 1e-9) -> int:
    """Return numeric sign with epsilon deadzone."""
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _normalize_event_type(value: Any) -> str:
    """Normalize event type to allowed vocabulary."""
    normalized = str(value or "").strip().lower()
    if normalized in LLM_V2_EVENT_TYPES:
        return normalized
    return "mixed_unclear"


def _infer_event_type_from_text(text: str) -> str:
    """Heuristic event inference used only for deterministic fallback."""
    lower = (text or "").lower()
    if not lower or "copper" not in lower:
        return "non_copper"

    supply_disruption_keywords = [
        "outage",
        "strike",
        "disruption",
        "shutdown",
        "halt",
        "sanction",
        "inventory draw",
        "stocks fell",
        "warehouse draw",
    ]
    supply_expansion_keywords = [
        "ramp-up",
        "ramp up",
        "increase output",
        "production increase",
        "new mine",
        "inventory build",
        "stocks rose",
    ]
    demand_increase_keywords = [
        "stimulus",
        "grid investment",
        "ev demand",
        "demand rise",
        "stockpile purchase",
        "import growth",
    ]
    demand_decrease_keywords = [
        "slowdown",
        "weak demand",
        "demand decline",
        "construction slump",
        "pmi contraction",
        "import decline",
    ]
    if any(token in lower for token in supply_disruption_keywords):
        return "supply_disruption"
    if any(token in lower for token in supply_expansion_keywords):
        return "supply_expansion"
    if any(token in lower for token in demand_increase_keywords):
        return "demand_increase"
    if any(token in lower for token in demand_decrease_keywords):
        return "demand_decrease"
    if "dollar strengthens" in lower or "usd stronger" in lower:
        return "macro_usd_up"
    if "dollar weakens" in lower or "usd weaker" in lower:
        return "macro_usd_down"
    return "mixed_unclear"


def _build_llm_v2_user_prompt(articles: list[dict], horizon_days: int) -> str:
    """Build compact JSON prompt for batch scoring."""
    normalized_articles = [
        {
            "id": int(article["id"]),
            "title": str(article.get("title") or "")[:500],
            "description": str(article.get("description") or "")[:800],
        }
        for article in articles
    ]
    return (
        f"Classify each article for {horizon_days}-day HG=F copper futures impact.\n"
        "Return one object per id.\n\n"
        f"Input articles JSON:\n{json.dumps(normalized_articles, ensure_ascii=True)}"
    )


def _parse_llm_v2_items(
    *,
    raw_results: Any,
    expected_ids: list[int],
    model_name: str,
) -> tuple[dict[int, dict], list[int]]:
    """
    Parse/validate V2 LLM outputs.

    Returns:
        (valid_results_by_id, failed_ids)
    """
    if not isinstance(raw_results, list):
        raise ValueError(f"Structured result must be a list, got {type(raw_results).__name__}")

    expected = set(expected_ids)
    valid: dict[int, dict] = {}
    failed_ids: set[int] = set()

    for item in raw_results:
        if not isinstance(item, dict):
            logger.debug("V2 parse: skipping non-dict item: %s", type(item).__name__)
            continue
        if "id" not in item:
            logger.debug("V2 parse: item missing 'id' key, keys=%s", list(item.keys()))
            continue

        try:
            article_id = int(item["id"])
        except (TypeError, ValueError):
            logger.debug("V2 parse: invalid id value: %r", item.get("id"))
            continue

        if article_id not in expected:
            continue
        if article_id in valid:
            failed_ids.add(article_id)
            continue

        raw_label = item.get("label", item.get("classification"))
        raw_impact = item.get("impact_score", item.get("score"))
        raw_confidence = item.get("confidence")
        raw_relevance = item.get("relevance", item.get("relevance_score"))
        raw_event_type = item.get("event_type")
        raw_reasoning = item.get("reasoning", "")

        try:
            # impact_score is required; try label-based inference if missing
            if raw_impact is None:
                if raw_label and str(raw_label).upper().strip() in LLM_LABELS:
                    lbl = str(raw_label).upper().strip()
                    raw_impact = {"BULLISH": 0.3, "BEARISH": -0.3, "NEUTRAL": 0.0}.get(lbl, 0.0)
                    logger.debug("V2 parse: inferred impact_score=%.1f from label=%s for id=%d", raw_impact, lbl, article_id)
                else:
                    raise ValueError("missing impact_score and no valid label")

            impact_score = _clip(float(raw_impact), -1.0, 1.0)
            # confidence and relevance: default to 0.5 if missing
            confidence = _clip(float(raw_confidence), 0.0, 1.0) if raw_confidence is not None else 0.5
            relevance = _clip(float(raw_relevance), 0.0, 1.0) if raw_relevance is not None else 0.5
        except (TypeError, ValueError) as exc:
            logger.debug("V2 parse: field error for id=%d: %s (keys=%s)", article_id, exc, list(item.keys()))
            failed_ids.add(article_id)
            continue

        event_type = _normalize_event_type(raw_event_type)
        label_from_impact = _label_from_impact_score(impact_score)
        if raw_label is None:
            label = label_from_impact
        else:
            label = str(raw_label).upper().strip()
            if label not in LLM_LABELS:
                label = label_from_impact
            if label != label_from_impact:
                # Keep deterministic consistency between score and class.
                label = label_from_impact

        reasoning = _sanitize_reasoning_text(raw_reasoning)[:160]

        valid[article_id] = {
            "id": article_id,
            "label": label,
            "impact_score": impact_score,
            "confidence": confidence,
            "relevance": relevance,
            "event_type": event_type,
            "reasoning": reasoning,
            "llm_model": model_name,
        }

    # Mark missing ids as failed.
    missing_ids = []
    for article_id in expected_ids:
        if article_id not in valid:
            failed_ids.add(article_id)
            missing_ids.append(article_id)

    if missing_ids:
        logger.warning(
            "V2 parse: %d/%d articles missing from LLM response (model=%s, returned=%d items, missing_ids=%s)",
            len(missing_ids), len(expected_ids), model_name, len(raw_results),
            missing_ids[:10],
        )

    return valid, sorted(failed_ids)


async def _repair_json_response_v2(
    *,
    settings: Any,
    model_name: str,
    malformed_content: str,
    expected_ids: list[int],
) -> str:
    """Repair malformed JSON into V2 contract with no semantic rewrite."""
    repair_prompt = (
        "Convert the malformed output into valid JSON array.\n"
        f"Expected ids: {expected_ids}\n"
        "Keep keys: id,label,impact_score,confidence,relevance,event_type,reasoning.\n"
        "Do not add explanations.\n\n"
        f"MALFORMED:\n{malformed_content}"
    )
    repair_data = await create_chat_completion(
        api_key=settings.openrouter_api_key,
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You repair JSON only. Return valid JSON array without markdown.",
            },
            {"role": "user", "content": repair_prompt},
        ],
        max_tokens=2600,
        temperature=0.0,
        timeout_seconds=60.0,
        max_retries=settings.openrouter_max_retries,
        rpm=settings.openrouter_rpm,
        fallback_models=settings.openrouter_fallback_models_list,
        referer="https://copper-mind.vercel.app",
        title="CopperMind V2 JSON Repair",
        extra_payload={"reasoning": {"exclude": True}},
    )
    repaired_content = _extract_chat_message_content(repair_data)
    if not repaired_content:
        raise LLMStructuredOutputError("V2 JSON repair returned empty content")
    return repaired_content


async def _score_subset_with_model_v2(
    *,
    settings: Any,
    model_name: str,
    articles: list[dict],
    horizon_days: int,
) -> tuple[dict[int, dict], list[int], int]:
    """
    Score subset with one model.

    Returns:
        (valid_results_by_id, failed_ids, parse_fail_count)
    """
    if not articles:
        return {}, [], 0

    expected_ids = [int(article["id"]) for article in articles]
    user_prompt = _build_llm_v2_user_prompt(articles, horizon_days=horizon_days)

    async def _request(*, max_tokens: int) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "api_key": settings.openrouter_api_key,
            "model": model_name,
            "messages": [
                {"role": "system", "content": LLM_V2_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "timeout_seconds": 60.0,
            "max_retries": settings.openrouter_max_retries,
            "rpm": settings.openrouter_rpm,
            "fallback_models": settings.openrouter_fallback_models_list,
            "referer": "https://copper-mind.vercel.app",
            "title": "CopperMind Sentiment Analysis V2",
            "response_format": LLM_SCORING_RESPONSE_FORMAT_V2,
            "extra_payload": {"reasoning": {"exclude": True}},
        }
        return await create_chat_completion(**request_kwargs)

    parse_fail_count = 0
    try:
        data = await _request(max_tokens=LLM_SCORING_MAX_TOKENS_PRIMARY)
    except Exception:
        return {}, expected_ids, len(expected_ids)

    content = _extract_chat_message_content(data)
    if not content:
        finish_reason = _extract_finish_reason(data)
        if finish_reason == "length":
            data = await _request(max_tokens=LLM_SCORING_MAX_TOKENS_RETRY)
            content = _extract_chat_message_content(data)
        if not content:
            return {}, expected_ids, len(expected_ids)

    try:
        raw_results = json.loads(_clean_json_content(content))
        valid, failed = _parse_llm_v2_items(
            raw_results=raw_results,
            expected_ids=expected_ids,
            model_name=model_name,
        )
        parse_fail_count += len(failed)
        return valid, failed, parse_fail_count
    except Exception as exc:
        logger.warning(
            "V2 JSON parse failed for model=%s: %s | response_preview=%.500s",
            model_name, exc, content,
        )
        parse_fail_count += len(expected_ids)

    try:
        repaired = await _repair_json_response_v2(
            settings=settings,
            model_name=model_name,
            malformed_content=content,
            expected_ids=expected_ids,
        )
        raw_results = json.loads(_clean_json_content(repaired))
        valid, failed = _parse_llm_v2_items(
            raw_results=raw_results,
            expected_ids=expected_ids,
            model_name=model_name,
        )
        return valid, failed, parse_fail_count
    except Exception:
        return {}, expected_ids, parse_fail_count


async def score_batch_with_llm_v2(
    articles: list[dict],
    *,
    horizon_days: int = 5,
) -> dict[str, Any]:
    """
    Commodity-aware sentiment scoring with fast+reliable model escalation.
    """
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError("OpenRouter API key not configured")

    fast_model = settings.resolved_scoring_fast_model
    reliable_model = settings.resolved_scoring_reliable_model
    conflict_threshold = _clip(
        float(getattr(settings, "sentiment_escalate_conflict_threshold", 0.55)),
        0.0,
        1.0,
    )

    normalized_articles = [
        {
            "id": int(article["id"]),
            "title": str(article.get("title") or "")[:500],
            "description": str(article.get("description") or "")[:800],
            "text": str(article.get("text") or "")[:1800],
        }
        for article in articles
    ]
    expected_ids = [item["id"] for item in normalized_articles]
    article_by_id = {item["id"]: item for item in normalized_articles}

    fast_valid, fast_failed, parse_fail_fast = await _score_subset_with_model_v2(
        settings=settings,
        model_name=fast_model,
        articles=normalized_articles,
        horizon_days=horizon_days,
    )

    results_by_id = dict(fast_valid)
    parse_fail_total = int(parse_fail_fast)

    conflict_ids: list[int] = []
    for article_id, item in fast_valid.items():
        event_type = _normalize_event_type(item.get("event_type"))
        rule_sign = int(LLM_V2_EVENT_SIGN.get(event_type, 0))
        llm_sign = _sign(float(item.get("impact_score", 0.0)))
        conflict_strength = _clip(
            float(item.get("confidence", 0.0)) * float(item.get("relevance", 0.0)),
            0.0,
            1.0,
        )
        if rule_sign != 0 and llm_sign != 0 and llm_sign != rule_sign and conflict_strength >= conflict_threshold:
            conflict_ids.append(article_id)
            results_by_id.pop(article_id, None)

    escalation_ids = sorted(set(fast_failed).union(conflict_ids))
    escalation_count = len(escalation_ids)

    if escalation_ids:
        reliable_subset = [
            article_by_id[article_id]
            for article_id in escalation_ids
            if article_id in article_by_id
        ]
        reliable_valid, _reliable_failed, parse_fail_reliable = await _score_subset_with_model_v2(
            settings=settings,
            model_name=reliable_model,
            articles=reliable_subset,
            horizon_days=horizon_days,
        )
        results_by_id.update(reliable_valid)
        parse_fail_total += int(parse_fail_reliable)

    results = [results_by_id[article_id] for article_id in expected_ids if article_id in results_by_id]
    failed_ids = [article_id for article_id in expected_ids if article_id not in results_by_id]
    fallback_count = len(failed_ids)

    return {
        "results": results,
        "failed_ids": failed_ids,
        "parse_fail_count": parse_fail_total,
        "escalation_count": escalation_count,
        "fallback_count": fallback_count,
        "model_fast": fast_model,
        "model_reliable": reliable_model,
    }


def score_batch_with_finbert_v2(articles: list[dict]) -> dict[int, dict]:
    """Score text with FinBERT for tone/intensity features."""
    pipe = get_finbert_pipeline()
    results: dict[int, dict] = {}

    for article in articles:
        article_id = int(article["id"])
        text = str(
            article.get("text")
            or f"{article.get('title', '')} {article.get('description', '')}"
        )[:1200]
        scores = score_text_with_finbert(pipe, text)
        results[article_id] = {
            "prob_positive": float(scores["prob_positive"]),
            "prob_neutral": float(scores["prob_neutral"]),
            "prob_negative": float(scores["prob_negative"]),
            "tone": float(scores["score"]),
            "magnitude": abs(float(scores["prob_positive"]) - float(scores["prob_negative"])),
        }

    return results


def compute_final_score_v2(
    *,
    impact_score_llm: float,
    confidence_llm: float,
    relevance_score: float,
    event_type: str,
    prob_positive: float,
    prob_negative: float,
) -> dict[str, float | int]:
    """Compute deterministic ensemble score for V2."""
    llm_impact = _clip(float(impact_score_llm), -1.0, 1.0)
    llm_conf = _clip(float(confidence_llm), 0.0, 1.0)
    relevance = _clip(float(relevance_score), 0.0, 1.0)
    tone = _clip(float(prob_positive) - float(prob_negative), -1.0, 1.0)
    tone_mag = abs(tone)

    normalized_event = _normalize_event_type(event_type)
    rule_sign = int(LLM_V2_EVENT_SIGN.get(normalized_event, 0))
    rule_strength = float(LLM_V2_EVENT_STRENGTH.get(normalized_event, 0.25))

    llm_sign = _sign(llm_impact)
    final_sign = llm_sign if llm_sign != 0 else rule_sign
    if final_sign == 0 and tone_mag >= 0.2:
        final_sign = _sign(tone)

    impact_mag = _clip(
        (0.55 * abs(llm_impact))
        + (0.25 * tone_mag)
        + (0.20 * _clip(rule_strength, 0.0, 1.0)),
        0.0,
        1.0,
    )
    if final_sign == 0:
        impact_mag = min(impact_mag, 0.12)

    final_score = float(final_sign) * impact_mag
    agreement = 1.0 if (rule_sign == 0 or llm_sign == 0 or llm_sign == rule_sign) else 0.4
    confidence_cal = _clip((0.50 * llm_conf) + (0.30 * agreement) + (0.20 * relevance), 0.01, 0.99)

    return {
        "rule_sign": rule_sign,
        "rule_strength": rule_strength,
        "final_score": final_score,
        "confidence_calibrated": confidence_cal,
    }


def _build_article_fallback_v2(
    *,
    article: dict,
    finbert: dict,
    model_fast: str,
    model_reliable: str,
) -> dict:
    """Deterministic article-level fallback without zero-only outputs."""
    text = str(article.get("text") or f"{article.get('title', '')} {article.get('description', '')}")
    event_type = _infer_event_type_from_text(text)
    rule_sign = int(LLM_V2_EVENT_SIGN.get(event_type, 0))
    tone = float(finbert.get("tone", 0.0))
    tone_sign = _sign(tone)
    direction = rule_sign if rule_sign != 0 else tone_sign
    if direction == 0:
        impact_score = 0.0
    else:
        impact_score = float(direction) * _clip((abs(tone) * 0.35) + 0.08, 0.08, 0.25)

    relevance = 0.10 if event_type == "non_copper" else 0.45
    confidence = 0.18 if direction == 0 else _clip(0.22 + (abs(tone) * 0.22), 0.22, 0.45)

    return {
        "id": int(article["id"]),
        "label": _label_from_impact_score(impact_score),
        "impact_score": impact_score,
        "confidence": confidence,
        "relevance": relevance,
        "event_type": event_type,
        "reasoning": "deterministic_fallback",
        "llm_model": model_fast,
        "model_fast": model_fast,
        "model_reliable": model_reliable,
        "fallback_used": True,
    }


def score_unscored_processed_articles(
    session: Session,
    *,
    chunk_size: int = 12,
    backfill_days: Optional[int] = None,
) -> dict[str, int]:
    """
    Score unscored `news_processed` articles into `news_sentiments_v2`.
    """
    settings = get_settings()
    horizon_days = max(1, int(getattr(settings, "sentiment_horizon_days", 5)))
    relevance_min = _clip(float(getattr(settings, "sentiment_relevance_min", 0.35)), 0.0, 1.0)

    query = (
        session.query(
            NewsProcessed.id.label("processed_id"),
            NewsProcessed.canonical_title,
            NewsProcessed.cleaned_text,
            NewsRaw.title.label("raw_title"),
            NewsRaw.description.label("raw_description"),
            NewsRaw.published_at,
        )
        .join(NewsRaw, NewsProcessed.raw_id == NewsRaw.id)
        .outerjoin(
            NewsSentimentV2,
            (NewsProcessed.id == NewsSentimentV2.news_processed_id)
            & (NewsSentimentV2.horizon_days == horizon_days),
        )
        .filter(NewsSentimentV2.id.is_(None))
        .order_by(NewsRaw.published_at.asc(), NewsProcessed.id.asc())
    )

    if backfill_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(backfill_days)))
        query = query.filter(NewsRaw.published_at >= cutoff)

    rows = query.all()
    if not rows:
        logger.info("No unscored processed articles found")
        return {
            "scored_count": 0,
            "parse_fail_count": 0,
            "escalation_count": 0,
            "fallback_count": 0,
            "finbert_used": 0,
        }

    logger.info("Found %s unscored processed articles for V2 scoring", len(rows))

    scored_count = 0
    parse_fail_count = 0
    escalation_count = 0
    fallback_count = 0
    finbert_used = 0
    llm_budget_remaining = max(0, int(settings.max_llm_articles_per_run))
    fast_model = settings.resolved_scoring_fast_model
    reliable_model = settings.resolved_scoring_reliable_model

    for chunk_idx in range(0, len(rows), chunk_size):
        chunk_rows = rows[chunk_idx:chunk_idx + chunk_size]
        chunk_items: list[dict] = []
        for row in chunk_rows:
            title = str(row.raw_title or row.canonical_title or "")[:500]
            description = str(row.raw_description or "")[:1000]
            text = str(row.cleaned_text or f"{title} {description}")[:2000]
            chunk_items.append(
                {
                    "id": int(row.processed_id),
                    "title": title,
                    "description": description,
                    "text": text,
                    "published_at": row.published_at,
                }
            )

        finbert_by_id = score_batch_with_finbert_v2(chunk_items)
        finbert_used += len(finbert_by_id)

        llm_results_by_id: dict[int, dict] = {}
        llm_candidates: list[dict] = []
        if settings.openrouter_api_key and llm_budget_remaining > 0:
            llm_take = min(len(chunk_items), llm_budget_remaining)
            llm_candidates = chunk_items[:llm_take]
            llm_budget_remaining -= llm_take

        if llm_candidates:
            try:
                llm_bundle = run_async_from_sync(
                    score_batch_with_llm_v2,
                    llm_candidates,
                    horizon_days=horizon_days,
                )
                for item in llm_bundle.get("results", []):
                    llm_results_by_id[int(item["id"])] = item
                parse_fail_count += int(llm_bundle.get("parse_fail_count", 0))
                escalation_count += int(llm_bundle.get("escalation_count", 0))
                fast_model = str(llm_bundle.get("model_fast", fast_model))
                reliable_model = str(llm_bundle.get("model_reliable", reliable_model))
            except Exception as exc:
                logger.warning("V2 LLM scoring failed for chunk starting at %s: %s", chunk_idx, exc)
                parse_fail_count += len(llm_candidates)

        for article in chunk_items:
            article_id = int(article["id"])
            finbert = finbert_by_id.get(article_id, _neutral_finbert_score())
            llm = llm_results_by_id.get(article_id)

            if llm is None:
                llm = _build_article_fallback_v2(
                    article=article,
                    finbert=finbert if isinstance(finbert, dict) else {},
                    model_fast=fast_model,
                    model_reliable=reliable_model,
                )
            else:
                llm["model_fast"] = fast_model
                llm["model_reliable"] = reliable_model
                llm["fallback_used"] = False

            if bool(llm.get("fallback_used", False)):
                fallback_count += 1

            if float(llm.get("relevance", 0.0)) < relevance_min and llm.get("event_type") != "non_copper":
                llm["event_type"] = "non_copper"
                llm["label"] = "NEUTRAL"
                llm["impact_score"] = 0.0

            metrics = compute_final_score_v2(
                impact_score_llm=float(llm.get("impact_score", 0.0)),
                confidence_llm=float(llm.get("confidence", 0.01)),
                relevance_score=float(llm.get("relevance", 0.01)),
                event_type=str(llm.get("event_type", "mixed_unclear")),
                prob_positive=float(finbert.get("prob_positive", 0.33)),
                prob_negative=float(finbert.get("prob_negative", 0.33)),
            )

            payload = {
                "label": llm.get("label", "NEUTRAL"),
                "impact_score": round(float(llm.get("impact_score", 0.0)), 4),
                "confidence": round(float(llm.get("confidence", 0.01)), 4),
                "relevance": round(float(llm.get("relevance", 0.01)), 4),
                "event_type": llm.get("event_type", "mixed_unclear"),
                "reasoning": llm.get("reasoning", ""),
                "rule_sign": metrics["rule_sign"],
                "rule_strength": round(float(metrics["rule_strength"]), 4),
                "confidence_calibrated": round(float(metrics["confidence_calibrated"]), 4),
                "fallback_used": bool(llm.get("fallback_used", False)),
                "llm_model": llm.get("llm_model", fast_model),
                "scoring_version": SCORING_V2_VERSION,
            }

            sentiment_v2 = NewsSentimentV2(
                news_processed_id=article_id,
                horizon_days=horizon_days,
                label=str(llm.get("label", "NEUTRAL")),
                impact_score_llm=float(llm.get("impact_score", 0.0)),
                confidence_llm=float(llm.get("confidence", 0.01)),
                confidence_calibrated=float(metrics["confidence_calibrated"]),
                relevance_score=float(llm.get("relevance", 0.01)),
                event_type=str(llm.get("event_type", "mixed_unclear")),
                rule_sign=int(metrics["rule_sign"]),
                final_score=float(metrics["final_score"]),
                finbert_pos=float(finbert.get("prob_positive", 0.33)),
                finbert_neu=float(finbert.get("prob_neutral", 0.34)),
                finbert_neg=float(finbert.get("prob_negative", 0.33)),
                reasoning_json=json.dumps(payload, ensure_ascii=True),
                model_fast=fast_model,
                model_reliable=reliable_model,
                scored_at=datetime.now(timezone.utc),
            )
            session.add(sentiment_v2)
            scored_count += 1

        session.commit()

    logger.info(
        "V2 scoring summary: scored=%s parse_fail=%s escalations=%s fallback=%s finbert_used=%s",
        scored_count,
        parse_fail_count,
        escalation_count,
        fallback_count,
        finbert_used,
    )
    return {
        "scored_count": scored_count,
        "parse_fail_count": parse_fail_count,
        "escalation_count": escalation_count,
        "fallback_count": fallback_count,
        "finbert_used": finbert_used,
    }


def aggregate_daily_sentiment_v2(
    session: Session,
    *,
    tau_hours: float = 12.0,
) -> int:
    """Aggregate V2 article scores into daily_sentiments_v2."""
    settings = get_settings()
    tau_hours = tau_hours or settings.sentiment_tau_hours
    horizon_days = max(1, int(getattr(settings, "sentiment_horizon_days", 5)))
    relevance_min = _clip(float(getattr(settings, "sentiment_relevance_min", 0.35)), 0.0, 1.0)

    rows = (
        session.query(
            NewsRaw.published_at,
            NewsSentimentV2.final_score,
            NewsSentimentV2.confidence_calibrated,
            NewsSentimentV2.relevance_score,
        )
        .join(NewsProcessed, NewsProcessed.raw_id == NewsRaw.id)
        .join(
            NewsSentimentV2,
            (NewsSentimentV2.news_processed_id == NewsProcessed.id)
            & (NewsSentimentV2.horizon_days == horizon_days),
        )
        .filter(NewsSentimentV2.relevance_score >= relevance_min)
        .all()
    )

    if not rows:
        logger.info("No V2 scored articles available for daily aggregation")
        return 0

    df = pd.DataFrame(
        rows,
        columns=["published_at", "final_score", "confidence_calibrated", "relevance_score"],
    )
    df["date"] = pd.to_datetime(df["published_at"]).dt.normalize()

    def calc_weights(group):
        hours = (group["published_at"] - group["date"]).dt.total_seconds() / 3600.0
        weights = np.exp(hours / tau_hours)
        return weights / weights.sum()

    daily_rows = []
    for date, group in df.groupby("date"):
        weights = calc_weights(group)
        daily_rows.append(
            {
                "date": date,
                "sentiment_index": float((group["final_score"] * weights).sum()),
                "news_count": int(len(group)),
                "avg_confidence": float(group["confidence_calibrated"].mean()),
                "avg_relevance": float(group["relevance_score"].mean()),
            }
        )

    count = 0
    for row in daily_rows:
        date_dt = row["date"].to_pydatetime()
        if date_dt.tzinfo is None:
            date_dt = date_dt.replace(tzinfo=timezone.utc)

        existing = session.query(DailySentimentV2).filter(
            func.date(DailySentimentV2.date) == func.date(date_dt)
        ).first()
        if existing:
            existing.sentiment_index = row["sentiment_index"]
            existing.news_count = row["news_count"]
            existing.avg_confidence = row["avg_confidence"]
            existing.avg_relevance = row["avg_relevance"]
            existing.source_version = "v2"
            existing.aggregated_at = datetime.now(timezone.utc)
        else:
            session.add(
                DailySentimentV2(
                    date=date_dt,
                    sentiment_index=row["sentiment_index"],
                    news_count=row["news_count"],
                    avg_confidence=row["avg_confidence"],
                    avg_relevance=row["avg_relevance"],
                    source_version="v2",
                    aggregated_at=datetime.now(timezone.utc),
                )
            )
        count += 1

    session.commit()
    logger.info("Aggregated V2 sentiment for %s days", count)
    return count


def backfill_sentiment_v2(
    session: Session,
    *,
    days: int = 180,
    batch_size: int = 50,
) -> dict[str, int]:
    """Idempotent V2 backfill helper for last N days."""
    logger.info("Starting V2 backfill for last %s days (batch_size=%s)", days, batch_size)
    return score_unscored_processed_articles(
        session=session,
        chunk_size=batch_size,
        backfill_days=days,
    )


def score_unscored_articles(
    session: Session,
    chunk_size: int = 12
) -> int:
    """
    Score all articles that don't have sentiment scores yet.
    
    Strategy:
    - Primary direction: OpenRouter LLM label + confidence
    - Intensity: FinBERT probabilities for every article
    - Non-neutral boost: (0.7*llm_conf + 0.3*finbert_strength) * boost
    - Soft neutral: NEUTRAL labels can emit small directional score from FinBERT polarity
    - Chunk size: 12 articles for lower free-tier rate-limit pressure
    - Run budget: cap LLM-scored articles per run, overflow uses FinBERT
    
    Returns:
        Number of articles scored
    """
    settings = get_settings()
    
    # Find unscored articles
    unscored = session.query(NewsArticle).outerjoin(
        NewsSentiment,
        NewsArticle.id == NewsSentiment.news_article_id
    ).filter(NewsSentiment.id.is_(None)).all()
    
    if not unscored:
        logger.info("No unscored articles found")
        return 0
    
    logger.info(f"Found {len(unscored)} unscored articles")
    
    scored_count = 0
    total_chunks = (len(unscored) + chunk_size - 1) // chunk_size
    llm_model = settings.resolved_scoring_model
    llm_budget_remaining = max(0, settings.max_llm_articles_per_run)
    non_neutral_boost = float(getattr(settings, "sentiment_non_neutral_boost", 1.35))
    soft_neutral_polarity_threshold = float(
        getattr(settings, "sentiment_soft_neutral_polarity_threshold", 0.12)
    )
    soft_neutral_max_mag = float(getattr(settings, "sentiment_soft_neutral_max_mag", 0.25))
    soft_neutral_scale = float(getattr(settings, "sentiment_soft_neutral_scale", 0.8))
    budget_exhausted_logged = False
    logger.info("LLM scoring budget for this run: %s articles", llm_budget_remaining)
    llm_success = 0
    json_repair_success = 0
    fallback_429 = 0
    fallback_parse = 0
    finbert_used = 0
    
    # Process in chunks
    for chunk_idx in range(0, len(unscored), chunk_size):
        chunk = unscored[chunk_idx:chunk_idx + chunk_size]
        chunk_num = chunk_idx // chunk_size + 1
        
        logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} articles)")

        llm_candidates: list[Any] = []
        non_llm_candidates: list[Any] = []

        if settings.openrouter_api_key and llm_budget_remaining > 0:
            llm_take = min(len(chunk), llm_budget_remaining)
            llm_candidates = chunk[:llm_take]
            non_llm_candidates = chunk[llm_take:]
        else:
            non_llm_candidates = chunk
            if settings.openrouter_api_key and llm_budget_remaining <= 0 and not budget_exhausted_logged:
                logger.info(
                    "LLM budget exhausted (%s articles). Remaining chunks use soft-neutral FinBERT fallback.",
                    settings.max_llm_articles_per_run,
                )
                budget_exhausted_logged = True

        finbert_results = score_batch_with_finbert(chunk)
        finbert_used += len(finbert_results)
        finbert_by_id = {result["id"]: result for result in finbert_results}

        llm_results_by_id: dict[int, dict] = {}

        if llm_candidates:
            articles_data = [
                {"id": a.id, "title": a.title, "description": a.description}
                for a in llm_candidates
            ]
            try:
                llm_results = run_async_from_sync(score_batch_with_llm, articles_data)
                llm_success += len(llm_results)
                json_repair_success += sum(1 for result in llm_results if result.get("json_repair_used"))
                for llm_result in llm_results:
                    llm_results_by_id[int(llm_result["id"])] = llm_result
                llm_budget_remaining -= len(llm_candidates)
                logger.info(
                    "LLM scored %s article(s) in chunk %s. Budget remaining: %s",
                    len(llm_candidates),
                    chunk_num,
                    llm_budget_remaining,
                )
            except Exception as e:
                if _is_rate_limit_error(e):
                    fallback_model_name = HYBRID_FALLBACK_429_MODEL_NAME
                    fallback_reason = "429 rate-limit fallback"
                    fallback_429 += len(llm_candidates)
                else:
                    fallback_model_name = HYBRID_FALLBACK_PARSE_MODEL_NAME
                    fallback_reason = "structured parse/repair fallback"
                    fallback_parse += len(llm_candidates)
                logger.warning("LLM scoring failed for chunk %s: %s", chunk_num, e)
                for article in llm_candidates:
                    llm_results_by_id[article.id] = _neutral_llm_result(
                        article_id=article.id,
                        llm_model=llm_model,
                        reason=fallback_reason,
                        model_name_override=fallback_model_name,
                    )

        for article in non_llm_candidates:
            reason = (
                "llm_skipped_budget"
                if settings.openrouter_api_key and llm_budget_remaining <= 0
                else "llm_unavailable_no_api_key"
            )
            llm_results_by_id[article.id] = _neutral_llm_result(
                article_id=article.id,
                llm_model=llm_model,
                reason=reason,
                model_name_override=_hybrid_model_name(llm_model),
            )
        
        # Save to database
        for article in chunk:
            finbert = finbert_by_id.get(article.id)
            if not finbert:
                neutral_finbert = _neutral_finbert_score()
                finbert = {
                    "prob_positive": neutral_finbert["prob_positive"],
                    "prob_neutral": neutral_finbert["prob_neutral"],
                    "prob_negative": neutral_finbert["prob_negative"],
                    "finbert_strength": abs(
                        neutral_finbert["prob_positive"] - neutral_finbert["prob_negative"]
                    ),
                }

            llm_result = llm_results_by_id.get(article.id)
            if not llm_result:
                llm_result = _neutral_llm_result(
                    article_id=article.id,
                    llm_model=llm_model,
                    reason="llm_result_missing",
                    model_name_override=HYBRID_FALLBACK_PARSE_MODEL_NAME,
                )
                fallback_parse += 1

            label = str(llm_result.get("label", "NEUTRAL")).upper().strip()
            if label not in LLM_LABELS:
                label = "NEUTRAL"

            llm_confidence = float(llm_result.get("llm_confidence", 0.0))
            finbert_polarity = float(finbert["prob_positive"]) - float(finbert["prob_negative"])
            finbert_strength = float(
                finbert.get(
                    "finbert_strength",
                    abs(finbert_polarity),
                )
            )
            final_score, soft_neutral_applied = _compute_hybrid_score(
                label=label,
                llm_confidence=llm_confidence,
                finbert_strength=finbert_strength,
                finbert_polarity=finbert_polarity,
                non_neutral_boost=non_neutral_boost,
                soft_neutral_polarity_threshold=soft_neutral_polarity_threshold,
                soft_neutral_max_mag=soft_neutral_max_mag,
                soft_neutral_scale=soft_neutral_scale,
                return_metadata=True,
            )

            reasoning_payload = _build_hybrid_reasoning_payload(
                label=label,
                llm_confidence=llm_confidence,
                finbert_strength=finbert_strength,
                finbert_polarity=finbert_polarity,
                llm_reasoning=llm_result.get("llm_reasoning", ""),
                llm_model=llm_result.get("llm_model", llm_model),
                soft_neutral_applied=soft_neutral_applied,
            )
            
            sentiment = NewsSentiment(
                news_article_id=article.id,
                prob_positive=float(finbert["prob_positive"]),
                prob_neutral=float(finbert["prob_neutral"]),
                prob_negative=float(finbert["prob_negative"]),
                score=float(final_score),
                reasoning=reasoning_payload,
                model_name=str(llm_result.get("model_name", _hybrid_model_name(llm_model))),
                scored_at=datetime.now(timezone.utc)
            )
            
            session.add(sentiment)
            scored_count += 1
        
        # Commit after each chunk
        session.commit()
        logger.info(f"Committed chunk {chunk_num}: {len(chunk)} articles")
    
    logger.info(
        "Hybrid scoring summary: llm_success=%s json_repair_success=%s fallback_429=%s "
        "fallback_parse=%s finbert_used=%s",
        llm_success,
        json_repair_success,
        fallback_429,
        fallback_parse,
        finbert_used,
    )
    logger.info(f"Total articles scored: {scored_count}")
    return scored_count


# =============================================================================
# Daily Sentiment Aggregation
# =============================================================================

def aggregate_daily_sentiment(
    session: Session,
    tau_hours: float = 12.0
) -> int:
    """
    Aggregate sentiment scores by day with recency weighting.
    
    Weighting formula: w = exp(-(hours_since_publish) / tau)
    
    Returns:
        Number of days aggregated
    """
    settings = get_settings()
    tau_hours = tau_hours or settings.sentiment_tau_hours
    
    # Get scored articles with copper keyword filter to reduce symbol-unrelated noise.
    copper_filter = (
        func.lower(NewsArticle.title).like("%copper%")
        | func.lower(func.coalesce(NewsArticle.description, "")).like("%copper%")
    )

    scored_articles = session.query(
        NewsArticle.published_at,
        NewsSentiment.score,
        NewsSentiment.prob_positive,
        NewsSentiment.prob_neutral,
        NewsSentiment.prob_negative
    ).join(
        NewsSentiment,
        NewsArticle.id == NewsSentiment.news_article_id
    ).filter(
        copper_filter
    ).all()
    
    if not scored_articles:
        logger.info("No copper-filtered scored articles for aggregation")
        return 0
    
    # Convert to DataFrame
    df = pd.DataFrame(scored_articles, columns=[
        "published_at", "score", "prob_positive", "prob_neutral", "prob_negative"
    ])
    
    # Extract date
    df["date"] = pd.to_datetime(df["published_at"]).dt.normalize()
    
    # Calculate recency weight within each day
    # Higher weight for articles later in the day (closer to market close)
    def calc_weights(group):
        # Hours since start of day
        hours = (group["published_at"] - group["date"]).dt.total_seconds() / 3600
        # Exponential weighting: later = higher weight
        weights = np.exp(hours / tau_hours)
        return weights / weights.sum()  # Normalize
    
    # Group by date and aggregate
    daily_data = []
    
    for date, group in df.groupby("date"):
        weights = calc_weights(group)
        
        # Convert numpy types to native Python types for database compatibility
        daily_data.append({
            "date": date,
            "sentiment_index": float((group["score"] * weights).sum()),
            "news_count": int(len(group)),
            "avg_positive": float(group["prob_positive"].mean()),
            "avg_neutral": float(group["prob_neutral"].mean()),
            "avg_negative": float(group["prob_negative"].mean()),
        })
    
    # Upsert daily sentiments
    count = 0
    for row in daily_data:
        date_dt = row["date"].to_pydatetime()
        if date_dt.tzinfo is None:
            date_dt = date_dt.replace(tzinfo=timezone.utc)
        
        # Check if exists
        existing = session.query(DailySentiment).filter(
            func.date(DailySentiment.date) == func.date(date_dt)
        ).first()
        
        if existing:
            # Update
            existing.sentiment_index = row["sentiment_index"]
            existing.news_count = row["news_count"]
            existing.avg_positive = row["avg_positive"]
            existing.avg_neutral = row["avg_neutral"]
            existing.avg_negative = row["avg_negative"]
            existing.aggregated_at = datetime.now(timezone.utc)
        else:
            # Insert
            daily = DailySentiment(
                date=date_dt,
                sentiment_index=row["sentiment_index"],
                news_count=row["news_count"],
                avg_positive=row["avg_positive"],
                avg_neutral=row["avg_neutral"],
                avg_negative=row["avg_negative"],
                weighting_method="recency_exponential",
                aggregated_at=datetime.now(timezone.utc)
            )
            session.add(daily)
        
        count += 1
    
    session.commit()
    logger.info(f"Aggregated sentiment for {count} days")
    
    return count


# =============================================================================
# XGBoost Model Training
# =============================================================================

def train_xgboost_model(
    session: Session,
    target_symbol: str = "HG=F",
    lookback_days: int = 365,
    validation_days: int = 30,
    early_stopping_rounds: int = 10
) -> Optional[dict]:
    """
    Train XGBoost model for price prediction.
    
    Target: Next-day return (more stationary, avoids direct price level issues)
    
    Returns:
        Dict with model path, metrics, and feature importance
    """
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Build feature matrix
    X, y = build_feature_matrix(
        session,
        target_symbol=target_symbol,
        lookback_days=lookback_days
    )
    
    if X.empty or len(X) < 50:
        logger.error(f"Insufficient data for training: {len(X)} samples")
        return None
    
    # Time-series split: last N days for validation
    split_date = X.index.max() - timedelta(days=validation_days)
    
    train_mask = X.index <= split_date
    val_mask = X.index > split_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    if len(X_train) < 30 or len(X_val) < 5:
        logger.error("Not enough samples for train/val split")
        return None
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # XGBoost parameters - tuned for overfitting prevention
    # With 250 samples / 198 features, we need strong regularization
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 4,              # Shallower trees = less memorization
        "learning_rate": 0.05,       # Slower learning = better generalization
        "subsample": 0.8,
        "colsample_bytree": 0.6,     # Use fewer features per tree
        "min_child_weight": 5,       # Require more samples per leaf
        "reg_alpha": 0.5,            # L1 regularization (sparsity)
        "reg_lambda": 2.0,           # L2 regularization (smoothness)
        "seed": 42,
    }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    # Train with early stopping
    evals = [(dtrain, "train"), (dval, "val")]
    
    logger.info("Training XGBoost model...")
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10  # More frequent logging
    )
    
    # Evaluate
    y_pred_train = model.predict(dtrain)
    y_pred_val = model.predict(dval)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    
    logger.info(f"Training MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}")
    logger.info(f"Validation MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
    
    # Feature importance
    importance = model.get_score(importance_type="gain")
    
    # Sort by importance
    sorted_importance = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Normalize importance
    total_importance = sum(v for _, v in sorted_importance)
    normalized_importance = [
        {"feature": k, "importance": v / total_importance}
        for k, v in sorted_importance
    ]
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_{timestamp}.json"
    model.save_model(str(model_path))
    
    # Save latest symlink/copy
    latest_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.json"
    model.save_model(str(latest_path))
    
    # Save metrics (including training symbols audit)
    # TARGET_TYPE: "simple_return" means model predicts next-day return, not price
    # This MUST be read by inference to correctly compute predicted_price
    metrics = {
        "target_symbol": target_symbol,
        # Target definition audit (prevents semantic confusion)
        "target_type": "simple_return",  # Model predicts: close(t+1)/close(t) - 1
        "target_shift_days": 1,  # Predict 1 day ahead
        "target_definition": "simple_return(close,1).shift(-1)",  # Exact pandas formula
        "baseline_price_source": "yfinance_close",  # Which close normalizes returns
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "best_iteration": model.best_iteration,
        "feature_count": len(feature_names),
        # Audit: which symbols were used for training
        "symbol_set_name": settings.symbol_set,
        "training_symbols": settings.training_symbols,
        "training_symbols_hash": settings.training_symbols_hash,
        "training_symbols_source": settings.training_symbols_source,
    }
    
    metrics_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature names
    features_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.features.json"
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    
    # Save importance (Overwrite to reflect the latest model training)
    importance_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.importance.json"
    with open(importance_path, "w") as f:
        json.dump(normalized_importance, f, indent=2)
    
    logger.info(f"Model saved to: {model_path}")
    
    # Log top influencers
    logger.info("Top 10 feature influencers:")
    descriptions = get_feature_descriptions()
    for item in normalized_importance[:10]:
        feat = item["feature"]
        imp = item["importance"]
        desc = descriptions.get(feat, feat)
        logger.info(f"  {feat}: {imp:.4f} ({desc})")
    
    # Save metadata to database for persistence across HF Space restarts
    try:
        from app.db import SessionLocal
        with SessionLocal() as session:
            save_model_metadata_to_db(
                session=session,
                symbol=target_symbol,
                importance=normalized_importance,
                features=feature_names,
                metrics=metrics,
            )
    except Exception as e:
        logger.warning(f"Could not save model metadata to DB: {e}")
    
    return {
        "model_path": str(model_path),
        "metrics": metrics,
        "top_influencers": normalized_importance[:10],
        "all_features": feature_names,
    }


def load_model(target_symbol: str = "HG=F") -> Optional[xgb.Booster]:
    """Load the latest trained model for a symbol."""
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    
    model_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.json"
    
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None
    
    model = xgb.Booster()
    model.load_model(str(model_path))
    
    return model


def save_model_metadata_to_db(
    session,
    symbol: str,
    importance: list,
    features: list,
    metrics: dict
) -> None:
    """
    Save model metadata to database for persistence across restarts.
    Called after train_model=True pipeline runs.
    """
    from .models import ModelMetadata
    from datetime import datetime
    
    # Try to find existing record
    existing = session.query(ModelMetadata).filter(ModelMetadata.symbol == symbol).first()
    
    if existing:
        existing.importance_json = json.dumps(importance)
        existing.features_json = json.dumps(features)
        existing.metrics_json = json.dumps(metrics)
        existing.trained_at = datetime.now(timezone.utc)
        logger.info(f"Updated model metadata in DB for {symbol}")
    else:
        new_record = ModelMetadata(
            symbol=symbol,
            importance_json=json.dumps(importance),
            features_json=json.dumps(features),
            metrics_json=json.dumps(metrics),
        )
        session.add(new_record)
        logger.info(f"Saved new model metadata to DB for {symbol}")
    
    session.commit()


def load_model_metadata_from_db(session, symbol: str) -> dict:
    """
    Load model metadata from database.
    Returns dict with importance, features, metrics or None values if not found.
    """
    from .models import ModelMetadata
    
    metadata = {
        "metrics": None,
        "features": None,
        "importance": None,
    }
    
    record = session.query(ModelMetadata).filter(ModelMetadata.symbol == symbol).first()
    
    if record:
        try:
            if record.importance_json:
                metadata["importance"] = json.loads(record.importance_json)
            if record.features_json:
                metadata["features"] = json.loads(record.features_json)
            if record.metrics_json:
                metadata["metrics"] = json.loads(record.metrics_json)
            logger.info(f"Loaded model metadata from DB for {symbol}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse model metadata from DB: {e}")
    
    return metadata


def load_model_metadata(target_symbol: str = "HG=F") -> dict:
    """
    Load metrics and feature info for a model.
    
    Priority:
    1. Database (survives HF Space restarts)
    2. Local JSON files (fallback for development)
    """
    from app.db import SessionLocal
    
    # Try database first
    try:
        with SessionLocal() as session:
            db_metadata = load_model_metadata_from_db(session, target_symbol)
            if db_metadata.get("importance") and db_metadata.get("features"):
                return db_metadata
    except Exception as e:
        logger.debug(f"Could not load metadata from DB: {e}")
    
    # Fallback to local files
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    
    prefix = f"xgb_{target_symbol.replace('=', '_')}_latest"
    
    metadata = {
        "metrics": None,
        "features": None,
        "importance": None,
    }
    
    # Load metrics
    metrics_path = model_dir / f"{prefix}.metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metadata["metrics"] = json.load(f)
    
    # Load features
    features_path = model_dir / f"{prefix}.features.json"
    if features_path.exists():
        with open(features_path) as f:
            metadata["features"] = json.load(f)
    
    # Load importance
    importance_path = model_dir / f"{prefix}.importance.json"
    if importance_path.exists():
        with open(importance_path) as f:
            metadata["importance"] = json.load(f)
    
    return metadata


# =============================================================================
# Main Entry Point
# =============================================================================

def run_full_pipeline(
    target_symbol: str = "HG=F",
    score_sentiment: bool = True,
    aggregate_sentiment: bool = True,
    train_model: bool = True
) -> dict:
    """
    Run the full AI pipeline.
    
    Returns:
        Dict with results from each stage
    """
    settings = get_settings()
    results = {
        "scored_articles": 0,
        "scored_articles_v2": 0,
        "aggregated_days": 0,
        "aggregated_days_v2": 0,
        "model_result": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    with SessionLocal() as session:
        if score_sentiment:
            if settings.scoring_source == "news_processed":
                scoring_stats = score_unscored_processed_articles(session)
                results["scored_articles"] = int(scoring_stats.get("scored_count", 0))
                results["scored_articles_v2"] = int(scoring_stats.get("scored_count", 0))
                results["llm_parse_fail_count"] = int(scoring_stats.get("parse_fail_count", 0))
                results["escalation_count"] = int(scoring_stats.get("escalation_count", 0))
                results["fallback_count"] = int(scoring_stats.get("fallback_count", 0))
            else:
                results["scored_articles"] = score_unscored_articles(session)
        
        if aggregate_sentiment:
            if settings.scoring_source == "news_processed":
                results["aggregated_days_v2"] = aggregate_daily_sentiment_v2(session)
            results["aggregated_days"] = aggregate_daily_sentiment(session)
        
        if train_model:
            results["model_result"] = train_xgboost_model(
                session,
                target_symbol=target_symbol
            )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run AI pipeline: LLM sentiment scoring (with FinBERT fallback) and XGBoost training"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run full pipeline (score + aggregate + train)"
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Only run sentiment scoring (LLM primary, FinBERT fallback)"
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only run sentiment aggregation"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only run XGBoost training"
    )
    parser.add_argument(
        "--refresh-sentiment",
        action="store_true",
        help="Run sentiment scoring + daily aggregation (no training)"
    )
    parser.add_argument(
        "--backfill-v2-days",
        type=int,
        default=0,
        help="Backfill unscored V2 sentiment for last N days (idempotent)"
    )
    parser.add_argument(
        "--backfill-v2-batch-size",
        type=int,
        default=50,
        help="Batch size for V2 backfill mode"
    )
    parser.add_argument(
        "--target-symbol",
        type=str,
        default="HG=F",
        help="Target symbol for training (default: HG=F)"
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip pipeline lock (for testing)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine what to run
    score = args.run_all or args.score_only or args.refresh_sentiment
    aggregate = args.run_all or args.aggregate_only or args.refresh_sentiment
    train = args.run_all or args.train_only
    backfill_v2 = args.backfill_v2_days > 0
    
    if not (score or aggregate or train or backfill_v2):
        parser.print_help()
        return
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Run pipeline
    def do_run():
        if backfill_v2:
            with SessionLocal() as session:
                scoring_stats = backfill_sentiment_v2(
                    session,
                    days=args.backfill_v2_days,
                    batch_size=max(1, int(args.backfill_v2_batch_size)),
                )
                aggregated_days_v2 = aggregate_daily_sentiment_v2(session)
                return {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "scored_articles": int(scoring_stats.get("scored_count", 0)),
                    "scored_articles_v2": int(scoring_stats.get("scored_count", 0)),
                    "aggregated_days": 0,
                    "aggregated_days_v2": int(aggregated_days_v2),
                    "llm_parse_fail_count": int(scoring_stats.get("parse_fail_count", 0)),
                    "escalation_count": int(scoring_stats.get("escalation_count", 0)),
                    "fallback_count": int(scoring_stats.get("fallback_count", 0)),
                    "model_result": None,
                    "backfill_days": int(args.backfill_v2_days),
                }
        return run_full_pipeline(
            target_symbol=args.target_symbol,
            score_sentiment=score,
            aggregate_sentiment=aggregate,
            train_model=train
        )
    
    if args.no_lock:
        results = do_run()
    else:
        try:
            with pipeline_lock():
                results = do_run()
        except RuntimeError as e:
            logger.error(f"Could not acquire lock: {e}")
            return
    
    # Print summary
    print("\n" + "=" * 50)
    print("AI PIPELINE SUMMARY")
    print("=" * 50)
    
    if score:
        print(f"\nSentiment Scoring: {results['scored_articles']} articles")
        if "scored_articles_v2" in results:
            print(f"V2 Sentiment Scoring: {results.get('scored_articles_v2', 0)} articles")
        if "llm_parse_fail_count" in results:
            print(f"  - LLM parse failures: {results.get('llm_parse_fail_count', 0)}")
            print(f"  - Escalations: {results.get('escalation_count', 0)}")
            print(f"  - Deterministic fallbacks: {results.get('fallback_count', 0)}")
    
    if aggregate:
        print(f"Daily Aggregation: {results['aggregated_days']} days")
        if results.get("aggregated_days_v2") is not None:
            print(f"Daily Aggregation V2 (shadow): {results.get('aggregated_days_v2', 0)} days")

    if backfill_v2:
        print(f"\nBackfill V2 Days: {results.get('backfill_days', 0)}")
        print(f"V2 Aggregation Days: {results.get('aggregated_days_v2', 0)}")
    
    if train and results.get("model_result"):
        mr = results["model_result"]
        metrics = mr.get("metrics", {})
        print(f"\nModel Training:")
        print(f"  - Validation MAE: {metrics.get('val_mae', 'N/A'):.6f}")
        print(f"  - Validation RMSE: {metrics.get('val_rmse', 'N/A'):.6f}")
        print(f"  - Model saved to: {mr.get('model_path', 'N/A')}")
        
        print("\nTop Influencers:")
        for item in mr.get("top_influencers", [])[:5]:
            print(f"  - {item['feature']}: {item['importance']:.4f}")
    
    print(f"\nTimestamp: {results.get('timestamp', 'N/A')}")


if __name__ == "__main__":
    main()

