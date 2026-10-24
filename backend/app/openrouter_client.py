"""Bounded, observable OpenRouter client with client-controlled model fallback."""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)
_RATE_LOCK = threading.Lock()
_NEXT_ALLOWED_TS = 0.0


class OpenRouterError(RuntimeError):
    """Classified OpenRouter failure safe to surface in pipeline metadata."""

    def __init__(self, message: str, status_code: Optional[int] = None, *, category: str = "network", model: Optional[str] = None, attempts: Optional[list[dict[str, Any]]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.category = category
        self.model = model
        self.attempts = attempts or []


class OpenRouterRateLimitError(OpenRouterError):
    """Compatibility subtype for callers that treat exhausted quota specially."""


def _ordered_models(primary: str, fallbacks: Optional[list[str]]) -> list[str]:
    ordered: list[str] = []
    for candidate in [primary, *(fallbacks or [])]:
        candidate = str(candidate or "").strip()
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _parse_retry_after_seconds(response: httpx.Response) -> Optional[float]:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _classify_status(status: int, body: str) -> str:
    lowered = body.lower()
    if status in (401, 403):
        return "auth"
    if status == 402:
        return "spend_limit"
    if status == 404:
        return "model_unavailable"
    if status == 408:
        return "timeout"
    if status == 429:
        return "rate_limit"
    if 500 <= status < 600:
        return "provider_5xx"
    if status in (400, 413) and any(token in lowered for token in ("context", "token", "maximum length")):
        return "context_limit"
    if status in (400, 422) and any(token in lowered for token in ("response_format", "json_schema", "unsupported")):
        return "unsupported_contract"
    return "unsupported_contract"


def _is_free_tier_daily_quota_exhausted(status: int, body: str, model: str) -> bool:
    """Return True when a 429 indicates the account-wide :free daily quota is exhausted."""
    if status != 429 or not str(model or "").lower().endswith(":free"):
        return False
    lowered = (body or "").lower()
    return "openrouter_free_tier_daily" in lowered or "free-models-per-day" in lowered


def _log_rate_limit_headers(response: httpx.Response, model: str) -> None:
    remaining = response.headers.get("X-Ratelimit-Remaining") or response.headers.get("x-ratelimit-remaining")
    limit = response.headers.get("X-Ratelimit-Limit") or response.headers.get("x-ratelimit-limit")
    reset = response.headers.get("X-Ratelimit-Reset") or response.headers.get("x-ratelimit-reset")
    if remaining is not None or limit is not None or reset is not None:
        logger.info("OpenRouter quota model=%s remaining=%s limit=%s reset=%s", model, remaining, limit, reset)


async def _throttle_request(rpm: int) -> None:
    if rpm <= 0:
        return
    minimum_interval = 60.0 / float(rpm)
    now = time.monotonic()
    wait_seconds = 0.0
    global _NEXT_ALLOWED_TS
    with _RATE_LOCK:
        if now < _NEXT_ALLOWED_TS:
            wait_seconds = _NEXT_ALLOWED_TS - now
            _NEXT_ALLOWED_TS += minimum_interval
        else:
            _NEXT_ALLOWED_TS = now + minimum_interval
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)


def _attempt(model: str, ordinal: int, category: str, status: Optional[int], retried: bool) -> dict[str, Any]:
    return {"model": model, "attempt": ordinal, "category": category, "status_code": status, "retried": retried}


def get_configured_api_keys(settings: Any) -> list[str]:
    """Return configured OpenRouter keys without exposing them to logs."""
    configured = getattr(settings, "openrouter_api_keys", None)
    candidates = configured if configured is not None else [
        getattr(settings, "openrouter_api_key", None),
        getattr(settings, "openrouter_api_key_backup", None),
    ]
    if isinstance(candidates, str):
        candidates = [candidates]
    keys: list[str] = []
    for candidate in candidates or []:
        normalized = str(candidate or "").strip()
        if normalized and normalized not in keys:
            keys.append(normalized)
    return keys


def _api_key_slot(index: int) -> str:
    return "primary" if index == 0 else f"backup-{index}"


def _extract_message_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _candidate_payload(base_payload: dict[str, Any], candidate: str) -> dict[str, Any]:
    payload = {**base_payload, "model": candidate}
    existing_reasoning = payload.get("reasoning")
    if candidate.lower().startswith("nvidia/nemotron-3.5-lightning"):
        # This model is the low-cost scoring route. Explicitly disable its
        # thinking mode; a low reasoning effort can consume the whole output
        # budget without returning the JSON scoring payload.
        payload["reasoning"] = {"enabled": False, "exclude": True}
    elif candidate.lower().startswith("nvidia/nemotron-3.5-super-vl"):
        if isinstance(existing_reasoning, dict) and existing_reasoning.get("enabled") is False:
            payload["reasoning"] = {"enabled": False, "exclude": True}
        else:
            reasoning = dict(existing_reasoning) if isinstance(existing_reasoning, dict) else {}
            reasoning.setdefault("effort", "low")
            payload["reasoning"] = reasoning
    elif candidate.lower().startswith("nvidia/nemotron"):
        if isinstance(existing_reasoning, dict) and "effort" in existing_reasoning:
            payload["reasoning"] = {"effort": existing_reasoning["effort"]}
        else:
            payload["reasoning"] = {"effort": "low"}
    elif isinstance(existing_reasoning, dict) and set(existing_reasoning.keys()) == {"effort"}:
        payload.pop("reasoning", None)
    return payload


async def create_chat_completion(
    *,
    api_key: Optional[str] = None,
    api_keys: Optional[list[str]] = None,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout_seconds: float = 45.0,
    max_retries: int = 1,
    rpm: int = 18,
    response_format: Optional[dict[str, Any]] = None,
    provider: Optional[dict[str, Any]] = None,
    fallback_models: Optional[list[str]] = None,
    referer: Optional[str] = None,
    title: Optional[str] = None,
    extra_payload: Optional[dict[str, Any]] = None,
    chain_deadline_seconds: float = 120.0,
) -> dict[str, Any]:
    """Try model fallbacks and rotate keys on quota or spend-limit responses."""
    configured_keys: list[str] = []
    for candidate in [api_key, *(api_keys or [])]:
        normalized = str(candidate or "").strip()
        if normalized and normalized not in configured_keys:
            configured_keys.append(normalized)
    if not configured_keys:
        raise OpenRouterError("OpenRouter API key not configured", category="auth")
    models = _ordered_models(model, fallback_models)
    if not models:
        raise OpenRouterError("No OpenRouter model configured", category="model_unavailable")

    base_payload: dict[str, Any] = {"messages": messages}
    if max_tokens is not None:
        base_payload["max_tokens"] = max_tokens
    if temperature is not None:
        base_payload["temperature"] = temperature
    if response_format is not None:
        base_payload["response_format"] = response_format
    if provider is not None:
        base_payload["provider"] = provider
    if extra_payload:
        base_payload.update(extra_payload)

    attempts: list[dict[str, Any]] = []
    deadline = time.monotonic() + max(float(chain_deadline_seconds), 1.0)
    last_error: Optional[OpenRouterError] = None
    first_rate_limit_error: Optional[OpenRouterRateLimitError] = None
    free_tier_daily_exhausted_keys: set[int] = set()
    retryable_categories = {"rate_limit", "timeout", "network", "provider_5xx"}
    key_rotation_categories = {"rate_limit", "spend_limit"}

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for candidate in models:
            candidate_is_free = candidate.lower().endswith(":free")
            if candidate_is_free and len(free_tier_daily_exhausted_keys) == len(configured_keys):
                logger.warning("OpenRouter free-tier daily quota exhausted for all configured keys; skipping model=%s", candidate)
                continue
            attempted_for_model = False
            for key_index, active_key in enumerate(configured_keys):
                if candidate_is_free and key_index in free_tier_daily_exhausted_keys:
                    continue
                attempted_for_model = True
                key_slot = _api_key_slot(key_index)
                headers = {"Authorization": f"Bearer {active_key}", "Content-Type": "application/json"}
                if referer:
                    headers["HTTP-Referer"] = referer
                if title:
                    headers["X-Title"] = title

                rotate_key = False
                daily_quota_exhausted = False
                for attempt_index in range(max(0, int(max_retries)) + 1):
                    if time.monotonic() >= deadline:
                        raise OpenRouterError("OpenRouter model chain deadline exceeded", category="timeout", model=candidate, attempts=attempts)
                    await _throttle_request(rpm)
                    remaining_before_call = deadline - time.monotonic()
                    if remaining_before_call <= 0:
                        raise OpenRouterError("OpenRouter model chain deadline exceeded", category="timeout", model=candidate, attempts=attempts)
                    attempt_timeout = min(float(timeout_seconds), max(remaining_before_call, 0.1))
                    payload = _candidate_payload(base_payload, candidate)
                    response: Optional[httpx.Response] = None
                    try:
                        response = await asyncio.wait_for(
                            client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload),
                            timeout=attempt_timeout,
                        )
                    except (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError) as exc:
                        last_error = OpenRouterError(str(exc) or "OpenRouter request timed out", category="timeout", model=candidate)
                    except httpx.RequestError as exc:
                        last_error = OpenRouterError(str(exc), category="network", model=candidate)
                    else:
                        if response.status_code == 200:
                            _log_rate_limit_headers(response, candidate)
                            try:
                                data = response.json()
                            except ValueError:
                                last_error = OpenRouterError("OpenRouter returned a non-JSON response", status_code=200, category="empty_response", model=candidate)
                            else:
                                if not data.get("choices"):
                                    last_error = OpenRouterError("OpenRouter returned no completion choices", status_code=200, category="empty_response", model=candidate)
                                elif not _extract_message_text(data):
                                    last_error = OpenRouterError("OpenRouter returned empty message content", status_code=200, category="empty_response", model=candidate)
                                else:
                                    attempts.append(_attempt(candidate, attempt_index + 1, "success", 200, attempt_index > 0) | {"key_slot": key_slot})
                                    data["_coppermind"] = {
                                        "requested_model": candidate,
                                        "actual_model": data.get("model") or candidate,
                                        "key_slot": key_slot,
                                        "attempts": attempts,
                                    }
                                    return data
                        else:
                            body = response.text[:1000]
                            category = _classify_status(response.status_code, body)
                            error_cls = OpenRouterRateLimitError if category == "rate_limit" else OpenRouterError
                            last_error = error_cls(f"OpenRouter {category}: HTTP {response.status_code} - {body[:500]}", status_code=response.status_code, category=category, model=candidate)
                            if isinstance(last_error, OpenRouterRateLimitError) and first_rate_limit_error is None:
                                first_rate_limit_error = last_error
                            if _is_free_tier_daily_quota_exhausted(response.status_code, body, candidate):
                                daily_quota_exhausted = True
                                free_tier_daily_exhausted_keys.add(key_index)
                                logger.warning(
                                    "OpenRouter free-tier daily quota exhausted; skipping this key for remaining free models in this request key_slot=%s",
                                    key_slot,
                                )

                    status = response.status_code if response is not None else None
                    retryable = last_error is not None and last_error.category in retryable_categories
                    # When fallback models exist, advance immediately on a timeout instead of
                    # burning a second 45s window on the same unresponsive upstream model.
                    skip_same_model_timeout_retry = (
                        last_error is not None
                        and last_error.category == "timeout"
                        and len(models) > 1
                    )
                    will_retry = (
                        retryable
                        and not skip_same_model_timeout_retry
                        and not daily_quota_exhausted
                        and not (
                            last_error
                            and last_error.category == "rate_limit"
                            and key_index < len(configured_keys) - 1
                        )
                        and attempt_index < max(0, int(max_retries))
                    )
                    attempts.append(
                        _attempt(candidate, attempt_index + 1, last_error.category if last_error else "network", status, will_retry)
                        | {"key_slot": key_slot}
                    )
                    if last_error and last_error.category == "auth":
                        last_error.attempts = attempts
                        raise last_error
                    if daily_quota_exhausted:
                        break
                    if last_error and last_error.category in key_rotation_categories and key_index < len(configured_keys) - 1:
                        logger.warning(
                            "OpenRouter key quota/spend limit reached; advancing to backup key model=%s category=%s key_slot=%s",
                            candidate,
                            last_error.category,
                            key_slot,
                        )
                        rotate_key = True
                        break
                    if will_retry:
                        retry_after = _parse_retry_after_seconds(response) if response is not None else None
                        delay = min(retry_after if retry_after is not None else (2 ** attempt_index) + random.uniform(0.0, 0.5), 30.0)
                        remaining = max(deadline - time.monotonic(), 0.0)
                        if delay >= remaining:
                            break
                        await asyncio.sleep(delay)
                        continue
                    break
                if rotate_key:
                    continue

            if not attempted_for_model and candidate_is_free:
                logger.warning("OpenRouter daily free-tier quota exhausted for all keys; skipping model=%s", candidate)
            else:
                logger.warning("OpenRouter model failed; advancing chain model=%s category=%s", candidate, last_error.category if last_error else "unknown")

    if first_rate_limit_error is not None and not isinstance(last_error, OpenRouterRateLimitError):
        last_error = first_rate_limit_error
    if last_error is None:
        last_error = OpenRouterRateLimitError(
            "OpenRouter free-tier daily quota exhausted for all configured API keys",
            status_code=429,
            category="rate_limit",
        )
    last_error.attempts = attempts
    raise last_error
