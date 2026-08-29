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


async def create_chat_completion(
    *,
    api_key: str,
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
    """Call each model explicitly so failure category and actual model are known."""
    if not api_key:
        raise OpenRouterError("OpenRouter API key not configured", category="auth")
    models = _ordered_models(model, fallback_models)
    if not models:
        raise OpenRouterError("No OpenRouter model configured", category="model_unavailable")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
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
    retryable_categories = {"rate_limit", "timeout", "network", "provider_5xx"}

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for candidate in models:
            for attempt_index in range(max(0, int(max_retries)) + 1):
                if time.monotonic() >= deadline:
                    raise OpenRouterError("OpenRouter model chain deadline exceeded", category="timeout", model=candidate, attempts=attempts)
                await _throttle_request(rpm)
                payload = {**base_payload, "model": candidate}
                response: Optional[httpx.Response] = None
                try:
                    response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                except httpx.TimeoutException as exc:
                    last_error = OpenRouterError(str(exc), category="timeout", model=candidate)
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
                            else:
                                attempts.append(_attempt(candidate, attempt_index + 1, "success", 200, attempt_index > 0))
                                data["_coppermind"] = {"requested_model": candidate, "actual_model": data.get("model") or candidate, "attempts": attempts}
                                return data
                    else:
                        body = response.text[:1000]
                        category = _classify_status(response.status_code, body)
                        error_cls = OpenRouterRateLimitError if category == "rate_limit" else OpenRouterError
                        last_error = error_cls(f"OpenRouter {category}: HTTP {response.status_code} - {body[:500]}", status_code=response.status_code, category=category, model=candidate)

                status = response.status_code if response is not None else None
                retryable = last_error is not None and last_error.category in retryable_categories
                will_retry = retryable and attempt_index < max(0, int(max_retries))
                attempts.append(_attempt(candidate, attempt_index + 1, last_error.category if last_error else "network", status, will_retry))
                if last_error and last_error.category == "auth":
                    last_error.attempts = attempts
                    raise last_error
                if will_retry:
                    retry_after = _parse_retry_after_seconds(response) if response is not None else None
                    delay = min(retry_after if retry_after is not None else (2 ** attempt_index) + random.uniform(0.0, 0.5), 30.0)
                    remaining = max(deadline - time.monotonic(), 0.0)
                    if delay >= remaining:
                        break
                    await asyncio.sleep(delay)
                    continue
                break
            logger.warning("OpenRouter model failed; advancing chain model=%s category=%s", candidate, last_error.category if last_error else "unknown")

    if last_error is None:
        last_error = OpenRouterError("OpenRouter request unexpectedly terminated", category="network")
    last_error.attempts = attempts
    raise last_error
