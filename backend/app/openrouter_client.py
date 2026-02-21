"""
Shared OpenRouter client with retry, throttling, and model fallback support.
"""

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
    """Base error raised for OpenRouter client failures."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class OpenRouterRateLimitError(OpenRouterError):
    """Raised when OpenRouter rate limiting persists after retries."""


def _parse_retry_after_seconds(response: httpx.Response) -> Optional[float]:
    """Parse Retry-After header in seconds if provided."""
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        seconds = float(value)
        return max(seconds, 0.0)
    except ValueError:
        return None


def _build_model_payload(primary_model: str, fallback_models: Optional[list[str]]) -> dict[str, Any]:
    """
    Build model payload for OpenRouter.
    Uses `models` only when fallback models are provided.
    """
    if not fallback_models:
        return {"model": primary_model}

    ordered: list[str] = []
    for model in [primary_model, *fallback_models]:
        if model and model not in ordered:
            ordered.append(model)

    if len(ordered) == 1:
        return {"model": ordered[0]}

    return {"models": ordered}


async def _throttle_request(rpm: int) -> None:
    """
    Global soft-throttle shared across all OpenRouter requests in this process.
    """
    if rpm <= 0:
        return

    min_interval = 60.0 / float(rpm)
    now = time.monotonic()
    wait_seconds = 0.0

    global _NEXT_ALLOWED_TS
    with _RATE_LOCK:
        if now < _NEXT_ALLOWED_TS:
            wait_seconds = _NEXT_ALLOWED_TS - now
            _NEXT_ALLOWED_TS += min_interval
        else:
            _NEXT_ALLOWED_TS = now + min_interval

    if wait_seconds > 0:
        logger.debug("OpenRouter throttle wait: %.3fs", wait_seconds)
        await asyncio.sleep(wait_seconds)


async def create_chat_completion(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout_seconds: float = 60.0,
    max_retries: int = 3,
    rpm: int = 18,
    response_format: Optional[dict[str, Any]] = None,
    provider: Optional[dict[str, Any]] = None,
    fallback_models: Optional[list[str]] = None,
    referer: Optional[str] = None,
    title: Optional[str] = None,
    extra_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Call OpenRouter chat completions with retry/backoff and soft throttling.

    Retry policy:
    - Retry on 429 and 5xx
    - Retry on transient network errors
    - Delay: Retry-After (if present) else 2^attempt + jitter(0..0.5)
    """
    if not api_key:
        raise OpenRouterError("OpenRouter API key not configured")

    payload: dict[str, Any] = {
        **_build_model_payload(model, fallback_models),
        "messages": messages,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format is not None:
        payload["response_format"] = response_format
    if provider is not None:
        payload["provider"] = provider
    if extra_payload:
        payload.update(extra_payload)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for attempt in range(max_retries + 1):
            await _throttle_request(rpm)
            try:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
            except httpx.RequestError as exc:
                if attempt >= max_retries:
                    raise OpenRouterError(
                        f"OpenRouter request failed after retries: {exc}"
                    ) from exc

                retry_num = attempt + 1
                delay = float(2 ** retry_num) + random.uniform(0.0, 0.5)
                logger.warning(
                    "OpenRouter network error (attempt %s/%s). Retrying in %.2fs: %s",
                    retry_num,
                    max_retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise OpenRouterError("OpenRouter returned non-JSON response body") from exc

            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if retryable and attempt < max_retries:
                retry_num = attempt + 1
                retry_after = _parse_retry_after_seconds(response)
                delay = retry_after if retry_after is not None else float(2 ** retry_num) + random.uniform(0.0, 0.5)
                logger.warning(
                    "OpenRouter retryable error status=%s (attempt %s/%s). Retrying in %.2fs",
                    response.status_code,
                    retry_num,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            body_preview = response.text[:500]
            if response.status_code == 429:
                raise OpenRouterRateLimitError(
                    f"OpenRouter rate limit exceeded after retries: {body_preview}",
                    status_code=response.status_code,
                )
            raise OpenRouterError(
                f"OpenRouter API error: {response.status_code} - {body_preview}",
                status_code=response.status_code,
            )

    raise OpenRouterError("OpenRouter request unexpectedly terminated")
