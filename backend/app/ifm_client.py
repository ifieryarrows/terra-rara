"""Bounded, observable IFM (Institute of Foundation Models) client for K2-Horizon."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

from .settings import Settings, get_settings

logger = logging.getLogger(__name__)


class IFMError(RuntimeError):
    """Classified IFM failure safe to surface in pipeline metadata."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        *,
        category: str = "network",
        model: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.category = category
        self.model = model


def get_configured_ifm_key(settings: Optional[Settings] = None) -> Optional[str]:
    """Return the IFM API key if configured."""
    active_settings = settings or get_settings()
    key = getattr(active_settings, "ifm_api_key", None)
    if isinstance(key, str) and key.strip():
        return key.strip()
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
    if status in (400, 422) and any(token in lowered for token in ("response_format", "json_schema", "unsupported")):
        return "unsupported_contract"
    return "network"


async def create_ifm_chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1500,
    response_format: Optional[dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    settings: Optional[Settings] = None,
) -> tuple[str, dict[str, Any]]:
    """Execute a chat completion request to IFM API.

    Returns:
        tuple[str, dict]: (content_text, full_response_json)
    """
    active_settings = settings or get_settings()
    api_key = get_configured_ifm_key(active_settings)
    if not api_key:
        raise IFMError("IFM API key is not configured", category="auth")

    target_model = model or getattr(active_settings, "ifm_model", "IFM/K2-Horizon-375B-A23B")
    base_url = (getattr(active_settings, "ifm_base_url", None) or "https://api.ifm.ai/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    timeout = timeout_seconds or getattr(active_settings, "ifm_timeout_seconds", 45.0)
    effort = reasoning_effort or getattr(active_settings, "ifm_reasoning_effort", "low")

    payload: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if effort and effort in ("low", "medium", "high"):
        payload["reasoning_effort"] = effort
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "CopperMind-Backend/1.0",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
    except (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError) as exc:
        raise IFMError(f"IFM request timed out after {timeout}s: {exc}", category="timeout", model=target_model) from exc
    except httpx.RequestError as exc:
        raise IFMError(f"IFM network failure: {exc}", category="network", model=target_model) from exc

    status = response.status_code
    body = response.text
    if status != 200:
        category = _classify_status(status, body)
        raise IFMError(
            f"IFM request failed ({status}): {body[:250]}",
            status_code=status,
            category=category,
            model=target_model,
        )

    try:
        data = response.json()
    except Exception as exc:
        raise IFMError(f"IFM response is not valid JSON: {body[:200]}", status_code=status, category="malformed_json", model=target_model) from exc

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise IFMError("IFM response contained no choices", status_code=status, category="empty_response", model=target_model)

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        content = content.strip()
    elif isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        content = "\n".join(parts).strip()
    else:
        content = str(content or "").strip()

    if not content:
        raise IFMError("IFM choice message content was empty", status_code=status, category="empty_response", model=target_model)

    return content, data
