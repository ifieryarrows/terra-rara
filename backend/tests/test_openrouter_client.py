"""
Tests for shared OpenRouter client behavior.
"""

import asyncio

import httpx
import pytest

from app import openrouter_client as orc


def _make_response(status_code: int, *, json_body=None, text_body: str = "", headers=None) -> httpx.Response:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    if json_body is not None:
        return httpx.Response(status_code, json=json_body, headers=headers, request=request)
    return httpx.Response(status_code, text=text_body, headers=headers, request=request)


def _patch_async_client(monkeypatch, responses, calls):
    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            calls.append({"url": url, "headers": headers, "json": json})
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(orc.httpx, "AsyncClient", DummyAsyncClient)


def test_retry_after_small_429_respects_header_under_cap(monkeypatch):
    """
    A valid Retry-After is honored while the client enforces a 30-second cap.
    """
    responses = [
        _make_response(429, text_body="rate limit", headers={"Retry-After": "1"}),
        _make_response(200, json_body={"choices": [{"message": {"content": "ok"}}]}),
    ]
    calls = []
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(orc.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(orc.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(orc, "_NEXT_ALLOWED_TS", 0.0)
    _patch_async_client(monkeypatch, responses, calls)

    async def run_call():
        return await orc.create_chat_completion(
            api_key="test-key",
            model="primary-model",
            messages=[{"role": "user", "content": "hello"}],
            rpm=0,
            max_retries=3,
        )

    result = asyncio.run(run_call())

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 2
    assert sleeps and sleeps[0] == pytest.approx(1.0)


def test_retry_after_large_429_is_capped_at_30_seconds(monkeypatch):
    responses = [
        _make_response(429, text_body="rate limit", headers={"Retry-After": "120"}),
        _make_response(200, json_body={"choices": [{"message": {"content": "ok"}}]}),
    ]
    calls = []
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(orc.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(orc.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(orc, "_NEXT_ALLOWED_TS", 0.0)
    _patch_async_client(monkeypatch, responses, calls)

    async def run_call():
        return await orc.create_chat_completion(
            api_key="test-key",
            model="primary-model",
            messages=[{"role": "user", "content": "hello"}],
            rpm=0,
            max_retries=3,
        )

    asyncio.run(run_call())
    assert sleeps and sleeps[0] == pytest.approx(30.0)


def test_rate_limit_error_after_max_retries(monkeypatch):
    responses = [
        _make_response(429, text_body="rate limit"),
        _make_response(429, text_body="rate limit"),
        _make_response(429, text_body="rate limit"),
        _make_response(429, text_body="rate limit"),
    ]
    calls = []

    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(orc.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(orc.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(orc, "_NEXT_ALLOWED_TS", 0.0)
    _patch_async_client(monkeypatch, responses, calls)

    async def run_call():
        await orc.create_chat_completion(
            api_key="test-key",
            model="primary-model",
            messages=[{"role": "user", "content": "hello"}],
            rpm=0,
            max_retries=3,
        )

    with pytest.raises(orc.OpenRouterRateLimitError):
        asyncio.run(run_call())

    assert len(calls) == 4


def test_models_payload_with_fallbacks_and_provider(monkeypatch):
    responses = [
        _make_response(200, json_body={"choices": [{"message": {"content": "ok"}}]}),
    ]
    calls = []

    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(orc.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(orc, "_NEXT_ALLOWED_TS", 0.0)
    _patch_async_client(monkeypatch, responses, calls)

    async def run_call():
        await orc.create_chat_completion(
            api_key="test-key",
            model="primary-model",
            fallback_models=["fallback-a", "fallback-b"],
            provider={"require_parameters": True},
            response_format={"type": "json_schema"},
            messages=[{"role": "user", "content": "hello"}],
            rpm=0,
            max_retries=0,
        )

    asyncio.run(run_call())

    payload = calls[0]["json"]
    assert payload["model"] == "primary-model"
    assert "models" not in payload
    assert payload["provider"] == {"require_parameters": True}
    assert payload["response_format"] == {"type": "json_schema"}


def test_404_advances_to_next_model_without_retry(monkeypatch):
    responses = [
        _make_response(404, text_body="model unavailable"),
        _make_response(200, json_body={"model": "fallback-a", "choices": [{"message": {"content": "ok"}}]}),
    ]
    calls = []

    async def fake_sleep(_delay):
        raise AssertionError("404 must not sleep/retry")

    monkeypatch.setattr(orc.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(orc, "_NEXT_ALLOWED_TS", 0.0)
    _patch_async_client(monkeypatch, responses, calls)

    result = asyncio.run(orc.create_chat_completion(
        api_key="test-key",
        model="removed-model",
        fallback_models=["fallback-a"],
        messages=[{"role": "user", "content": "hello"}],
        rpm=0,
        max_retries=1,
    ))
    assert [call["json"]["model"] for call in calls] == ["removed-model", "fallback-a"]
    assert result["_coppermind"]["actual_model"] == "fallback-a"


def test_auth_error_aborts_whole_model_chain(monkeypatch):
    responses = [_make_response(401, text_body="invalid key")]
    calls = []
    monkeypatch.setattr(orc, "_NEXT_ALLOWED_TS", 0.0)
    _patch_async_client(monkeypatch, responses, calls)

    with pytest.raises(orc.OpenRouterError) as exc_info:
        asyncio.run(orc.create_chat_completion(
            api_key="bad-key",
            model="primary",
            fallback_models=["fallback"],
            messages=[{"role": "user", "content": "hello"}],
            rpm=0,
        ))
    assert exc_info.value.category == "auth"
    assert len(calls) == 1


def test_provider_5xx_retries_once_then_advances_model(monkeypatch):
    responses = [
        _make_response(503, text_body="provider unavailable"),
        _make_response(503, text_body="provider unavailable"),
        _make_response(200, json_body={"model": "fallback", "choices": [{"message": {"content": "ok"}}]}),
    ]
    calls = []

    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(orc.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(orc, "_NEXT_ALLOWED_TS", 0.0)
    _patch_async_client(monkeypatch, responses, calls)

    result = asyncio.run(orc.create_chat_completion(
        api_key="test-key",
        model="primary",
        fallback_models=["fallback"],
        messages=[{"role": "user", "content": "hello"}],
        rpm=0,
        max_retries=1,
    ))
    assert [call["json"]["model"] for call in calls] == ["primary", "primary", "fallback"]
    assert result["_coppermind"]["actual_model"] == "fallback"


def test_timeout_is_classified_and_advances_model(monkeypatch):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    responses = [
        httpx.ReadTimeout("slow provider", request=request),
        _make_response(200, json_body={"model": "fallback", "choices": [{"message": {"content": "ok"}}]}),
    ]
    calls = []
    monkeypatch.setattr(orc, "_NEXT_ALLOWED_TS", 0.0)
    _patch_async_client(monkeypatch, responses, calls)

    result = asyncio.run(orc.create_chat_completion(
        api_key="test-key",
        model="primary",
        fallback_models=["fallback"],
        messages=[{"role": "user", "content": "hello"}],
        rpm=0,
        max_retries=0,
    ))
    assert result["_coppermind"]["attempts"][0]["category"] == "timeout"
    assert result["_coppermind"]["actual_model"] == "fallback"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (400, "maximum context length exceeded", "context_limit"),
        (422, "json_schema response_format unsupported", "unsupported_contract"),
        (404, "model unavailable", "model_unavailable"),
    ],
)
def test_http_failure_classification(status, body, expected):
    assert orc._classify_status(status, body) == expected


def test_rpm_throttle_waits_between_calls(monkeypatch):
    responses = [
        _make_response(200, json_body={"choices": [{"message": {"content": "ok-1"}}]}),
    ]
    calls = []
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(orc.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(orc, "_NEXT_ALLOWED_TS", orc.time.monotonic() + 0.25)
    _patch_async_client(monkeypatch, responses, calls)

    async def run_call():
        await orc.create_chat_completion(
            api_key="test-key",
            model="primary-model",
            messages=[{"role": "user", "content": "hello"}],
            rpm=120,
            max_retries=0,
        )

    asyncio.run(run_call())

    assert any(delay > 0 for delay in sleeps)
    assert len(calls) == 1
