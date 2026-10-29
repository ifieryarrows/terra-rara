"""Tests for IFM (K2-Horizon) client."""

import pytest
from types import SimpleNamespace

from app.ifm_client import IFMError, create_ifm_chat_completion, get_configured_ifm_key


def test_get_configured_ifm_key():
    settings_with_key = SimpleNamespace(ifm_api_key="  test-ifm-key-123  ")
    assert get_configured_ifm_key(settings_with_key) == "test-ifm-key-123"

    settings_empty = SimpleNamespace(ifm_api_key="   ")
    assert get_configured_ifm_key(settings_empty) is None

    settings_none = SimpleNamespace(ifm_api_key=None)
    assert get_configured_ifm_key(settings_none) is None


@pytest.mark.asyncio
async def test_create_ifm_chat_completion_requires_key():
    settings = SimpleNamespace(ifm_api_key=None)
    with pytest.raises(IFMError) as exc_info:
        await create_ifm_chat_completion(
            [{"role": "user", "content": "hi"}],
            settings=settings,
        )
    assert exc_info.value.category == "auth"


@pytest.mark.asyncio
async def test_create_ifm_chat_completion_success(monkeypatch):
    settings = SimpleNamespace(
        ifm_api_key="test-key",
        ifm_model="IFM/K2-Horizon-375B-A23B",
        ifm_base_url="https://api.ifm.ai/v1",
        ifm_timeout_seconds=30.0,
        ifm_reasoning_effort="low",
    )

    class DummyResponse:
        status_code = 200
        text = '{"choices":[{"message":{"content":"{\\"stance\\":\\"BULLISH\\",\\"commentary\\":\\"test\\"}"}}]}'

        def json(self):
            import json
            return json.loads(self.text)

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json, headers):
            assert url == "https://api.ifm.ai/v1/chat/completions"
            assert headers["Authorization"] == "Bearer test-key"
            assert json["model"] == "IFM/K2-Horizon-375B-A23B"
            assert json["reasoning_effort"] == "low"
            return DummyResponse()

    from app import ifm_client
    monkeypatch.setattr(ifm_client.httpx, "AsyncClient", DummyClient)

    content, data = await create_ifm_chat_completion(
        [{"role": "user", "content": "test"}],
        settings=settings,
    )
    assert '{"stance":"BULLISH"' in content
    assert data["choices"][0]["message"]["content"] == content
