"""
Tests for commentary generation and fallback behavior.
"""

import asyncio
from types import SimpleNamespace

from app import commentary as commentary_module
from app.openrouter_client import OpenRouterError


def test_generate_commentary_uses_template_fallback_on_openrouter_error(monkeypatch):
    fake_settings = SimpleNamespace(
        openrouter_api_key="test-key",
        resolved_commentary_model="stepfun/step-3.5-flash:free",
        openrouter_max_retries=3,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: fake_settings)

    async def fail_openrouter(**_kwargs):
        raise OpenRouterError("429")

    monkeypatch.setattr(commentary_module, "create_chat_completion", fail_openrouter)

    async def run_call():
        return await commentary_module.generate_commentary(
            current_price=4.1000,
            predicted_price=4.1400,
            predicted_return=0.01,
            sentiment_index=0.2,
            sentiment_label="Bullish",
            top_influencers=[{"feature": "sentiment__index", "importance": 0.2}],
            news_count=12,
        )

    text = asyncio.run(run_call())

    assert text is not None
    assert "Risks:" in text
    assert "Opportunities:" in text
    assert "Bias warning:" in text
    assert text.strip().endswith("This is NOT financial advice.")


def test_generate_commentary_uses_template_when_api_key_missing(monkeypatch):
    fake_settings = SimpleNamespace(
        openrouter_api_key=None,
        resolved_commentary_model="stepfun/step-3.5-flash:free",
        openrouter_max_retries=3,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: fake_settings)

    async def run_call():
        return await commentary_module.generate_commentary(
            current_price=4.1000,
            predicted_price=4.0500,
            predicted_return=-0.012,
            sentiment_index=-0.1,
            sentiment_label="Bearish",
            top_influencers=[],
            news_count=3,
        )

    text = asyncio.run(run_call())

    assert text is not None
    assert "Risks:" in text
    assert "Opportunities:" in text
    assert text.strip().endswith("This is NOT financial advice.")


def test_determine_ai_stance_falls_back_to_keywords(monkeypatch):
    fake_settings = SimpleNamespace(
        openrouter_api_key="test-key",
        resolved_commentary_model="stepfun/step-3.5-flash:free",
        openrouter_max_retries=3,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: fake_settings)

    async def fail_openrouter(**_kwargs):
        raise OpenRouterError("rate limit")

    monkeypatch.setattr(commentary_module, "create_chat_completion", fail_openrouter)

    async def run_call():
        return await commentary_module.determine_ai_stance(
            "Bullish momentum, upside potential, strong growth setup with higher highs."
        )

    stance = asyncio.run(run_call())

    assert stance == "BULLISH"


def test_generate_and_save_commentary_saves_template_fallback(monkeypatch):
    fake_settings = SimpleNamespace(
        openrouter_api_key=None,
        resolved_commentary_model="stepfun/step-3.5-flash:free",
        openrouter_max_retries=3,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: fake_settings)
    async def fake_determine_ai_stance(_text):
        return "NEUTRAL"

    monkeypatch.setattr(commentary_module, "determine_ai_stance", fake_determine_ai_stance)

    captured = {}

    def fake_save_commentary_to_db(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(commentary_module, "save_commentary_to_db", fake_save_commentary_to_db)

    async def run_call():
        return await commentary_module.generate_and_save_commentary(
            session=object(),
            symbol="HG=F",
            current_price=4.1,
            predicted_price=4.12,
            predicted_return=0.005,
            sentiment_index=0.1,
            sentiment_label="Bullish",
            top_influencers=[{"feature": "driver_a", "importance": 0.3}],
            news_count=5,
        )

    result = asyncio.run(run_call())

    assert result is not None
    assert captured["symbol"] == "HG=F"
    assert "Risks:" in captured["commentary"]
