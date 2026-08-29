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


def test_determine_ai_stance_uses_keywords(monkeypatch):
    fake_settings = SimpleNamespace(
        openrouter_api_key="test-key",
        resolved_commentary_model="stepfun/step-3.5-flash:free",
        openrouter_max_retries=3,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: fake_settings)

    async def run_call():
        return await commentary_module.determine_ai_stance(
            "Bullish momentum, upside potential, strong growth setup with higher highs."
        )

    stance = asyncio.run(run_call())
    assert stance == "BULLISH"


def test_generate_commentary_single_call_json_success(monkeypatch):
    fake_settings = SimpleNamespace(
        openrouter_api_key="test-key",
        resolved_commentary_model="stepfun/step-3.5-flash:free",
        openrouter_max_retries=3,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: fake_settings)

    async def fake_openrouter(**kwargs):
        assert kwargs.get("response_format") == commentary_module.COMMENTARY_RESPONSE_FORMAT
        assert kwargs.get("provider") == {"require_parameters": True}
        prompt = kwargs["messages"][1]["content"]
        assert "Model Baseline Price: 6.5870" in prompt
        assert "Current Live/Display Price (not the model baseline): 6.6590" in prompt
        assert "Predicted-vs-Live Display Gap" in prompt
        assert "never call a lower numeric target an advance" in prompt
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"stance":"BEARISH","commentary":"Risks:\\n1. Risk a\\n2. Risk b\\n3. Risk c\\n'
                            'Opportunities:\\n1. Opp a\\n2. Opp b\\n3. Opp c\\n'
                            'Summary: cautious.\\nBias warning: model risk.\\nThis is NOT financial advice."}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(commentary_module, "create_chat_completion", fake_openrouter)

    async def run_call():
        return await commentary_module._generate_commentary_and_stance(
            current_price=6.659,
            baseline_price=6.587,
            baseline_price_date="2026-08-27",
            price_basis="predicted_price = baseline_price * (1 + predicted_return)",
            predicted_price=6.6046,
            predicted_return=0.002678,
            sentiment_index=-0.1,
            sentiment_label="Bearish",
            top_influencers=[],
            news_count=8,
        )

    commentary, stance = asyncio.run(run_call())
    assert stance == "BEARISH"
    assert "This is NOT financial advice." in commentary


def test_commentary_fallback_does_not_describe_baseline_forecast_as_live_price_path(monkeypatch):
    fake_settings = SimpleNamespace(openrouter_api_key=None)
    monkeypatch.setattr(commentary_module, "get_settings", lambda: fake_settings)

    result = asyncio.run(commentary_module._generate_commentary_and_stance(
        current_price=6.659,
        baseline_price=6.587,
        baseline_price_date="2026-08-27",
        price_basis="predicted_price = baseline_price * (1 + predicted_return)",
        predicted_price=6.6046,
        predicted_return=0.002678,
        sentiment_index=0.1955,
        sentiment_label="Bullish",
        top_influencers=[],
        news_count=189,
    ))

    assert "from $6.6590 to $6.6046" not in result.commentary
    assert "baseline $6.5870" in result.commentary
    assert "live display price is $6.6590 and is a separate observation" in result.commentary


def test_generate_commentary_repairs_invalid_json(monkeypatch):
    fake_settings = SimpleNamespace(
        openrouter_api_key="test-key",
        resolved_commentary_model="stepfun/step-3.5-flash:free",
        openrouter_max_retries=3,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: fake_settings)

    call_count = {"n": 0}

    async def fake_openrouter(**_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"choices": [{"message": {"content": '{"stance":"BULLISH","commentary":"oops"'}}]}
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"stance":"BULLISH","commentary":"Risks:\\n1. r1\\n2. r2\\n3. r3\\n'
                            'Opportunities:\\n1. o1\\n2. o2\\n3. o3\\nSummary: s.\\n'
                            'Bias warning: b.\\nThis is NOT financial advice."}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(commentary_module, "create_chat_completion", fake_openrouter)

    async def run_call():
        return await commentary_module._generate_commentary_and_stance(
            current_price=4.1,
            predicted_price=4.2,
            predicted_return=0.02,
            sentiment_index=0.1,
            sentiment_label="Bullish",
            top_influencers=[],
            news_count=8,
        )

    commentary, stance = asyncio.run(run_call())
    assert call_count["n"] == 2
    assert stance == "BULLISH"
    assert "This is NOT financial advice." in commentary


def test_generate_and_save_commentary_uses_deterministic_stance_fallback(monkeypatch):
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

    captured = {}

    def fake_save_commentary_to_db(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(commentary_module, "save_commentary_to_db", fake_save_commentary_to_db)

    async def run_call():
        return await commentary_module.generate_and_save_commentary(
            session=object(),
            symbol="HG=F",
            current_price=4.1,
            predicted_price=4.2,
            predicted_return=0.02,
            sentiment_index=-0.005,
            sentiment_label="Bullish",
            top_influencers=[],
            news_count=4,
        )

    commentary = asyncio.run(run_call())
    assert commentary is not None
    assert "Risks:" in commentary
    assert captured["ai_stance"] == "BULLISH"


def test_generate_and_save_commentary_includes_degraded_model_note(monkeypatch):
    fake_settings = SimpleNamespace(
        openrouter_api_key=None,
        resolved_commentary_model="stepfun/step-3.5-flash:free",
        openrouter_max_retries=3,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(commentary_module, "save_commentary_to_db", lambda **_kwargs: None)

    async def run_call():
        return await commentary_module.generate_and_save_commentary(
            session=object(),
            symbol="HG=F",
            current_price=4.1,
            predicted_price=4.2,
            predicted_return=0.02,
            sentiment_index=0.1,
            sentiment_label="Bullish",
            top_influencers=[],
            news_count=4,
            model_status_note="TFT weekly model unavailable; commentary excludes TFT signal.",
        )

    commentary = asyncio.run(run_call())
    assert "commentary excludes TFT signal" in commentary


def test_commentary_fallback_is_observable_and_does_not_invent_zero_price(monkeypatch):
    fake_settings = SimpleNamespace(
        openrouter_api_key="test-key",
        resolved_commentary_model="removed-model",
        resolved_scoring_fast_model="different-repair-model",
        openrouter_max_retries=1,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: fake_settings)

    async def fail_openrouter(**_kwargs):
        raise OpenRouterError(
            "model removed",
            status_code=404,
            category="model_unavailable",
            attempts=[{"model": "removed-model", "category": "model_unavailable"}],
        )

    monkeypatch.setattr(commentary_module, "create_chat_completion", fail_openrouter)
    result = asyncio.run(commentary_module._generate_commentary_and_stance(
        current_price=6.6,
        predicted_price=None,
        predicted_return=0.002,
        sentiment_index=0.1,
        sentiment_label="Bullish",
        top_influencers=[],
        news_count=5,
    ))

    assert result.generation_mode == "deterministic_fallback"
    assert result.fallback_reason == "model_unavailable"
    assert result.attempts[0]["model"] == "removed-model"
    assert "$0.0000" not in result.commentary
    assert "unavailable" in result.commentary.lower()
