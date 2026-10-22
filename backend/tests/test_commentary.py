"""
Tests for commentary generation and fallback behavior.
"""

import asyncio
from types import SimpleNamespace

from app import commentary as commentary_module
from app.openrouter_client import OpenRouterError
from app.tft_explain import summarize_explain_for_commentary


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


# ---------------------------------------------------------------------------
# TFT-ASRO XAI commentary integration (Phase 2 core)
# ---------------------------------------------------------------------------


def _xai_explain_payload() -> dict:
    """Explain payload shaped exactly like ``app.tft_explain`` output."""
    return {
        "schema_version": 1,
        "symbol": "HG=F",
        "reference_price_date": "2026-10-14",
        "horizon": {"label": "5-day cumulative log return (primary)", "days": 5},
        "variable_selection": {
            "method": "vsn",
            "encoder": [
                {
                    "feature": "HG=F_close",
                    "weight": 0.423,
                    "label": "Copper Close",
                    "category": "Price",
                    "recent": {
                        "basis": "5 sessions",
                        "kind": "pct",
                        "last_value": 6.5,
                        "change": 0.13,
                        "change_pct": 2.1,
                        "as_of": "2026-10-14",
                    },
                },
                {
                    "feature": "HG=F_sentiment_mean",
                    "weight": 0.2,
                    "label": "Sentiment Mean",
                    "category": "Sentiment",
                    "recent": {
                        "basis": "5 sessions",
                        "kind": "abs",
                        "last_value": 0.15,
                        "change": 0.12,
                        "change_pct": None,
                        "as_of": "2026-10-14",
                    },
                },
                {
                    "feature": "HG=F_emb_03",
                    "weight": 0.1,
                    "label": "Embedding 03",
                    "category": "Embedding",
                },
                {"feature": "HG=F_volume", "weight": 0.05, "label": "Volume", "category": "Liquidity"},
                {"feature": "HG=F_sma_5", "weight": 0.03, "label": "SMA 5", "category": "Trend"},
                {"feature": "HG=F_inventory", "weight": 0.01, "label": "Inventory", "category": "Fundamental"},
            ],
            "decoder": [],
            "static": [],
        },
        "attention": {
            "method": "tft_attention",
            "top_days": [
                {"date": "2026-10-14", "weight": 0.5},
                {"date": "2026-10-01", "weight": 0.3},
                {"date": "2026-09-20", "weight": 0.12},
                {"date": "2026-09-01", "weight": 0.08},
            ],
            "window": {"start": "2026-09-01", "end": "2026-10-14", "days": 30},
        },
        "notes": "Model focus only; not causal.",
    }


def _xai_tft_summary() -> dict:
    return {
        "primary_forecast_return": 0.0212,
        "primary_forecast_q10": -0.01,
        "primary_forecast_q90": 0.05,
        "direction": "BULLISH",
        "risk_level": "MEDIUM",
        "reference_price_date": "2026-10-14",
    }


def _capturing_openrouter(captured: dict):
    async def fake_openrouter(**_kwargs):
        captured["prompt"] = _kwargs["messages"][1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"stance":"BULLISH","commentary":"RISKS (3 factors):\\n'
                            '• r1\\n• r2\\n• r3\\n'
                            'OPPORTUNITIES (3 factors):\\n• o1\\n• o2\\n• o3\\n'
                            'SYNTHESIS:\\nWeaving the model outlook.\\n'
                            'BIAS ACKNOWLEDGMENT:\\nAssumptions may fail.\\n'
                            'DISCLAIMER:\\nThis is NOT financial advice."}'
                        )
                    }
                }
            ]
        }

    return fake_openrouter


def _patch_xai_settings(monkeypatch):
    fake_settings = SimpleNamespace(
        openrouter_api_key="test-key",
        resolved_commentary_model="stepfun/step-3.5-flash:free",
        openrouter_max_retries=3,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: fake_settings)


def test_summarize_explain_for_commentary_builds_prompt_safe_digest():
    digest = summarize_explain_for_commentary(_xai_explain_payload())

    assert digest["horizon_label"] == "5-day cumulative log return (primary)"
    assert digest["reference_price_date"] == "2026-10-14"
    assert [signal["feature"] for signal in digest["top_encoder_signals"]] == [
        "HG=F_close",
        "HG=F_sentiment_mean",
        "HG=F_emb_03",
        "HG=F_volume",
        "HG=F_sma_5",
    ]
    assert digest["top_encoder_signals"][0]["weight_pct"] == 42.3
    assert digest["top_encoder_signals"][0]["recent_summary"] == "+2.10% over 5 sessions"
    assert digest["top_encoder_signals"][1]["recent_summary"] == "+0.120 over 5 sessions"
    assert digest["top_encoder_signals"][2]["recent_summary"] is None  # no recent block
    assert digest["memory_anchors"] == [
        {"date": "2026-10-14", "weight_pct": 50.0},
        {"date": "2026-10-01", "weight_pct": 30.0},
        {"date": "2026-09-20", "weight_pct": 12.0},
    ]
    assert digest["window_span"] == {"start": "2026-09-01", "end": "2026-10-14", "days": 30}


def test_summarize_explain_for_commentary_is_fail_open_for_missing_and_malformed_input():
    assert summarize_explain_for_commentary(None) is None
    assert summarize_explain_for_commentary("not-a-payload") is None
    assert summarize_explain_for_commentary(123) is None

    sparse = summarize_explain_for_commentary(
        {"variable_selection": {"encoder": [{"feature": "HG=F_close", "weight": 0.5}]}}
    )
    assert sparse["top_encoder_signals"][0]["recent_summary"] is None
    assert sparse["top_encoder_signals"][0]["weight_pct"] == 50.0

    malformed = summarize_explain_for_commentary(
        {"variable_selection": "nope", "attention": {"top_days": None, "window": 7}}
    )
    assert malformed["top_encoder_signals"] == []
    assert malformed["memory_anchors"] == []
    assert malformed["window_span"] is None
    assert malformed["horizon_label"] == "5-day cumulative log return (primary)"


def test_tft_xai_block_is_injected_into_openrouter_prompt(monkeypatch):
    _patch_xai_settings(monkeypatch)
    captured = {}
    monkeypatch.setattr(commentary_module, "create_chat_completion", _capturing_openrouter(captured))

    async def run_call():
        return await commentary_module._generate_commentary_and_stance(
            current_price=6.5,
            predicted_price=6.6,
            predicted_return=0.002,
            sentiment_index=0.1,
            sentiment_label="Bullish",
            top_influencers=[{"feature": "HG=F_close", "importance": 0.4}],
            news_count=8,
            tft_explain=_xai_explain_payload(),
            tft_summary=_xai_tft_summary(),
        )

    result = asyncio.run(run_call())
    assert result.generation_mode == "llm"

    prompt = captured["prompt"]
    assert "TFT-ASRO 5-DAY PRIMARY HORIZON & XAI FOCUS (PRE-CALIBRATION, NON-CAUSAL):" in prompt
    assert "T+1 next-day simple-return" in prompt
    assert "5-day cumulative log-return" in prompt
    assert "Never merge, conflate, or re-label the two horizons" in prompt
    assert "+0.0212" in prompt
    assert "q10 -0.0100" in prompt
    assert "q90 +0.0500" in prompt
    assert "direction BULLISH" in prompt
    assert "risk level MEDIUM" in prompt
    assert "HG=F_close" in prompt
    assert "42.3% of encoder focus" in prompt
    assert "recent: +2.10% over 5 sessions" in prompt
    assert "2026-10-14 — 50.0%" in prompt
    assert "2026-10-01" in prompt
    assert "Weave the TFT-ASRO 5-day primary outlook" in prompt
    assert "never causal proof" in prompt


def test_tft_summary_alone_still_renders_horizon_block_in_prompt(monkeypatch):
    _patch_xai_settings(monkeypatch)
    captured = {}
    monkeypatch.setattr(commentary_module, "create_chat_completion", _capturing_openrouter(captured))

    async def run_call():
        return await commentary_module._generate_commentary_and_stance(
            current_price=6.5,
            predicted_price=6.6,
            predicted_return=0.002,
            sentiment_index=0.1,
            sentiment_label="Bullish",
            top_influencers=[],
            news_count=8,
            tft_summary=_xai_tft_summary(),
        )

    asyncio.run(run_call())

    prompt = captured["prompt"]
    assert "TFT-ASRO 5-DAY PRIMARY HORIZON & XAI FOCUS (PRE-CALIBRATION, NON-CAUSAL):" in prompt
    assert "+0.0212" in prompt
    assert "direction BULLISH" in prompt


def test_generate_commentary_prompt_without_tft_context_stays_unchanged(monkeypatch):
    _patch_xai_settings(monkeypatch)
    captured = {}
    monkeypatch.setattr(commentary_module, "create_chat_completion", _capturing_openrouter(captured))

    async def run_call():
        return await commentary_module._generate_commentary_and_stance(
            current_price=6.5,
            predicted_price=6.6,
            predicted_return=0.002,
            sentiment_index=0.1,
            sentiment_label="Bullish",
            top_influencers=[],
            news_count=8,
        )

    result = asyncio.run(run_call())
    assert result.generation_mode == "llm"

    prompt = captured["prompt"]
    assert "TFT-ASRO 5-DAY PRIMARY HORIZON & XAI FOCUS" not in prompt
    assert "Weave the TFT-ASRO 5-day primary outlook" not in prompt


_LLM_COMMENTARY = (
    "RISKS (3 factors):\n"
    "• Demand destruction if China property stimulus disappoints.\n"
    "• Positioning is crowded long after the recent squeeze.\n"
    "• A stronger dollar would cap the rally.\n"
    "\n"
    "OPPORTUNITIES (3 factors):\n"
    "• Grid investment keeps copper demand resilient.\n"
    "• LME inventories remain historically tight.\n"
    "\n"
    "SYNTHESIS:\n"
    "The model's constructive next-day signal aligns with firm physical tightness. "
    "Momentum remains positive but crowding argues for tight risk control.\n"
    "\n"
    "BIAS ACKNOWLEDGMENT:\n"
    "Volatility inputs can shift quickly around macro data releases.\n"
    "\n"
    "DISCLAIMER:\n"
    "This is NOT financial advice."
)


def test_parse_commentary_sections_extracts_llm_format():
    sections = commentary_module.parse_commentary_sections(
        _LLM_COMMENTARY, tft_explain=_xai_explain_payload()
    )

    assert sections["synthesis"].startswith("The model's constructive next-day signal")
    assert sections["synthesis"].endswith("risk control.")
    assert sections["risks"] == [
        "Demand destruction if China property stimulus disappoints.",
        "Positioning is crowded long after the recent squeeze.",
        "A stronger dollar would cap the rally.",
    ]
    assert sections["opportunities"] == [
        "Grid investment keeps copper demand resilient.",
        "LME inventories remain historically tight.",
    ]
    assert sections["bias_warning"] == "Volatility inputs can shift quickly around macro data releases."

    brief = sections["xai_brief"]
    assert "Model focus" in brief
    assert "HG=F_close" in brief
    assert "42.3%" in brief
    assert "2026-10-14 (50.0%)" in brief
    assert "not causal drivers" in brief


def test_parse_commentary_sections_extracts_deterministic_fallback():
    fallback = commentary_module._build_commentary_template_fallback(
        current_price=6.659,
        baseline_price=6.587,
        predicted_price=6.6046,
        predicted_return=0.002678,
        sentiment_index=0.1955,
        sentiment_label="Bullish",
        top_influencers=[{"feature": "HG=F_close", "importance": 0.4}],
        news_count=189,
        tft_explain=_xai_explain_payload(),
    )

    assert fallback.strip().endswith("This is NOT financial advice.")
    xai_line = next(line for line in fallback.splitlines() if line.startswith("XAI Focus:"))
    assert "`HG=F_close`" in xai_line
    assert "42.3%" in xai_line
    assert "2026-10-14 (50.0%)" in xai_line
    assert "not causal drivers" in xai_line
    assert fallback.index("XAI Focus:") < fallback.index("This is NOT financial advice.")

    sections = commentary_module.parse_commentary_sections(fallback)

    assert sections["synthesis"] == (
        "News sentiment is bullish (+0.196), so the daily read leans up; "
        "no 5-day model outlook is available for this run."
    )
    assert len(sections["risks"]) == 3
    assert sections["risks"][0].startswith("Model indicates upside uncertainty")
    assert len(sections["opportunities"]) == 2
    # TFT-ASRO focus outranks the XGBoost importance list, so this payload
    # yields the price line plus a single TFT driver line (the parser caps at 3).
    assert sections["opportunities"][0].startswith("The model-derived price is $6.6046")
    assert sections["bias_warning"].startswith("This view is model-driven")
    # The XAI line must never leak into the bias section.
    assert "XAI Focus" not in sections["bias_warning"]
    assert sections["xai_brief"] is None  # no explain payload was passed to the parser


def test_parse_commentary_sections_handles_none_and_unstructured_text():
    empty = commentary_module.parse_commentary_sections(None, tft_explain=_xai_explain_payload())
    assert empty["synthesis"] == ""
    assert empty["risks"] == []
    assert empty["opportunities"] == []
    assert empty["bias_warning"] is None
    assert "HG=F_close" in empty["xai_brief"]

    bare = commentary_module.parse_commentary_sections(
        "A single unstructured paragraph without sections."
    )
    assert bare["synthesis"] == "A single unstructured paragraph without sections."
    assert bare["xai_brief"] is None


def test_worker_stage6_passes_tft_xai_context_into_commentary_call():
    """Source-level canary mirroring tests/test_worker_tft_explain_wiring.py: a
    refactor that drops these kwargs silently disables XAI-aware commentary.
    """
    import inspect

    from worker.tasks import _execute_pipeline_stages_v2

    source = inspect.getsource(_execute_pipeline_stages_v2)
    assert "generate_and_save_commentary(" in source
    assert "tft_explain=tft_explain_payload" in source
    assert "tft_summary=tft_commentary_summary" in source
    assert "tft_commentary_summary = {" in source
