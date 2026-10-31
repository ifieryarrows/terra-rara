"""Regressions: IFM is really primary, CUPRUM DIALOGUE parses, headline is LLM-written."""

import asyncio
from types import SimpleNamespace

import app.commentary as commentary_module
from app.commentary import parse_commentary_sections
from app.companion import build_companion_state

# Verbatim shape of a production commentary row (2026-10-31).
PRODUCTION_COMMENTARY = (
    "RISKS: \n• Risk a.\n• Risk b.\n• Risk c.\n\n"
    "OPPORTUNITIES: \n• Opp a.\n• Opp b.\n• Opp c.\n\n"
    "SYNTHESIS: Long institutional synthesis. Second sentence.\n\n"
    "BIAS ACKNOWLEDGMENT: Model focus is not causal.\n\n"
    "DISCLAIMER:\nThis is NOT financial advice.\n\n"
    "CUPRUM DIALOGUE:\n"
    '{"greeting": "The wires hum with caution.", "drivers_reply": "The dollar is pulling hardest.", '
    '"outlook_reply": "It may hold.", "mood": "patina_bear", "headline": "Copper leans lower this week as the dollar firms."}'
)


def test_cuprum_dialogue_block_is_parsed_from_stored_commentary():
    sections = parse_commentary_sections(PRODUCTION_COMMENTARY)
    assert sections["cuprum_daily"]["greeting"] == "The wires hum with caution."
    assert sections["cuprum_daily"]["drivers_reply"] == "The dollar is pulling hardest."
    # The dialogue block must not leak into the bias/disclaimer text.
    assert "greeting" not in (sections["bias_warning"] or "")


def test_companion_headline_comes_from_llm_headline():
    state = build_companion_state(
        symbol="HG=F",
        commentary_record={"commentary": PRODUCTION_COMMENTARY, "ai_stance": "BEARISH"},
    )
    assert state["headline_quote"] == "Copper leans lower this week as the dollar firms."
    assert state["cuprum_daily"]["outlook_reply"] == "It may hold."


def test_ifm_is_primary_provider_and_receives_the_full_prompt(monkeypatch):
    settings = SimpleNamespace(
        ifm_api_key="ifm-key",
        ifm_model="IFM/K2-Horizon-375B-A23B",
        ifm_base_url="https://api.ifm.ai/v1",
        ifm_timeout_seconds=30.0,
        ifm_reasoning_effort="low",
        openrouter_api_key="or-key",
        resolved_commentary_model="nvidia/nemotron-3.5-super-vl:free",
        openrouter_max_retries=1,
        openrouter_rpm=12,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: settings)
    captured = {}

    async def fake_ifm(messages, **kwargs):
        captured["prompt"] = messages[1]["content"]
        return (
            '{"stance":"BEARISH","headline":"Copper leans lower.",'
            '"commentary":"RISKS:\\n• a\\nOPPORTUNITIES:\\n• b\\nSYNTHESIS: c\\n'
            'BIAS ACKNOWLEDGMENT: d\\nDISCLAIMER:\\nThis is NOT financial advice.",'
            '"cuprum_daily":{"mood":"patina_bear","greeting":"g","drivers_reply":"r","outlook_reply":"o"}}',
            {},
        )

    async def openrouter_must_not_run(**kwargs):
        raise AssertionError("OpenRouter called although IFM succeeded")

    import app.ifm_client as ifm_client

    monkeypatch.setattr(ifm_client, "create_ifm_chat_completion", fake_ifm)
    monkeypatch.setattr(commentary_module, "create_chat_completion", openrouter_must_not_run)

    result = asyncio.run(
        commentary_module._generate_commentary_and_stance(
            current_price=6.48,
            predicted_price=6.44,
            predicted_return=-0.0025,
            sentiment_index=-0.39,
            sentiment_label="Bearish",
            top_influencers=[{"feature": "SCCO_EMA_20", "importance": 0.005}],
            news_count=251,
        )
    )

    assert result.generation_mode == "ifm_k2"
    assert result.model_name == "IFM/K2-Horizon-375B-A23B"
    assert result.cuprum_daily["headline"] == "Copper leans lower."
    prompt = captured["prompt"]
    assert "WRITING RULES" in prompt
    # raw identifiers never reach the model as data (the rule text cites one as an example)
    assert "- SCCO_EMA_20:" not in prompt
    assert "CUPRUM DIALOGUE" in result.commentary
