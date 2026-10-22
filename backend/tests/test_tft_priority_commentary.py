"""The pet speaks from the news; the 5-day TFT view is primary, XGBoost diagnostic.

Product decision (2026-10-22): news changes daily, so the companion's stance and
mood follow the news regime. Driving it from the price direction would leave no
neutral pet at all, because the 5-day model is almost always mildly directional.
Where the same information exists in both models, TFT-ASRO wins and the XGBoost
next-day view is labelled as a diagnostic.
"""

import asyncio
from types import SimpleNamespace

from app import commentary as commentary_module

XGB_DRIVER_LABEL = "US Dollar Index · Price vs. 20-day average"


def _tft_summary() -> dict:
    return {
        "primary_forecast_return": 0.01297,
        "primary_forecast_q10": -0.0275,
        "primary_forecast_q90": 0.0556,
        "direction": "BULLISH",
        "risk_level": "LOW",
        "reference_price_date": "2026-10-21",
    }


def _tft_explain() -> dict:
    return {
        "schema_version": 1,
        "symbol": "HG=F",
        "reference_price_date": "2026-10-21",
        "horizon": {"label": "5-day cumulative log return (primary)", "days": 5},
        "variable_selection": {
            "method": "vsn",
            "encoder": [
                {
                    "feature": "DX_Y_NYB_ret1",
                    "weight": 0.307,
                    "label": "US Dollar Index · 1-day price change",
                    "category": "Macro",
                    "recent": {
                        "basis": "5 sessions",
                        "kind": "pct",
                        "last_value": 99.1,
                        "change": 0.6,
                        "change_pct": 0.6,
                        "as_of": "2026-10-21",
                    },
                },
                {
                    "feature": "GXC_lag_ret1_1",
                    "weight": 0.091,
                    "label": "GXC · return lag 1 days",
                    "category": "ETF",
                },
            ],
            "decoder": [],
            "static": [],
        },
        "attention": {
            "method": "tft_attention",
            "top_days": [{"date": "2026-09-28", "weight": 0.029}],
            "window": {"start": "2026-08-20", "end": "2026-10-21", "days": 30},
        },
        "notes": "Model focus only; not causal.",
    }


def _fallback(
    tft: bool = True,
    sentiment_index: float = -0.0646,
    sentiment_label: str = "Neutral",
) -> str:
    return commentary_module._build_commentary_template_fallback(
        current_price=6.4815,
        baseline_price=6.5430,
        predicted_price=6.5497,
        predicted_return=0.0012,
        sentiment_index=sentiment_index,
        sentiment_label=sentiment_label,
        top_influencers=[
            {
                "feature": "DX_Y_NYB_price_sma_ratio",
                "label": XGB_DRIVER_LABEL,
                "importance": 0.081,
            }
        ],
        news_count=279,
        tft_explain=_tft_explain() if tft else None,
        tft_summary=_tft_summary() if tft else None,
    )


class TestStanceFollowsTheNews:
    def test_balanced_news_is_neutral_even_when_the_next_day_model_is_positive(self):
        # Production case: news -0.0646 (balanced), XGBoost +0.12% for tomorrow.
        assert (
            commentary_module._deterministic_stance_from_inputs(-0.0646, "Neutral")
            == "NEUTRAL"
        )

    def test_directional_news_sets_the_stance(self):
        assert (
            commentary_module._deterministic_stance_from_inputs(0.32, "Bullish")
            == "BULLISH"
        )
        assert (
            commentary_module._deterministic_stance_from_inputs(-0.32, "Bearish")
            == "BEARISH"
        )

    def test_label_is_optional_and_falls_back_to_the_shared_classification(self):
        assert commentary_module._deterministic_stance_from_inputs(-0.32) == "BEARISH"
        assert commentary_module._deterministic_stance_from_inputs(0.0) == "NEUTRAL"

    def test_legacy_positive_negative_labels_are_understood(self):
        assert (
            commentary_module._deterministic_stance_from_inputs(0.2, "Positive")
            == "BULLISH"
        )
        assert (
            commentary_module._deterministic_stance_from_inputs(-0.2, "Negative")
            == "BEARISH"
        )


class TestOpportunitiesLeadWithTft:
    def test_the_five_day_primary_is_the_first_opportunity(self):
        sections = commentary_module.parse_commentary_sections(_fallback(tft=True))

        assert len(sections["opportunities"]) == 3
        first = sections["opportunities"][0]
        assert first.startswith("5-day primary outlook (TFT-ASRO)")
        assert "+1.30%" in first
        assert "q10 -2.75%" in first
        assert "q90 +5.56%" in first

    def test_tft_encoder_focus_outranks_the_xgboost_drivers(self):
        text = _fallback(tft=True)

        assert "US Dollar Index · 1-day price change" in text
        assert XGB_DRIVER_LABEL not in text

    def test_the_xgboost_price_is_labelled_a_next_day_diagnostic(self):
        sections = commentary_module.parse_commentary_sections(_fallback(tft=True))
        joined = " ".join(sections["opportunities"])

        assert "Next-day diagnostic" in joined
        assert "$6.5497" in joined

    def test_without_a_tft_outlook_the_previous_ordering_is_kept(self):
        sections = commentary_module.parse_commentary_sections(_fallback(tft=False))
        joined = " ".join(sections["opportunities"])

        assert sections["opportunities"][0].startswith(
            "The model-derived price is $6.5497"
        )
        assert XGB_DRIVER_LABEL in joined
        assert "Next-day diagnostic" not in joined


class TestSummaryFollowsTheNews:
    def test_balanced_news_does_not_produce_a_directional_summary(self):
        sections = commentary_module.parse_commentary_sections(_fallback(tft=True))
        synthesis = sections["synthesis"].lower()

        assert "upside bias" not in synthesis
        assert "balanced" in synthesis

    def test_directional_news_states_the_lean(self):
        sections = commentary_module.parse_commentary_sections(
            _fallback(tft=True, sentiment_index=0.32, sentiment_label="Bullish")
        )

        assert "leans up" in sections["synthesis"].lower()


class TestPromptKeepsThePriority:
    def _prompt(self, monkeypatch) -> str:
        captured: dict = {}
        settings = SimpleNamespace(
            openrouter_api_key="test-key",
            resolved_commentary_model="test/model",
            openrouter_max_retries=1,
            openrouter_rpm=18,
            openrouter_fallback_models_list=[],
        )
        monkeypatch.setattr(commentary_module, "get_settings", lambda: settings)

        async def fake_openrouter(**kwargs):
            captured["prompt"] = kwargs["messages"][1]["content"]
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"stance":"NEUTRAL","commentary":"Summary: balanced."}'
                        }
                    }
                ]
            }

        monkeypatch.setattr(
            commentary_module, "create_chat_completion", fake_openrouter
        )

        asyncio.run(
            commentary_module._generate_commentary_and_stance(
                current_price=6.4815,
                predicted_price=6.5497,
                predicted_return=0.0012,
                sentiment_index=-0.0646,
                sentiment_label="Neutral",
                top_influencers=[],
                news_count=279,
                tft_explain=_tft_explain(),
                tft_summary=_tft_summary(),
            )
        )
        return captured["prompt"]

    def test_the_stance_rule_is_news_driven(self, monkeypatch):
        prompt = self._prompt(monkeypatch)

        assert "Derive the stance from the news sentiment regime" in prompt
        assert "never the stance" in prompt

    def test_xgboost_drivers_are_flagged_as_diagnostic(self, monkeypatch):
        prompt = self._prompt(monkeypatch)

        assert "Next-day XGBoost drivers (diagnostic" in prompt
        assert "TFT-ASRO focus takes priority" in prompt


class TestStanceIsDerivedNotNegotiated:
    """The LLM writes the prose; the news regime owns the stance.

    Prompt rules alone cannot guarantee the model's answer, and a stance that
    disagrees with the label rendered next to the same sentiment index would
    put the pet in a mood the dashboard contradicts.
    """

    def _run(
        self,
        monkeypatch,
        *,
        llm_stance: str,
        sentiment_index: float,
        sentiment_label: str,
    ):
        captured: dict = {}
        settings = SimpleNamespace(
            openrouter_api_key="test-key",
            resolved_commentary_model="test/model",
            openrouter_max_retries=1,
            openrouter_rpm=18,
            openrouter_fallback_models_list=[],
        )
        monkeypatch.setattr(commentary_module, "get_settings", lambda: settings)

        async def fake_openrouter(**kwargs):
            captured["prompt"] = kwargs["messages"][1]["content"]
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"stance":"' + llm_stance + '",'
                                '"commentary":"Summary: model prose."}'
                            )
                        }
                    }
                ]
            }

        monkeypatch.setattr(
            commentary_module, "create_chat_completion", fake_openrouter
        )

        result = asyncio.run(
            commentary_module._generate_commentary_and_stance(
                current_price=6.4815,
                predicted_price=6.5497,
                predicted_return=0.0012,
                sentiment_index=sentiment_index,
                sentiment_label=sentiment_label,
                top_influencers=[],
                news_count=279,
                tft_explain=_tft_explain(),
                tft_summary=_tft_summary(),
            )
        )
        return result, captured["prompt"]

    def test_the_llm_cannot_turn_balanced_news_bullish(self, monkeypatch):
        result, _ = self._run(
            monkeypatch,
            llm_stance="BULLISH",
            sentiment_index=-0.0646,
            sentiment_label="Neutral",
        )

        assert result.generation_mode == "llm"
        assert result.stance == "NEUTRAL"

    def test_the_llm_cannot_override_bearish_news(self, monkeypatch):
        result, _ = self._run(
            monkeypatch,
            llm_stance="BULLISH",
            sentiment_index=-0.40,
            sentiment_label="Bearish",
        )

        assert result.stance == "BEARISH"

    def test_the_prompt_states_the_stance_the_run_will_publish(self, monkeypatch):
        _, prompt = self._run(
            monkeypatch,
            llm_stance="NEUTRAL",
            sentiment_index=-0.0646,
            sentiment_label="Neutral",
        )

        assert "Stance for this run is fixed at NEUTRAL" in prompt
        assert "Do not contradict it" in prompt
