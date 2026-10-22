"""
Balanced sentiment must never be read as directional.

The dashboard renders the sentiment *label* (Bullish / Neutral / Bearish) next to
the raw index. A consumer that needs a directional read must use that
classification, never the sign of the raw number: an index inside the balance
band carries no directional information at all.

Production case (2026-10-22): index -0.0287 -> label "Neutral" (Balanced on the
dashboard), while the deterministic stance read the number and returned BEARISH,
putting the Cuprum pet in bear mood while the model read +0.10% upside.
"""

import asyncio
from types import SimpleNamespace

import pytest

from app import commentary as commentary_module
from app import inference as inference_module
from app.sentiment import (
    SENTIMENT_BALANCE_BAND,
    classify_sentiment,
    is_balanced,
    sentiment_direction,
)


def _fake_inference_settings():
    return SimpleNamespace(
        training_symbols_hash="sha256:test",
        inference_sentiment_multiplier_max=2.0,
        inference_sentiment_multiplier_min=0.5,
        inference_sentiment_news_ref=30,
        inference_sentiment_power_ref=0.20,
        inference_tiny_signal_threshold=0.0015,
        inference_tiny_signal_floor=0.0025,
        inference_return_cap=0.02,
    )


def _fake_commentary_settings():
    return SimpleNamespace(
        openrouter_api_key="test-key",
        resolved_commentary_model="stepfun/step-3.5-flash:free",
        openrouter_max_retries=3,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )


class TestClassification:
    def test_index_inside_the_band_is_neutral(self):
        assert classify_sentiment(-0.029) == "Neutral"
        assert classify_sentiment(0.0) == "Neutral"
        assert classify_sentiment(SENTIMENT_BALANCE_BAND) == "Neutral"
        assert classify_sentiment(-SENTIMENT_BALANCE_BAND) == "Neutral"

    def test_index_outside_the_band_is_directional(self):
        assert classify_sentiment(0.11) == "Bullish"
        assert classify_sentiment(-0.11) == "Bearish"

    def test_direction_is_zero_for_a_balanced_index(self):
        assert sentiment_direction(-0.029) == 0
        assert sentiment_direction(0.2) == 1
        assert sentiment_direction(-0.2) == -1

    def test_is_balanced(self):
        assert is_balanced(-0.029) is True
        assert is_balanced(-0.2) is False


class TestDeterministicStance:
    """The stance follows the news regime; the price model never sets it."""

    def test_balanced_sentiment_is_neutral_whatever_the_model_says(self):
        # Production case: news -0.0646 (Neutral), XGBoost +0.12% for tomorrow.
        stance = commentary_module._deterministic_stance_from_inputs(-0.0287, "Neutral")

        assert stance == "NEUTRAL"

    def test_label_is_optional_and_falls_back_to_the_classification(self):
        stance = commentary_module._deterministic_stance_from_inputs(-0.0287)

        assert stance == "NEUTRAL"

    def test_directional_sentiment_drives_the_stance(self):
        assert commentary_module._deterministic_stance_from_inputs(-0.30, "Bearish") == "BEARISH"
        assert commentary_module._deterministic_stance_from_inputs(0.30, "Bullish") == "BULLISH"


class TestFallbackCommentary:
    def _fallback(self, sentiment_index, sentiment_label):
        return commentary_module._build_commentary_template_fallback(
            current_price=6.4815,
            baseline_price=6.5430,
            predicted_price=6.5497,
            predicted_return=0.001,
            sentiment_index=sentiment_index,
            sentiment_label=sentiment_label,
            top_influencers=[],
            news_count=271,
        )

    def test_balanced_regime_is_described_as_balanced(self):
        text = self._fallback(-0.0287, "Neutral")

        risk_lines = [line for line in text.splitlines() if line.startswith("2.")]
        assert risk_lines, "the fallback must keep a second risk line"
        assert "balanced" in risk_lines[0].lower()
        assert "balance band" in risk_lines[0].lower()

    def test_directional_regime_keeps_the_regime_line(self):
        text = self._fallback(-0.30, "Bearish")

        assert "Sentiment regime is Bearish with score -0.300" in text


class TestPrompt:
    def test_prompt_makes_the_label_authoritative(self, monkeypatch):
        captured = {}

        async def fake_openrouter(**kwargs):
            captured["prompt"] = kwargs["messages"][1]["content"]
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"stance":"NEUTRAL","commentary":"Summary: flat.\\n'
                                'This is NOT financial advice."}'
                            )
                        }
                    }
                ]
            }

        monkeypatch.setattr(commentary_module, "get_settings", _fake_commentary_settings)
        monkeypatch.setattr(commentary_module, "create_chat_completion", fake_openrouter)

        asyncio.run(
            commentary_module._generate_commentary_and_stance(
                current_price=6.4815,
                baseline_price=6.5430,
                baseline_price_date="2026-10-21",
                price_basis="predicted_price = baseline_price * (1 + predicted_return)",
                predicted_price=6.5497,
                predicted_return=0.001,
                sentiment_index=-0.0287,
                sentiment_label="Neutral",
                top_influencers=[],
                news_count=271,
            )
        )

        prompt = captured["prompt"]
        assert "Sentiment Balance Band" in prompt
        assert "no directional information" in prompt.lower()
        assert "Sentiment Label: Neutral" in prompt


class TestSentimentAdjustment:
    def test_balanced_index_does_not_tilt_the_forecast(self, monkeypatch):
        monkeypatch.setattr(inference_module, "get_settings", _fake_inference_settings)

        adjusted, multiplier, applied, capped = inference_module._apply_sentiment_adjustment(
            raw_predicted_return=0.01,
            sentiment_index=-0.029,
            news_count_7d=271,
        )

        assert multiplier == pytest.approx(1.0)
        assert adjusted == pytest.approx(0.01)
        assert applied is False
        assert capped is False

    def test_directional_index_still_tilts_the_forecast(self, monkeypatch):
        monkeypatch.setattr(inference_module, "get_settings", _fake_inference_settings)

        adjusted, multiplier, applied, _capped = inference_module._apply_sentiment_adjustment(
            raw_predicted_return=0.01,
            sentiment_index=-0.30,
            news_count_7d=30,
        )

        assert multiplier < 1.0
        assert adjusted < 0.01
        assert applied is True
