"""Tests for XGBoost + TFT ensemble directional voting."""

from deep_learning.inference.predictor import ensemble_directional_vote


def test_both_agree_bullish():
    result = ensemble_directional_vote(xgb_return=0.01, tft_return=0.02)
    assert result["consensus_direction"] == "BULLISH"
    assert result["confidence"] == "HIGH"
    assert result["position_scale"] == 1.0


def test_both_agree_bearish():
    result = ensemble_directional_vote(xgb_return=-0.01, tft_return=-0.02)
    assert result["consensus_direction"] == "BEARISH"
    assert result["confidence"] == "HIGH"


def test_disagreement_yields_neutral():
    result = ensemble_directional_vote(xgb_return=0.01, tft_return=-0.02)
    assert result["consensus_direction"] == "NEUTRAL"
    assert result["confidence"] == "LOW"
    assert result["position_scale"] == 0.0


def test_xgb_neutral_defers_to_tft():
    result = ensemble_directional_vote(xgb_return=0.001, tft_return=0.02)
    assert result["consensus_direction"] == "BULLISH"
    assert result["confidence"] == "MEDIUM"


def test_tft_neutral_defers_to_xgb():
    result = ensemble_directional_vote(xgb_return=0.01, tft_return=0.0005)
    assert result["consensus_direction"] == "BULLISH"
    assert result["confidence"] == "MEDIUM"
    assert result["position_scale"] == 0.5


def test_bias_correction_applied():
    result = ensemble_directional_vote(
        xgb_return=0.005, tft_return=-0.01, xgb_bias_correction=0.008,
    )
    assert result["xgb_return_adjusted"] < 0
    assert result["xgb_direction"] == -1


def test_blended_return_calculation():
    result = ensemble_directional_vote(xgb_return=0.01, tft_return=0.02)
    expected = 0.4 * 0.01 + 0.6 * 0.02
    assert abs(result["blended_return"] - expected) < 1e-9
