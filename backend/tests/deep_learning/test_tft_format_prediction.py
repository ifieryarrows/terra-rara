"""Tests for TFT prediction output formatting."""

import numpy as np

from deep_learning.models.tft_copper import format_prediction


def test_format_prediction_sorts_crossed_quantiles_and_flags_audit_fields():
    raw = np.array([
        [0.0002, -0.0215, -0.0140, 0.0459, 0.0050, 0.0057, 0.0171],
    ])

    result = format_prediction(raw, baseline_price=6.12)

    quantiles = list(result["quantiles"].values())
    raw_quantiles = list(result["raw_quantiles"].values())

    assert result["quantile_crossing_detected"] is True
    assert result["raw_quantile_crossing_rate"] > 0.0
    assert result["quantile_crossing_rate"] == 0.0
    assert result["ordered_quantile_crossing_rate"] == 0.0
    assert result["public_quantile_crossing_rate"] == 0.0
    assert result["anomaly_detected"] is True
    assert quantiles == sorted(quantiles)
    assert raw_quantiles != quantiles
    assert result["confidence_band_96"][0] <= result["predicted_price_median"]
    assert result["predicted_price_median"] <= result["confidence_band_96"][1]


def test_format_prediction_keeps_monotonic_quantiles_unflagged():
    raw = np.array([
        [-0.04, -0.02, -0.01, 0.00, 0.01, 0.02, 0.04],
    ])

    result = format_prediction(raw, baseline_price=6.12)

    assert result["quantile_crossing_detected"] is False
    assert result["quantile_crossing_rate"] == 0.0
    assert result["raw_quantile_crossing_rate"] == 0.0
    assert result["anomaly_detected"] is False


def test_format_prediction_uses_log_return_price_conversion():
    raw = np.tile(np.array([[0.005, 0.008, 0.009, 0.01, 0.011, 0.012, 0.015]]), (5, 1))
    result = format_prediction(raw, baseline_price=100.0)
    assert np.isclose(result["weekly_log_return"], 0.05)
    assert np.isclose(result["weekly_price"], 100.0 * np.exp(0.05))
    assert np.isclose(result["weekly_return"], np.exp(0.05) - 1.0)
    assert result["return_basis"] == "daily_log_return_path"


def test_format_prediction_calibrated_interval_widens_without_moving_weekly_median():
    raw = np.tile(np.array([[-0.020, -0.010, -0.005, 0.004, 0.005, 0.010, 0.020]]), (5, 1))
    base = format_prediction(raw, baseline_price=100.0, conformal_adjustment=0.0)
    widened = format_prediction(raw, baseline_price=100.0, conformal_adjustment=0.030)

    assert widened["weekly_log_return"] == base["weekly_log_return"]
    assert widened["weekly_log_return_q10_calibrated"] < base["weekly_log_return_q10_calibrated"]
    assert widened["weekly_log_return_q90_calibrated"] > base["weekly_log_return_q90_calibrated"]
    assert widened["weekly_interval_calibrated"] is True
    assert widened["public_quantile_crossing_rate"] == 0.0
    assert np.isclose(
        widened["daily_forecasts"][-1]["price_q10"],
        100.0 * np.exp(widened["weekly_log_return_q10_calibrated"]),
    )
    assert np.isclose(
        widened["daily_forecasts"][-1]["price_q90"],
        100.0 * np.exp(widened["weekly_log_return_q90_calibrated"]),
    )


def test_format_prediction_clips_negative_anomaly_before_monotonic_and_preserves_ordered():
    # Extreme negative median must clip before monotonic gaps so zero crossings occur
    raw_anomalous = np.array([[-1.0, -0.5, -0.2, -0.25, -0.2, -0.5, -1.0]])
    res_raw = format_prediction(raw_anomalous, baseline_price=100.0)
    vals_raw = list(res_raw["quantiles_log"].values())
    assert res_raw["anomaly_detected"] is True
    assert res_raw["quantile_crossing_rate"] == 0.0
    assert vals_raw == sorted(vals_raw)
    assert res_raw["quantiles_log"]["q0.50"] == -0.12

    # already_ordered=True preserves spreads exactly (no double softplus) and shifts vector cohesively
    ordered_in = np.tile(np.array([[-0.03, -0.02, -0.01, 0.005, 0.015, 0.025, 0.035]]), (5, 1))
    res_ord = format_prediction(ordered_in, baseline_price=100.0, already_ordered=True)
    assert np.isclose(res_ord["predicted_log_return_q10"], -0.02)
    assert np.isclose(res_ord["predicted_log_return_q90"], 0.025)

    ordered_anom = np.array([[-0.30, -0.28, -0.26, -0.25, -0.24, -0.22, -0.20]])
    res_shifted = format_prediction(ordered_anom, baseline_price=100.0, already_ordered=True)
    vals_shifted = list(res_shifted["quantiles_log"].values())
    assert vals_shifted == sorted(vals_shifted)
    assert np.isclose(res_shifted["quantiles_log"]["q0.50"], -0.12)
    assert np.isclose(res_shifted["quantiles_log"]["q0.90"] - res_shifted["quantiles_log"]["q0.10"], 0.06)
