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
    assert result["anomaly_detected"] is False
