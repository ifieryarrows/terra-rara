"""Tests for shared TFT quality gate thresholds."""

from app.quality_gate import evaluate_quality_gate


GOOD_WEEKLY = {
    "weekly_directional_accuracy": 0.55,
    "weekly_magnitude_ratio": 1.0,
    "weekly_tail_capture_rate": 0.50,
    "weekly_pi80_coverage": 0.80,
    "weekly_quantile_crossing_rate": 0.0,
    "weekly_median_sort_gap_max": 0.0,
    "weekly_sample_count": 120,
}


def test_quality_gate_rejects_negative_sharpe_and_low_da():
    passed, reasons = evaluate_quality_gate(da=0.4377, sharpe=-2.4054, vr=0.9424, **GOOD_WEEKLY)

    assert passed is False
    assert any("DA=" in reason for reason in reasons)
    assert any("Sharpe=" in reason for reason in reasons)


def test_quality_gate_rejects_quantile_incoherence():
    passed, reasons = evaluate_quality_gate(
        da=0.52,
        sharpe=0.4,
        vr=0.9,
        quantile_crossing_rate=0.30,
        median_sort_gap_max=0.02,
        **GOOD_WEEKLY,
    )

    assert passed is False
    assert any("QuantileCrossing" in reason for reason in reasons)
    assert any("MedianSortGapMax" in reason for reason in reasons)


def test_quality_gate_fails_when_weekly_metrics_missing():
    passed, reasons = evaluate_quality_gate(da=0.60, sharpe=1.0, vr=1.0)
    assert passed is False
    assert any("Missing weekly_directional_accuracy" in reason for reason in reasons)


def test_quality_gate_relaxes_only_weekly_da_for_small_sample():
    passed, reasons = evaluate_quality_gate(
        da=0.52,
        sharpe=0.0,
        vr=1.0,
        **{**GOOD_WEEKLY, "weekly_directional_accuracy": 0.515, "weekly_sample_count": 50},
    )
    assert passed is True, reasons
