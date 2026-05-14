"""Tests for shared TFT quality gate thresholds."""

import pytest

from app.quality_gate import evaluate_quality_gate, evaluate_quality_gate_warnings


GOOD_WEEKLY = {
    "weekly_directional_accuracy": 0.55,
    "weekly_magnitude_ratio": 1.0,
    "weekly_tail_capture_rate": 0.50,
    "weekly_pi80_coverage": 0.80,
    "weekly_pi80_width_ratio": 1.0,
    "weekly_pi96_coverage": 0.96,
    "weekly_pi96_width_ratio": 1.0,
    "weekly_quantile_crossing_rate": 0.0,
    "weekly_sorted_quantile_crossing_rate": 0.0,
    "weekly_median_sort_gap_max": 0.0,
    "weekly_sample_count": 120,
}
GOOD_QUANTILE = {
    "quantile_crossing_rate": 0.0,
    "median_sort_gap_max": 0.0,
}


def test_quality_gate_rejects_negative_sharpe_but_not_low_da():
    passed, reasons = evaluate_quality_gate(da=0.4377, sharpe=-2.4054, vr=0.9424, **GOOD_QUANTILE, **GOOD_WEEKLY)

    assert passed is False
    assert not any("DA=" in reason for reason in reasons)
    assert any("Sharpe=" in reason for reason in reasons)


def test_quality_gate_asserts_public_quantile_incoherence():
    with pytest.raises(AssertionError, match="PublicQuantileCrossing"):
        evaluate_quality_gate(
            da=0.52,
            sharpe=0.4,
            vr=0.9,
            quantile_crossing_rate=0.30,
            median_sort_gap_max=0.0,
            **GOOD_WEEKLY,
        )

    with pytest.raises(AssertionError, match="OrderedMedianSortGapMax"):
        evaluate_quality_gate(
            da=0.52,
            sharpe=0.4,
            vr=0.9,
            quantile_crossing_rate=0.0,
            median_sort_gap_max=0.02,
            **GOOD_WEEKLY,
        )


def test_quality_gate_fails_when_weekly_metrics_missing():
    passed, reasons = evaluate_quality_gate(da=0.60, sharpe=1.0, vr=1.0)
    assert passed is False
    assert any("Missing weekly_directional_accuracy" in reason for reason in reasons)


def test_quality_gate_relaxes_only_weekly_da_for_small_sample():
    passed, reasons = evaluate_quality_gate(
        da=0.52,
        sharpe=0.0,
        vr=1.0,
        **GOOD_QUANTILE,
        **{**GOOD_WEEKLY, "weekly_directional_accuracy": 0.515, "weekly_sample_count": 50},
    )
    assert passed is True, reasons


def test_quality_gate_does_not_fail_on_low_t1_da_alone():
    passed, reasons = evaluate_quality_gate(
        da=0.42,
        sharpe=0.1,
        vr=1.0,
        **GOOD_QUANTILE,
        **GOOD_WEEKLY,
    )
    assert passed is True, reasons


def test_quality_gate_warns_on_t1_variance_ratio_above_stabilization_band():
    passed, reasons = evaluate_quality_gate(
        da=0.60,
        sharpe=0.1,
        vr=3.1,
        **GOOD_QUANTILE,
        **GOOD_WEEKLY,
    )
    warnings = evaluate_quality_gate_warnings(vr=3.1)
    assert passed is True, reasons
    assert "VR=3.1000 > 2.5 - model overdispersed" in warnings


def test_quality_gate_rejects_weekly_pi96_width_explosion():
    passed, reasons = evaluate_quality_gate(
        da=0.60,
        sharpe=0.1,
        vr=1.0,
        **GOOD_QUANTILE,
        **{**GOOD_WEEKLY, "weekly_pi96_width_ratio": 10.6438},
    )

    assert passed is False
    assert "WeeklyPI96WidthRatio=10.6438 > 3.0" in reasons


def test_quality_gate_asserts_weekly_public_crossing_above_bug_threshold():
    with pytest.raises(AssertionError, match="WeeklyPublicQuantileCrossing"):
        evaluate_quality_gate(
            da=0.60,
            sharpe=0.1,
            vr=1.0,
            **GOOD_QUANTILE,
            **{**GOOD_WEEKLY, "weekly_quantile_crossing_rate": 0.06},
        )


def test_quality_gate_rejects_negative_public_widths():
    passed, reasons = evaluate_quality_gate(
        da=0.60,
        sharpe=0.1,
        vr=1.0,
        pi80_width=-0.001,
        weekly_pi96_width=-0.001,
        **GOOD_QUANTILE,
        **GOOD_WEEKLY,
    )

    assert passed is False
    assert "PI80Width=-0.0010 < 0.0" in reasons
    assert "WeeklyPI96Width=-0.0010 < 0.0" in reasons
