"""Tests for shared TFT quality gate thresholds."""

import pytest

from app.quality_gate import (
    evaluate_quality_gate,
    evaluate_quality_gate_metrics,
    evaluate_quality_gate_warnings,
)


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
    "weekly_pred_positive_rate": 0.55,
    "weekly_actual_positive_rate": 0.55,
    "weekly_sharpe_ratio": 0.40,
    "weekly_raw_magnitude_ratio": 1.0,
    "weekly_median_bound_applied_rate": 0.10,
}
GOOD_QUANTILE = {
    "quantile_crossing_rate": 0.0,
    "median_sort_gap_max": 0.0,
}


def test_quality_gate_rejects_negative_sharpe_but_not_low_da():
    # Weekly-primary contract: T+1 Sharpe triggers a diagnostic warning, while
    # negative weekly HAC Sharpe below -0.35 blocks promotion cleanly.
    t1_only_passed, t1_only_reasons = evaluate_quality_gate(
        da=0.4377, sharpe=-2.4054, vr=0.9424, **GOOD_QUANTILE, **GOOD_WEEKLY
    )
    assert t1_only_passed is True, t1_only_reasons
    t1_warnings = evaluate_quality_gate_warnings(vr=0.9424, sharpe_ratio=-2.4054)
    assert any("T1Sharpe=-2.4054" in w for w in t1_warnings)

    weekly_failed, weekly_reasons = evaluate_quality_gate(
        da=0.4377,
        sharpe=0.0,
        vr=0.9424,
        **GOOD_QUANTILE,
        **{**GOOD_WEEKLY, "weekly_sharpe_ratio_hac": -0.45},
    )
    assert weekly_failed is False
    assert not any("DA=" in reason for reason in weekly_reasons)
    assert any("WeeklySharpe=-0.4500" in reason for reason in weekly_reasons)


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
    # n < 150 -> min_weekly_da = 0.45
    passed_small, reasons_small = evaluate_quality_gate(
        da=0.52,
        sharpe=0.0,
        vr=1.0,
        **GOOD_QUANTILE,
        **{**GOOD_WEEKLY, "weekly_directional_accuracy": 0.46, "weekly_sample_count": 105},
    )
    assert passed_small is True, reasons_small

    # 150 <= n < 300 -> min_weekly_da = 0.47
    failed_mid, reasons_mid = evaluate_quality_gate(
        da=0.52,
        sharpe=0.0,
        vr=1.0,
        **GOOD_QUANTILE,
        **{**GOOD_WEEKLY, "weekly_directional_accuracy": 0.46, "weekly_sample_count": 180},
    )
    assert failed_mid is False
    assert "WeeklyDA=0.4600 < 0.47" in reasons_mid

    # n >= 300 -> min_weekly_da = 0.49
    failed_large, reasons_large = evaluate_quality_gate(
        da=0.52,
        sharpe=0.0,
        vr=1.0,
        **GOOD_QUANTILE,
        **{**GOOD_WEEKLY, "weekly_directional_accuracy": 0.48, "weekly_sample_count": 320},
    )
    assert failed_large is False
    assert "WeeklyDA=0.4800 < 0.49" in reasons_large


def test_quality_gate_magnitude_floor_mae_cap_and_low_magnitude_warning():
    # Bayes-optimal low-magnitude MR=0.03 passes gate with non-blocking warning
    passed_low_mr, reasons_low_mr = evaluate_quality_gate(
        da=0.52,
        sharpe=0.0,
        vr=1.0,
        **GOOD_QUANTILE,
        **{**GOOD_WEEKLY, "weekly_magnitude_ratio": 0.03, "weekly_mae_vs_naive_zero": 1.05},
    )
    assert passed_low_mr is True, reasons_low_mr
    warns = evaluate_quality_gate_warnings(vr=1.0, weekly_magnitude_ratio=0.03)
    assert any("WeeklyMagnitudeRatio=0.0300 < 0.05" in w for w in warns)

    # Collapse below 0.02 and MAE > 1.12 fail
    failed, reasons = evaluate_quality_gate(
        da=0.52,
        sharpe=0.0,
        vr=1.0,
        **GOOD_QUANTILE,
        **{**GOOD_WEEKLY, "weekly_magnitude_ratio": 0.01, "weekly_mae_vs_naive_zero": 1.13},
    )
    assert failed is False
    assert "WeeklyMagnitudeCollapse=0.0100 < 0.02" in reasons
    assert "WeeklyMAEvsNaiveZero=1.1300 > 1.12" in reasons


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
    assert "WeeklyPI96WidthRatio=10.6438 > 3.2" in reasons


def test_quality_gate_rejects_weekly_majority_sign_collapse():
    passed, reasons = evaluate_quality_gate(
        da=0.55,
        sharpe=0.5,
        vr=1.0,
        **GOOD_QUANTILE,
        **{
            **GOOD_WEEKLY,
            "weekly_pred_positive_rate": 1.0,
            "weekly_actual_positive_rate": 0.5484,
        },
    )

    assert passed is False
    assert any("WeeklySignCollapse=" in reason for reason in reasons)


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

def test_quality_gate_rejects_weekly_raw_magnitude_explosion():
    passed, reasons = evaluate_quality_gate(
        da=0.60,
        sharpe=0.1,
        vr=1.0,
        **GOOD_QUANTILE,
        **{**GOOD_WEEKLY, "weekly_raw_magnitude_ratio": 3.5},
    )
    assert passed is False
    assert any("WeeklyRawMagnitudeExplosion=" in r for r in reasons)


def test_quality_gate_warns_on_excessive_cap_clipping():
    warnings = evaluate_quality_gate_warnings(
        vr=1.0,
        weekly_median_bound_applied_rate=0.75,
        weekly_raw_magnitude_ratio=2.4,
    )
    assert any("WeeklyMedianBoundRate=" in w for w in warnings)
    assert any("WeeklyRawMagnitudeRatio=" in w for w in warnings)


def test_quality_gate_warns_on_moderate_cap_clipping():
    warnings = evaluate_quality_gate_warnings(
        vr=1.0,
        weekly_median_bound_applied_rate=0.45,
        weekly_raw_magnitude_ratio=1.9,
    )
    assert any("WeeklyMedianBoundRate=" in w for w in warnings)
    assert any("WeeklyRawMagnitudeRatio=" in w for w in warnings)

def test_quality_gate_rejects_negative_weekly_sharpe():
    passed, reasons = evaluate_quality_gate(
        da=0.60,
        sharpe=0.1,
        vr=1.0,
        **GOOD_QUANTILE,
        **{**GOOD_WEEKLY, "weekly_sharpe_ratio": -0.45},
    )
    assert passed is False
    assert any("WeeklySharpe=" in r for r in reasons)


def test_quality_gate_rejects_missing_weekly_sharpe():
    weekly_without_sharpe = {
        key: value
        for key, value in GOOD_WEEKLY.items()
        if key != "weekly_sharpe_ratio"
    }
    passed, reasons = evaluate_quality_gate(
        da=0.60,
        sharpe=0.1,
        vr=1.0,
        **GOOD_QUANTILE,
        **weekly_without_sharpe,
    )

    assert passed is False
    assert "Missing weekly_sharpe_ratio" in reasons


def test_quality_gate_rejects_missing_raw_magnitude_audit():
    weekly_without_raw_audit = {
        key: value
        for key, value in GOOD_WEEKLY.items()
        if key not in {
            "weekly_raw_magnitude_ratio",
            "weekly_median_bound_applied_rate",
        }
    }
    passed, reasons = evaluate_quality_gate(
        da=0.60,
        sharpe=0.1,
        vr=1.0,
        **GOOD_QUANTILE,
        **weekly_without_raw_audit,
    )

    assert passed is False
    assert "Missing weekly_raw_magnitude_ratio" in reasons
    assert "Missing weekly_median_bound_applied_rate" in reasons


def test_serialized_metric_mapping_uses_full_gate_contract():
    passed, reasons = evaluate_quality_gate_metrics(
        {
            "directional_accuracy": "0.60",
            "sharpe_ratio": "0.10",
            "variance_ratio": "1.0",
            **GOOD_QUANTILE,
            **GOOD_WEEKLY,
        }
    )

    assert passed is True, reasons


def test_quality_gate_warns_on_low_weekly_sharpe():
    warnings = evaluate_quality_gate_warnings(
        vr=1.0,
        weekly_sharpe_ratio=-0.12,
    )
    assert any("WeeklySharpe=-0.1200 < 0.00" in w for w in warnings)

