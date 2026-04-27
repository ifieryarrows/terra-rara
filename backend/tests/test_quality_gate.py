"""Tests for shared TFT quality gate thresholds."""

from app.quality_gate import evaluate_quality_gate


def test_quality_gate_rejects_negative_sharpe_and_low_da():
    passed, reasons = evaluate_quality_gate(da=0.4377, sharpe=-2.4054, vr=0.9424)

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
    )

    assert passed is False
    assert any("QuantileCrossing" in reason for reason in reasons)
    assert any("MedianSortGapMax" in reason for reason in reasons)
