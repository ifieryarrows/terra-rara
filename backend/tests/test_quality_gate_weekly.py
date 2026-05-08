from app.quality_gate import evaluate_quality_gate


def test_good_t1_without_weekly_metrics_fails():
    passed, reasons = evaluate_quality_gate(da=0.75, sharpe=2.0, vr=1.0)
    assert passed is False
    assert "Missing weekly_directional_accuracy" in reasons


def test_missing_quantile_crossing_rate_fails():
    passed, reasons = evaluate_quality_gate(
        da=0.75,
        sharpe=2.0,
        vr=1.0,
        tail_capture=0.50,
        weekly_directional_accuracy=0.55,
        weekly_magnitude_ratio=1.0,
        weekly_tail_capture_rate=0.50,
        weekly_pi80_coverage=0.80,
        weekly_quantile_crossing_rate=0.0,
        weekly_median_sort_gap_max=0.0,
        weekly_sample_count=120,
    )

    assert passed is False
    assert "Missing quantile_crossing_rate" in reasons


def test_missing_weekly_quantile_crossing_rate_fails():
    passed, reasons = evaluate_quality_gate(
        da=0.75,
        sharpe=2.0,
        vr=1.0,
        tail_capture=0.50,
        quantile_crossing_rate=0.0,
        median_sort_gap_max=0.0,
        weekly_directional_accuracy=0.55,
        weekly_magnitude_ratio=1.0,
        weekly_tail_capture_rate=0.50,
        weekly_pi80_coverage=0.80,
        weekly_median_sort_gap_max=0.0,
        weekly_sample_count=120,
    )

    assert passed is False
    assert "Missing weekly_quantile_crossing_rate" in reasons


def test_weekly_magnitude_explosion_has_explicit_reason():
    passed, reasons = evaluate_quality_gate(
        da=0.75,
        sharpe=2.0,
        vr=1.0,
        tail_capture=0.50,
        quantile_crossing_rate=0.0,
        median_sort_gap_max=0.0,
        weekly_directional_accuracy=0.55,
        weekly_magnitude_ratio=12.5,
        weekly_tail_capture_rate=0.50,
        weekly_pi80_coverage=0.80,
        weekly_quantile_crossing_rate=0.0,
        weekly_median_sort_gap_max=0.0,
        weekly_sample_count=120,
    )

    assert passed is False
    assert any("WeeklyMagnitudeExplosion=12.5000 > 3.0" == reason for reason in reasons)
