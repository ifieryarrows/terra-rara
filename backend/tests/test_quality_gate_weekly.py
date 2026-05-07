from app.quality_gate import evaluate_quality_gate


def test_good_t1_without_weekly_metrics_fails():
    passed, reasons = evaluate_quality_gate(da=0.75, sharpe=2.0, vr=1.0)
    assert passed is False
    assert "Missing weekly_directional_accuracy" in reasons
