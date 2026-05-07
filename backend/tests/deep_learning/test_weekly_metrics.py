import numpy as np

from deep_learning.training.metrics import compute_weekly_metrics, cumulative_horizon, cumulative_quantiles


def test_cumulative_horizon_sums_first_five_steps():
    y = np.array([[0.01, 0.02, -0.01, 0.00, 0.03, 0.99]])
    assert np.isclose(cumulative_horizon(y, horizon=5)[0], 0.05)


def test_compute_weekly_metrics_returns_sample_count():
    actual = np.tile(np.array([[0.01, 0.01, 0.0, -0.005, 0.005]]), (40, 1))
    pred = np.zeros((40, 5, 7), dtype=float)
    pred[..., 3] = actual
    pred[..., 1] = actual - 0.01
    pred[..., 5] = actual + 0.01
    pred[..., 0] = actual - 0.02
    pred[..., 2] = actual - 0.005
    pred[..., 4] = actual + 0.005
    pred[..., 6] = actual + 0.02

    metrics = compute_weekly_metrics(actual, pred)
    assert metrics["weekly_sample_count"] == 40
    assert metrics["weekly_directional_accuracy"] == 1.0
    assert "weekly_magnitude_ratio" in metrics


def test_cumulative_quantiles_rejects_short_path():
    pred = np.zeros((2, 4, 7))
    try:
        cumulative_quantiles(pred, horizon=5)
    except ValueError as exc:
        assert "Need at least 5 horizons" in str(exc)
    else:
        raise AssertionError("expected ValueError")
