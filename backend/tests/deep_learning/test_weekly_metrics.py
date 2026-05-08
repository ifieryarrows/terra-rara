import numpy as np

from deep_learning.training.metrics import (
    compute_weekly_metrics,
    cumulative_horizon,
    cumulative_quantiles,
    interval_score,
)


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


def test_compute_weekly_metrics_uses_configured_horizon():
    actual = np.tile(np.array([[0.01, 0.02, 0.03, 0.50, 0.50]]), (4, 1))
    pred = np.zeros((4, 5, 7), dtype=float)
    pred[..., 3] = actual
    pred[..., 1] = actual - 0.01
    pred[..., 5] = actual + 0.01
    pred[..., 0] = actual - 0.02
    pred[..., 2] = actual - 0.005
    pred[..., 4] = actual + 0.005
    pred[..., 6] = actual + 0.02

    metrics = compute_weekly_metrics(actual, pred, horizon=3)

    assert metrics["weekly_sample_count"] == 4
    assert np.isclose(metrics["weekly_mean_actual_abs"], 0.06)
    assert np.isfinite(metrics["weekly_pi80_width_ratio"])
    assert np.isfinite(metrics["weekly_pi96_width_ratio"])
    assert np.isfinite(metrics["weekly_interval_score_80"])


def test_weekly_width_ratios_match_expected_formulas():
    actual = np.array(
        [
            [0.01, 0.01, 0.0, 0.0, 0.0],
            [-0.01, -0.01, 0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 0.0, 0.0],
            [-0.02, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    pred = np.zeros((4, 5, 7), dtype=float)
    pred[..., 3] = actual
    pred[..., 1] = actual - 0.002
    pred[..., 5] = actual + 0.002
    pred[..., 0] = actual - 0.004
    pred[..., 2] = actual - 0.001
    pred[..., 4] = actual + 0.001
    pred[..., 6] = actual + 0.004

    metrics = compute_weekly_metrics(actual, pred, horizon=5)
    weekly_actual = cumulative_horizon(actual, horizon=5)
    expected_pi80_width = 0.004 * 5
    expected_pi96_width = 0.008 * 5

    assert np.isclose(
        metrics["weekly_pi80_width_ratio"],
        expected_pi80_width / (2.56 * weekly_actual.std() + 1e-8),
    )
    assert np.isclose(
        metrics["weekly_pi96_width_ratio"],
        expected_pi96_width / (4.10 * weekly_actual.std() + 1e-8),
    )


def test_interval_score_penalizes_width_and_misses():
    actual = np.array([0.0, 0.0])
    tight = interval_score(actual, np.array([-0.1, -0.1]), np.array([0.1, 0.1]), alpha=0.20)
    wide = interval_score(actual, np.array([-0.5, -0.5]), np.array([0.5, 0.5]), alpha=0.20)
    missed = interval_score(actual, np.array([0.1, 0.1]), np.array([0.2, 0.2]), alpha=0.20)

    assert wide > tight
    assert missed > tight


def test_cumulative_quantiles_rejects_short_path():
    pred = np.zeros((2, 4, 7))
    try:
        cumulative_quantiles(pred, horizon=5)
    except ValueError as exc:
        assert "Need at least 5 horizons" in str(exc)
    else:
        raise AssertionError("expected ValueError")
