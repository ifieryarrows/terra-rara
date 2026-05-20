import numpy as np

from deep_learning.training.metrics import (
    compute_all_metrics,
    compute_weekly_metrics,
    cumulative_horizon,
    cumulative_quantiles,
    evaluate_quantile_predictions,
    interval_score,
    monotonic_quantiles_np,
    quantile_crossing_rate,
    quantile_median_sort_gap,
    select_prediction_horizon,
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
    assert "weekly_directional_accuracy_flipped" in metrics
    assert "weekly_sharpe_ratio_flipped" in metrics
    assert "weekly_tail_capture_rate_flipped" in metrics
    assert "weekly_sign_correlation" in metrics
    assert "weekly_pred_positive_rate" in metrics
    assert "weekly_actual_positive_rate" in metrics
    assert "weekly_pred_mean" in metrics
    assert "weekly_actual_mean" in metrics
    assert "weekly_pred_median" in metrics
    assert "weekly_actual_median" in metrics


def test_evaluate_quantile_predictions_matches_existing_metric_paths():
    actual = np.array(
        [
            [0.010, 0.005, -0.002, 0.003, 0.004],
            [-0.012, -0.004, 0.001, -0.003, -0.002],
            [0.008, 0.003, 0.001, 0.002, 0.004],
            [-0.006, -0.003, -0.002, -0.001, -0.002],
        ],
        dtype=float,
    )
    pred = np.zeros((4, 5, 7), dtype=float)
    offsets = np.array([-0.020, -0.010, -0.005, 0.000, 0.005, 0.010, 0.020])
    for q_idx, offset in enumerate(offsets):
        pred[:, :, q_idx] = actual + offset

    metrics = evaluate_quantile_predictions(actual, pred, horizon=5)

    ordered = monotonic_quantiles_np(pred, median_idx=3)
    y_actual_t1 = select_prediction_horizon(actual, horizon_idx=0)
    expected_t1 = compute_all_metrics(
        y_actual_t1,
        ordered[:, 0, 3],
        y_pred_q10=ordered[:, 0, 1],
        y_pred_q90=ordered[:, 0, 5],
        y_pred_q02=ordered[:, 0, 0],
        y_pred_q98=ordered[:, 0, 6],
        y_pred_quantiles=ordered[:, 0, :],
    )
    raw_gap_mean, raw_gap_max = quantile_median_sort_gap(pred[:, 0, :], 3)
    expected_weekly = compute_weekly_metrics(actual, pred, horizon=5)

    assert np.isclose(metrics["mae"], expected_t1["mae"])
    assert np.isclose(metrics["variance_ratio"], expected_t1["variance_ratio"])
    assert metrics["quantile_crossing_rate"] == expected_t1["quantile_crossing_rate"]
    assert metrics["raw_quantile_crossing_rate"] == quantile_crossing_rate(pred[:, 0, :])
    assert metrics["raw_median_sort_gap_mean"] == raw_gap_mean
    assert metrics["raw_median_sort_gap_max"] == raw_gap_max
    assert np.isclose(
        metrics["weekly_magnitude_ratio"],
        expected_weekly["weekly_magnitude_ratio"],
    )
    assert np.isclose(
        metrics["weekly_mae_vs_naive_zero"],
        expected_weekly["weekly_mae_vs_naive_zero"],
    )


def test_evaluate_quantile_predictions_keeps_raw_and_sorted_crossing_separate():
    actual = np.tile(np.array([[0.01, 0.0, 0.0, 0.0, 0.0]]), (4, 1))
    pred = np.zeros((4, 5, 7), dtype=float)
    pred[..., 3] = actual
    pred[..., 1] = actual + 0.02
    pred[..., 5] = actual - 0.02
    pred[..., 0] = actual - 0.03
    pred[..., 2] = actual - 0.01
    pred[..., 4] = actual + 0.01
    pred[..., 6] = actual + 0.03

    metrics = evaluate_quantile_predictions(actual, pred, horizon=5)

    assert metrics["quantile_crossing_rate"] == 0.0
    assert metrics["sorted_quantile_crossing_rate"] == 0.0
    assert metrics["raw_quantile_crossing_rate"] > 0.0
    assert metrics["weekly_quantile_crossing_rate"] == 0.0
    assert metrics["weekly_sorted_quantile_crossing_rate"] == 0.0
    assert metrics["weekly_raw_quantile_crossing_rate"] > 0.0


def test_weekly_metrics_report_flipped_direction_diagnostics():
    actual = np.array(
        [
            [0.020, 0.000, 0.000, 0.000, 0.000],
            [-0.030, 0.000, 0.000, 0.000, 0.000],
            [0.040, 0.000, 0.000, 0.000, 0.000],
            [-0.050, 0.000, 0.000, 0.000, 0.000],
        ]
    )
    median = -actual
    pred = np.zeros((4, 5, 7), dtype=float)
    pred[..., 3] = median
    pred[..., 1] = median - 0.01
    pred[..., 5] = median + 0.01
    pred[..., 0] = median - 0.02
    pred[..., 2] = median - 0.005
    pred[..., 4] = median + 0.005
    pred[..., 6] = median + 0.02

    metrics = compute_weekly_metrics(actual, pred)

    assert metrics["weekly_directional_accuracy"] == 0.0
    assert metrics["weekly_directional_accuracy_flipped"] == 1.0
    assert metrics["weekly_tail_capture_rate"] == 0.0
    assert metrics["weekly_tail_capture_rate_flipped"] == 1.0
    assert metrics["weekly_sharpe_ratio_flipped"] > metrics["weekly_sharpe_ratio"]
    assert metrics["weekly_sign_correlation"] < -0.99


def test_weekly_metrics_report_prediction_and_actual_sign_bias():
    actual = np.array(
        [
            [0.020, 0.000, 0.000, 0.000, 0.000],
            [-0.030, 0.000, 0.000, 0.000, 0.000],
            [0.040, 0.000, 0.000, 0.000, 0.000],
            [-0.050, 0.000, 0.000, 0.000, 0.000],
        ]
    )
    median = -actual
    pred = np.zeros((4, 5, 7), dtype=float)
    pred[..., 3] = median
    pred[..., 1] = median - 0.01
    pred[..., 5] = median + 0.01
    pred[..., 0] = median - 0.02
    pred[..., 2] = median - 0.005
    pred[..., 4] = median + 0.005
    pred[..., 6] = median + 0.02

    metrics = compute_weekly_metrics(actual, pred)

    assert metrics["weekly_pred_positive_rate"] == 0.5
    assert metrics["weekly_actual_positive_rate"] == 0.5
    assert np.isclose(metrics["weekly_pred_mean"], 0.005)
    assert np.isclose(metrics["weekly_actual_mean"], -0.005)
    assert np.isclose(metrics["weekly_pred_median"], 0.005)
    assert np.isclose(metrics["weekly_actual_median"], -0.005)


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
    assert np.isfinite(metrics["weekly_interval_score_96"])
    assert metrics["weekly_sorted_quantile_crossing_rate"] == 0.0


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
    weekly_quantiles = cumulative_quantiles(monotonic_quantiles_np(pred), horizon=5)
    expected_pi80_width = np.mean(weekly_quantiles[:, 5] - weekly_quantiles[:, 1])
    expected_pi96_width = np.mean(weekly_quantiles[:, 6] - weekly_quantiles[:, 0])

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


def test_weekly_metrics_preserve_raw_crossing_and_promote_sorted_quantiles():
    actual = np.tile(np.array([[0.01, 0.0, 0.0, 0.0, 0.0]]), (4, 1))
    pred = np.zeros((4, 5, 7), dtype=float)
    pred[..., 3] = actual
    pred[..., 1] = actual + 0.02
    pred[..., 5] = actual - 0.02
    pred[..., 0] = actual - 0.03
    pred[..., 2] = actual - 0.01
    pred[..., 4] = actual + 0.01
    pred[..., 6] = actual + 0.03

    metrics = compute_weekly_metrics(actual, pred, horizon=5)

    assert metrics["weekly_quantile_crossing_rate"] == 0.0
    assert metrics["weekly_ordered_quantile_crossing_rate"] == 0.0
    assert metrics["weekly_public_quantile_crossing_rate"] == 0.0
    assert metrics["weekly_raw_quantile_crossing_rate"] > 0.0
    assert metrics["weekly_sorted_quantile_crossing_rate"] == 0.0
    assert metrics["weekly_pi80_width"] >= 0.0
