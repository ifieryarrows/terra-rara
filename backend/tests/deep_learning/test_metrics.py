"""Tests for financial evaluation metrics."""

import numpy as np
import pytest

from deep_learning.training.metrics import (
    sharpe_ratio,
    sortino_ratio,
    directional_accuracy,
    tail_capture_rate,
    prediction_interval_coverage,
    prediction_interval_width,
    compute_all_metrics,
)


def test_sharpe_ratio_positive_returns():
    returns = np.array([0.01, 0.02, 0.015, 0.005, 0.01])
    sr = sharpe_ratio(returns)
    assert sr > 0, "Positive returns -> positive Sharpe"


def test_sharpe_ratio_zero_variance():
    returns = np.array([0.01, 0.01, 0.01])
    sr = sharpe_ratio(returns)
    assert sr == 0.0 or np.isfinite(sr)


def test_sortino_ratio_no_downside():
    returns = np.array([0.01, 0.02, 0.03, 0.01])
    sr = sortino_ratio(returns)
    assert sr == 0.0 or sr > 0


def test_directional_accuracy_perfect():
    actual = np.array([0.01, -0.02, 0.03, -0.01])
    pred = np.array([0.02, -0.01, 0.01, -0.03])
    da = directional_accuracy(actual, pred)
    assert da == 1.0


def test_directional_accuracy_zero():
    actual = np.array([0.01, -0.02, 0.03, -0.01])
    pred = np.array([-0.02, 0.01, -0.01, 0.03])
    da = directional_accuracy(actual, pred)
    assert da == 0.0


def test_tail_capture_rate_no_tails():
    actual = np.array([0.001, -0.002, 0.003])
    pred = np.array([0.001, -0.001, 0.002])
    tcr = tail_capture_rate(actual, pred, tail_threshold=0.015)
    assert tcr == 0.0


def test_prediction_interval_coverage():
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = np.array([0.5, 1.5, 2.0, 3.5, 4.5])
    upper = np.array([1.5, 2.5, 4.0, 4.5, 5.5])
    cov = prediction_interval_coverage(actual, lower, upper)
    assert 0 <= cov <= 1.0
    assert cov == 1.0


def test_prediction_interval_width():
    lower = np.array([0.0, 1.0, 2.0])
    upper = np.array([1.0, 3.0, 4.0])
    w = prediction_interval_width(lower, upper)
    assert w == pytest.approx(5.0 / 3.0)


def test_compute_all_metrics_full():
    np.random.seed(42)
    n = 100
    actual = np.random.normal(0, 0.02, n)
    pred = actual + np.random.normal(0, 0.005, n)
    q10 = pred - 0.02
    q90 = pred + 0.02
    q02 = pred - 0.04
    q98 = pred + 0.04

    m = compute_all_metrics(actual, pred, q10, q90, q02, q98)

    assert "mae" in m
    assert "rmse" in m
    assert "directional_accuracy" in m
    assert "sharpe_ratio" in m
    assert "pi80_coverage" in m
    assert "pi96_coverage" in m
    assert "variance_ratio" in m
    assert m["mae"] > 0
    assert 0 < m["pi80_coverage"] <= 1
