import numpy as np

from deep_learning.calibration.conformal import (
    apply_conformal_interval,
    rolling_conformal_adjustment,
    select_bucket_adjustment,
)


def test_undercovered_intervals_produce_positive_adjustment():
    actual = np.linspace(-0.05, 0.05, 60)
    lower = actual - 0.001
    upper = actual + 0.001
    upper[:30] = actual[:30] - 0.01
    adj = rolling_conformal_adjustment(actual, lower, upper, alpha=0.20)
    assert adj > 0


def test_adjustment_widens_interval():
    lower, upper = apply_conformal_interval(np.array([0.0]), np.array([1.0]), 0.2)
    assert lower[0] == -0.2
    assert upper[0] == 1.2


def test_fewer_than_30_samples_returns_zero_adjustment():
    actual = np.ones(20)
    assert rolling_conformal_adjustment(actual, actual - 0.1, actual + 0.1) == 0.0


def test_bucket_fallback_uses_global_adjustment():
    assert select_bucket_adjustment({"global_adjustment": 0.12, "bucket_adjustments": {}}, "high_vol") == 0.12
