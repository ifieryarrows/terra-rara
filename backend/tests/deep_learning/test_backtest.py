"""Tests for the walk-forward backtest module."""

import numpy as np

from deep_learning.validation.backtest import run_backtest, compare_with_baseline


def test_run_backtest_basic():
    np.random.seed(42)
    n = 100
    y_actual = np.random.randn(n) * 0.02
    y_pred = y_actual * 0.5 + np.random.randn(n) * 0.01

    result = run_backtest(y_actual, y_pred, window_size=30, step_size=10)

    assert "summary" in result
    assert "windows" in result
    assert result["summary"]["n_windows"] > 0
    assert 0.0 <= result["summary"]["mean_da"] <= 1.0


def test_run_backtest_with_quantiles():
    np.random.seed(42)
    n = 60
    y_actual = np.random.randn(n) * 0.02
    y_pred = y_actual * 0.3
    y_q10 = y_pred - 0.01
    y_q90 = y_pred + 0.01

    result = run_backtest(y_actual, y_pred, y_q10, y_q90, window_size=50)
    assert "mean_pi80_coverage" in result["summary"]


def test_compare_with_baseline():
    tft_result = {
        "summary": {
            "mean_da": 0.55,
            "mean_sharpe": 0.5,
            "mean_mae": 0.010,
        }
    }
    theta_result = {
        "directional_accuracy": 0.52,
        "sharpe_ratio": 0.3,
        "mae": 0.012,
    }

    comp = compare_with_baseline(tft_result, theta_result)
    assert comp["tft_wins"] == 3
    assert comp["verdict"] == "TFT_SUPERIOR"


def test_compare_theta_wins():
    tft_result = {
        "summary": {
            "mean_da": 0.48,
            "mean_sharpe": -0.1,
            "mean_mae": 0.015,
        }
    }
    theta_result = {
        "directional_accuracy": 0.52,
        "sharpe_ratio": 0.3,
        "mae": 0.012,
    }

    comp = compare_with_baseline(tft_result, theta_result)
    assert comp["verdict"] == "THETA_SUPERIOR"
