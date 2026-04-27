"""
Financial evaluation metrics for TFT-ASRO.

These go beyond standard ML metrics (MAE/RMSE) to measure what actually
matters in a trading context:
- Risk-adjusted return (Sharpe, Sortino)
- Directional accuracy
- Tail-event capture rate
- Calibration of prediction intervals
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_prediction_horizon(values: np.ndarray, horizon_idx: int = 0) -> np.ndarray:
    """
    Select one forecast horizon from a target/prediction matrix.

    TFT emits multi-horizon targets with shape ``(n_samples, prediction_length)``.
    Financial metrics used for promotion are T+1 metrics, so predictions from
    horizon 0 must be compared with actuals from horizon 0 only.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if horizon_idx < 0 or horizon_idx >= arr.shape[1]:
        raise IndexError(f"horizon_idx={horizon_idx} outside prediction length {arr.shape[1]}")
    return arr[:, horizon_idx].reshape(-1)


def quantile_crossing_rate(y_pred_quantiles: np.ndarray, eps: float = 1e-12) -> float:
    """
    Fraction of adjacent quantile pairs that violate monotonic ordering.
    """
    arr = np.asarray(y_pred_quantiles, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < 2:
        return 0.0
    violations = np.diff(arr, axis=-1) < -eps
    return float(violations.mean())


def quantile_median_sort_gap(
    y_pred_quantiles: np.ndarray,
    median_idx: int | None = None,
) -> tuple[float, float]:
    """
    Mean and max absolute movement of q50 after monotonic sorting.
    """
    arr = np.asarray(y_pred_quantiles, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] == 0:
        return 0.0, 0.0
    if median_idx is None:
        median_idx = arr.shape[-1] // 2
    sorted_arr = np.sort(arr, axis=-1)
    gap = np.abs(arr[..., median_idx] - sorted_arr[..., median_idx])
    return float(gap.mean()), float(gap.max())


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualisation: float = 252.0,
) -> float:
    """
    Annualised Sharpe Ratio.

    SR = sqrt(252) * (mean(r - rf) / std(r))
    """
    excess = returns - risk_free_rate / annualisation
    std = excess.std()
    if std < 1e-9:
        return 0.0
    return float(np.sqrt(annualisation) * excess.mean() / std)


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualisation: float = 252.0,
) -> float:
    """
    Annualised Sortino Ratio (penalises only downside volatility).
    """
    excess = returns - risk_free_rate / annualisation
    downside = excess[excess < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-9
    if downside_std < 1e-9:
        return 0.0
    return float(np.sqrt(annualisation) * excess.mean() / downside_std)


def directional_accuracy(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Fraction of days where predicted and actual returns share the same sign.
    """
    actual_sign = np.sign(y_actual)
    pred_sign = np.sign(y_pred)
    matches = (actual_sign == pred_sign) | ((actual_sign == 0) & (pred_sign == 0))
    return float(matches.mean())


def tail_capture_rate(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    tail_threshold: float = 0.015,
) -> float:
    """
    Directional accuracy on days where |actual_return| > threshold.

    These are the high-impact days that the low-variance trap misses.
    """
    tail_mask = np.abs(y_actual) > tail_threshold
    if tail_mask.sum() == 0:
        return 0.0
    return directional_accuracy(y_actual[tail_mask], y_pred[tail_mask])


def prediction_interval_coverage(
    y_actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Empirical coverage of the [lower, upper] prediction interval.
    Ideal value equals the nominal confidence level (e.g. 0.80 for 80% PI).
    """
    covered = (y_actual >= lower) & (y_actual <= upper)
    return float(covered.mean())


def prediction_interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Mean width of prediction intervals (narrower is better given coverage)."""
    return float((upper - lower).mean())


def compute_all_metrics(
    y_actual: np.ndarray,
    y_pred_median: np.ndarray,
    y_pred_q10: np.ndarray | None = None,
    y_pred_q90: np.ndarray | None = None,
    y_pred_q02: np.ndarray | None = None,
    y_pred_q98: np.ndarray | None = None,
    y_pred_quantiles: np.ndarray | None = None,
    tail_threshold: float = 0.015,
) -> dict[str, float]:
    """
    Compute the full financial metric suite.
    """
    y_actual = np.asarray(y_actual, dtype=np.float64)
    y_pred_median = np.asarray(y_pred_median, dtype=np.float64)

    # Simulated binary long/short strategy: take direction from model, size from actual return.
    # This is the correct series to compute Sharpe/Sortino on — not the raw predictions.
    # Using y_pred_median directly produces an inflated ratio because pred_std << actual_std.
    strategy_returns = np.sign(y_pred_median) * y_actual

    metrics: dict[str, float] = {
        "mae": float(np.abs(y_actual - y_pred_median).mean()),
        "rmse": float(np.sqrt(((y_actual - y_pred_median) ** 2).mean())),
        "directional_accuracy": directional_accuracy(y_actual, y_pred_median),
        "tail_capture_rate": tail_capture_rate(y_actual, y_pred_median, tail_threshold),
        "sharpe_ratio": sharpe_ratio(strategy_returns),
        "sortino_ratio": sortino_ratio(strategy_returns),
    }

    pred_std = float(y_pred_median.std())
    actual_std = float(y_actual.std())
    metrics["pred_std"] = pred_std
    metrics["actual_std"] = actual_std
    metrics["variance_ratio"] = pred_std / actual_std if actual_std > 1e-9 else 0.0

    if y_pred_q10 is not None and y_pred_q90 is not None:
        q10 = np.asarray(y_pred_q10, dtype=np.float64)
        q90 = np.asarray(y_pred_q90, dtype=np.float64)
        metrics["pi80_coverage"] = prediction_interval_coverage(y_actual, q10, q90)
        metrics["pi80_width"] = prediction_interval_width(q10, q90)

    if y_pred_q02 is not None and y_pred_q98 is not None:
        q02 = np.asarray(y_pred_q02, dtype=np.float64)
        q98 = np.asarray(y_pred_q98, dtype=np.float64)
        metrics["pi96_coverage"] = prediction_interval_coverage(y_actual, q02, q98)
        metrics["pi96_width"] = prediction_interval_width(q02, q98)

    if y_pred_quantiles is not None:
        q_arr = np.asarray(y_pred_quantiles, dtype=np.float64)
        metrics["quantile_crossing_rate"] = quantile_crossing_rate(q_arr)
        gap_mean, gap_max = quantile_median_sort_gap(q_arr)
        metrics["median_sort_gap_mean"] = gap_mean
        metrics["median_sort_gap_max"] = gap_max

    return metrics
