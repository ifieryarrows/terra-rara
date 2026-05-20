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
import torch

from deep_learning.models.monotonic_quantiles import (
    DEFAULT_MONOTONIC_GAP_SCALE,
    enforce_monotonic_quantiles,
)


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


def cumulative_horizon(values: np.ndarray, horizon: int = 5) -> np.ndarray:
    """Sum the first ``horizon`` daily log-return targets into a weekly target."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] < horizon:
        raise ValueError(f"Need at least {horizon} horizons, got {arr.shape[1]}")
    return arr[:, :horizon].sum(axis=1)


def cumulative_quantiles(pred: np.ndarray, horizon: int = 5) -> np.ndarray:
    """Sum same-quantile daily path values into approximate weekly quantiles."""
    arr = np.asarray(pred, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected [n, horizon, q], got shape {arr.shape}")
    if arr.shape[1] < horizon:
        raise ValueError(f"Need at least {horizon} horizons, got {arr.shape[1]}")
    return arr[:, :horizon, :].sum(axis=1)


def monotonic_quantiles_np(
    pred: np.ndarray,
    median_idx: int | None = None,
) -> np.ndarray:
    """Apply the production monotonic quantile transform to a numpy tensor."""
    arr = np.asarray(pred, dtype=np.float64)
    if arr.shape[-1] == 0:
        return arr.copy()
    if median_idx is None:
        median_idx = arr.shape[-1] // 2
    tensor = torch.as_tensor(arr, dtype=torch.float64)
    ordered = enforce_monotonic_quantiles(
        tensor,
        median_idx=median_idx,
        min_gap=1e-5,
        gap_scale=DEFAULT_MONOTONIC_GAP_SCALE,
        init_bias=-3.0,
    )
    return ordered.detach().cpu().numpy()


def apply_weekly_median_cap_np(
    pred: np.ndarray,
    *,
    weekly_median_cap: float | None,
    quantiles: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    horizon: int = 5,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Bound the cumulative q50 path without using validation/test targets.

    The cap itself must be resolved from training targets before this helper is
    called. Lower/upper raw quantile channels are left untouched; the production
    monotonic transform rebuilds public quantiles around the bounded median.
    """
    arr = np.array(pred, dtype=np.float64, copy=True)
    if arr.ndim != 3:
        raise ValueError(f"Expected quantile prediction tensor [n,horizon,q], got {arr.shape}")
    if arr.shape[1] < horizon:
        raise ValueError(f"Prediction horizon too short: {arr.shape[1]} < {horizon}")
    if arr.shape[2] != len(quantiles):
        raise ValueError(
            f"Quantile dim mismatch: prediction has {arr.shape[2]}, config has {len(quantiles)}"
        )

    median_idx = len(quantiles) // 2
    raw_weekly = arr[:, :horizon, median_idx].sum(axis=1)
    cap_value = 0.0 if weekly_median_cap is None else float(weekly_median_cap)
    applied = np.zeros_like(raw_weekly, dtype=bool)

    if cap_value > 0.0 and raw_weekly.size:
        raw_abs = np.abs(raw_weekly)
        scale = np.minimum(1.0, cap_value / np.maximum(raw_abs, eps))
        applied = raw_abs > cap_value + eps
        arr[:, :, median_idx] = arr[:, :, median_idx] * scale.reshape(-1, 1)

    bounded_weekly = arr[:, :horizon, median_idx].sum(axis=1)
    diagnostics = {
        "weekly_median_cap": cap_value,
        "weekly_median_bound_applied_rate": float(np.mean(applied)) if applied.size else 0.0,
        "weekly_raw_pred_min": float(np.min(raw_weekly)) if raw_weekly.size else 0.0,
        "weekly_raw_pred_max": float(np.max(raw_weekly)) if raw_weekly.size else 0.0,
        "weekly_raw_pred_mean_abs": float(np.mean(np.abs(raw_weekly))) if raw_weekly.size else 0.0,
        "weekly_raw_pred_abs_median": float(np.median(np.abs(raw_weekly))) if raw_weekly.size else 0.0,
        "weekly_bounded_pred_min": float(np.min(bounded_weekly)) if bounded_weekly.size else 0.0,
        "weekly_bounded_pred_max": float(np.max(bounded_weekly)) if bounded_weekly.size else 0.0,
        "weekly_bounded_pred_mean_abs": (
            float(np.mean(np.abs(bounded_weekly))) if bounded_weekly.size else 0.0
        ),
        "weekly_bounded_pred_abs_median": (
            float(np.median(np.abs(bounded_weekly))) if bounded_weekly.size else 0.0
        ),
    }
    return arr, diagnostics


def _target_from_batch(batch) -> np.ndarray:
    y = batch[1] if isinstance(batch, (tuple, list)) and len(batch) > 1 else batch
    if isinstance(y, (tuple, list)):
        y = y[0]
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _target_scale_from_batch(batch) -> np.ndarray | None:
    if not isinstance(batch, (tuple, list)) or not batch:
        return None
    x = batch[0]
    if not isinstance(x, dict) or "target_scale" not in x:
        return None
    scale = x["target_scale"]
    if hasattr(scale, "detach"):
        scale = scale.detach().cpu().numpy()
    return np.asarray(scale, dtype=np.float64)


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "finite_rate": 0.0}
    finite = np.isfinite(arr)
    if not finite.any():
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "finite_rate": 0.0}
    clean = arr[finite]
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "finite_rate": float(np.mean(finite)),
    }


def summarize_dataloader_target_scale(
    dataloader,
    *,
    horizon: int = 5,
    max_batches: int | None = None,
) -> dict[str, float | int | bool]:
    """
    Summarize target and target-scale tensors emitted by a TFT dataloader.

    This helper is intentionally dataloader-based so trainer and hyperopt can
    audit the exact encoded decoder target tensors used by PyTorch Forecasting.
    """
    target_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        target_parts.append(_target_from_batch(batch))
        target_scale = _target_scale_from_batch(batch)
        if target_scale is not None:
            scale_parts.append(target_scale)

    target = np.concatenate(target_parts, axis=0) if target_parts else np.empty((0, horizon))
    target_stats = _finite_stats(target)
    if target.shape[1] >= horizon:
        weekly_actual = cumulative_horizon(target, horizon=horizon)
    else:
        weekly_actual = np.asarray([], dtype=np.float64)
    weekly_abs = np.abs(weekly_actual)

    audit: dict[str, float | int | bool] = {
        "target_decoder_samples": int(target.shape[0]) if target.ndim >= 1 else 0,
        "target_decoder_horizon": int(target.shape[1]) if target.ndim >= 2 else 0,
        "target_decoder_mean": target_stats["mean"],
        "target_decoder_std": target_stats["std"],
        "target_decoder_min": target_stats["min"],
        "target_decoder_max": target_stats["max"],
        "target_decoder_finite_rate": target_stats["finite_rate"],
        "actual_weekly_std": float(np.std(weekly_actual)) if weekly_actual.size else 0.0,
        "actual_weekly_mean_abs": float(np.mean(weekly_abs)) if weekly_abs.size else 0.0,
        "actual_weekly_abs_median": float(np.median(weekly_abs)) if weekly_abs.size else 0.0,
        "actual_weekly_min": float(np.min(weekly_actual)) if weekly_actual.size else 0.0,
        "actual_weekly_max": float(np.max(weekly_actual)) if weekly_actual.size else 0.0,
        "target_scale_present": bool(scale_parts),
    }

    if scale_parts:
        target_scale = np.concatenate(scale_parts, axis=0)
        scale_stats = _finite_stats(target_scale)
        audit.update({
            "target_scale_mean": scale_stats["mean"],
            "target_scale_std": scale_stats["std"],
            "target_scale_min": scale_stats["min"],
            "target_scale_max": scale_stats["max"],
            "target_scale_finite_rate": scale_stats["finite_rate"],
        })
    else:
        audit.update({
            "target_scale_mean": 0.0,
            "target_scale_std": 0.0,
            "target_scale_min": 0.0,
            "target_scale_max": 0.0,
            "target_scale_finite_rate": 0.0,
        })
    return audit


def resolve_weekly_median_cap(
    scale_audit: dict,
    *,
    floor: float = 0.08,
    std_multiple: float = 4.0,
) -> float:
    """Resolve the structural weekly median cap from training-target scale only."""
    weekly_std = float(scale_audit.get("actual_weekly_std", 0.0) or 0.0)
    if not np.isfinite(weekly_std) or weekly_std < 0.0:
        weekly_std = 0.0
    return float(max(float(floor), float(std_multiple) * weekly_std))


def magnitude_ratio(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """Median predicted absolute move divided by median actual absolute move."""
    denom = np.median(np.abs(np.asarray(y_actual, dtype=np.float64)))
    if denom < 1e-9:
        return 0.0
    return float(np.median(np.abs(np.asarray(y_pred, dtype=np.float64))) / denom)


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


def directional_accuracy_count(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int]:
    """Return ``(matches, n)`` for directional accuracy confidence intervals."""
    actual_sign = np.sign(y_actual)
    pred_sign = np.sign(y_pred)
    matches = (actual_sign == pred_sign) | ((actual_sign == 0) & (pred_sign == 0))
    return int(matches.sum()), int(matches.size)


def wilson_interval(
    successes: int,
    n: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Two-sided Wilson confidence interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return float((centre - margin) / denom), float((centre + margin) / denom)


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


def interval_score(
    y_actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.20,
) -> float:
    """
    Mean interval score for a central prediction interval.

    Lower is better. The score balances interval width with penalties when
    actual values fall below/above the interval.
    """
    y = np.asarray(y_actual, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return float(np.mean(width + (2.0 / alpha) * below + (2.0 / alpha) * above))


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
    direction_hits, direction_n = directional_accuracy_count(y_actual, y_pred_median)
    da_ci_low, da_ci_high = wilson_interval(direction_hits, direction_n)
    zero_mae = float(np.abs(y_actual).mean())
    zero_rmse = float(np.sqrt((y_actual ** 2).mean()))

    metrics: dict[str, float] = {
        "mae": float(np.abs(y_actual - y_pred_median).mean()),
        "rmse": float(np.sqrt(((y_actual - y_pred_median) ** 2).mean())),
        "directional_accuracy": directional_accuracy(y_actual, y_pred_median),
        "directional_accuracy_ci_low": da_ci_low,
        "directional_accuracy_ci_high": da_ci_high,
        "directional_accuracy_n": float(direction_n),
        "tail_capture_rate": tail_capture_rate(y_actual, y_pred_median, tail_threshold),
        "sharpe_ratio": sharpe_ratio(strategy_returns),
        "sortino_ratio": sortino_ratio(strategy_returns),
        "naive_zero_mae": zero_mae,
        "naive_zero_rmse": zero_rmse,
        "mae_vs_naive_zero": float(np.abs(y_actual - y_pred_median).mean() / (zero_mae + 1e-12)),
        "rmse_vs_naive_zero": float(
            np.sqrt(((y_actual - y_pred_median) ** 2).mean()) / (zero_rmse + 1e-12)
        ),
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
        metrics["pi80_interval_score"] = interval_score(y_actual, q10, q90, alpha=0.20)

    if y_pred_q02 is not None and y_pred_q98 is not None:
        q02 = np.asarray(y_pred_q02, dtype=np.float64)
        q98 = np.asarray(y_pred_q98, dtype=np.float64)
        metrics["pi96_coverage"] = prediction_interval_coverage(y_actual, q02, q98)
        metrics["pi96_width"] = prediction_interval_width(q02, q98)
        metrics["pi96_interval_score"] = interval_score(y_actual, q02, q98, alpha=0.04)

    if y_pred_quantiles is not None:
        q_arr = np.asarray(y_pred_quantiles, dtype=np.float64)
        ordered_q = monotonic_quantiles_np(q_arr)
        raw_crossing = quantile_crossing_rate(q_arr)
        ordered_crossing = quantile_crossing_rate(ordered_q)
        metrics["quantile_crossing_rate"] = ordered_crossing
        metrics["raw_quantile_crossing_rate"] = raw_crossing
        metrics["ordered_quantile_crossing_rate"] = ordered_crossing
        metrics["public_quantile_crossing_rate"] = ordered_crossing
        metrics["sorted_quantile_crossing_rate"] = ordered_crossing
        gap_mean, gap_max = quantile_median_sort_gap(q_arr)
        metrics["raw_median_sort_gap_mean"] = gap_mean
        metrics["raw_median_sort_gap_max"] = gap_max
        ordered_gap_mean, ordered_gap_max = quantile_median_sort_gap(ordered_q)
        metrics["median_sort_gap_mean"] = ordered_gap_mean
        metrics["median_sort_gap_max"] = ordered_gap_max
        metrics["ordered_median_sort_gap_mean"] = ordered_gap_mean
        metrics["ordered_median_sort_gap_max"] = ordered_gap_max
        metrics["sorted_median_sort_gap_mean"] = ordered_gap_mean
        metrics["sorted_median_sort_gap_max"] = ordered_gap_max

    return metrics


def compute_weekly_metrics(
    y_actual_path: np.ndarray,
    y_pred_quantiles_path: np.ndarray,
    quantiles: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    horizon: int = 5,
) -> dict[str, float]:
    """
    Compute weekly-first metrics from a daily log-return path.

    Internal evaluation remains in log-return space. Public API/UI conversion
    to simple returns happens only during inference formatting.
    """
    weekly_actual = cumulative_horizon(y_actual_path, horizon=horizon)
    raw_path = np.asarray(y_pred_quantiles_path, dtype=np.float64)
    ordered_path = monotonic_quantiles_np(raw_path, median_idx=len(quantiles) // 2)
    raw_weekly_quantiles = cumulative_quantiles(raw_path, horizon=horizon)
    weekly_quantiles = cumulative_quantiles(ordered_path, horizon=horizon)

    median_idx = len(quantiles) // 2
    q10_idx = quantiles.index(0.10)
    q90_idx = quantiles.index(0.90)
    q02_idx = quantiles.index(0.02)
    q98_idx = quantiles.index(0.98)

    weekly_pred = weekly_quantiles[:, median_idx]
    tail_threshold = (
        float(np.nanpercentile(np.abs(weekly_actual), 75))
        if len(weekly_actual)
        else 0.0
    )

    metrics = compute_all_metrics(
        weekly_actual,
        weekly_pred,
        y_pred_q10=weekly_quantiles[:, q10_idx],
        y_pred_q90=weekly_quantiles[:, q90_idx],
        y_pred_q02=weekly_quantiles[:, q02_idx],
        y_pred_q98=weekly_quantiles[:, q98_idx],
        y_pred_quantiles=weekly_quantiles,
        tail_threshold=tail_threshold,
    )

    weekly_metrics = {f"weekly_{k}": v for k, v in metrics.items()}
    if len(weekly_pred):
        weekly_metrics["weekly_pred_positive_rate"] = float(np.mean(weekly_pred > 0))
        weekly_metrics["weekly_actual_positive_rate"] = float(np.mean(weekly_actual > 0))
        weekly_metrics["weekly_pred_mean"] = float(np.mean(weekly_pred))
        weekly_metrics["weekly_actual_mean"] = float(np.mean(weekly_actual))
        weekly_metrics["weekly_pred_median"] = float(np.median(weekly_pred))
        weekly_metrics["weekly_actual_median"] = float(np.median(weekly_actual))
    else:
        weekly_metrics["weekly_pred_positive_rate"] = 0.0
        weekly_metrics["weekly_actual_positive_rate"] = 0.0
        weekly_metrics["weekly_pred_mean"] = 0.0
        weekly_metrics["weekly_actual_mean"] = 0.0
        weekly_metrics["weekly_pred_median"] = 0.0
        weekly_metrics["weekly_actual_median"] = 0.0
    weekly_metrics["weekly_directional_accuracy_flipped"] = directional_accuracy(
        weekly_actual,
        -weekly_pred,
    )
    flipped_strategy_returns = np.sign(-weekly_pred) * weekly_actual
    weekly_metrics["weekly_sharpe_ratio_flipped"] = sharpe_ratio(flipped_strategy_returns)
    weekly_metrics["weekly_tail_capture_rate_flipped"] = tail_capture_rate(
        weekly_actual,
        -weekly_pred,
        tail_threshold=tail_threshold,
    )
    if np.std(weekly_pred) > 1e-9 and np.std(weekly_actual) > 1e-9:
        weekly_metrics["weekly_sign_correlation"] = float(
            np.corrcoef(weekly_pred, weekly_actual)[0, 1]
        )
    else:
        weekly_metrics["weekly_sign_correlation"] = 0.0
    weekly_metrics["weekly_interval_quantile_source"] = 1.0
    weekly_metrics["weekly_approx_quantile_crossing_rate"] = quantile_crossing_rate(
        raw_weekly_quantiles
    )
    approx_gap_mean, approx_gap_max = quantile_median_sort_gap(raw_weekly_quantiles)
    weekly_metrics["weekly_approx_median_sort_gap_mean"] = approx_gap_mean
    weekly_metrics["weekly_approx_median_sort_gap_max"] = approx_gap_max
    weekly_metrics["weekly_raw_quantile_crossing_rate"] = quantile_crossing_rate(
        raw_weekly_quantiles
    )
    weekly_metrics["weekly_ordered_quantile_crossing_rate"] = quantile_crossing_rate(
        weekly_quantiles
    )
    weekly_metrics["weekly_public_quantile_crossing_rate"] = weekly_metrics[
        "weekly_ordered_quantile_crossing_rate"
    ]
    weekly_metrics["weekly_magnitude_ratio"] = magnitude_ratio(weekly_actual, weekly_pred)
    weekly_metrics["weekly_mean_actual_abs"] = float(np.mean(np.abs(weekly_actual)))
    weekly_metrics["weekly_mean_pred_abs"] = float(np.mean(np.abs(weekly_pred)))
    weekly_std = float(np.std(weekly_actual))
    weekly_metrics["weekly_actual_std"] = weekly_std
    weekly_metrics["weekly_pred_std"] = float(np.std(weekly_pred))
    weekly_metrics["weekly_pi80_width_ratio"] = float(
        weekly_metrics.get("weekly_pi80_width", 0.0) / (2.56 * weekly_std + 1e-8)
    )
    weekly_metrics["weekly_pi96_width_ratio"] = float(
        weekly_metrics.get("weekly_pi96_width", 0.0) / (4.10 * weekly_std + 1e-8)
    )
    weekly_metrics["weekly_interval_score_80"] = interval_score(
        weekly_actual,
        weekly_quantiles[:, q10_idx],
        weekly_quantiles[:, q90_idx],
        alpha=0.20,
    )
    weekly_metrics["weekly_interval_score_96"] = interval_score(
        weekly_actual,
        weekly_quantiles[:, q02_idx],
        weekly_quantiles[:, q98_idx],
        alpha=0.04,
    )
    weekly_metrics["weekly_sample_count"] = int(len(weekly_actual))
    return weekly_metrics


def evaluate_quantile_predictions(
    y_actual_path: np.ndarray,
    y_pred_quantiles_path: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    horizon: int = 5,
    weekly_median_cap: float | None = None,
) -> dict[str, float]:
    """
    Evaluate multi-horizon quantile predictions through the production metric path.

    Trainer and hyperopt both consume this helper so T+1 and weekly promotion
    signals are calculated with the same monotonic transform, raw-crossing
    diagnostics, and weekly cumulative target semantics.
    """
    quantiles = tuple(quantiles)
    pred_np = np.asarray(y_pred_quantiles_path, dtype=np.float64)
    if pred_np.ndim != 3:
        raise ValueError(f"Expected quantile prediction tensor [n,horizon,q], got {pred_np.shape}")
    if pred_np.shape[1] < horizon:
        raise ValueError(f"Prediction horizon too short: {pred_np.shape[1]} < {horizon}")
    if pred_np.shape[2] != len(quantiles):
        raise ValueError(
            f"Quantile dim mismatch: prediction has {pred_np.shape[2]}, config has {len(quantiles)}"
        )

    median_idx = len(quantiles) // 2
    q10_idx = quantiles.index(0.10) if 0.10 in quantiles else 1
    q90_idx = quantiles.index(0.90) if 0.90 in quantiles else len(quantiles) - 2
    q02_idx = quantiles.index(0.02) if 0.02 in quantiles else 0
    q98_idx = quantiles.index(0.98) if 0.98 in quantiles else len(quantiles) - 1

    y_actual_path = np.asarray(y_actual_path, dtype=np.float64)
    eval_pred_np, cap_diagnostics = apply_weekly_median_cap_np(
        pred_np,
        weekly_median_cap=weekly_median_cap,
        quantiles=quantiles,
        horizon=horizon,
    )
    ordered_pred_np = monotonic_quantiles_np(eval_pred_np, median_idx=median_idx)
    raw_pred_t1 = pred_np[:, 0, :]
    pred_t1 = ordered_pred_np[:, 0, :]
    y_actual_t1 = select_prediction_horizon(y_actual_path, horizon_idx=0)

    n = min(len(y_actual_t1), len(pred_t1))
    metrics = compute_all_metrics(
        y_actual_t1[:n],
        pred_t1[:n, median_idx],
        y_pred_q10=pred_t1[:n, q10_idx],
        y_pred_q90=pred_t1[:n, q90_idx],
        y_pred_q02=pred_t1[:n, q02_idx],
        y_pred_q98=pred_t1[:n, q98_idx],
        y_pred_quantiles=pred_t1[:n],
    )
    raw_gap_mean, raw_gap_max = quantile_median_sort_gap(raw_pred_t1[:n], median_idx)
    metrics["raw_quantile_crossing_rate"] = quantile_crossing_rate(raw_pred_t1[:n])
    metrics["raw_median_sort_gap_mean"] = raw_gap_mean
    metrics["raw_median_sort_gap_max"] = raw_gap_max

    n_path = min(len(y_actual_path), len(pred_np))
    weekly_metrics = compute_weekly_metrics(
        y_actual_path[:n_path],
        eval_pred_np[:n_path],
        quantiles=quantiles,
        horizon=horizon,
    )
    metrics.update(weekly_metrics)
    if weekly_median_cap is not None:
        weekly_actual = cumulative_horizon(y_actual_path[:n_path], horizon=horizon)
        raw_ordered_pred_np = monotonic_quantiles_np(pred_np[:n_path], median_idx=median_idx)
        bounded_ordered_pred_np = ordered_pred_np[:n_path]
        raw_weekly_pred = raw_ordered_pred_np[:, :horizon, median_idx].sum(axis=1)
        bounded_weekly_pred = bounded_ordered_pred_np[:, :horizon, median_idx].sum(axis=1)
        metrics.update(cap_diagnostics)
        metrics["weekly_raw_magnitude_ratio"] = magnitude_ratio(weekly_actual, raw_weekly_pred)
        metrics["weekly_bounded_magnitude_ratio"] = magnitude_ratio(
            weekly_actual,
            bounded_weekly_pred,
        )
        metrics["weekly_magnitude_ratio"] = metrics["weekly_bounded_magnitude_ratio"]
    return metrics
