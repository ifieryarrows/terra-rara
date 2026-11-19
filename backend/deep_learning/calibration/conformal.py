"""Regime-aware conformal calibration for weekly TFT intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConformalCalibrationResult:
    lower: np.ndarray
    upper: np.ndarray
    adjustment: float
    coverage_window: int
    bucket: str


def bucketize_regime(row: pd.Series) -> str:
    if row.get("regime_supply_shock", 0.0) >= 0.5:
        return "supply_shock"
    if row.get("regime_usd_pressure", 0.0) >= 0.5:
        return "usd_pressure"
    if row.get("regime_high_vol_chop", 0.0) >= 0.5:
        return "high_vol"
    if row.get("regime_risk_on_demand", 0.0) >= 0.5:
        return "risk_on"
    return "neutral"


def compute_nonconformity(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum(lower - actual, actual - upper)


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual_np = np.asarray(actual, dtype=np.float64)
    lower_np = np.asarray(lower, dtype=np.float64)
    upper_np = np.asarray(upper, dtype=np.float64)
    if actual_np.size == 0:
        return 0.0
    return float(np.mean((actual_np >= lower_np) & (actual_np <= upper_np)))


def rolling_conformal_adjustment(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.20,
) -> float:
    scores = compute_nonconformity(
        np.asarray(actual, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    scores = scores[np.isfinite(scores)]
    if len(scores) < 30:
        return 0.0
    return float(np.quantile(scores, 1.0 - alpha))


def apply_conformal_interval(
    lower_raw: np.ndarray,
    upper_raw: np.ndarray,
    adjustment: float,
) -> tuple[np.ndarray, np.ndarray]:
    return lower_raw - adjustment, upper_raw + adjustment


def select_bucket_adjustment(
    calibration: dict,
    bucket: str,
) -> float:
    """Return bucket adjustment, falling back to global adjustment."""
    bucket_adjustments = calibration.get("bucket_adjustments") or {}
    value = bucket_adjustments.get(bucket)
    if value is None:
        value = calibration.get("global_adjustment", 0.0)
    return float(value or 0.0)


def _vol_scale_factors(
    realized_vol: np.ndarray | None,
    n_samples: int,
    median_val_vol: float | None = None,
) -> tuple[np.ndarray, float | None]:
    """Compute bounded volatility-normalization factors ``clip(vol / med, 0.5, 2.0)``."""
    ones = np.ones(n_samples, dtype=np.float64)
    if realized_vol is None:
        return ones, median_val_vol

    vol = np.asarray(realized_vol, dtype=np.float64).reshape(-1)
    if vol.size == 1 and n_samples > 1:
        vol = np.full(n_samples, float(vol[0]), dtype=np.float64)
    if vol.size != n_samples:
        return ones, median_val_vol

    ref = median_val_vol
    if ref is None:
        valid = vol[np.isfinite(vol) & (vol > 1e-12)]
        if valid.size == 0:
            return ones, None
        ref = float(np.median(valid))
    if not np.isfinite(ref) or ref <= 1e-12:
        return ones, None

    factors = np.ones(n_samples, dtype=np.float64)
    valid_mask = np.isfinite(vol) & (vol > 1e-12)
    factors[valid_mask] = np.clip(vol[valid_mask] / ref, 0.5, 2.0)
    return factors, float(ref)


def _split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split conformal quantile at level ``ceil((1 - alpha)(n + 1)) / n``."""
    clean = np.asarray(scores, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    n = len(clean)
    if n == 0:
        return 0.0
    level = min(1.0, max(0.0, float(np.ceil((1.0 - alpha) * (n + 1))) / float(n)))
    try:
        return float(np.quantile(clean, level, method="higher" if level < 1.0 else "linear"))
    except TypeError:  # pragma: no cover - numpy<1.22 fallback
        return float(np.quantile(clean, level, interpolation="higher" if level < 1.0 else "linear"))


def fit_weekly_cqr(
    y_actual_weekly: np.ndarray,
    ordered_weekly_quantiles: np.ndarray,
    *,
    quantiles: tuple[float, ...] | list[float] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    realized_vol: np.ndarray | None = None,
    alpha_80: float = 0.20,
    alpha_96: float = 0.04,
    min_samples: int = 20,
) -> dict[str, float | int | bool | str | None]:
    """Fit Split / Volatility-Normalized Conformalized Quantile Regression (CQR).

    Operates strictly on ordered weekly log-return quantiles and validation
    targets, computing normalized signed nonconformity scores for both PI80
    ``[q10, q90]`` and PI96 ``[q02, q98]`` without touching ``q50``.
    """
    y = np.asarray(y_actual_weekly, dtype=np.float64).reshape(-1)
    q_arr = np.asarray(ordered_weekly_quantiles, dtype=np.float64)
    q_list = list(quantiles)
    n = min(len(y), len(q_arr))
    if q_arr.ndim != 2 or q_arr.shape[1] != len(q_list) or n < int(min_samples):
        return {
            "cqr_enabled": False,
            "method": "split_cqr_vol_normalized",
            "n_calibration": int(n),
            "q_hat_80": 0.0,
            "q_hat_96": 0.0,
            "median_val_vol": None,
            "vol_normalized": False,
            "alpha_80": float(alpha_80),
            "alpha_96": float(alpha_96),
            "validation_pi80_coverage_raw": interval_coverage(
                y[:n], q_arr[:n, q_list.index(0.10)], q_arr[:n, q_list.index(0.90)]
            ) if n > 0 else 0.0,
            "validation_pi80_coverage_calibrated": interval_coverage(
                y[:n], q_arr[:n, q_list.index(0.10)], q_arr[:n, q_list.index(0.90)]
            ) if n > 0 else 0.0,
            "validation_pi96_coverage_raw": interval_coverage(
                y[:n], q_arr[:n, q_list.index(0.02)], q_arr[:n, q_list.index(0.98)]
            ) if n > 0 else 0.0,
            "validation_pi96_coverage_calibrated": interval_coverage(
                y[:n], q_arr[:n, q_list.index(0.02)], q_arr[:n, q_list.index(0.98)]
            ) if n > 0 else 0.0,
        }

    y = y[:n]
    q_arr = q_arr[:n]
    vol_factors, median_val_vol = _vol_scale_factors(
        realized_vol[:n] if realized_vol is not None else None,
        n,
    )

    med_idx = len(q_list) // 2
    q02_idx = q_list.index(0.02) if 0.02 in q_list else 0
    q10_idx = q_list.index(0.10) if 0.10 in q_list else 1
    q90_idx = q_list.index(0.90) if 0.90 in q_list else len(q_list) - 2
    q98_idx = q_list.index(0.98) if 0.98 in q_list else len(q_list) - 1

    base_vol_power = 1.0
    base_vol_scale = np.clip(vol_factors, 0.25, 4.0) ** base_vol_power
    med = q_arr[:, med_idx]
    scaled_q = q_arr.copy()
    for lo_idx in (q02_idx, q10_idx):
        hw = np.maximum(med - q_arr[:, lo_idx], 1e-4) * base_vol_scale
        scaled_q[:, lo_idx] = med - hw
    for hi_idx in (q90_idx, q98_idx):
        hw = np.maximum(q_arr[:, hi_idx] - med, 1e-4) * base_vol_scale
        scaled_q[:, hi_idx] = med + hw

    scores_80 = compute_nonconformity(y, scaled_q[:, q10_idx], scaled_q[:, q90_idx]) / vol_factors
    scores_96 = compute_nonconformity(y, scaled_q[:, q02_idx], scaled_q[:, q98_idx]) / vol_factors

    q_hat_80 = _split_conformal_quantile(scores_80, alpha_80)
    q_hat_96 = _split_conformal_quantile(scores_96, alpha_96)

    params: dict[str, float | int | bool | str | None] = {
        "cqr_enabled": True,
        "method": "split_cqr_vol_normalized",
        "n_calibration": int(n),
        "q_hat_80": float(q_hat_80),
        "q_hat_96": float(q_hat_96),
        "median_val_vol": float(median_val_vol) if median_val_vol is not None else None,
        "vol_normalized": median_val_vol is not None,
        "base_vol_power": base_vol_power,
        "alpha_80": float(alpha_80),
        "alpha_96": float(alpha_96),
        "min_half_width_retention": 0.35,
    }
    calibrated = apply_weekly_cqr(
        q_arr,
        quantiles=tuple(q_list),
        cqr_params=params,
        realized_vol=realized_vol[:n] if realized_vol is not None else None,
    )
    params.update(
        {
            "validation_pi80_coverage_raw": interval_coverage(y, q_arr[:, q10_idx], q_arr[:, q90_idx]),
            "validation_pi80_coverage_calibrated": interval_coverage(
                y, calibrated[:, q10_idx], calibrated[:, q90_idx]
            ),
            "validation_pi96_coverage_raw": interval_coverage(y, q_arr[:, q02_idx], q_arr[:, q98_idx]),
            "validation_pi96_coverage_calibrated": interval_coverage(
                y, calibrated[:, q02_idx], calibrated[:, q98_idx]
            ),
        }
    )
    return params


def apply_weekly_cqr(
    ordered_weekly_quantiles: np.ndarray,
    *,
    quantiles: tuple[float, ...] | list[float] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    cqr_params: dict | None = None,
    realized_vol: np.ndarray | float | None = None,
) -> np.ndarray:
    """Apply validation-fitted Split CQR adjustments in ordered weekly quantile space.

    Preserves the weekly median (``q50``) exactly and guarantees strictly
    monotonic outer envelopes ``q02 < q10 < q25 < q50 < q75 < q90 < q98``.
    """
    arr = np.asarray(ordered_weekly_quantiles, dtype=np.float64).copy()
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(1, -1)

    if not cqr_params or not bool(cqr_params.get("cqr_enabled", False)) or arr.shape[0] == 0:
        return arr[0] if was_1d else arr

    q_list = list(quantiles)
    n = arr.shape[0]
    q_dim = arr.shape[1]
    if q_dim != len(q_list):
        raise ValueError(f"Quantile dim mismatch: {q_dim} != {len(q_list)}")

    med_idx = q_dim // 2
    q02_idx = q_list.index(0.02) if 0.02 in q_list else 0
    q10_idx = q_list.index(0.10) if 0.10 in q_list else 1
    q25_idx = q_list.index(0.25) if 0.25 in q_list else max(0, med_idx - 1)
    q75_idx = q_list.index(0.75) if 0.75 in q_list else min(q_dim - 1, med_idx + 1)
    q90_idx = q_list.index(0.90) if 0.90 in q_list else max(med_idx + 1, q_dim - 2)
    q98_idx = q_list.index(0.98) if 0.98 in q_list else q_dim - 1

    median_val_vol = cqr_params.get("median_val_vol")
    vol_arr = (
        np.full(n, float(realized_vol), dtype=np.float64)
        if isinstance(realized_vol, (int, float))
        else realized_vol
    )
    vol_factors, _ = _vol_scale_factors(
        vol_arr,
        n,
        median_val_vol=float(median_val_vol) if median_val_vol is not None else None,
    )

    q_hat_80 = float(cqr_params.get("q_hat_80", 0.0) or 0.0)
    q_hat_96 = float(cqr_params.get("q_hat_96", 0.0) or 0.0)
    min_retention = float(cqr_params.get("min_half_width_retention", 0.35) or 0.35)
    base_vol_power = float(cqr_params.get("base_vol_power", 0.0) or 0.0)
    base_vol_scale = (
        np.clip(vol_factors, 0.25, 4.0) ** base_vol_power
        if base_vol_power > 0.0
        else np.ones_like(vol_factors)
    )
    min_step = 1e-4

    med = arr[:, med_idx]
    low_80_raw_hw = np.maximum(med - arr[:, q10_idx], min_step) * base_vol_scale
    hi_80_raw_hw = np.maximum(arr[:, q90_idx] - med, min_step) * base_vol_scale
    low_80_cal_hw = np.maximum(low_80_raw_hw + q_hat_80 * vol_factors, np.maximum(min_step, min_retention * low_80_raw_hw))
    hi_80_cal_hw = np.maximum(hi_80_raw_hw + q_hat_80 * vol_factors, np.maximum(min_step, min_retention * hi_80_raw_hw))

    arr[:, q10_idx] = med - low_80_cal_hw
    arr[:, q90_idx] = med + hi_80_cal_hw

    low_96_raw_hw = np.maximum(med - arr[:, q02_idx], min_step) * base_vol_scale
    hi_96_raw_hw = np.maximum(arr[:, q98_idx] - med, min_step) * base_vol_scale
    low_96_cal_hw = np.maximum(
        np.maximum(low_96_raw_hw + q_hat_96 * vol_factors, np.maximum(min_step, min_retention * low_96_raw_hw)),
        low_80_cal_hw + min_step,
    )
    hi_96_cal_hw = np.maximum(
        np.maximum(hi_96_raw_hw + q_hat_96 * vol_factors, np.maximum(min_step, min_retention * hi_96_raw_hw)),
        hi_80_cal_hw + min_step,
    )

    arr[:, q02_idx] = med - low_96_cal_hw
    arr[:, q98_idx] = med + hi_96_cal_hw

    # Keep inner quartiles strictly inside [q10, q50] and [q50, q90]
    if q10_idx < q25_idx < med_idx:
        arr[:, q25_idx] = np.clip(arr[:, q25_idx], arr[:, q10_idx] + min_step * 0.5, med - min_step * 0.5)
    if med_idx < q75_idx < q90_idx:
        arr[:, q75_idx] = np.clip(arr[:, q75_idx], med + min_step * 0.5, arr[:, q90_idx] - min_step * 0.5)

    return arr[0] if was_1d else arr
