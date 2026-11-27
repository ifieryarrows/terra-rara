"""Regime-aware conformal calibration for weekly TFT-ASRO risk cones.

Supports:
- V5 FF-NexCP (Feedforward Student-t + 65/35 hybrid cone, damped volatility
  scaling, elevated-stress left-tail asymmetry, exponential-weighted NexCP
  quantile calibration with MA(h-1) effective sample size adjustment).
- Full backward compatibility with legacy V1 Split-CQR JSON artifacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

_DEFAULT_QUANTILES: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)


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
    *,
    clip_low: float = 0.5,
    clip_high: float = 2.0,
) -> tuple[np.ndarray, float | None]:
    """Compute bounded volatility-normalization factors ``clip(vol / med, lo, hi)``."""
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
    factors[valid_mask] = np.clip(vol[valid_mask] / ref, float(clip_low), float(clip_high))
    return factors, float(ref)


def _effective_sample_size(n: int, overlap_horizon: int = 5) -> float:
    """MA(h-1)-adjusted effective sample count for rolling horizon forecasts."""
    if n <= 0:
        return 0.0
    horizon = max(int(overlap_horizon), 1)
    return max(float(n) / float(horizon), 1.0)


def _split_conformal_quantile(
    scores: np.ndarray,
    alpha: float,
    *,
    overlap_horizon: int = 5,
    weights: np.ndarray | None = None,
) -> float:
    """Finite-sample (optionally NexCP-weighted) conformal quantile with MA-overlap ``n_eff``."""
    raw = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(raw)
    clean = raw[mask]
    n = len(clean)
    if n == 0:
        return 0.0
    alpha_f = min(max(float(alpha), 1e-4), 0.99)
    n_eff = _effective_sample_size(n, overlap_horizon=overlap_horizon)
    raw_level = (1.0 - alpha_f) * (n_eff + 1.0) / n_eff
    upper_cap = min(1.0 - 0.5 * alpha_f, 1.0 - 1.0 / max(float(n) * 2.0, 4.0))
    upper_cap = max(upper_cap, 1.0 - alpha_f)
    level = min(upper_cap, max(1.0 - alpha_f, float(raw_level)))

    if weights is not None:
        w = np.maximum(np.asarray(weights, dtype=np.float64)[mask], 0.0)
        w_sum = float(w.sum())
        if w_sum > 1e-12:
            order = np.argsort(clean)
            s_sorted = clean[order]
            w_sorted = w[order]
            cdf = np.cumsum(w_sorted) / w_sum
            idx = int(np.searchsorted(cdf, level, side="left"))
            return float(s_sorted[min(idx, len(s_sorted) - 1)])

    try:
        return float(np.quantile(clean, level, method="linear"))
    except TypeError:  # pragma: no cover - numpy<1.22 fallback
        return float(np.quantile(clean, level, interpolation="linear"))


def _blend_studentt_prior_quantiles(
    q_arr: np.ndarray,
    quantiles: list[float],
    realized_vol: np.ndarray | None,
    *,
    lam: float,
    student_t_df: float,
    horizon: int = 5,
) -> np.ndarray:
    """Blend 65/35 raw quantiles with unit-variance Student-t(df) volatility cone, preserving q50."""
    lam_f = min(max(float(lam), 0.0), 1.0)
    if lam_f >= 0.999 or realized_vol is None:
        return q_arr

    n = q_arr.shape[0]
    vol = np.asarray(realized_vol, dtype=np.float64).reshape(-1)
    if vol.size == 1 and n > 1:
        vol = np.full(n, float(vol[0]), dtype=np.float64)
    if vol.size != n:
        return q_arr

    vol_clean = np.where(np.isfinite(vol) & (vol > 1e-12), vol, np.nan)
    valid_v = vol_clean[np.isfinite(vol_clean)]
    if valid_v.size == 0:
        return q_arr
    fill_v = float(np.median(valid_v))
    vol5 = np.maximum(np.where(np.isfinite(vol_clean), vol_clean, fill_v), 1e-4) * math.sqrt(float(max(horizon, 1)))

    df = max(float(student_t_df), 3.2)
    t_scale = math.sqrt((df - 2.0) / df)
    q_probs = np.clip(np.asarray(quantiles, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    z_t = stats.t.ppf(q_probs, df=df) * t_scale
    med_idx = len(quantiles) // 2
    z_t = z_t - z_t[med_idx]

    q_vol = q_arr[:, med_idx : med_idx + 1] + vol5[:, None] * z_t[None, :]
    blended = lam_f * q_arr + (1.0 - lam_f) * q_vol
    blended[:, med_idx] = q_arr[:, med_idx]

    min_step = 1e-4
    for j in range(med_idx - 1, -1, -1):
        blended[:, j] = np.minimum(blended[:, j], blended[:, j + 1] - min_step)
    for j in range(med_idx + 1, len(quantiles)):
        blended[:, j] = np.maximum(blended[:, j], blended[:, j - 1] + min_step)
    return blended


def _apply_feedforward_vol_scaling(
    q_arr: np.ndarray,
    q_list: list[float],
    bscale: np.ndarray,
    *,
    left_tail_extra: float,
) -> np.ndarray:
    """Scale prior quantile widths by damped volatility factor before score extraction."""
    out = q_arr.copy()
    med_idx = len(q_list) // 2
    med = out[:, med_idx]
    min_step = 1e-4
    lt_mult = 1.0 + max(float(left_tail_extra), 0.0) * np.maximum(bscale - 1.0, 0.0)

    for j in range(0, med_idx):
        mult = bscale * lt_mult if q_list[j] <= 0.15 else bscale
        hw = np.maximum(med - out[:, j], min_step) * mult
        out[:, j] = med - hw
    for j in range(med_idx + 1, len(q_list)):
        hw = np.maximum(out[:, j] - med, min_step) * bscale
        out[:, j] = med + hw

    for j in range(med_idx - 1, -1, -1):
        out[:, j] = np.minimum(out[:, j], out[:, j + 1] - 0.5 * min_step)
    for j in range(med_idx + 1, len(q_list)):
        out[:, j] = np.maximum(out[:, j], out[:, j - 1] + 0.5 * min_step)
    return out


def fit_weekly_cqr(
    y_actual_weekly: np.ndarray,
    ordered_weekly_quantiles: np.ndarray,
    *,
    quantiles: tuple[float, ...] | list[float] = _DEFAULT_QUANTILES,
    realized_vol: np.ndarray | None = None,
    alpha_80: float = 0.20,
    alpha_96: float = 0.04,
    min_samples: int = 20,
    overlap_horizon: int = 5,
    prior_blend_lambda: float = 0.75,
    prior_student_t_df: float = 6.0,
    base_vol_power: float = 0.40,
    vol_clip_low: float = 0.45,
    vol_clip_high: float = 2.80,
    nexcp_gamma: float = 0.985,
    left_tail_extra: float = 0.15,
) -> dict[str, float | int | bool | str | None]:
    """Fit Feedforward-Feedback NexCP / Split-CQR calibration on validation predictions.

    Combines Benchmark-V5 validated design:
    - Feedforward blend (`prior_blend_lambda=0.75`) of raw 65/35 quantiles with
      heavy-tailed Student-t(df=6) volatility cones;
    - Damped multiplicative volatility scaling (`base_vol_power=0.40`, clip `[0.45, 2.80]`)
      + asymmetric downside stress expansion (`left_tail_extra=0.15`);
    - Exponential-weighted NexCP conformal quantiles (`nexcp_gamma=0.985`) adjusted for
      5-day overlapping MA(4) effective sample count.

    Preserves ``q50`` invariance and strict quantile monotonicity.
    """
    y = np.asarray(y_actual_weekly, dtype=np.float64).reshape(-1)
    q_arr = np.asarray(ordered_weekly_quantiles, dtype=np.float64)
    q_list = list(quantiles)
    n = min(len(y), len(q_arr))
    n_eff = _effective_sample_size(n, overlap_horizon=overlap_horizon)
    if q_arr.ndim != 2 or q_arr.shape[1] != len(q_list) or n < int(min_samples):
        return {
            "cqr_enabled": False,
            "method": "ff_nexcp_vol_normalized",
            "n_calibration": int(n),
            "n_eff": float(n_eff),
            "overlap_horizon": int(max(int(overlap_horizon), 1)),
            "q_hat_80": 0.0,
            "q_hat_96": 0.0,
            "median_val_vol": None,
            "vol_normalized": False,
            "base_vol_power": float(base_vol_power),
            "prior_blend_lambda": float(prior_blend_lambda),
            "prior_student_t_df": float(prior_student_t_df),
            "vol_clip_low": float(vol_clip_low),
            "vol_clip_high": float(vol_clip_high),
            "nexcp_gamma": float(nexcp_gamma),
            "left_tail_extra": float(left_tail_extra),
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
    vol_slice = realized_vol[:n] if realized_vol is not None else None
    vol_factors, median_val_vol = _vol_scale_factors(
        vol_slice,
        n,
        clip_low=vol_clip_low,
        clip_high=vol_clip_high,
    )

    med_idx = len(q_list) // 2
    q02_idx = q_list.index(0.02) if 0.02 in q_list else 0
    q10_idx = q_list.index(0.10) if 0.10 in q_list else 1
    q90_idx = q_list.index(0.90) if 0.90 in q_list else len(q_list) - 2
    q98_idx = q_list.index(0.98) if 0.98 in q_list else len(q_list) - 1

    q_prior = _blend_studentt_prior_quantiles(
        q_arr,
        q_list,
        vol_slice,
        lam=prior_blend_lambda,
        student_t_df=prior_student_t_df,
        horizon=overlap_horizon,
    )
    bscale = (
        np.clip(vol_factors, 0.25, 4.0) ** float(base_vol_power)
        if float(base_vol_power) > 0.0
        else np.ones_like(vol_factors)
    )
    scaled_q = _apply_feedforward_vol_scaling(
        q_prior,
        q_list,
        bscale,
        left_tail_extra=left_tail_extra,
    )

    scores_80 = compute_nonconformity(y, scaled_q[:, q10_idx], scaled_q[:, q90_idx]) / vol_factors
    scores_96 = compute_nonconformity(y, scaled_q[:, q02_idx], scaled_q[:, q98_idx]) / vol_factors

    gamma_f = float(nexcp_gamma)
    nexcp_weights = (
        np.power(gamma_f, np.arange(n - 1, -1, -1, dtype=np.float64))
        if 0.5 < gamma_f < 1.0
        else None
    )
    q_hat_80 = _split_conformal_quantile(scores_80, alpha_80, overlap_horizon=overlap_horizon, weights=nexcp_weights)
    q_hat_96 = _split_conformal_quantile(scores_96, alpha_96, overlap_horizon=overlap_horizon, weights=nexcp_weights)

    params: dict[str, float | int | bool | str | None] = {
        "cqr_enabled": True,
        "method": "ff_nexcp_vol_normalized",
        "n_calibration": int(n),
        "n_eff": float(n_eff),
        "overlap_horizon": int(max(int(overlap_horizon), 1)),
        "q_hat_80": float(q_hat_80),
        "q_hat_96": float(q_hat_96),
        "median_val_vol": float(median_val_vol) if median_val_vol is not None else None,
        "vol_normalized": median_val_vol is not None,
        "base_vol_power": float(base_vol_power),
        "prior_blend_lambda": float(prior_blend_lambda),
        "prior_student_t_df": float(prior_student_t_df),
        "vol_clip_low": float(vol_clip_low),
        "vol_clip_high": float(vol_clip_high),
        "nexcp_gamma": float(nexcp_gamma),
        "left_tail_extra": float(left_tail_extra),
        "alpha_80": float(alpha_80),
        "alpha_96": float(alpha_96),
        "min_half_width_retention": 0.35,
    }
    calibrated = apply_weekly_cqr(
        q_arr,
        quantiles=tuple(q_list),
        cqr_params=params,
        realized_vol=vol_slice,
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
    quantiles: tuple[float, ...] | list[float] = _DEFAULT_QUANTILES,
    cqr_params: dict | None = None,
    realized_vol: np.ndarray | float | None = None,
) -> np.ndarray:
    """Apply validation-fitted CQR / FF-NexCP adjustments in ordered weekly quantile space.

    Preserves the weekly median (``q50``) exactly and guarantees strictly
    monotonic outer envelopes ``q02 < q10 < q25 < q50 < q75 < q90 < q98``.
    Backward-compatible with legacy V1 ``split_cqr_vol_normalized`` artifacts.
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

    # Legacy V1 checkpoints omit V2 keys -> preserve exact V1 defaults when absent
    vol_clip_low = float(cqr_params.get("vol_clip_low", 0.5) or 0.5)
    vol_clip_high = float(cqr_params.get("vol_clip_high", 2.0) or 2.0)
    prior_blend_lambda = float(cqr_params.get("prior_blend_lambda", 1.0) if cqr_params.get("prior_blend_lambda") is not None else 1.0)
    prior_student_t_df = float(cqr_params.get("prior_student_t_df", 6.0) or 6.0)
    left_tail_extra = float(cqr_params.get("left_tail_extra", 0.0) or 0.0)
    overlap_horizon = int(cqr_params.get("overlap_horizon", 5) or 5)

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
        clip_low=vol_clip_low,
        clip_high=vol_clip_high,
    )

    arr = _blend_studentt_prior_quantiles(
        arr,
        q_list,
        vol_arr,
        lam=prior_blend_lambda,
        student_t_df=prior_student_t_df,
        horizon=overlap_horizon,
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
    low_80_raw_unscaled = np.maximum(med - arr[:, q10_idx], min_step)
    hi_80_raw_unscaled = np.maximum(arr[:, q90_idx] - med, min_step)
    frac_lo_25 = (
        np.clip(
            med - arr[:, q25_idx],
            min_step * 0.5,
            np.maximum(low_80_raw_unscaled - min_step * 0.5, min_step * 0.5),
        )
        / low_80_raw_unscaled
    )
    frac_hi_75 = (
        np.clip(
            arr[:, q75_idx] - med,
            min_step * 0.5,
            np.maximum(hi_80_raw_unscaled - min_step * 0.5, min_step * 0.5),
        )
        / hi_80_raw_unscaled
    )

    lt_mult = 1.0 + max(left_tail_extra, 0.0) * np.maximum(base_vol_scale - 1.0, 0.0)
    low_80_raw_hw = low_80_raw_unscaled * base_vol_scale * lt_mult
    hi_80_raw_hw = hi_80_raw_unscaled * base_vol_scale
    low_80_cal_hw = np.maximum(low_80_raw_hw + q_hat_80 * vol_factors, np.maximum(min_step, min_retention * low_80_raw_hw))
    hi_80_cal_hw = np.maximum(hi_80_raw_hw + q_hat_80 * vol_factors, np.maximum(min_step, min_retention * hi_80_raw_hw))

    arr[:, q10_idx] = med - low_80_cal_hw
    arr[:, q90_idx] = med + hi_80_cal_hw

    low_96_raw_hw = np.maximum(med - arr[:, q02_idx], min_step) * base_vol_scale * lt_mult
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

    if q10_idx < q25_idx < med_idx:
        lo_25_hw = np.clip(
            frac_lo_25 * low_80_cal_hw,
            min_step * 0.5,
            np.maximum(low_80_cal_hw - min_step * 0.5, min_step * 0.5),
        )
        arr[:, q25_idx] = np.clip(med - lo_25_hw, arr[:, q10_idx] + min_step * 0.5, med - min_step * 0.5)
    if med_idx < q75_idx < q90_idx:
        hi_75_hw = np.clip(
            frac_hi_75 * hi_80_cal_hw,
            min_step * 0.5,
            np.maximum(hi_80_cal_hw - min_step * 0.5, min_step * 0.5),
        )
        arr[:, q75_idx] = np.clip(med + hi_75_hw, med + min_step * 0.5, arr[:, q90_idx] - min_step * 0.5)

    return arr[0] if was_1d else arr
