"""Causal Theta prior, macro fair-value residual z-score, and hybrid median blend.

Every routine here is strictly causal at time t (fit/evaluated solely on data
available at or before bar t, with blend weights fit exclusively on the
chronological validation split).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_MACRO_SYMBOLS = (
    "DX-Y.NYB",
    "CL=F",
    "FCX",
    "SCCO",
    "COPX",
    "GXC",
    "BHP",
)


def _column_variants(prefix: str, suffix: str) -> tuple[str, str]:
    sanitized = prefix.replace(".", "_").replace("-", "_")
    return f"{prefix}_{suffix}", f"{sanitized}_{suffix}"


def _find_feature_column(columns: pd.Index | Sequence[str], prefix: str, suffix: str) -> Optional[str]:
    col_set = set(columns)
    for candidate in _column_variants(prefix, suffix):
        if candidate in col_set:
            return candidate
    return None


def compute_causal_theta_5d_series(
    close: pd.Series,
    *,
    horizon: int = 5,
    min_window: int = 60,
    max_eval_tail: Optional[int] = None,
) -> pd.Series:
    """Compute expanding-origin causal 5D cumulative log-return Theta forecasts.

    For each bar index ``i >= max(min_window - 1, start_idx)``, evaluates
    ``theta_forecast(close.iloc[: i + 1], horizon=horizon)`` and converts the
    cumulative price forecast to log-return ``log(pred_price / last_price)``.
    When ``max_eval_tail`` is supplied (e.g. live inference), earlier rows stay
    neutral ``0.0`` so latency stays sub-second.
    """
    from deep_learning.baselines.theta_model import theta_forecast

    out = pd.Series(0.0, index=close.index, dtype=float, name="theta_5d_prior")
    clean = pd.to_numeric(close, errors="coerce")
    n = len(clean)
    if n < max(int(min_window), 30):
        return out

    start_pos = max(int(min_window) - 1, 29)
    if max_eval_tail is not None and int(max_eval_tail) > 0:
        start_pos = max(start_pos, n - int(max_eval_tail))

    values = np.zeros(n, dtype=np.float64)
    for idx in range(start_pos, n):
        window_series = clean.iloc[: idx + 1].dropna()
        if len(window_series) < 30:
            continue
        last_px = float(window_series.iloc[-1])
        if not np.isfinite(last_px) or last_px <= 0.0:
            continue
        try:
            fc = theta_forecast(window_series, horizon=horizon)
        except Exception:
            continue
        if "error" in fc:
            continue
        pred_px = float(fc.get("predicted_price_final") or 0.0)
        if np.isfinite(pred_px) and pred_px > 0.0:
            log_ret = float(np.clip(np.log(pred_px / last_px), -0.25, 0.25))
            values[idx] = log_ret

    out[:] = values
    return out


def compute_causal_macro_fair_value_zscore(
    price_features: pd.DataFrame,
    *,
    target_prefix: str = "HG=F",
    macro_symbols: Sequence[str] = _DEFAULT_MACRO_SYMBOLS,
    window: int = 60,
    min_periods: int = 40,
    ridge_alpha: float = 1.0,
) -> pd.Series:
    """Compute a strictly causal rolling Ridge macro fair-value residual z-score.

    Regresses the target's 20-day return against demeaned backward-window
    cross-asset 20-day returns (USD, oil, miners, China equity proxies). A
    negative z-score indicates copper lags its macro basket ("cheap" relative to
    macro peers); positive indicates overshoot. Zero-centered by construction.
    """
    out = pd.Series(0.0, index=price_features.index, dtype=float, name="macro_fair_value_zscore")
    if price_features.empty:
        return out

    target_col = _find_feature_column(price_features.columns, target_prefix, "ret20")
    if target_col is None:
        return out

    predictor_cols: list[str] = []
    target_variants = set(_column_variants(target_prefix, "ret20"))
    for sym in macro_symbols:
        col = _find_feature_column(price_features.columns, sym, "ret20")
        if col is not None and col not in target_variants and col not in predictor_cols:
            predictor_cols.append(col)
        if len(predictor_cols) >= 6:
            break

    if not predictor_cols:
        return out

    y_all = pd.to_numeric(price_features[target_col], errors="coerce").to_numpy(dtype=np.float64)
    X_all = (
        price_features[predictor_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )

    n, p = X_all.shape
    zscores = np.zeros(n, dtype=np.float64)
    win = max(int(window), int(min_periods), p + 5)
    min_req = max(int(min_periods), p + 5)
    eye = np.eye(p, dtype=np.float64) * float(max(ridge_alpha, 1e-6))

    for i in range(min_req - 1, n):
        start = max(0, i - win + 1)
        yw = y_all[start : i + 1]
        Xw = X_all[start : i + 1]
        valid = np.isfinite(yw) & np.isfinite(Xw).all(axis=1)
        if valid.sum() < min_req or not valid[-1]:
            continue

        yv = yw[valid]
        Xv = Xw[valid]
        x_mean = Xv.mean(axis=0, keepdims=True)
        y_mean = float(yv.mean())
        Xc = Xv - x_mean
        yc = yv - y_mean

        try:
            beta = np.linalg.solve(Xc.T @ Xc + eye, Xc.T @ yc)
        except np.linalg.LinAlgError:
            continue

        residuals = yc - (Xc @ beta)
        res_std = float(np.std(residuals))
        if res_std > 1e-6 and np.isfinite(residuals[-1]):
            zscores[i] = float(np.clip(residuals[-1] / res_std, -4.0, 4.0))

    out[:] = zscores
    return out


def _blend_validation_score(
    actual_weekly: np.ndarray,
    pred_weekly: np.ndarray,
    *,
    horizon: int = 5,
) -> dict[str, float]:
    from deep_learning.training.metrics import directional_accuracy, sharpe_ratio_hac

    actual = np.asarray(actual_weekly, dtype=np.float64)
    pred = np.asarray(pred_weekly, dtype=np.float64)
    actual_std = max(float(np.std(actual)), 1e-8)
    mae = float(np.mean(np.abs(actual - pred)))
    naive_mae = max(float(np.mean(np.abs(actual))), 1e-8)
    mae_vs_naive = float(mae / naive_mae)
    da = float(directional_accuracy(actual, pred))
    strat = np.sign(pred) * actual
    sh = float(sharpe_ratio_hac(strat, annualisation=52.0, max_lag=max(int(horizon) - 1, 0)))
    score = (
        0.50 * (mae / actual_std)
        - 0.12 * float(np.clip(sh, -2.0, 2.0))
        - 0.40 * (da - 0.50)
    )
    return {
        "score": float(score),
        "da": da,
        "sharpe_hac": sh,
        "mae_vs_naive": mae_vs_naive,
    }


def fit_theta_hybrid_blend(
    val_actual_weekly: np.ndarray,
    val_tft_weekly_median: np.ndarray,
    val_theta_weekly_pred: np.ndarray,
    *,
    horizon: int = 5,
    min_tft_weight: float = 0.35,
    max_theta_weight: float = 0.65,
    min_improvement: float = 0.005,
    max_mae_degradation: float = 0.01,
    min_theta_weight: float = 0.25,
    theta_insurance_min_da: float = 0.52,
    theta_insurance_max_mae_ratio: float = 1.01,
) -> dict:
    """Fit a convex ``w_tft * TFT + w_theta * Theta`` blend on validation only.

    Enforces ``w_tft >= min_tft_weight`` so the multi-factor TFT prior remains a
    substantive contributor. Activates when validation composite skill improves
    cleanly OR when Theta shows standalone validation edge and applying the
    structural ``min_theta_weight`` regime-insurance floor keeps MAE well-behaved.
    """
    y = np.asarray(val_actual_weekly, dtype=np.float64)
    p_tft = np.asarray(val_tft_weekly_median, dtype=np.float64)
    p_theta = np.asarray(val_theta_weekly_pred, dtype=np.float64)
    n = min(len(y), len(p_tft), len(p_theta))
    y, p_tft, p_theta = y[:n], p_tft[:n], p_theta[:n]
    valid = np.isfinite(y) & np.isfinite(p_tft) & np.isfinite(p_theta)
    y, p_tft, p_theta = y[valid], p_tft[valid], p_theta[valid]

    if len(y) < 20 or float(np.std(y)) <= 1e-8 or float(np.std(p_theta)) <= 1e-8:
        return {
            "enabled": False,
            "fit_split": "validation",
            "prior_feature": "HG=F_theta_5d_prior",
            "w_tft": 1.0,
            "w_theta": 0.0,
            "sample_count": int(len(y)),
            "reason": "insufficient_validation_signal",
        }

    base = _blend_validation_score(y, p_tft, horizon=horizon)
    theta_base = _blend_validation_score(y, p_theta, horizon=horizon)
    theta_has_edge = bool(
        (theta_base["da"] >= float(theta_insurance_min_da) or theta_base["sharpe_hac"] > 0.0)
        and theta_base["mae_vs_naive"] <= float(theta_insurance_max_mae_ratio)
    )
    upper_theta = float(np.clip(max_theta_weight, 0.0, 1.0 - max(0.0, min_tft_weight)))
    grid = [round(w, 2) for w in np.arange(0.0, upper_theta + 1e-9, 0.05)]

    best_w_theta = 0.0
    best_stats = base
    for w_th in grid:
        if w_th <= 0.0:
            continue
        w_tf = 1.0 - w_th
        blended = w_tf * p_tft + w_th * p_theta
        stats = _blend_validation_score(y, blended, horizon=horizon)
        if (
            stats["score"] < best_stats["score"]
            and stats["mae_vs_naive"] <= base["mae_vs_naive"] + float(max_mae_degradation)
        ):
            best_w_theta = w_th
            best_stats = stats

    floor_mae_slack = max(float(max_mae_degradation), 0.025)
    w_floor = round(min(max(0.0, float(min_theta_weight)), upper_theta), 4)

    improved = bool(
        best_w_theta > 0.0
        and best_stats["score"] <= base["score"] - float(min_improvement)
        and best_stats["mae_vs_naive"] <= base["mae_vs_naive"] + float(max_mae_degradation)
    )

    enabled = False
    selected_theta = 0.0
    selected_stats = base
    reason = "tft_standalone_preferred"

    if improved:
        selected_theta = float(best_w_theta)
        selected_stats = best_stats
        if theta_has_edge and w_floor > selected_theta:
            blended_floor = (1.0 - w_floor) * p_tft + w_floor * p_theta
            floor_stats = _blend_validation_score(y, blended_floor, horizon=horizon)
            if floor_stats["mae_vs_naive"] <= base["mae_vs_naive"] + floor_mae_slack:
                selected_theta = float(w_floor)
                selected_stats = floor_stats
        enabled = True
        reason = "validation_skill_improved"
    elif theta_has_edge and w_floor > 0.0:
        blended_floor = (1.0 - w_floor) * p_tft + w_floor * p_theta
        floor_stats = _blend_validation_score(y, blended_floor, horizon=horizon)
        if floor_stats["mae_vs_naive"] <= base["mae_vs_naive"] + floor_mae_slack:
            enabled = True
            selected_theta = float(w_floor)
            selected_stats = floor_stats
            reason = "theta_regime_insurance_floor"

    selected_tft = float(round(1.0 - selected_theta, 4))

    return {
        "enabled": enabled,
        "fit_split": "validation",
        "prior_feature": "HG=F_theta_5d_prior",
        "w_tft": selected_tft,
        "w_theta": float(selected_theta),
        "sample_count": int(len(y)),
        "val_tft_score": round(base["score"], 6),
        "val_hybrid_score": round(selected_stats["score"], 6),
        "val_tft_da": round(base["da"], 4),
        "val_hybrid_da": round(selected_stats["da"], 4),
        "val_tft_sharpe_hac": round(base["sharpe_hac"], 4),
        "val_hybrid_sharpe_hac": round(selected_stats["sharpe_hac"], 4),
        "val_tft_mae_vs_naive": round(base["mae_vs_naive"], 4),
        "val_hybrid_mae_vs_naive": round(selected_stats["mae_vs_naive"], 4),
        "val_theta_da": round(theta_base["da"], 4),
        "val_theta_sharpe_hac": round(theta_base["sharpe_hac"], 4),
        "val_theta_mae_vs_naive": round(theta_base["mae_vs_naive"], 4),
        "reason": reason,
    }


def apply_theta_hybrid_shift_to_path(
    pred_path_np: np.ndarray,
    theta_weekly_pred: np.ndarray | float,
    *,
    hybrid_blend: Optional[dict],
    quantiles: Sequence[float] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    horizon: int = 5,
) -> np.ndarray:
    """Shift only the raw median logits so the weekly median equals the hybrid blend.

    Outer softplus gap logits remain untouched, so ``monotonic_quantiles_np``
    deterministically shifts the entire quantile distribution by the exact
    blended delta without introducing quantile crossings.
    """
    out = np.asarray(pred_path_np, dtype=np.float64).copy()
    if not hybrid_blend or not hybrid_blend.get("enabled"):
        return out

    w_theta = float(hybrid_blend.get("w_theta", 0.0) or 0.0)
    if not np.isfinite(w_theta) or w_theta <= 0.0:
        return out
    w_theta = float(np.clip(w_theta, 0.0, 1.0))
    w_tft = 1.0 - w_theta

    squeezed = out.ndim == 2
    if squeezed:
        out = out[None, ...]
    if out.ndim != 3 or out.shape[1] < 1:
        return out[0] if squeezed else out

    h = min(int(horizon), out.shape[1])
    median_idx = list(quantiles).index(0.50) if 0.50 in quantiles else len(quantiles) // 2

    theta_arr = np.asarray(theta_weekly_pred, dtype=np.float64)
    if theta_arr.ndim == 0:
        theta_arr = np.full(out.shape[0], float(theta_arr), dtype=np.float64)
    else:
        theta_arr = theta_arr.reshape(-1)
    if len(theta_arr) != out.shape[0]:
        raise ValueError(
            f"theta_weekly_pred length {len(theta_arr)} != batch size {out.shape[0]}"
        )

    valid = np.isfinite(theta_arr)
    if valid.any():
        tft_med = out[:, :h, median_idx].sum(axis=1)
        blended = np.where(valid, w_tft * tft_med + w_theta * theta_arr, tft_med)
        delta_per_day = (blended - tft_med) / float(max(h, 1))
        out[:, :h, median_idx] = out[:, :h, median_idx] + delta_per_day[:, None]

    return out[0] if squeezed else out
