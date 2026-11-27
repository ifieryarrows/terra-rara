"""Causal HAR-GJR forward volatility and market stress engine.

Fits strictly on the chronological training partition (with scale alignment on
validation origins), exports pure JSON-serializable linear/logistic parameters
(zero pickle/joblib fragility), and predicts 5-day forward volatility + shock
regime stress to normalize Split CQR risk cones across all 4 pipeline paths:
    1. Training & validation Split CQR fitting (`trainer.py`)
    2. Untouched test evaluation (`trainer.py`)
    3. Authentic walk-forward backtests (`TFTBacktestRunner`)
    4. Live weekly inference (`TFTPredictor`)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOL5_FLOOR = 0.005
DEFAULT_SHOCK_WEIGHT = 0.75
DEFAULT_SHOCK_SIGMA_MULT = 1.75
STRESS_ELEVATED_RATIO = 1.15
STRESS_HIGH_RATIO = 1.35
SHOCK_ELEVATED_PCT = 75.0
SHOCK_HIGH_PCT = 90.0

FEATURE_NAMES: tuple[str, ...] = (
    "log_rv3",
    "log_rv5",
    "log_rv10",
    "log_rv20",
    "log_rv60",
    "log_down5",
    "gjr1",
    "vol_accel_3_20",
    "vol_accel_5_60",
    "abs_r1_z",
    "abs_r5_z",
    "dd60",
    "dd_vel5",
    "rsi_ext",
    "dxy5",
    "cl5",
    "signed_vol",
    "cross_stress",
)


def _first_numeric_series(
    df: pd.DataFrame,
    candidates: Sequence[str],
    default: float = 0.0,
) -> Optional[pd.Series]:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    return None


def build_har_gjr_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build strictly causal multi-scale volatility & leverage stress features.

    Zero backward-fill (``.bfill()``) is used anywhere in this function so
    mutating rows at index ``> t`` can never alter feature rows at ``<= t``.
    Missing optional cross-asset or technical columns degrade gracefully to 0.
    """
    n = len(df)
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64), list(FEATURE_NAMES), np.empty(0, dtype=np.float64)

    close_s = _first_numeric_series(df, ("target_close", "close"))
    r1_s = _first_numeric_series(df, ("observed_return_1d", "observed_ret_1d", "HG=F_ret1", "target"))
    if r1_s is None and close_s is not None:
        safe_close = close_s.where(close_s > 0.0, np.nan)
        r1_s = np.log(safe_close).diff()
    if r1_s is None:
        r1_s = pd.Series(0.0, index=df.index, dtype=np.float64)
    else:
        r1_s = r1_s.fillna(0.0)

    rv20_s = _first_numeric_series(df, ("realized_vol_20d",))
    if rv20_s is None:
        rv20_s = r1_s.rolling(20, min_periods=5).std()
    rv20_s = rv20_s.ffill().fillna(0.012).clip(lower=1e-4)

    rv3_s = r1_s.rolling(3, min_periods=2).std().ffill().fillna(rv20_s).clip(lower=1e-4)
    rv5_s = r1_s.rolling(5, min_periods=3).std().ffill().fillna(rv20_s).clip(lower=1e-4)
    rv10_s = r1_s.rolling(10, min_periods=5).std().ffill().fillna(rv20_s).clip(lower=1e-4)
    rv60_s = r1_s.rolling(60, min_periods=15).std().ffill().fillna(rv20_s).clip(lower=1e-4)

    down5_sq = (np.minimum(r1_s, 0.0) ** 2).rolling(5, min_periods=3).mean()
    down5_s = np.sqrt(down5_sq.ffill().fillna(rv20_s**2)).clip(lower=1e-4)
    gjr1_s = np.abs(r1_s) * (1.0 + 0.5 * (r1_s < 0.0).astype(np.float64))

    vol_accel_3_20 = rv3_s / (rv20_s + 1e-6)
    vol_accel_5_60 = rv5_s / (rv60_s + 1e-6)
    abs_r1_z = np.abs(r1_s) / (rv20_s + 1e-6)

    r5_s = _first_numeric_series(df, ("HG=F_ret5",))
    if r5_s is None:
        r5_s = r1_s.rolling(5, min_periods=3).sum().fillna(0.0)
    else:
        r5_s = r5_s.fillna(0.0)
    abs_r5_z = np.abs(r5_s) / (rv20_s * math.sqrt(5.0) + 1e-6)

    dd60_s = _first_numeric_series(df, ("HG=F_dd_60",))
    if dd60_s is None:
        if close_s is not None:
            safe_c = close_s.where(close_s > 0.0, np.nan).ffill()
            roll_max = safe_c.rolling(60, min_periods=1).max()
            dd60_s = ((safe_c / roll_max.replace(0.0, np.nan)) - 1.0).fillna(0.0)
        else:
            cum_log = r1_s.cumsum()
            roll_peak = cum_log.rolling(60, min_periods=1).max()
            dd60_s = (np.exp(np.minimum(cum_log - roll_peak, 0.0)) - 1.0).fillna(0.0)
    else:
        dd60_s = dd60_s.fillna(0.0)
    dd_vel5 = dd60_s.diff(5).fillna(0.0)

    rsi_c = _first_numeric_series(df, ("HG=F_rsi_14_c",))
    if rsi_c is not None:
        rsi_ext = np.abs(rsi_c.fillna(0.0))
    else:
        rsi_raw = _first_numeric_series(df, ("HG=F_RSI_14",))
        if rsi_raw is not None:
            rsi_ext = np.abs(((rsi_raw - 50.0) / 50.0).fillna(0.0))
        else:
            rsi_ext = pd.Series(0.0, index=df.index, dtype=np.float64)

    dxy5_s = _first_numeric_series(df, ("DX-Y.NYB_ret5", "DX-Y_NYB_ret5", "dxy_ret5"))
    dxy5_s = dxy5_s.fillna(0.0) if dxy5_s is not None else pd.Series(0.0, index=df.index, dtype=np.float64)

    cl5_s = _first_numeric_series(df, ("CL=F_ret5", "CL_F_ret5", "cl_ret5"))
    cl5_s = cl5_s.fillna(0.0) if cl5_s is not None else pd.Series(0.0, index=df.index, dtype=np.float64)

    signed_vol_s = _first_numeric_series(df, ("HG=F_signed_vol_ratio",))
    if signed_vol_s is None:
        r10_s = r1_s.rolling(10, min_periods=3).sum().fillna(0.0)
        signed_vol_s = np.sign(r10_s) * ((rv5_s / (rv20_s + 1e-6)) - 1.0).clip(-3.0, 3.0)
    else:
        signed_vol_s = signed_vol_s.fillna(0.0)

    cross_stress = np.abs(dxy5_s) + np.abs(cl5_s) + np.abs(r5_s)

    col_map: dict[str, np.ndarray] = {
        "log_rv3": np.log(rv3_s.to_numpy(dtype=np.float64) + 1e-4),
        "log_rv5": np.log(rv5_s.to_numpy(dtype=np.float64) + 1e-4),
        "log_rv10": np.log(rv10_s.to_numpy(dtype=np.float64) + 1e-4),
        "log_rv20": np.log(rv20_s.to_numpy(dtype=np.float64) + 1e-4),
        "log_rv60": np.log(rv60_s.to_numpy(dtype=np.float64) + 1e-4),
        "log_down5": np.log(down5_s.to_numpy(dtype=np.float64) + 1e-4),
        "gjr1": gjr1_s.to_numpy(dtype=np.float64),
        "vol_accel_3_20": vol_accel_3_20.to_numpy(dtype=np.float64),
        "vol_accel_5_60": vol_accel_5_60.to_numpy(dtype=np.float64),
        "abs_r1_z": abs_r1_z.to_numpy(dtype=np.float64),
        "abs_r5_z": abs_r5_z.to_numpy(dtype=np.float64),
        "dd60": dd60_s.to_numpy(dtype=np.float64),
        "dd_vel5": dd_vel5.to_numpy(dtype=np.float64),
        "rsi_ext": rsi_ext.to_numpy(dtype=np.float64),
        "dxy5": dxy5_s.to_numpy(dtype=np.float64),
        "cl5": cl5_s.to_numpy(dtype=np.float64),
        "signed_vol": signed_vol_s.to_numpy(dtype=np.float64),
        "cross_stress": cross_stress.to_numpy(dtype=np.float64),
    }
    mat = np.column_stack(
        [np.nan_to_num(col_map[name], nan=0.0, posinf=0.0, neginf=0.0) for name in FEATURE_NAMES]
    )
    rv20_arr = np.nan_to_num(rv20_s.to_numpy(dtype=np.float64), nan=0.012, posinf=0.012, neginf=0.012)
    rv20_arr = np.maximum(rv20_arr, 1e-4)
    return mat, list(FEATURE_NAMES), rv20_arr


def _extract_forward_5d_returns(df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
    """Resolve 5D cumulative forward log returns for training label construction."""
    for col in ("target_5d_log_return", "target_5d_log"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)

    r1_s = _first_numeric_series(df, ("target", "observed_return_1d", "observed_ret_1d", "HG=F_ret1"))
    if r1_s is None:
        close_s = _first_numeric_series(df, ("target_close", "close"))
        if close_s is not None:
            safe_c = close_s.where(close_s > 0.0, np.nan)
            log_c = np.log(safe_c)
            return (log_c.shift(-int(horizon)) - log_c).to_numpy(dtype=np.float64)
        return np.full(len(df), np.nan, dtype=np.float64)

    acc = pd.Series(0.0, index=df.index, dtype=np.float64)
    for h in range(1, int(horizon) + 1):
        acc = acc + r1_s.shift(-h)
    return acc.to_numpy(dtype=np.float64)


def fit_har_gjr_volatility_stress(
    master_df: pd.DataFrame,
    *,
    train_end_pos: int,
    val_origin_pos: Optional[np.ndarray | Sequence[int]] = None,
    horizon: int = 5,
    shock_weight: float = DEFAULT_SHOCK_WEIGHT,
    shock_sigma_mult: float = DEFAULT_SHOCK_SIGMA_MULT,
    min_train_samples: int = 60,
    stress_elevated_threshold: float = STRESS_ELEVATED_RATIO,
    stress_high_threshold: float = STRESS_HIGH_RATIO,
) -> dict[str, Any]:
    """Fit JSON-serializable HAR-GJR volatility + shock parameters without lookahead.

    - Regressor & shock classifier fit strictly on settled training rows
      ``[0 .. train_end_pos - horizon]``.
    - Median scale ratio aligns raw log-|y| predictions to 5D volatility scale
      using validation origins (falling back to training rows when omitted).
    """
    n = len(master_df)
    if n == 0:
        return {"version": 1, "enabled": False, "reason": "empty_dataframe", "n_train": 0}

    h = max(int(horizon), 1)
    # Strict purge: row `i` predicts `close[i+h] / close[i]`, so only `i <= train_end_pos - h` is settled inside train.
    train_slice_end = max(0, min(int(train_end_pos) - h + 1, n))
    if train_slice_end < int(min_train_samples):
        return {
            "version": 1,
            "enabled": False,
            "reason": "insufficient_train_slice",
            "n_train": int(train_slice_end),
        }

    vmat, feature_names, rv20_arr = build_har_gjr_features(master_df)
    weekly_y = _extract_forward_5d_returns(master_df, horizon=h)

    y_tr_raw = weekly_y[:train_slice_end]
    valid_tr = np.isfinite(y_tr_raw) & np.isfinite(rv20_arr[:train_slice_end])
    n_valid_tr = int(valid_tr.sum())
    if n_valid_tr < int(min_train_samples):
        return {
            "version": 1,
            "enabled": False,
            "reason": "insufficient_finite_targets",
            "n_train": n_valid_tr,
        }

    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler

    X_tr = vmat[:train_slice_end][valid_tr]
    y_5d_tr = y_tr_raw[valid_tr]
    vol5_tr = np.maximum(rv20_arr[:train_slice_end][valid_tr] * math.sqrt(float(h)), VOL5_FLOOR)

    y_abs_log_tr = np.log(np.abs(y_5d_tr) + 1e-3)
    y_shock_tr = (np.abs(y_5d_tr) >= float(shock_sigma_mult) * vol5_tr).astype(int)

    scaler = StandardScaler().fit(X_tr)
    scale_safe = np.where(np.asarray(scaler.scale_, dtype=np.float64) > 1e-12, scaler.scale_, 1.0)
    mean_arr = np.asarray(scaler.mean_, dtype=np.float64)
    X_tr_s = np.clip((X_tr - mean_arr) / scale_safe, -6.0, 6.0)

    ridge = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(X_tr_s, y_abs_log_tr)

    # Align median scale ratio on validation origins when available (zero test leakage)
    val_idx: Optional[np.ndarray] = None
    if val_origin_pos is not None:
        raw_idx = np.asarray(val_origin_pos, dtype=int).reshape(-1)
        raw_idx = raw_idx[(raw_idx >= 0) & (raw_idx < n)]
        if raw_idx.size > 0:
            val_idx = raw_idx

    if val_idx is not None and val_idx.size >= 5:
        X_ref = np.clip((vmat[val_idx] - mean_arr) / scale_safe, -6.0, 6.0)
        vol5_ref = np.maximum(rv20_arr[val_idx] * math.sqrt(float(h)), VOL5_FLOOR)
        n_val = int(val_idx.size)
    else:
        X_ref = X_tr_s
        vol5_ref = vol5_tr
        n_val = 0

    raw_ref_vol = np.exp(
        np.clip(X_ref @ np.asarray(ridge.coef_, dtype=np.float64) + float(ridge.intercept_), -10.0, 2.0)
    )
    med_raw_ref = float(np.median(raw_ref_vol)) if raw_ref_vol.size else 1.0
    med_vol5_ref = float(np.median(vol5_ref)) if vol5_ref.size else VOL5_FLOOR
    vol_scale_ratio = float(np.clip(med_vol5_ref / max(med_raw_ref, 1e-6), 0.1, 10.0))

    n_shock_pos = int(y_shock_tr.sum())
    n_shock_neg = int(len(y_shock_tr) - n_shock_pos)
    if n_shock_pos >= 5 and n_shock_neg >= 5:
        clf_lr = LogisticRegression(C=0.25, max_iter=500, random_state=42).fit(X_tr_s, y_shock_tr)
        shock_enabled = True
        shock_coef = np.asarray(clf_lr.coef_[0], dtype=np.float64)
        shock_intercept = float(clf_lr.intercept_[0])
        logits_tr = np.clip(X_tr_s @ shock_coef + shock_intercept, -30.0, 30.0)
        p_shock_tr = 1.0 / (1.0 + np.exp(-logits_tr))
    else:
        shock_enabled = False
        shock_coef = np.zeros(len(feature_names), dtype=np.float64)
        shock_intercept = -10.0
        p_shock_tr = np.zeros(n_valid_tr, dtype=np.float64)

    ref_quantiles = np.quantile(p_shock_tr, np.linspace(0.0, 1.0, 101)).astype(np.float64)
    shock_p75 = float(np.quantile(p_shock_tr, 0.75)) if p_shock_tr.size else 0.0
    shock_p90 = float(np.quantile(p_shock_tr, 0.90)) if p_shock_tr.size else 0.0

    return {
        "version": 1,
        "enabled": True,
        "model_family": "har_gjr_ridge_logistic_json",
        "horizon": int(h),
        "feature_names": feature_names,
        "scaler_mean": [float(x) for x in mean_arr.tolist()],
        "scaler_scale": [float(x) for x in scale_safe.tolist()],
        "vol_coef": [float(x) for x in np.asarray(ridge.coef_, dtype=np.float64).tolist()],
        "vol_intercept": float(ridge.intercept_),
        "vol_scale_ratio": float(vol_scale_ratio),
        "shock_enabled": bool(shock_enabled),
        "shock_coef": [float(x) for x in shock_coef.tolist()],
        "shock_intercept": float(shock_intercept),
        "shock_weight": float(shock_weight),
        "shock_sigma_mult": float(shock_sigma_mult),
        "shock_p75": float(shock_p75),
        "shock_p90": float(shock_p90),
        "shock_ref_quantiles": [float(x) for x in ref_quantiles.tolist()],
        "stress_elevated_threshold": float(stress_elevated_threshold),
        "stress_high_threshold": float(stress_high_threshold),
        "shock_elevated_pct": float(SHOCK_ELEVATED_PCT),
        "shock_high_pct": float(SHOCK_HIGH_PCT),
        "n_train": int(n_valid_tr),
        "n_val": int(n_val),
        "train_shock_rate": float(y_shock_tr.mean()) if len(y_shock_tr) else 0.0,
    }


def is_valid_har_gjr_params(params: Optional[Mapping[str, Any]]) -> bool:
    """Return True when ``params`` contains finite, enabled JSON HAR-GJR weights."""
    if not isinstance(params, Mapping) or not bool(params.get("enabled", False)):
        return False
    required = ("feature_names", "scaler_mean", "scaler_scale", "vol_coef", "vol_intercept", "vol_scale_ratio")
    if not all(k in params for k in required):
        return False
    try:
        names = list(params["feature_names"])
        d = len(names)
        if d == 0:
            return False
        for key in ("scaler_mean", "scaler_scale", "vol_coef"):
            arr = np.asarray(params[key], dtype=np.float64)
            if arr.shape != (d,) or not np.isfinite(arr).all():
                return False
        if not np.isfinite(float(params["vol_intercept"])) or not np.isfinite(float(params["vol_scale_ratio"])):
            return False
        if float(params["vol_scale_ratio"]) <= 0.0:
            return False
        if bool(params.get("shock_enabled", False)):
            sc = np.asarray(params.get("shock_coef", []), dtype=np.float64)
            if sc.shape != (d,) or not np.isfinite(sc).all():
                return False
            if not np.isfinite(float(params.get("shock_intercept", 0.0))):
                return False
        return True
    except Exception:
        return False


def _classify_stress_regime(
    stress_ratio: float,
    shock_percentile: float,
    *,
    elevated_ratio: float = STRESS_ELEVATED_RATIO,
    high_ratio: float = STRESS_HIGH_RATIO,
    elevated_pct: float = SHOCK_ELEVATED_PCT,
    high_pct: float = SHOCK_HIGH_PCT,
) -> str:
    if stress_ratio >= high_ratio or shock_percentile >= high_pct:
        return "HIGH_STRESS"
    if stress_ratio >= elevated_ratio or shock_percentile >= elevated_pct:
        return "ELEVATED"
    return "CALM"


def predict_har_gjr_volatility_stress(
    df: pd.DataFrame,
    har_gjr_params: Mapping[str, Any],
    *,
    positions: Optional[np.ndarray | Sequence[int]] = None,
) -> dict[str, np.ndarray]:
    """Run causal pure-NumPy HAR-GJR inference on ``df`` (optionally sliced at ``positions``)."""
    if not is_valid_har_gjr_params(har_gjr_params):
        raise ValueError("Invalid or disabled HAR-GJR parameters")

    vmat, computed_names, rv20_arr = build_har_gjr_features(df)
    target_names = [str(x) for x in har_gjr_params["feature_names"]]
    if target_names != computed_names:
        name_to_col = {name: i for i, name in enumerate(computed_names)}
        cols = [
            vmat[:, name_to_col[name]] if name in name_to_col else np.zeros(len(df), dtype=np.float64)
            for name in target_names
        ]
        X = np.column_stack(cols) if len(df) > 0 else np.empty((0, len(target_names)), dtype=np.float64)
    else:
        X = vmat

    if positions is not None:
        pos = np.asarray(positions, dtype=int).reshape(-1)
        pos = np.clip(pos, 0, max(len(df) - 1, 0))
        X = X[pos]
        rv20_sub = rv20_arr[pos]
    else:
        rv20_sub = rv20_arr

    horizon = max(int(har_gjr_params.get("horizon", 5) or 5), 1)
    sqrt_h = math.sqrt(float(horizon))
    mean_arr = np.asarray(har_gjr_params["scaler_mean"], dtype=np.float64)
    scale_arr = np.maximum(np.asarray(har_gjr_params["scaler_scale"], dtype=np.float64), 1e-12)
    vol_coef = np.asarray(har_gjr_params["vol_coef"], dtype=np.float64)
    vol_intercept = float(har_gjr_params["vol_intercept"])
    vol_scale_ratio = float(har_gjr_params["vol_scale_ratio"])

    X_s = np.clip((X - mean_arr) / scale_arr, -6.0, 6.0)
    raw_log_vol = np.clip(X_s @ vol_coef + vol_intercept, -10.0, 2.0)
    hat_vol_5d = np.maximum(np.exp(raw_log_vol) * vol_scale_ratio, VOL5_FLOOR)
    rv20_vol5 = np.maximum(rv20_sub * sqrt_h, VOL5_FLOOR)

    if bool(har_gjr_params.get("shock_enabled", False)):
        shock_coef = np.asarray(har_gjr_params["shock_coef"], dtype=np.float64)
        shock_intercept = float(har_gjr_params["shock_intercept"])
        logits = np.clip(X_s @ shock_coef + shock_intercept, -30.0, 30.0)
        p_shock = 1.0 / (1.0 + np.exp(-logits))
    else:
        p_shock = np.zeros(len(X_s), dtype=np.float64)

    shock_weight = float(har_gjr_params.get("shock_weight", DEFAULT_SHOCK_WEIGHT) or DEFAULT_SHOCK_WEIGHT)
    cqr_vol_scale = np.maximum((hat_vol_5d / sqrt_h) * (1.0 + shock_weight * p_shock), 1e-4)
    stress_ratio = hat_vol_5d / np.maximum(rv20_vol5, VOL5_FLOOR)

    ref_q = np.asarray(har_gjr_params.get("shock_ref_quantiles") or [], dtype=np.float64)
    if ref_q.size >= 2 and np.isfinite(ref_q).all():
        ranks = np.searchsorted(ref_q, p_shock, side="right")
        shock_percentile = np.clip((ranks.astype(np.float64) / float(len(ref_q))) * 100.0, 0.0, 100.0)
    else:
        shock_p75 = float(har_gjr_params.get("shock_p75", 0.02) or 0.02)
        shock_p90 = float(har_gjr_params.get("shock_p90", 0.04) or 0.04)
        shock_percentile = np.where(
            p_shock >= shock_p90,
            95.0,
            np.where(p_shock >= shock_p75, 80.0, 40.0),
        )

    elevated_ratio = float(har_gjr_params.get("stress_elevated_threshold", STRESS_ELEVATED_RATIO) or STRESS_ELEVATED_RATIO)
    high_ratio = float(har_gjr_params.get("stress_high_threshold", STRESS_HIGH_RATIO) or STRESS_HIGH_RATIO)
    elevated_pct = float(har_gjr_params.get("shock_elevated_pct", SHOCK_ELEVATED_PCT) or SHOCK_ELEVATED_PCT)
    high_pct = float(har_gjr_params.get("shock_high_pct", SHOCK_HIGH_PCT) or SHOCK_HIGH_PCT)

    regimes = np.asarray(
        [
            _classify_stress_regime(
                float(sr),
                float(sp),
                elevated_ratio=elevated_ratio,
                high_ratio=high_ratio,
                elevated_pct=elevated_pct,
                high_pct=high_pct,
            )
            for sr, sp in zip(stress_ratio, shock_percentile)
        ],
        dtype=object,
    )

    return {
        "hat_vol_5d": hat_vol_5d,
        "rv20_vol5": rv20_vol5,
        "p_shock": p_shock,
        "shock_percentile": shock_percentile,
        "stress_ratio": stress_ratio,
        "cqr_vol_scale": cqr_vol_scale,
        "regime": regimes,
    }


def resolve_cqr_vol_scale(
    df: Optional[pd.DataFrame],
    positions: Optional[np.ndarray | Sequence[int]],
    *,
    har_gjr_params: Optional[Mapping[str, Any]],
    fallback_vol: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Return causal HAR-GJR ``cqr_vol_scale`` array or fail-open ``fallback_vol``."""
    if (
        df is None
        or df.empty
        or positions is None
        or len(positions) == 0
        or not is_valid_har_gjr_params(har_gjr_params)
    ):
        return fallback_vol
    try:
        out = predict_har_gjr_volatility_stress(df, har_gjr_params, positions=positions)
        scale = np.asarray(out["cqr_vol_scale"], dtype=np.float64)
        if scale.shape[0] != len(positions) or not np.isfinite(scale).all():
            return fallback_vol
        return scale
    except Exception as exc:
        logger.warning("HAR-GJR CQR vol resolution skipped (fail-open to realized_vol_20d): %s", exc)
        return fallback_vol


def predict_latest_volatility_stress(
    df: Optional[pd.DataFrame],
    har_gjr_params: Optional[Mapping[str, Any]],
) -> tuple[Optional[float], Optional[dict[str, Any]]]:
    """Evaluate HAR-GJR at the latest forecast origin for live inference (fail-open)."""
    if df is None or df.empty or not is_valid_har_gjr_params(har_gjr_params):
        return None, None
    try:
        out = predict_har_gjr_volatility_stress(df, har_gjr_params, positions=[len(df) - 1])
        cqr_vol = float(out["cqr_vol_scale"][0])
        hat_vol5 = float(out["hat_vol_5d"][0])
        rv20_vol5 = float(out["rv20_vol5"][0])
        stress_ratio = float(out["stress_ratio"][0])
        p_shock = float(out["p_shock"][0])
        shock_pct = float(out["shock_percentile"][0])
        regime = str(out["regime"][0])
        if not (math.isfinite(cqr_vol) and cqr_vol > 0.0):
            return None, None
        payload = {
            "enabled": True,
            "regime": regime,
            "har_vol5_pct": round(hat_vol5 * 100.0, 2),
            "rv20_vol5_pct": round(rv20_vol5 * 100.0, 2),
            "stress_ratio": round(stress_ratio, 3),
            "shock_score": round(p_shock, 4),
            "shock_percentile": round(shock_pct, 1),
            "vol_scale_source": "har_gjr",
        }
        return cqr_vol, payload
    except Exception as exc:
        logger.warning("Live HAR-GJR evaluation skipped (fail-open): %s", exc)
        return None, None
