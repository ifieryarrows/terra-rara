"""Causal EWMA-0.94 + HAR-YZ-Rough forward volatility & dual-head market stress engine.

V2 (Benchmark-V5 champion):
- Combines 50% causal RiskMetrics EWMA-0.94 variance with 50% QLIKE-debiased
  HAR-YZ-SHAR-HARQ-Rough Ridge variance (`F1_rough`: Yang-Zhang/Garman-Klass/
  Rogers-Satchell range estimators, overnight gap intensity, SHAR downside/upside
  semivariance, HARQ quarticity, local Hurst roughness proxy, cross-asset stress).
- Automatically falls back gracefully when OHLC columns are absent.
- Uses strictly validation-only QLIKE-optimal mean debiasing (``E[y_va / v_va] = 1``)
  eliminating Jensen under-scaling bias on right-skewed 5D realized variance.
- Fits dual causal market-stress heads (relative regime shock + 5D volatility surge)
  and feeds unified ``cqr_vol_scale`` across train/val, test, backtest, and live paths.
- Maintains 100% pure-JSON serialization and backward compatibility with V1 artifacts.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOL5_FLOOR = 0.005
VAR5_FLOOR = VOL5_FLOOR ** 2
DEFAULT_SHOCK_WEIGHT = 0.55
DEFAULT_SHOCK_TAU = 0.22
DEFAULT_SHOCK_SIGMA_MULT = 1.75
DEFAULT_EWMA_BLEND = 0.50
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

V2_EXTRA_FEATURES: tuple[str, ...] = (
    "log_yz5",
    "log_yz10",
    "log_yz22",
    "log_yz60",
    "log_gk5",
    "log_gk22",
    "yz_accel_5_22",
    "yz_accel_5_60",
    "intraday_hl",
    "gap_var_5",
    "gap_ratio_5",
    "rsv_neg_5",
    "rsv_asym_5",
    "rsv_asym_20",
    "rq5",
    "rq20",
    "rough_hurst_proxy",
    "rough_2_22_logratio",
    "overnight_jump_ratio",
    "jump_max5",
    "ewma_fast_slow_ratio",
    "log_ewma_094",
    "tail_kurt_20d",
    "vol_surge",
)

V2_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + V2_EXTRA_FEATURES


def _first_numeric_series(
    df: pd.DataFrame,
    candidates: Sequence[str],
    default: float = 0.0,
) -> Optional[pd.Series]:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    return None


def _compute_ewma_daily_var(r1: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty(len(r1), dtype=np.float64)
    if len(r1) == 0:
        return out
    out[0] = 0.012 ** 2
    for i in range(1, len(r1)):
        out[i] = lam * out[i - 1] + (1.0 - lam) * (r1[i] ** 2)
    return np.maximum(out, 1e-10)


def _daily_ohlc_components(df: pd.DataFrame, r1_s: pd.Series, rv20_s: pd.Series) -> dict[str, Any]:
    """Extract strictly causal OHLC range/gap variance components (graceful close-only fallback)."""
    close_s = _first_numeric_series(df, ("target_close", "close"))
    open_s = _first_numeric_series(df, ("target_open", "open"))
    high_s = _first_numeric_series(df, ("target_high", "high"))
    low_s = _first_numeric_series(df, ("target_low", "low"))
    vol_s = _first_numeric_series(df, ("target_volume", "volume"))

    cc_var = (r1_s ** 2).clip(lower=1e-10)
    has_ohlc = (
        close_s is not None
        and open_s is not None
        and high_s is not None
        and low_s is not None
        and int(((high_s > low_s) & (close_s > 0.0) & (open_s > 0.0)).sum()) >= max(10, int(0.25 * len(df)))
    )
    if not has_ohlc:
        return {
            "has_ohlc": False,
            "daily_var_yz": cc_var,
            "rs_var": cc_var,
            "gk_var": cc_var,
            "log_oc": pd.Series(0.0, index=df.index, dtype=np.float64),
            "log_co": pd.Series(0.0, index=df.index, dtype=np.float64),
            "log_hl": pd.Series(0.0, index=df.index, dtype=np.float64),
            "volume": vol_s.fillna(0.0) if vol_s is not None else pd.Series(0.0, index=df.index, dtype=np.float64),
        }

    c = close_s.where(close_s > 0.0, np.nan).ffill()
    o = open_s.where(open_s > 0.0, c).fillna(c)
    h = np.maximum(high_s.where(high_s > 0.0, c).fillna(c), np.maximum(o, c))
    l = np.minimum(low_s.where(low_s > 0.0, c).fillna(c), np.minimum(o, c))
    prev_c = c.shift(1).fillna(o)

    valid_hl = (h > l) & (l > 0.0) & (o > 0.0) & (c > 0.0)
    log_oc = np.log(o / prev_c).where(valid_hl, 0.0).fillna(0.0)
    log_co = np.log(c / o).where(valid_hl, 0.0).fillna(0.0)
    log_ho = np.log(h / o).where(valid_hl, 0.0).fillna(0.0)
    log_lo = np.log(l / o).where(valid_hl, 0.0).fillna(0.0)
    log_hl = np.log(h / l).where(valid_hl, 0.0).fillna(0.0)
    log_hc = np.log(h / c).where(valid_hl, 0.0).fillna(0.0)
    log_lc = np.log(l / c).where(valid_hl, 0.0).fillna(0.0)

    rs_raw = (log_ho * log_hc + log_lo * log_lc).clip(lower=0.0)
    gk_raw = (0.5 * (log_hl ** 2) - (2.0 * math.log(2.0) - 1.0) * (log_co ** 2)).clip(lower=0.0)
    zero_range = (~valid_hl) | (rs_raw <= 1e-12)
    rs_clean = rs_raw.where(~zero_range, cc_var).fillna(1e-4)
    gk_clean = gk_raw.where(~zero_range, cc_var).fillna(1e-4)
    # Daily YZ-compatible forward variance contribution per bar
    daily_var_yz = ((log_oc ** 2) + 0.5 * (log_co ** 2) + rs_clean).clip(lower=1e-8)
    return {
        "has_ohlc": True,
        "daily_var_yz": daily_var_yz,
        "rs_var": rs_clean,
        "gk_var": gk_clean,
        "log_oc": log_oc,
        "log_co": log_co,
        "log_hl": log_hl,
        "volume": vol_s.fillna(0.0) if vol_s is not None else pd.Series(0.0, index=df.index, dtype=np.float64),
    }


def _rolling_yang_zhang_daily_vol(comp: dict[str, Any], window: int, fallback_rv: pd.Series) -> pd.Series:
    if not comp["has_ohlc"]:
        return fallback_rv
    w = max(int(window), 2)
    min_p = max(2, w // 2)
    k = 0.34 / (1.34 + (w + 1.0) / (w - 1.0))
    sigma_o2 = comp["log_oc"].rolling(w, min_periods=min_p).var().fillna(0.0)
    sigma_c2 = comp["log_co"].rolling(w, min_periods=min_p).var().fillna(0.0)
    sigma_rs2 = comp["rs_var"].rolling(w, min_periods=min_p).mean().fillna(1e-4)
    yz_var = (sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2).clip(lower=1e-8)
    return np.sqrt(yz_var).ffill().fillna(fallback_rv).clip(lower=1e-4)


def build_har_gjr_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build strictly causal F0 multi-scale volatility & leverage stress features (18 cols)."""
    mat_v2, _, rv20_arr, _ = build_v2_vol_stress_features(df)
    return mat_v2[:, : len(FEATURE_NAMES)], list(FEATURE_NAMES), rv20_arr


def build_v2_vol_stress_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, np.ndarray]]:
    """Build strictly causal V2 F1_rough volatility & stress features + EWMA/YZ anchors.

    Zero backward-fill (``.bfill()``) is used anywhere so mutating rows at index ``> t``
    can never alter feature rows or EWMA state at ``<= t``.
    """
    n = len(df)
    if n == 0:
        return (
            np.empty((0, len(V2_FEATURE_NAMES)), dtype=np.float64),
            list(V2_FEATURE_NAMES),
            np.empty(0, dtype=np.float64),
            {"ewma_094_var5": np.empty(0, dtype=np.float64), "yz22_vol5": np.empty(0, dtype=np.float64), "daily_var_yz": np.empty(0, dtype=np.float64)},
        )

    close_s = _first_numeric_series(df, ("target_close", "close"))
    r1_s = _first_numeric_series(df, ("observed_return_1d", "observed_ret_1d", "HG=F_ret1", "target"))
    if r1_s is None and close_s is not None:
        safe_close = close_s.where(close_s > 0.0, np.nan)
        r1_s = np.log(safe_close).diff()
    if r1_s is None:
        r1_s = pd.Series(0.0, index=df.index, dtype=np.float64)
    else:
        r1_s = r1_s.fillna(0.0)

    # Always derive causal daily RV20 from observed_return_1d if realized_vol_20d is absent
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

    # ---- V2 Causal F1_rough OHLC, SHAR, HARQ, Roughness & EWMA features ----
    comp = _daily_ohlc_components(df, r1_s, rv20_s)
    yz5 = _rolling_yang_zhang_daily_vol(comp, 5, rv5_s)
    yz10 = _rolling_yang_zhang_daily_vol(comp, 10, rv10_s)
    yz22 = _rolling_yang_zhang_daily_vol(comp, 22, rv20_s)
    yz60 = _rolling_yang_zhang_daily_vol(comp, 60, rv60_s)

    gk5 = np.sqrt(comp["gk_var"].rolling(5, min_periods=2).mean().ffill().fillna(rv5_s ** 2).clip(lower=1e-8))
    gk22 = np.sqrt(comp["gk_var"].rolling(22, min_periods=5).mean().ffill().fillna(rv20_s ** 2).clip(lower=1e-8))

    neg_sq = np.minimum(r1_s, 0.0) ** 2
    pos_sq = np.maximum(r1_s, 0.0) ** 2
    d5 = neg_sq.rolling(5, min_periods=2).mean().fillna(1e-5)
    u5 = pos_sq.rolling(5, min_periods=2).mean().fillna(1e-5)
    d20 = neg_sq.rolling(20, min_periods=5).mean().fillna(1e-5)
    u20 = pos_sq.rolling(20, min_periods=5).mean().fillna(1e-5)

    r4_series = comp["daily_var_yz"] ** 2
    rq5 = np.sqrt(r4_series.rolling(5, min_periods=2).mean().ffill().fillna(1e-8).clip(lower=1e-12)) * 100.0
    rq20 = np.sqrt(r4_series.rolling(20, min_periods=5).mean().ffill().fillna(1e-8).clip(lower=1e-12)) * 100.0

    gap_var_5 = (comp["log_oc"] ** 2).rolling(5, min_periods=2).mean().fillna(0.0)
    gap_ratio_5 = gap_var_5 / ((yz5 ** 2) + 1e-8)

    # Local log-range roughness slope regressing log(yz_w) on log(w) across {5, 10, 22, 60}
    log_yz5 = np.log(yz5.to_numpy(dtype=np.float64) + 1e-4)
    log_yz10 = np.log(yz10.to_numpy(dtype=np.float64) + 1e-4)
    log_yz22 = np.log(yz22.to_numpy(dtype=np.float64) + 1e-4)
    log_yz60 = np.log(yz60.to_numpy(dtype=np.float64) + 1e-4)
    ws = np.log(np.array([5.0, 10.0, 22.0, 60.0], dtype=np.float64))
    ws_c = ws - ws.mean()
    ws_denom = float(np.sum(ws_c ** 2))
    lyz_mat = np.column_stack([log_yz5, log_yz10, log_yz22, log_yz60])
    rough_hurst_proxy = np.sum((lyz_mat - lyz_mat.mean(axis=1, keepdims=True)) * ws_c[None, :], axis=1) / ws_denom

    rv2 = r1_s.rolling(2, min_periods=1).std().ffill().fillna(rv20_s).clip(lower=1e-4)
    rough_2_22_logratio = np.log(rv2.to_numpy(dtype=np.float64) + 1e-4) - log_yz22

    overnight_jump_ratio = np.sqrt(np.maximum(gap_var_5.to_numpy(dtype=np.float64), 0.0)) / (yz5.to_numpy(dtype=np.float64) + 1e-4)
    jump_max5 = abs_r1_z.rolling(5, min_periods=1).max().fillna(1.0).to_numpy(dtype=np.float64)

    r1_arr = r1_s.to_numpy(dtype=np.float64)
    ewma_086 = _compute_ewma_daily_var(r1_arr, 0.86)
    ewma_094 = _compute_ewma_daily_var(r1_arr, 0.94)
    ewma_097 = _compute_ewma_daily_var(r1_arr, 0.97)
    ewma_094_var5 = np.clip(5.0 * ewma_094, VAR5_FLOOR, 0.25)
    ewma_fast_slow_ratio = np.sqrt(ewma_086) / (np.sqrt(ewma_097) + 1e-8)
    log_ewma_094 = 0.5 * np.log(np.maximum(ewma_094, 1e-10))

    z4 = np.clip(abs_r1_z.to_numpy(dtype=np.float64), 0.0, 8.0) ** 4
    tail_kurt_20d = pd.Series(z4, index=df.index).rolling(20, min_periods=5).mean().ffill().fillna(3.0).to_numpy(dtype=np.float64)

    vol_s = comp["volume"]
    vol_ma20 = vol_s.rolling(20, min_periods=5).mean().replace(0.0, np.nan)
    vol_surge = np.log1p((vol_s / vol_ma20).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 10.0)).to_numpy(dtype=np.float64)

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
        # V2 rough + OHLC + EWMA
        "log_yz5": log_yz5,
        "log_yz10": log_yz10,
        "log_yz22": log_yz22,
        "log_yz60": log_yz60,
        "log_gk5": np.log(gk5.to_numpy(dtype=np.float64) + 1e-4),
        "log_gk22": np.log(gk22.to_numpy(dtype=np.float64) + 1e-4),
        "yz_accel_5_22": (yz5 / (yz22 + 1e-6)).to_numpy(dtype=np.float64),
        "yz_accel_5_60": (yz5 / (yz60 + 1e-6)).to_numpy(dtype=np.float64),
        "intraday_hl": comp["log_hl"].to_numpy(dtype=np.float64),
        "gap_var_5": gap_var_5.to_numpy(dtype=np.float64),
        "gap_ratio_5": gap_ratio_5.to_numpy(dtype=np.float64),
        "rsv_neg_5": np.log(np.sqrt(d5.to_numpy(dtype=np.float64)) + 1e-4),
        "rsv_asym_5": ((d5 - u5) / (d5 + u5 + 1e-8)).to_numpy(dtype=np.float64),
        "rsv_asym_20": ((d20 - u20) / (d20 + u20 + 1e-8)).to_numpy(dtype=np.float64),
        "rq5": rq5.to_numpy(dtype=np.float64),
        "rq20": rq20.to_numpy(dtype=np.float64),
        "rough_hurst_proxy": rough_hurst_proxy,
        "rough_2_22_logratio": rough_2_22_logratio,
        "overnight_jump_ratio": overnight_jump_ratio,
        "jump_max5": jump_max5,
        "ewma_fast_slow_ratio": ewma_fast_slow_ratio,
        "log_ewma_094": log_ewma_094,
        "tail_kurt_20d": tail_kurt_20d,
        "vol_surge": vol_surge,
    }
    mat = np.column_stack(
        [np.nan_to_num(col_map[name], nan=0.0, posinf=0.0, neginf=0.0) for name in V2_FEATURE_NAMES]
    )
    rv20_arr = np.nan_to_num(rv20_s.to_numpy(dtype=np.float64), nan=0.012, posinf=0.012, neginf=0.012)
    rv20_arr = np.maximum(rv20_arr, 1e-4)
    yz22_vol5 = np.maximum(np.nan_to_num(yz22.to_numpy(dtype=np.float64), nan=0.012) * math.sqrt(5.0), VOL5_FLOOR)
    anchors = {
        "ewma_094_var5": ewma_094_var5,
        "yz22_vol5": yz22_vol5,
        "daily_var_yz": np.nan_to_num(comp["daily_var_yz"].to_numpy(dtype=np.float64), nan=1e-4),
    }
    return mat, list(V2_FEATURE_NAMES), rv20_arr, anchors


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


def _extract_forward_5d_variance(df: pd.DataFrame, daily_var_yz: np.ndarray, horizon: int = 5) -> np.ndarray:
    """Resolve settled 5D forward realized variance (Yang-Zhang when OHLC exists, else squared return)."""
    if "fwd_var_5d_yz" in df.columns:
        return pd.to_numeric(df["fwd_var_5d_yz"], errors="coerce").to_numpy(dtype=np.float64)
    h = max(int(horizon), 1)
    dv = pd.Series(daily_var_yz, index=df.index, dtype=np.float64)
    acc = pd.Series(0.0, index=df.index, dtype=np.float64)
    for step in range(1, h + 1):
        acc = acc + dv.shift(-step)
    return acc.to_numpy(dtype=np.float64)


def fit_har_gjr_volatility_stress(
    master_df: pd.DataFrame,
    *,
    train_end_pos: int,
    val_origin_pos: Optional[np.ndarray | Sequence[int]] = None,
    horizon: int = 5,
    shock_weight: float = DEFAULT_SHOCK_WEIGHT,
    shock_tau: float = DEFAULT_SHOCK_TAU,
    ewma_blend: float = DEFAULT_EWMA_BLEND,
    shock_sigma_mult: float = DEFAULT_SHOCK_SIGMA_MULT,
    min_train_samples: int = 60,
    stress_elevated_threshold: float = STRESS_ELEVATED_RATIO,
    stress_high_threshold: float = STRESS_HIGH_RATIO,
) -> dict[str, Any]:
    """Fit JSON-serializable V2 EWMA-0.94 + HAR-YZ-Rough volatility & dual stress parameters.

    - Strictly fitted on settled training origins ``[0 .. train_end_pos - horizon]``.
    - Validation origins align HAR-YZ-Rough variance via QLIKE-optimal mean debiasing
      ``c* = mean(y_va / v_va)`` (zero Jensen collapse, zero test leakage).
    """
    n = len(master_df)
    if n == 0:
        return {"version": 2, "enabled": False, "reason": "empty_dataframe", "n_train": 0}

    h = max(int(horizon), 1)
    train_slice_end = max(0, min(int(train_end_pos) - h + 1, n))
    if train_slice_end < int(min_train_samples):
        return {
            "version": 2,
            "enabled": False,
            "reason": "insufficient_train_slice",
            "n_train": int(train_slice_end),
        }

    vmat, feature_names, rv20_arr, anchors = build_v2_vol_stress_features(master_df)
    weekly_y = _extract_forward_5d_returns(master_df, horizon=h)
    fwd_var_5d = _extract_forward_5d_variance(master_df, anchors["daily_var_yz"], horizon=h)

    y_tr_raw = weekly_y[:train_slice_end]
    v_tr_raw = fwd_var_5d[:train_slice_end]
    valid_tr = np.isfinite(y_tr_raw) & np.isfinite(rv20_arr[:train_slice_end])
    n_valid_tr = int(valid_tr.sum())
    if n_valid_tr < int(min_train_samples):
        return {
            "version": 2,
            "enabled": False,
            "reason": "insufficient_finite_targets",
            "n_train": n_valid_tr,
        }

    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler

    X_tr = vmat[:train_slice_end][valid_tr]
    y_5d_tr = y_tr_raw[valid_tr]
    var_tr_eff = np.where(np.isfinite(v_tr_raw[valid_tr]) & (v_tr_raw[valid_tr] > 1e-8), v_tr_raw[valid_tr], y_5d_tr ** 2)
    var_tr_eff = np.clip(var_tr_eff, 1e-6, 0.25)
    vol5_tr = np.maximum(rv20_arr[:train_slice_end][valid_tr] * math.sqrt(float(h)), VOL5_FLOOR)
    yz22_vol5_tr = anchors["yz22_vol5"][:train_slice_end][valid_tr]

    y_half_logvar_tr = 0.5 * np.log(var_tr_eff)
    y_shock_tr = (np.abs(y_5d_tr) >= float(shock_sigma_mult) * vol5_tr).astype(int)
    y_vol_surge_tr = (np.sqrt(var_tr_eff) >= 1.35 * yz22_vol5_tr).astype(int)

    scaler = StandardScaler().fit(X_tr)
    scale_safe = np.where(np.asarray(scaler.scale_, dtype=np.float64) > 1e-12, scaler.scale_, 1.0)
    mean_arr = np.asarray(scaler.mean_, dtype=np.float64)
    X_tr_s = np.clip((X_tr - mean_arr) / scale_safe, -6.0, 6.0)

    ridge = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(X_tr_s, y_half_logvar_tr)

    # Validation origins for QLIKE-optimal mean debiasing c* = mean(y_va / v_va)
    val_idx: Optional[np.ndarray] = None
    if val_origin_pos is not None:
        raw_idx = np.asarray(val_origin_pos, dtype=int).reshape(-1)
        raw_idx = raw_idx[(raw_idx >= 0) & (raw_idx < n)]
        if raw_idx.size > 0:
            val_idx = raw_idx

    if val_idx is not None and val_idx.size >= 5:
        X_ref = np.clip((vmat[val_idx] - mean_arr) / scale_safe, -6.0, 6.0)
        v_ref_raw = fwd_var_5d[val_idx]
        y_ref_raw = weekly_y[val_idx]
        fallback_ref_var = np.maximum(rv20_arr[val_idx] * math.sqrt(float(h)), VOL5_FLOOR) ** 2
        ref_var = np.where(
            np.isfinite(v_ref_raw) & (v_ref_raw > 1e-8),
            v_ref_raw,
            np.where(np.isfinite(y_ref_raw), np.clip(y_ref_raw ** 2, 1e-6, 0.25), fallback_ref_var),
        )
        n_val = int(val_idx.size)
    else:
        X_ref = X_tr_s
        ref_var = var_tr_eff
        n_val = 0

    raw_har_var = np.clip(
        np.exp(np.clip(2.0 * (X_ref @ np.asarray(ridge.coef_, dtype=np.float64) + float(ridge.intercept_)), -12.0, -1.0)),
        VAR5_FLOOR,
        0.25,
    )
    ok_ref = np.isfinite(ref_var) & (ref_var > 1e-8) & np.isfinite(raw_har_var) & (raw_har_var > 1e-8)
    var_scale_ratio = (
        float(np.clip(np.mean(np.clip(ref_var[ok_ref], 1e-6, 0.25) / raw_har_var[ok_ref]), 0.25, 4.0))
        if ok_ref.any()
        else 1.0
    )

    def _fit_lr(y_bin: np.ndarray) -> tuple[bool, np.ndarray, float, np.ndarray]:
        pos = int(y_bin.sum())
        neg = int(len(y_bin) - pos)
        if pos >= 2 and neg >= 5:
            clf = LogisticRegression(C=0.25, max_iter=500, random_state=42).fit(X_tr_s, y_bin)
            coef = np.asarray(clf.coef_[0], dtype=np.float64)
            inter = float(clf.intercept_[0])
            logits = np.clip(X_tr_s @ coef + inter, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            return True, coef, inter, probs
        return False, np.zeros(len(feature_names), dtype=np.float64), -10.0, np.zeros(n_valid_tr, dtype=np.float64)

    shock_enabled, shock_coef, shock_intercept, p_shock_tr = _fit_lr(y_shock_tr)
    vs_enabled, vs_coef, vs_intercept, p_vs_tr = _fit_lr(y_vol_surge_tr)

    if shock_enabled and vs_enabled:
        p_stress_tr = 0.5 * p_shock_tr + 0.5 * p_vs_tr
    elif shock_enabled:
        p_stress_tr = p_shock_tr
    else:
        p_stress_tr = p_vs_tr

    ref_quantiles = np.quantile(p_stress_tr, np.linspace(0.0, 1.0, 101)).astype(np.float64)
    shock_p75 = float(np.quantile(p_stress_tr, 0.75)) if p_stress_tr.size else 0.0
    shock_p90 = float(np.quantile(p_stress_tr, 0.90)) if p_stress_tr.size else 0.0

    ewma_tr = anchors["ewma_094_var5"][:train_slice_end][valid_tr]
    har_var_tr = np.clip(
        np.exp(np.clip(2.0 * (X_tr_s @ np.asarray(ridge.coef_, dtype=np.float64) + float(ridge.intercept_)), -12.0, -1.0))
        * var_scale_ratio,
        VAR5_FLOOR,
        0.25,
    )
    blend_var_tr = np.clip(float(ewma_blend) * ewma_tr + (1.0 - float(ewma_blend)) * har_var_tr, VAR5_FLOOR, 0.25)
    hat_vol_5d_tr = np.maximum(np.sqrt(blend_var_tr), VOL5_FLOOR)
    vol5_ref_quantiles = np.quantile(hat_vol_5d_tr, np.linspace(0.0, 1.0, 101)).astype(np.float64)
    vol5_p75 = float(np.quantile(hat_vol_5d_tr, 0.75)) if hat_vol_5d_tr.size else 0.0
    vol5_p90 = float(np.quantile(hat_vol_5d_tr, 0.90)) if hat_vol_5d_tr.size else 0.0

    return {
        "version": 2,
        "enabled": True,
        "model_family": "ewma_har_yz_rough_v2_json",
        "horizon": int(h),
        "ewma_blend": float(ewma_blend),
        "ewma_lambda": 0.94,
        "feature_names": feature_names,
        "scaler_mean": [float(x) for x in mean_arr.tolist()],
        "scaler_scale": [float(x) for x in scale_safe.tolist()],
        "vol_coef": [float(x) for x in np.asarray(ridge.coef_, dtype=np.float64).tolist()],
        "vol_intercept": float(ridge.intercept_),
        "var_scale_ratio": float(var_scale_ratio),
        # Keep vol_scale_ratio for legacy inspection symmetry
        "vol_scale_ratio": float(math.sqrt(var_scale_ratio)),
        "shock_enabled": bool(shock_enabled),
        "shock_coef": [float(x) for x in shock_coef.tolist()],
        "shock_intercept": float(shock_intercept),
        "vol_surge_enabled": bool(vs_enabled),
        "vol_surge_coef": [float(x) for x in vs_coef.tolist()],
        "vol_surge_intercept": float(vs_intercept),
        "shock_weight": float(shock_weight),
        "shock_tau": float(shock_tau),
        "shock_sigma_mult": float(shock_sigma_mult),
        "shock_p75": float(shock_p75),
        "shock_p90": float(shock_p90),
        "shock_ref_quantiles": [float(x) for x in ref_quantiles.tolist()],
        "vol5_p75": float(vol5_p75),
        "vol5_p90": float(vol5_p90),
        "vol5_ref_quantiles": [float(x) for x in vol5_ref_quantiles.tolist()],
        "stress_elevated_threshold": float(stress_elevated_threshold),
        "stress_high_threshold": float(stress_high_threshold),
        "shock_elevated_pct": float(SHOCK_ELEVATED_PCT),
        "shock_high_pct": float(SHOCK_HIGH_PCT),
        "n_train": int(n_valid_tr),
        "n_val": int(n_val),
        "train_shock_rate": float(y_shock_tr.mean()) if len(y_shock_tr) else 0.0,
        "train_vol_surge_rate": float(y_vol_surge_tr.mean()) if len(y_vol_surge_tr) else 0.0,
    }


def is_valid_har_gjr_params(params: Optional[Mapping[str, Any]]) -> bool:
    """Return True when ``params`` contains finite, enabled JSON V1 or V2 volatility-stress weights."""
    if not isinstance(params, Mapping) or not bool(params.get("enabled", False)):
        return False
    required = ("feature_names", "scaler_mean", "scaler_scale", "vol_coef", "vol_intercept")
    if not all(k in params for k in required):
        return False
    if "vol_scale_ratio" not in params and "var_scale_ratio" not in params:
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
        if not np.isfinite(float(params["vol_intercept"])):
            return False
        scale_val = float(params.get("var_scale_ratio", params.get("vol_scale_ratio", 0.0)))
        if not np.isfinite(scale_val) or scale_val <= 0.0:
            return False
        if bool(params.get("shock_enabled", False)):
            sc = np.asarray(params.get("shock_coef", []), dtype=np.float64)
            if sc.shape != (d,) or not np.isfinite(sc).all():
                return False
            if not np.isfinite(float(params.get("shock_intercept", 0.0))):
                return False
        if bool(params.get("vol_surge_enabled", False)):
            vc = np.asarray(params.get("vol_surge_coef", []), dtype=np.float64)
            if vc.shape != (d,) or not np.isfinite(vc).all():
                return False
            if not np.isfinite(float(params.get("vol_surge_intercept", 0.0))):
                return False
        return True
    except Exception:
        return False


def _classify_stress_regime(
    stress_ratio: float,
    shock_percentile: float,
    *,
    hat_vol_5d: Optional[float] = None,
    vol5_p75: Optional[float] = None,
    vol5_p90: Optional[float] = None,
    elevated_ratio: float = STRESS_ELEVATED_RATIO,
    high_ratio: float = STRESS_HIGH_RATIO,
    elevated_pct: float = SHOCK_ELEVATED_PCT,
    high_pct: float = SHOCK_HIGH_PCT,
) -> str:
    if (
        stress_ratio >= high_ratio
        or shock_percentile >= high_pct
        or (hat_vol_5d is not None and vol5_p90 is not None and vol5_p90 > 0.0 and hat_vol_5d >= vol5_p90)
    ):
        return "HIGH_STRESS"
    if (
        stress_ratio >= elevated_ratio
        or shock_percentile >= elevated_pct
        or (hat_vol_5d is not None and vol5_p75 is not None and vol5_p75 > 0.0 and hat_vol_5d >= vol5_p75)
    ):
        return "ELEVATED"
    return "CALM"


def predict_har_gjr_volatility_stress(
    df: pd.DataFrame,
    har_gjr_params: Mapping[str, Any],
    *,
    positions: Optional[np.ndarray | Sequence[int]] = None,
) -> dict[str, np.ndarray]:
    """Run causal pure-NumPy V1 or V2 volatility & stress inference on ``df``."""
    if not is_valid_har_gjr_params(har_gjr_params):
        raise ValueError("Invalid or disabled volatility stress parameters")

    version = int(har_gjr_params.get("version", 1) or 1)
    vmat, computed_names, rv20_arr, anchors = build_v2_vol_stress_features(df)
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

    ewma_var5 = anchors["ewma_094_var5"]
    if positions is not None:
        pos = np.asarray(positions, dtype=int).reshape(-1)
        pos = np.clip(pos, 0, max(len(df) - 1, 0))
        X = X[pos]
        rv20_sub = rv20_arr[pos]
        ewma_sub = ewma_var5[pos]
    else:
        rv20_sub = rv20_arr
        ewma_sub = ewma_var5

    horizon = max(int(har_gjr_params.get("horizon", 5) or 5), 1)
    sqrt_h = math.sqrt(float(horizon))
    mean_arr = np.asarray(har_gjr_params["scaler_mean"], dtype=np.float64)
    scale_arr = np.maximum(np.asarray(har_gjr_params["scaler_scale"], dtype=np.float64), 1e-12)
    vol_coef = np.asarray(har_gjr_params["vol_coef"], dtype=np.float64)
    vol_intercept = float(har_gjr_params["vol_intercept"])
    X_s = np.clip((X - mean_arr) / scale_arr, -6.0, 6.0)

    if version >= 2:
        var_scale_ratio = float(har_gjr_params.get("var_scale_ratio", 1.0) or 1.0)
        har_var_5d = np.clip(
            np.exp(np.clip(2.0 * (X_s @ vol_coef + vol_intercept), -12.0, -1.0)) * var_scale_ratio,
            VAR5_FLOOR,
            0.25,
        )
        ewma_w = min(max(float(har_gjr_params.get("ewma_blend", DEFAULT_EWMA_BLEND)), 0.0), 1.0)
        blend_var_5d = np.clip(ewma_w * ewma_sub + (1.0 - ewma_w) * har_var_5d, VAR5_FLOOR, 0.25)
        hat_vol_5d = np.maximum(np.sqrt(blend_var_5d), VOL5_FLOOR)
    else:
        # Legacy V1 exact math
        vol_scale_ratio = float(har_gjr_params["vol_scale_ratio"])
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

    if bool(har_gjr_params.get("vol_surge_enabled", False)):
        vs_coef = np.asarray(har_gjr_params["vol_surge_coef"], dtype=np.float64)
        vs_intercept = float(har_gjr_params["vol_surge_intercept"])
        vs_logits = np.clip(X_s @ vs_coef + vs_intercept, -30.0, 30.0)
        p_vol_surge = 1.0 / (1.0 + np.exp(-vs_logits))
    else:
        p_vol_surge = p_shock.copy()

    if version >= 2 and bool(har_gjr_params.get("vol_surge_enabled", False)) and bool(har_gjr_params.get("shock_enabled", False)):
        p_stress = 0.5 * p_shock + 0.5 * p_vol_surge
    else:
        p_stress = p_shock

    shock_weight = float(har_gjr_params.get("shock_weight", DEFAULT_SHOCK_WEIGHT if version >= 2 else 0.75) or 0.0)
    if version >= 2:
        shock_tau = float(har_gjr_params.get("shock_tau", DEFAULT_SHOCK_TAU) or DEFAULT_SHOCK_TAU)
        excess = np.maximum(p_stress - shock_tau, 0.0) / max(1.0 - shock_tau, 1e-4)
        cqr_vol_scale = np.maximum((hat_vol_5d / sqrt_h) * (1.0 + shock_weight * excess), 1e-4)
    else:
        cqr_vol_scale = np.maximum((hat_vol_5d / sqrt_h) * (1.0 + shock_weight * p_shock), 1e-4)

    stress_ratio = hat_vol_5d / np.maximum(rv20_vol5, VOL5_FLOOR)

    ref_q = np.asarray(har_gjr_params.get("shock_ref_quantiles") or [], dtype=np.float64)
    if ref_q.size >= 2 and np.isfinite(ref_q).all():
        ranks = np.searchsorted(ref_q, p_stress, side="right")
        shock_percentile = np.clip((ranks.astype(np.float64) / float(len(ref_q))) * 100.0, 0.0, 100.0)
    else:
        shock_p75 = float(har_gjr_params.get("shock_p75", 0.02) or 0.02)
        shock_p90 = float(har_gjr_params.get("shock_p90", 0.04) or 0.04)
        shock_percentile = np.where(
            p_stress >= shock_p90,
            95.0,
            np.where(p_stress >= shock_p75, 80.0, 40.0),
        )

    elevated_ratio = float(har_gjr_params.get("stress_elevated_threshold", STRESS_ELEVATED_RATIO) or STRESS_ELEVATED_RATIO)
    high_ratio = float(har_gjr_params.get("stress_high_threshold", STRESS_HIGH_RATIO) or STRESS_HIGH_RATIO)
    elevated_pct = float(har_gjr_params.get("shock_elevated_pct", SHOCK_ELEVATED_PCT) or SHOCK_ELEVATED_PCT)
    high_pct = float(har_gjr_params.get("shock_high_pct", SHOCK_HIGH_PCT) or SHOCK_HIGH_PCT)

    raw_vol5_p75 = har_gjr_params.get("vol5_p75")
    raw_vol5_p90 = har_gjr_params.get("vol5_p90")
    vol5_p75 = float(raw_vol5_p75) if raw_vol5_p75 is not None and float(raw_vol5_p75) > 0.0 else None
    vol5_p90 = float(raw_vol5_p90) if raw_vol5_p90 is not None and float(raw_vol5_p90) > 0.0 else None

    vol_ref_q = np.asarray(har_gjr_params.get("vol5_ref_quantiles") or [], dtype=np.float64)
    if vol_ref_q.size >= 2 and np.isfinite(vol_ref_q).all():
        vol_ranks = np.searchsorted(vol_ref_q, hat_vol_5d, side="right")
        vol_percentile = np.clip((vol_ranks.astype(np.float64) / float(len(vol_ref_q))) * 100.0, 0.0, 100.0)
    elif vol5_p75 is not None and vol5_p90 is not None:
        vol_percentile = np.where(
            hat_vol_5d >= vol5_p90,
            92.0,
            np.where(hat_vol_5d >= vol5_p75, 80.0, 40.0),
        )
    else:
        vol_percentile = np.full(len(hat_vol_5d), np.nan, dtype=np.float64)

    regimes = np.asarray(
        [
            _classify_stress_regime(
                float(sr),
                float(sp),
                hat_vol_5d=float(hv),
                vol5_p75=vol5_p75,
                vol5_p90=vol5_p90,
                elevated_ratio=elevated_ratio,
                high_ratio=high_ratio,
                elevated_pct=elevated_pct,
                high_pct=high_pct,
            )
            for sr, sp, hv in zip(stress_ratio, shock_percentile, hat_vol_5d)
        ],
        dtype=object,
    )
    risk_budget_mult = np.where(regimes == "HIGH_STRESS", 0.70, np.where(regimes == "ELEVATED", 0.85, 1.00))

    return {
        "hat_vol_5d": hat_vol_5d,
        "rv20_vol5": rv20_vol5,
        "p_shock": p_shock,
        "p_vol_surge": p_vol_surge,
        "p_stress": p_stress,
        "shock_percentile": shock_percentile,
        "vol_percentile": vol_percentile,
        "stress_ratio": stress_ratio,
        "cqr_vol_scale": cqr_vol_scale,
        "regime": regimes,
        "risk_budget_mult": risk_budget_mult,
    }


def resolve_cqr_vol_scale(
    df: Optional[pd.DataFrame],
    positions: Optional[np.ndarray | Sequence[int]],
    *,
    har_gjr_params: Optional[Mapping[str, Any]],
    fallback_vol: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Return causal volatility-stress ``cqr_vol_scale`` array or fail-open ``fallback_vol``."""
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
    """Evaluate volatility & stress engine at the latest forecast origin (fail-open)."""
    if df is None or df.empty or not is_valid_har_gjr_params(har_gjr_params):
        return None, None
    try:
        out = predict_har_gjr_volatility_stress(df, har_gjr_params, positions=[len(df) - 1])
        cqr_vol = float(out["cqr_vol_scale"][0])
        hat_vol5 = float(out["hat_vol_5d"][0])
        rv20_vol5 = float(out["rv20_vol5"][0])
        stress_ratio = float(out["stress_ratio"][0])
        p_shock = float(out["p_shock"][0])
        p_vs = float(out["p_vol_surge"][0])
        p_stress = float(out["p_stress"][0])
        shock_pct = float(out["shock_percentile"][0])
        vol_pct_raw = float(out["vol_percentile"][0]) if "vol_percentile" in out else float("nan")
        vol_pct = round(vol_pct_raw, 1) if math.isfinite(vol_pct_raw) else None
        regime = str(out["regime"][0])
        budget_mult = float(out["risk_budget_mult"][0])
        version = int(har_gjr_params.get("version", 1) or 1)
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
            "vol_percentile": vol_pct,
            "vol_surge_prob": round(p_vs, 4),
            "tail_risk_score": round(p_stress, 4),
            "risk_budget_mult": round(budget_mult, 2),
            "engine_version": "v2_ewma_har_rough" if version >= 2 else "v1_har_gjr",
            "vol_scale_source": "har_gjr",
        }
        return cqr_vol, payload
    except Exception as exc:
        logger.warning("Live volatility-stress evaluation skipped (fail-open): %s", exc)
        return None, None
