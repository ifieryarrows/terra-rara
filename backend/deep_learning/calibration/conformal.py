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
