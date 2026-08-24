"""Leakage-safe auxiliary weekly direction calibration.

The TFT quantile path remains responsible for return magnitude and intervals.
This module provides a small, deterministic direction model fitted only on
chronological training origins.  It is enabled only when its untouched
validation predictions improve on the TFT direction with a balanced sign rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _finite_feature_frame(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Direction model features missing from frame: {missing}")
    values = frame[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def fit_weekly_direction_model(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    train_cutoff: int,
    horizon: int = 5,
    max_encoder_length: int = 50,
    target_col: str = "target_5d_log_return",
) -> dict[str, Any]:
    """Fit a balanced logistic direction model on training origins only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    times = frame["time_idx"].to_numpy(dtype=np.int64)
    target = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=np.float64)
    origin_mask = (
        (times >= int(max_encoder_length) - 1)
        & (times <= int(train_cutoff) - int(horizon))
        & np.isfinite(target)
    )
    if int(origin_mask.sum()) < 60:
        return {
            "version": 1,
            "enabled": False,
            "reason": "insufficient_training_origins",
            "fit_split": "train",
            "feature_names": list(feature_names),
            "horizon": int(horizon),
        }

    x = _finite_feature_frame(frame.loc[origin_mask], feature_names)
    y = (target[origin_mask] > 0.0).astype(np.int64)
    if np.unique(y).size < 2:
        return {
            "version": 1,
            "enabled": False,
            "reason": "single_training_sign_class",
            "fit_split": "train",
            "feature_names": list(feature_names),
            "horizon": int(horizon),
        }

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        C=0.10,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(x_scaled, y)
    return {
        "version": 1,
        "enabled": False,
        "reason": "awaiting_validation_selection",
        "fit_split": "train",
        "feature_names": list(feature_names),
        "horizon": int(horizon),
        "decision_threshold": 0.50,
        "train_origin_count": int(len(y)),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }


def predict_weekly_direction(
    calibrator: dict[str, Any],
    frame: pd.DataFrame,
    *,
    start_exclusive: int,
    end_inclusive: int,
) -> np.ndarray:
    """Predict validation/test-origin positive probabilities in time order."""
    times = frame["time_idx"].to_numpy(dtype=np.int64)
    mask = (times > int(start_exclusive)) & (times <= int(end_inclusive))
    selected = frame.loc[mask].sort_values("time_idx")
    if selected.empty:
        return np.empty(0, dtype=np.float64)
    x = _finite_feature_frame(selected, list(calibrator["feature_names"]))
    mean = np.asarray(calibrator["mean"], dtype=np.float64)
    scale = np.asarray(calibrator["scale"], dtype=np.float64)
    coef = np.asarray(calibrator["coef"], dtype=np.float64)
    scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
    logits = ((x - mean) / scale) @ coef + float(calibrator["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def apply_weekly_direction_model(
    pred: np.ndarray,
    positive_probability: np.ndarray,
    *,
    threshold: float = 0.50,
    horizon: int = 5,
) -> np.ndarray:
    """Align each TFT weekly median sign to the causal direction model.

    The adjustment shifts every quantile by the median-path delta, preserving
    interval widths and ordering while changing only the forecast location.
    """
    arr = np.asarray(pred, dtype=np.float64).copy()
    probs = np.asarray(positive_probability, dtype=np.float64).reshape(-1)
    if arr.ndim != 3 or arr.shape[1] < horizon:
        raise ValueError(f"Expected predictions [n,>={horizon},q], got {arr.shape}")
    if len(probs) != len(arr):
        raise ValueError(f"Direction probabilities {len(probs)} do not match predictions {len(arr)}")
    median_idx = arr.shape[2] // 2
    median_path = arr[:, :horizon, median_idx]
    weekly_median = median_path.sum(axis=1)
    current_sign = np.where(weekly_median >= 0.0, 1.0, -1.0)
    desired_sign = np.where(probs >= float(threshold), 1.0, -1.0)
    new_median_path = median_path * (desired_sign * current_sign)[:, None]
    arr[:, :horizon, :] += (new_median_path - median_path)[:, :, None]
    return arr
