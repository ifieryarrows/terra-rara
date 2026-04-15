"""
Time Series Data Augmentation for TFT-ASRO.

Applies conservative augmentation techniques to increase effective training
set size without introducing unrealistic patterns.

Techniques:
    - Jittering: Add small Gaussian noise to feature values
    - Magnitude Warping: Scale features by small random factors
    - Window Slicing: Create shifted sub-windows from the training data

Reference: Um et al. (2017) "Data Augmentation of Wearable Sensor Data" (ICMI)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def jitter(
    df: pd.DataFrame,
    feature_cols: list[str],
    sigma: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add Gaussian noise to feature columns.

    The noise magnitude is relative to each feature's standard deviation
    to maintain scale consistency across features with different ranges.
    """
    rng = np.random.RandomState(seed)
    augmented = df.copy()

    for col in feature_cols:
        col_std = augmented[col].std()
        if col_std < 1e-12:
            continue
        noise = rng.normal(0, sigma * col_std, size=len(augmented))
        augmented[col] = augmented[col] + noise

    return augmented


def magnitude_warp(
    df: pd.DataFrame,
    feature_cols: list[str],
    sigma: float = 0.02,
    seed: int = 43,
) -> pd.DataFrame:
    """
    Multiply feature values by smooth random factors centered at 1.0.

    Uses cubic spline interpolation over a few knots to create slowly-varying
    scale factors, preserving local structure.
    """
    from scipy.interpolate import CubicSpline

    rng = np.random.RandomState(seed)
    augmented = df.copy()
    n = len(augmented)
    n_knots = 4
    knot_positions = np.linspace(0, n - 1, n_knots)
    x = np.arange(n)

    for col in feature_cols:
        knot_values = rng.normal(1.0, sigma, size=n_knots)
        cs = CubicSpline(knot_positions, knot_values)
        warp_factor = cs(x)
        augmented[col] = augmented[col] * warp_factor

    return augmented


def augment_training_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    augment_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Augment training DataFrame with jittered and warped copies.

    Appends augmented rows to the original, preserving time_idx ordering
    by offsetting augmented indices past the original range.

    Args:
        df:             Training DataFrame (must have time_idx and group_id).
        feature_cols:   Feature columns to augment (target is preserved exact).
        augment_ratio:  Fraction of original data to add (0.15 = 15%).
        seed:           Random seed.

    Returns:
        Augmented DataFrame with updated time_idx for new rows.
    """
    n_original = len(df)
    n_augment = int(n_original * augment_ratio)
    if n_augment < 10:
        logger.info("Augmentation: ratio=%.2f yields <10 rows, skipping", augment_ratio)
        return df

    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(n_original, size=n_augment, replace=False)
    sample = df.iloc[sample_idx].copy()

    aug_features = [c for c in feature_cols if c != target_col]

    aug_jitter = jitter(sample, aug_features, sigma=0.005, seed=seed)
    aug_warped = magnitude_warp(aug_jitter, aug_features, sigma=0.02, seed=seed + 1)

    max_time_idx = df["time_idx"].max()
    aug_warped["time_idx"] = np.arange(max_time_idx + 1, max_time_idx + 1 + n_augment)
    aug_warped["group_id"] = "copper_aug"

    combined = pd.concat([df, aug_warped], ignore_index=True)
    combined = combined.sort_values("time_idx").reset_index(drop=True)

    logger.info(
        "Augmentation: added %d rows (%.0f%%) → total %d rows",
        n_augment, augment_ratio * 100, len(combined),
    )
    return combined
