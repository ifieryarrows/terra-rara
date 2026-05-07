"""Validation helpers for the weekly TFT target contract."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


WEEKLY_TARGET_HELPER_COLS = {
    "target_1d_log_return",
    "target_5d_log_return",
    "realized_vol_20d",
    "material_move_5d",
}


def validate_weekly_target_contract(
    df: pd.DataFrame,
    *,
    mode: Literal["train", "inference"] = "train",
) -> None:
    """Validate weekly target/helper columns without treating helpers as inputs."""
    required = [
        "target",
        "target_1d_log_return",
        "target_5d_log_return",
        "realized_vol_20d",
        "material_move_5d",
        "time_idx",
        "group_id",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing weekly target columns: {missing}")

    if mode == "train":
        for col in ("target", "target_1d_log_return", "target_5d_log_return"):
            if df[col].isna().any():
                raise ValueError(f"{col} contains NaN after feature-store construction")

    comparable = df[["target", "target_1d_log_return"]].dropna()
    if not comparable.empty and not np.allclose(
        comparable["target"].to_numpy(),
        comparable["target_1d_log_return"].to_numpy(),
        atol=1e-10,
        equal_nan=True,
    ):
        raise ValueError("target and target_1d_log_return are not identical")

    material_values = set(df["material_move_5d"].dropna().unique())
    if not material_values.issubset({0.0, 1.0}):
        raise ValueError("material_move_5d must be binary 0/1")

    if not df["time_idx"].is_monotonic_increasing:
        raise ValueError("time_idx must be monotonic increasing")
