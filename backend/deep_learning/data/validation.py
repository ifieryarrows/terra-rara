"""Validation helpers for the weekly TFT target contract."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


WEEKLY_TARGET_HELPER_COLS = {
    "target_close",
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
        for col in ("target", "target_1d_log_return"):
            valid_1d = df[col].dropna()
            max_abs_1d = float(valid_1d.abs().max()) if not valid_1d.empty else 0.0
            if max_abs_1d > 0.50:
                raise ValueError(
                    f"Corrupt single-day log return detected in training frame ({col}): max_abs={max_abs_1d:.4f} > 0.50"
                )

    if len(df) >= 2:
        next_realized = pd.to_numeric(df["target"].iloc[1:], errors="coerce").to_numpy(dtype=float)
        prev_fwd_1d = pd.to_numeric(df["target_1d_log_return"].iloc[:-1], errors="coerce").to_numpy(dtype=float)
        if mode == "train":
            finite_mask = np.isfinite(next_realized) & np.isfinite(prev_fwd_1d)
            if finite_mask.any() and not np.allclose(
                next_realized[finite_mask],
                prev_fwd_1d[finite_mask],
                atol=1e-10,
            ):
                raise ValueError("Causal alignment broken: target[t+1] does not match target_1d_log_return[t]")
        else:
            # In inference mode, trailing unknown bars have dummy target_1d_log_return=0.0;
            # verify causal alignment on all bars except dummy trailing placeholders.
            non_dummy = np.isfinite(next_realized) & np.isfinite(prev_fwd_1d) & (np.abs(prev_fwd_1d) > 1e-12)
            if len(next_realized) > 1:
                non_dummy[-1] = False
            if non_dummy.any() and not np.allclose(
                next_realized[non_dummy],
                prev_fwd_1d[non_dummy],
                atol=1e-10,
            ):
                raise ValueError("Causal alignment broken: target[t+1] does not match target_1d_log_return[t]")

    material_values = set(df["material_move_5d"].dropna().unique())
    if not material_values.issubset({0.0, 1.0}):
        raise ValueError("material_move_5d must be binary 0/1")

    if not df["time_idx"].is_monotonic_increasing:
        raise ValueError("time_idx must be monotonic increasing")
