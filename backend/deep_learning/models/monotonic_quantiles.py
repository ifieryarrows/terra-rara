from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_MONOTONIC_GAP_SCALE = 0.03


def enforce_monotonic_quantiles(
    y_pred: torch.Tensor,
    median_idx: int = 3,
    min_gap: float = 1e-5,
    gap_scale: float = DEFAULT_MONOTONIC_GAP_SCALE,
    init_bias: float = -3.0,
) -> torch.Tensor:
    """
    Transform unconstrained quantile outputs into structurally monotonic
    quantile outputs.

    The median dimension is preserved exactly. Lower/upper quantile distances
    are positive by construction and scaled for log-return targets.
    """
    base = y_pred[..., median_idx]

    lower_raw = y_pred[..., :median_idx]
    upper_raw = y_pred[..., median_idx + 1 :]

    lower_steps = min_gap + gap_scale * F.softplus(
        torch.flip(lower_raw, dims=[-1]) + init_bias
    )
    upper_steps = min_gap + gap_scale * F.softplus(upper_raw + init_bias)

    lower_from_median = torch.cumsum(lower_steps, dim=-1)
    upper_from_median = torch.cumsum(upper_steps, dim=-1)

    lower = base.unsqueeze(-1) - lower_from_median
    lower = torch.flip(lower, dims=[-1])
    upper = base.unsqueeze(-1) + upper_from_median

    ordered = torch.cat([lower, base.unsqueeze(-1), upper], dim=-1)

    assert ordered.shape == y_pred.shape, (
        f"Monotonic transform output shape {ordered.shape} "
        f"does not match input shape {y_pred.shape}"
    )
    return ordered


def aggregate_weekly_quantiles_torch(
    ordered_pred: torch.Tensor,
    median_idx: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Aggregate ordered daily log-return quantiles ``[..., H, Q]`` into
    a valid weekly distribution ``[..., Q]``.

    The median sums linearly across the horizon (additive log returns), while
    quantile half-widths combine by root-sum-of-squares (L2 across time) so the
    interval width scales as ``sqrt(H)`` rather than assuming perfect daily
    comonotonicity (``H``). Non-crossing monotonicity is preserved strictly
    whenever ``ordered_pred`` is non-decreasing across the last axis.
    """
    if ordered_pred.ndim < 2:
        raise ValueError(
            f"Expected quantile tensor [..., horizon, q], got {tuple(ordered_pred.shape)}"
        )
    q_dim = int(ordered_pred.shape[-1])
    if median_idx is None:
        median_idx = q_dim // 2

    median_daily = ordered_pred[..., median_idx]
    weekly_median = median_daily.sum(dim=-1)
    med_exp = median_daily.unsqueeze(-1)

    parts: list[torch.Tensor] = []
    if median_idx > 0:
        lower_offsets = torch.clamp(med_exp - ordered_pred[..., :median_idx], min=0.0)
        weekly_lower_dist = torch.sqrt(lower_offsets.pow(2).sum(dim=-2) + eps)
        parts.append(weekly_median.unsqueeze(-1) - weekly_lower_dist)

    parts.append(weekly_median.unsqueeze(-1))

    if median_idx + 1 < q_dim:
        upper_offsets = torch.clamp(ordered_pred[..., median_idx + 1 :] - med_exp, min=0.0)
        weekly_upper_dist = torch.sqrt(upper_offsets.pow(2).sum(dim=-2) + eps)
        parts.append(weekly_median.unsqueeze(-1) + weekly_upper_dist)

    return torch.cat(parts, dim=-1)


def aggregate_weekly_quantiles_np(
    ordered_pred: np.ndarray,
    median_idx: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NumPy counterpart of :func:`aggregate_weekly_quantiles_torch`."""
    arr = np.asarray(ordered_pred, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"Expected quantile array [..., horizon, q], got {arr.shape}")
    q_dim = arr.shape[-1]
    if median_idx is None:
        median_idx = q_dim // 2

    median_daily = arr[..., median_idx]
    weekly_median = median_daily.sum(axis=-1)
    med_exp = median_daily[..., None]

    parts: list[np.ndarray] = []
    if median_idx > 0:
        lower_offsets = np.maximum(med_exp - arr[..., :median_idx], 0.0)
        weekly_lower_dist = np.sqrt(np.square(lower_offsets).sum(axis=-2) + eps)
        parts.append(weekly_median[..., None] - weekly_lower_dist)

    parts.append(weekly_median[..., None])

    if median_idx + 1 < q_dim:
        upper_offsets = np.maximum(arr[..., median_idx + 1 :] - med_exp, 0.0)
        weekly_upper_dist = np.sqrt(np.square(upper_offsets).sum(axis=-2) + eps)
        parts.append(weekly_median[..., None] + weekly_upper_dist)

    return np.concatenate(parts, axis=-1)


def aggregate_cumulative_quantiles_np(
    ordered_pred: np.ndarray,
    median_idx: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute cumulative horizon quantiles ``[H, Q]`` with RSS spread growth."""
    arr = np.asarray(ordered_pred, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [horizon, q], got {arr.shape}")
    q_dim = arr.shape[1]
    if median_idx is None:
        median_idx = q_dim // 2

    median_daily = arr[:, median_idx]
    cum_median = np.cumsum(median_daily, axis=0)
    med_exp = median_daily[:, None]

    parts: list[np.ndarray] = []
    if median_idx > 0:
        lower_offsets = np.maximum(med_exp - arr[:, :median_idx], 0.0)
        cum_lower_dist = np.sqrt(np.cumsum(np.square(lower_offsets), axis=0) + eps)
        parts.append(cum_median[:, None] - cum_lower_dist)

    parts.append(cum_median[:, None])

    if median_idx + 1 < q_dim:
        upper_offsets = np.maximum(arr[:, median_idx + 1 :] - med_exp, 0.0)
        cum_upper_dist = np.sqrt(np.cumsum(np.square(upper_offsets), axis=0) + eps)
        parts.append(cum_median[:, None] + cum_upper_dist)

    return np.concatenate(parts, axis=1)


def validate_monotonicity(
    y_pred: torch.Tensor,
    tolerance: float = 1e-6,
) -> dict:
    """Return crossing diagnostics for an ordered quantile tensor."""
    diffs = y_pred[..., 1:] - y_pred[..., :-1]
    violations = diffs < -tolerance
    crossing_rate = violations.float().mean().item()
    max_violation = (
        (-diffs[violations]).max().item() if violations.any().item() else 0.0
    )

    return {
        "crossing_rate": crossing_rate,
        "max_violation": max_violation,
        "is_valid": crossing_rate == 0.0,
    }
