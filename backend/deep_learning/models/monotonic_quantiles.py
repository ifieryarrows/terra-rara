from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_MONOTONIC_GAP_SCALE = 0.02


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
