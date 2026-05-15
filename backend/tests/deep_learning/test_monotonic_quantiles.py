import torch

from deep_learning.models.monotonic_quantiles import (
    DEFAULT_MONOTONIC_GAP_SCALE,
    enforce_monotonic_quantiles,
    validate_monotonicity,
)


def test_enforce_monotonic_quantiles_orders_random_path_and_preserves_median():
    torch.manual_seed(20260514)
    raw = torch.randn(64, 5, 7)

    ordered = enforce_monotonic_quantiles(raw)
    diagnostics = validate_monotonicity(ordered)

    assert ordered.shape == raw.shape
    assert diagnostics["crossing_rate"] == 0.0
    assert diagnostics["max_violation"] == 0.0
    assert diagnostics["is_valid"] is True
    assert torch.allclose(ordered[..., 3], raw[..., 3])


def test_default_monotonic_gap_scale_matches_deterministic_recovery_config():
    assert DEFAULT_MONOTONIC_GAP_SCALE == 0.03
