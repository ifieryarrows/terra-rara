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


def test_initialize_tft_output_layer_calibrates_initial_median_and_pi80_width():
    from types import SimpleNamespace

    from deep_learning.models.monotonic_quantiles import aggregate_weekly_quantiles_torch
    from deep_learning.models.tft_copper import _initialize_tft_output_layer

    quantiles = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)
    layer = torch.nn.Linear(24, len(quantiles))
    stub_model = SimpleNamespace(output_layer=layer)
    _initialize_tft_output_layer(stub_model, quantiles)

    torch.manual_seed(0)
    hidden = torch.randn(32, 5, 24)
    raw = layer(hidden)
    ordered = enforce_monotonic_quantiles(raw, median_idx=3)
    weekly = aggregate_weekly_quantiles_torch(ordered, median_idx=3)
    weekly_median_abs_mean = weekly[:, 3].abs().mean().item()
    weekly_pi80_width = (weekly[:, 5] - weekly[:, 1]).mean().item()

    assert weekly_median_abs_mean < 0.03
    assert 0.06 <= weekly_pi80_width <= 0.095
