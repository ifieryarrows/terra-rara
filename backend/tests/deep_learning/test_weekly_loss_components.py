import torch

from deep_learning.models.tft_copper import (
    _bound_weekly_median_path,
    _weekly_saturation_loss,
    _weekly_scale_losses,
)


def test_weekly_scale_losses_penalize_structural_magnitude_explosion():
    actual_weekly = torch.tensor([0.020, -0.015, 0.025, -0.020, 0.018, -0.022])
    pred_calibrated = actual_weekly * 1.10
    pred_exploded = actual_weekly * 8.00

    calibrated = _weekly_scale_losses(pred_calibrated, actual_weekly)
    exploded = _weekly_scale_losses(pred_exploded, actual_weekly)

    assert calibrated["magnitude_ratio"].item() < 1.35
    assert exploded["magnitude_ratio"].item() > 3.0
    assert exploded["magnitude_loss"].item() > calibrated["magnitude_loss"].item() * 20.0


def test_weekly_median_cap_bounds_exploded_paths_and_keeps_gradients():
    pred = torch.zeros((2, 5, 7), dtype=torch.float32, requires_grad=True)
    pred.data[0, :, 3] = 0.20
    pred.data[1, :, 3] = -0.18

    bounded = _bound_weekly_median_path(
        pred,
        median_idx=3,
        weekly_median_cap=0.08,
        horizon=5,
    )
    weekly_median = bounded[:, :5, 3].sum(dim=1)

    assert torch.max(torch.abs(weekly_median)).item() <= 0.080001

    bounded.sum().backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_weekly_saturation_loss_penalizes_raw_paths_near_and_above_cap():
    in_range = torch.tensor([0.030, -0.035])
    near_cap = torch.tensor([0.046, -0.047])
    above_cap = torch.tensor([0.120, -0.130])

    assert _weekly_saturation_loss(in_range, weekly_median_cap=0.05).item() == 0.0
    assert _weekly_saturation_loss(near_cap, weekly_median_cap=0.05).item() > 0.0
    assert _weekly_saturation_loss(above_cap, weekly_median_cap=0.05).item() > (
        _weekly_saturation_loss(near_cap, weekly_median_cap=0.05).item() * 10.0
    )
