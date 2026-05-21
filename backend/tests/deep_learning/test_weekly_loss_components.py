import torch

from deep_learning.models.tft_copper import (
    _bound_weekly_median_path,
    _weekly_interval_undercoverage_loss,
    _weekly_positive_rate_loss,
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


def test_weekly_scale_losses_penalize_bullish_mean_and_median_bias():
    actual_weekly = torch.tensor([-0.040, -0.030, 0.020, 0.010])
    pred_centered = actual_weekly.clone()
    pred_bullish = actual_weekly + 0.030

    centered = _weekly_scale_losses(pred_centered, actual_weekly)
    bullish = _weekly_scale_losses(pred_bullish, actual_weekly)

    assert bullish["bias_loss"].item() > centered["bias_loss"].item() + 1.0


def test_weekly_positive_rate_loss_only_penalizes_extreme_sign_collapse():
    actual_weekly = torch.tensor([-0.030, 0.010, 0.020, 0.040])
    pred_mid_rate = torch.tensor([-0.018, -0.012, 0.014, 0.020])
    pred_all_positive = torch.tensor([0.040, 0.050, 0.060, 0.070])
    pred_all_negative = torch.tensor([-0.070, -0.060, -0.050, -0.040])

    mid_loss = _weekly_positive_rate_loss(pred_mid_rate, actual_weekly)
    all_positive_loss = _weekly_positive_rate_loss(pred_all_positive, actual_weekly)
    all_negative_loss = _weekly_positive_rate_loss(pred_all_negative, actual_weekly)

    assert mid_loss.item() == 0.0
    assert all_positive_loss.item() > mid_loss.item()
    assert all_negative_loss.item() > mid_loss.item()


def test_weekly_interval_undercoverage_loss_prefers_wider_pi80_without_moving_q50():
    actual_weekly = torch.tensor([-0.040, -0.020, 0.020, 0.040])
    median = torch.tensor([-0.010, -0.010, 0.010, 0.010])
    narrow = torch.stack(
        [
            median - 0.003,
            median - 0.002,
            median - 0.001,
            median,
            median + 0.001,
            median + 0.002,
            median + 0.003,
        ],
        dim=1,
    )
    wide = torch.stack(
        [
            median - 0.060,
            median - 0.050,
            median - 0.025,
            median,
            median + 0.025,
            median + 0.050,
            median + 0.060,
        ],
        dim=1,
    )

    narrow_loss = _weekly_interval_undercoverage_loss(narrow, actual_weekly, [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98])
    wide_loss = _weekly_interval_undercoverage_loss(wide, actual_weekly, [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98])

    assert torch.equal(narrow[:, 3], wide[:, 3])
    assert torch.sign(narrow[:, 3]).tolist() == torch.sign(wide[:, 3]).tolist()
    assert wide_loss.item() < narrow_loss.item()


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
