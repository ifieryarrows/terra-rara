import torch

from deep_learning.models.tft_copper import _weekly_scale_losses


def test_weekly_scale_losses_penalize_structural_magnitude_explosion():
    actual_weekly = torch.tensor([0.020, -0.015, 0.025, -0.020, 0.018, -0.022])
    pred_calibrated = actual_weekly * 1.10
    pred_exploded = actual_weekly * 8.00

    calibrated = _weekly_scale_losses(pred_calibrated, actual_weekly)
    exploded = _weekly_scale_losses(pred_exploded, actual_weekly)

    assert calibrated["magnitude_ratio"].item() < 1.35
    assert exploded["magnitude_ratio"].item() > 3.0
    assert exploded["magnitude_loss"].item() > calibrated["magnitude_loss"].item() * 20.0
