import pytest
import torch

from deep_learning.models import tft_copper


pytestmark = pytest.mark.skipif(
    getattr(tft_copper, "WeeklyASROPFLoss", None) is None,
    reason="pytorch_forecasting is not installed",
)


QUANTILES = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]


def _path_from_median(median: torch.Tensor, spread: float = 0.004) -> torch.Tensor:
    offsets = torch.tensor([-3, -2, -1, 0, 1, 2, 3], dtype=median.dtype) * spread
    return median.unsqueeze(-1) + offsets.view(1, 1, -1)


def test_correct_weekly_direction_has_lower_loss_than_anti_direction():
    actual = torch.tensor([[0.01, 0.01, 0.0, 0.0, 0.0], [-0.01, -0.01, 0.0, 0.0, 0.0]])
    pred_good = _path_from_median(actual)
    pred_bad = _path_from_median(-actual)
    loss = tft_copper.WeeklyASROPFLoss(QUANTILES)
    assert loss.loss(pred_good, actual) < loss.loss(pred_bad, actual)


def test_material_move_prefers_larger_correct_magnitude():
    actual = torch.tensor([[0.02, 0.02, 0.01, 0.0, 0.0], [-0.02, -0.02, -0.01, 0.0, 0.0]])
    pred_tiny = _path_from_median(torch.sign(actual) * 0.001)
    pred_scaled = _path_from_median(actual * 0.8)
    loss = tft_copper.WeeklyASROPFLoss(QUANTILES)
    assert loss.loss(pred_scaled, actual) < loss.loss(pred_tiny, actual)


def test_quantile_crossing_increases_loss():
    actual = torch.tensor([[0.01, 0.01, 0.0, 0.0, 0.0], [-0.01, -0.01, 0.0, 0.0, 0.0]])
    pred = _path_from_median(actual)
    crossed = pred.clone()
    crossed[..., 1], crossed[..., 5] = crossed[..., 5].clone(), crossed[..., 1].clone()
    loss = tft_copper.WeeklyASROPFLoss(QUANTILES, lambda_crossing=10.0)
    assert loss.loss(crossed, actual) > loss.loss(pred, actual)
