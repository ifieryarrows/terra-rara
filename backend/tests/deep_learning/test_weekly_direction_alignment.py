import pytest
import torch

from deep_learning.models import tft_copper


pytestmark = pytest.mark.skipif(
    getattr(tft_copper, "WeeklyASROPFLoss", None) is None,
    reason="pytorch_forecasting is not installed",
)


def test_weekly_directional_loss_prefers_same_sign_prediction():
    quantiles = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
    loss_fn = tft_copper.WeeklyASROPFLoss(
        quantiles=quantiles,
        lambda_weekly_quantile=0.0,
        lambda_t1_quantile=0.0,
        lambda_dispersion=0.0,
        lambda_magnitude=0.0,
        lambda_naive=0.0,
        lambda_directional=1.0,
    )

    batch, horizon, n_quantiles = 8, 5, 7
    actual = torch.full((batch, horizon), 0.01)

    pred_correct = torch.zeros(batch, horizon, n_quantiles)
    pred_correct[..., 3] = 0.01

    pred_wrong = torch.zeros(batch, horizon, n_quantiles)
    pred_wrong[..., 3] = -0.01

    correct_loss = loss_fn.loss(pred_correct, actual)
    wrong_loss = loss_fn.loss(pred_wrong, actual)

    assert correct_loss < wrong_loss
