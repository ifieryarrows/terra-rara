"""Tests for ASRO and quantile loss functions."""

import pytest
import torch
import numpy as np

from deep_learning.models.losses import CombinedQuantileLoss, AdaptiveSharpeRatioLoss


@pytest.fixture
def sample_predictions():
    """(batch=16, prediction_length=1, quantiles=7)"""
    torch.manual_seed(42)
    y_pred = torch.randn(16, 1, 7).sort(dim=-1).values
    y_actual = torch.randn(16, 1)
    return y_pred, y_actual


def test_quantile_loss_is_positive(sample_predictions):
    y_pred, y_actual = sample_predictions
    loss_fn = CombinedQuantileLoss()
    loss = loss_fn(y_pred, y_actual)
    assert loss.item() > 0, "Quantile loss should be positive"


def test_quantile_loss_perfect_prediction():
    y = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]])
    y_actual = torch.tensor([[4.0]])
    loss_fn = CombinedQuantileLoss()
    loss = loss_fn(y, y_actual)
    assert loss.item() >= 0


def test_asro_loss_returns_scalar(sample_predictions):
    y_pred, y_actual = sample_predictions
    loss_fn = AdaptiveSharpeRatioLoss()
    loss = loss_fn(y_pred, y_actual)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), "Loss should be finite"


def test_asro_loss_lambda_sensitivity():
    torch.manual_seed(0)
    y_pred = torch.randn(32, 1, 7).sort(dim=-1).values
    y_actual = torch.randn(32, 1)

    loss_low = AdaptiveSharpeRatioLoss(lambda_vol=0.1, lambda_quantile=0.1)
    loss_high = AdaptiveSharpeRatioLoss(lambda_vol=0.9, lambda_quantile=0.9)

    l1 = loss_low(y_pred, y_actual).item()
    l2 = loss_high(y_pred, y_actual).item()

    assert l1 != l2, "Different lambdas should produce different losses"


def test_asro_from_config():
    from deep_learning.config import ASROConfig
    cfg = ASROConfig(lambda_vol=0.4, lambda_quantile=0.3)
    quantiles = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)
    loss_fn = AdaptiveSharpeRatioLoss.from_config(cfg, quantiles)
    assert loss_fn.lambda_vol == 0.4
    assert loss_fn.lambda_quantile == 0.3
