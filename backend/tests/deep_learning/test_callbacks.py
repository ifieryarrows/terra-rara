"""Tests for training callbacks."""

import pytest
from unittest.mock import MagicMock

pytest.importorskip("lightning", reason="lightning/pytorch_lightning not installed locally")

from deep_learning.training.callbacks import (
    CurriculumLossScheduler,
    SWACallback,
    WeeklyLossComponentLogger,
)


def test_curriculum_warmup_starts_with_high_quantile_weight():
    cb = CurriculumLossScheduler(
        warmup_epochs=10,
        initial_lambda_quantile=0.65,
        target_lambda_quantile=0.35,
        initial_lambda_madl=0.05,
        target_lambda_madl=0.25,
    )

    loss = MagicMock()
    loss.lambda_quantile = 0.35
    loss.lambda_madl = 0.25

    pl_module = MagicMock()
    pl_module.loss = loss
    trainer = MagicMock()
    trainer.current_epoch = 0

    cb.on_train_epoch_start(trainer, pl_module)

    assert loss.lambda_quantile == 0.65
    assert loss.lambda_madl == 0.05


def test_curriculum_reaches_target_after_warmup():
    cb = CurriculumLossScheduler(
        warmup_epochs=10,
        initial_lambda_quantile=0.65,
        target_lambda_quantile=0.35,
        initial_lambda_madl=0.05,
        target_lambda_madl=0.25,
    )

    loss = MagicMock()
    loss.lambda_quantile = 0.65
    loss.lambda_madl = 0.05

    pl_module = MagicMock()
    pl_module.loss = loss
    trainer = MagicMock()
    trainer.current_epoch = 15

    cb.on_train_epoch_start(trainer, pl_module)

    assert loss.lambda_quantile == 0.35
    assert loss.lambda_madl == 0.25


def test_curriculum_midway_interpolation():
    cb = CurriculumLossScheduler(
        warmup_epochs=10,
        initial_lambda_quantile=0.60,
        target_lambda_quantile=0.40,
    )

    loss = MagicMock()
    loss.lambda_quantile = 0.60

    pl_module = MagicMock()
    pl_module.loss = loss
    trainer = MagicMock()
    trainer.current_epoch = 5

    cb.on_train_epoch_start(trainer, pl_module)

    assert 0.45 < loss.lambda_quantile < 0.55


def test_swa_callback_does_not_average_before_start():
    cb = SWACallback(swa_start_pct=0.75)
    assert cb._n_averaged == 0

    trainer = MagicMock()
    trainer.max_epochs = 100
    trainer.current_epoch = 50

    pl_module = MagicMock()
    cb.on_train_epoch_end(trainer, pl_module)

    assert cb._n_averaged == 0


def test_weekly_loss_logger_publishes_validation_objective():
    cb = WeeklyLossComponentLogger()
    loss = MagicMock()
    loss.component_means.return_value = {
        "n_batches": 2,
        "weekly_q_loss_mean": 0.01,
        "t1_q_loss_mean": 0.02,
        "t1_directional_loss_mean": 0.03,
        "dispersion_loss_mean": 0.04,
        "magnitude_loss_mean": 0.05,
        "naive_loss_mean": 0.06,
        "bias_loss_mean": 0.07,
        "saturation_loss_mean": 0.08,
        "positive_rate_loss_mean": 0.09,
        "interval_loss_mean": 0.10,
        "directional_loss_mean": 0.11,
        "total_loss_mean": 0.42,
        "dominant_component": "directional",
    }
    pl_module = MagicMock()
    pl_module.loss = loss
    trainer = MagicMock()
    trainer.current_epoch = 3

    cb.on_validation_epoch_end(trainer, pl_module)

    pl_module.log.assert_any_call(
        "val_weekly_loss",
        0.42,
        on_step=False,
        on_epoch=True,
        prog_bar=False,
        logger=True,
        sync_dist=False,
    )
