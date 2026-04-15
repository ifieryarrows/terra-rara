"""Tests for CurriculumLossScheduler and SWACallback."""

import pytest
from unittest.mock import MagicMock

pytest.importorskip("lightning", reason="lightning/pytorch_lightning not installed locally")

from deep_learning.training.callbacks import CurriculumLossScheduler, SWACallback


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
