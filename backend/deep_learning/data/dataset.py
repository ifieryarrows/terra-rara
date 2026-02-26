"""
TimeSeriesDataSet builder for pytorch_forecasting.

Wraps the feature_store output into train / validation / test splits
with proper temporal ordering (no leakage).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from deep_learning.config import TFTASROConfig, get_tft_config

logger = logging.getLogger(__name__)


def build_datasets(
    master_df: pd.DataFrame,
    time_varying_unknown_reals: list[str],
    time_varying_known_reals: list[str],
    target_cols: list[str],
    cfg: Optional[TFTASROConfig] = None,
):
    """
    Create pytorch_forecasting TimeSeriesDataSet objects for train / val / test.

    Uses chronological splitting:
        [train  |  val  |  test]

    Returns:
        (training_dataset, validation_dataset, test_dataset)
    """
    from pytorch_forecasting import TimeSeriesDataSet

    if cfg is None:
        cfg = get_tft_config()

    n = len(master_df)
    test_size = int(n * cfg.training.test_ratio)
    val_size = int(n * cfg.training.val_ratio)
    train_size = n - val_size - test_size

    if train_size < cfg.model.max_encoder_length + cfg.model.max_prediction_length:
        raise ValueError(
            f"Not enough data for TFT: {train_size} train rows, "
            f"need at least {cfg.model.max_encoder_length + cfg.model.max_prediction_length}"
        )

    train_cutoff = master_df["time_idx"].iloc[train_size - 1]
    val_cutoff = master_df["time_idx"].iloc[train_size + val_size - 1]

    logger.info(
        "Data split: train=%d (idx<=%.0f), val=%d (idx<=%.0f), test=%d",
        train_size, train_cutoff, val_size, val_cutoff, test_size,
    )

    target = target_cols[0] if target_cols else "target"

    training = TimeSeriesDataSet(
        master_df[master_df["time_idx"] <= train_cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=["group_id"],
        max_encoder_length=cfg.model.max_encoder_length,
        max_prediction_length=cfg.model.max_prediction_length,
        time_varying_unknown_reals=time_varying_unknown_reals,
        time_varying_known_reals=time_varying_known_reals,
        static_categoricals=["group_id"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        master_df[
            (master_df["time_idx"] > train_cutoff - cfg.model.max_encoder_length)
            & (master_df["time_idx"] <= val_cutoff)
        ],
        stop_randomization=True,
    )

    test = TimeSeriesDataSet.from_dataset(
        training,
        master_df[master_df["time_idx"] > val_cutoff - cfg.model.max_encoder_length],
        stop_randomization=True,
    )

    logger.info(
        "Datasets created: train=%d samples, val=%d, test=%d | "
        "encoder_len=%d, prediction_len=%d | "
        "%d unknown reals, %d known reals",
        len(training),
        len(validation),
        len(test),
        cfg.model.max_encoder_length,
        cfg.model.max_prediction_length,
        len(time_varying_unknown_reals),
        len(time_varying_known_reals),
    )

    return training, validation, test


def create_dataloaders(
    training_dataset,
    validation_dataset,
    test_dataset=None,
    cfg: Optional[TFTASROConfig] = None,
):
    """
    Create PyTorch DataLoaders from TimeSeriesDataSet objects.
    """
    if cfg is None:
        cfg = get_tft_config()

    train_dl = training_dataset.to_dataloader(
        train=True,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    val_dl = validation_dataset.to_dataloader(
        train=False,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    test_dl = None
    if test_dataset is not None:
        test_dl = test_dataset.to_dataloader(
            train=False,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
        )

    return train_dl, val_dl, test_dl
