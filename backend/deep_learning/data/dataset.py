"""
TimeSeriesDataSet builder for pytorch_forecasting.

Wraps the feature_store output into train / validation / test splits
with proper temporal ordering (no leakage).
"""

from __future__ import annotations

import logging
import os
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


def build_cv_folds(
    master_df: pd.DataFrame,
    time_varying_unknown_reals: list[str],
    time_varying_known_reals: list[str],
    target_cols: list[str],
    cfg: Optional[TFTASROConfig] = None,
    n_folds: int = 3,
    purge_gap: int = 5,
):
    """
    Purged Walk-Forward Temporal CV with expanding training windows.

    Test set (last ``test_ratio`` %) is excluded from the CV pool entirely.
    The remaining data is split into ``n_folds`` expanding-window folds::

        Fold 1: [===TRAIN 60%===][GAP][=VAL=][................]
        Fold 2: [======TRAIN 73%======][GAP][=VAL=][.........]
        Fold 3: [=========TRAIN 87%=========][GAP][=VAL=]

    The ``purge_gap`` removes N samples between train and validation to
    prevent autocovariance-based data leakage (de Prado, 2018).

    Each validation block covers a different market regime, so Optuna
    cannot overfit to a single time window (REG-2026-001 root cause).

    When ``n_folds=1``, returns a single fold equivalent to the old
    single-split behaviour (backward-compatible fallback).

    Returns:
        List of ``(training_dataset, validation_dataset)`` tuples.
    """
    from pytorch_forecasting import TimeSeriesDataSet

    if cfg is None:
        cfg = get_tft_config()

    n = len(master_df)
    test_size = int(n * cfg.training.test_ratio)
    cv_pool_size = n - test_size

    min_seq = cfg.model.max_encoder_length + cfg.model.max_prediction_length

    # Minimum training size: 60 % of CV pool (ensures enough history)
    min_train_size = max(int(cv_pool_size * 0.60), min_seq + 10)
    if min_train_size >= cv_pool_size:
        raise ValueError(
            f"Not enough data for {n_folds}-fold CV: "
            f"cv_pool={cv_pool_size}, min_train={min_train_size}"
        )

    # Divide the remaining space into n_folds equal validation blocks
    available = cv_pool_size - min_train_size
    fold_step = max(1, available // n_folds)

    target = target_cols[0] if target_cols else "target"
    folds: list[tuple] = []

    for fold_idx in range(n_folds):
        train_end_pos = min(min_train_size + fold_idx * fold_step, cv_pool_size - fold_step)
        val_start_pos = train_end_pos + purge_gap
        val_end_pos = min(val_start_pos + fold_step, cv_pool_size)

        if val_start_pos >= cv_pool_size or val_end_pos <= val_start_pos:
            logger.warning("Fold %d skipped: purge gap exhausts remaining data", fold_idx)
            continue

        train_cutoff = master_df["time_idx"].iloc[train_end_pos - 1]
        val_start_idx = master_df["time_idx"].iloc[val_start_pos]
        val_cutoff = master_df["time_idx"].iloc[val_end_pos - 1]

        train_data = master_df[master_df["time_idx"] <= train_cutoff]
        val_data = master_df[
            (master_df["time_idx"] >= val_start_idx - cfg.model.max_encoder_length)
            & (master_df["time_idx"] <= val_cutoff)
        ]

        training_ds = TimeSeriesDataSet(
            train_data,
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

        validation_ds = TimeSeriesDataSet.from_dataset(
            training_ds,
            val_data,
            stop_randomization=True,
        )

        logger.info(
            "CV Fold %d/%d: train=%d samples (idx<=%.0f), "
            "purge_gap=%d, val=%d (idx %.0f–%.0f)",
            fold_idx + 1, n_folds,
            len(training_ds), train_cutoff,
            purge_gap,
            len(validation_ds), val_start_idx, val_cutoff,
        )

        folds.append((training_ds, validation_ds))

    return folds


def _resolve_num_workers(configured: int) -> int:
    """
    Return a safe num_workers value for the current platform.

    On Windows (os.name == 'nt'), PyTorch DataLoader multiprocessing requires
    the script to be inside an ``if __name__ == '__main__'`` guard, which is
    not the case in training scripts. Force 0 to avoid deadlocks.

    On Linux/macOS (GitHub Actions, HF Spaces), use the configured value;
    default to 2 when the config still carries the old 0.
    """
    if os.name == "nt":
        return 0
    # On POSIX: honour config; upgrade 0 → 2 as a sensible floor
    return max(configured, 2)


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

    nw = _resolve_num_workers(cfg.training.num_workers)
    logger.info(
        "DataLoader workers: %d (platform=%s, configured=%d)",
        nw, os.name, cfg.training.num_workers,
    )

    train_dl = training_dataset.to_dataloader(
        train=True,
        batch_size=cfg.training.batch_size,
        num_workers=nw,
    )
    val_dl = validation_dataset.to_dataloader(
        train=False,
        batch_size=cfg.training.batch_size,
        num_workers=nw,
    )
    test_dl = None
    if test_dataset is not None:
        test_dl = test_dataset.to_dataloader(
            train=False,
            batch_size=cfg.training.batch_size,
            num_workers=nw,
        )

    return train_dl, val_dl, test_dl
