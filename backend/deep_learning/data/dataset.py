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


def _identity_target_normalizer():
    """No-op target normalizer so weekly sums stay in raw log-return space."""
    from pytorch_forecasting.data.encoders import TorchNormalizer

    return TorchNormalizer(
        method="identity",
        center=False,
        transformation=None,
    )


def compute_dataset_split_bounds(
    master_df: pd.DataFrame,
    cfg: Optional[TFTASROConfig] = None,
    *,
    purge_gap: int = 5,
) -> dict[str, int]:
    """Compute chronological train/val/test split boundaries with optional purge gaps.

    Places ``purge_gap`` unpredicted rows between train -> val and val -> test
    so overlapping 5-day return horizons and encoder edges cannot leak across
    split boundaries.
    """
    if cfg is None:
        cfg = get_tft_config()

    n = len(master_df)
    test_size = int(n * cfg.training.test_ratio)
    val_size = int(n * cfg.training.val_ratio)
    min_seq = cfg.model.max_encoder_length + cfg.model.max_prediction_length
    raw_train_size = n - val_size - test_size

    if raw_train_size < min_seq:
        raise ValueError(
            f"Not enough data for TFT: {raw_train_size} train rows, "
            f"need at least {min_seq}"
        )

    requested_gap = max(0, int(purge_gap))
    effective_gap = requested_gap
    if val_size > 0 and test_size > 0 and raw_train_size - 2 * effective_gap < min_seq:
        effective_gap = max(0, (raw_train_size - min_seq) // 2)
        if effective_gap < requested_gap:
            logger.warning(
                "Clamped dataset purge_gap from %d to %d on short frame (n=%d, train=%d)",
                requested_gap,
                effective_gap,
                n,
                raw_train_size,
            )

    test_start_pos = n - test_size
    val_end_pos = max(0, test_start_pos - effective_gap)
    val_start_pos = max(0, val_end_pos - val_size)
    train_end_pos = max(1, val_start_pos - (effective_gap if val_size > 0 else 0))

    train_cutoff = int(master_df["time_idx"].iloc[train_end_pos - 1])
    val_start_idx = int(master_df["time_idx"].iloc[min(val_start_pos, n - 1)])
    val_cutoff = int(master_df["time_idx"].iloc[max(0, min(val_end_pos - 1, n - 1))])
    test_start_idx = int(master_df["time_idx"].iloc[min(test_start_pos, n - 1)])
    max_time_idx = int(master_df["time_idx"].iloc[-1])

    return {
        "train_size": int(train_end_pos),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "purge_gap": int(effective_gap),
        "train_end_pos": int(train_end_pos),
        "val_start_pos": int(val_start_pos),
        "val_end_pos": int(val_end_pos),
        "test_start_pos": int(test_start_pos),
        "train_cutoff": train_cutoff,
        "val_start_idx": val_start_idx,
        "val_cutoff": val_cutoff,
        "test_start_idx": test_start_idx,
        "max_time_idx": max_time_idx,
    }


def build_datasets(
    master_df: pd.DataFrame,
    time_varying_unknown_reals: list[str],
    time_varying_known_reals: list[str],
    target_cols: list[str],
    cfg: Optional[TFTASROConfig] = None,
    purge_gap: int = 5,
):
    """
    Create pytorch_forecasting TimeSeriesDataSet objects for train / val / test.

    Uses chronological splitting with a ``purge_gap`` (default 5 bars) between
    train -> val and val -> test:
        [train  | gap |  val  | gap |  test]

    Returns:
        (training_dataset, validation_dataset, test_dataset)
    """
    from pytorch_forecasting import TimeSeriesDataSet

    if cfg is None:
        cfg = get_tft_config()

    bounds = compute_dataset_split_bounds(master_df, cfg, purge_gap=purge_gap)
    train_cutoff = bounds["train_cutoff"]
    val_start_idx = bounds["val_start_idx"]
    val_cutoff = bounds["val_cutoff"]
    test_start_idx = bounds["test_start_idx"]

    logger.info(
        "Data split: train=%d (idx<=%d), purge_gap=%d, val=%d (idx %d..%d), test=%d (idx>=%d)",
        bounds["train_size"],
        train_cutoff,
        bounds["purge_gap"],
        bounds["val_size"],
        val_start_idx,
        val_cutoff,
        bounds["test_size"],
        test_start_idx,
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
        target_normalizer=_identity_target_normalizer(),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        master_df[
            (master_df["time_idx"] >= val_start_idx - cfg.model.max_encoder_length)
            & (master_df["time_idx"] <= val_cutoff)
        ],
        stop_randomization=True,
    )

    test = TimeSeriesDataSet.from_dataset(
        training,
        master_df[master_df["time_idx"] >= test_start_idx - cfg.model.max_encoder_length],
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


def compute_cv_fold_bounds(
    master_df: pd.DataFrame,
    cfg: Optional[TFTASROConfig] = None,
    *,
    n_folds: int = 3,
    purge_gap: int = 5,
) -> list[dict[str, int]]:
    """Compute expanding-window purged CV fold boundaries strictly before test pool."""
    if cfg is None:
        cfg = get_tft_config()

    n = len(master_df)
    test_size = int(n * cfg.training.test_ratio)
    eff_gap = max(0, int(purge_gap))
    min_seq = cfg.model.max_encoder_length + cfg.model.max_prediction_length
    cv_pool_size = n - test_size - (eff_gap if (n - test_size - eff_gap) >= min_seq + 15 else 0)

    min_train_size = max(int(cv_pool_size * 0.60), min_seq + 10)
    if min_train_size >= cv_pool_size:
        raise ValueError(
            f"Not enough data for {n_folds}-fold CV: "
            f"cv_pool={cv_pool_size}, min_train={min_train_size}"
        )

    available = cv_pool_size - min_train_size
    fold_step = max(1, available // n_folds)
    bounds_list: list[dict[str, int]] = []

    for fold_idx in range(n_folds):
        train_end_pos = min(min_train_size + fold_idx * fold_step, cv_pool_size - fold_step)
        val_start_pos = train_end_pos + eff_gap
        val_end_pos = min(val_start_pos + fold_step, cv_pool_size)

        if val_start_pos >= cv_pool_size or val_end_pos <= val_start_pos:
            logger.warning("Fold %d skipped: purge gap exhausts remaining data", fold_idx)
            continue

        bounds_list.append(
            {
                "fold_idx": int(fold_idx),
                "purge_gap": int(eff_gap),
                "train_end_pos": int(train_end_pos),
                "val_start_pos": int(val_start_pos),
                "val_end_pos": int(val_end_pos),
                "train_cutoff": int(master_df["time_idx"].iloc[train_end_pos - 1]),
                "val_start_idx": int(master_df["time_idx"].iloc[val_start_pos]),
                "val_cutoff": int(master_df["time_idx"].iloc[val_end_pos - 1]),
            }
        )
    return bounds_list


def build_cv_fold_dataset(
    master_df: pd.DataFrame,
    time_varying_unknown_reals: list[str],
    time_varying_known_reals: list[str],
    target_cols: list[str],
    cfg: TFTASROConfig,
    bounds: dict[str, int],
):
    """Materialize a single ``(training_ds, validation_ds)`` pair from fold bounds."""
    from pytorch_forecasting import TimeSeriesDataSet

    target = target_cols[0] if target_cols else "target"
    train_cutoff = int(bounds["train_cutoff"])
    val_start_idx = int(bounds["val_start_idx"])
    val_cutoff = int(bounds["val_cutoff"])

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
        target_normalizer=_identity_target_normalizer(),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=True,
    )

    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        val_data,
        stop_randomization=True,
    )
    return training_ds, validation_ds


def build_cv_folds(
    master_df: pd.DataFrame,
    time_varying_unknown_reals: list[str],
    time_varying_known_reals: list[str],
    target_cols: list[str],
    cfg: Optional[TFTASROConfig] = None,
    n_folds: int = 3,
    purge_gap: int = 5,
    mrmr_top_k: Optional[int] = None,
):
    """
    Purged Walk-Forward Temporal CV with expanding training windows.

    Test set (last ``test_ratio`` %) is excluded from the CV pool entirely.
    The remaining data is split into ``n_folds`` expanding-window folds::

        Fold 1: [===TRAIN 60%===][GAP][=VAL=][................]
        Fold 2: [======TRAIN 73%======][GAP][=VAL=][.........]
        Fold 3: [=========TRAIN 87%=========][GAP][=VAL=]

    The ``purge_gap`` removes N samples between train and validation to
    prevent autocovariance-based data leakage (de Prado, 2018). When
    ``mrmr_top_k`` is supplied and > 0, runs leak-free per-fold MRMR
    feature selection strictly on each fold's training rows.

    Returns:
        List of ``(training_dataset, validation_dataset)`` tuples.
    """
    if cfg is None:
        cfg = get_tft_config()

    fold_bounds_list = compute_cv_fold_bounds(
        master_df,
        cfg,
        n_folds=n_folds,
        purge_gap=purge_gap,
    )
    folds: list[tuple] = []

    for bounds in fold_bounds_list:
        fold_df = master_df
        fold_unknown = list(time_varying_unknown_reals)
        fold_known = list(time_varying_known_reals)

        if mrmr_top_k is not None and int(mrmr_top_k) > 0:
            from deep_learning.data.feature_selection import build_fold_scoped_features

            fold_df, fold_unknown, fold_known = build_fold_scoped_features(
                master_df,
                train_end_pos=bounds["train_end_pos"],
                target_col=cfg.forecast.primary_target_col,
                mrmr_top_k=int(mrmr_top_k),
                horizon=cfg.forecast.primary_horizon_days,
                known_features=fold_known,
            )

        training_ds, validation_ds = build_cv_fold_dataset(
            fold_df,
            fold_unknown,
            fold_known,
            target_cols,
            cfg,
            bounds,
        )

        logger.info(
            "CV Fold %d/%d: train=%d samples (idx<=%d), "
            "purge_gap=%d, val=%d (idx %d–%d, %d unknown feats)",
            bounds["fold_idx"] + 1,
            n_folds,
            len(training_ds),
            bounds["train_cutoff"],
            bounds["purge_gap"],
            len(validation_ds),
            bounds["val_start_idx"],
            bounds["val_cutoff"],
            len(fold_unknown),
        )

        folds.append((training_ds, validation_ds))

    return folds


def _resolve_num_workers(configured: int) -> int:
    """
    Return a safe num_workers value for the current platform.

    On Windows (os.name == 'nt'), PyTorch DataLoader multiprocessing requires
    the script to be inside an ``if __name__ == '__main__'`` guard, which is
    not the case in training scripts. Force 0 to avoid deadlocks.

    On Linux/macOS (GitHub Actions, HF Spaces), honor the configured value as
    well. In particular, configured ``0`` is intentional for strict
    reproducibility and must not silently become multiprocessing.
    """
    if os.name == "nt":
        return 0
    return max(int(configured), 0)


def make_dataloader_generator(seed: int):
    """Create an isolated CPU ``torch.Generator`` seeded deterministically."""
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def create_train_dataloader(
    training_dataset,
    cfg: Optional[TFTASROConfig] = None,
    *,
    seed: Optional[int] = None,
):
    """Create a shuffled training DataLoader with an independent per-seed generator."""
    if cfg is None:
        cfg = get_tft_config()
    nw = _resolve_num_workers(cfg.training.num_workers)
    eff_seed = int(cfg.training.seed if seed is None else seed)
    train_generator = make_dataloader_generator(eff_seed)
    return training_dataset.to_dataloader(
        train=True,
        batch_size=cfg.training.batch_size,
        num_workers=nw,
        shuffle=True,
        generator=train_generator,
    )


def create_dataloaders(
    training_dataset,
    validation_dataset,
    test_dataset=None,
    cfg: Optional[TFTASROConfig] = None,
    *,
    seed: Optional[int] = None,
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

    # Keep the training windows shuffled, as required by the stochastic
    # optimizer, but give every seed its own isolated generator so multi-seed
    # runs never couple seed 137/256 batch order to seed 42's epoch count.
    train_dl = create_train_dataloader(training_dataset, cfg, seed=seed)
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
