"""
TFT-ASRO Training Pipeline.

Orchestrates the full training flow:
    1. Build feature store from DB
    2. Create TimeSeriesDataSet splits
    3. Instantiate TFT-ASRO model
    4. Train with PyTorch Lightning
    5. Evaluate on test set with financial metrics
    6. Persist model checkpoint and metadata

Usage:
    python -m deep_learning.training.trainer --symbol HG=F
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import os
import platform
import sys
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from deep_learning.config import TFTASROConfig, get_tft_config
from deep_learning.contract import (
    FORECAST_CONTRACT_VERSION,
    PUBLIC_RETURN_SPACE,
    RETURN_SPACE,
    TARGET_RETURN_TYPE,
)
from deep_learning.logging_utils import configure_cli_logging, suppress_lightning_noise

# pytorch_forecasting prescalers are fit on DataFrames but transform numpy arrays
# internally on every batch — this produces thousands of identical sklearn warnings.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)

logger = logging.getLogger(__name__)


def _build_data_snapshot_metadata(master_df: pd.DataFrame) -> dict:
    """Describe the exact feature frame used for a train/test split."""
    from deep_learning.data.feature_store import data_snapshot_sha256

    return {
        "sha256": data_snapshot_sha256(master_df),
        "rows": int(len(master_df)),
        "columns": int(master_df.shape[1]),
        "first_index": str(master_df.index.min()) if len(master_df) else None,
        "last_index": str(master_df.index.max()) if len(master_df) else None,
        "tft_data_as_of": os.environ.get("TFT_DATA_AS_OF") or None,
    }


def _runtime_environment_metadata() -> dict:
    """Capture versions that can materially change a TFT run."""
    package_names = (
        "torch",
        "lightning",
        "pytorch-forecasting",
        "optuna",
        "optuna-integration",
        "numpy",
        "pandas",
        "scikit-learn",
    )
    packages = {}
    for package_name in package_names:
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None
    try:
        import torch

        torch_runtime = {
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
        }
    except Exception:
        torch_runtime = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "github_sha": os.environ.get("GITHUB_SHA") or None,
        "runner_image": os.environ.get("ImageOS") or os.environ.get("RUNNER_OS") or None,
        "packages": packages,
        "process_controls": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
        "torch_runtime": torch_runtime,
    }


def _configure_tft_reproducibility() -> None:
    """Keep the historical private entry point for callers and tests."""
    from deep_learning.training.reproducibility import configure_tft_reproducibility

    configure_tft_reproducibility()

KNOWN_GOOD_CONFIG = {
    "max_encoder_length": 15,
    "hidden_size": 12,
    "attention_head_size": 1,
    "dropout": 0.30,
    "hidden_continuous_size": 6,
    "learning_rate": 5e-4,
    "output_head_lr_mult": 0.20,
    "weight_decay": 2e-2,
    "mrmr_top_k": 5,
    "lambda_vol": 0.30,
    "lambda_quantile": 0.25,
    "lambda_madl": 0.40,
    "lambda_weekly_quantile": 1.00,
    "lambda_t1_quantile": 0.20,
    "lambda_subhorizon_quantile": 0.15,
    "lambda_t1_directional": 0.0,
    "lambda_dispersion": 0.0,
    "lambda_magnitude": 0.0,
    "lambda_naive": 0.0,
    "lambda_bias": 0.0,
    "lambda_drift_prior": 1.00,
    "lambda_directional": 0.10,
    "lambda_saturation": 0.0,
    "lambda_positive_rate": 0.0,
    "lambda_interval": 0.0,
    "batch_size": 32,
}

CONTROLLED_WEEKLY_OPTUNA_PARAMS = (
    "max_encoder_length",
    "hidden_size",
    "hidden_continuous_size",
    "attention_head_size",
    "dropout",
    "learning_rate",
    "output_head_lr_mult",
    "weight_decay",
    "mrmr_top_k",
    "batch_size",
    "lambda_weekly_quantile",
    "lambda_t1_quantile",
    "lambda_subhorizon_quantile",
    "lambda_t1_directional",
    "lambda_dispersion",
    "lambda_magnitude",
    "lambda_naive",
    "lambda_bias",
    "lambda_drift_prior",
    "lambda_directional",
    "lambda_saturation",
    "lambda_positive_rate",
    "lambda_interval",
)

DETERMINISTIC_WEEKLY_CONFIG = dict(KNOWN_GOOD_CONFIG)

REQUIRED_PROMOTABLE_METRICS = (
    "weekly_directional_accuracy",
    "weekly_magnitude_ratio",
    "weekly_tail_capture_rate",
    "weekly_mae_vs_naive_zero",
    "weekly_pi80_coverage",
    "weekly_pi80_width_ratio",
    "weekly_pi96_coverage",
    "weekly_pi96_width_ratio",
    "weekly_sample_count",
    "weekly_quantile_crossing_rate",
    "weekly_sorted_quantile_crossing_rate",
    "weekly_pred_positive_rate",
    "weekly_actual_positive_rate",
    "quantile_crossing_rate",
    "sorted_quantile_crossing_rate",
)

WEEKLY_INTERVAL_CONDITIONING_FEATURE = "realized_vol_20d"


def _validation_ranked_checkpoint_paths(
    checkpoint_callback,
    *,
    max_count: int = 2,
) -> list[tuple[Path, float]]:
    """Return the best saved checkpoints using validation monitor scores only."""
    ranked: list[tuple[Path, float]] = []
    saved_scores = getattr(checkpoint_callback, "best_k_models", {}) or {}
    for raw_path, raw_score in saved_scores.items():
        try:
            score = float(raw_score.detach().cpu())
        except AttributeError:
            score = float(raw_score)
        ranked.append((Path(raw_path), score))

    ranked.sort(key=lambda item: (item[1], str(item[0])))
    if len(ranked) > 1:
        best_score = ranked[0][1]
        max_allowed_score = (
            best_score * 1.03
            if best_score > 0.0
            else best_score + 0.03 * max(1.0, abs(best_score))
        )
        ranked = [ranked[0]] + [
            item for item in ranked[1:] if item[1] <= max_allowed_score
        ]
    if not ranked:
        best_path = str(getattr(checkpoint_callback, "best_model_path", "") or "")
        if best_path:
            raw_score = getattr(
                checkpoint_callback,
                "best_model_score",
                float("inf"),
            )
            try:
                score = float(raw_score.detach().cpu())
            except AttributeError:
                score = float(raw_score)
            ranked.append((Path(best_path), score))
    return ranked[: max(1, int(max_count))]


def _build_uniform_checkpoint_soup(
    source_paths: list[Path],
    destination: Path,
    *,
    max_epoch_gap: Optional[int] = 5,
) -> bool:
    """Average compatible model weights into one deployable Lightning checkpoint.

    Source checkpoints have already been ranked by the validation monitor. Test
    predictions and labels are unavailable here by design. Non-floating state
    (for example counters) is retained from the best-ranked checkpoint.
    Enforces an epoch-proximity constraint ``abs(ep_a - ep_b) <= max_epoch_gap``
    when integer epoch metadata is present so disjoint training basins are not
    destructively averaged. Returns ``True`` if multiple checkpoints were
    averaged, else ``False``.
    """
    if not source_paths:
        raise ValueError("At least one checkpoint is required for promotion")

    import shutil
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(source_paths) == 1:
        shutil.copy2(source_paths[0], destination)
        return False

    def _load(path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # torch<2.6 compatibility
            return torch.load(path, map_location="cpu")

    payloads = [_load(path) for path in source_paths]
    state_dicts = [payload.get("state_dict") for payload in payloads]
    if any(not isinstance(state, dict) for state in state_dicts):
        raise RuntimeError("Lightning checkpoint is missing a state_dict")

    reference_keys = tuple(state_dicts[0])
    for path, state in zip(source_paths[1:], state_dicts[1:]):
        if tuple(state) != reference_keys:
            raise RuntimeError(f"Checkpoint state keys differ: {path}")

    # Filter out secondary checkpoints whose training epoch is farther than max_epoch_gap
    # from the primary validation-best checkpoint (distinct basins fail linear interpolation).
    selected_indices = [0]
    ref_epoch = payloads[0].get("epoch")
    for idx in range(1, len(payloads)):
        cand_epoch = payloads[idx].get("epoch")
        if (
            max_epoch_gap is not None
            and isinstance(ref_epoch, int)
            and isinstance(cand_epoch, int)
            and abs(int(cand_epoch) - int(ref_epoch)) > int(max_epoch_gap)
        ):
            logger.info(
                "Skipping checkpoint soup member %s (epoch=%s too far from best epoch=%s; max_gap=%d)",
                source_paths[idx].name,
                cand_epoch,
                ref_epoch,
                int(max_epoch_gap),
            )
            continue
        selected_indices.append(idx)

    if len(selected_indices) <= 1:
        shutil.copy2(source_paths[0], destination)
        return False

    selected_states = [state_dicts[i] for i in selected_indices]
    averaged_payload = payloads[0]
    averaged_state = averaged_payload["state_dict"]
    for key in reference_keys:
        tensors = [state[key] for state in selected_states]
        reference = tensors[0]
        if any(tensor.shape != reference.shape for tensor in tensors[1:]):
            raise RuntimeError(f"Checkpoint tensor shape differs for {key}")
        if torch.is_floating_point(reference) or torch.is_complex(reference):
            accumulator_dtype = (
                torch.complex128 if torch.is_complex(reference) else torch.float64
            )
            averaged_state[key] = torch.stack(
                [tensor.to(dtype=accumulator_dtype) for tensor in tensors]
            ).mean(dim=0).to(dtype=reference.dtype)
        else:
            averaged_state[key] = reference

    torch.save(averaged_payload, destination)
    return True


def _validate_quantile_prediction_shape(pred_np: np.ndarray, cfg: TFTASROConfig) -> None:
    if pred_np.ndim != 3:
        raise RuntimeError(
            f"Expected quantile prediction tensor [n, horizon, q], got shape={pred_np.shape}. "
            "Weekly quality gate cannot run without full multi-horizon quantile predictions."
        )
    if pred_np.shape[1] < cfg.forecast.primary_horizon_days:
        raise RuntimeError(
            f"Prediction horizon too short: got {pred_np.shape[1]}, "
            f"need {cfg.forecast.primary_horizon_days}"
        )
    if pred_np.shape[2] != len(cfg.model.quantiles):
        raise RuntimeError(
            f"Quantile dim mismatch: {pred_np.shape[2]} != {len(cfg.model.quantiles)}"
        )


def _predict_quantiles_to_np(
    mdl,
    dataloader,
    cfg: TFTASROConfig,
    return_y: bool = False,
    *,
    ordered: bool = False,
):
    y_actual_np = None
    pred = None
    try:
        pred_out = mdl.predict(dataloader, mode="quantiles", return_y=True)
        if hasattr(pred_out, "output") and hasattr(pred_out, "y"):
            pred = pred_out.output
            y_val = pred_out.y[0] if isinstance(pred_out.y, (list, tuple)) else pred_out.y
            y_actual_np = y_val.cpu().numpy() if hasattr(y_val, "cpu") else np.asarray(y_val)
        elif isinstance(pred_out, (list, tuple)) and len(pred_out) == 2:
            pred, y_val = pred_out
            y_val = y_val[0] if isinstance(y_val, (list, tuple)) else y_val
            y_actual_np = y_val.cpu().numpy() if hasattr(y_val, "cpu") else np.asarray(y_val)
        else:
            pred = pred_out
    except TypeError:
        pred = mdl.predict(dataloader, mode="quantiles")

    if return_y and y_actual_np is None:
        y_parts = []
        for batch in dataloader:
            y_parts.append(batch[1][0] if isinstance(batch[1], (list, tuple)) else batch[1])
        if y_parts:
            import torch
            y_actual_np = torch.cat(y_parts).cpu().numpy()

    pred_np = pred.cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)
    _validate_quantile_prediction_shape(pred_np, cfg)
    if ordered:
        from deep_learning.models.tft_copper import ensure_ordered_quantiles_from_model_output

        median_idx = len(cfg.model.quantiles) // 2
        pred_np = ensure_ordered_quantiles_from_model_output(
            pred_np,
            model=mdl,
            median_idx=median_idx,
        )
    if return_y:
        return pred_np, y_actual_np
    return pred_np


def _require_promotable_metrics(metrics: dict) -> None:
    missing = [
        key for key in REQUIRED_PROMOTABLE_METRICS
        if key not in metrics or metrics.get(key) is None
    ]
    if missing:
        raise RuntimeError(
            f"Required TFT promotion metrics missing after evaluation: {missing}. "
            "Refusing to write promotable TFT metadata."
        )


def _compute_test_metrics_from_quantiles(
    y_actual_path: np.ndarray,
    pred_np: np.ndarray,
    cfg: TFTASROConfig,
    *,
    weekly_interval_scale: float = 1.0,
    weekly_interval_calibration: Optional[dict] = None,
    weekly_interval_conditioning_values: Optional[np.ndarray] = None,
    weekly_cqr_params: Optional[dict] = None,
    realized_vol: Optional[np.ndarray] = None,
    already_ordered: bool = False,
) -> dict[str, float]:
    from deep_learning.training.metrics import evaluate_quantile_predictions

    pred_np = np.asarray(pred_np)
    _validate_quantile_prediction_shape(pred_np, cfg)

    n_path = min(len(y_actual_path), len(pred_np))
    evaluator_kwargs = {
        "quantiles": cfg.model.quantiles,
        "horizon": cfg.forecast.primary_horizon_days,
        "weekly_median_cap": cfg.weekly_loss.weekly_median_cap,
    }
    if already_ordered:
        evaluator_kwargs["already_ordered"] = True
    # Keep the historical evaluator call shape for downstream diagnostics and
    # tests when no validation-fitted interval adjustment is needed.
    calibration = weekly_interval_calibration or {}
    effective_scale = float(calibration.get("weekly_interval_scale", weekly_interval_scale))
    if effective_scale != 1.0:
        evaluator_kwargs["weekly_interval_scale"] = effective_scale
    if weekly_cqr_params is not None:
        evaluator_kwargs["weekly_cqr_params"] = weekly_cqr_params
    if realized_vol is not None:
        evaluator_kwargs["realized_vol"] = realized_vol
    if calibration.get("weekly_interval_conditioning_enabled"):
        if weekly_interval_conditioning_values is None:
            raise RuntimeError(
                "Weekly interval calibration requires forecast-origin conditioning values"
            )
        evaluator_kwargs.update(
            {
                "weekly_interval_conditioning_values": weekly_interval_conditioning_values,
                "weekly_interval_conditioning_reference": calibration.get(
                    "weekly_interval_conditioning_reference"
                ),
                "weekly_interval_conditioning_power": calibration.get(
                    "weekly_interval_conditioning_power", 0.35
                ),
                "weekly_interval_conditioning_min_factor": calibration.get(
                    "weekly_interval_conditioning_min_factor", 0.5
                ),
                "weekly_interval_conditioning_max_factor": calibration.get(
                    "weekly_interval_conditioning_max_factor", 2.0
                ),
                "weekly_interval_conditioning_min_scale": calibration.get(
                    "weekly_interval_conditioning_min_scale", 0.20
                ),
                "weekly_interval_conditioning_max_scale": calibration.get(
                    "weekly_interval_conditioning_max_scale", 2.50
                ),
            }
        )
    test_metrics = evaluate_quantile_predictions(
        y_actual_path[:n_path],
        pred_np[:n_path],
        **evaluator_kwargs,
    )
    _log_weekly_alignment_sample(
        y_actual_path[:n_path],
        pred_np[:n_path],
        cfg,
        already_ordered=already_ordered,
    )
    _require_promotable_metrics(test_metrics)
    return test_metrics


def _weekly_interval_conditioning_values(
    feature_frame: Optional[pd.DataFrame],
    *,
    feature_name: str,
    start_exclusive: int,
    end_inclusive: int,
    expected_count: int,
) -> np.ndarray:
    """Return condition values for the exact forecast origins being scored."""
    if feature_frame is None or feature_name not in feature_frame.columns:
        raise RuntimeError(
            f"Weekly interval conditioning feature is missing: {feature_name}"
        )
    if "time_idx" not in feature_frame.columns:
        raise RuntimeError("Weekly interval conditioning requires a time_idx column")
    times = pd.to_numeric(feature_frame["time_idx"], errors="coerce")
    selected = feature_frame.loc[
        (times > int(start_exclusive)) & (times <= int(end_inclusive))
    ].sort_values("time_idx")
    if len(selected) != int(expected_count):
        raise RuntimeError(
            "Weekly interval conditioning origin count does not match predictions: "
            f"{len(selected)} != {expected_count}"
        )
    return pd.to_numeric(selected[feature_name], errors="coerce").to_numpy(dtype=np.float64)


def _dataset_origin_conditioning_values(
    feature_frame: Optional[pd.DataFrame],
    dataset,
    *,
    feature_name: str,
    horizon: int,
    start_exclusive: int,
    end_inclusive: int,
    expected_count: int,
) -> np.ndarray:
    """Extract condition values aligned directly via TimeSeriesDataSet global ``time_idx`` origins."""
    if feature_frame is None or feature_name not in feature_frame.columns:
        raise RuntimeError(f"Weekly interval conditioning feature is missing: {feature_name}")
    ds_index = getattr(dataset, "index", None)
    if (
        ds_index is not None
        and "time" in ds_index
        and "sequence_length" in ds_index
        and "time_idx" in feature_frame.columns
    ):
        origin_time = (
            ds_index["time"].to_numpy(dtype=int)
            + ds_index["sequence_length"].to_numpy(dtype=int)
            - int(horizon)
            - 1
        )[: int(expected_count)]
        if len(origin_time) == int(expected_count):
            lookup = feature_frame.drop_duplicates(subset=["time_idx"]).set_index("time_idx")[feature_name]
            values = pd.to_numeric(lookup.reindex(origin_time), errors="coerce").to_numpy(
                dtype=np.float64
            )
            if np.isfinite(values).all():
                return values
    return _weekly_interval_conditioning_values(
        feature_frame,
        feature_name=feature_name,
        start_exclusive=start_exclusive,
        end_inclusive=end_inclusive,
        expected_count=expected_count,
    )


def _log_weekly_alignment_sample(
    y_actual_path: np.ndarray,
    pred_np: np.ndarray,
    cfg: TFTASROConfig,
    max_rows: int = 10,
    already_ordered: bool = False,
) -> None:
    from deep_learning.training.metrics import (
        apply_weekly_median_cap_np,
        cumulative_horizon,
        monotonic_quantiles_np,
    )

    pred_np = np.asarray(pred_np)
    y_actual_path = np.asarray(y_actual_path)
    if len(y_actual_path) == 0 or len(pred_np) == 0:
        return

    _validate_quantile_prediction_shape(pred_np, cfg)
    n = min(len(y_actual_path), len(pred_np))
    median_idx = len(cfg.model.quantiles) // 2
    horizon = cfg.forecast.primary_horizon_days
    weekly_actual = cumulative_horizon(y_actual_path[:n], horizon=horizon)
    if already_ordered:
        eval_pred_np, _ = apply_weekly_median_cap_np(
            pred_np[:n],
            weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
            quantiles=cfg.model.quantiles,
            horizon=horizon,
            already_ordered=True,
        )
        ordered_pred_np = eval_pred_np
    else:
        eval_pred_np, _ = apply_weekly_median_cap_np(
            pred_np[:n],
            weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
            quantiles=cfg.model.quantiles,
            horizon=horizon,
        )
        ordered_pred_np = monotonic_quantiles_np(eval_pred_np, median_idx=median_idx)
    weekly_pred = ordered_pred_np[:, :horizon, median_idx].sum(axis=1)

    logger.info("WEEKLY ALIGNMENT SAMPLE:")
    for i in range(min(max_rows, len(weekly_actual))):
        logger.info(
            "sample=%d actual_weekly=%.5f pred_weekly=%.5f actual_sign=%+.0f pred_sign=%+.0f",
            i,
            weekly_actual[i],
            weekly_pred[i],
            np.sign(weekly_actual[i]),
            np.sign(weekly_pred[i]),
        )


def _iso_index_date(value) -> Optional[str]:
    """Convert a DataFrame index label to YYYY-MM-DD when date-like."""
    if value is None or isinstance(value, (int, np.integer)):
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.notna(ts):
            return ts.date().isoformat()
    except Exception:
        return None
    return None


def _compute_split_dates(
    master_df: pd.DataFrame,
    *,
    train_size: int,
    val_size: int,
    train_end_pos: Optional[int] = None,
    val_start_pos: Optional[int] = None,
    val_end_pos: Optional[int] = None,
    test_start_pos: Optional[int] = None,
) -> dict[str, Optional[str]]:
    """Record ISO date boundaries for chronological train / val / test partitions."""
    if master_df.empty:
        return {}
    n = len(master_df)
    tr_end = max(0, min((train_size - 1) if train_end_pos is None else int(train_end_pos), n - 1))
    val_start = max(0, min(train_size if val_start_pos is None else int(val_start_pos), n - 1))
    val_end = max(
        val_start,
        min((train_size + val_size - 1) if val_end_pos is None else int(val_end_pos), n - 1),
    )
    test_start = min((train_size + val_size) if test_start_pos is None else int(test_start_pos), n - 1)
    return {
        "train_start_date": _iso_index_date(master_df.index[0]),
        "train_end_date": _iso_index_date(master_df.index[tr_end]),
        "val_start_date": _iso_index_date(master_df.index[val_start]),
        "val_end_date": _iso_index_date(master_df.index[val_end]),
        "test_start_date": _iso_index_date(master_df.index[test_start]),
        "test_end_date": _iso_index_date(master_df.index[-1]),
    }


def _train_single_tft_seed(
    *,
    training_ds,
    train_dl,
    val_dl,
    cfg: TFTASROConfig,
    seed: int,
    ckpt_dir: Path,
    destination_path: Path,
    use_asro: bool,
    weekly_target_scale: Optional[float],
    allow_soup: bool,
):
    """Train one deterministic TFT seed and promote its best validation checkpoint."""
    import shutil
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    except ImportError:
        import pytorch_lightning as pl  # type: ignore[no-redef]
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint  # type: ignore[no-redef]

    from deep_learning.models.tft_copper import create_tft_model, load_tft_model
    from deep_learning.training.callbacks import (
        CurriculumLossScheduler,
        WeeklyLossComponentLogger,
    )

    pl.seed_everything(int(seed), workers=True)
    model = create_tft_model(
        training_ds,
        cfg,
        use_asro=use_asro,
        weekly_target_scale=weekly_target_scale,
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor_metric = "val_loss"
    callbacks = [
        WeeklyLossComponentLogger(),
        EarlyStopping(
            monitor=monitor_metric,
            patience=cfg.training.early_stopping_patience,
            mode="min",
            verbose=allow_soup,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="tft-asro-{epoch:02d}-{val_loss:.4f}",
            monitor=monitor_metric,
            mode="min",
            save_top_k=3 if allow_soup else 1,
            save_last=allow_soup,
        ),
    ]
    if use_asro and cfg.forecast.primary_horizon_days != 5:
        callbacks.append(CurriculumLossScheduler(
            warmup_epochs=10,
            initial_lambda_quantile=0.65,
            target_lambda_quantile=cfg.asro.lambda_quantile,
            initial_lambda_madl=0.05,
            target_lambda_madl=cfg.asro.lambda_madl,
        ))

    n_batches = len(train_dl)
    log_steps = max(1, min(5, n_batches))
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        gradient_clip_val=cfg.model.gradient_clip_val,
        callbacks=callbacks,
        enable_progress_bar=allow_soup,
        log_every_n_steps=log_steps,
        deterministic=True,
    )
    logger.info("Starting TFT-ASRO training (seed=%d) ...", int(seed))
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_path = trainer.checkpoint_callback.best_model_path
    checkpoint_selection = {
        "fit_split": "validation",
        "monitor": monitor_metric,
        "method": "best_validation_checkpoint",
        "source_count": 0,
        "source_checkpoints": [],
        "source_scores": [],
        "test_labels_used": False,
        "seed": int(seed),
    }
    if best_path:
        source_count = 2 if (allow_soup and use_asro and cfg.forecast.primary_horizon_days == 5) else 1
        ranked_checkpoints = _validation_ranked_checkpoint_paths(
            trainer.checkpoint_callback,
            max_count=source_count,
        )
        source_paths = [path for path, _score in ranked_checkpoints]
        soup_averaged = _build_uniform_checkpoint_soup(source_paths, destination_path)
        if len(source_paths) > 1 and soup_averaged:
            best_score = float(ranked_checkpoints[0][1])
            soup_accepted = False
            try:
                souped_model = load_tft_model(str(destination_path))
                val_results = trainer.validate(souped_model, dataloaders=val_dl, verbose=False)
                soup_score = float((val_results[0] if val_results else {}).get(monitor_metric, float("inf")))
                if math.isfinite(soup_score) and soup_score <= best_score:
                    checkpoint_selection["method"] = "uniform_top2_weight_soup"
                    checkpoint_selection["soup_val_score"] = soup_score
                    soup_accepted = True
                else:
                    logger.info(
                        "Reverting top-2 checkpoint soup (val=%.6f > best=%.6f) -> single best %s",
                        soup_score,
                        best_score,
                        source_paths[0].name,
                    )
            except Exception as exc:
                logger.warning("Checkpoint soup validation failed (%s); keeping single best", exc)

            if not soup_accepted:
                shutil.copy2(source_paths[0], destination_path)
                source_paths = [source_paths[0]]
                ranked_checkpoints = [ranked_checkpoints[0]]
                checkpoint_selection["method"] = "best_validation_checkpoint_soup_reverted"
        elif len(source_paths) > 1 and not soup_averaged:
            source_paths = [source_paths[0]]
            ranked_checkpoints = [ranked_checkpoints[0]]
            checkpoint_selection["method"] = "best_validation_checkpoint_epoch_gap_skipped"
        effective_val_loss = (
            float(checkpoint_selection["soup_val_score"])
            if checkpoint_selection.get("method") == "uniform_top2_weight_soup"
            and "soup_val_score" in checkpoint_selection
            else (float(ranked_checkpoints[0][1]) if ranked_checkpoints else float("inf"))
        )
        checkpoint_selection.update(
            {
                "source_count": len(source_paths),
                "source_checkpoints": [path.name for path in source_paths],
                "source_scores": [score for _path, score in ranked_checkpoints],
                "effective_val_loss": effective_val_loss,
            }
        )
    return model, trainer, best_path, checkpoint_selection


def _evaluate_seed_member_stats(
    pred_val_np: np.ndarray,
    val_actual_path: np.ndarray,
    pred_train_np: np.ndarray,
    cfg: TFTASROConfig,
    *,
    val_loss: float,
    already_ordered: bool = False,
) -> dict:
    """Compute validation skill and train/val drift statistics for one ensemble seed."""
    from deep_learning.models.monotonic_quantiles import aggregate_weekly_quantiles_np
    from deep_learning.training.metrics import (
        apply_weekly_median_cap_np,
        cumulative_horizon,
        monotonic_quantiles_np,
    )

    horizon = int(cfg.forecast.primary_horizon_days)
    quantiles = tuple(cfg.model.quantiles)
    median_idx = quantiles.index(0.50) if 0.50 in quantiles else len(quantiles) // 2

    n_val = min(len(pred_val_np), len(val_actual_path))
    if n_val <= 0:
        return {
            "val_loss": float(val_loss),
            "val_mae_vs_naive": float("inf"),
            "val_pred_mean": 0.0,
            "train_pred_mean": 0.0,
        }

    if already_ordered:
        ord_val, _ = apply_weekly_median_cap_np(
            pred_val_np[:n_val],
            weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
            quantiles=quantiles,
            horizon=horizon,
            already_ordered=True,
        )
    else:
        eval_val, _ = apply_weekly_median_cap_np(
            pred_val_np[:n_val],
            weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
            quantiles=quantiles,
            horizon=horizon,
        )
        ord_val = monotonic_quantiles_np(eval_val, median_idx=median_idx)
    wq_val = aggregate_weekly_quantiles_np(ord_val[:, :horizon, :], median_idx=median_idx)
    val_med = wq_val[:, median_idx]
    actual_w = cumulative_horizon(val_actual_path[:n_val], horizon=horizon)
    naive_mae = float(np.mean(np.abs(actual_w))) if len(actual_w) else 0.0
    val_mae = float(np.mean(np.abs(actual_w - val_med))) if len(actual_w) else float("inf")
    val_mae_ratio = val_mae / (naive_mae + 1e-8) if naive_mae > 1e-12 else float("inf")

    train_med_mean = 0.0
    if len(pred_train_np) > 0:
        if already_ordered:
            ord_tr, _ = apply_weekly_median_cap_np(
                pred_train_np,
                weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
                quantiles=quantiles,
                horizon=horizon,
                already_ordered=True,
            )
        else:
            eval_tr, _ = apply_weekly_median_cap_np(
                pred_train_np,
                weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
                quantiles=quantiles,
                horizon=horizon,
            )
            ord_tr = monotonic_quantiles_np(eval_tr, median_idx=median_idx)
        wq_tr = aggregate_weekly_quantiles_np(ord_tr[:, :horizon, :], median_idx=median_idx)
        train_med_mean = float(np.mean(wq_tr[:, median_idx]))

    return {
        "val_loss": float(val_loss),
        "val_mae_vs_naive": float(val_mae_ratio),
        "val_pred_mean": float(np.mean(val_med)) if len(val_med) else 0.0,
        "train_pred_mean": float(train_med_mean),
    }


def _filter_ensemble_members(
    candidates: list[dict],
    *,
    max_val_mae_vs_naive: float = 1.06,
    max_val_loss_ratio: float = 1.08,
    max_abs_val_pred_mean: float = 0.010,
    max_abs_train_pred_mean: float = 0.008,
) -> tuple[list[dict], list[dict]]:
    """Keep seeds passing validation skill, loss ratio, and train/val drift gates.

    Calibration gates are sized for small (~80-100 bar / ~19 independent weeks)
    chronological validation windows without subtracting validation means. If no
    seed clears all strict gates simultaneously, keeps the top 2 finite
    non-collapsed seeds to preserve multi-seed variance reduction.
    """
    if not candidates:
        return [], []

    finite_losses = [
        float(c.get("val_loss", float("inf")))
        for c in candidates
        if np.isfinite(float(c.get("val_loss", float("inf"))))
    ]
    best_loss = min(finite_losses) if finite_losses else float("inf")
    loss_limit = (
        best_loss * float(max_val_loss_ratio)
        if np.isfinite(best_loss) and best_loss > 0.0
        else (
            best_loss + (float(max_val_loss_ratio) - 1.0) * max(1.0, abs(best_loss))
            if np.isfinite(best_loss)
            else float("inf")
        )
    )

    accepted: list[dict] = []
    diagnostics: list[dict] = []
    for item in candidates:
        v_loss = float(item.get("val_loss", float("inf")))
        v_mae = float(item.get("val_mae_vs_naive", float("inf")))
        v_mean = float(item.get("val_pred_mean", 0.0))
        t_mean = float(item.get("train_pred_mean", 0.0))

        reasons: list[str] = []
        if not np.isfinite(v_mae) or v_mae > float(max_val_mae_vs_naive):
            reasons.append(f"val_mae_vs_naive={v_mae:.4f}>{max_val_mae_vs_naive:.2f}")
        if np.isfinite(loss_limit) and (not np.isfinite(v_loss) or v_loss > loss_limit):
            reasons.append(f"val_loss={v_loss:.4f}>{loss_limit:.4f}")
        if not np.isfinite(v_mean) or abs(v_mean) > float(max_abs_val_pred_mean):
            reasons.append(f"|val_pred_mean|={abs(v_mean):.5f}>{max_abs_val_pred_mean:.4f}")
        if not np.isfinite(t_mean) or abs(t_mean) > float(max_abs_train_pred_mean):
            reasons.append(f"|train_pred_mean|={abs(t_mean):.5f}>{max_abs_train_pred_mean:.4f}")

        ok = len(reasons) == 0
        ckpt_obj = item.get("path")
        diag = {
            "seed": item.get("seed"),
            "checkpoint": Path(ckpt_obj).name if ckpt_obj else None,
            "val_loss": v_loss,
            "val_mae_vs_naive": v_mae,
            "val_pred_mean": v_mean,
            "train_pred_mean": t_mean,
            "accepted": ok,
            "reject_reasons": reasons,
        }
        diagnostics.append(diag)
        if ok:
            accepted.append(item)

    if not accepted:
        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda i: (
                float(candidates[i].get("val_mae_vs_naive", float("inf")))
                if np.isfinite(float(candidates[i].get("val_mae_vs_naive", float("inf"))))
                else float("inf"),
                float(candidates[i].get("val_loss", float("inf")))
                if np.isfinite(float(candidates[i].get("val_loss", float("inf"))))
                else float("inf"),
            ),
        )
        non_degenerate = [
            i
            for i in ranked_indices
            if np.isfinite(float(candidates[i].get("val_mae_vs_naive", float("inf"))))
            and np.isfinite(float(candidates[i].get("val_loss", float("inf"))))
            and float(candidates[i].get("val_mae_vs_naive", float("inf"))) <= 1.25
            and abs(float(candidates[i].get("val_pred_mean", 0.0))) <= 0.03
        ]
        fallback_indices = non_degenerate[:2] if len(non_degenerate) >= 2 else [ranked_indices[0]]
        accepted = [candidates[i] for i in fallback_indices]
        for i in fallback_indices:
            diagnostics[i]["accepted"] = True
            diagnostics[i]["fallback_best"] = True

    return accepted, diagnostics


def train_tft_model(
    cfg: Optional[TFTASROConfig] = None,
    use_asro: bool = True,
    upload_to_hub: bool = False,
    deterministic_weekly_validation: bool = False,
) -> dict:
    """
    End-to-end TFT-ASRO training.

    Returns:
        Dict with metrics, checkpoint path, and feature importance.
    """
    # Configure torch before importing Lightning.  Importing Lightning may
    # initialize torch's inter-op pool, after which set_num_interop_threads()
    # can no longer make the process single-threaded.  The hosted-runner
    # replay showed that leaving that pool at its default produced different
    # validation trajectories for the same seed and data snapshot.
    _configure_tft_reproducibility()

    # pytorch_forecasting >=1.0 uses the unified `lightning` package.
    # Importing from `pytorch_lightning` gives a different LightningModule
    # base class, causing "model must be a LightningModule" at trainer.fit().
    suppress_lightning_noise()
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    except ImportError:
        import pytorch_lightning as pl  # type: ignore[no-redef]
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint  # type: ignore[no-redef]

    from app.db import SessionLocal, init_db
    import deep_learning.data.dataset as _dataset_mod
    from deep_learning.data.feature_store import build_tft_dataframe
    from deep_learning.data.dataset import build_datasets, create_dataloaders
    from deep_learning.models.tft_copper import create_tft_model, get_variable_importance, format_prediction
    from deep_learning.training.metrics import (
        resolve_weekly_median_cap,
        summarize_dataloader_target_scale,
    )
    from deep_learning.training.callbacks import (
        CurriculumLossScheduler,
        WeeklyLossComponentLogger,
    )

    if cfg is None:
        cfg = get_tft_config()

    # ---- 0a. Load training params ----
    # Deterministic validation bypasses Optuna so structural changes can be
    # measured before investing in search.
    if deterministic_weekly_validation:
        cfg = _overlay_training_config(cfg, DETERMINISTIC_WEEKLY_CONFIG)
        logger.info("Using deterministic weekly validation config: %s", DETERMINISTIC_WEEKLY_CONFIG)
    else:
        cfg = _apply_optuna_results(cfg)

    # ---- 0b. ASRO loss sanity check (runs before any training) ----
    try:
        from deep_learning.models.losses import debug_asro_loss_direction
        debug = debug_asro_loss_direction()
        logger.info(
            "ASRO loss direction check: %s | "
            "correct_dir loss=%.4f sharpe=%.4f | "
            "anti_dir loss=%.4f sharpe=%.4f | "
            "zero loss=%.4f sharpe=%.4f",
            debug["diagnostics"],
            debug["results"]["correct_direction"]["loss"],
            debug["results"]["correct_direction"]["strategy_sharpe"],
            debug["results"]["anti_direction"]["loss"],
            debug["results"]["anti_direction"]["strategy_sharpe"],
            debug["results"]["zero_predictions"]["loss"],
            debug["results"]["zero_predictions"]["strategy_sharpe"],
        )
        if not debug["passed"]:
            logger.error("ASRO loss direction check FAILED — stopping training")
            return {"error": "ASRO loss check failed", "debug": debug}
    except Exception as exc:
        logger.warning("Could not run ASRO debug check: %s", exc)

    init_db()
    pl.seed_everything(cfg.training.seed, workers=True)
    logger.info(
        "Experiment protocol | seed=%d deterministic=%s split=chronological train/validation/test",
        cfg.training.seed,
        True,
    )

    # ---- 1. Feature store ----
    logger.info("Building feature store ...")
    with SessionLocal() as session:
        master_df, tv_unknown, tv_known, target_cols, _ = build_tft_dataframe(session, cfg)

    logger.info("Master DataFrame: %d rows x %d cols", *master_df.shape)

    # ---- 2. Datasets ----
    training_ds, validation_ds, test_ds = build_datasets(
        master_df, tv_unknown, tv_known, target_cols, cfg,
    )
    train_dl, val_dl, test_dl = create_dataloaders(training_ds, validation_ds, test_ds, cfg)

    split_bounds_fn = getattr(_dataset_mod, "compute_dataset_split_bounds", None)
    if callable(split_bounds_fn):
        bounds = split_bounds_fn(
            master_df,
            cfg,
            purge_gap=int(cfg.forecast.primary_horizon_days),
        )
        train_cutoff = int(bounds["train_cutoff"])
        val_cutoff = int(bounds["val_cutoff"])
        max_time_idx = int(bounds["max_time_idx"])
        split_dates = _compute_split_dates(
            master_df,
            train_size=int(bounds["train_size"]),
            val_size=int(bounds["val_size"]),
            train_end_pos=int(bounds["train_end_pos"]),
            val_start_pos=int(bounds["val_start_pos"]),
            val_end_pos=int(bounds["val_end_pos"]),
            test_start_pos=int(bounds["test_start_pos"]),
        )
    else:
        n_rows = len(master_df)
        test_size = int(n_rows * cfg.training.test_ratio)
        val_size = int(n_rows * cfg.training.val_ratio)
        train_size = n_rows - val_size - test_size
        train_cutoff = int(master_df["time_idx"].iloc[train_size - 1])
        val_cutoff = int(master_df["time_idx"].iloc[train_size + val_size - 1])
        max_time_idx = int(master_df["time_idx"].iloc[-1])
        split_dates = _compute_split_dates(master_df, train_size=train_size, val_size=val_size)
    weekly_direction_model: dict = {
        "version": 1,
        "enabled": False,
        "reason": "disabled_raw_logit_distortion_removed",
        "fit_split": "train",
        "horizon": int(cfg.forecast.primary_horizon_days),
    }

    train_scale_audit = summarize_dataloader_target_scale(
        train_dl,
        horizon=cfg.forecast.primary_horizon_days,
    )
    val_scale_audit = summarize_dataloader_target_scale(
        val_dl,
        horizon=cfg.forecast.primary_horizon_days,
    )
    weekly_median_cap = resolve_weekly_median_cap(
        train_scale_audit,
        abs_median_multiple=cfg.weekly_loss.weekly_median_cap_abs_median_multiple,
        mean_abs_multiple=cfg.weekly_loss.weekly_median_cap_mean_abs_multiple,
        std_multiple=cfg.weekly_loss.weekly_median_cap_std_multiple,
    )
    cfg = replace(
        cfg,
        weekly_loss=replace(cfg.weekly_loss, weekly_median_cap=weekly_median_cap),
    )
    target_scale_audit = {
        "cap_source": "train_decoder_targets",
        "weekly_median_cap": weekly_median_cap,
        "train": train_scale_audit,
        "validation": val_scale_audit,
    }
    logger.info(
        "Target scale audit train | samples=%d target_std=%.6f weekly_std=%.6f "
        "weekly_abs_median=%.6f cap=%.6f target_scale_present=%s",
        train_scale_audit["target_decoder_samples"],
        train_scale_audit["target_decoder_std"],
        train_scale_audit["actual_weekly_std"],
        train_scale_audit["actual_weekly_abs_median"],
        weekly_median_cap,
        train_scale_audit["target_scale_present"],
    )
    logger.info(
        "Target scale audit val   | samples=%d target_std=%.6f weekly_std=%.6f "
        "weekly_abs_median=%.6f target_scale_present=%s",
        val_scale_audit["target_decoder_samples"],
        val_scale_audit["target_decoder_std"],
        val_scale_audit["actual_weekly_std"],
        val_scale_audit["actual_weekly_abs_median"],
        val_scale_audit["target_scale_present"],
    )

    # ---- 3-6. Multi-seed TFT-ASRO training & validation checkpoint promotion ----
    from deep_learning.calibration.hybrid import (
        apply_theta_hybrid_shift_to_path,
        fit_theta_hybrid_blend,
    )
    from deep_learning.models.tft_copper import load_tft_model

    weekly_target_scale = float(train_scale_audit.get("actual_weekly_std") or 0.0) or None
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    final_path = Path(cfg.training.best_model_path)
    configured_seeds = tuple(getattr(cfg.training, "ensemble_seeds", (cfg.training.seed,)) or (cfg.training.seed,))
    if use_asro and cfg.forecast.primary_horizon_days == 5 and len(configured_seeds) > 1:
        ensemble_seeds = (int(cfg.training.seed),) + tuple(
            int(s) for s in configured_seeds[1:] if int(s) != int(cfg.training.seed)
        )
    else:
        ensemble_seeds = (int(cfg.training.seed),)

    candidate_records: list[dict] = []
    primary_model = None
    primary_trainer = None
    primary_best_path = ""
    checkpoint_selection: dict = {}

    create_train_dl_fn = getattr(_dataset_mod, "create_train_dataloader", None)
    for seed_idx, seed_val in enumerate(ensemble_seeds):
        member_ckpt_dir = ckpt_dir if seed_idx == 0 else ckpt_dir / f"seed_{seed_val}"
        member_dest = final_path if seed_idx == 0 else final_path.parent / f"best_tft_asro_seed{seed_idx}.ckpt"
        seed_train_dl = (
            create_train_dl_fn(training_ds, cfg, seed=int(seed_val))
            if (callable(create_train_dl_fn) and (seed_idx > 0 or int(seed_val) != int(cfg.training.seed)))
            else train_dl
        )
        mdl, trn, bpath, sel = _train_single_tft_seed(
            training_ds=training_ds,
            train_dl=seed_train_dl,
            val_dl=val_dl,
            cfg=cfg,
            seed=seed_val,
            ckpt_dir=member_ckpt_dir,
            destination_path=member_dest,
            use_asro=use_asro,
            weekly_target_scale=weekly_target_scale,
            allow_soup=True,
        )
        if seed_idx == 0:
            primary_model = mdl
            primary_trainer = trn
            primary_best_path = bpath
            checkpoint_selection = sel
            n_batches = len(seed_train_dl)
            total_params = sum(p.numel() for p in primary_model.parameters())
            trainable_params = sum(p.numel() for p in primary_model.parameters() if p.requires_grad)
            logger.info(
                "Model params   | total=%s trainable=%s (batches/epoch=%d, seeds=%s)",
                f"{total_params:,}", f"{trainable_params:,}", n_batches, list(ensemble_seeds),
            )
        loaded_member = mdl
        if bpath and member_dest.exists():
            try:
                loaded_member = load_tft_model(str(member_dest))
            except Exception as exc:
                logger.warning("Could not load ensemble checkpoint %s: %s", member_dest, exc)
        candidate_records.append(
            {
                "seed": int(seed_val),
                "path": member_dest if (bpath and member_dest.exists()) else None,
                "model": loaded_member,
                "trainer": trn,
                "selection": sel,
                "val_loss": float(sel.get("effective_val_loss", float("inf"))),
            }
        )

    model = primary_model
    trainer = primary_trainer
    best_path = str(final_path) if primary_best_path else ""
    ensemble_models = [rec["model"] for rec in candidate_records if rec.get("model") is not None] or [model]
    ensemble_ckpt_paths = [rec["path"] for rec in candidate_records if rec.get("path") is not None]
    evaluation_model = ensemble_models[0]

    # ---- 7. Ensemble quality filter -> Validation-only Hybrid Blend + Split CQR -> Test ----
    direction_calibration = {
        "fit_split": "validation",
        "sample_count": 0,
        "direction_sign_multiplier": 1,
        "weekly_sign_threshold": 0.0,
        "weekly_magnitude_scale": 1.0,
    }
    magnitude_calibration = {
        "fit_split": "validation",
        "sample_count": 0,
        "applied": False,
        "scale": 1.0,
        "weekly_magnitude_scale": 1.0,
    }
    interval_calibration = {
        "fit_split": "validation",
        "sample_count": 0,
        "weekly_interval_scale": 1.0,
        "validation_pi80_coverage": 0.0,
        "target_pi80_coverage": 0.80,
    }
    hybrid_blend: dict = {
        "enabled": False,
        "fit_split": "validation",
        "prior_feature": f"{cfg.feature_store.target_symbol.replace('.', '_').replace('-', '_')}_theta_5d_prior",
        "w_tft": 1.0,
        "w_theta": 0.0,
        "sample_count": 0,
        "reason": "not_evaluated",
    }
    weekly_cqr_params: Optional[dict] = None
    val_actual_path = None
    val_pred_np = None
    ensemble_member_diagnostics: list[dict] = []
    theta_prior_col = hybrid_blend["prior_feature"]

    from deep_learning.calibration.conformal import fit_weekly_cqr
    from deep_learning.models.monotonic_quantiles import aggregate_weekly_quantiles_np
    from deep_learning.training.metrics import (
        apply_weekly_median_cap_np,
        cumulative_horizon,
    )

    train_eval_dl = training_ds.to_dataloader(
        train=False,
        batch_size=cfg.training.batch_size,
        num_workers=0,
    )
    for rec_idx, rec in enumerate(candidate_records):
        if rec_idx == 0:
            m_val_pred, val_actual_path = _predict_quantiles_to_np(
                rec["model"], val_dl, cfg, return_y=True, ordered=True
            )
        else:
            m_val_pred = _predict_quantiles_to_np(
                rec["model"], val_dl, cfg, return_y=False, ordered=True
            )
        m_train_pred = _predict_quantiles_to_np(
            rec["model"], train_eval_dl, cfg, return_y=False, ordered=True
        )
        stats = _evaluate_seed_member_stats(
            m_val_pred,
            val_actual_path if val_actual_path is not None else np.empty((0, cfg.forecast.primary_horizon_days)),
            m_train_pred,
            cfg,
            val_loss=float(rec.get("val_loss", float("inf"))),
            already_ordered=True,
        )
        rec["val_pred_np"] = m_val_pred
        rec.update(stats)

    if len(candidate_records) > 1:
        accepted_records, ensemble_member_diagnostics = _filter_ensemble_members(candidate_records)
        for diag in ensemble_member_diagnostics:
            logger.info(
                "Ensemble seed=%s accepted=%s val_loss=%.4f val_mae/0=%.4f val_mean=%+.5f train_mean=%+.5f reasons=%s",
                diag["seed"],
                diag["accepted"],
                diag["val_loss"],
                diag["val_mae_vs_naive"],
                diag["val_pred_mean"],
                diag["train_pred_mean"],
                diag["reject_reasons"],
            )
    else:
        accepted_records = candidate_records
        if candidate_records:
            rec0 = candidate_records[0]
            ensemble_member_diagnostics = [
                {
                    "seed": rec0.get("seed"),
                    "checkpoint": Path(rec0["path"]).name if rec0.get("path") else None,
                    "val_loss": float(rec0.get("val_loss", float("inf"))),
                    "val_mae_vs_naive": float(rec0.get("val_mae_vs_naive", float("inf"))),
                    "val_pred_mean": float(rec0.get("val_pred_mean", 0.0)),
                    "train_pred_mean": float(rec0.get("train_pred_mean", 0.0)),
                    "accepted": True,
                    "reject_reasons": [],
                }
            ]

    # Reconcile surviving checkpoints cleanly onto contiguous HF Hub slots
    staged_payloads = [
        Path(rec["path"]).read_bytes()
        for rec in accepted_records
        if rec.get("path") is not None and Path(rec["path"]).exists()
    ]
    if len(staged_payloads) == len(accepted_records) and accepted_records:
        for sec_i in range(1, len(ensemble_seeds)):
            sec_path = final_path.parent / f"best_tft_asro_seed{sec_i}.ckpt"
            if sec_path.exists():
                sec_path.unlink()
        reconciled_paths: list[Path] = []
        for slot_i, payload_bytes in enumerate(staged_payloads):
            slot_path = (
                final_path
                if slot_i == 0
                else final_path.parent / f"best_tft_asro_seed{slot_i}.ckpt"
            )
            slot_path.write_bytes(payload_bytes)
            reconciled_paths.append(slot_path)
        ensemble_ckpt_paths = reconciled_paths

    ensemble_models = [rec["model"] for rec in accepted_records if rec.get("model") is not None] or [model]
    evaluation_model = ensemble_models[0]
    if accepted_records and accepted_records[0].get("selection"):
        checkpoint_selection = accepted_records[0]["selection"]
    if accepted_records and accepted_records[0].get("trainer"):
        trainer = accepted_records[0]["trainer"]
    logger.info(
        "Promoted validation checkpoint artifact | method=%s sources=%s scores=%s path=%s ensemble=%d/%d",
        checkpoint_selection.get("method"),
        checkpoint_selection.get("source_checkpoints"),
        checkpoint_selection.get("source_scores"),
        final_path,
        len(ensemble_models),
        len(candidate_records),
    )

    val_pred_members = [rec["val_pred_np"] for rec in accepted_records if rec.get("val_pred_np") is not None]
    val_pred_np = (
        np.mean(np.stack(val_pred_members, axis=0), axis=0)
        if len(val_pred_members) > 1
        else val_pred_members[0]
    )

    if val_actual_path is not None and len(val_actual_path) > 0:
        n_val = min(len(val_actual_path), len(val_pred_np))
        direction_calibration["sample_count"] = int(n_val)
        magnitude_calibration["sample_count"] = int(n_val)
        interval_calibration["sample_count"] = int(n_val)

        val_realized_vol = None
        if WEEKLY_INTERVAL_CONDITIONING_FEATURE in master_df.columns:
            try:
                val_realized_vol = _dataset_origin_conditioning_values(
                    master_df,
                    validation_ds,
                    feature_name=WEEKLY_INTERVAL_CONDITIONING_FEATURE,
                    horizon=cfg.forecast.primary_horizon_days,
                    start_exclusive=train_cutoff - 1,
                    end_inclusive=val_cutoff - cfg.forecast.primary_horizon_days,
                    expected_count=n_val,
                )
            except Exception as vol_exc:
                logger.debug("Validation realized_vol lookup skipped: %s", vol_exc)

        val_theta_prior = None
        if theta_prior_col in master_df.columns:
            try:
                val_theta_prior = _dataset_origin_conditioning_values(
                    master_df,
                    validation_ds,
                    feature_name=theta_prior_col,
                    horizon=cfg.forecast.primary_horizon_days,
                    start_exclusive=train_cutoff - 1,
                    end_inclusive=val_cutoff - cfg.forecast.primary_horizon_days,
                    expected_count=n_val,
                )
            except Exception as th_exc:
                logger.debug("Validation Theta prior lookup skipped: %s", th_exc)

        median_idx = len(cfg.model.quantiles) // 2
        val_ordered, _ = apply_weekly_median_cap_np(
            val_pred_np[:n_val],
            weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
            quantiles=tuple(cfg.model.quantiles),
            horizon=cfg.forecast.primary_horizon_days,
            already_ordered=True,
        )
        val_weekly_q = aggregate_weekly_quantiles_np(
            val_ordered[:, : cfg.forecast.primary_horizon_days, :],
            median_idx=median_idx,
        )
        val_weekly_actual = cumulative_horizon(
            val_actual_path[:n_val],
            horizon=cfg.forecast.primary_horizon_days,
        )

        if val_theta_prior is not None:
            hybrid_blend = fit_theta_hybrid_blend(
                val_weekly_actual,
                val_weekly_q[:, median_idx],
                val_theta_prior[:n_val],
                horizon=cfg.forecast.primary_horizon_days,
            )
            if hybrid_blend.get("enabled"):
                val_pred_np[:n_val] = apply_theta_hybrid_shift_to_path(
                    val_pred_np[:n_val],
                    val_theta_prior[:n_val],
                    hybrid_blend=hybrid_blend,
                    quantiles=tuple(cfg.model.quantiles),
                    horizon=cfg.forecast.primary_horizon_days,
                    already_ordered=True,
                )
                val_ordered, _ = apply_weekly_median_cap_np(
                    val_pred_np[:n_val],
                    weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
                    quantiles=tuple(cfg.model.quantiles),
                    horizon=cfg.forecast.primary_horizon_days,
                    already_ordered=True,
                )
                val_weekly_q = aggregate_weekly_quantiles_np(
                    val_ordered[:, : cfg.forecast.primary_horizon_days, :],
                    median_idx=median_idx,
                )

        weekly_cqr_params = fit_weekly_cqr(
            val_weekly_actual,
            val_weekly_q,
            quantiles=tuple(cfg.model.quantiles),
            realized_vol=val_realized_vol,
        )
        interval_calibration.update(
            {
                "cqr_enabled": bool(weekly_cqr_params.get("cqr_enabled", False)),
                "q_hat_80": float(weekly_cqr_params.get("q_hat_80", 0.0) or 0.0),
                "q_hat_96": float(weekly_cqr_params.get("q_hat_96", 0.0) or 0.0),
                "vol_normalized": bool(weekly_cqr_params.get("vol_normalized", False)),
                "base_vol_power": float(weekly_cqr_params.get("base_vol_power", 1.0) or 1.0),
                "n_eff": float(weekly_cqr_params.get("n_eff", 0.0) or 0.0),
                "validation_pi80_coverage": float(
                    weekly_cqr_params.get("validation_pi80_coverage_calibrated", 0.0) or 0.0
                ),
            }
        )

    direction_sign_multiplier = 1
    daily_sign_multiplier = 1
    weekly_sign_threshold = 0.0
    weekly_magnitude_scale = 1.0
    weekly_interval_scale = 1.0
    logger.info("Validation-only Hybrid Theta blend: %s", hybrid_blend)
    logger.info("Validation-only Split CQR calibration: %s", weekly_cqr_params or interval_calibration)

    # ---- 8. Evaluate on the promoted ensemble (in ordered quantile space) ----
    test_metrics = {}
    if test_dl is not None:
        test_pred_members = []
        pred_0, y_actual_path = _predict_quantiles_to_np(
            ensemble_models[0], test_dl, cfg, return_y=True, ordered=True
        )
        test_pred_members.append(pred_0)
        for extra_mdl in ensemble_models[1:]:
            test_pred_members.append(
                _predict_quantiles_to_np(extra_mdl, test_dl, cfg, return_y=False, ordered=True)
            )
        pred_np = (
            np.mean(np.stack(test_pred_members, axis=0), axis=0)
            if len(test_pred_members) > 1
            else pred_0
        )
        ensemble_size = len(ensemble_models)
        logger.info("Promoted ensemble evaluation: %d model(s)", ensemble_size)

        if hybrid_blend.get("enabled") and theta_prior_col in master_df.columns:
            try:
                test_theta_prior = _dataset_origin_conditioning_values(
                    master_df,
                    test_ds,
                    feature_name=theta_prior_col,
                    horizon=cfg.forecast.primary_horizon_days,
                    start_exclusive=val_cutoff - 1,
                    end_inclusive=max_time_idx - cfg.forecast.primary_horizon_days,
                    expected_count=len(pred_np),
                )
                pred_np = apply_theta_hybrid_shift_to_path(
                    pred_np,
                    test_theta_prior,
                    hybrid_blend=hybrid_blend,
                    quantiles=tuple(cfg.model.quantiles),
                    horizon=cfg.forecast.primary_horizon_days,
                    already_ordered=True,
                )
            except Exception as th_test_exc:
                logger.warning("Test Theta hybrid shift skipped: %s", th_test_exc)

        test_realized_vol = None
        if WEEKLY_INTERVAL_CONDITIONING_FEATURE in master_df.columns:
            try:
                test_realized_vol = _dataset_origin_conditioning_values(
                    master_df,
                    test_ds,
                    feature_name=WEEKLY_INTERVAL_CONDITIONING_FEATURE,
                    horizon=cfg.forecast.primary_horizon_days,
                    start_exclusive=val_cutoff - 1,
                    end_inclusive=max_time_idx - cfg.forecast.primary_horizon_days,
                    expected_count=len(pred_np),
                )
            except Exception as vol_exc:
                logger.debug("Test realized_vol lookup skipped: %s", vol_exc)

        test_metrics = _compute_test_metrics_from_quantiles(
            y_actual_path,
            pred_np,
            cfg,
            weekly_cqr_params=weekly_cqr_params,
            realized_vol=test_realized_vol,
            already_ordered=True,
        )
        test_metrics["ensemble_size"] = ensemble_size
        logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in test_metrics.items()})

    _require_promotable_metrics(test_metrics)

    calibration_artifact = _write_conformal_calibration_artifact(
        cfg=cfg,
        model=evaluation_model,
        val_dl=val_dl,
        feature_frame=master_df,
        direction_sign_multiplier=direction_sign_multiplier,
        daily_sign_multiplier=daily_sign_multiplier,
        weekly_sign_threshold=weekly_sign_threshold,
        weekly_magnitude_scale=weekly_magnitude_scale,
        weekly_direction_model=weekly_direction_model,
        validation_start_time=train_cutoff - 1,
        validation_end_time=val_cutoff - cfg.forecast.primary_horizon_days,
        weekly_interval_scale=weekly_interval_scale,
        weekly_interval_calibration=interval_calibration,
        weekly_cqr_params=weekly_cqr_params,
        precomputed_val_pred_np=val_pred_np,
        precomputed_val_actual_path=val_actual_path,
        already_ordered=True,
    )

    # ---- 8. Variable importance ----
    var_importance = get_variable_importance(evaluation_model, val_dataloader=val_dl)

    # ---- 9. Persist metadata ----
    result = {
        "checkpoint_path": str(final_path),
        "ensemble_checkpoints": [p.name for p in ensemble_ckpt_paths] if ensemble_ckpt_paths else [final_path.name],
        "ensemble_size": len(ensemble_models) if ensemble_models else 1,
        "ensemble_member_diagnostics": ensemble_member_diagnostics,
        "hybrid_blend": hybrid_blend,
        "split_dates": split_dates,
        "test_metrics": test_metrics,
        "variable_importance": var_importance,
        "config": {
            "hidden_size": cfg.model.hidden_size,
            "attention_head_size": cfg.model.attention_head_size,
            "dropout": cfg.model.dropout,
            "quantiles": list(cfg.model.quantiles),
            "use_asro": use_asro,
            "lambda_vol": cfg.asro.lambda_vol,
            "lambda_quantile": cfg.asro.lambda_quantile,
            "lambda_madl": cfg.asro.lambda_madl,
            "lambda_crossing": cfg.asro.lambda_crossing,
            "lambda_weekly_quantile": cfg.weekly_loss.lambda_weekly_quantile,
            "lambda_t1_quantile": cfg.weekly_loss.lambda_t1_quantile,
            "lambda_subhorizon_quantile": cfg.weekly_loss.lambda_subhorizon_quantile,
            "lambda_t1_directional": cfg.weekly_loss.lambda_t1_directional,
            "lambda_dispersion": cfg.weekly_loss.lambda_dispersion,
            "lambda_magnitude": cfg.weekly_loss.lambda_magnitude,
            "lambda_naive": cfg.weekly_loss.lambda_naive,
            "lambda_bias": cfg.weekly_loss.lambda_bias,
            "lambda_drift_prior": cfg.weekly_loss.lambda_drift_prior,
            "lambda_directional": cfg.weekly_loss.lambda_directional,
            "lambda_saturation": cfg.weekly_loss.lambda_saturation,
            "lambda_positive_rate": cfg.weekly_loss.lambda_positive_rate,
            "lambda_interval": cfg.weekly_loss.lambda_interval,
            "weekly_median_cap_abs_median_multiple": (
                cfg.weekly_loss.weekly_median_cap_abs_median_multiple
            ),
            "weekly_median_cap_mean_abs_multiple": (
                cfg.weekly_loss.weekly_median_cap_mean_abs_multiple
            ),
            "weekly_median_cap_std_multiple": cfg.weekly_loss.weekly_median_cap_std_multiple,
            "weekly_median_cap": cfg.weekly_loss.weekly_median_cap,
            "monotonic_quantile_transform": True,
            "max_encoder_length": cfg.model.max_encoder_length,
            "max_prediction_length": cfg.model.max_prediction_length,
            "mrmr_top_k": cfg.feature_store.mrmr_top_k,
            "forecast_contract_version": FORECAST_CONTRACT_VERSION,
            "target_return_type": TARGET_RETURN_TYPE,
            "primary_horizon_days": cfg.forecast.primary_horizon_days,
            "public_return_space": PUBLIC_RETURN_SPACE,
            "return_space": RETURN_SPACE,
        },
        "forecast_contract_version": FORECAST_CONTRACT_VERSION,
        "target_return_type": TARGET_RETURN_TYPE,
        "primary_horizon_days": cfg.forecast.primary_horizon_days,
        "public_return_space": PUBLIC_RETURN_SPACE,
        "return_space": RETURN_SPACE,
        "target_scale_audit": target_scale_audit,
        "data_snapshot": _build_data_snapshot_metadata(master_df),
        "runtime_environment": _runtime_environment_metadata(),
        "direction_calibration": direction_calibration,
        "magnitude_calibration": magnitude_calibration,
        "weekly_direction_model": weekly_direction_model,
        "interval_calibration": interval_calibration,
        "checkpoint_selection": checkpoint_selection,
        "experiment": {
            "seed": cfg.training.seed,
            "deterministic": True,
            "split_protocol": "chronological train/validation/test; validation-only calibration; untouched test",
        },
        "conformal_calibration_path": str(calibration_artifact) if calibration_artifact else None,
        "n_unknown_features": len(tv_unknown),
        "n_known_features": len(tv_known),
        "train_samples": len(training_ds),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write metadata JSON to disk for CI quality gate
    meta_json_path = Path(cfg.training.best_model_path).parent / "tft_metadata.json"
    try:
        result["artifact_manifest_path"] = str(meta_json_path.parent / "artifact_manifest.json")
        meta_json_path.write_text(json.dumps(result, indent=2, default=str))
        logger.info("Training metadata written to %s", meta_json_path)
        try:
            from deep_learning.models.hub import write_artifact_manifest

            manifest_path = write_artifact_manifest(meta_json_path.parent)
            result["artifact_manifest_path"] = str(manifest_path)
            logger.info("Artifact manifest written to %s", manifest_path)
        except Exception as exc:
            logger.warning("Could not write artifact manifest: %s", exc)
    except Exception as exc:
        logger.warning("Could not write metadata JSON: %s", exc)

    # ---- 10. Optional HF Hub upload ----
    # CI promotes artifacts only after scripts/tft_quality_gate.py passes.
    # Keeping upload disabled by default prevents a failed model from replacing
    # the production checkpoint before the gate has evaluated test metrics.
    result["hub_uploaded"] = False
    if upload_to_hub:
        uploaded = False
        try:
            from deep_learning.models.hub import upload_tft_artifacts

            tft_dir = final_path.parent
            uploaded = upload_tft_artifacts(
                local_dir=tft_dir,
                repo_id=cfg.training.hf_model_repo,
                commit_message=f"TFT-ASRO checkpoint (val_loss={trainer.checkpoint_callback.best_model_score:.4f})"
                if trainer.checkpoint_callback.best_model_score
                else "TFT-ASRO checkpoint",
            )
            result["hub_uploaded"] = uploaded
        except Exception as exc:
            logger.warning("HF Hub upload skipped: %s", exc)
        if uploaded:
            persist_promoted_tft_metadata(cfg.feature_store.target_symbol, result)
            result["promotion_metadata_persisted"] = True
    else:
        result["hub_upload_skipped"] = "disabled_until_quality_gate_passes"

    return result


def _write_conformal_calibration_artifact(
    *,
    cfg: TFTASROConfig,
    model,
    val_dl,
    feature_frame,
    direction_sign_multiplier: int = 1,
    daily_sign_multiplier: int = 1,
    weekly_sign_threshold: float = 0.0,
    weekly_magnitude_scale: float = 1.0,
    weekly_direction_model: Optional[dict] = None,
    validation_start_time: Optional[int] = None,
    validation_end_time: Optional[int] = None,
    weekly_interval_scale: float = 1.0,
    weekly_interval_calibration: Optional[dict] = None,
    weekly_cqr_params: Optional[dict] = None,
    precomputed_val_pred_np: Optional[np.ndarray] = None,
    precomputed_val_actual_path: Optional[np.ndarray] = None,
    already_ordered: bool = False,
) -> Optional[Path]:
    """
    Fit interval adjustment on validation/calibration data, never final test.

    Writes both Split / Volatility-Normalized CQR parameters and backward-compatible
    scalar conformal keys so production inference and legacy consumers agree.
    """
    try:
        from deep_learning.calibration.conformal import (
            apply_conformal_interval,
            fit_weekly_cqr,
            interval_coverage,
            rolling_conformal_adjustment,
        )
        from deep_learning.models.monotonic_quantiles import aggregate_weekly_quantiles_np
        from deep_learning.training.metrics import (
            apply_direction_sign_multiplier_np,
            apply_weekly_magnitude_scale_np,
            apply_weekly_sign_correction_np,
            apply_weekly_median_cap_np,
            apply_weekly_interval_scale_np,
            cumulative_horizon,
            monotonic_quantiles_np,
        )

        # Reuse precomputed ensemble + hybrid-shifted validation tensors when supplied
        if precomputed_val_pred_np is not None and precomputed_val_actual_path is not None:
            pred_np = np.asarray(precomputed_val_pred_np, dtype=np.float64)
            y_actual_path = np.asarray(precomputed_val_actual_path, dtype=np.float64)
        else:
            pred_np, y_actual_path = _predict_quantiles_to_np(
                model, val_dl, cfg, return_y=True, ordered=already_ordered
            )
        if y_actual_path is None or len(y_actual_path) == 0:
            return None
        pred_np = apply_direction_sign_multiplier_np(pred_np, int(direction_sign_multiplier))
        pred_np = apply_weekly_sign_correction_np(
            pred_np,
            float(weekly_sign_threshold),
            horizon=cfg.forecast.primary_horizon_days,
        )
        if weekly_direction_model and weekly_direction_model.get("enabled"):
            from deep_learning.training.direction_model import (
                apply_weekly_direction_model,
                predict_weekly_direction,
            )

            if validation_start_time is None or validation_end_time is None:
                raise ValueError("Validation time bounds are required for weekly direction calibration")
            validation_probability = predict_weekly_direction(
                weekly_direction_model,
                feature_frame,
                start_exclusive=validation_start_time,
                end_inclusive=validation_end_time,
            )
            if len(validation_probability) != len(pred_np):
                raise ValueError(
                    "Weekly direction model origin count does not match validation predictions"
                )
            pred_np = apply_weekly_direction_model(
                pred_np,
                validation_probability,
                threshold=float(weekly_direction_model.get("decision_threshold", 0.50)),
                horizon=cfg.forecast.primary_horizon_days,
            )
        pred_np = apply_weekly_magnitude_scale_np(
            pred_np,
            float(weekly_magnitude_scale),
            quantiles=tuple(cfg.model.quantiles),
        )
        interval_calibration = weekly_interval_calibration or {}
        effective_interval_scale = float(
            interval_calibration.get("weekly_interval_scale", weekly_interval_scale)
        )
        interval_kwargs = {
            "quantiles": tuple(cfg.model.quantiles),
        }
        val_realized_vol = None
        if (
            feature_frame is not None
            and WEEKLY_INTERVAL_CONDITIONING_FEATURE in feature_frame.columns
            and validation_start_time is not None
            and validation_end_time is not None
        ):
            try:
                val_realized_vol = _dataset_origin_conditioning_values(
                    feature_frame,
                    getattr(val_dl, "dataset", None),
                    feature_name=WEEKLY_INTERVAL_CONDITIONING_FEATURE,
                    horizon=cfg.forecast.primary_horizon_days,
                    start_exclusive=validation_start_time,
                    end_inclusive=validation_end_time,
                    expected_count=len(pred_np),
                )
            except Exception:
                val_realized_vol = None
        if interval_calibration.get("weekly_interval_conditioning_enabled"):
            if val_realized_vol is None:
                raise ValueError(
                    "Validation realized volatility values missing for conditioning"
                )
            interval_kwargs.update(
                {
                    "conditioning_values": val_realized_vol,
                    "conditioning_reference": interval_calibration.get(
                        "weekly_interval_conditioning_reference"
                    ),
                    "conditioning_power": interval_calibration.get(
                        "weekly_interval_conditioning_power", 0.35
                    ),
                    "conditioning_min_factor": interval_calibration.get(
                        "weekly_interval_conditioning_min_factor", 0.5
                    ),
                    "conditioning_max_factor": interval_calibration.get(
                        "weekly_interval_conditioning_max_factor", 2.0
                    ),
                    "conditioning_min_scale": interval_calibration.get(
                        "weekly_interval_conditioning_min_scale", 0.20
                    ),
                    "conditioning_max_scale": interval_calibration.get(
                        "weekly_interval_conditioning_max_scale", 2.50
                    ),
                }
            )
        pred_np = apply_weekly_interval_scale_np(
            pred_np,
            effective_interval_scale,
            **interval_kwargs,
        )
        n = min(len(y_actual_path), len(pred_np))
        if n <= 0:
            return None

        weekly_actual = cumulative_horizon(y_actual_path[:n], horizon=cfg.forecast.primary_horizon_days)
        if already_ordered:
            ordered_pred_np, _ = apply_weekly_median_cap_np(
                pred_np[:n],
                weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
                quantiles=cfg.model.quantiles,
                horizon=cfg.forecast.primary_horizon_days,
                already_ordered=True,
            )
        else:
            eval_pred_np, _ = apply_weekly_median_cap_np(
                pred_np[:n],
                weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
                quantiles=cfg.model.quantiles,
                horizon=cfg.forecast.primary_horizon_days,
            )
            ordered_pred_np = monotonic_quantiles_np(
                eval_pred_np,
                median_idx=len(cfg.model.quantiles) // 2,
            )
        weekly_quantiles = aggregate_weekly_quantiles_np(
            ordered_pred_np[:, : cfg.forecast.primary_horizon_days, :],
            median_idx=len(cfg.model.quantiles) // 2,
        )
        cqr_data = weekly_cqr_params or fit_weekly_cqr(
            weekly_actual,
            weekly_quantiles,
            quantiles=tuple(cfg.model.quantiles),
            realized_vol=val_realized_vol[:n] if val_realized_vol is not None else None,
        )
        q = tuple(cfg.model.quantiles)
        q10_idx = q.index(0.10)
        q90_idx = q.index(0.90)
        raw_lower = weekly_quantiles[:, q10_idx]
        raw_upper = weekly_quantiles[:, q90_idx]
        validation_pi80_coverage = interval_coverage(weekly_actual, raw_lower, raw_upper)
        validation_pi80_width = float(np.mean(raw_upper - raw_lower))
        validation_weekly_std = float(np.std(weekly_actual))
        target_min_pi80_width_ratio = 0.70
        validation_pi80_width_ratio = (
            validation_pi80_width / (2.56 * validation_weekly_std + 1e-8)
            if validation_weekly_std > 1e-12
            else 0.0
        )
        target_min_pi80_width = target_min_pi80_width_ratio * 2.56 * validation_weekly_std
        width_floor_adjustment = max(
            0.0,
            (target_min_pi80_width - validation_pi80_width) / 2.0,
        )

        from app.state_machines import TFTCalibrationStatus

        calibration_status = TFTCalibrationStatus.FIT.value
        if validation_pi80_coverage >= 0.90:
            conformal_adj = 0.0
            calibration_status = TFTCalibrationStatus.SKIPPED_OVERCOVERED.value
        else:
            conformal_adj = rolling_conformal_adjustment(
                weekly_actual,
                raw_lower,
                raw_upper,
                alpha=0.20,
            )
        global_adj = max(float(conformal_adj), float(width_floor_adjustment))
        if width_floor_adjustment > conformal_adj and width_floor_adjustment > 0.0:
            calibration_status = TFTCalibrationStatus.FIT_WIDTH_FLOOR.value
        calibrated_lower, calibrated_upper = apply_conformal_interval(
            raw_lower,
            raw_upper,
            global_adj,
        )
        calibrated_validation_pi80_coverage = interval_coverage(
            weekly_actual,
            calibrated_lower,
            calibrated_upper,
        )
        calibrated_validation_pi80_width = float(np.mean(calibrated_upper - calibrated_lower))
        calibrated_validation_pi80_width_ratio = (
            calibrated_validation_pi80_width / (2.56 * validation_weekly_std + 1e-8)
            if validation_weekly_std > 1e-12
            else 0.0
        )
        artifact = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target": "weekly_5d_log_return",
            "alpha": 0.20,
            "cqr_enabled": bool(cqr_data.get("cqr_enabled", False)),
            "method": cqr_data.get("method", "split_cqr_vol_normalized"),
            "q_hat_80": float(cqr_data.get("q_hat_80", 0.0) or 0.0),
            "q_hat_96": float(cqr_data.get("q_hat_96", 0.0) or 0.0),
            "median_val_vol": cqr_data.get("median_val_vol"),
            "vol_normalized": bool(cqr_data.get("vol_normalized", False)),
            "base_vol_power": (
                float(cqr_data["base_vol_power"])
                if cqr_data.get("base_vol_power") is not None
                else 1.0
            ),
            "n_eff": cqr_data.get("n_eff"),
            "overlap_horizon": int(
                cqr_data.get("overlap_horizon", cfg.forecast.primary_horizon_days)
                or cfg.forecast.primary_horizon_days
            ),
            "min_half_width_retention": float(cqr_data.get("min_half_width_retention", 0.35) or 0.35),
            "global_adjustment": float(global_adj),
            "base_conformal_adjustment": float(conformal_adj),
            "width_floor_adjustment": float(width_floor_adjustment),
            "bucket_adjustments": {
                "neutral": float(global_adj),
                "risk_on": float(global_adj),
                "usd_pressure": float(global_adj),
                "supply_shock": float(global_adj),
                "high_vol": float(global_adj),
            },
            "min_bucket_samples": 30,
            "window": int(min(252, n)),
            "fit_split": "validation",
            "test_split_used_for_fit": False,
            "direction_sign_multiplier": int(direction_sign_multiplier),
            "daily_sign_multiplier": int(daily_sign_multiplier),
            "weekly_sign_threshold": float(weekly_sign_threshold),
            "weekly_interval_scale": float(effective_interval_scale),
            "weekly_interval_conditioning_enabled": bool(
                interval_calibration.get("weekly_interval_conditioning_enabled", False)
            ),
            "weekly_interval_conditioning_feature": interval_calibration.get(
                "weekly_interval_conditioning_feature"
            ),
            "weekly_interval_conditioning_reference": interval_calibration.get(
                "weekly_interval_conditioning_reference"
            ),
            "weekly_interval_conditioning_power": interval_calibration.get(
                "weekly_interval_conditioning_power", 0.35
            ),
            "weekly_interval_conditioning_min_factor": interval_calibration.get(
                "weekly_interval_conditioning_min_factor", 0.5
            ),
            "weekly_interval_conditioning_max_factor": interval_calibration.get(
                "weekly_interval_conditioning_max_factor", 2.0
            ),
            "weekly_interval_conditioning_min_scale": interval_calibration.get(
                "weekly_interval_conditioning_min_scale", 0.20
            ),
            "weekly_interval_conditioning_max_scale": interval_calibration.get(
                "weekly_interval_conditioning_max_scale", 2.50
            ),
            "validation_pi80_coverage": validation_pi80_coverage,
            "calibrated_validation_pi80_coverage": calibrated_validation_pi80_coverage,
            "cqr_validation_pi80_coverage": float(
                cqr_data.get("validation_pi80_coverage_calibrated", validation_pi80_coverage) or 0.0
            ),
            "cqr_validation_pi96_coverage": float(
                cqr_data.get("validation_pi96_coverage_calibrated", 0.0) or 0.0
            ),
            "validation_pi80_width": validation_pi80_width,
            "calibrated_validation_pi80_width": calibrated_validation_pi80_width,
            "validation_pi80_width_ratio": validation_pi80_width_ratio,
            "calibrated_validation_pi80_width_ratio": calibrated_validation_pi80_width_ratio,
            "target_min_pi80_width_ratio": target_min_pi80_width_ratio,
            "calibration_status": calibration_status,
        }

        calibration_path = Path(cfg.training.best_model_path).parent / "conformal_calibration.json"
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        logger.info("Conformal calibration written to %s", calibration_path)
        return calibration_path
    except Exception as exc:
        logger.warning("Could not write conformal calibration artifact: %s", exc)
        return None


def _apply_optuna_results(cfg: TFTASROConfig) -> TFTASROConfig:
    """
    If an optuna_results.json exists in the checkpoint directory, overlay the
    best hyperparameters onto cfg and return the updated config.  This connects
    the hyperopt step to the final training run so search results are not wasted.
    """
    results_path = Path(cfg.training.best_model_path).parent / "optuna_results.json"
    if not results_path.exists():
        logger.info(
            "Optuna results not found at %s; using known-good fallback config: %s",
            results_path,
            KNOWN_GOOD_CONFIG,
        )
        return _overlay_training_config(cfg, KNOWN_GOOD_CONFIG)

    try:
        data = json.loads(results_path.read_text())
        params = data.get("best_params", {})
        structural_report = data.get("structural_invalidity_report") or {}
        preflight = data.get("best_trial_preflight") or {}
        if (
            structural_report.get("verdict") == "STRUCTURAL_FAILURE"
            or preflight.get("preflight_passed") is False
        ):
            logger.warning(
                "Optuna results at %s failed structural preflight "
                "(verdict=%s, best_trial=%s); using known-good fallback config: %s",
                results_path,
                structural_report.get("verdict", "unknown"),
                data.get("best_trial", "unknown"),
                KNOWN_GOOD_CONFIG,
            )
            return _overlay_training_config(cfg, KNOWN_GOOD_CONFIG)

        if not params:
            logger.warning(
                "Optuna results did not contain finite best params (status=%s); "
                "using known-good fallback config: %s",
                data.get("status", "unknown"),
                KNOWN_GOOD_CONFIG,
            )
            return _overlay_training_config(cfg, KNOWN_GOOD_CONFIG)

        params = dict(params)
        if "max_encoder_length" in params:
            params["max_encoder_length"] = min(max(int(params["max_encoder_length"]), 10), 90)
        if "hidden_size" in params:
            params["hidden_size"] = min(max(int(params["hidden_size"]), 8), 128)
        if "hidden_continuous_size" in params:
            params["hidden_continuous_size"] = min(max(int(params["hidden_continuous_size"]), 4), 64)
        if "attention_head_size" in params:
            params["attention_head_size"] = min(max(int(params["attention_head_size"]), 1), 8)
        if "dropout" in params:
            params["dropout"] = min(max(float(params["dropout"]), 0.05), 0.60)
        if "learning_rate" in params:
            params["learning_rate"] = min(max(float(params["learning_rate"]), 1e-5), 1e-2)
        if "output_head_lr_mult" in params:
            params["output_head_lr_mult"] = min(max(float(params["output_head_lr_mult"]), 0.05), 1.0)
        if "weight_decay" in params:
            params["weight_decay"] = min(max(float(params["weight_decay"]), 0.0), 5e-2)
        if "batch_size" in params:
            params["batch_size"] = min(max(int(params["batch_size"]), 8), 128)

        controlled_params = dict(KNOWN_GOOD_CONFIG)
        controlled_params.update({
            key: params[key]
            for key in CONTROLLED_WEEKLY_OPTUNA_PARAMS
            if key in params
        })

        logger.info(
            "Loaded controlled Optuna weekly params (trial #%d, weekly_objective=%.4f): %s",
            data.get("best_trial", -1),
            data.get("best_value", float("nan")),
            controlled_params,
        )
        return _overlay_training_config(cfg, controlled_params)

    except Exception as exc:
        logger.warning("Could not apply Optuna results: %s", exc)
        return _overlay_training_config(cfg, KNOWN_GOOD_CONFIG)


def _overlay_training_config(cfg: TFTASROConfig, params: dict) -> TFTASROConfig:
    """Overlay model, ASRO and batch-size params onto a TFT-ASRO config."""
    model_overrides = {
        k: params[k] for k in (
            "hidden_size", "attention_head_size", "dropout",
            "hidden_continuous_size", "learning_rate", "output_head_lr_mult",
            "gradient_clip_val", "max_encoder_length",
            "weight_decay",
        ) if k in params
    }
    asro_overrides = {
        k: params[k] for k in (
            "lambda_vol", "lambda_quantile", "lambda_madl", "lambda_crossing",
        ) if k in params
    }
    training_overrides = {
        k: params[k] for k in ("batch_size",) if k in params
    }
    feature_store_overrides = {
        k: int(params[k]) for k in ("mrmr_top_k",) if k in params
    }
    weekly_loss_overrides = {
        k: params[k] for k in (
            "lambda_weekly_quantile", "lambda_t1_quantile", "lambda_subhorizon_quantile",
            "lambda_t1_directional", "lambda_directional", "lambda_dispersion",
            "lambda_magnitude", "lambda_naive", "lambda_bias", "lambda_drift_prior",
            "lambda_saturation", "lambda_positive_rate", "lambda_interval",
            "weekly_median_cap_abs_median_multiple",
            "weekly_median_cap_mean_abs_multiple", "weekly_median_cap_std_multiple",
        ) if k in params
    }

    new_model = replace(cfg.model, **model_overrides) if model_overrides else cfg.model
    new_asro = replace(cfg.asro, **asro_overrides) if asro_overrides else cfg.asro
    new_weekly_loss = (
        replace(cfg.weekly_loss, **weekly_loss_overrides)
        if weekly_loss_overrides
        else cfg.weekly_loss
    )
    new_training = replace(cfg.training, **training_overrides) if training_overrides else cfg.training
    new_feature_store = (
        replace(cfg.feature_store, **feature_store_overrides)
        if feature_store_overrides
        else cfg.feature_store
    )
    return replace(
        cfg,
        model=new_model,
        asro=new_asro,
        weekly_loss=new_weekly_loss,
        training=new_training,
        feature_store=new_feature_store,
    )


def _importance_metadata_payload(result: dict) -> str:
    """Serialize VSN encoder importance for informational storage at promotion.

    Never let metadata-only concerns block promotion: missing or malformed
    importance degrades to an empty encoder map. Non-finite weights (exploding
    checkpoints can emit NaN softmax weights) are dropped — ``json.dumps``
    would otherwise emit ``NaN``/``Infinity`` literals, which are invalid JSON
    and choke strict readers of the summary endpoint.
    """
    importance = result.get("variable_importance")
    if not isinstance(importance, dict):
        importance = {}
    cleaned_importance = {}
    for name, value in importance.items():
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(weight):
            cleaned_importance[name] = weight
    return json.dumps(
        {
            "schema_version": 1,
            "encoder": cleaned_importance,
            "trained_at": result.get("trained_at"),
            "checkpoint": result.get("checkpoint_path"),
        }
    )


def persist_promoted_tft_metadata(symbol: str, result: dict) -> None:
    """Persist the active model only after gate and Hub promotion succeed.

    Candidate training must not overwrite the single active row. The caller is
    expected to invoke this after the artifact upload returns successfully;
    the gate is deliberately re-evaluated here so a workflow ordering mistake
    still fails closed.
    """
    from app.db import SessionLocal, ensure_tft_model_metadata_schema
    from app.models import TFTModelMetadata
    from app.quality_gate import evaluate_quality_gate_metrics

    metrics = result.get("test_metrics") or {}
    gate_passed, reasons = evaluate_quality_gate_metrics(metrics)
    if not gate_passed:
        raise ValueError(
            "Refusing to persist rejected TFT candidate as active: "
            + "; ".join(reasons)
        )

    ensure_tft_model_metadata_schema()

    trained_at = datetime.now(timezone.utc)
    raw_trained_at = result.get("trained_at")
    if isinstance(raw_trained_at, str):
        try:
            trained_at = datetime.fromisoformat(raw_trained_at.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Invalid TFT trained_at timestamp during promotion: %s", raw_trained_at)

    with SessionLocal() as session:
        existing = session.query(TFTModelMetadata).filter(
            TFTModelMetadata.symbol == symbol
        ).first()

        importance_json = _importance_metadata_payload(result)

        if existing:
            existing.config_json = json.dumps(result.get("config", {}))
            existing.metrics_json = json.dumps(metrics)
            existing.importance_json = importance_json
            existing.checkpoint_path = result.get("checkpoint_path", "")
            existing.trained_at = trained_at
            existing.quality_gate_passed = True
        else:
            session.add(
                TFTModelMetadata(
                    symbol=symbol,
                    config_json=json.dumps(result.get("config", {})),
                    metrics_json=json.dumps(metrics),
                    importance_json=importance_json,
                    checkpoint_path=result.get("checkpoint_path", ""),
                    trained_at=trained_at,
                    quality_gate_passed=True,
                )
            )

        session.commit()
        logger.info("Promoted TFT metadata persisted for %s", symbol)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configure_cli_logging(logging.INFO)

    parser = argparse.ArgumentParser(description="Train TFT-ASRO model")
    parser.add_argument("--symbol", default="HG=F")
    parser.add_argument("--no-asro", action="store_true", help="Use standard QuantileLoss instead of ASRO")
    parser.add_argument("--upload-hub", action="store_true", help="Upload artifacts to HF Hub after training")
    parser.add_argument(
        "--deterministic-weekly-validation",
        action="store_true",
        help="Bypass Optuna overlays and run the fixed monotonic weekly validation config",
    )
    args = parser.parse_args()

    cfg = get_tft_config()
    result = train_tft_model(
        cfg,
        use_asro=not args.no_asro,
        upload_to_hub=args.upload_hub,
        deterministic_weekly_validation=args.deterministic_weekly_validation,
    )

    print("\n" + "=" * 60)
    print("TFT-ASRO TRAINING COMPLETE")
    print("=" * 60)
    for k, v in result.get("test_metrics", {}).items():
        print(f"  {k}: {v:.4f}")
    print(f"\nCheckpoint: {result.get('checkpoint_path')}")
