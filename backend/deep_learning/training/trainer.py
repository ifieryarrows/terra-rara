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
    "max_encoder_length": 10,
    "hidden_size": 12,
    "attention_head_size": 2,
    "dropout": 0.35,
    "hidden_continuous_size": 6,
    "learning_rate": 5e-4,
    "output_head_lr_mult": 0.05,
    "weight_decay": 1e-2,
    "lambda_vol": 0.30,
    "lambda_quantile": 0.25,
    "lambda_madl": 0.40,
    "lambda_weekly_quantile": 1.00,
    "lambda_t1_quantile": 0.25,
    "lambda_t1_directional": 0.0,
    "lambda_dispersion": 0.0,
    "lambda_magnitude": 0.0,
    "lambda_naive": 0.0,
    "lambda_bias": 0.0,
    "lambda_drift_prior": 1.0,
    "lambda_directional": 0.0,
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
    "batch_size",
    "lambda_weekly_quantile",
    "lambda_t1_quantile",
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
) -> None:
    """Average compatible model weights into one deployable Lightning checkpoint.

    Source checkpoints have already been ranked by the validation monitor. Test
    predictions and labels are unavailable here by design. Non-floating state
    (for example counters) is retained from the best-ranked checkpoint.
    """
    if not source_paths:
        raise ValueError("At least one checkpoint is required for promotion")

    import shutil
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(source_paths) == 1:
        shutil.copy2(source_paths[0], destination)
        return

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

    averaged_payload = payloads[0]
    averaged_state = averaged_payload["state_dict"]
    for key in reference_keys:
        tensors = [state[key] for state in state_dicts]
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
    _log_weekly_alignment_sample(y_actual_path[:n_path], pred_np[:n_path], cfg)
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
    from deep_learning.data.feature_store import build_tft_dataframe
    from deep_learning.data.dataset import build_datasets, create_dataloaders
    from deep_learning.models.tft_copper import create_tft_model, get_variable_importance, format_prediction
    from deep_learning.training.metrics import (
        resolve_weekly_median_cap,
        summarize_dataloader_target_scale,
    )
    from deep_learning.training.callbacks import (
        CurriculumLossScheduler,
        SWACallback,
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

    n_rows = len(master_df)
    test_size = int(n_rows * cfg.training.test_ratio)
    val_size = int(n_rows * cfg.training.val_ratio)
    train_size = n_rows - val_size - test_size
    train_cutoff = int(master_df["time_idx"].iloc[train_size - 1])
    val_cutoff = int(master_df["time_idx"].iloc[train_size + val_size - 1])
    max_time_idx = int(master_df["time_idx"].iloc[-1])
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

    # ---- 3. Model ----
    weekly_target_scale = float(train_scale_audit.get("actual_weekly_std") or 0.0) or None
    model = create_tft_model(
        training_ds,
        cfg,
        use_asro=use_asro,
        weekly_target_scale=weekly_target_scale,
    )

    # Log active config so every run is fully reproducible from logs
    n_batches = len(train_dl)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Training config | hidden=%d hidden_cont=%d dropout=%.2f "
        "heads=%d lr=%.0e clip=%.1f",
        cfg.model.hidden_size, cfg.model.hidden_continuous_size,
        cfg.model.dropout, cfg.model.attention_head_size,
        cfg.model.learning_rate, cfg.model.gradient_clip_val,
    )
    if cfg.forecast.primary_horizon_days == 5:
        logger.info(
            "Training data  | samples=%d batch_size=%d batches/epoch=%d patience=%d",
            len(training_ds), cfg.training.batch_size, n_batches,
            cfg.training.early_stopping_patience,
        )
        logger.info(
            "Weekly loss   | weekly_q=%.2f t1_q=%.2f dispersion=%.2f "
            "magnitude=%.2f naive=%.2f bias=%.2f directional=%.2f saturation=%.2f "
            "positive_rate=%.2f interval=%.2f "
            "median_cap=%.6f "
            "monotonic_transform=true",
            cfg.weekly_loss.lambda_weekly_quantile,
            cfg.weekly_loss.lambda_t1_quantile,
            cfg.weekly_loss.lambda_dispersion,
            cfg.weekly_loss.lambda_magnitude,
            cfg.weekly_loss.lambda_naive,
            cfg.weekly_loss.lambda_bias,
            cfg.weekly_loss.lambda_directional,
            cfg.weekly_loss.lambda_saturation,
            cfg.weekly_loss.lambda_positive_rate,
            cfg.weekly_loss.lambda_interval,
            cfg.weekly_loss.weekly_median_cap or 0.0,
        )
    else:
        logger.info(
            "Training data  | samples=%d batch_size=%d batches/epoch=%d "
            "patience=%d w_quantile=%.2f w_sharpe=%.2f lambda_vol=%.2f",
            len(training_ds), cfg.training.batch_size, n_batches,
            cfg.training.early_stopping_patience,
            cfg.asro.lambda_quantile, 1.0 - cfg.asro.lambda_quantile,
            cfg.asro.lambda_vol,
        )
    logger.info(
        "Model params   | total=%s trainable=%s",
        f"{total_params:,}", f"{trainable_params:,}",
    )

    # ---- 4. Callbacks ----
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # WeeklyASROPFLoss overrides Metric.update() so val_loss equals the exact
    # undivided sample-weighted validation objective; WeeklyLossComponentLogger
    # logs component breakdowns alongside val_loss.
    monitor_metric = "val_loss"
    checkpoint_filename = "tft-asro-{epoch:02d}-{val_loss:.4f}"

    callbacks = [
        # Must run before EarlyStopping/ModelCheckpoint so the monitored
        # validation metric is present in callback_metrics for this epoch.
        WeeklyLossComponentLogger(),
        EarlyStopping(
            monitor=monitor_metric,
            patience=cfg.training.early_stopping_patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=checkpoint_filename,
            monitor=monitor_metric,
            mode="min",
            save_top_k=3,
            save_last=True,
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

    callbacks.append(SWACallback(swa_start_pct=0.75))

    # log_every_n_steps must not exceed the number of training batches
    log_steps = max(1, min(5, n_batches))

    # ---- 5. Train ----
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        gradient_clip_val=cfg.model.gradient_clip_val,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=log_steps,
        deterministic=True,
    )

    logger.info("Starting TFT-ASRO training ...")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # ---- 6. Validation-ranked checkpoint promotion ----
    # A single best epoch remained sensitive to small training-trajectory
    # changes on an immutable snapshot. For weekly ASRO, average the two best
    # validation checkpoints into one deployable checkpoint. This keeps live
    # inference/calibration/gate evaluation aligned while reducing dependence
    # on one noisy epoch; test labels are not available during this selection.
    final_path = Path(cfg.training.best_model_path)
    best_path = trainer.checkpoint_callback.best_model_path
    checkpoint_selection = {
        "fit_split": "validation",
        "monitor": monitor_metric,
        "method": "best_validation_checkpoint",
        "source_count": 0,
        "source_checkpoints": [],
        "source_scores": [],
        "test_labels_used": False,
    }
    if best_path:
        import shutil
        from deep_learning.models.tft_copper import load_tft_model

        source_count = (
            2 if use_asro and cfg.forecast.primary_horizon_days == 5 else 1
        )
        ranked_checkpoints = _validation_ranked_checkpoint_paths(
            trainer.checkpoint_callback,
            max_count=source_count,
        )
        source_paths = [path for path, _score in ranked_checkpoints]
        _build_uniform_checkpoint_soup(source_paths, final_path)
        if len(source_paths) > 1:
            best_score = float(ranked_checkpoints[0][1])
            soup_accepted = False
            try:
                souped_model = load_tft_model(str(final_path))
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
                shutil.copy2(source_paths[0], final_path)
                source_paths = [source_paths[0]]
                ranked_checkpoints = [ranked_checkpoints[0]]
                checkpoint_selection["method"] = "best_validation_checkpoint_soup_reverted"
        checkpoint_selection.update(
            {
                "source_count": len(source_paths),
                "source_checkpoints": [path.name for path in source_paths],
                "source_scores": [score for _path, score in ranked_checkpoints],
            }
        )
        best_path = str(final_path)
        logger.info(
            "Promoted validation checkpoint artifact | method=%s sources=%s scores=%s path=%s",
            checkpoint_selection["method"],
            checkpoint_selection["source_checkpoints"],
            checkpoint_selection["source_scores"],
            final_path,
        )

    # Calibration, gate evaluation, and live inference must all use the same
    # promoted checkpoint.  ``trainer.fit`` leaves ``model`` at the last
    # training state, while production loads ``best_tft_asro.ckpt``; using the
    # former for validation calibration would make the persisted correction
    # depend on a checkpoint that is never deployed.
    evaluation_model = model
    if best_path:
        try:
            from deep_learning.models.tft_copper import load_tft_model

            evaluation_model = load_tft_model(str(best_path))
            logger.info(
                "Using promoted checkpoint for calibration and gate evaluation: %s",
                best_path,
            )
        except Exception as exc:
            logger.warning(
                "Could not reload promoted checkpoint for evaluation; "
                "using in-memory model: %s",
                exc,
            )

    # ---- 7. Fit validation-only ordered Split CQR, then evaluate test ----
    # Median point forecasts come directly from the trained proper-scoring model
    # (without raw-logit sign/magnitude hacks). Uncertainty intervals [q10, q90]
    # and [q02, q98] are calibrated in ordered weekly space on the chronological
    # validation split using Volatility-Normalized Split CQR and evaluated identically
    # on held-out test and live inference.
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
    weekly_cqr_params: Optional[dict] = None
    val_actual_path = None
    val_pred_np = None
    try:
        from deep_learning.calibration.conformal import fit_weekly_cqr
        from deep_learning.models.monotonic_quantiles import aggregate_weekly_quantiles_np
        from deep_learning.training.metrics import (
            apply_weekly_median_cap_np,
            cumulative_horizon,
            monotonic_quantiles_np,
        )

        val_pred_np, val_actual_path = _predict_quantiles_to_np(
            evaluation_model, val_dl, cfg, return_y=True
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

            val_eval_pred, _ = apply_weekly_median_cap_np(
                val_pred_np[:n_val],
                weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
                quantiles=tuple(cfg.model.quantiles),
                horizon=cfg.forecast.primary_horizon_days,
            )
            median_idx = len(cfg.model.quantiles) // 2
            val_ordered = monotonic_quantiles_np(val_eval_pred, median_idx=median_idx)
            val_weekly_q = aggregate_weekly_quantiles_np(
                val_ordered[:, : cfg.forecast.primary_horizon_days, :],
                median_idx=median_idx,
            )
            val_weekly_actual = cumulative_horizon(
                val_actual_path[:n_val],
                horizon=cfg.forecast.primary_horizon_days,
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
                    "validation_pi80_coverage": float(
                        weekly_cqr_params.get("validation_pi80_coverage_calibrated", 0.0) or 0.0
                    ),
                }
            )
    except Exception as exc:
        logger.warning(
            "Validation Split CQR unavailable; retaining raw RSS intervals: %s",
            exc,
        )

    direction_sign_multiplier = 1
    daily_sign_multiplier = 1
    weekly_sign_threshold = 0.0
    weekly_magnitude_scale = 1.0
    weekly_interval_scale = 1.0
    logger.info("Validation-only Split CQR calibration: %s", weekly_cqr_params or interval_calibration)

    # ---- 8. Evaluate on the promoted best checkpoint ----
    test_metrics = {}
    if test_dl is not None:
        # Single-pass prediction and target extraction (eliminates dataloader double-iteration)
        pred_np, y_actual_path = _predict_quantiles_to_np(
            evaluation_model, test_dl, cfg, return_y=True
        )
        ensemble_size = 1
        logger.info("Promoted checkpoint evaluation: 1 model")

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
    )

    # ---- 8. Variable importance ----
    var_importance = get_variable_importance(evaluation_model, val_dataloader=val_dl)

    # ---- 9. Persist metadata ----
    result = {
        "checkpoint_path": str(final_path),
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
            "lambda_t1_directional": cfg.weekly_loss.lambda_t1_directional,
            "lambda_dispersion": cfg.weekly_loss.lambda_dispersion,
            "lambda_magnitude": cfg.weekly_loss.lambda_magnitude,
            "lambda_naive": cfg.weekly_loss.lambda_naive,
            "lambda_bias": cfg.weekly_loss.lambda_bias,
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

        # Single-pass prediction and target extraction to avoid DataLoader double iteration
        pred_np, y_actual_path = _predict_quantiles_to_np(
            model, val_dl, cfg, return_y=True
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
    weekly_loss_overrides = {
        k: params[k] for k in (
            "lambda_weekly_quantile", "lambda_t1_quantile", "lambda_t1_directional",
            "lambda_directional", "lambda_dispersion", "lambda_magnitude",
            "lambda_naive", "lambda_bias", "lambda_drift_prior", "lambda_saturation",
            "lambda_positive_rate", "lambda_interval",
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
    return replace(cfg, model=new_model, asro=new_asro, weekly_loss=new_weekly_loss, training=new_training)


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
