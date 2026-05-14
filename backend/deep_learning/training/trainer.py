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
import json
import logging
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

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

KNOWN_GOOD_CONFIG = {
    "max_encoder_length": 50,
    "hidden_size": 48,
    "attention_head_size": 2,
    "dropout": 0.30,
    "hidden_continuous_size": 16,
    "learning_rate": 2e-4,
    "weight_decay": 5e-5,
    "lambda_vol": 0.30,
    "lambda_quantile": 0.25,
    "lambda_madl": 0.40,
    "lambda_weekly_quantile": 0.55,
    "lambda_t1_quantile": 0.15,
    "lambda_dispersion": 0.20,
    "lambda_directional": 0.10,
    "batch_size": 32,
}

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
    "quantile_crossing_rate",
    "sorted_quantile_crossing_rate",
)


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


def _predict_quantiles_to_np(mdl, dataloader, cfg: TFTASROConfig) -> np.ndarray:
    pred = mdl.predict(dataloader, mode="quantiles")
    pred_np = pred.cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)
    _validate_quantile_prediction_shape(pred_np, cfg)
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
) -> dict[str, float]:
    from deep_learning.training.metrics import (
        compute_all_metrics,
        compute_weekly_metrics,
        monotonic_quantiles_np,
        quantile_crossing_rate,
        quantile_median_sort_gap,
        select_prediction_horizon,
    )

    pred_np = np.asarray(pred_np)
    _validate_quantile_prediction_shape(pred_np, cfg)

    median_idx = len(cfg.model.quantiles) // 2
    ordered_pred_np = monotonic_quantiles_np(pred_np, median_idx=median_idx)
    raw_pred_t1 = pred_np[:, 0, :]
    pred_t1 = ordered_pred_np[:, 0, :]
    y_pred_median = pred_t1[:, median_idx]
    y_pred_q10 = pred_t1[:, 1]
    y_pred_q90 = pred_t1[:, -2]
    y_pred_q02 = pred_t1[:, 0]
    y_pred_q98 = pred_t1[:, -1]

    y_actual = select_prediction_horizon(y_actual_path, horizon_idx=0)
    n = min(len(y_actual), len(y_pred_median))
    test_metrics = compute_all_metrics(
        y_actual[:n],
        y_pred_median[:n],
        y_pred_q10=y_pred_q10[:n],
        y_pred_q90=y_pred_q90[:n],
        y_pred_q02=y_pred_q02[:n],
        y_pred_q98=y_pred_q98[:n],
        y_pred_quantiles=pred_t1[:n],
    )
    raw_gap_mean, raw_gap_max = quantile_median_sort_gap(raw_pred_t1[:n], median_idx)
    test_metrics["raw_quantile_crossing_rate"] = quantile_crossing_rate(raw_pred_t1[:n])
    test_metrics["raw_median_sort_gap_mean"] = raw_gap_mean
    test_metrics["raw_median_sort_gap_max"] = raw_gap_max

    n_path = min(len(y_actual_path), len(pred_np))
    weekly_metrics = compute_weekly_metrics(
        y_actual_path[:n_path],
        pred_np[:n_path],
        quantiles=cfg.model.quantiles,
        horizon=cfg.forecast.primary_horizon_days,
    )
    test_metrics.update(weekly_metrics)
    _require_promotable_metrics(test_metrics)
    return test_metrics


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
    pl.seed_everything(cfg.training.seed)

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

    # ---- 3. Model ----
    model = create_tft_model(training_ds, cfg, use_asro=use_asro)

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
            "Weekly loss   | weekly_q=%.2f t1_q=%.2f dispersion=%.2f directional=%.2f monotonic_transform=true",
            cfg.weekly_loss.lambda_weekly_quantile,
            cfg.weekly_loss.lambda_t1_quantile,
            cfg.weekly_loss.lambda_dispersion,
            cfg.weekly_loss.lambda_directional,
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

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.early_stopping_patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="tft-asro-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        WeeklyLossComponentLogger(),
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
    )

    logger.info("Starting TFT-ASRO training ...")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # ---- 6. Best checkpoint ----
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        final_path = Path(cfg.training.best_model_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best_path, final_path)
        logger.info("Best model saved to %s (val_loss=%.6f)", final_path, trainer.checkpoint_callback.best_model_score)
    else:
        final_path = Path(cfg.training.best_model_path)

    # ---- 7. Evaluate on test set (Snapshot Ensemble) ----
    # Use the top-k checkpoints saved by ModelCheckpoint and take the
    # element-wise median of their predictions.  This smooths stochastic
    # outliers and improves directional robustness (REG-2026-001 P2-2).
    test_metrics = {}
    if test_dl is not None:
        import torch
        from deep_learning.models.tft_copper import load_tft_model

        # Collect actual values (same regardless of which model predicts)
        y_actual_parts = []
        for batch in test_dl:
            y_actual_parts.append(
                batch[1][0] if isinstance(batch[1], (list, tuple)) else batch[1]
            )
        y_actual_path = torch.cat(y_actual_parts).cpu().numpy()
        # Gather top-k checkpoint paths
        best_k = getattr(trainer.checkpoint_callback, "best_k_models", {})
        ckpt_paths = sorted(best_k.keys(), key=lambda p: best_k[p]) if best_k else []

        # Always include the just-trained model as a baseline
        all_pred_arrays = []

        # Predictions from the best model (already in memory)
        all_pred_arrays.append(_predict_quantiles_to_np(model, test_dl, cfg))

        # Load additional checkpoints for ensemble
        for cp in ckpt_paths:
            if str(cp) == str(best_path):
                continue  # already have this one
            try:
                ckpt_model = load_tft_model(str(cp))
                all_pred_arrays.append(_predict_quantiles_to_np(ckpt_model, test_dl, cfg))
                del ckpt_model
            except Exception as exc:
                logger.warning("Skipping incompatible ensemble checkpoint %s: %s", cp, exc)

        ensemble_size = len(all_pred_arrays)
        logger.info(
            "Snapshot Ensemble: %d model(s) for test evaluation", ensemble_size,
        )

        # Element-wise median across all models
        if ensemble_size >= 2:
            pred_np = np.median(np.stack(all_pred_arrays, axis=0), axis=0)
        else:
            pred_np = all_pred_arrays[0]

        test_metrics = _compute_test_metrics_from_quantiles(y_actual_path, pred_np, cfg)
        test_metrics["ensemble_size"] = ensemble_size
        logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in test_metrics.items()})

    _require_promotable_metrics(test_metrics)

    calibration_artifact = _write_conformal_calibration_artifact(
        cfg=cfg,
        model=model,
        val_dl=val_dl,
        feature_frame=master_df,
    )

    # ---- 8. Variable importance ----
    var_importance = get_variable_importance(model, val_dataloader=val_dl)

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
            "lambda_directional": cfg.weekly_loss.lambda_directional,
            "lambda_dispersion": cfg.weekly_loss.lambda_dispersion,
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
        "conformal_calibration_path": str(calibration_artifact) if calibration_artifact else None,
        "n_unknown_features": len(tv_unknown),
        "n_known_features": len(tv_known),
        "train_samples": len(training_ds),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    _persist_tft_metadata(cfg.feature_store.target_symbol, result)

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
    else:
        result["hub_upload_skipped"] = "disabled_until_quality_gate_passes"

    return result


def _write_conformal_calibration_artifact(
    *,
    cfg: TFTASROConfig,
    model,
    val_dl,
    feature_frame,
) -> Optional[Path]:
    """
    Fit interval adjustment on validation/calibration data, never final test.

    If the validation window is too small, the adjustment is intentionally
    zero; this preserves interval honesty without leaking test information.
    """
    try:
        import torch

        from deep_learning.calibration.conformal import rolling_conformal_adjustment
        from deep_learning.training.metrics import (
            cumulative_horizon,
            cumulative_quantiles,
            monotonic_quantiles_np,
        )

        y_parts = []
        for batch in val_dl:
            y_parts.append(batch[1][0] if isinstance(batch[1], (list, tuple)) else batch[1])
        if not y_parts:
            return None

        y_actual_path = torch.cat(y_parts).cpu().numpy()
        pred = model.predict(val_dl, mode="quantiles")
        pred_np = pred.cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)
        n = min(len(y_actual_path), len(pred_np))
        if n <= 0:
            return None

        weekly_actual = cumulative_horizon(y_actual_path[:n], horizon=cfg.forecast.primary_horizon_days)
        ordered_pred_np = monotonic_quantiles_np(
            pred_np[:n],
            median_idx=len(cfg.model.quantiles) // 2,
        )
        weekly_quantiles = cumulative_quantiles(
            ordered_pred_np,
            horizon=cfg.forecast.primary_horizon_days,
        )
        q = tuple(cfg.model.quantiles)
        q10_idx = q.index(0.10)
        q90_idx = q.index(0.90)
        raw_lower = weekly_quantiles[:, q10_idx]
        raw_upper = weekly_quantiles[:, q90_idx]
        validation_pi80_coverage = float(
            np.mean((weekly_actual >= raw_lower) & (weekly_actual <= raw_upper))
        )

        calibration_status = "fit"
        if validation_pi80_coverage >= 0.90:
            global_adj = 0.0
            calibration_status = "skipped_interval_already_overcovered"
        else:
            global_adj = rolling_conformal_adjustment(
                weekly_actual,
                raw_lower,
                raw_upper,
                alpha=0.20,
            )
        artifact = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target": "weekly_5d_log_return",
            "alpha": 0.20,
            "global_adjustment": float(global_adj),
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
            "validation_pi80_coverage": validation_pi80_coverage,
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
    # optuna_results.json is saved at tft/ root (alongside best_tft_asro.ckpt),
    # not inside the checkpoints/ subdirectory.
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
        if not params:
            logger.warning(
                "Optuna results did not contain finite best params (status=%s); "
                "using known-good fallback config: %s",
                data.get("status", "unknown"),
                KNOWN_GOOD_CONFIG,
            )
            return _overlay_training_config(cfg, KNOWN_GOOD_CONFIG)

        # Guard against legacy optuna_results.json files produced before the
        # 2026-04-27 metric/coherence fixes.  Reusing those artifacts with
        # run_hyperopt=false can otherwise re-apply the known weak-direction
        # regime (for example lambda_madl=0.2).
        params = dict(params)
        if "lambda_vol" in params:
            params["lambda_vol"] = max(float(params["lambda_vol"]), 0.30)
        if "lambda_quantile" in params:
            params["lambda_quantile"] = min(max(float(params["lambda_quantile"]), 0.25), 0.40)
        if "lambda_madl" in params:
            params["lambda_madl"] = max(float(params["lambda_madl"]), 0.30)
        if "max_encoder_length" in params and int(params["max_encoder_length"]) < 40:
            logger.warning(
                "Optuna max_encoder_length=%s is below weekly-safe floor; clamping to 40",
                params["max_encoder_length"],
            )
            params["max_encoder_length"] = 40
        if "learning_rate" in params:
            params["learning_rate"] = min(float(params["learning_rate"]), 6e-4)
        if "weight_decay" in params:
            params["weight_decay"] = min(float(params["weight_decay"]), 5e-4)
        if "lambda_directional" in params:
            params["lambda_directional"] = min(float(params["lambda_directional"]), 0.12)
        if "lambda_dispersion" in params:
            params["lambda_dispersion"] = max(float(params["lambda_dispersion"]), 0.20)

        logger.info(
            "Loaded Optuna best params (trial #%d, weekly_objective=%.4f): %s",
            data.get("best_trial", -1),
            data.get("best_value", float("nan")),
            params,
        )
        return _overlay_training_config(cfg, params)

    except Exception as exc:
        logger.warning("Could not apply Optuna results: %s", exc)
        return _overlay_training_config(cfg, KNOWN_GOOD_CONFIG)


def _overlay_training_config(cfg: TFTASROConfig, params: dict) -> TFTASROConfig:
    """Overlay model, ASRO and batch-size params onto a TFT-ASRO config."""
    model_overrides = {
        k: params[k] for k in (
            "hidden_size", "attention_head_size", "dropout",
            "hidden_continuous_size", "learning_rate",
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
            "lambda_weekly_quantile", "lambda_t1_quantile", "lambda_directional",
            "lambda_dispersion",
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


def _persist_tft_metadata(symbol: str, result: dict) -> None:
    """Save TFT model metadata to DB."""
    try:
        from app.db import SessionLocal
        from app.models import TFTModelMetadata

        with SessionLocal() as session:
            existing = session.query(TFTModelMetadata).filter(
                TFTModelMetadata.symbol == symbol
            ).first()

            if existing:
                existing.config_json = json.dumps(result.get("config", {}))
                existing.metrics_json = json.dumps(result.get("test_metrics", {}))
                existing.checkpoint_path = result.get("checkpoint_path", "")
                existing.trained_at = datetime.now(timezone.utc)
            else:
                session.add(TFTModelMetadata(
                    symbol=symbol,
                    config_json=json.dumps(result.get("config", {})),
                    metrics_json=json.dumps(result.get("test_metrics", {})),
                    checkpoint_path=result.get("checkpoint_path", ""),
                ))

            session.commit()
            logger.info("TFT metadata persisted for %s", symbol)
    except Exception as exc:
        logger.warning("Could not persist TFT metadata: %s", exc)


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
