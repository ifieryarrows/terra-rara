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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from deep_learning.config import TFTASROConfig, get_tft_config

# pytorch_forecasting prescalers are fit on DataFrames but transform numpy arrays
# internally on every batch — this produces thousands of identical sklearn warnings.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)

logger = logging.getLogger(__name__)


def train_tft_model(
    cfg: Optional[TFTASROConfig] = None,
    use_asro: bool = True,
    upload_to_hub: bool = False,
) -> dict:
    """
    End-to-end TFT-ASRO training.

    Returns:
        Dict with metrics, checkpoint path, and feature importance.
    """
    # pytorch_forecasting >=1.0 uses the unified `lightning` package.
    # Importing from `pytorch_lightning` gives a different LightningModule
    # base class, causing "model must be a LightningModule" at trainer.fit().
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
    from deep_learning.training.metrics import compute_all_metrics, select_prediction_horizon
    from deep_learning.training.callbacks import CurriculumLossScheduler, SWACallback

    if cfg is None:
        cfg = get_tft_config()

    # ---- 0a. Load Optuna best params if available ----
    # When the hyperopt step ran before this trainer, it writes best params to
    # optuna_results.json. We apply those params over the default config so that
    # the final training run actually benefits from the search.
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
    ]

    if use_asro:
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
        y_actual = select_prediction_horizon(torch.cat(y_actual_parts).cpu().numpy(), horizon_idx=0)

        # Gather top-k checkpoint paths
        best_k = getattr(trainer.checkpoint_callback, "best_k_models", {})
        ckpt_paths = sorted(best_k.keys(), key=lambda p: best_k[p]) if best_k else []

        # Always include the just-trained model as a baseline
        all_pred_arrays = []

        def _predict_to_np(mdl):
            pred = mdl.predict(test_dl, return_x=True)
            pt = pred.output if hasattr(pred, "output") else pred
            return pt.cpu().numpy() if hasattr(pt, "cpu") else np.array(pt)

        # Predictions from the best model (already in memory)
        all_pred_arrays.append(_predict_to_np(model))

        # Load additional checkpoints for ensemble
        for cp in ckpt_paths:
            if str(cp) == str(best_path):
                continue  # already have this one
            try:
                ckpt_model = load_tft_model(str(cp))
                all_pred_arrays.append(_predict_to_np(ckpt_model))
                del ckpt_model
            except Exception as exc:
                logger.debug("Skipping ensemble checkpoint %s: %s", cp, exc)

        ensemble_size = len(all_pred_arrays)
        logger.info(
            "Snapshot Ensemble: %d model(s) for test evaluation", ensemble_size,
        )

        # Element-wise median across all models
        if ensemble_size >= 2:
            pred_np = np.median(np.stack(all_pred_arrays, axis=0), axis=0)
        else:
            pred_np = all_pred_arrays[0]

        median_idx = len(cfg.model.quantiles) // 2
        if pred_np.ndim == 3:
            pred_t1 = pred_np[:, 0, :]
            y_pred_median = pred_t1[:, median_idx]
            y_pred_q10 = pred_t1[:, 1] if pred_t1.shape[1] > 2 else None
            y_pred_q90 = pred_t1[:, -2] if pred_t1.shape[1] > 2 else None
            y_pred_q02 = pred_t1[:, 0] if pred_t1.shape[1] > 2 else None
            y_pred_q98 = pred_t1[:, -1] if pred_t1.shape[1] > 2 else None
        else:
            y_pred_median = pred_np.flatten()
            pred_t1 = None
            y_pred_q10 = y_pred_q90 = y_pred_q02 = y_pred_q98 = None

        n = min(len(y_actual), len(y_pred_median))
        test_metrics = compute_all_metrics(
            y_actual[:n],
            y_pred_median[:n],
            y_pred_q10=y_pred_q10[:n] if y_pred_q10 is not None else None,
            y_pred_q90=y_pred_q90[:n] if y_pred_q90 is not None else None,
            y_pred_q02=y_pred_q02[:n] if y_pred_q02 is not None else None,
            y_pred_q98=y_pred_q98[:n] if y_pred_q98 is not None else None,
            y_pred_quantiles=pred_t1[:n] if pred_t1 is not None else None,
        )
        test_metrics["ensemble_size"] = ensemble_size
        logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in test_metrics.items()})

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
            "max_encoder_length": cfg.model.max_encoder_length,
            "max_prediction_length": cfg.model.max_prediction_length,
        },
        "n_unknown_features": len(tv_unknown),
        "n_known_features": len(tv_known),
        "train_samples": len(training_ds),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    _persist_tft_metadata(cfg.feature_store.target_symbol, result)

    # Write metadata JSON to disk for CI quality gate
    meta_json_path = Path(cfg.training.best_model_path).parent / "tft_metadata.json"
    try:
        meta_json_path.write_text(json.dumps(result, indent=2, default=str))
        logger.info("Training metadata written to %s", meta_json_path)
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


def _apply_optuna_results(cfg: TFTASROConfig) -> TFTASROConfig:
    """
    If an optuna_results.json exists in the checkpoint directory, overlay the
    best hyperparameters onto cfg and return the updated config.  This connects
    the hyperopt step to the final training run so search results are not wasted.
    """
    import json
    from dataclasses import replace
    from deep_learning.config import ASROConfig, TFTModelConfig, TrainingConfig

    # optuna_results.json is saved at tft/ root (alongside best_tft_asro.ckpt),
    # not inside the checkpoints/ subdirectory.
    results_path = Path(cfg.training.best_model_path).parent / "optuna_results.json"
    if not results_path.exists():
        return cfg

    try:
        data = json.loads(results_path.read_text())
        params = data.get("best_params", {})
        if not params:
            return cfg

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

        model_overrides = {
            k: params[k] for k in (
                "hidden_size", "attention_head_size", "dropout",
                "hidden_continuous_size", "learning_rate",
                "gradient_clip_val", "max_encoder_length",
                "weight_decay",
            ) if k in params
        }
        asro_overrides = {
            k: params[k] for k in ("lambda_vol", "lambda_quantile", "lambda_madl", "lambda_crossing")
            if k in params
        }
        training_overrides = {
            k: params[k] for k in ("batch_size",) if k in params
        }

        new_model = replace(cfg.model, **model_overrides) if model_overrides else cfg.model
        new_asro = replace(cfg.asro, **asro_overrides) if asro_overrides else cfg.asro
        new_training = replace(cfg.training, **training_overrides) if training_overrides else cfg.training

        logger.info(
            "Loaded Optuna best params (trial #%d, val_loss=%.4f): %s",
            data.get("best_trial", -1),
            data.get("best_value", float("nan")),
            params,
        )
        return replace(cfg, model=new_model, asro=new_asro, training=new_training)

    except Exception as exc:
        logger.warning("Could not apply Optuna results: %s", exc)
        return cfg


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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train TFT-ASRO model")
    parser.add_argument("--symbol", default="HG=F")
    parser.add_argument("--no-asro", action="store_true", help="Use standard QuantileLoss instead of ASRO")
    parser.add_argument("--upload-hub", action="store_true", help="Upload artifacts to HF Hub after training")
    args = parser.parse_args()

    cfg = get_tft_config()
    result = train_tft_model(cfg, use_asro=not args.no_asro, upload_to_hub=args.upload_hub)

    print("\n" + "=" * 60)
    print("TFT-ASRO TRAINING COMPLETE")
    print("=" * 60)
    for k, v in result.get("test_metrics", {}).items():
        print(f"  {k}: {v:.4f}")
    print(f"\nCheckpoint: {result.get('checkpoint_path')}")
