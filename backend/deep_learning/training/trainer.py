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
    from deep_learning.training.metrics import compute_all_metrics

    if cfg is None:
        cfg = get_tft_config()

    # ---- 0. ASRO loss sanity check (runs before any training) ----
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
        master_df, tv_unknown, tv_known, target_cols = build_tft_dataframe(session, cfg)

    logger.info("Master DataFrame: %d rows x %d cols", *master_df.shape)

    # ---- 2. Datasets ----
    training_ds, validation_ds, test_ds = build_datasets(
        master_df, tv_unknown, tv_known, target_cols, cfg,
    )
    train_dl, val_dl, test_dl = create_dataloaders(training_ds, validation_ds, test_ds, cfg)

    # ---- 3. Model ----
    model = create_tft_model(training_ds, cfg, use_asro=use_asro)

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

    # ---- 5. Train ----
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        gradient_clip_val=cfg.model.gradient_clip_val,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
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

    # ---- 7. Evaluate on test set ----
    test_metrics = {}
    if test_dl is not None:
        logger.info("Evaluating on test set ...")
        predictions = model.predict(test_dl, return_x=True)
        pred_tensor = predictions.output if hasattr(predictions, "output") else predictions

        if hasattr(pred_tensor, "cpu"):
            pred_np = pred_tensor.cpu().numpy()
        else:
            pred_np = np.array(pred_tensor)

        if pred_np.ndim == 3:
            median_idx = len(cfg.model.quantiles) // 2
            y_pred_median = pred_np[:, 0, median_idx]
            y_pred_q10 = pred_np[:, 0, 1] if pred_np.shape[2] > 2 else None
            y_pred_q90 = pred_np[:, 0, -2] if pred_np.shape[2] > 2 else None
            y_pred_q02 = pred_np[:, 0, 0] if pred_np.shape[2] > 2 else None
            y_pred_q98 = pred_np[:, 0, -1] if pred_np.shape[2] > 2 else None
        else:
            y_pred_median = pred_np.flatten()
            y_pred_q10 = y_pred_q90 = y_pred_q02 = y_pred_q98 = None

        y_actual_parts = []
        for batch in test_dl:
            y_actual_parts.append(batch[1][0] if isinstance(batch[1], (list, tuple)) else batch[1])
        import torch
        y_actual = torch.cat(y_actual_parts).cpu().numpy().flatten()

        n = min(len(y_actual), len(y_pred_median))
        test_metrics = compute_all_metrics(
            y_actual[:n],
            y_pred_median[:n],
            y_pred_q10=y_pred_q10[:n] if y_pred_q10 is not None else None,
            y_pred_q90=y_pred_q90[:n] if y_pred_q90 is not None else None,
            y_pred_q02=y_pred_q02[:n] if y_pred_q02 is not None else None,
            y_pred_q98=y_pred_q98[:n] if y_pred_q98 is not None else None,
        )
        logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in test_metrics.items()})

    # ---- 8. Variable importance ----
    var_importance = get_variable_importance(model)

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
            "max_encoder_length": cfg.model.max_encoder_length,
            "max_prediction_length": cfg.model.max_prediction_length,
        },
        "n_unknown_features": len(tv_unknown),
        "n_known_features": len(tv_known),
        "train_samples": len(training_ds),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    _persist_tft_metadata(cfg.feature_store.target_symbol, result)

    # ---- 10. Upload to HF Hub (for persistence across HF Space rebuilds) ----
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
        result["hub_uploaded"] = False

    return result


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
    args = parser.parse_args()

    cfg = get_tft_config()
    result = train_tft_model(cfg, use_asro=not args.no_asro)

    print("\n" + "=" * 60)
    print("TFT-ASRO TRAINING COMPLETE")
    print("=" * 60)
    for k, v in result.get("test_metrics", {}).items():
        print(f"  {k}: {v:.4f}")
    print(f"\nCheckpoint: {result.get('checkpoint_path')}")
