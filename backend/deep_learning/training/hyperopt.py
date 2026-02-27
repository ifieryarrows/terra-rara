"""
Optuna-based Hyperparameter Optimization for TFT-ASRO.

Searches across model architecture, training, and ASRO loss parameters
using Tree-structured Parzen Estimator (TPE) with early pruning.

Usage:
    python -m deep_learning.training.hyperopt --n-trials 50
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)

from deep_learning.config import (
    ASROConfig,
    TFTASROConfig,
    TFTModelConfig,
    TrainingConfig,
    get_tft_config,
)

logger = logging.getLogger(__name__)


def create_trial_config(trial, base_cfg: TFTASROConfig) -> TFTASROConfig:
    """Map an Optuna trial to a TFT-ASRO configuration."""
    model_cfg = TFTModelConfig(
        max_encoder_length=trial.suggest_int("max_encoder_length", 30, 90, step=10),
        max_prediction_length=base_cfg.model.max_prediction_length,
        hidden_size=trial.suggest_int("hidden_size", 32, 128, step=16),
        attention_head_size=trial.suggest_int("attention_head_size", 1, 8),
        dropout=trial.suggest_float("dropout", 0.05, 0.3, step=0.05),
        hidden_continuous_size=trial.suggest_int("hidden_continuous_size", 16, 64, step=8),
        quantiles=base_cfg.model.quantiles,
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        reduce_on_plateau_patience=4,
        gradient_clip_val=trial.suggest_float("gradient_clip_val", 0.1, 1.0, step=0.1),
    )

    asro_cfg = ASROConfig(
        lambda_vol=trial.suggest_float("lambda_vol", 0.1, 0.5, step=0.05),
        lambda_quantile=trial.suggest_float("lambda_quantile", 0.1, 0.5, step=0.05),
        risk_free_rate=0.0,
    )

    training_cfg = TrainingConfig(
        max_epochs=50,
        early_stopping_patience=8,
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        val_ratio=base_cfg.training.val_ratio,
        test_ratio=base_cfg.training.test_ratio,
        lookback_days=base_cfg.training.lookback_days,
        seed=base_cfg.training.seed,
        num_workers=base_cfg.training.num_workers,
        optuna_n_trials=base_cfg.training.optuna_n_trials,
        checkpoint_dir=str(Path(base_cfg.training.checkpoint_dir) / f"trial_{trial.number}"),
        best_model_path=str(Path(base_cfg.training.checkpoint_dir) / f"trial_{trial.number}" / "best.ckpt"),
    )

    return TFTASROConfig(
        embedding=base_cfg.embedding,
        sentiment=base_cfg.sentiment,
        lme=base_cfg.lme,
        model=model_cfg,
        asro=asro_cfg,
        training=training_cfg,
        feature_store=base_cfg.feature_store,
    )


def _objective(trial, base_cfg: TFTASROConfig, master_data: tuple) -> float:
    """
    Single Optuna trial: train a TFT variant and return the validation metric.

    Optimises for a composite score:
        score = -val_loss + 0.5 * directional_accuracy
    """
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping
    except ImportError:
        import pytorch_lightning as pl  # type: ignore[no-redef]
        from pytorch_lightning.callbacks import EarlyStopping  # type: ignore[no-redef]
    from optuna.integration import PyTorchLightningPruningCallback

    from deep_learning.data.dataset import build_datasets, create_dataloaders
    from deep_learning.models.tft_copper import create_tft_model
    from deep_learning.training.metrics import compute_all_metrics

    trial_cfg = create_trial_config(trial, base_cfg)
    master_df, tv_unknown, tv_known, target_cols = master_data

    try:
        training_ds, validation_ds, test_ds = build_datasets(
            master_df, tv_unknown, tv_known, target_cols, trial_cfg,
        )
        train_dl, val_dl, _ = create_dataloaders(training_ds, validation_ds, cfg=trial_cfg)
        model = create_tft_model(training_ds, trial_cfg, use_asro=True)
    except Exception as exc:
        logger.warning("Trial %d setup failed: %s", trial.number, exc)
        return float("inf")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=trial_cfg.training.early_stopping_patience, mode="min"),
        PyTorchLightningPruningCallback(trial, monitor="val_loss"),
    ]

    ckpt_dir = Path(trial_cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=trial_cfg.training.max_epochs,
        accelerator="auto",
        gradient_clip_val=trial_cfg.model.gradient_clip_val,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=20,
    )

    try:
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    except Exception as exc:
        logger.warning("Trial %d training failed: %s", trial.number, exc)
        return float("inf")

    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is None:
        return float("inf")

    return float(val_loss)


def run_hyperopt(
    base_cfg: Optional[TFTASROConfig] = None,
    n_trials: int = 50,
    study_name: str = "tft_asro_optuna",
    storage: Optional[str] = None,
) -> dict:
    """
    Launch Optuna hyperparameter search.

    Returns:
        Dict with best params, best value, and study summary.
    """
    import optuna
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl  # type: ignore[no-redef]

    from app.db import SessionLocal, init_db
    from deep_learning.data.feature_store import build_tft_dataframe

    if base_cfg is None:
        base_cfg = get_tft_config()

    init_db()
    pl.seed_everything(base_cfg.training.seed)

    logger.info("Building feature store for hyperopt ...")
    with SessionLocal() as session:
        master_data = build_tft_dataframe(session, base_cfg)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: _objective(trial, base_cfg, master_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info("Optuna best trial #%d: val_loss=%.6f", best.number, best.value)
    logger.info("Best params: %s", best.params)

    results_path = Path(base_cfg.training.checkpoint_dir) / "optuna_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
    }, indent=2))

    return {
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="TFT-ASRO hyperparameter optimisation")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", default="tft_asro_optuna")
    args = parser.parse_args()

    result = run_hyperopt(n_trials=args.n_trials, study_name=args.study_name)

    print("\n" + "=" * 60)
    print("HYPEROPT COMPLETE")
    print("=" * 60)
    print(f"Best trial: #{result['best_trial']}")
    print(f"Best val_loss: {result['best_value']:.6f}")
    for k, v in result["best_params"].items():
        print(f"  {k}: {v}")
