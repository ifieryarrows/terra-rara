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
        # Floor at 32: hidden=16 with dropout>0.3 leaves ~8 active neurons,
        # compressing output distribution and preventing amplitude learning.
        hidden_size=trial.suggest_int("hidden_size", 32, 64, step=16),
        attention_head_size=trial.suggest_int("attention_head_size", 1, 4),
        # Floor at 0.20: 313 samples with dropout<0.20 causes co-adaptation
        # and memorization (REG-2026-001).  Cap at 0.35: dropout>0.35 with
        # small hidden_size collapses the output range.
        dropout=trial.suggest_float("dropout", 0.20, 0.35, step=0.05),
        # Cap at 24: hidden_cont=32 doubled the VSN parameter surface and
        # contributed to overfitting in the 31-Mar regression.
        hidden_continuous_size=trial.suggest_int("hidden_continuous_size", 8, 24, step=8),
        quantiles=base_cfg.model.quantiles,
        # Range [1e-4, 1e-3]: LR < 1e-4 produces near-zero pred_std (VR=0.14);
        # LR > 1e-3 causes 1-epoch divergence. This band is the stable zone.
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        reduce_on_plateau_patience=4,
        gradient_clip_val=trial.suggest_float("gradient_clip_val", 0.5, 2.0, step=0.5),
    )

    asro_cfg = ASROConfig(
        # Floor at 0.25: three Optuna runs consistently selected 0.30-0.35.
        # Lower values let the model collapse to near-zero pred_std.
        lambda_vol=trial.suggest_float("lambda_vol", 0.25, 0.45, step=0.05),
        # lambda_quantile is the explicit w_quantile weight (w_sharpe = 1 - w_q)
        lambda_quantile=trial.suggest_float("lambda_quantile", 0.2, 0.6, step=0.05),
        risk_free_rate=0.0,
    )

    training_cfg = TrainingConfig(
        # Reduced for CV: each fold is smaller, needs fewer epochs.
        max_epochs=35,
        early_stopping_patience=6,
        # 16 gives 19 batches/epoch, 32 gives ~10.  64 produced only 4
        # batches/epoch with noisy gradients — removed after REG-2026-001.
        batch_size=trial.suggest_categorical("batch_size", [16, 32]),
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
    Single Optuna trial with Walk-Forward k-Fold Temporal CV.

    Each trial trains k models (one per fold) and returns the mean
    composite score.  This prevents overfitting to a single validation
    window — the core structural issue identified in REG-2026-001.

    Composite score per fold (lower is better):
        fold_score = val_loss + vr_penalty

    Final score:
        mean(fold_scores) + consistency_penalty + da_penalty

    After each fold, an intermediate score is reported to Optuna so
    the MedianPruner can kill clearly-bad trials early (after 1 fold
    instead of waiting for all 3).
    """
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping
    except ImportError:
        import pytorch_lightning as pl  # type: ignore[no-redef]
        from pytorch_lightning.callbacks import EarlyStopping  # type: ignore[no-redef]

    import optuna
    import numpy as np
    import torch
    from deep_learning.data.dataset import build_cv_folds, create_dataloaders
    from deep_learning.models.tft_copper import create_tft_model

    trial_cfg = create_trial_config(trial, base_cfg)
    master_df, tv_unknown, tv_known, target_cols, _ = master_data
    n_folds = getattr(trial_cfg.training, "cv_n_folds", 3)

    try:
        cv_folds = build_cv_folds(
            master_df, tv_unknown, tv_known, target_cols,
            trial_cfg, n_folds=n_folds,
        )
    except Exception as exc:
        logger.warning("Trial %d CV fold creation failed: %s", trial.number, exc)
        return float("inf")

    fold_scores: list[float] = []
    fold_da_list: list[float] = []
    fold_sharpe_list: list[float] = []
    fold_vr_list: list[float] = []

    for fold_idx, (fold_train_ds, fold_val_ds) in enumerate(cv_folds):
        # ---- setup ----
        try:
            fold_train_dl, fold_val_dl, _ = create_dataloaders(
                fold_train_ds, fold_val_ds, cfg=trial_cfg,
            )
            model = create_tft_model(fold_train_ds, trial_cfg, use_asro=True)
        except Exception as exc:
            logger.warning(
                "Trial %d fold %d setup failed: %s",
                trial.number, fold_idx, exc,
            )
            return float("inf")

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=trial_cfg.training.early_stopping_patience,
                mode="min",
            ),
        ]

        ckpt_dir = Path(trial_cfg.training.checkpoint_dir) / f"fold_{fold_idx}"
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

        # ---- train ----
        try:
            trainer.fit(model, train_dataloaders=fold_train_dl, val_dataloaders=fold_val_dl)
        except Exception as exc:
            logger.warning("Trial %d fold %d training failed: %s", trial.number, fold_idx, exc)
            return float("inf")

        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return float("inf")
        fold_val_loss = float(val_loss)

        # ---- per-fold metrics ----
        fold_vr_penalty = 0.0
        fold_da = 0.5
        fold_sharpe = 0.0
        fold_vr = 0.0

        try:
            pred_tensor = model.predict(fold_val_dl, mode="quantiles")
            if hasattr(pred_tensor, "cpu"):
                pred_np = pred_tensor.cpu().numpy()
            else:
                pred_np = np.array(pred_tensor)

            median_idx = len(trial_cfg.model.quantiles) // 2
            y_pred = pred_np[:, 0, median_idx] if pred_np.ndim == 3 else pred_np.flatten()

            y_actual_parts = []
            for batch in fold_val_dl:
                y_actual_parts.append(
                    batch[1][0] if isinstance(batch[1], (list, tuple)) else batch[1]
                )
            y_actual = torch.cat(y_actual_parts).cpu().numpy().flatten()

            fn = min(len(y_actual), len(y_pred))
            pred_std = float(y_pred[:fn].std())
            actual_std = float(y_actual[:fn].std())
            fold_vr = pred_std / actual_std if actual_std > 1e-9 else 0.0

            if fold_vr < 0.5:
                fold_vr_penalty = 2.0 * (1.0 - fold_vr / 0.5)
            elif fold_vr > 1.5:
                fold_vr_penalty = 0.5 * (fold_vr - 1.5)

            pred_sign = np.sign(y_pred[:fn])
            actual_sign = np.sign(y_actual[:fn])
            fold_da = float(np.mean(pred_sign == actual_sign))

            strategy_returns = np.sign(y_pred[:fn]) * y_actual[:fn]
            sr_mean = float(strategy_returns.mean())
            sr_std = float(strategy_returns.std()) + 1e-9
            fold_sharpe = sr_mean / sr_std
        except Exception as exc:
            logger.debug(
                "Trial %d fold %d metrics failed: %s", trial.number, fold_idx, exc
            )

        fold_vr_list.append(fold_vr)
        fold_da_list.append(fold_da)
        fold_sharpe_list.append(fold_sharpe)

        fold_score = fold_val_loss + fold_vr_penalty
        fold_scores.append(fold_score)

        logger.debug(
            "Trial %d fold %d/%d: val_loss=%.4f vr=%.3f da=%.1f%% sharpe=%.4f",
            trial.number, fold_idx + 1, n_folds,
            fold_val_loss, fold_vr, fold_da * 100, fold_sharpe,
        )

        # Report running average so MedianPruner can kill bad trials early
        running_avg = float(np.mean(fold_scores))
        trial.report(running_avg, fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Free GPU memory between folds
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- cross-fold aggregation ----
    avg_score = float(np.mean(fold_scores))
    avg_da = float(np.mean(fold_da_list)) if fold_da_list else 0.5
    avg_sharpe = float(np.mean(fold_sharpe_list)) if fold_sharpe_list else 0.0
    avg_vr = float(np.mean(fold_vr_list)) if fold_vr_list else 0.0

    # High fold-score variance = trial is unreliable (works in one regime, fails in another)
    consistency_penalty = (
        float(np.std(fold_scores)) * 0.5 if len(fold_scores) > 1 else 0.0
    )

    trial.set_user_attr("avg_variance_ratio", round(avg_vr, 4))
    trial.set_user_attr("avg_directional_accuracy", round(avg_da, 4))
    trial.set_user_attr("avg_val_sharpe", round(avg_sharpe, 4))
    trial.set_user_attr(
        "fold_score_std",
        round(float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0, 4),
    )

    # Hard prune: avg Sharpe negative across folds = systematically wrong
    if avg_sharpe < 0.0:
        logger.warning(
            "Trial %d PRUNED: avg_sharpe=%.4f < 0 across %d folds (DA=%.1f%%)",
            trial.number, avg_sharpe, n_folds, avg_da * 100,
        )
        raise optuna.exceptions.TrialPruned()

    # Soft penalty: avg DA below coin-flip
    da_penalty = 2.0 * max(0.0, 0.50 - avg_da) if avg_da < 0.50 else 0.0

    final_score = avg_score + consistency_penalty + da_penalty
    logger.info(
        "Trial %d [%d-fold CV]: avg_score=%.4f consistency=%.4f "
        "da_penalty=%.4f → final=%.4f | DA=%.1f%% Sharpe=%.3f VR=%.3f",
        trial.number, n_folds, avg_score, consistency_penalty, da_penalty,
        final_score, avg_da * 100, avg_sharpe, avg_vr,
    )
    return final_score


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

    # Save alongside best_tft_asro.ckpt (tft/ root) so upload_tft_artifacts picks it up.
    results_path = Path(base_cfg.training.best_model_path).parent / "optuna_results.json"
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
