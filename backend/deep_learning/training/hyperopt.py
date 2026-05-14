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
import sys
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
    WeeklyLossConfig,
    get_tft_config,
)
from deep_learning.logging_utils import configure_cli_logging, suppress_lightning_noise

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hyperopt_diagnostics import (
    best_trial_preflight_check,
    compute_structural_invalidity_report,
    compute_trial_distribution_summary,
)

logger = logging.getLogger(__name__)

MIN_COMPLETED_TRIALS = 10
SHARPE_PRUNE_THRESHOLD = -0.3
FOLD_SHARPE_PRUNE_THRESHOLD = -1.0

KNOWN_GOOD_TRIAL_PARAMS = {
    "max_encoder_length": 50,
    "hidden_size": 48,
    "attention_head_size": 2,
    "dropout": 0.30,
    "hidden_continuous_size": 16,
    "learning_rate": 2e-4,
    "gradient_clip_val": 1.0,
    "weight_decay": 5e-5,
    "lambda_vol": 0.30,
    "lambda_quantile": 0.25,
    "lambda_madl": 0.40,
    "lambda_weekly_quantile": 0.70,
    "lambda_t1_quantile": 0.20,
    "lambda_dispersion": 0.35,
    "lambda_magnitude": 0.50,
    "lambda_naive": 0.50,
    "lambda_directional": 0.00,
    "batch_size": 32,
}


def _trial_state_counts(study) -> dict[str, int]:
    """Return lowercase Optuna trial-state counts for logs and artifacts."""
    counts: dict[str, int] = {}
    for trial in study.trials:
        state = getattr(trial.state, "name", str(trial.state)).lower()
        counts[state] = counts.get(state, 0) + 1
    return counts


def _best_finite_completed_trial(study):
    """Optuna raises when no trial completed; select the usable best trial safely."""
    completed = []
    for trial in study.trials:
        if getattr(trial.state, "name", None) != "COMPLETE":
            continue
        if trial.value is None or not np.isfinite(float(trial.value)):
            continue
        completed.append(trial)

    if not completed:
        return None

    return min(completed, key=lambda trial: float(trial.value))


def _finite_completed_trial_count(study) -> int:
    """Count completed trials with finite objective values."""
    return sum(
        1
        for trial in getattr(study, "trials", [])
        if getattr(trial.state, "name", None) == "COMPLETE"
        and trial.value is not None
        and np.isfinite(float(trial.value))
    )


def _weekly_pinball_loss(
    actual_path: np.ndarray,
    pred_path: np.ndarray,
    quantiles: tuple[float, ...],
    horizon: int = 5,
) -> float:
    actual = np.asarray(actual_path, dtype=np.float64)[:, :horizon].sum(axis=1)
    pred = np.asarray(pred_path, dtype=np.float64)[:, :horizon, :].sum(axis=1)
    q = np.asarray(quantiles, dtype=np.float64).reshape(1, -1)
    err = actual.reshape(-1, 1) - pred
    return float(np.maximum(q * err, (q - 1.0) * err).mean())


def _is_startup_protected(trial) -> bool:
    """Protect early trials until Optuna has enough finite completed evidence."""
    study = getattr(trial, "study", None)
    if study is None:
        return False
    return _finite_completed_trial_count(study) < MIN_COMPLETED_TRIALS


def _build_prune_diagnostics(study) -> tuple[dict[str, int], list[dict]]:
    prune_reasons = {
        "sharpe_prune": 0,
        "crossing_prune": 0,
        "median_prune": 0,
        "fold_sharpe_prune": 0,
        "weekly_magnitude_collapse": 0,
        "weekly_magnitude_explosion": 0,
        "weekly_interval_width_explosion": 0,
        "weekly_tail_width_explosion": 0,
        "weekly_raw_crossing_prune": 0,
        "weekly_overcoverage_width_explosion": 0,
        "error": 0,
    }
    fold_diagnostics: list[dict] = []
    metric_keys = (
        "avg_variance_ratio",
        "avg_directional_accuracy",
        "avg_val_sharpe",
        "avg_raw_quantile_crossing_rate",
        "avg_quantile_crossing_rate",
        "avg_raw_median_sort_gap",
        "avg_median_sort_gap",
        "avg_weekly_magnitude_ratio",
        "avg_weekly_pi80_coverage",
        "avg_weekly_pi80_width_ratio",
        "avg_weekly_pi96_width_ratio",
        "avg_weekly_raw_crossing_rate",
        "avg_weekly_sorted_crossing_rate",
        "avg_weekly_interval_score_80",
        "avg_weekly_interval_score_96",
        "fold_score_std",
    )

    for trial in study.trials:
        state = getattr(trial.state, "name", str(trial.state))
        user_attrs = getattr(trial, "user_attrs", {}) or {}
        if state == "PRUNED":
            reason = user_attrs.get("prune_reason", "median_prune")
            prune_reasons[reason] = prune_reasons.get(reason, 0) + 1

        metrics = {key: user_attrs[key] for key in metric_keys if key in user_attrs}
        if metrics:
            fold_diagnostics.append({
                "trial": trial.number,
                "state": state,
                **metrics,
            })

    return prune_reasons, fold_diagnostics


def _build_result_payload(study) -> dict:
    """Build the persisted hyperopt artifact without assuming a best trial exists."""
    trial_state_counts = _trial_state_counts(study)
    best = _best_finite_completed_trial(study)
    prune_reasons, fold_diagnostics = _build_prune_diagnostics(study)
    structural_report = compute_structural_invalidity_report(fold_diagnostics)
    distribution_summary = compute_trial_distribution_summary(fold_diagnostics)

    if best is None:
        return {
            "status": "no_finite_completed_trials",
            "best_trial": None,
            "best_value": None,
            "best_params": {},
            "n_trials": len(study.trials),
            "trial_state_counts": trial_state_counts,
            "prune_reasons": prune_reasons,
            "fold_diagnostics": fold_diagnostics,
            "structural_invalidity_report": structural_report,
            "trial_distribution_summary": distribution_summary,
            "best_trial_preflight": None,
            "message": (
                "No Optuna trials completed with a finite objective value; "
                "final training will use the known-good fallback config "
                "(weekly interval calibration warm-start parameters)."
            ),
        }

    best_diagnostics = next(
        (d for d in fold_diagnostics if d.get("trial") == best.number),
        {},
    )
    preflight = best_trial_preflight_check(best_diagnostics)
    return {
        "status": "completed",
        "best_trial": best.number,
        "best_value": float(best.value),
        "best_params": best.params,
        "n_trials": len(study.trials),
        "trial_state_counts": trial_state_counts,
        "prune_reasons": prune_reasons,
        "fold_diagnostics": fold_diagnostics,
        "structural_invalidity_report": structural_report,
        "trial_distribution_summary": distribution_summary,
        "best_trial_preflight": preflight,
    }


def _enqueue_known_good_trial(study, base_cfg: TFTASROConfig) -> bool:
    """
    Start a fresh Optuna study from the It.4 known-good parameter set.

    The static warm-start is intentionally a single enqueued trial; the
    remaining trials still explore the configured search space freely.
    """
    if getattr(study, "trials", []):
        return False

    study.enqueue_trial(dict(KNOWN_GOOD_TRIAL_PARAMS))
    logger.info("Enqueued known-good TFT-ASRO warm-start trial: %s", KNOWN_GOOD_TRIAL_PARAMS)
    return True


def create_trial_config(trial, base_cfg: TFTASROConfig) -> TFTASROConfig:
    """Map an Optuna trial to a TFT-ASRO configuration."""
    model_cfg = TFTModelConfig(
        max_encoder_length=trial.suggest_categorical("max_encoder_length", [40, 50, 60, 75, 90]),
        max_prediction_length=base_cfg.model.max_prediction_length,
        # Post-MRMR pruning (~60-80 features), smaller models generalise better.
        # 24 is viable now that feature count dropped from 200+ to ~60-80.
        hidden_size=trial.suggest_categorical("hidden_size", [24, 32, 48]),
        attention_head_size=trial.suggest_categorical("attention_head_size", [1, 2]),
        # Floor at 0.20: 313 samples with dropout<0.20 causes co-adaptation
        # and memorization (REG-2026-001).  Cap at 0.35: dropout>0.35 with
        # small hidden_size collapses the output range.
        dropout=trial.suggest_categorical("dropout", [0.20, 0.25, 0.30, 0.35]),
        # Paired reduction: with hidden=24-48 and ~60-80 features,
        # 8-16 is the sweet spot for continuous variable processing.
        hidden_continuous_size=trial.suggest_categorical("hidden_continuous_size", [8, 16]),
        quantiles=base_cfg.model.quantiles,
        # Range [1e-4, 1e-3]: LR < 1e-4 produces near-zero pred_std (VR=0.14);
        # LR > 1e-3 causes 1-epoch divergence. This band is the stable zone.
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 6e-4, log=True),
        reduce_on_plateau_patience=4,
        gradient_clip_val=trial.suggest_categorical("gradient_clip_val", [0.5, 1.0, 1.5]),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
    )

    asro_cfg = ASROConfig(
        # Floor at 0.25: three Optuna runs consistently selected 0.30-0.35.
        # Lower values let the model collapse to near-zero pred_std.
        lambda_vol=trial.suggest_float("lambda_vol", 0.25, 0.40, step=0.05),
        # lambda_quantile is the explicit w_quantile weight (w_sharpe = 1 - w_q)
        # Capped at 0.40 to ensure Sharpe (directional) component always has
        # ≥60% weight.  Higher values caused the "perfect calibration, coin-flip
        # direction" pathology where the model optimised volatility at the
        # expense of directional signal.
        lambda_quantile=trial.suggest_float("lambda_quantile", 0.25, 0.4, step=0.05),
        # MADL weight: how much the directional loss contributes relative to Sharpe.
        lambda_madl=trial.suggest_float("lambda_madl", 0.35, 0.60, step=0.05),
        risk_free_rate=0.0,
    )

    weekly_loss_cfg = WeeklyLossConfig(
        lambda_weekly_quantile=trial.suggest_float("lambda_weekly_quantile", 0.60, 0.80, step=0.05),
        lambda_t1_quantile=trial.suggest_float("lambda_t1_quantile", 0.10, 0.25, step=0.05),
        lambda_dispersion=trial.suggest_float("lambda_dispersion", 0.25, 0.45, step=0.05),
        lambda_magnitude=trial.suggest_float("lambda_magnitude", 0.25, 0.75, step=0.05),
        lambda_naive=trial.suggest_float("lambda_naive", 0.25, 0.75, step=0.05),
        lambda_directional=trial.suggest_float("lambda_directional", 0.00, 0.08, step=0.02),
    )

    training_cfg = TrainingConfig(
        # CI budget: 3h limit @ CPU-only.
        # 15 trials × 3 folds × 25 epochs ≈ 108 min → leaves 70 min for final trainer.
        # (Was 35/6, causing 3h+ timeout with 20 trials.)
        max_epochs=25,
        early_stopping_patience=4,
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
        forecast=base_cfg.forecast,
        weekly_loss=weekly_loss_cfg,
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
    suppress_lightning_noise()
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
    from deep_learning.training.callbacks import CurriculumLossScheduler
    from deep_learning.training.metrics import (
        compute_weekly_metrics,
        monotonic_quantiles_np,
        quantile_crossing_rate,
        quantile_median_sort_gap,
        select_prediction_horizon,
    )

    trial_cfg = create_trial_config(trial, base_cfg)
    protect_trial = _is_startup_protected(trial)
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
    fold_crossing_list: list[float] = []
    fold_raw_crossing_list: list[float] = []
    fold_median_gap_list: list[float] = []
    fold_raw_median_gap_list: list[float] = []
    fold_weekly_objectives: list[float] = []
    fold_weekly_mr_list: list[float] = []
    fold_weekly_pi80_coverage_list: list[float] = []
    fold_weekly_pi80_width_ratio_list: list[float] = []
    fold_weekly_pi96_width_ratio_list: list[float] = []
    fold_weekly_raw_crossing_list: list[float] = []
    fold_weekly_sorted_crossing_list: list[float] = []
    fold_weekly_interval_score_80_list: list[float] = []
    fold_weekly_interval_score_96_list: list[float] = []

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
        if trial_cfg.forecast.primary_horizon_days != 5:
            callbacks.append(
                CurriculumLossScheduler(
                    warmup_epochs=5,
                    initial_lambda_quantile=0.55,
                    target_lambda_quantile=trial_cfg.asro.lambda_quantile,
                    initial_lambda_madl=0.10,
                    target_lambda_madl=trial_cfg.asro.lambda_madl,
                )
            )

        ckpt_dir = Path(trial_cfg.training.checkpoint_dir) / f"fold_{fold_idx}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        log_steps = max(1, min(5, len(fold_train_dl)))

        trainer = pl.Trainer(
            max_epochs=trial_cfg.training.max_epochs,
            accelerator="auto",
            gradient_clip_val=trial_cfg.model.gradient_clip_val,
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            log_every_n_steps=log_steps,
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
        fold_crossing_rate = 0.0
        fold_median_gap = 0.0
        fold_weekly_objective = fold_val_loss
        fold_weekly_mr = 1.0
        fold_weekly_pi80_coverage = 0.0
        fold_weekly_pi80_width_ratio = 1.0
        fold_weekly_pi96_width_ratio = 1.0
        fold_weekly_raw_crossing = 0.0
        fold_weekly_sorted_crossing = 0.0
        fold_weekly_interval_score_80 = 0.0
        fold_weekly_interval_score_96 = 0.0

        try:
            pred_tensor = model.predict(fold_val_dl, mode="quantiles")
            if hasattr(pred_tensor, "cpu"):
                pred_np = pred_tensor.cpu().numpy()
            else:
                pred_np = np.array(pred_tensor)

            median_idx = len(trial_cfg.model.quantiles) // 2
            if pred_np.ndim != 3:
                raise ValueError(f"Expected quantile prediction tensor [n,horizon,q], got {pred_np.shape}")
            if pred_np.shape[1] < trial_cfg.forecast.primary_horizon_days:
                raise ValueError(
                    f"Prediction horizon too short: {pred_np.shape[1]} < {trial_cfg.forecast.primary_horizon_days}"
                )
            ordered_pred_np = monotonic_quantiles_np(pred_np, median_idx=median_idx)
            raw_pred_t1 = pred_np[:, 0, :]
            pred_t1 = ordered_pred_np[:, 0, :]
            y_pred = pred_t1[:, median_idx]
            fold_crossing_rate = quantile_crossing_rate(pred_t1)
            fold_raw_crossing_rate = quantile_crossing_rate(raw_pred_t1)
            _, fold_median_gap = quantile_median_sort_gap(pred_t1, median_idx)
            _, fold_raw_median_gap = quantile_median_sort_gap(raw_pred_t1, median_idx)

            y_actual_parts = []
            for batch in fold_val_dl:
                y_actual_parts.append(
                    batch[1][0] if isinstance(batch[1], (list, tuple)) else batch[1]
                )
            y_actual_path = torch.cat(y_actual_parts).cpu().numpy()
            y_actual = select_prediction_horizon(y_actual_path, horizon_idx=0)

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

            n_path = min(len(y_actual_path), len(pred_np))
            weekly = compute_weekly_metrics(
                y_actual_path[:n_path],
                pred_np[:n_path],
                quantiles=trial_cfg.model.quantiles,
                horizon=trial_cfg.forecast.primary_horizon_days,
            )
            weekly_pinball = _weekly_pinball_loss(
                y_actual_path[:n_path],
                ordered_pred_np[:n_path],
                tuple(trial_cfg.model.quantiles),
                horizon=trial_cfg.forecast.primary_horizon_days,
            )
            fold_weekly_mr = float(weekly.get("weekly_magnitude_ratio", 1.0))
            fold_weekly_pi80_coverage = float(weekly.get("weekly_pi80_coverage", 0.0))
            fold_weekly_pi80_width_ratio = float(weekly.get("weekly_pi80_width_ratio", 1.0))
            fold_weekly_pi96_width_ratio = float(weekly.get("weekly_pi96_width_ratio", 1.0))
            fold_weekly_raw_crossing = float(weekly.get("weekly_raw_quantile_crossing_rate", 0.0))
            fold_weekly_sorted_crossing = float(
                weekly.get("weekly_ordered_quantile_crossing_rate", 0.0)
            )
            fold_weekly_interval_score_80 = float(weekly.get("weekly_interval_score_80", 0.0))
            fold_weekly_interval_score_96 = float(weekly.get("weekly_interval_score_96", 0.0))
            weekly_actual_std = float(weekly.get("weekly_actual_std", 0.0))
            interval_score_penalty = fold_weekly_interval_score_80 / (weekly_actual_std + 1e-8)
            interval_score_96_penalty = fold_weekly_interval_score_96 / (weekly_actual_std + 1e-8)
            coverage_penalty = abs(fold_weekly_pi80_coverage - 0.80)
            width_penalty = max(0.0, fold_weekly_pi80_width_ratio - 1.5)
            tail_width_penalty = max(0.0, fold_weekly_pi96_width_ratio - 3.0)
            fold_weekly_objective = (
                0.35 * weekly_pinball
                + 0.15 * (1.0 - float(weekly.get("weekly_directional_accuracy", 0.5)))
                + 0.50 * abs(np.log(fold_weekly_mr + 1e-8))
                + 0.20 * coverage_penalty
                + 0.25 * width_penalty
                + 0.35 * tail_width_penalty
                + 0.10 * interval_score_penalty
                + 0.05 * interval_score_96_penalty
                + 0.25 * fold_weekly_sorted_crossing
            )
        except Exception as exc:
            logger.warning(
                "Trial %d fold %d metrics failed: %s", trial.number, fold_idx, exc
            )
            return float("inf")

        fold_vr_list.append(fold_vr)
        fold_da_list.append(fold_da)
        fold_sharpe_list.append(fold_sharpe)
        fold_crossing_list.append(fold_crossing_rate)
        fold_raw_crossing_list.append(fold_raw_crossing_rate)
        fold_median_gap_list.append(fold_median_gap)
        fold_raw_median_gap_list.append(fold_raw_median_gap)
        fold_weekly_objectives.append(fold_weekly_objective)
        fold_weekly_mr_list.append(fold_weekly_mr)
        fold_weekly_pi80_coverage_list.append(fold_weekly_pi80_coverage)
        fold_weekly_pi80_width_ratio_list.append(fold_weekly_pi80_width_ratio)
        fold_weekly_pi96_width_ratio_list.append(fold_weekly_pi96_width_ratio)
        fold_weekly_raw_crossing_list.append(fold_weekly_raw_crossing)
        fold_weekly_sorted_crossing_list.append(fold_weekly_sorted_crossing)
        fold_weekly_interval_score_80_list.append(fold_weekly_interval_score_80)
        fold_weekly_interval_score_96_list.append(fold_weekly_interval_score_96)

        # Incorporate DA directly into fold_score as a reward (not just penalty).
        # DA > 50% (coin-flip) is rewarded, < 50% penalised.
        # This ensures the hyperopt objective actively selects for directional
        # accuracy, not just low calibration loss.
        da_baseline = 0.50
        da_adjustment = (fold_da - da_baseline) * 2.0  # reward when DA > 50%, penalty when < 50%
        crossing_penalty = 2.0 * max(0.0, fold_crossing_rate - 0.05)
        median_gap_penalty = 5.0 * max(0.0, fold_median_gap - 0.005)
        fold_score = fold_weekly_objective + fold_vr_penalty + crossing_penalty + median_gap_penalty - da_adjustment
        fold_scores.append(fold_score)

        logger.debug(
            "Trial %d fold %d/%d: val_loss=%.4f vr=%.3f da=%.1f%% "
            "sharpe=%.4f q_cross=%.3f q_gap=%.4f",
            trial.number, fold_idx + 1, n_folds,
            fold_val_loss, fold_vr, fold_da * 100, fold_sharpe,
            fold_crossing_rate, fold_median_gap,
        )

        # Per-fold Sharpe pruning: if a fold has deeply negative Sharpe,
        # the trial is systematically predicting the wrong direction for
        # that market regime — no point continuing to subsequent folds.
        if (
            fold_sharpe < FOLD_SHARPE_PRUNE_THRESHOLD
            and fold_idx >= 1
            and not protect_trial
        ):
            logger.warning(
                "Trial %d PRUNED at fold %d: fold_sharpe=%.4f < %.1f",
                trial.number, fold_idx + 1, fold_sharpe,
                FOLD_SHARPE_PRUNE_THRESHOLD,
            )
            trial.set_user_attr("prune_reason", "fold_sharpe_prune")
            raise optuna.exceptions.TrialPruned()

        if fold_weekly_mr < 0.40 and fold_idx >= 1 and not protect_trial:
            logger.warning(
                "Trial %d PRUNED at fold %d: weekly_magnitude_ratio=%.4f < 0.40",
                trial.number, fold_idx + 1, fold_weekly_mr,
            )
            trial.set_user_attr("prune_reason", "weekly_magnitude_collapse")
            raise optuna.exceptions.TrialPruned()

        if fold_weekly_mr > 3.0 and fold_idx >= 1 and not protect_trial:
            logger.warning(
                "Trial %d PRUNED at fold %d: weekly_magnitude_ratio=%.4f > 3.0",
                trial.number, fold_idx + 1, fold_weekly_mr,
            )
            trial.set_user_attr("prune_reason", "weekly_magnitude_explosion")
            raise optuna.exceptions.TrialPruned()

        if fold_weekly_pi80_width_ratio > 4.0 and fold_idx >= 1 and not protect_trial:
            logger.warning(
                "Trial %d PRUNED at fold %d: weekly_pi80_width_ratio=%.4f > 4.0",
                trial.number, fold_idx + 1, fold_weekly_pi80_width_ratio,
            )
            trial.set_user_attr("prune_reason", "weekly_interval_width_explosion")
            raise optuna.exceptions.TrialPruned()

        if fold_weekly_pi96_width_ratio > 3.0 and fold_idx >= 1 and not protect_trial:
            logger.warning(
                "Trial %d PRUNED at fold %d: weekly_pi96_width_ratio=%.4f > 3.0",
                trial.number, fold_idx + 1, fold_weekly_pi96_width_ratio,
            )
            trial.set_user_attr("prune_reason", "weekly_tail_width_explosion")
            raise optuna.exceptions.TrialPruned()

        if fold_weekly_raw_crossing > 0.05 and fold_idx >= 1 and not protect_trial:
            logger.warning(
                "Trial %d PRUNED at fold %d: weekly raw crossing=%.4f > 0.05",
                trial.number, fold_idx + 1, fold_weekly_raw_crossing,
            )
            trial.set_user_attr("prune_reason", "weekly_raw_crossing_prune")
            raise optuna.exceptions.TrialPruned()

        if (
            fold_weekly_pi80_coverage >= 0.98
            and fold_weekly_pi80_width_ratio > 3.0
            and fold_idx >= 1
            and not protect_trial
        ):
            logger.warning(
                "Trial %d PRUNED at fold %d: overcoverage=%.4f width_ratio=%.4f",
                trial.number, fold_idx + 1,
                fold_weekly_pi80_coverage, fold_weekly_pi80_width_ratio,
            )
            trial.set_user_attr("prune_reason", "weekly_overcoverage_width_explosion")
            raise optuna.exceptions.TrialPruned()

        # Report running average so MedianPruner can kill bad trials early
        running_avg = float(np.mean(fold_scores))
        trial.report(running_avg, fold_idx)
        if trial.should_prune() and not protect_trial:
            trial.set_user_attr("prune_reason", "median_prune")
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
    avg_crossing = float(np.mean(fold_crossing_list)) if fold_crossing_list else 0.0
    avg_raw_crossing = (
        float(np.mean(fold_raw_crossing_list)) if fold_raw_crossing_list else 0.0
    )
    avg_median_gap = float(np.mean(fold_median_gap_list)) if fold_median_gap_list else 0.0
    avg_raw_median_gap = (
        float(np.mean(fold_raw_median_gap_list)) if fold_raw_median_gap_list else 0.0
    )
    avg_weekly_mr = float(np.mean(fold_weekly_mr_list)) if fold_weekly_mr_list else 1.0
    avg_weekly_pi80_coverage = (
        float(np.mean(fold_weekly_pi80_coverage_list))
        if fold_weekly_pi80_coverage_list
        else 0.0
    )
    avg_weekly_pi80_width_ratio = (
        float(np.mean(fold_weekly_pi80_width_ratio_list))
        if fold_weekly_pi80_width_ratio_list
        else 1.0
    )
    avg_weekly_pi96_width_ratio = (
        float(np.mean(fold_weekly_pi96_width_ratio_list))
        if fold_weekly_pi96_width_ratio_list
        else 1.0
    )
    avg_weekly_raw_crossing = (
        float(np.mean(fold_weekly_raw_crossing_list))
        if fold_weekly_raw_crossing_list
        else 0.0
    )
    avg_weekly_sorted_crossing = (
        float(np.mean(fold_weekly_sorted_crossing_list))
        if fold_weekly_sorted_crossing_list
        else 0.0
    )
    avg_weekly_interval_score_80 = (
        float(np.mean(fold_weekly_interval_score_80_list))
        if fold_weekly_interval_score_80_list
        else 0.0
    )
    avg_weekly_interval_score_96 = (
        float(np.mean(fold_weekly_interval_score_96_list))
        if fold_weekly_interval_score_96_list
        else 0.0
    )

    # High fold-score variance = trial is unreliable (works in one regime, fails in another)
    consistency_penalty = (
        float(np.std(fold_scores)) * 0.5 if len(fold_scores) > 1 else 0.0
    )

    trial.set_user_attr("avg_variance_ratio", round(avg_vr, 4))
    trial.set_user_attr("avg_directional_accuracy", round(avg_da, 4))
    trial.set_user_attr("avg_val_sharpe", round(avg_sharpe, 4))
    trial.set_user_attr("avg_raw_quantile_crossing_rate", round(avg_raw_crossing, 4))
    trial.set_user_attr("avg_quantile_crossing_rate", round(avg_crossing, 4))
    trial.set_user_attr("avg_raw_median_sort_gap", round(avg_raw_median_gap, 4))
    trial.set_user_attr("avg_median_sort_gap", round(avg_median_gap, 4))
    trial.set_user_attr("avg_weekly_magnitude_ratio", round(avg_weekly_mr, 4))
    trial.set_user_attr("avg_weekly_pi80_coverage", round(avg_weekly_pi80_coverage, 4))
    trial.set_user_attr("avg_weekly_pi80_width_ratio", round(avg_weekly_pi80_width_ratio, 4))
    trial.set_user_attr("avg_weekly_pi96_width_ratio", round(avg_weekly_pi96_width_ratio, 4))
    trial.set_user_attr("avg_weekly_raw_crossing_rate", round(avg_weekly_raw_crossing, 4))
    trial.set_user_attr("avg_weekly_sorted_crossing_rate", round(avg_weekly_sorted_crossing, 4))
    trial.set_user_attr("avg_weekly_interval_score_80", round(avg_weekly_interval_score_80, 4))
    trial.set_user_attr("avg_weekly_interval_score_96", round(avg_weekly_interval_score_96, 4))
    trial.set_user_attr(
        "fold_score_std",
        round(float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0, 4),
    )

    # Hard prune: avg Sharpe negative across folds = systematically wrong
    if avg_sharpe < SHARPE_PRUNE_THRESHOLD and not protect_trial:
        logger.warning(
            "Trial %d PRUNED: avg_sharpe=%.4f < %.1f across %d folds (DA=%.1f%%)",
            trial.number, avg_sharpe, SHARPE_PRUNE_THRESHOLD, n_folds, avg_da * 100,
        )
        trial.set_user_attr("prune_reason", "sharpe_prune")
        raise optuna.exceptions.TrialPruned()

    if avg_crossing > 0.001 or avg_weekly_sorted_crossing > 0.001:
        raise RuntimeError(
            "Monotonic quantile transform produced public crossings: "
            f"daily={avg_crossing:.6f}, weekly={avg_weekly_sorted_crossing:.6f}"
        )

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
    suppress_lightning_noise()
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
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=max(5, n_trials // 3),
            n_warmup_steps=1,
        ),
    )
    _enqueue_known_good_trial(study, base_cfg)

    study.optimize(
        lambda trial: _objective(trial, base_cfg, master_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Save alongside best_tft_asro.ckpt (tft/ root) so upload_tft_artifacts picks it up.
    results_path = Path(base_cfg.training.best_model_path).parent / "optuna_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    result = _build_result_payload(study)
    results_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    logger.info(
        "Optuna structural invalidity report: %s",
        result.get("structural_invalidity_report"),
    )
    logger.info(
        "Optuna trial distribution summary: %s",
        result.get("trial_distribution_summary"),
    )
    logger.info("Optuna best trial preflight: %s", result.get("best_trial_preflight"))

    if result["best_trial"] is None:
        logger.warning(
            "Optuna finished without a finite completed trial; state counts=%s. "
            "Wrote fallback artifact to %s",
            result["trial_state_counts"],
            results_path,
        )
    else:
        logger.info(
            "Optuna best trial #%d: weekly_objective=%.6f",
            result["best_trial"],
            result["best_value"],
        )
        logger.info("Best params: %s", result["best_params"])

    structural_report = result.get("structural_invalidity_report") or {}
    if structural_report.get("verdict") == "STRUCTURAL_FAILURE":
        raise RuntimeError(structural_report.get("next_action", "Structural failure in hyperopt."))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configure_cli_logging(logging.INFO)

    parser = argparse.ArgumentParser(description="TFT-ASRO hyperparameter optimisation")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", default="tft_asro_optuna")
    args = parser.parse_args()

    result = run_hyperopt(n_trials=args.n_trials, study_name=args.study_name)

    print("\n" + "=" * 60)
    print("HYPEROPT COMPLETE")
    print("=" * 60)
    if result["best_trial"] is None:
        print(f"Status: {result['status']}")
        print(result["message"])
        if result.get("trial_state_counts"):
            counts = ", ".join(
                f"{state}={count}"
                for state, count in sorted(result["trial_state_counts"].items())
            )
            print(f"Trial states: {counts}")
    else:
        print(f"Best trial: #{result['best_trial']}")
        print(f"Best weekly objective: {result['best_value']:.6f}")
        for k, v in result["best_params"].items():
            print(f"  {k}: {v}")
