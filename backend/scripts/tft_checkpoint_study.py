"""Evaluate saved TFT checkpoints on one immutable validation/test snapshot.

This is a diagnostic tool: checkpoint ranking is computed from validation
metrics only.  Test metrics are printed separately to measure whether the
validation rule transfers; they never participate in the ranking.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app.quality_gate import evaluate_quality_gate_metrics
from deep_learning.config import get_tft_config
from deep_learning.data.dataset import build_datasets, create_dataloaders
from deep_learning.models.tft_copper import load_tft_model
from deep_learning.training.direction_model import (
    apply_weekly_direction_model,
    predict_weekly_direction,
)
from deep_learning.training.metrics import (
    apply_weekly_sign_correction_np,
    cumulative_horizon,
    directional_accuracy,
    fit_direction_sign_calibration,
    fit_weekly_interval_scale,
)
from deep_learning.training.trainer import (
    WEEKLY_INTERVAL_CONDITIONING_FEATURE,
    _build_uniform_checkpoint_soup,
    _compute_test_metrics_from_quantiles,
    _validate_quantile_prediction_shape,
    _weekly_interval_conditioning_values,
)


def _actual_path(dataloader) -> np.ndarray:
    parts = [
        batch[1][0] if isinstance(batch[1], (list, tuple)) else batch[1]
        for batch in dataloader
    ]
    return torch.cat(parts).cpu().numpy()


def _predict_quantiles(model, dataloader, cfg) -> np.ndarray:
    prediction = model.predict(
        dataloader,
        mode="quantiles",
        trainer_kwargs={"logger": False, "enable_checkpointing": False},
    )
    prediction_np = prediction.cpu().numpy()
    _validate_quantile_prediction_shape(prediction_np, cfg)
    return prediction_np


def _prepare_candidate(
    *,
    prediction: np.ndarray,
    actual: np.ndarray,
    weekly_direction_model: dict,
    direction_probability: np.ndarray,
    conditioning_values: np.ndarray | None,
    cfg,
) -> tuple[np.ndarray, dict, dict, dict, dict]:
    direction_calibration = fit_direction_sign_calibration(
        actual,
        prediction,
        horizon=cfg.forecast.primary_horizon_days,
    )
    oriented = prediction * int(
        direction_calibration.get("direction_sign_multiplier", 1)
    )
    oriented = apply_weekly_sign_correction_np(
        oriented,
        float(direction_calibration.get("weekly_sign_threshold", 0.0)),
        horizon=cfg.forecast.primary_horizon_days,
    )

    selected_direction_model = copy.deepcopy(weekly_direction_model)
    selected_direction_model["enabled"] = False
    selected_direction_model["reason"] = "validation_selection_rejected"
    if selected_direction_model.get("coef") and len(direction_probability) == len(
        oriented
    ):
        candidate = apply_weekly_direction_model(
            oriented,
            direction_probability,
            threshold=float(
                selected_direction_model.get("decision_threshold", 0.50)
            ),
            horizon=cfg.forecast.primary_horizon_days,
        )
        actual_weekly = cumulative_horizon(
            actual,
            horizon=cfg.forecast.primary_horizon_days,
        )
        median_idx = len(cfg.model.quantiles) // 2
        base_da = directional_accuracy(
            actual_weekly,
            cumulative_horizon(
                oriented[:, :, median_idx],
                horizon=cfg.forecast.primary_horizon_days,
            ),
        )
        candidate_da = directional_accuracy(
            actual_weekly,
            cumulative_horizon(
                candidate[:, :, median_idx],
                horizon=cfg.forecast.primary_horizon_days,
            ),
        )
        candidate_rate = float(np.mean(direction_probability >= 0.50))
        selected_direction_model.update(
            {
                "validation_base_weekly_da": float(base_da),
                "validation_weekly_da": float(candidate_da),
                "validation_pred_positive_rate": candidate_rate,
            }
        )
        if (
            candidate_da >= 0.51
            and candidate_da >= base_da + 0.01
            and 0.25 <= candidate_rate <= 0.75
        ):
            selected_direction_model["enabled"] = True
            selected_direction_model["reason"] = "validation_improved"
            oriented = candidate

    interval_calibration = fit_weekly_interval_scale(
        actual,
        oriented,
        quantiles=tuple(cfg.model.quantiles),
        horizon=cfg.forecast.primary_horizon_days,
        weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
        conditioning_values=conditioning_values,
        conditioning_feature=(
            WEEKLY_INTERVAL_CONDITIONING_FEATURE
            if conditioning_values is not None
            else None
        ),
    )
    metrics = _compute_test_metrics_from_quantiles(
        actual,
        oriented,
        cfg,
        weekly_interval_scale=float(
            interval_calibration.get("weekly_interval_scale", 1.0)
        ),
        weekly_interval_calibration=interval_calibration,
        weekly_interval_conditioning_values=conditioning_values,
    )
    return (
        oriented,
        metrics,
        direction_calibration,
        interval_calibration,
        selected_direction_model,
    )


def _validation_rank(metrics: dict, checkpoint_loss: float) -> tuple:
    """Rank without test labels: fewer contract misses, then safer margins."""
    _passed, reasons = evaluate_quality_gate_metrics(metrics)

    # Proper interval score and bounded scale are continuous tie-breakers.
    return (
        len(reasons),
        float(metrics["weekly_pi80_interval_score"]),
        float(metrics["weekly_mae_vs_naive_zero"]),
        -float(metrics["weekly_sharpe_ratio"]),
        float(checkpoint_loss),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--include-soups", action="store_true")
    parser.add_argument("--candidate", help="Evaluate only one checkpoint filename")
    parser.add_argument(
        "--weekly-median-cap-factor",
        type=float,
        default=1.0,
        help="Diagnostic multiplier applied to the recorded weekly median cap",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    snapshot_path = artifact_dir / "feature_snapshot.pkl"
    snapshot_meta = json.loads(
        snapshot_path.with_suffix(".pkl.json").read_text(encoding="utf-8")
    )
    run_meta = json.loads(
        (artifact_dir / "tft_metadata.json").read_text(encoding="utf-8")
    )

    frame = pd.read_pickle(snapshot_path)
    cfg = get_tft_config()
    effective_weekly_cap = (
        float(run_meta["config"]["weekly_median_cap"])
        * args.weekly_median_cap_factor
    )
    if effective_weekly_cap <= 0.0:
        raise ValueError("The effective weekly median cap must be positive")
    cfg = replace(
        cfg,
        weekly_loss=replace(
            cfg.weekly_loss,
            weekly_median_cap=effective_weekly_cap,
        ),
    )
    training_ds, validation_ds, test_ds = build_datasets(
        frame,
        snapshot_meta["time_varying_unknown_reals"],
        snapshot_meta["time_varying_known_reals"],
        snapshot_meta["target_cols"],
        cfg,
    )
    _, val_dl, test_dl = create_dataloaders(
        training_ds,
        validation_ds,
        test_ds,
        cfg,
    )
    val_actual = _actual_path(val_dl)
    test_actual = _actual_path(test_dl)

    n_rows = len(frame)
    test_size = int(n_rows * cfg.training.test_ratio)
    val_size = int(n_rows * cfg.training.val_ratio)
    train_size = n_rows - val_size - test_size
    train_cutoff = int(frame["time_idx"].iloc[train_size - 1])
    val_cutoff = int(frame["time_idx"].iloc[train_size + val_size - 1])
    max_time_idx = int(frame["time_idx"].iloc[-1])

    base_direction_model = run_meta["weekly_direction_model"]
    val_direction_probability = predict_weekly_direction(
        base_direction_model,
        frame,
        start_exclusive=train_cutoff - 1,
        end_inclusive=val_cutoff - cfg.forecast.primary_horizon_days,
    )
    test_direction_probability = predict_weekly_direction(
        base_direction_model,
        frame,
        start_exclusive=val_cutoff - 1,
        end_inclusive=max_time_idx - cfg.forecast.primary_horizon_days,
    )
    val_conditioning = _weekly_interval_conditioning_values(
        frame,
        feature_name=WEEKLY_INTERVAL_CONDITIONING_FEATURE,
        start_exclusive=train_cutoff - 1,
        end_inclusive=val_cutoff - cfg.forecast.primary_horizon_days,
        expected_count=len(val_actual),
    )
    test_conditioning = _weekly_interval_conditioning_values(
        frame,
        feature_name=WEEKLY_INTERVAL_CONDITIONING_FEATURE,
        start_exclusive=val_cutoff - 1,
        end_inclusive=max_time_idx - cfg.forecast.primary_horizon_days,
        expected_count=len(test_actual),
    )

    checkpoint_specs = []
    checkpoints = sorted(
        (artifact_dir / "checkpoints").glob("tft-asro-*.ckpt"),
        key=lambda path: float(path.stem.rsplit("=", 1)[-1]),
    )
    for checkpoint in checkpoints:
        checkpoint_specs.append(
            (checkpoint, float(checkpoint.stem.rsplit("=", 1)[-1]))
        )
    if args.include_soups:
        for count in range(2, len(checkpoints) + 1):
            source = checkpoints[:count]
            soup_path = (
                artifact_dir
                / "checkpoints"
                / f"validation-top-{count}-soup.ckpt"
            )
            _build_uniform_checkpoint_soup(source, soup_path)
            checkpoint_specs.append(
                (
                    soup_path,
                    float(
                        np.mean(
                            [
                                float(path.stem.rsplit("=", 1)[-1])
                                for path in source
                            ]
                        )
                    ),
                )
            )

    if args.candidate:
        checkpoint_specs = [
            spec for spec in checkpoint_specs if spec[0].name == args.candidate
        ]
        if not checkpoint_specs:
            raise FileNotFoundError(
                f"Checkpoint candidate not found: {args.candidate}"
            )

    rows = []
    for checkpoint, loss in checkpoint_specs:
        model = load_tft_model(str(checkpoint))
        val_prediction = _predict_quantiles(model, val_dl, cfg)
        (
            _,
            val_metrics,
            direction_calibration,
            interval_calibration,
            selected_direction_model,
        ) = _prepare_candidate(
            prediction=val_prediction,
            actual=val_actual,
            weekly_direction_model=base_direction_model,
            direction_probability=val_direction_probability,
            conditioning_values=val_conditioning,
            cfg=cfg,
        )

        test_prediction = _predict_quantiles(model, test_dl, cfg)
        test_prediction *= int(
            direction_calibration.get("direction_sign_multiplier", 1)
        )
        test_prediction = apply_weekly_sign_correction_np(
            test_prediction,
            float(direction_calibration.get("weekly_sign_threshold", 0.0)),
            horizon=cfg.forecast.primary_horizon_days,
        )
        if selected_direction_model.get("enabled"):
            test_prediction = apply_weekly_direction_model(
                test_prediction,
                test_direction_probability,
                threshold=float(
                    selected_direction_model.get("decision_threshold", 0.50)
                ),
                horizon=cfg.forecast.primary_horizon_days,
            )
        test_metrics = _compute_test_metrics_from_quantiles(
            test_actual,
            test_prediction,
            cfg,
            weekly_interval_scale=float(
                interval_calibration["weekly_interval_scale"]
            ),
            weekly_interval_calibration=interval_calibration,
            weekly_interval_conditioning_values=test_conditioning,
        )
        rank = _validation_rank(val_metrics, loss)
        validation_passed, validation_reasons = evaluate_quality_gate_metrics(
            val_metrics
        )
        test_passed, test_reasons = evaluate_quality_gate_metrics(test_metrics)
        rows.append(
            {
                "checkpoint": checkpoint.name,
                "checkpoint_loss": loss,
                "weekly_median_cap_factor": args.weekly_median_cap_factor,
                "effective_weekly_median_cap": effective_weekly_cap,
                "validation_rank": rank,
                "validation_gate_passed": validation_passed,
                "validation_gate_reasons": validation_reasons,
                "validation": val_metrics,
                "test_diagnostic_gate_passed": test_passed,
                "test_diagnostic_gate_reasons": test_reasons,
                "test_diagnostic": test_metrics,
            }
        )

    rows.sort(key=lambda row: row["validation_rank"])
    print(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
