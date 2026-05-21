"""Tests for Optuna hyperopt result handling."""

import inspect
from types import SimpleNamespace

import numpy as np

import deep_learning.training.hyperopt as hyperopt_module
from deep_learning.training.hyperopt import (
    KNOWN_GOOD_TRIAL_PARAMS,
    MIN_COMPLETED_TRIALS,
    _build_result_payload,
    _enqueue_known_good_trial,
    _finite_completed_trial_count,
    _fold_scale_diagnostic,
    _is_startup_protected,
    create_trial_config,
)
from deep_learning.config import get_tft_config


def _trial(number: int, state: str, value=None, params=None, user_attrs=None):
    return SimpleNamespace(
        number=number,
        state=SimpleNamespace(name=state),
        value=value,
        params=params or {},
        user_attrs=user_attrs or {},
    )


def _study(*trials):
    return SimpleNamespace(trials=list(trials))


class _RecordingTrial:
    number = 0

    def __init__(self):
        self.categorical_choices = {}
        self.float_ranges = {}

    def suggest_categorical(self, name, choices):
        self.categorical_choices[name] = list(choices)
        return choices[0]

    def suggest_float(self, name, low, high, step=None, log=False):
        self.float_ranges[name] = {
            "low": low,
            "high": high,
            "step": step,
            "log": log,
        }
        return low


def test_build_result_payload_handles_all_pruned_trials():
    result = _build_result_payload(
        _study(
            _trial(0, "PRUNED"),
            _trial(1, "PRUNED"),
            _trial(2, "PRUNED"),
        )
    )

    assert result["status"] == "no_finite_completed_trials"
    assert result["best_trial"] is None
    assert result["best_value"] is None
    assert result["best_params"] == {}
    assert result["n_trials"] == 3
    assert result["trial_state_counts"] == {"pruned": 3}
    assert result["prune_reasons"] == {
        "sharpe_prune": 0,
        "crossing_prune": 0,
        "median_prune": 3,
        "fold_sharpe_prune": 0,
        "weekly_magnitude_collapse": 0,
        "weekly_magnitude_explosion": 0,
        "weekly_positive_rate_explosion": 0,
        "weekly_pi80_undercoverage": 0,
        "weekly_mae_vs_naive_explosion": 0,
        "weekly_interval_width_explosion": 0,
        "weekly_tail_width_explosion": 0,
        "weekly_raw_crossing_prune": 0,
        "weekly_overcoverage_width_explosion": 0,
        "error": 0,
    }
    assert result["fold_diagnostics"] == []


def test_build_result_payload_selects_best_finite_completed_trial():
    result = _build_result_payload(
        _study(
            _trial(0, "COMPLETE", 1.25, {"hidden_size": 24}),
            _trial(1, "COMPLETE", float("inf"), {"hidden_size": 32}),
            _trial(2, "COMPLETE", 0.75, {"hidden_size": 48}),
        )
    )

    assert result["status"] == "completed"
    assert result["best_trial"] == 2
    assert result["best_value"] == 0.75
    assert result["best_params"] == {"hidden_size": 48}
    assert result["trial_state_counts"] == {"complete": 3}


def test_build_result_payload_marks_structural_failure_as_artifact_status():
    result = _build_result_payload(
        _study(
            _trial(
                3,
                "COMPLETE",
                21.361586,
                {"lambda_magnitude": 0.54},
                {
                    "avg_quantile_crossing_rate": 0.0,
                    "avg_weekly_magnitude_ratio": 21.0704,
                    "avg_weekly_pi80_coverage": 0.0127,
                    "avg_weekly_pi80_width_ratio": 0.3531,
                    "avg_weekly_mae_vs_naive_zero": 16.6480,
                    "avg_variance_ratio": 2.5120,
                    "avg_directional_accuracy": 0.5209,
                },
            )
        )
    )

    assert result["status"] == "structural_failure"
    assert result["best_trial"] == 3
    assert result["best_value"] == 21.361586
    assert result["best_params"] == {"lambda_magnitude": 0.54}
    assert result["structural_invalidity_report"]["verdict"] == "STRUCTURAL_FAILURE"
    assert result["best_trial_preflight"]["preflight_passed"] is False


def test_build_result_payload_records_prune_reasons_and_fold_diagnostics():
    result = _build_result_payload(
        _study(
            _trial(
                0,
                "PRUNED",
                user_attrs={
                    "prune_reason": "fold_sharpe_prune",
                    "avg_val_sharpe": -1.2,
                    "avg_directional_accuracy": 0.43,
                    "fold_score_std": 0.18,
                },
            ),
            _trial(
                1,
                "PRUNED",
                user_attrs={
                    "prune_reason": "crossing_prune",
                    "avg_quantile_crossing_rate": 0.28,
                    "avg_median_sort_gap": 0.014,
                },
            ),
        )
    )

    assert result["prune_reasons"]["fold_sharpe_prune"] == 1
    assert result["prune_reasons"]["crossing_prune"] == 1
    assert result["fold_diagnostics"] == [
        {
            "trial": 0,
            "state": "PRUNED",
            "avg_directional_accuracy": 0.43,
            "avg_val_sharpe": -1.2,
            "fold_score_std": 0.18,
        },
        {
            "trial": 1,
            "state": "PRUNED",
            "avg_quantile_crossing_rate": 0.28,
            "avg_median_sort_gap": 0.014,
        },
    ]


def test_build_result_payload_persists_fold_scale_diagnostics():
    scale_diagnostic = {
        "trial": 0,
        "fold": 1,
        "train_samples": 120,
        "val_samples": 24,
        "actual_weekly_std": 0.021,
        "actual_weekly_mean_abs": 0.018,
        "actual_weekly_abs_median": 0.015,
        "pred_weekly_mean_abs": 0.020,
        "pred_weekly_abs_median": 0.017,
        "weekly_magnitude_ratio": 1.1333,
        "weekly_mae_vs_naive_zero": 0.9000,
        "weekly_pred_min": -0.035,
        "weekly_pred_max": 0.041,
        "weekly_actual_min": -0.030,
        "weekly_actual_max": 0.038,
    }

    result = _build_result_payload(
        _study(
            _trial(
                0,
                "COMPLETE",
                0.42,
                {"lambda_magnitude": 0.55},
                {
                    "avg_quantile_crossing_rate": 0.0,
                    "avg_weekly_magnitude_ratio": 1.1333,
                    "avg_weekly_pi80_coverage": 0.80,
                    "avg_weekly_pi80_width_ratio": 1.0,
                    "avg_weekly_mae_vs_naive_zero": 0.9,
                    "avg_variance_ratio": 1.1,
                    "avg_directional_accuracy": 0.55,
                    "fold_scale_diagnostics": [scale_diagnostic],
                },
            )
        )
    )

    assert result["fold_scale_diagnostics"] == [scale_diagnostic]


def test_fold_scale_diagnostic_includes_target_audit_and_raw_bounded_scale():
    weekly_actual = np.array([0.04, -0.04, 0.05, -0.05])
    raw_weekly_pred = np.array([1.00, -1.00, 0.90, -0.90])
    bounded_weekly_pred = np.array([0.08, -0.08, 0.08, -0.08])
    train_audit = {
        "target_decoder_mean": 0.001,
        "target_decoder_std": 0.012,
        "target_decoder_min": -0.03,
        "target_decoder_max": 0.04,
        "target_scale_present": True,
        "target_scale_mean": 0.5,
        "target_scale_std": 0.5,
        "target_scale_min": 0.0,
        "target_scale_max": 1.0,
        "actual_weekly_std": 0.035,
        "actual_weekly_mean_abs": 0.03,
        "actual_weekly_abs_median": 0.025,
        "actual_weekly_min": -0.06,
        "actual_weekly_max": 0.07,
    }
    val_audit = {
        "target_decoder_mean": -0.001,
        "target_decoder_std": 0.010,
        "target_decoder_min": -0.02,
        "target_decoder_max": 0.03,
        "target_scale_present": False,
        "actual_weekly_std": 0.040,
        "actual_weekly_mean_abs": 0.045,
        "actual_weekly_abs_median": 0.045,
        "actual_weekly_min": -0.05,
        "actual_weekly_max": 0.05,
    }

    diagnostic = _fold_scale_diagnostic(
        trial_number=2,
        fold_idx=0,
        train_samples=120,
        val_samples=24,
        weekly_actual=weekly_actual,
        weekly_pred=bounded_weekly_pred,
        weekly_metrics={
            "weekly_magnitude_ratio": 1.777,
            "weekly_mae_vs_naive_zero": 0.95,
            "weekly_raw_magnitude_ratio": 22.22,
            "weekly_bounded_magnitude_ratio": 1.777,
            "weekly_median_cap": 0.08,
            "weekly_median_bound_applied_rate": 1.0,
            "weekly_raw_pred_min": -1.0,
            "weekly_raw_pred_max": 1.0,
            "weekly_bounded_pred_min": -0.08,
            "weekly_bounded_pred_max": 0.08,
            "cap_to_actual_abs_median_ratio": 3.2,
            "cap_to_actual_mean_abs_ratio": 2.667,
        },
        raw_weekly_pred=raw_weekly_pred,
        train_scale_audit=train_audit,
        val_scale_audit=val_audit,
        weekly_median_cap=0.08,
    )

    assert diagnostic["weekly_median_cap"] == 0.08
    assert diagnostic["weekly_raw_magnitude_ratio"] == 22.22
    assert diagnostic["weekly_bounded_magnitude_ratio"] == 1.777
    assert diagnostic["weekly_median_bound_applied_rate"] == 1.0
    assert diagnostic["cap_to_actual_abs_median_ratio"] == 3.2
    assert diagnostic["cap_to_actual_mean_abs_ratio"] == 2.667
    assert diagnostic["raw_pred_weekly_mean_abs"] == 0.95
    assert diagnostic["pred_weekly_mean_abs"] == 0.08
    assert diagnostic["train_target_decoder_std"] == 0.012
    assert diagnostic["val_target_decoder_std"] == 0.01
    assert diagnostic["train_target_scale_present"] is True
    assert diagnostic["val_target_scale_present"] is False


def test_startup_protection_requires_min_finite_completed_trials():
    protected_study = _study(
        _trial(0, "COMPLETE", 0.5),
        _trial(1, "COMPLETE", float("inf")),
        _trial(2, "PRUNED"),
    )
    protected_trial = SimpleNamespace(study=protected_study)

    assert _finite_completed_trial_count(protected_study) == 1
    assert _is_startup_protected(protected_trial)

    enough_completed = _study(
        *[_trial(i, "COMPLETE", 0.1 + i) for i in range(MIN_COMPLETED_TRIALS)]
    )
    unprotected_trial = SimpleNamespace(study=enough_completed)

    assert _finite_completed_trial_count(enough_completed) == MIN_COMPLETED_TRIALS
    assert not _is_startup_protected(unprotected_trial)


def test_enqueue_known_good_trial_only_for_empty_study():
    class FakeStudy:
        def __init__(self, trials=None):
            self.trials = trials or []
            self.enqueued = []

        def enqueue_trial(self, params):
            self.enqueued.append(params)

    study = FakeStudy()
    assert _enqueue_known_good_trial(study, base_cfg=None)
    assert study.enqueued == [KNOWN_GOOD_TRIAL_PARAMS]

    existing_study = FakeStudy(trials=[_trial(0, "COMPLETE", 0.5)])
    assert not _enqueue_known_good_trial(existing_study, base_cfg=None)
    assert existing_study.enqueued == []


def test_known_good_trial_includes_weekly_loss_search_params():
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_weekly_quantile"] == 0.70
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_t1_quantile"] == 0.20
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_dispersion"] == 0.35
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_magnitude"] == 0.55
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_naive"] == 0.40
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_bias"] == 0.19
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_directional"] == 0.06
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_positive_rate"] == 0.20
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_interval"] == 0.15
    assert "weekly_lambda_vol" not in KNOWN_GOOD_TRIAL_PARAMS
    assert "lambda_width" not in KNOWN_GOOD_TRIAL_PARAMS
    assert "lambda_tail_width" not in KNOWN_GOOD_TRIAL_PARAMS


def test_hyperopt_reports_weekly_objective_label():
    source = inspect.getsource(hyperopt_module)

    assert "Best weekly objective" in source
    assert "weekly_objective=%.6f" in source
    assert "Best val_loss" not in source


def test_hyperopt_objective_penalizes_interval_width_and_overcoverage():
    source = inspect.getsource(hyperopt_module)

    assert "coverage_penalty = abs(fold_weekly_pi80_coverage - 0.80)" in source
    assert "width_penalty = max(0.0, fold_weekly_pi80_width_ratio - 1.5)" in source
    assert "tail_width_penalty = max(0.0, fold_weekly_pi96_width_ratio - 3.0)" in source
    assert "interval_score_penalty" in source
    assert "weekly_interval_width_explosion" in source
    assert "weekly_tail_width_explosion" in source
    assert "weekly_raw_crossing_prune" in source
    assert "weekly_overcoverage_width_explosion" in source


def test_controlled_hyperopt_search_only_tunes_weekly_loss_weights():
    trial = _RecordingTrial()

    cfg = create_trial_config(trial, get_tft_config())

    assert cfg.model.max_encoder_length == 50
    assert cfg.model.hidden_size == 48
    assert cfg.model.attention_head_size == 2
    assert cfg.model.dropout == 0.30
    assert cfg.model.hidden_continuous_size == 16
    assert cfg.model.learning_rate == 2e-4
    assert cfg.model.gradient_clip_val == 1.0
    assert cfg.model.weight_decay == 5e-5

    assert cfg.asro.lambda_vol == 0.30
    assert cfg.asro.lambda_quantile == 0.25
    assert cfg.asro.lambda_madl == 0.40

    assert cfg.training.batch_size == 32

    assert cfg.weekly_loss.lambda_weekly_quantile == 0.70
    assert cfg.weekly_loss.lambda_t1_quantile == 0.20
    assert cfg.weekly_loss.lambda_dispersion == 0.35
    assert cfg.weekly_loss.weekly_median_cap is None
    assert cfg.weekly_loss.weekly_median_cap_abs_median_multiple == 2.0
    assert cfg.weekly_loss.weekly_median_cap_mean_abs_multiple == 1.6
    assert cfg.weekly_loss.weekly_median_cap_std_multiple == 1.2
    assert cfg.weekly_loss.lambda_saturation == 0.25
    assert cfg.weekly_loss.lambda_positive_rate == 0.20
    assert cfg.weekly_loss.lambda_interval == 0.15

    assert trial.float_ranges == {}
    assert trial.categorical_choices == {
        "lambda_magnitude": [0.50, 0.55, 0.58],
        "lambda_naive": [0.35, 0.40, 0.45],
        "lambda_bias": [0.14, 0.17, 0.19],
        "lambda_directional": [0.05, 0.06, 0.07],
    }
    assert "lambda_positive_rate" not in trial.categorical_choices
    assert "lambda_interval" not in trial.categorical_choices


def test_hyperopt_objective_penalizes_positive_rate_and_prunes_explosions():
    source = inspect.getsource(hyperopt_module)

    assert "positive_rate_penalty = abs(" in source
    assert "weekly_pred_positive_rate" in source
    assert "weekly_actual_positive_rate" in source
    assert "fold_weekly_pred_positive_rate > 0.90" in source
    assert "fold_weekly_actual_positive_rate < 0.75" in source
    assert "weekly_positive_rate_explosion" in source
    assert "fold_weekly_pi80_coverage < 0.15" in source
    assert "weekly_pi80_undercoverage" in source
    assert "fold_weekly_mae_vs_naive_zero > 3.0" in source
    assert "weekly_mae_vs_naive_explosion" in source


def test_run_hyperopt_persists_structural_failure_without_runtime_raise():
    source = inspect.getsource(hyperopt_module.run_hyperopt)

    assert "STRUCTURAL_FAILURE" in source
    assert "raise RuntimeError(structural_report" not in source
