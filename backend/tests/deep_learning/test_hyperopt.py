"""Tests for Optuna hyperopt result handling."""

import inspect
from types import SimpleNamespace

import deep_learning.training.hyperopt as hyperopt_module
from deep_learning.training.hyperopt import (
    KNOWN_GOOD_TRIAL_PARAMS,
    MIN_COMPLETED_TRIALS,
    _build_result_payload,
    _enqueue_known_good_trial,
    _finite_completed_trial_count,
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
        self.float_ranges = {}

    def suggest_categorical(self, name, choices):
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
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_bias"] == 0.17
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_directional"] == 0.06
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


def test_controlled_hyperopt_search_stays_near_midpoint_weights():
    trial = _RecordingTrial()

    create_trial_config(trial, get_tft_config())

    assert trial.float_ranges["lambda_magnitude"] == {
        "low": 0.50,
        "high": 0.58,
        "step": 0.01,
        "log": False,
    }
    assert trial.float_ranges["lambda_naive"] == {
        "low": 0.35,
        "high": 0.45,
        "step": 0.05,
        "log": False,
    }
    assert trial.float_ranges["lambda_bias"] == {
        "low": 0.14,
        "high": 0.19,
        "step": 0.01,
        "log": False,
    }
    assert trial.float_ranges["lambda_directional"] == {
        "low": 0.05,
        "high": 0.07,
        "step": 0.01,
        "log": False,
    }


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
