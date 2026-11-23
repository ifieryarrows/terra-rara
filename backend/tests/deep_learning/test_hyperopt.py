"""Tests for Optuna hyperopt result handling."""

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd

import deep_learning.training.hyperopt as hyperopt_module
from deep_learning.training.hyperopt import (
    KNOWN_GOOD_TRIAL_PARAMS,
    MIN_COMPLETED_TRIALS,
    _build_result_payload,
    _data_snapshot_id,
    _enqueue_known_good_trial,
    _finite_completed_trial_count,
    _fold_scale_diagnostic,
    _is_startup_protected,
    create_trial_config,
)
from deep_learning.config import get_tft_config


def test_data_snapshot_id_is_stable_and_changes_with_frame_content():
    frame = pd.DataFrame({"target": [0.1, -0.2]})
    same_frame = frame.copy()
    changed_frame = frame.copy()
    changed_frame.loc[1, "target"] = -0.3

    assert _data_snapshot_id(frame) == _data_snapshot_id(same_frame)
    assert _data_snapshot_id(frame) != _data_snapshot_id(changed_frame)


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


def test_build_result_payload_prefers_preflight_safe_trial_over_lower_objective():
    healthy = {
        "avg_quantile_crossing_rate": 0.0,
        "avg_weekly_magnitude_ratio": 1.0,
        "avg_weekly_pred_positive_rate": 0.55,
        "avg_weekly_actual_positive_rate": 0.57,
        "avg_weekly_pi80_coverage": 0.80,
        "avg_weekly_pi80_width_ratio": 1.0,
        "avg_weekly_mae_vs_naive_zero": 1.0,
        "avg_variance_ratio": 1.0,
    }
    result = _build_result_payload(
        _study(
            _trial(
                0,
                "COMPLETE",
                0.10,
                {"lambda_directional": 0.15},
                {**healthy, "avg_weekly_directional_accuracy": 0.44},
            ),
            _trial(
                1,
                "COMPLETE",
                0.50,
                {"lambda_directional": 0.25},
                {**healthy, "avg_weekly_directional_accuracy": 0.52},
            ),
        )
    )

    assert result["best_trial"] == 1
    assert result["best_params"] == {"lambda_directional": 0.25}
    assert result["best_trial_preflight"]["preflight_passed"] is True
    assert result["best_trial_selection"]["mode"] == "lowest_objective_preflight_pass"
    assert result["best_trial_selection"]["preflight_eligible_trials"] == [1]


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


def test_shared_reproducibility_contract_sets_deterministic_cpu():
    import torch

    from deep_learning.training.reproducibility import configure_tft_reproducibility

    configure_tft_reproducibility()

    assert torch.get_num_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()


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
    assert KNOWN_GOOD_TRIAL_PARAMS["max_encoder_length"] == 15
    assert KNOWN_GOOD_TRIAL_PARAMS["hidden_size"] == 12
    assert KNOWN_GOOD_TRIAL_PARAMS["hidden_continuous_size"] == 6
    assert KNOWN_GOOD_TRIAL_PARAMS["attention_head_size"] == 1
    assert KNOWN_GOOD_TRIAL_PARAMS["output_head_lr_mult"] == 0.20
    assert KNOWN_GOOD_TRIAL_PARAMS["weight_decay"] == 2e-2
    assert KNOWN_GOOD_TRIAL_PARAMS["mrmr_top_k"] == 5
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_t1_quantile"] == 0.20
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_drift_prior"] == 1.00
    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_directional"] == 0.10
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

    assert "pinball_norm" in source
    assert "interval_norm" in source
    assert "mae_excess_penalty" in source
    assert "sharpe_bonus" in source
    assert "weekly_interval_width_explosion" in source
    assert "weekly_tail_width_explosion" in source
    assert "fold_weekly_pi80_width_ratio > 4.0" in source
    assert "fold_weekly_pi96_width_ratio > 4.0" in source
    assert "crossing_prune" in source
    assert "weekly_raw_crossing_prune" not in source


def test_controlled_hyperopt_search_only_tunes_weekly_loss_weights():
    trial = _RecordingTrial()

    cfg = create_trial_config(trial, get_tft_config())

    assert cfg.model.max_encoder_length in {10, 15, 20}
    assert cfg.model.hidden_size in {8, 12}
    assert cfg.model.attention_head_size in {1, 2}
    assert cfg.model.hidden_continuous_size in {4, 6}
    assert cfg.model.output_head_lr_mult in {0.15, 0.20, 0.25}
    assert cfg.model.weight_decay in {1e-2, 2e-2, 3e-2}
    assert cfg.feature_store.mrmr_top_k in {5, 6, 8, 12}
    assert cfg.training.batch_size == 32
    assert cfg.weekly_loss.lambda_weekly_quantile == 1.00
    assert cfg.weekly_loss.lambda_t1_quantile in {0.15, 0.20, 0.25}
    assert cfg.weekly_loss.lambda_subhorizon_quantile == 0.15
    assert cfg.weekly_loss.lambda_drift_prior in {0.15, 0.35, 0.75, 1.00}
    assert cfg.weekly_loss.lambda_directional in {0.05, 0.10, 0.15}
    assert cfg.weekly_loss.lambda_dispersion == 0.0
    assert cfg.weekly_loss.lambda_magnitude == 0.0
    assert cfg.weekly_loss.lambda_naive == 0.0
    assert cfg.weekly_loss.lambda_positive_rate == 0.0
    assert cfg.weekly_loss.lambda_interval == 0.0

    assert "learning_rate" in trial.float_ranges
    assert "output_head_lr_mult" in trial.categorical_choices
    assert "dropout" in trial.categorical_choices
    assert "max_encoder_length" in trial.categorical_choices
    assert "hidden_size" in trial.categorical_choices


def test_hyperopt_objective_penalizes_positive_rate_and_prunes_explosions():
    source = inspect.getsource(hyperopt_module)

    assert "weekly_pred_positive_rate" in source
    assert "weekly_actual_positive_rate" in source
    assert "fold_weekly_pred_positive_rate > 0.94" in source
    assert "weekly_positive_rate_explosion" in source
    assert "fold_weekly_pi80_coverage < 0.15" in source
    assert "weekly_pi80_undercoverage" in source
    assert "fold_weekly_mae_vs_naive_zero > 2.5" in source
    assert "weekly_mae_vs_naive_explosion" in source


def test_run_hyperopt_persists_structural_failure_without_runtime_raise():
    source = inspect.getsource(hyperopt_module.run_hyperopt)

    assert "STRUCTURAL_FAILURE" in source
    assert "raise RuntimeError(structural_report" not in source


def test_select_preflight_safe_trial_prefers_quality_gate_magnitude_compliance():
    base_healthy = {
        "avg_quantile_crossing_rate": 0.0,
        "avg_weekly_pred_positive_rate": 0.50,
        "avg_weekly_actual_positive_rate": 0.50,
        "avg_weekly_pi80_coverage": 0.80,
        "avg_weekly_pi80_width_ratio": 1.0,
        "avg_weekly_mae_vs_naive_zero": 0.95,
        "avg_variance_ratio": 1.0,
        "avg_directional_accuracy": 0.54,
    }
    exploded_raw = _trial(
        0,
        "COMPLETE",
        0.30,
        {"lambda_magnitude": 0.50},
        {
            **base_healthy,
            "avg_weekly_magnitude_ratio": 1.65,
            "avg_weekly_raw_magnitude_ratio": 4.20,
        },
    )
    gate_compliant = _trial(
        1,
        "COMPLETE",
        0.45,
        {"lambda_magnitude": 0.58},
        {
            **base_healthy,
            "avg_weekly_magnitude_ratio": 1.05,
            "avg_weekly_raw_magnitude_ratio": 1.25,
        },
    )

    result = _build_result_payload(_study(exploded_raw, gate_compliant))

    assert result["best_trial"] == 1
    assert result["best_params"] == {"lambda_magnitude": 0.58}


def test_hyperopt_avoids_final_fold_pruning_and_t1_da_leakage():
    source = inspect.getsource(hyperopt_module._objective)
    run_source = inspect.getsource(hyperopt_module.run_hyperopt)
    assert "effective_da = avg_weekly_da" in source
    assert "max(avg_da, avg_weekly_da)" not in source
    assert "fold_vr_penalty" not in source
    assert "fold_idx < (n_folds - 1)" in source
    assert "mrmr_top_k" in source
    assert "purge_gap" in source
    assert "apply_mrmr=False" in run_source

    base_cfg = get_tft_config()
    trial_cfg = create_trial_config(_RecordingTrial(), base_cfg)
    assert trial_cfg.training.max_epochs == base_cfg.training.max_epochs
    assert trial_cfg.training.early_stopping_patience == base_cfg.training.early_stopping_patience
    assert trial_cfg.model.reduce_on_plateau_patience == base_cfg.model.reduce_on_plateau_patience


def test_run_hyperopt_writes_post_study_snapshot_with_winning_mrmr_top_k(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from dataclasses import replace
    import pandas as pd
    import deep_learning.data.feature_store as feature_store_module
    import app.db as app_db

    snap_file = tmp_path / "feature_snapshot.pkl"
    monkeypatch.setenv("TFT_FEATURE_SNAPSHOT_PATH", str(snap_file))
    monkeypatch.setattr(app_db, "init_db", lambda *a, **kw: None)
    monkeypatch.setattr(app_db, "SessionLocal", lambda: nullcontext(object()))

    calls = []
    idx = pd.bdate_range("2024-01-02", periods=30)
    dummy_df = pd.DataFrame(
        {
            "time_idx": list(range(30)),
            "group_id": ["copper"] * 30,
            "target": [0.001] * 30,
            "target_5d_log_return": [0.005] * 30,
        },
        index=idx,
    )

    def fake_build_tft_dataframe(session, cfg=None, *, drop_missing_target=True, apply_mrmr=True, use_snapshot=True):
        calls.append({
            "mrmr_top_k": int(cfg.feature_store.mrmr_top_k),
            "apply_mrmr": bool(apply_mrmr),
            "use_snapshot": bool(use_snapshot),
        })
        if apply_mrmr and use_snapshot:
            snap_file.write_bytes(b"snapshot")
        return dummy_df, ["f1"], ["day_of_week"], ["target"], 4.5

    monkeypatch.setattr(feature_store_module, "build_tft_dataframe", fake_build_tft_dataframe)

    base_cfg = replace(
        get_tft_config(),
        training=replace(
            get_tft_config().training,
            checkpoint_dir=str(tmp_path / "ckpts"),
            best_model_path=str(tmp_path / "best_tft_asro.ckpt"),
        ),
    )

    def fake_objective(trial, cfg, master_data):
        t_cfg = create_trial_config(trial, cfg)
        chosen_k = int(t_cfg.feature_store.mrmr_top_k)
        trial.set_user_attr("avg_quantile_crossing_rate", 0.0)
        trial.set_user_attr("avg_weekly_pred_positive_rate", 0.50)
        trial.set_user_attr("avg_weekly_actual_positive_rate", 0.50)
        trial.set_user_attr("avg_weekly_pi80_coverage", 0.80)
        trial.set_user_attr("avg_weekly_pi80_width_ratio", 1.0)
        trial.set_user_attr("avg_weekly_mae_vs_naive_zero", 0.95)
        trial.set_user_attr("avg_variance_ratio", 0.80)
        trial.set_user_attr("avg_weekly_magnitude_ratio", 0.85)
        trial.set_user_attr("avg_weekly_raw_magnitude_ratio", 0.90)
        trial.set_user_attr("avg_directional_accuracy", 0.55)
        # Prefer mrmr_top_k=12 in this synthetic objective to prove post-study snapshot follows best_params
        return 0.10 if chosen_k == 12 else 0.90

    monkeypatch.setattr(hyperopt_module, "_objective", fake_objective)

    import sys
    from types import ModuleType

    class _FakeTrialObj(_RecordingTrial):
        def __init__(self, number, preset_mrmr_k):
            super().__init__()
            self.number = number
            self._preset_mrmr_k = preset_mrmr_k
            self.user_attrs = {}
            self.params = {}

        def suggest_categorical(self, name, choices):
            val = self._preset_mrmr_k if name == "mrmr_top_k" else choices[0]
            self.params[name] = val
            return val

        def suggest_float(self, name, low, high, log=False):
            val = float(low)
            self.params[name] = val
            return val

        def set_user_attr(self, key, val):
            self.user_attrs[key] = val

    class _FakeStudyObj:
        def __init__(self):
            self.trials = []
            self.enqueued = []

        def enqueue_trial(self, params):
            self.enqueued.append(params)

        def optimize(self, func, n_trials=2, show_progress_bar=False):
            for i, k_val in enumerate((5, 12)[:n_trials]):
                t = _FakeTrialObj(i, k_val)
                val = func(t)
                self.trials.append(
                    _trial(i, "COMPLETE", val, dict(t.params), dict(t.user_attrs))
                )

    fake_optuna = ModuleType("optuna")
    fake_optuna.samplers = SimpleNamespace(TPESampler=lambda **kw: None)
    fake_optuna.pruners = SimpleNamespace(MedianPruner=lambda **kw: None)
    fake_optuna.create_study = lambda **kw: _FakeStudyObj()
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    res = hyperopt_module.run_hyperopt(n_trials=2, base_cfg=base_cfg)
    assert res["best_params"]["mrmr_top_k"] == 12
    assert len(calls) == 2
    assert calls[0] == {"mrmr_top_k": 5, "apply_mrmr": False, "use_snapshot": False}
    assert calls[1] == {"mrmr_top_k": 12, "apply_mrmr": True, "use_snapshot": True}
    assert snap_file.exists()

