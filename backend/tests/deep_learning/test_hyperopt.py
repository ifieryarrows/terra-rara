"""Tests for Optuna hyperopt result handling."""

from types import SimpleNamespace

from deep_learning.training.hyperopt import (
    KNOWN_GOOD_TRIAL_PARAMS,
    MIN_COMPLETED_TRIALS,
    _build_result_payload,
    _enqueue_known_good_trial,
    _finite_completed_trial_count,
    _is_startup_protected,
)


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
