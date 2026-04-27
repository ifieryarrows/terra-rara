"""Tests for Optuna hyperopt result handling."""

from types import SimpleNamespace

from deep_learning.training.hyperopt import _build_result_payload


def _trial(number: int, state: str, value=None, params=None):
    return SimpleNamespace(
        number=number,
        state=SimpleNamespace(name=state),
        value=value,
        params=params or {},
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
