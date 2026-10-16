"""Tests for the VSN variable-importance metadata payload written at promotion."""

import json

from deep_learning.training.trainer import _importance_metadata_payload


def test_importance_metadata_payload_round_trips_encoder_weights():
    result = {
        "trained_at": "2026-10-15T12:00:00+00:00",
        "checkpoint_path": "/tmp/models/tft/best_tft_asro.ckpt",
        "variable_importance": {"HG_F_lag_ret1": 0.7, "HG_F_rsi14": 0.3},
    }

    payload = json.loads(_importance_metadata_payload(result))

    assert payload["schema_version"] == 1
    assert payload["encoder"] == {"HG_F_lag_ret1": 0.7, "HG_F_rsi14": 0.3}
    assert payload["trained_at"] == "2026-10-15T12:00:00+00:00"
    assert payload["checkpoint"] == "/tmp/models/tft/best_tft_asro.ckpt"


def test_importance_metadata_payload_tolerates_missing_importance():
    payload = json.loads(_importance_metadata_payload({"trained_at": None}))

    assert payload == {
        "schema_version": 1,
        "encoder": {},
        "trained_at": None,
        "checkpoint": None,
    }


def test_importance_metadata_payload_drops_non_finite_weights():
    # json.dumps would otherwise emit NaN/Infinity literals (invalid JSON) that
    # poison the metadata TEXT column and break strict readers downstream.
    result = {
        "variable_importance": {"ok": 0.7, "nan_entry": float("nan"), "inf_entry": None},
        "trained_at": None,
        "checkpoint_path": None,
    }

    payload_text = _importance_metadata_payload(result)

    assert "NaN" not in payload_text and "Infinity" not in payload_text
    payload = json.loads(payload_text)
    assert payload["encoder"] == {"ok": 0.7}
    # Round-trips under the strict reader the summary endpoint relies on.
    assert json.loads(payload_text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c))) == payload
