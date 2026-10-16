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
