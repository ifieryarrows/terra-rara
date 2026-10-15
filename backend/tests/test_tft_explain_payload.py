"""Tests for the TFT explain payload builder (``app.tft_explain``)."""

import json
import logging
import re as regex
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.tft_explain import (
    EXPLAIN_SCHEMA_VERSION,
    RECENT_BASIS_SESSIONS,
    TOP_RECENT,
    build_explain_payload,
    try_build_explain_payload,
)

WINDOW_DATES = ["2026-10-10", "2026-10-13", "2026-10-14"]


def _happy_interpretation() -> dict:
    return {
        "encoder": [
            {"feature": "HG=F_close", "weight": 0.5},
            {"feature": "HG=F_sentiment_mean", "weight": 0.3},
            {"feature": "HG=F_emb_03", "weight": 0.2},
        ],
        "decoder": [{"feature": "dow_sin", "weight": 1.0}],
        "static": [{"feature": "symbol_id", "weight": 1.0}],
        "attention": [0.2, 0.3, 0.5],
        "encoder_length": 3,
    }


def _happy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "HG=F_close": [6.0, 6.1, 6.061],
            "HG=F_sentiment_mean": [0.1, 0.2, 0.15],
            "HG=F_emb_03": [0.0, 1.0, 2.0],
        },
        index=WINDOW_DATES,
    )


def _base_kwargs(**overrides) -> dict:
    kwargs = {
        "interpretation": _happy_interpretation(),
        "window_dates": list(WINDOW_DATES),
        "feature_frame": _happy_frame(),
        "reference_price_date": WINDOW_DATES[-1],
        "symbol": "HG=F",
        "checkpoint": "best.ckpt",
        "generated_at": datetime(2026, 10, 15, tzinfo=timezone.utc),
    }
    kwargs.update(overrides)
    return kwargs


def _build(**overrides) -> dict:
    return build_explain_payload(**_base_kwargs(**overrides))


def _json_dump(payload: dict) -> str:
    return json.dumps(payload, allow_nan=False)


def test_constants_match_interface():
    assert EXPLAIN_SCHEMA_VERSION == 1
    assert TOP_RECENT == 5
    assert RECENT_BASIS_SESSIONS == 5


def test_happy_path_groups_labels_attention_and_notes():
    payload = _build()

    assert payload["schema_version"] == 1
    assert payload["symbol"] == "HG=F"
    assert payload["reference_price_date"] == "2026-10-14"
    assert payload["generated_at"] == "2026-10-15T00:00:00+00:00"
    assert payload["run_id"] is None
    assert payload["model"] == {"checkpoint": "best.ckpt", "trained_at": None}
    assert payload["horizon"] == {"label": "5-day cumulative log return (primary)", "days": 5}
    assert "not causal" in payload["notes"]

    selection = payload["variable_selection"]
    assert selection["method"]
    assert [entry["feature"] for entry in selection["encoder"]] == [
        "HG=F_close",
        "HG=F_sentiment_mean",
        "HG=F_emb_03",
    ]
    assert [entry["weight"] for entry in selection["encoder"]] == [0.5, 0.3, 0.2]
    assert [entry["feature"] for entry in selection["decoder"]] == ["dow_sin"]
    assert [entry["feature"] for entry in selection["static"]] == ["symbol_id"]

    # Groups stay separate and every entry carries describe_feature metadata.
    all_features = [
        entry["feature"]
        for group in ("encoder", "decoder", "static")
        for entry in selection[group]
    ]
    assert sorted(all_features) == [
        "HG=F_close",
        "HG=F_emb_03",
        "HG=F_sentiment_mean",
        "dow_sin",
        "symbol_id",
    ]
    for entry in selection["encoder"] + selection["decoder"] + selection["static"]:
        assert set(entry) >= {"feature", "weight", "label", "description", "category", "time_horizon"}

    close_entry = selection["encoder"][0]
    assert close_entry["recent"]["kind"] == "pct"
    assert close_entry["recent"]["basis"] == "5 sessions"
    assert close_entry["recent"]["last_value"] == 6.061
    assert close_entry["recent"]["change"] == pytest.approx(0.061)
    assert close_entry["recent"]["change_pct"] == pytest.approx(1.0166667, abs=1e-6)
    assert close_entry["recent"]["as_of"] == "2026-10-14"

    sentiment_entry = selection["encoder"][1]
    assert sentiment_entry["recent"]["kind"] == "abs"
    assert sentiment_entry["recent"]["change"] == pytest.approx(0.05)
    assert sentiment_entry["recent"]["change_pct"] is None
    assert sentiment_entry["recent"]["as_of"] == "2026-10-14"

    emb_entry = selection["encoder"][2]
    assert "recent" not in emb_entry
    assert emb_entry["category"] == "Embedding"

    attention = payload["attention"]
    assert attention["method"]
    assert attention["step"] == {"kind": "first_forecast_day", "horizon_index": 0}
    assert attention["window"] == {"start": "2026-10-10", "end": "2026-10-14", "days": 3}
    assert [point["date"] for point in attention["series"]] == list(WINDOW_DATES)
    assert attention["series"][-1] == {"date": "2026-10-14", "weight": 0.5}
    assert attention["top_days"][0] == {"date": "2026-10-14", "weight": 0.5}
    assert [point["weight"] for point in attention["top_days"]] == [0.5, 0.3, 0.2]

    for key in ("encoder", "decoder", "static"):
        assert regex.fullmatch(r"[0-9a-f]{12}", payload["layout_hash"][key])

    # Fully JSON-safe: NaN/Inf would make allow_nan=False raise, and the
    # round-trip must be loss-less for the ORM JSON column.
    assert json.loads(_json_dump(payload)) == payload


def test_weights_are_rounded_to_six_decimals():
    interpretation = _happy_interpretation()
    interpretation["encoder"] = [{"feature": "HG=F_close", "weight": 1 / 3}]

    payload = _build(interpretation=interpretation)

    assert payload["variable_selection"]["encoder"][0]["weight"] == 0.333333


def test_length_mismatches_raise_value_error_and_try_is_fail_open():
    with pytest.raises(ValueError):
        _build(window_dates=WINDOW_DATES[:2])

    with pytest.raises(ValueError):
        _build(feature_frame=_happy_frame().iloc[:2])

    assert try_build_explain_payload(**_base_kwargs(window_dates=WINDOW_DATES[:2])) is None
    assert try_build_explain_payload(**_base_kwargs(feature_frame=_happy_frame().iloc[:2])) is None


def test_non_finite_weights_are_dropped_from_their_group():
    frame = _happy_frame()
    frame["HG=F_open"] = [5.9, 6.0, 6.05]
    interpretation = _happy_interpretation()
    interpretation["encoder"] = [
        {"feature": "HG=F_close", "weight": float("nan")},
        {"feature": "HG=F_sentiment_mean", "weight": np.inf},
        {"feature": "HG=F_emb_03", "weight": np.nan},
        {"feature": "HG=F_open", "weight": np.float64(0.1)},
    ]
    interpretation["decoder"] = [{"feature": "dow_sin", "weight": float("-inf")}]

    payload = _build(interpretation=interpretation, feature_frame=frame)

    assert [entry["feature"] for entry in payload["variable_selection"]["encoder"]] == ["HG=F_open"]
    assert payload["variable_selection"]["encoder"][0]["weight"] == 0.1
    assert payload["variable_selection"]["decoder"] == []
    assert [entry["feature"] for entry in payload["variable_selection"]["static"]] == ["symbol_id"]
    _json_dump(payload)


def test_encoder_entry_without_frame_column_keeps_label_but_no_recent():
    interpretation = _happy_interpretation()
    interpretation["encoder"] = [
        {"feature": "HG=F_close", "weight": 0.9},
        {"feature": "HG=F_unmapped_driver", "weight": 0.1},
    ]

    payload = _build(interpretation=interpretation)

    entries = payload["variable_selection"]["encoder"]
    assert [entry["feature"] for entry in entries] == ["HG=F_close", "HG=F_unmapped_driver"]
    assert "recent" not in entries[1]
    assert entries[1]["label"]  # describe_feature fallback still labels it
    assert entries[1]["category"]
    assert entries[0]["recent"]["basis"] == "5 sessions"


def test_recent_is_attached_only_to_top_five_encoder_entries():
    names = [
        "HG=F_close",
        "HG=F_open",
        "HG=F_high",
        "HG=F_low",
        "HG=F_SMA_5",
        "HG=F_EMA_5",
        "HG=F_volume",
    ]
    frame = pd.DataFrame({name: [1.0, 2.0, 3.0] for name in names}, index=WINDOW_DATES)
    interpretation = _happy_interpretation()
    interpretation["encoder"] = [
        {"feature": name, "weight": round(0.7 - index * 0.1, 6)}
        for index, name in enumerate(names)
    ]

    payload = _build(interpretation=interpretation, feature_frame=frame)

    entries = payload["variable_selection"]["encoder"]
    assert [entry["feature"] for entry in entries] == names
    assert ["recent" in entry for entry in entries] == [True] * TOP_RECENT + [False] * (
        len(names) - TOP_RECENT
    )
    assert all(entry["recent"]["kind"] == "pct" for entry in entries[:TOP_RECENT])


def test_recent_uses_the_last_five_sessions_only():
    dates = [f"2026-09-{day:02d}" for day in range(1, 8)]  # 7 sessions
    frame = pd.DataFrame({"HG=F_close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]}, index=dates)
    interpretation = _happy_interpretation()
    interpretation["encoder"] = [{"feature": "HG=F_close", "weight": 1.0}]
    interpretation["decoder"] = []
    interpretation["static"] = []
    interpretation["attention"] = [0.1] * 7

    payload = _build(interpretation=interpretation, window_dates=dates, feature_frame=frame)

    recent = payload["variable_selection"]["encoder"][0]["recent"]
    assert recent["last_value"] == 16.0
    assert recent["change"] == 4.0  # 12 -> 16 across the last five sessions
    assert recent["change_pct"] == pytest.approx(4.0 / 12.0 * 100.0)
    assert recent["change_pct"] == 33.333333
    assert recent["as_of"] == "2026-09-07"
    assert recent["basis"] == f"{RECENT_BASIS_SESSIONS} sessions"


def test_recent_omitted_when_column_has_fewer_than_two_finite_values():
    frame = _happy_frame()
    frame["HG=F_close"] = [float("nan"), float("inf"), 6.061]

    payload = _build(feature_frame=frame)

    assert "recent" not in payload["variable_selection"]["encoder"][0]
    _json_dump(payload)


def test_non_finite_attention_is_zeroed_and_zero_series_still_built():
    interpretation = _happy_interpretation()
    interpretation["attention"] = [0.2, float("nan"), float("inf")]

    payload = _build(interpretation=interpretation)

    series = payload["attention"]["series"]
    assert series[0]["weight"] == 0.2
    assert series[1]["weight"] == 0.0
    assert series[2]["weight"] == 0.0
    assert payload["attention"]["top_days"][0] == {"date": "2026-10-10", "weight": 0.2}
    _json_dump(payload)

    zero_interpretation = _happy_interpretation()
    zero_interpretation["attention"] = [0.0, 0.0, 0.0]
    zero_payload = _build(interpretation=zero_interpretation)
    assert zero_payload["attention"]["series"] == [
        {"date": "2026-10-10", "weight": 0.0},
        {"date": "2026-10-13", "weight": 0.0},
        {"date": "2026-10-14", "weight": 0.0},
    ]
    assert zero_payload["attention"]["top_days"][0]["weight"] == 0.0


def test_try_build_returns_payload_on_success():
    payload = try_build_explain_payload(**_base_kwargs())

    assert isinstance(payload, dict)
    assert payload["schema_version"] == EXPLAIN_SCHEMA_VERSION


def test_try_build_is_fail_open_and_logs_a_warning(caplog):
    broken = _happy_interpretation()
    broken.pop("attention")

    with caplog.at_level(logging.WARNING, logger="app.tft_explain"):
        result = try_build_explain_payload(**_base_kwargs(interpretation=broken))

    assert result is None
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    assert any("explain" in record.getMessage().lower() for record in caplog.records)


def test_optional_metadata_is_json_normalized():
    trained_at = datetime(2026, 10, 1, 12, 30, tzinfo=timezone.utc)

    payload = _build(run_id="run-123", trained_at=trained_at)

    assert payload["run_id"] == "run-123"
    assert payload["model"]["trained_at"] == "2026-10-01T12:30:00+00:00"
    assert payload["generated_at"] == "2026-10-15T00:00:00+00:00"

    string_payload = _build(trained_at="2026-10-01T00:00:00")
    assert string_payload["model"]["trained_at"] == "2026-10-01T00:00:00"

    default_payload = _build(generated_at=None)
    parsed = datetime.fromisoformat(default_payload["generated_at"])
    assert parsed.tzinfo is not None
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds()) < 60


def test_layout_hash_is_deterministic_and_layout_sensitive():
    first = _build()
    second = _build()
    assert first["layout_hash"] == second["layout_hash"]

    changed = _happy_interpretation()
    changed["encoder"] = [
        {"feature": "HG=F_close", "weight": 0.5},
        {"feature": "HG=F_sentiment_mean", "weight": 0.3},
        {"feature": "HG=F_emb_07", "weight": 0.2},
    ]
    third = _build(interpretation=changed)
    assert third["layout_hash"]["encoder"] != first["layout_hash"]["encoder"]
    assert third["layout_hash"]["decoder"] == first["layout_hash"]["decoder"]
