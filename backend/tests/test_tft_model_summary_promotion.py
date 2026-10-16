"""Regression coverage for active-model filtering on the Models API."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import main


def _fake_session_for(meta):
    class FakeQuery:
        def filter(self, *_args):
            return self

        def order_by(self, *_args):
            return self

        def first(self):
            return meta

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def query(self, _model):
            return FakeQuery()

    return FakeSession()


def _summary_meta(*, config_json="{}", importance_json=None, metrics_json="{}"):
    return SimpleNamespace(
        symbol="HG=F",
        trained_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        checkpoint_path="models/tft/best_tft_asro.ckpt",
        config_json=config_json,
        metrics_json=metrics_json,
        importance_json=importance_json,
        quality_gate_passed=True,
    )


def _feature_importance_pairs(summary):
    return [
        (item["feature"], item["importance"])
        for item in summary["variable_importance"]
    ]


def test_tft_summary_does_not_fall_back_to_rejected_candidate(monkeypatch):
    class EmptyQuery:
        def filter(self, *_args):
            return self

        def order_by(self, *_args):
            return self

        def first(self):
            return None

    class FakeSession:
        query_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def query(self, _model):
            self.query_count += 1
            return EmptyQuery()

    session = FakeSession()
    monkeypatch.setattr(main, "SessionLocal", lambda: session)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.get_tft_summary("HG=F"))

    assert exc_info.value.status_code == 404
    assert "quality-gate-passed" in str(exc_info.value.detail)
    assert session.query_count == 1


def test_tft_summary_prefers_persisted_encoder_importance(monkeypatch, tmp_path):
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "mrmr_results.json").write_text(
        json.dumps({"scores": {"HG_F_rsi14": 0.99}}), encoding="utf-8"
    )
    meta = _summary_meta(
        config_json=json.dumps({"feature_store": {"artifact_dir": str(tmp_path)}}),
        importance_json=json.dumps(
            {"schema_version": 1, "encoder": {"a": 0.4, "b": 0.6}}
        ),
    )
    monkeypatch.setattr(main, "SessionLocal", lambda: _fake_session_for(meta))

    summary = asyncio.run(main.get_tft_summary("HG=F"))

    assert _feature_importance_pairs(summary) == [("b", 0.6), ("a", 0.4)]


def test_tft_summary_falls_back_to_mrmr_when_importance_malformed(
    monkeypatch, tmp_path
):
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "mrmr_results.json").write_text(
        json.dumps({"scores": {"HG_F_lag_ret1": 0.8, "HG_F_rsi14": 0.2}}),
        encoding="utf-8",
    )
    meta = _summary_meta(
        config_json=json.dumps({"feature_store": {"artifact_dir": str(tmp_path)}}),
        importance_json="not json",
    )
    monkeypatch.setattr(main, "SessionLocal", lambda: _fake_session_for(meta))

    summary = asyncio.run(main.get_tft_summary("HG=F"))

    assert _feature_importance_pairs(summary) == [
        ("HG_F_lag_ret1", 0.8),
        ("HG_F_rsi14", 0.2),
    ]


def test_tft_summary_returns_empty_importance_without_any_source(
    monkeypatch, tmp_path
):
    meta = _summary_meta(
        config_json=json.dumps({"feature_store": {"artifact_dir": str(tmp_path)}}),
        importance_json=None,
    )
    monkeypatch.setattr(main, "SessionLocal", lambda: _fake_session_for(meta))

    summary = asyncio.run(main.get_tft_summary("HG=F"))

    assert summary["variable_importance"] == []

