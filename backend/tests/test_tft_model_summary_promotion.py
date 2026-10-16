"""Regression coverage for active-model filtering on the Models API."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.models import TFTModelMetadata


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
    assert summary["variable_importance_method"] == "vsn_encoder"


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
    assert summary["variable_importance_method"] == "mrmr"


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
    assert summary["variable_importance_method"] is None


def test_tft_summary_never_serves_a_rejected_candidate_row(
    monkeypatch, tmp_path
):
    # Real-session guard for the promotion contract: the endpoint query must
    # keep `TFTModelMetadata.quality_gate_passed.is_(True)`, so a REJECTED row
    # is never served, no matter how shiny its stored metrics look. The table
    # keeps a single row per symbol (UNIQUE constraint; `_persist_tft_model_
    # metadata` re-runs the gate and refuses to persist rejected candidates),
    # so the realistic regression is dropping the filter and reporting whichever
    # row happens to exist. The FakeSession-based tests above cannot catch that
    # (their `filter()` swallows its arguments); a real DB engine can.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    try:
        TFTModelMetadata.__table__.create(bind=engine)
        ServerSession = sessionmaker(bind=engine)
        with ServerSession() as session:
            session.add(
                TFTModelMetadata(
                    symbol="HG=F",
                    trained_at=datetime(2026, 10, 10, tzinfo=timezone.utc),
                    checkpoint_path="/artifacts/tft/rejected_candidate.ckpt",
                    config_json=json.dumps(
                        {"feature_store": {"artifact_dir": str(tmp_path)}}
                    ),
                    metrics_json=json.dumps({"directional_accuracy": 0.9}),
                    importance_json=json.dumps(
                        {"schema_version": 1, "encoder": {"rejected": 1.0}}
                    ),
                    quality_gate_passed=False,  # REJECTED
                )
            )
            session.commit()
        monkeypatch.setattr(main, "SessionLocal", ServerSession)

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(main.get_tft_summary("HG=F"))

        assert excinfo.value.status_code == 404
        assert "quality-gate-passed" in excinfo.value.detail
    finally:
        engine.dispose()

