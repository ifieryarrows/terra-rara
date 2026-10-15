"""SQLite round-trip tests for the per-forecast ``explain_json`` TFT column."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.db as db_module
from app.db import Base, _run_migrations
from app.models import TFTPredictionSnapshot

EXPLAIN_PAYLOAD = {
    "schema_version": 1,
    "symbol": "HG=F",
    "reference_price_date": "2026-10-15",
    "generated_at": "2026-10-16T08:00:00+00:00",
    "run_id": "run-123",
    "model": {
        "checkpoint": "artifacts/tft/best.ckpt",
        "trained_at": "2026-10-01T00:00:00+00:00",
    },
    "horizon": {"label": "5-day cumulative log return (primary)", "days": 5},
    "variable_selection": {
        "method": "vsn",
        "encoder": [
            {"feature": "HG=F_close", "weight": 0.5, "label": "Copper close"},
            {"feature": "HG=F_sentiment_mean", "weight": 0.25, "label": "News sentiment"},
        ],
        "decoder": [],
        "static": [],
    },
    "attention": {
        "method": "tft_attention",
        "window": {"start": "2026-10-08", "end": "2026-10-15", "days": 2},
        "series": [
            {"date": "2026-10-08", "weight": 0.4},
            {"date": "2026-10-15", "weight": 0.6},
        ],
    },
}


def _engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _snapshot(**overrides):
    values = {
        "symbol": "HG=F",
        "payload_json": {"model_state": "ok"},
        "generated_at": datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc),
        "reference_price_date": "2026-10-15",
    }
    values.update(overrides)
    return TFTPredictionSnapshot(**values)


def _columns(engine, table_name: str) -> set[str]:
    with engine.connect() as conn:
        return {row[1] for row in conn.execute(text(f"PRAGMA table_info({table_name})"))}


def test_explain_json_round_trips_through_sqlite():
    engine = _engine()
    try:
        TFTPredictionSnapshot.__table__.create(bind=engine)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            session.add(_snapshot(run_id="run-123", explain_json=EXPLAIN_PAYLOAD))
            session.commit()
        with Session() as session:
            row = session.query(TFTPredictionSnapshot).one()
            assert row.explain_json == EXPLAIN_PAYLOAD
    finally:
        engine.dispose()


def test_explain_json_is_null_when_not_provided():
    engine = _engine()
    try:
        TFTPredictionSnapshot.__table__.create(bind=engine)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            session.add(_snapshot())
            session.commit()
        with Session() as session:
            row = session.query(TFTPredictionSnapshot).one()
            assert row.explain_json is None
    finally:
        engine.dispose()


def test_run_migrations_backfills_explain_json_on_legacy_table():
    engine = _engine()
    try:
        Base.metadata.create_all(bind=engine)
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE tft_prediction_snapshots DROP COLUMN explain_json")
            )
        assert "explain_json" not in _columns(engine, "tft_prediction_snapshots")

        _run_migrations(engine)
        assert "explain_json" in _columns(engine, "tft_prediction_snapshots")

        # Idempotent: a second migration pass must be a no-op, not an error.
        _run_migrations(engine)
        assert "explain_json" in _columns(engine, "tft_prediction_snapshots")

        Session = sessionmaker(bind=engine)
        with Session() as session:
            session.add(_snapshot(explain_json=EXPLAIN_PAYLOAD))
            session.commit()
        with Session() as session:
            row = session.query(TFTPredictionSnapshot).one()
            assert row.explain_json == EXPLAIN_PAYLOAD
    finally:
        engine.dispose()


def test_run_migrations_tolerates_missing_snapshot_table():
    engine = _engine()
    try:
        Base.metadata.create_all(bind=engine)
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE tft_prediction_snapshots"))
            conn.execute(text("ALTER TABLE ai_commentaries DROP COLUMN fallback_reason"))

        _run_migrations(engine)

        assert not inspect(engine).has_table("tft_prediction_snapshots")
        # Later migration blocks still ran after the tolerated skip.
        assert "fallback_reason" in _columns(engine, "ai_commentaries")
    finally:
        engine.dispose()


def test_run_migrations_reraises_non_missing_table_failures(monkeypatch):
    engine = _engine()
    try:
        Base.metadata.create_all(bind=engine)
        original_ensure_columns = db_module._ensure_columns

        def _fail_for_snapshot(conn, table_name, columns, is_sqlite):
            if table_name == "tft_prediction_snapshots":
                raise RuntimeError("permission denied for relation tft_prediction_snapshots")
            return original_ensure_columns(conn, table_name, columns, is_sqlite)

        monkeypatch.setattr(db_module, "_ensure_columns", _fail_for_snapshot)

        with pytest.raises(RuntimeError, match="permission denied"):
            _run_migrations(engine)
    finally:
        engine.dispose()
