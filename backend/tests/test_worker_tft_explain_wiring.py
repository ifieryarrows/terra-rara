"""Stage-5.5 wiring guards for the per-forecast TFT explanation.

Guards two loss-prone seams:
1. ``_persist_tft_report_snapshot`` must land ``explain_json`` (with the run
   id) on the snapshot row and fail open when the write fails.
2. The pipeline stage must keep requesting explanations — if a refactor drops
   ``include_explain=True`` from the ``generate_tft_analysis`` call, the whole
   feature silently stops producing payloads forever.
"""

import inspect
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import TFTPredictionSnapshot
from worker.tasks import _execute_pipeline_stages_v2, _persist_tft_report_snapshot


def _engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _report(explain=None):
    report = {
        "prediction": {"reference_price_date": "2026-10-15"},
        "is_forecast_healthy": True,
    }
    if explain is not None:
        report["explain"] = explain
    return report


def test_persist_writes_explain_json_with_run_id():
    engine = _engine()
    try:
        TFTPredictionSnapshot.__table__.create(bind=engine)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            explain, persisted, error = _persist_tft_report_snapshot(
                session,
                _report(explain={"schema_version": 1, "attention": {"series": []}}),
                "run-77",
            )
            session.commit()

            assert (persisted, error) == (True, None)
            assert explain["run_id"] == "run-77"
            row = session.query(TFTPredictionSnapshot).one()
            assert row.run_id == "run-77"
            assert row.explain_json["schema_version"] == 1
            assert row.explain_json["run_id"] == "run-77"
    finally:
        engine.dispose()


def test_persist_without_explain_keeps_column_null():
    engine = _engine()
    try:
        TFTPredictionSnapshot.__table__.create(bind=engine)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            explain, persisted, error = _persist_tft_report_snapshot(
                session, _report(), "run-78"
            )
            session.commit()

            assert (persisted, error) == (True, None)
            assert explain is None
            row = session.query(TFTPredictionSnapshot).one()
            assert row.explain_json is None
    finally:
        engine.dispose()


def test_persist_failure_is_fail_open(monkeypatch, tmp_path):
    engine = _engine()
    try:
        Session = sessionmaker(bind=engine)

        def _explode(*_args, **_kwargs):
            raise Exception("database offline")

        with Session() as session:
            # Patch the live session, not the sessionmaker: instances produced
            # by `sessionmaker(bind=...)` never see attributes set on the
            # factory itself.
            monkeypatch.setattr(session, "add", _explode)
            explain, persisted, error = _persist_tft_report_snapshot(
                session, _report(explain={"schema_version": 1}), "run-79"
            )

            # The payload survives so diagnostics are possible, nothing raises
            # and the caller records the failure without losing the run.
            assert persisted is False
            assert error == "database offline"
    finally:
        engine.dispose()


def test_stage_55_still_requests_explanations():
    # Canary: if the `include_explain=True` call disappears, the feature stops
    # producing explanations with every other test staying green (no worker
    # integration harness exists). This asserts the Phase 5.5 wiring explicitly.
    source = inspect.getsource(_execute_pipeline_stages_v2)
    assert "include_explain=True" in source
    assert "generate_tft_analysis(" in source


def test_run_id_is_stamped_when_worker_run_is_none():
    engine = _engine()
    try:
        TFTPredictionSnapshot.__table__.create(bind=engine)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            explain, persisted, _ = _persist_tft_report_snapshot(
                session, _report(explain={"schema_version": 1}), None
            )
            session.commit()

            assert persisted is True
            assert explain["run_id"] is None
    finally:
        engine.dispose()
