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


def test_persist_failure_is_fail_open(monkeypatch):
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


def test_persist_retries_with_explain_none_when_explain_write_fails(caplog):
    """An explain_json serialization/write failure must downgrade to explain_json=None
    and still persist the forecast snapshot (fail-open contract).
    """
    import logging

    engine = _engine()
    try:
        TFTPredictionSnapshot.__table__.create(bind=engine)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            # A set inside explain_json causes SQLAlchemy's JSON column serializer
            # to fail during the first session.flush(), exercising rollback + retry.
            bad_explain = {"schema_version": 1, "unserializable": {1, 2, 3}}
            with caplog.at_level(logging.WARNING, logger="worker.tasks"):
                explain, persisted, error = _persist_tft_report_snapshot(
                    session,
                    _report(explain=bad_explain),
                    "run-80",
                )
            session.commit()

            assert (explain, persisted, error) == (None, True, None)
            row = session.query(TFTPredictionSnapshot).one()
            assert row.run_id == "run-80"
            assert row.reference_price_date == "2026-10-15"
            assert row.explain_json is None
            assert "explain" not in row.payload_json
            assert row.payload_json["is_forecast_healthy"] is True
            assert any(
                "retrying with explain_json=None" in record.getMessage()
                for record in caplog.records
            )
    finally:
        engine.dispose()


def test_persist_drops_non_dict_explain_and_persists_snapshot():
    engine = _engine()
    try:
        TFTPredictionSnapshot.__table__.create(bind=engine)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            explain, persisted, error = _persist_tft_report_snapshot(
                session,
                _report(explain="unexpected-string-payload"),
                "run-81",
            )
            session.commit()

            assert (explain, persisted, error) == (None, True, None)
            row = session.query(TFTPredictionSnapshot).one()
            assert row.explain_json is None
            assert "explain" not in row.payload_json
    finally:
        engine.dispose()


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
