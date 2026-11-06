"""Unit tests for PostgreSQL advisory locks non-intrusive checking and acquisition."""

from unittest.mock import MagicMock
from adapters.db.lock import (
    _lock_key_to_id,
    try_acquire_lock,
    release_lock,
    is_lock_held,
    advisory_lock,
    PIPELINE_LOCK_KEY,
)


def test_lock_key_to_id_deterministic():
    id1 = _lock_key_to_id(PIPELINE_LOCK_KEY)
    id2 = _lock_key_to_id(PIPELINE_LOCK_KEY)
    assert id1 == id2
    assert isinstance(id1, int)
    assert id1 > 0
    # Must fit in signed 64-bit bigint
    assert id1 < (1 << 63)


def test_is_lock_held_queries_pg_locks_non_intrusively():
    session = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.scalar.return_value = 1
    session.execute.return_value = mock_cursor

    held = is_lock_held(session, PIPELINE_LOCK_KEY)

    assert held is True
    # Verify pg_locks was queried and pg_try_advisory_lock was NOT executed
    query_text = str(session.execute.call_args[0][0])
    params = session.execute.call_args[0][1]
    assert "pg_locks" in query_text
    assert "pg_try_advisory_lock" not in query_text
    assert "CAST(classid AS bigint)" in query_text
    assert "CAST(objid AS bigint)" in query_text
    assert "CAST(:classid AS bigint)" in query_text
    assert "CAST(:objid AS bigint)" in query_text
    assert "classid" in params
    assert "objid" in params
    # Ensure parameter values match expected 32-bit halves
    lock_id = _lock_key_to_id(PIPELINE_LOCK_KEY)
    assert params["classid"] == ((lock_id >> 32) & 0xFFFFFFFF)
    assert params["objid"] == (lock_id & 0xFFFFFFFF)


def test_is_lock_held_sqlalchemy_compilation_and_param_binding():
    """Verify that SQLAlchemy compiles the query and extracts bind parameters without syntax errors."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE pg_locks (locktype TEXT, objsubid INT, classid INT, objid INT, granted INT)"
            )
        )
        conn.commit()

    Session = sessionmaker(bind=engine)
    with Session() as s:
        # Should execute cleanly without bind errors (returning False because table is empty)
        held = is_lock_held(s, PIPELINE_LOCK_KEY)
        assert held is False


def test_is_lock_held_returns_false_when_not_held():
    session = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.scalar.return_value = None
    session.execute.return_value = mock_cursor

    held = is_lock_held(session, PIPELINE_LOCK_KEY)

    assert held is False


import pytest


def test_is_lock_held_reraises_on_query_error():
    session = MagicMock()
    session.execute.side_effect = Exception("connection failure: timeout")

    with pytest.raises(Exception, match="connection failure"):
        is_lock_held(session, PIPELINE_LOCK_KEY)


async def test_pipeline_run_status_preserves_running_on_inconclusive_lock(monkeypatch):
    from datetime import datetime, timezone, timedelta
    from types import SimpleNamespace
    import app.main as main

    fake_run = SimpleNamespace(
        run_id="run-123",
        status="running",
        quality_state=None,
        worker_started_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        enqueued_at=datetime.now(timezone.utc) - timedelta(minutes=6),
        run_completed_at=None,
        duration_seconds=None,
        error_message=None,
        stage_results_json="{}",
        llm_success_count=0,
        operational_fallback_count=0,
        policy_fallback_count=0,
        commentary_generation_mode=None,
        artifact_version=None,
        promoted_artifact_version=None,
        train_model_requested=False,
    )

    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.__exit__.return_value = None
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = fake_run
    mock_session.query.return_value = mock_query

    monkeypatch.setattr(main, "get_db_type", lambda: "postgresql")
    monkeypatch.setattr(main, "SessionLocal", lambda: mock_session)

    # Simulate transient DB failure when checking pg_locks
    def fake_is_lock_held(_session, _key):
        raise RuntimeError("Transient connection pool timeout")

    import adapters.db.lock as db_lock
    monkeypatch.setattr(db_lock, "is_lock_held", fake_is_lock_held)

    result = await main.get_pipeline_run_status("run-123")

    # Run should remain running, never failed with worker_lost_lock_before_finalizing
    assert result["status"] == "running"
    assert fake_run.status == "running"
    assert fake_run.error_message is None
