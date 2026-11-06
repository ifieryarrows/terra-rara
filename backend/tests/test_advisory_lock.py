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
    assert "classid::bigint" in query_text
    assert "objid::bigint" in query_text
    assert "classid" in params
    assert "objid" in params


def test_is_lock_held_returns_false_when_not_held():
    session = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.scalar.return_value = None
    session.execute.return_value = mock_cursor

    held = is_lock_held(session, PIPELINE_LOCK_KEY)

    assert held is False


def test_is_lock_held_safe_on_query_error():
    session = MagicMock()
    session.execute.side_effect = Exception("no such table: pg_locks")

    held = is_lock_held(session, PIPELINE_LOCK_KEY)

    assert held is False
