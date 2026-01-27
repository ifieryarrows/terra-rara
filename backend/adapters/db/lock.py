"""
Distributed lock using PostgreSQL advisory locks.

Advisory locks are:
- Session-based: automatically released when connection closes
- Non-blocking: can check without waiting
- Reliable: no stale locks after crash

This is the AUTHORITY for pipeline locking.
`pipeline_locks` table is for VISIBILITY only (best-effort).
"""

import hashlib
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _lock_key_to_id(lock_key: str) -> int:
    """
    Convert string lock key to bigint for pg_advisory_lock.
    Uses first 15 hex chars of SHA-256 to fit in signed bigint.
    """
    hash_hex = hashlib.sha256(lock_key.encode()).hexdigest()[:15]
    return int(hash_hex, 16)


def try_acquire_lock(session: Session, lock_key: str) -> bool:
    """
    Try to acquire advisory lock (non-blocking).
    
    Args:
        session: SQLAlchemy session (lock is tied to this connection)
        lock_key: String identifier for the lock (e.g., "pipeline:daily")
        
    Returns:
        True if lock acquired, False if already held by another session
        
    IMPORTANT: Lock is held until session.close() or explicit release.
    Keep the same session alive for the entire pipeline run.
    """
    lock_id = _lock_key_to_id(lock_key)
    
    result = session.execute(
        text("SELECT pg_try_advisory_lock(:lock_id)"),
        {"lock_id": lock_id}
    ).scalar()
    
    if result:
        logger.info(f"Advisory lock acquired: {lock_key} (id={lock_id})")
    else:
        logger.warning(f"Advisory lock NOT acquired (held by another): {lock_key}")
    
    return bool(result)


def release_lock(session: Session, lock_key: str) -> bool:
    """
    Release advisory lock explicitly.
    
    Usually not needed - lock auto-releases on session close.
    Use this for early release if pipeline completes before session ends.
    """
    lock_id = _lock_key_to_id(lock_key)
    
    result = session.execute(
        text("SELECT pg_advisory_unlock(:lock_id)"),
        {"lock_id": lock_id}
    ).scalar()
    
    if result:
        logger.info(f"Advisory lock released: {lock_key}")
    else:
        logger.warning(f"Advisory lock release failed (not held?): {lock_key}")
    
    return bool(result)


def is_lock_held(session: Session, lock_key: str) -> bool:
    """
    Check if lock is currently held by ANY session.
    
    This is a weak check - another session could acquire between check and use.
    Use try_acquire_lock for actual locking.
    """
    lock_id = _lock_key_to_id(lock_key)
    
    # Try to acquire, then immediately release if successful
    acquired = session.execute(
        text("SELECT pg_try_advisory_lock(:lock_id)"),
        {"lock_id": lock_id}
    ).scalar()
    
    if acquired:
        session.execute(
            text("SELECT pg_advisory_unlock(:lock_id)"),
            {"lock_id": lock_id}
        )
        return False  # Was NOT held
    else:
        return True  # IS held by another


@contextmanager
def advisory_lock(session: Session, lock_key: str, raise_on_fail: bool = True):
    """
    Context manager for advisory lock.
    
    Usage:
        with advisory_lock(session, "pipeline:daily"):
            # Do work - lock held
            pass
        # Lock released
        
    Args:
        session: SQLAlchemy session
        lock_key: Lock identifier
        raise_on_fail: If True, raise RuntimeError if lock not acquired
        
    Raises:
        RuntimeError: If lock not acquired and raise_on_fail=True
    """
    acquired = try_acquire_lock(session, lock_key)
    
    if not acquired:
        if raise_on_fail:
            raise RuntimeError(f"Could not acquire lock: {lock_key}")
        else:
            yield False
            return
    
    try:
        yield True
    finally:
        release_lock(session, lock_key)


# Lock key constants
PIPELINE_LOCK_KEY = "pipeline:daily"


def write_lock_visibility(
    session: Session,
    lock_key: str,
    run_id: str,
    holder_id: Optional[str] = None
) -> None:
    """
    Write lock info to pipeline_locks table for visibility.
    
    This is BEST-EFFORT only - not the authority.
    If this fails, pipeline continues.
    """
    try:
        # Upsert lock info
        session.execute(
            text("""
                INSERT INTO pipeline_locks (lock_key, holder_id, run_id, acquired_at)
                VALUES (:lock_key, :holder_id, :run_id, :acquired_at)
                ON CONFLICT (lock_key) DO UPDATE SET
                    holder_id = EXCLUDED.holder_id,
                    run_id = EXCLUDED.run_id,
                    acquired_at = EXCLUDED.acquired_at
            """),
            {
                "lock_key": lock_key,
                "holder_id": holder_id,
                "run_id": run_id,
                "acquired_at": datetime.now(timezone.utc),
            }
        )
        session.commit()
    except Exception as e:
        logger.debug(f"Failed to write lock visibility (best-effort): {e}")
        session.rollback()


def clear_lock_visibility(session: Session, lock_key: str) -> None:
    """
    Clear lock info from pipeline_locks table.
    Best-effort only.
    """
    try:
        session.execute(
            text("DELETE FROM pipeline_locks WHERE lock_key = :lock_key"),
            {"lock_key": lock_key}
        )
        session.commit()
    except Exception as e:
        logger.debug(f"Failed to clear lock visibility (best-effort): {e}")
        session.rollback()
