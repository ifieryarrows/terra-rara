"""Pipeline lock helpers.

PostgreSQL advisory locks are authoritative for the distributed production
worker. File locking remains the local/SQLite fallback.
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from filelock import FileLock, Timeout

from app.settings import get_settings

logger = logging.getLogger(__name__)


class PipelineLock:
    """
    File-based lock for pipeline operations.
    Prevents concurrent data ingestion or model training.
    """
    
    def __init__(self, lock_file: Optional[str] = None, timeout: int = 0):
        """
        Initialize the lock.
        
        Args:
            lock_file: Path to lock file. If None, uses settings.
            timeout: Seconds to wait for lock. 0 = non-blocking, -1 = wait forever.
        """
        settings = get_settings()
        self.lock_file = Path(lock_file or settings.pipeline_lock_file)
        self.timeout = timeout
        self._lock: Optional[FileLock] = None
        self._acquired = False
        
        # Ensure lock directory exists
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
    
    def acquire(self) -> bool:
        """
        Try to acquire the lock.
        
        Returns:
            True if lock acquired, False if already locked.
        """
        if self._acquired:
            return True
        
        self._lock = FileLock(self.lock_file, timeout=self.timeout)
        
        try:
            self._lock.acquire(timeout=self.timeout)
            self._acquired = True
            
            # Write lock info
            self._write_lock_info()
            
            logger.info(f"Pipeline lock acquired: {self.lock_file}")
            return True
            
        except Timeout:
            logger.warning(f"Could not acquire pipeline lock (already locked): {self.lock_file}")
            return False
    
    def release(self):
        """Release the lock."""
        if self._lock and self._acquired:
            self._lock.release()
            self._acquired = False
            
            # Remove lock info file
            info_file = Path(str(self.lock_file) + ".info")
            if info_file.exists():
                try:
                    info_file.unlink()
                except Exception:
                    pass
            
            logger.info(f"Pipeline lock released: {self.lock_file}")
    
    def _write_lock_info(self):
        """Write info about who holds the lock."""
        info_file = Path(str(self.lock_file) + ".info")
        try:
            info = {
                "pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
            info_file.write_text(str(info))
        except Exception as e:
            logger.debug(f"Could not write lock info: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise RuntimeError("Could not acquire pipeline lock")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def is_pipeline_locked() -> bool:
    """
    Check if the pipeline is currently locked.
    Non-blocking check.
    """
    # The API and ARQ worker are separate processes in production, so a local
    # file lock cannot describe the worker's PostgreSQL advisory lock. Use the
    # same authority as the worker for health and enqueue preflight checks.
    from app.db import SessionLocal, get_db_type

    if get_db_type() == "postgresql":
        from adapters.db.lock import PIPELINE_LOCK_KEY, is_lock_held

        try:
            with SessionLocal() as session:
                return is_lock_held(session, PIPELINE_LOCK_KEY)
        except Exception as exc:
            # This is only a preflight visibility check. The worker still uses
            # try_acquire_lock as the race-safe authority.
            logger.warning("Could not inspect PostgreSQL pipeline lock: %s", exc)

    settings = get_settings()
    lock_file = Path(settings.pipeline_lock_file)
    
    if not lock_file.exists():
        return False
    
    # Try to acquire briefly
    lock = FileLock(lock_file, timeout=0)
    try:
        lock.acquire(timeout=0)
        lock.release()
        return False
    except Timeout:
        return True


def get_lock_info() -> Optional[dict]:
    """
    Get information about the current lock holder.
    """
    settings = get_settings()
    info_file = Path(str(settings.pipeline_lock_file) + ".info")
    
    if not info_file.exists():
        return None
    
    try:
        content = info_file.read_text()
        # Parse the dict string (simple approach)
        import ast
        return ast.literal_eval(content)
    except Exception:
        return None


@contextmanager
def pipeline_lock(timeout: int = 0):
    """
    Context manager for pipeline locking.
    
    Usage:
        with pipeline_lock():
            # Do heavy work
            pass
    
    Args:
        timeout: Seconds to wait for lock. 0 = non-blocking, -1 = wait forever.
    
    Raises:
        RuntimeError: If lock cannot be acquired.
    """
    lock = PipelineLock(timeout=timeout)
    try:
        if not lock.acquire():
            raise RuntimeError("Pipeline is locked by another process")
        yield lock
    finally:
        lock.release()

