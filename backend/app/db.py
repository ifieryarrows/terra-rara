"""
Database connection and session management.
SQLite with WAL mode for concurrent read/write support.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base

from app.settings import get_settings

logger = logging.getLogger(__name__)

# SQLAlchemy declarative base
Base = declarative_base()

# Global engine and session factory (lazy initialized)
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    
    if _engine is None:
        settings = get_settings()
        database_url = settings.database_url
        
        # Determine if SQLite
        is_sqlite = database_url.startswith("sqlite")
        
        # Engine configuration
        engine_kwargs = {
            "echo": settings.log_level == "DEBUG",
            "pool_pre_ping": True,
        }
        
        if is_sqlite:
            # SQLite-specific settings
            engine_kwargs["connect_args"] = {
                "check_same_thread": False,
                "timeout": 30,
            }
        else:
            # PostgreSQL (Supabase) - connection pooling
            engine_kwargs["pool_size"] = 5
            engine_kwargs["max_overflow"] = 10
            engine_kwargs["pool_timeout"] = 30
        
        _engine = create_engine(database_url, **engine_kwargs)
        
        # SQLite WAL mode and pragmas
        if is_sqlite:
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                # WAL mode for concurrent reads
                cursor.execute("PRAGMA journal_mode=WAL")
                # Busy timeout (ms) - wait for locks instead of immediate failure
                cursor.execute("PRAGMA busy_timeout=5000")
                # Synchronous mode - balance between safety and speed
                cursor.execute("PRAGMA synchronous=NORMAL")
                # Foreign keys enforcement
                cursor.execute("PRAGMA foreign_keys=ON")
                # Memory-mapped I/O (faster reads)
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                cursor.close()
            
            logger.info("SQLite configured with WAL mode and performance pragmas")
        
        # Log sanitized URL (hide password)
        import re
        safe_url = re.sub(r'://[^:]+:[^@]+@', '://***:***@', database_url.split('?')[0])
        logger.info(f"Database engine created: {safe_url}")
    
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the session factory."""
    global _SessionLocal
    
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
    
    return _SessionLocal


def SessionLocal() -> Session:
    """Create a new database session."""
    factory = get_session_factory()
    return factory()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Automatically commits on success, rolls back on exception.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _run_migrations(engine):
    """
    Run necessary database migrations for schema changes.
    These are safe to run multiple times (idempotent).
    """
    settings = get_settings()
    is_sqlite = settings.database_url.startswith("sqlite")
    
    with engine.connect() as conn:
        # Migration: Add ai_stance column to ai_commentaries if it doesn't exist
        try:
            if is_sqlite:
                # SQLite doesn't support IF NOT EXISTS for columns, check manually
                result = conn.execute(text("PRAGMA table_info(ai_commentaries)"))
                columns = [row[1] for row in result.fetchall()]
                if 'ai_stance' not in columns:
                    conn.execute(text("ALTER TABLE ai_commentaries ADD COLUMN ai_stance VARCHAR(20) DEFAULT 'NEUTRAL'"))
                    conn.commit()
                    logger.info("Migration: Added ai_stance column to ai_commentaries")
            else:
                # PostgreSQL supports IF NOT EXISTS via DO block
                conn.execute(text("""
                    ALTER TABLE ai_commentaries 
                    ADD COLUMN IF NOT EXISTS ai_stance VARCHAR(20) DEFAULT 'NEUTRAL'
                """))
                conn.commit()
                logger.info("Migration: Ensured ai_stance column exists in ai_commentaries")
        except Exception as e:
            logger.debug(f"Migration check for ai_stance: {e}")


def init_db():
    """
    Initialize the database - create all tables.
    Safe to call multiple times (uses CREATE IF NOT EXISTS).
    Also runs any necessary migrations for schema changes.
    """
    # Import models to register them with Base
    from app import models  # noqa: F401
    
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")
    
    # Run migrations for existing tables
    _run_migrations(engine)


def get_db_type() -> str:
    """Return the database type (sqlite, postgresql, etc.)."""
    settings = get_settings()
    url = settings.database_url
    
    if url.startswith("sqlite"):
        return "sqlite"
    elif url.startswith("postgresql"):
        return "postgresql"
    elif url.startswith("mysql"):
        return "mysql"
    else:
        return "unknown"


def check_db_connection() -> bool:
    """Test database connectivity."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False

