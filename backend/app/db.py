"""
Database connection and session management.
SQLite with WAL mode for concurrent read/write support.
"""

import importlib.util
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base

from app.settings import get_settings

logger = logging.getLogger(__name__)

WEEKLY_NEWS_SENTIMENT_COLUMNS = (
    ("market_date", "DATE", "DATE"),
    ("available_at", "TIMESTAMPTZ", "TIMESTAMP"),
    ("cutoff_version", "TEXT DEFAULT 'market_close_v1'", "TEXT DEFAULT 'market_close_v1'"),
)

WEEKLY_DAILY_SENTIMENT_COLUMNS = (
    ("market_date", "DATE", "DATE"),
    ("material_news_count", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
    ("after_close_news_count", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
    ("stale_sentiment_flag", "BOOLEAN DEFAULT FALSE", "BOOLEAN DEFAULT FALSE"),
    ("days_since_last_material_news", "INTEGER DEFAULT 999", "INTEGER DEFAULT 999"),
    ("cutoff_version", "TEXT DEFAULT 'market_close_v1'", "TEXT DEFAULT 'market_close_v1'"),
)

TFT_MODEL_METADATA_COLUMNS = (
    ("quality_gate_passed", "BOOLEAN", "BOOLEAN"),
)

TFT_PREDICTION_SNAPSHOT_COLUMNS = (
    ("explain_json", "JSON", "JSON"),
)

PRODUCTION_FLOW_AI_COMMENTARY_COLUMNS = (
    ("generation_mode", "VARCHAR(32) DEFAULT 'unknown'", "VARCHAR(32) DEFAULT 'unknown'"),
    ("fallback_reason", "VARCHAR(64)", "VARCHAR(64)"),
    ("attempts_json", "TEXT", "TEXT"),
)

PRODUCTION_FLOW_PIPELINE_COLUMNS = (
    ("enqueued_at", "TIMESTAMPTZ", "TIMESTAMP"),
    ("worker_started_at", "TIMESTAMPTZ", "TIMESTAMP"),
    ("trigger_source", "VARCHAR(32)", "VARCHAR(32)"),
    ("train_model_requested", "BOOLEAN DEFAULT FALSE", "BOOLEAN DEFAULT FALSE"),
    ("job_id", "VARCHAR(128)", "VARCHAR(128)"),
    ("llm_success_count", "INTEGER", "INTEGER"),
    ("operational_fallback_count", "INTEGER", "INTEGER"),
    ("policy_fallback_count", "INTEGER", "INTEGER"),
    ("commentary_generation_mode", "VARCHAR(32)", "VARCHAR(32)"),
    ("artifact_version", "VARCHAR(80)", "VARCHAR(80)"),
    ("promoted_artifact_version", "VARCHAR(80)", "VARCHAR(80)"),
    ("stage_results_json", "TEXT", "TEXT"),
)

PRODUCTION_FLOW_NEWS_RAW_COLUMNS = (
    ("publisher", "VARCHAR(300)", "VARCHAR(300)"),
)

PRODUCTION_FLOW_NEWS_PROCESSED_COLUMNS = (
    ("dedup_version", "VARCHAR(20) DEFAULT 'legacy_v1'", "VARCHAR(20) DEFAULT 'legacy_v1'"),
    ("duplicate_of_id", "BIGINT", "BIGINT"),
)

# SQLAlchemy declarative base
Base = declarative_base()

# Global engine and session factory (lazy initialized)
_engine = None
_SessionLocal = None


def normalize_database_url(raw_url: str) -> tuple[str, dict]:
    """
    Normalize database URL and determine any driver-specific connect_args.

    Handles:
    - postgres:// -> postgresql:// legacy scheme
    - Driver selection:
      * If URL doesn't specify a driver (postgresql://):
        - If psycopg (v3) is available, use postgresql+psycopg://
        - If psycopg is not available but psycopg2 is, use postgresql+psycopg2://
      * If URL specifies postgresql+psycopg:// but psycopg is not installed and psycopg2 is,
        fallback to postgresql+psycopg2://
      * If URL specifies postgresql+psycopg2:// but psycopg2 is not installed and psycopg is,
        fallback to postgresql+psycopg://
    - Supabase / PgBouncer pooler support:
      * When using psycopg (v3) with a pooler (e.g. port 6543 / pooler.supabase.com / pgbouncer),
        prepared statements must be disabled via prepare_threshold=None to avoid
        DuplicatePreparedStatement / InvalidSqlStatementName errors in transaction mode.
    """
    connect_args: dict = {}
    if not raw_url:
        return "", connect_args

    url = raw_url

    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    if url.startswith("sqlite"):
        return url, connect_args

    has_psycopg = importlib.util.find_spec("psycopg") is not None
    has_psycopg2 = importlib.util.find_spec("psycopg2") is not None

    is_psycopg = False
    if url.startswith("postgresql://"):
        rest = url[len("postgresql://"):]
        if has_psycopg:
            url = "postgresql+psycopg://" + rest
            is_psycopg = True
        elif has_psycopg2:
            url = "postgresql+psycopg2://" + rest
    elif url.startswith("postgresql+psycopg://"):
        rest = url[len("postgresql+psycopg://"):]
        if not has_psycopg and has_psycopg2:
            logger.warning(
                "psycopg (v3) requested in DATABASE_URL but not installed; falling back to psycopg2"
            )
            url = "postgresql+psycopg2://" + rest
        elif has_psycopg:
            is_psycopg = True
    elif url.startswith("postgresql+psycopg2://"):
        rest = url[len("postgresql+psycopg2://"):]
        if not has_psycopg2 and has_psycopg:
            logger.warning(
                "psycopg2 requested in DATABASE_URL but not installed; falling back to psycopg"
            )
            url = "postgresql+psycopg://" + rest
            is_psycopg = True

    if is_psycopg and (
        ":6543" in url
        or "pooler.supabase.com" in url
        or "supabase.co" in url
        or "supabase.com" in url
        or "pgbouncer" in url.lower()
    ):
        connect_args["prepare_threshold"] = None

    return url, connect_args


def get_engine():
    """Get or create the database engine."""
    global _engine
    
    if _engine is None:
        settings = get_settings()
        raw_database_url = settings.database_url
        database_url, extra_connect_args = normalize_database_url(raw_database_url)
        
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
            engine_kwargs["pool_recycle"] = 300
            if extra_connect_args:
                engine_kwargs.setdefault("connect_args", {}).update(extra_connect_args)
        
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


def _sqlite_columns(conn, table_name: str) -> set[str]:
    result = conn.execute(text(f"PRAGMA table_info({table_name})"))
    return {row[1] for row in result.fetchall()}


def _ensure_columns(conn, table_name: str, columns, is_sqlite: bool) -> None:
    if is_sqlite:
        existing = _sqlite_columns(conn, table_name)
        if not existing:
            raise RuntimeError(f"Expected table {table_name} to exist before migration")
        for column_name, _pg_type, sqlite_type in columns:
            if column_name not in existing:
                conn.execute(
                    text(
                        f"ALTER TABLE {table_name} "
                        f"ADD COLUMN {column_name} {sqlite_type}"
                    )
                )
        return

    for column_name, pg_type, _sqlite_type in columns:
        conn.execute(
            text(
                f"ALTER TABLE {table_name} "
                f"ADD COLUMN IF NOT EXISTS {column_name} {pg_type}"
            )
        )


def _ensure_weekly_sentiment_schema(conn, is_sqlite: bool) -> None:
    """Ensure market-date sentiment columns from migration 003 exist."""
    _ensure_columns(conn, "news_sentiments_v2", WEEKLY_NEWS_SENTIMENT_COLUMNS, is_sqlite)
    _ensure_columns(conn, "daily_sentiments_v2", WEEKLY_DAILY_SENTIMENT_COLUMNS, is_sqlite)

    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_news_sentiments_v2_market_date "
            "ON news_sentiments_v2(market_date)"
        )
    )
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_daily_sentiments_v2_market_date "
            "ON daily_sentiments_v2(market_date)"
        )
    )


def _ensure_tft_model_metadata_schema(conn, is_sqlite: bool) -> None:
    """Ensure promotion status exists on databases created before migration 004."""
    _ensure_columns(
        conn,
        "tft_model_metadata",
        TFT_MODEL_METADATA_COLUMNS,
        is_sqlite,
    )


def ensure_tft_model_metadata_schema() -> None:
    """Apply the narrow promotion migration before an external CI DB write."""
    from app.models import TFTModelMetadata

    engine = get_engine()
    TFTModelMetadata.__table__.create(bind=engine, checkfirst=True)
    with engine.begin() as conn:
        _ensure_tft_model_metadata_schema(
            conn,
            is_sqlite=engine.dialect.name == "sqlite",
        )


def _ensure_production_flow_schema(conn, is_sqlite: bool) -> None:
    """Additive schema used by production-flow observability and news dedup."""
    _ensure_columns(conn, "ai_commentaries", PRODUCTION_FLOW_AI_COMMENTARY_COLUMNS, is_sqlite)
    _ensure_columns(conn, "pipeline_run_metrics", PRODUCTION_FLOW_PIPELINE_COLUMNS, is_sqlite)
    _ensure_columns(conn, "news_raw", PRODUCTION_FLOW_NEWS_RAW_COLUMNS, is_sqlite)
    _ensure_columns(conn, "news_processed", PRODUCTION_FLOW_NEWS_PROCESSED_COLUMNS, is_sqlite)

    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_news_raw_publisher ON news_raw(publisher)"))
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_news_processed_duplicate_of_id "
            "ON news_processed(duplicate_of_id)"
        )
    )
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_pipeline_run_metrics_enqueued_at "
            "ON pipeline_run_metrics(enqueued_at)"
        )
    )

def _run_migrations(engine):
    """
    Run necessary database migrations for schema changes.
    These are safe to run multiple times (idempotent).
    """
    is_sqlite = engine.dialect.name == "sqlite"
    
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

        # Migration: V2 stage-2 metrics columns
        v2_metric_columns = [
            ("articles_scored_v2", "INTEGER"),
            ("llm_parse_fail_count", "INTEGER"),
            ("escalation_count", "INTEGER"),
            ("fallback_count", "INTEGER"),
        ]
        try:
            if is_sqlite:
                result = conn.execute(text("PRAGMA table_info(pipeline_run_metrics)"))
                columns = [row[1] for row in result.fetchall()]
                for column_name, column_type in v2_metric_columns:
                    if column_name not in columns:
                        conn.execute(
                            text(
                                f"ALTER TABLE pipeline_run_metrics ADD COLUMN {column_name} {column_type}"
                            )
                        )
                conn.commit()
            else:
                for column_name, column_type in v2_metric_columns:
                    conn.execute(
                        text(
                            f"ALTER TABLE pipeline_run_metrics "
                            f"ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
                        )
                    )
                conn.commit()
            logger.info("Migration: Ensured V2 Stage-2 metric columns exist")
        except Exception as e:
            logger.debug(f"Migration check for V2 metric columns: {e}")

        # Migration: TFT-ASRO metric columns on pipeline_run_metrics
        tft_metric_columns = [
            ("tft_embeddings_computed", "INTEGER"),
            ("tft_trained", "BOOLEAN DEFAULT FALSE"),
            ("tft_val_loss", "FLOAT"),
            ("tft_sharpe", "FLOAT"),
            ("tft_directional_accuracy", "FLOAT"),
            ("tft_snapshot_generated", "BOOLEAN DEFAULT FALSE"),
        ]
        try:
            if is_sqlite:
                result = conn.execute(text("PRAGMA table_info(pipeline_run_metrics)"))
                columns = [row[1] for row in result.fetchall()]
                for column_name, column_type in tft_metric_columns:
                    col_name_only = column_name
                    if col_name_only not in columns:
                        conn.execute(
                            text(
                                f"ALTER TABLE pipeline_run_metrics ADD COLUMN {column_name} {column_type}"
                            )
                        )
                conn.commit()
            else:
                for column_name, column_type in tft_metric_columns:
                    conn.execute(
                        text(
                            f"ALTER TABLE pipeline_run_metrics "
                            f"ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
                        )
                    )
                conn.commit()
            logger.info("Migration: Ensured TFT-ASRO metric columns exist")
        except Exception as e:
            logger.debug(f"Migration check for TFT metric columns: {e}")

        try:
            _ensure_weekly_sentiment_schema(conn, is_sqlite)
            conn.commit()
            logger.info("Migration: Ensured weekly market-date sentiment schema exists")
        except Exception:
            conn.rollback()
            logger.exception("Migration failed for weekly market-date sentiment schema")
            raise

        try:
            _ensure_tft_model_metadata_schema(conn, is_sqlite)
            conn.commit()
            logger.info("Migration: Ensured TFT model promotion schema exists")
        except Exception:
            conn.rollback()
            logger.exception("Migration failed for TFT model promotion schema")
            raise

        # Only a missing snapshot table is tolerated (init_db creates it before
        # migrations run, but ad-hoc migration passes must not fail on it);
        # any other failure is a real schema error and must surface loudly.
        try:
            if inspect(conn).has_table("tft_prediction_snapshots"):
                _ensure_columns(
                    conn,
                    "tft_prediction_snapshots",
                    TFT_PREDICTION_SNAPSHOT_COLUMNS,
                    is_sqlite,
                )
                conn.commit()
                logger.info(
                    "Migration: Ensured TFT prediction snapshot explain schema exists"
                )
            else:
                logger.debug(
                    "Migration skipped: tft_prediction_snapshots table does not exist"
                )
        except Exception:
            conn.rollback()
            logger.exception(
                "Migration failed for TFT prediction snapshot explain schema"
            )
            raise

        try:
            _ensure_production_flow_schema(conn, is_sqlite)
            conn.commit()
            logger.info("Migration: Ensured production-flow schema exists")
        except Exception:
            conn.rollback()
            logger.exception("Migration failed for production-flow schema")
            raise


def init_db():
    """
    Initialize the database - create all tables.
    Safe to call multiple times (uses CREATE IF NOT EXISTS).
    Also runs any necessary migrations for schema changes.
    """
    # Import models to register them with Base
    from app import models  # noqa: F401

    engine = get_engine()
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
    except Exception as ddl_err:
        # On PostgreSQL with concurrent startup (API process + worker), two
        # processes may race to run CREATE TABLE simultaneously.  Even though
        # SQLAlchemy uses CREATE TABLE IF NOT EXISTS, the internal pg_type
        # catalog update is not fully atomic and can raise a UniqueViolation
        # on the pg_type_typname_nsp_index constraint.
        # If the error is that kind of DDL race we simply log a warning and
        # continue — the table was created by the other process.
        err_str = str(ddl_err)
        if "pg_type_typname_nsp_index" in err_str or (
            "UniqueViolation" in err_str and "already exists" in err_str.lower()
        ):
            logger.warning(
                "DDL race condition detected during startup (concurrent process "
                "already created the table). Continuing with existing schema."
            )
        else:
            raise

    # Run migrations for existing tables
    _run_migrations(engine)


def get_db_type() -> str:
    """Return the database type (sqlite, postgresql, etc.)."""
    settings = get_settings()
    url = settings.database_url
    if not url:
        return "unknown"

    if url.startswith("sqlite"):
        return "sqlite"
    elif url.startswith("postgresql") or url.startswith("postgres"):
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

