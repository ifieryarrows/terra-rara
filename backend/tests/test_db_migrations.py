"""Tests for idempotent database migration helpers."""

from sqlalchemy import create_engine, text

from app.db import _ensure_weekly_sentiment_schema


def _sqlite_columns(conn, table_name: str) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    return {row[1] for row in rows}


def _sqlite_indexes(conn, table_name: str) -> set[str]:
    rows = conn.execute(text(f"PRAGMA index_list({table_name})")).fetchall()
    return {row[1] for row in rows}


def test_weekly_sentiment_migration_adds_columns_to_old_sqlite_schema():
    engine = create_engine("sqlite:///:memory:")

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE news_sentiments_v2 (
                    id INTEGER PRIMARY KEY,
                    final_score FLOAT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE daily_sentiments_v2 (
                    id INTEGER PRIMARY KEY,
                    date TIMESTAMP NOT NULL,
                    sentiment_index FLOAT NOT NULL,
                    news_count INTEGER NOT NULL DEFAULT 0,
                    avg_confidence FLOAT,
                    avg_relevance FLOAT,
                    source_version TEXT DEFAULT 'v2',
                    aggregated_at TIMESTAMP
                )
                """
            )
        )

        _ensure_weekly_sentiment_schema(conn, is_sqlite=True)
        _ensure_weekly_sentiment_schema(conn, is_sqlite=True)

        news_columns = _sqlite_columns(conn, "news_sentiments_v2")
        daily_columns = _sqlite_columns(conn, "daily_sentiments_v2")

        assert {"market_date", "available_at", "cutoff_version"} <= news_columns
        assert {
            "market_date",
            "material_news_count",
            "after_close_news_count",
            "stale_sentiment_flag",
            "days_since_last_material_news",
            "cutoff_version",
        } <= daily_columns
        assert "ix_news_sentiments_v2_market_date" in _sqlite_indexes(
            conn, "news_sentiments_v2"
        )
        assert "ix_daily_sentiments_v2_market_date" in _sqlite_indexes(
            conn, "daily_sentiments_v2"
        )
