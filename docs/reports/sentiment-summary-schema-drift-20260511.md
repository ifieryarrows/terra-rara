# Sentiment Summary Schema Drift - 2026-05-11

## Summary

Fixed the `/api/sentiment/summary` 500 caused by schema drift between the ORM model and the deployed PostgreSQL table. `DailySentimentV2` expected the weekly market-date columns added by `003_weekly_tft_market_date.sql`, while the live `daily_sentiments_v2` table still had the older shape.

## Root Cause

- `Base.metadata.create_all()` verifies tables but does not add new columns to existing tables.
- `backend/app/db.py` did not run the weekly market-date sentiment migration during startup.
- `/api/sentiment/summary` selected the full `DailySentimentV2` entity, so SQLAlchemy included `daily_sentiments_v2.market_date` in the SELECT list even though the endpoint only needed stable summary fields.

## Changes

- Added a startup migration helper that ensures weekly market-date columns and indexes exist on `news_sentiments_v2` and `daily_sentiments_v2`.
- Made the weekly sentiment migration startup-critical: failures are logged and re-raised instead of hidden as debug-only migration checks.
- Changed `/api/sentiment/summary` to project only `date`, `sentiment_index`, `news_count`, and `avg_confidence` from `DailySentimentV2`.
- Changed current-sentiment inference to query only `DailySentimentV2.sentiment_index`.
- Added a SQLite old-schema regression test for the weekly sentiment migration helper.
- Added an API regression assertion that the sentiment summary query does not load the full `DailySentimentV2` entity.

## Validation

```bash
py -m py_compile backend/app/db.py backend/app/main.py backend/app/inference.py
```

Result: passed.

```bash
py -m pytest backend/tests/test_api.py backend/tests/test_db_migrations.py -q
```

Result: `28 passed, 5 warnings`.

```bash
$env:PYTHONPATH='backend'; py -c "from app.db import init_db; init_db(); print('init_db completed')"
```

Result: `init_db completed`; the startup migration path completed against the configured PostgreSQL database.

PostgreSQL schema check:

```text
DAILY_WEEKLY_COLUMNS=after_close_news_count,cutoff_version,days_since_last_material_news,market_date,material_news_count,stale_sentiment_flag
MISSING=
```

```bash
$env:PYTHONPATH='backend'; py -c "import asyncio; from app.main import get_sentiment_summary; result = asyncio.run(get_sentiment_summary(days=7, recent_limit=6)); print(result['source']); print(result['data_freshness']['window_days']); print(len(result['recent_articles']))"
```

Result: `daily_v2`, `7`, `6`.

## Runtime Notes

- Docker CLI was not available in this Windows environment, so the Docker-container `psql` path could not be used here.
- A temporary local Uvicorn smoke was blocked by missing local Redis on `localhost:6379`. The process was stopped and the endpoint was smoke-tested directly against the real DB session path instead.
- No API response shape changed.
