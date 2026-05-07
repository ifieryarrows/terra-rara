"""Backfill market-date sentiment alignment for weekly TFT training."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import date, datetime, timezone
from typing import Any

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import func

from app.ai_engine import aggregate_daily_sentiment_v2
from app.db import SessionLocal, init_db
from app.models import DailySentimentV2, NewsProcessed, NewsRaw, NewsSentimentV2
from pipelines.market_calendar import assign_market_date, is_after_close_news


def _parse_date(value: str) -> datetime:
    if value.lower() == "today":
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _audit_counts(session, start: datetime, end: datetime) -> dict[str, Any]:
    base = (
        session.query(NewsSentimentV2, NewsRaw)
        .join(NewsProcessed, NewsSentimentV2.news_processed_id == NewsProcessed.id)
        .join(NewsRaw, NewsProcessed.raw_id == NewsRaw.id)
        .filter(NewsRaw.published_at >= start, NewsRaw.published_at <= end)
    )
    rows = base.all()
    null_before = sum(1 for sentiment, _raw in rows if sentiment.market_date is None)
    mappings = []
    after_close = 0
    weekend = 0
    for _sentiment, raw in rows:
        md = assign_market_date(raw.published_at)
        local_weekend = raw.published_at.weekday() >= 5
        after = is_after_close_news(raw.published_at)
        after_close += int(after)
        weekend += int(local_weekend)
        if len(mappings) < 10 and (after or local_weekend):
            mappings.append(
                {
                    "published_at_utc": raw.published_at.isoformat(),
                    "market_date": md.isoformat(),
                }
            )
    if len(mappings) < 5:
        for _sentiment, raw in rows[: 10 - len(mappings)]:
            mappings.append(
                {
                    "published_at_utc": raw.published_at.isoformat(),
                    "market_date": assign_market_date(raw.published_at).isoformat(),
                }
            )

    return {
        "rows_to_update": len(rows),
        "null_market_date_before": null_before,
        "null_market_date_after_expected": 0 if rows else null_before,
        "after_close_shift_count": after_close,
        "weekend_shift_count": weekend,
        "daily_market_date_null_before": session.query(DailySentimentV2)
        .filter(DailySentimentV2.market_date.is_(None))
        .count(),
        "sample_mappings": mappings,
    }


def _backfill_news_sentiments(session, start: datetime, end: datetime) -> int:
    rows = (
        session.query(NewsSentimentV2, NewsRaw)
        .join(NewsProcessed, NewsSentimentV2.news_processed_id == NewsProcessed.id)
        .join(NewsRaw, NewsProcessed.raw_id == NewsRaw.id)
        .filter(NewsRaw.published_at >= start, NewsRaw.published_at <= end)
        .all()
    )
    for sentiment, raw in rows:
        sentiment.market_date = assign_market_date(raw.published_at)
        sentiment.available_at = raw.fetched_at or raw.published_at
        sentiment.cutoff_version = "market_close_v1"
    return len(rows)


def _duplicate_market_dates(session) -> list[dict[str, Any]]:
    rows = (
        session.query(DailySentimentV2.market_date, func.count(DailySentimentV2.id))
        .filter(DailySentimentV2.market_date.is_not(None))
        .group_by(DailySentimentV2.market_date)
        .having(func.count(DailySentimentV2.id) > 1)
        .all()
    )
    return [{"market_date": str(md), "count": int(count)} for md, count in rows]


def run_backfill(start: datetime, end: datetime, *, dry_run: bool) -> dict[str, Any]:
    init_db()
    with SessionLocal() as session:
        audit = _audit_counts(session, start, end)
        audit["dry_run"] = dry_run
        audit["start"] = start.date().isoformat()
        audit["end"] = end.date().isoformat()

        if dry_run:
            audit["daily_aggregate_rows_rebuilt"] = 0
            audit["duplicates_after_expected"] = _duplicate_market_dates(session)
            return audit

        updated = _backfill_news_sentiments(session, start, end)
        session.commit()
        rebuilt = aggregate_daily_sentiment_v2(session)
        session.commit()

        audit["rows_updated"] = updated
        audit["daily_aggregate_rows_rebuilt"] = rebuilt
        audit["news_market_date_null_after"] = (
            session.query(NewsSentimentV2)
            .filter(NewsSentimentV2.market_date.is_(None))
            .count()
        )
        audit["daily_market_date_null_after"] = (
            session.query(DailySentimentV2)
            .filter(DailySentimentV2.market_date.is_(None))
            .count()
        )
        audit["duplicates_after"] = _duplicate_market_dates(session)
        return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill market_date for TFT weekly sentiment features")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="today")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    audit = run_backfill(_parse_date(args.start), _parse_date(args.end), dry_run=bool(args.dry_run))
    print(json.dumps(audit, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
