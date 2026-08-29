"""Audit or repair production-flow derived data without deleting raw news.

The default invocation is a dry run. ``--apply`` marks duplicate relationships
and rebuilds the fully derived daily sentiment table. ``--refresh-prices`` also
performs overlap ingestion; invalid historical closes remain auditable but are
excluded everywhere by the shared finite-price contract.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ai_engine import aggregate_daily_sentiment_v2
from app.data_manager import ingest_prices
from app.db import SessionLocal, init_db
from app.models import DailySentimentV2, PriceBar
from pipelines.processing.news import backfill_content_dedup


def _invalid_price_count(session) -> int:
    count = 0
    for (close,) in session.query(PriceBar.close).all():
        try:
            value = float(close)
        except (TypeError, ValueError):
            count += 1
            continue
        if not math.isfinite(value) or value <= 0:
            count += 1
    return count


def run_repair(*, apply: bool, refresh_prices: bool) -> dict:
    init_db()
    with SessionLocal() as session:
        result = {
            "mode": "apply" if apply else "dry_run",
            "dedup": backfill_content_dedup(session, dry_run=not apply),
            "invalid_price_rows_before": _invalid_price_count(session),
            "daily_sentiment_rows_before": session.query(DailySentimentV2).count(),
        }

        if apply:
            # This table is fully derived from canonical article-level scores.
            # Rebuilding it prevents aggregate rows from retaining old duplicate
            # contributions after the non-destructive relationship backfill.
            session.query(DailySentimentV2).delete(synchronize_session=False)
            session.commit()
            result["daily_sentiment_rows_rebuilt"] = aggregate_daily_sentiment_v2(session)
            if refresh_prices:
                result["price_ingest"] = ingest_prices(session)
                session.commit()
        else:
            result["daily_sentiment_rows_rebuilt"] = 0
            result["price_ingest"] = "skipped_in_dry_run"

        result["invalid_price_rows_after"] = _invalid_price_count(session)
        result["daily_sentiment_rows_after"] = session.query(DailySentimentV2).count()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/repair news dedup and finite-price flow")
    parser.add_argument("--apply", action="store_true", help="Apply dedup relationships and rebuild derived aggregates")
    parser.add_argument("--refresh-prices", action="store_true", help="Run overlap price ingest (requires --apply)")
    args = parser.parse_args()
    if args.refresh_prices and not args.apply:
        parser.error("--refresh-prices requires --apply")
    print(json.dumps(run_repair(apply=args.apply, refresh_prices=args.refresh_prices), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
