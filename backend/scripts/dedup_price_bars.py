"""Deduplicate historical same-calendar-day PriceBar rows safely.

Defaults to dry-run mode (no writes). Pass ``--apply`` explicitly to delete
duplicate rows and normalize surviving timestamps to naive UTC midnight.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from datetime import datetime, timezone

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db import SessionLocal, init_db
from app.models import PriceBar


def _utc_midnight(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return datetime(dt.year, dt.month, dt.day)


def _is_valid_close(value) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def dedup_price_bars(session, *, dry_run: bool = True) -> dict:
    """Audit or remove duplicate ``PriceBar`` rows sharing (symbol, UTC date)."""
    rows = session.query(PriceBar).order_by(PriceBar.symbol.asc(), PriceBar.date.asc(), PriceBar.id.asc()).all()
    grouped: dict[tuple[str, datetime], list[PriceBar]] = defaultdict(list)
    symbols: set[str] = set()
    for row in rows:
        symbols.add(str(row.symbol))
        grouped[(str(row.symbol), _utc_midnight(row.date))].append(row)

    duplicate_groups = 0
    rows_to_delete = 0
    rows_deleted = 0
    rows_normalized = 0

    for (symbol, target_dt), bucket in grouped.items():
        if len(bucket) > 1:
            duplicate_groups += 1
            rows_to_delete += len(bucket) - 1

        # Prefer valid finite positive close, then latest raw timestamp, then highest id.
        canonical = max(
            bucket,
            key=lambda r: (
                1 if _is_valid_close(r.close) else 0,
                r.date.replace(tzinfo=None) if r.date.tzinfo is not None else r.date,
                int(r.id or 0),
            ),
        )
        needs_norm = canonical.date != target_dt
        if needs_norm:
            rows_normalized += 1

        if not dry_run:
            for extra in bucket:
                if extra is not canonical:
                    session.delete(extra)
                    rows_deleted += 1
            if needs_norm:
                canonical.date = target_dt

    if not dry_run and (rows_deleted > 0 or rows_normalized > 0):
        session.flush()
        session.commit()

    return {
        "mode": "dry_run" if dry_run else "apply",
        "total_rows_scanned": len(rows),
        "symbols_scanned": len(symbols),
        "duplicate_groups": duplicate_groups,
        "rows_to_delete": rows_to_delete,
        "rows_deleted": rows_deleted if not dry_run else 0,
        "rows_normalized": rows_normalized,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit or deduplicate PriceBar rows by (symbol, UTC day)")
    parser.add_argument("--apply", action="store_true", help="Delete duplicate rows and normalize dates")
    args = parser.parse_args()

    init_db()
    with SessionLocal() as session:
        summary = dedup_price_bars(session, dry_run=not args.apply)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
