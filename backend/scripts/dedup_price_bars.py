"""Deduplicate historical same-calendar-day PriceBar rows safely.

Defaults to dry-run mode (no writes). Pass ``--apply`` explicitly to delete
duplicate/weekend/invalid rows and normalize surviving timestamps to canonical
exchange-local midnight (UTC midnight representation).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from datetime import datetime, timezone

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db import SessionLocal, init_db
from app.models import PriceBar
from app.price_utils import (
    exchange_local_date,
    finite_positive_price,
    is_canonical_utc_midnight,
)


def _to_naive_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.min
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _row_preference_key(row: PriceBar) -> tuple:
    """Rank canonical candidate rows on the same exchange-local calendar day.

    Preference order:
    1. Valid finite positive ``close``
    2. Valid finite positive ``adj_close``
    3. Canonical ``00:00:00 UTC`` ingestion bar over legacy ``04:00/05:00/16:00`` bars
    4. Latest ``fetched_at`` timestamp
    5. Latest raw bar ``date`` timestamp
    6. Highest row ``id``
    """
    valid_close = 1 if finite_positive_price(row.close) is not None else 0
    valid_adj = 1 if finite_positive_price(getattr(row, "adj_close", None)) is not None else 0
    canonical_midnight = 1 if is_canonical_utc_midnight(row.date) else 0
    fetched = _to_naive_utc(getattr(row, "fetched_at", None))
    raw_dt = _to_naive_utc(row.date)
    return (
        valid_close,
        valid_adj,
        canonical_midnight,
        fetched,
        raw_dt,
        int(row.id or 0),
    )


def dedup_price_bars(session, *, dry_run: bool = True, drop_weekends: bool = True) -> dict:
    """Audit or remove duplicate ``PriceBar`` rows sharing (symbol, exchange-local date) and weekend ghost bars."""
    rows = session.query(PriceBar).order_by(PriceBar.symbol.asc(), PriceBar.date.asc(), PriceBar.id.asc()).all()
    grouped: dict[tuple[str, datetime], list[PriceBar]] = defaultdict(list)
    symbols: set[str] = set()
    invalid_date_rows: list[PriceBar] = []

    for row in rows:
        sym = str(row.symbol)
        symbols.add(sym)
        local_day = exchange_local_date(row.date, sym)
        if local_day is None:
            invalid_date_rows.append(row)
            continue
        grouped[(sym, datetime(local_day.year, local_day.month, local_day.day))].append(row)

    duplicate_groups = 0
    rows_to_delete = len(invalid_date_rows)
    rows_normalized = 0
    weekend_rows_to_delete = 0
    weekend_rows_deleted = 0
    invalid_close_rows_to_delete = 0

    to_delete: list[PriceBar] = list(invalid_date_rows)
    to_normalize: list[tuple[PriceBar, datetime]] = []

    for (_symbol, target_dt), bucket in grouped.items():
        if drop_weekends and target_dt.weekday() >= 5:
            weekend_rows_to_delete += len(bucket)
            rows_to_delete += len(bucket)
            to_delete.extend(bucket)
            continue

        valid_rows = [r for r in bucket if finite_positive_price(r.close) is not None]
        if not valid_rows:
            invalid_close_rows_to_delete += len(bucket)
            rows_to_delete += len(bucket)
            to_delete.extend(bucket)
            continue

        if len(bucket) > 1:
            duplicate_groups += 1
            rows_to_delete += len(bucket) - 1

        canonical = max(bucket, key=_row_preference_key)
        canonical_naive = _to_naive_utc(canonical.date)
        needs_norm = canonical_naive != target_dt
        if needs_norm:
            rows_normalized += 1
            norm_dt = (
                target_dt.replace(tzinfo=timezone.utc)
                if (canonical.date is not None and canonical.date.tzinfo is not None)
                else target_dt
            )
            to_normalize.append((canonical, norm_dt))

        for extra in bucket:
            if extra is not canonical:
                to_delete.append(extra)

    rows_deleted = 0
    if not dry_run and (to_delete or to_normalize):
        # Flush all deletes BEFORE normalizing canonical.date to prevent
        # Postgres UniqueViolation on uq_price_symbol_date.
        for row in to_delete:
            session.delete(row)
        rows_deleted = len(to_delete)
        weekend_rows_deleted = weekend_rows_to_delete
        if to_delete:
            session.flush()
        for canonical, norm_dt in to_normalize:
            canonical.date = norm_dt
        session.flush()
        session.commit()

    return {
        "mode": "dry_run" if dry_run else "apply",
        "drop_weekends": bool(drop_weekends),
        "total_rows_scanned": len(rows),
        "symbols_scanned": len(symbols),
        "duplicate_groups": duplicate_groups,
        "weekend_rows_to_delete": weekend_rows_to_delete,
        "weekend_rows_deleted": weekend_rows_deleted if not dry_run else 0,
        "invalid_rows_to_delete": len(invalid_date_rows) + invalid_close_rows_to_delete,
        "rows_to_delete": rows_to_delete,
        "rows_deleted": rows_deleted if not dry_run else 0,
        "rows_normalized": rows_normalized,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit or deduplicate PriceBar rows by (symbol, exchange-local day)")
    parser.add_argument("--apply", action="store_true", help="Delete duplicate/weekend rows and normalize dates")
    parser.add_argument("--keep-weekends", action="store_true", help="Keep Sat/Sun weekend bars instead of deleting them")
    args = parser.parse_args()

    init_db()
    with SessionLocal() as session:
        summary = dedup_price_bars(session, dry_run=not args.apply, drop_weekends=not args.keep_weekends)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
