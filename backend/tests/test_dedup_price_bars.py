"""Offline SQLite and unit tests for exchange-local dedup and PriceBar validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import PriceBar
from app.price_utils import exchange_local_date, latest_finite_price_bar
from scripts.dedup_price_bars import dedup_price_bars


def _sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)()


def test_exchange_local_date_handles_global_exchanges():
    # Canonical 00:00 UTC bars keep their exact calendar day across exchanges
    assert exchange_local_date("2026-11-16T00:00:00Z", "HG=F").isoformat() == "2026-11-16"
    assert exchange_local_date("2026-11-16T00:00:00Z", "2899.HK").isoformat() == "2026-11-16"

    # Legacy 16:00 UTC HK bars are 00:00 HKT next day (Sun 16:00 UTC -> Mon HKT!)
    assert exchange_local_date("2026-11-15T16:00:00Z", "2899.HK").isoformat() == "2026-11-16"

    # Legacy 04:00 (EDT) / 05:00 (EST) UTC US/TSX bars map to same NY/Toronto day; 23:00 UTC Sun stays Sun
    assert exchange_local_date("2026-11-16T05:00:00Z", "HG=F").isoformat() == "2026-11-16"
    assert exchange_local_date("2026-07-16T04:00:00Z", "HBM.TO").isoformat() == "2026-07-16"
    assert exchange_local_date("2026-11-16T04:00:00Z", "HBM.TO").isoformat() == "2026-11-15"
    assert exchange_local_date("2026-11-15T23:00:00Z", "HG=F").isoformat() == "2026-11-15"

    # London BST 23:00 UTC maps to 00:00 BST next day
    assert exchange_local_date("2026-07-01T23:00:00Z", "ANTO.L").isoformat() == "2026-07-02"


def test_dedup_preserves_2899_hk_monday_1600_utc_and_prefers_0000_utc():
    engine, session = _sqlite_session()
    try:
        session.add_all(
            [
                # Collision on 2026-11-16: Sun 16:00 UTC (Mon HKT) vs Mon 00:00 UTC
                PriceBar(
                    symbol="2899.HK",
                    date=datetime(2026, 11, 15, 16, 0, tzinfo=timezone.utc),
                    close=25.50,
                    adj_close=25.50,
                    fetched_at=datetime(2026, 11, 19, 7, 0, tzinfo=timezone.utc),
                ),
                PriceBar(
                    symbol="2899.HK",
                    date=datetime(2026, 11, 16, 0, 0, tzinfo=timezone.utc),
                    close=25.98,
                    adj_close=25.98,
                    fetched_at=datetime(2026, 11, 21, 6, 0, tzinfo=timezone.utc),
                ),
                # Standalone HK bar stored only as 16:00 UTC (Thu 16:00 UTC == Fri 2026-11-13 HKT)
                PriceBar(
                    symbol="2899.HK",
                    date=datetime(2026, 11, 12, 16, 0, tzinfo=timezone.utc),
                    close=26.88,
                    adj_close=26.88,
                    fetched_at=datetime(2026, 11, 19, 7, 0, tzinfo=timezone.utc),
                ),
            ]
        )
        session.commit()

        dry = dedup_price_bars(session, dry_run=True, drop_weekends=True)
        assert dry["weekend_rows_to_delete"] == 0
        assert dry["duplicate_groups"] == 1
        assert dry["rows_to_delete"] == 1
        assert dry["rows_normalized"] == 1

        applied = dedup_price_bars(session, dry_run=False, drop_weekends=True)
        assert applied["weekend_rows_deleted"] == 0
        assert applied["rows_deleted"] == 1
        remaining = session.query(PriceBar).order_by(PriceBar.date.asc()).all()
        assert [(r.date.date().isoformat(), r.close) for r in remaining] == [
            ("2026-11-13", 26.88),
            ("2026-11-16", 25.98),
        ]
    finally:
        session.close()
        engine.dispose()


def test_dedup_prefers_0000_over_0400_0500_and_flushes_deletes_before_update():
    engine, session = _sqlite_session()
    try:
        session.add_all(
            [
                # Case 1: BHP 00:00 UTC must beat 05:00 UTC even when 05:00 has later date hour
                PriceBar(
                    symbol="BHP",
                    date=datetime(2024, 1, 22, 0, 0, tzinfo=timezone.utc),
                    close=61.50,
                    adj_close=54.90,
                    fetched_at=datetime(2026, 11, 19, 23, 0, tzinfo=timezone.utc),
                ),
                PriceBar(
                    symbol="BHP",
                    date=datetime(2024, 1, 22, 5, 0, tzinfo=timezone.utc),
                    close=57.20,
                    adj_close=57.20,
                    fetched_at=datetime(2026, 1, 15, 23, 0, tzinfo=timezone.utc),
                ),
                # Case 2: SCCO 00:00 UTC is invalid (SQLite NOT NULL forbids NaN, so use <= 0;
                # Postgres NaN is handled identically via finite_positive_price) while 04:00 UTC is valid ->
                # 00:00 must be deleted & flushed BEFORE 04:00 row's date is updated to 00:00
                PriceBar(
                    symbol="SCCO",
                    date=datetime(2026, 8, 11, 0, 0, tzinfo=timezone.utc),
                    close=-1.0,
                    adj_close=None,
                    fetched_at=datetime(2026, 11, 19, 23, 0, tzinfo=timezone.utc),
                ),
                PriceBar(
                    symbol="SCCO",
                    date=datetime(2026, 8, 11, 4, 0, tzinfo=timezone.utc),
                    close=118.4,
                    adj_close=118.4,
                    fetched_at=datetime(2026, 8, 11, 23, 0, tzinfo=timezone.utc),
                ),
                # Case 3: Genuine Sunday futures ghost bar must be deleted
                PriceBar(
                    symbol="HG=F",
                    date=datetime(2026, 11, 15, 5, 0, tzinfo=timezone.utc),
                    close=5.95,
                    adj_close=5.95,
                ),
            ]
        )
        session.commit()

        applied = dedup_price_bars(session, dry_run=False, drop_weekends=True)
        assert applied["weekend_rows_deleted"] == 1
        assert applied["rows_deleted"] == 3

        bhp = session.query(PriceBar).filter(PriceBar.symbol == "BHP").one()
        assert bhp.close == pytest.approx(61.50)
        assert bhp.adj_close == pytest.approx(54.90)
        assert bhp.date.hour == 0

        scco = session.query(PriceBar).filter(PriceBar.symbol == "SCCO").one()
        assert scco.close == pytest.approx(118.4)
        assert scco.date.date().isoformat() == "2026-08-11"
        assert scco.date.hour == 0

        assert session.query(PriceBar).filter(PriceBar.symbol == "HG=F").count() == 0
    finally:
        session.close()
        engine.dispose()


def test_latest_finite_price_bar_filters_weekends_and_nan_and_prefers_canonical():
    engine, session = _sqlite_session()
    try:
        session.add_all(
            [
                # Friday valid canonical 00:00 and stale 05:00
                PriceBar(
                    symbol="HG=F",
                    date=datetime(2026, 11, 13, 0, 0, tzinfo=timezone.utc),
                    close=6.08,
                    adj_close=6.08,
                    fetched_at=datetime(2026, 11, 19, 23, 0, tzinfo=timezone.utc),
                ),
                PriceBar(
                    symbol="HG=F",
                    date=datetime(2026, 11, 13, 5, 0, tzinfo=timezone.utc),
                    close=6.01,
                    adj_close=6.01,
                    fetched_at=datetime(2026, 11, 13, 23, 0, tzinfo=timezone.utc),
                ),
                # Sunday ghost bar with newer timestamp (must be ignored)
                PriceBar(
                    symbol="HG=F",
                    date=datetime(2026, 11, 15, 5, 0, tzinfo=timezone.utc),
                    close=6.12,
                    adj_close=6.12,
                ),
                # Monday non-positive bar (must be ignored just like Postgres NaN)
                PriceBar(
                    symbol="HG=F",
                    date=datetime(2026, 11, 16, 0, 0, tzinfo=timezone.utc),
                    close=0.0,
                    adj_close=None,
                ),
            ]
        )
        session.commit()

        latest = latest_finite_price_bar(session, "HG=F")
        assert latest is not None
        assert latest.date.date().isoformat() == "2026-11-13"
        assert latest.close == pytest.approx(6.08)
    finally:
        session.close()
        engine.dispose()
