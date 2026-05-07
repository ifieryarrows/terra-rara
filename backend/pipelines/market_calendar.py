"""Market-date assignment helpers for leakage-safe news features."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


MARKET_TZ = ZoneInfo("America/New_York")
CUT_OFF_LOCAL = time(17, 30)


def assign_market_date(
    published_at_utc: datetime,
    *,
    cutoff_local: time = CUT_OFF_LOCAL,
) -> date:
    """Return the trading date on which an article is safely available."""
    if published_at_utc.tzinfo is None:
        published_at_utc = published_at_utc.replace(tzinfo=timezone.utc)

    local = published_at_utc.astimezone(MARKET_TZ)
    market_day = local.date()

    if local.time() > cutoff_local:
        market_day = market_day + timedelta(days=1)

    while market_day.weekday() >= 5:
        market_day = market_day + timedelta(days=1)

    return market_day


def is_after_close_news(
    published_at_utc: datetime,
    *,
    cutoff_local: time = CUT_OFF_LOCAL,
) -> bool:
    """Return True when the article should not affect the same market date."""
    if published_at_utc.tzinfo is None:
        published_at_utc = published_at_utc.replace(tzinfo=timezone.utc)
    local = published_at_utc.astimezone(MARKET_TZ)
    return local.time() > cutoff_local or local.weekday() >= 5
