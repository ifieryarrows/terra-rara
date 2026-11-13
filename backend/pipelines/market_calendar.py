"""Market-date assignment helpers for leakage-safe news features."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


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


def _cutoff_nanoseconds(cutoff_local: time) -> int:
    return (
        ((int(cutoff_local.hour) * 60 + int(cutoff_local.minute)) * 60 + int(cutoff_local.second))
        * 1_000_000_000
        + int(cutoff_local.microsecond) * 1_000
    )


def assign_market_date_and_after_close_series(
    published_at: Any,
    *,
    cutoff_local: time = CUT_OFF_LOCAL,
) -> tuple[pd.Series, pd.Series]:
    """
    Vectorized calculation of ``(market_date_ts, after_close_int)`` for a series of timestamps.

    Returns:
        - ``market_date_ts``: tz-naive normalized ``datetime64[ns]`` Series representing the market date.
        - ``after_close_int``: ``int64`` Series (1 if published after local cutoff or on weekend, else 0).
    """
    s = published_at if isinstance(published_at, pd.Series) else pd.Series(published_at)
    if s.empty:
        return (
            pd.Series(dtype="datetime64[ns]", index=s.index),
            pd.Series(dtype="int64", index=s.index),
        )

    pub_utc = pd.to_datetime(s, utc=True)
    local_dt = pub_utc.dt.tz_convert(MARKET_TZ)
    local_naive = local_dt.dt.tz_localize(None)
    local_day = local_naive.dt.floor("D")

    time_ns = (local_naive - local_day).dt.total_seconds() * 1_000_000_000
    cutoff_ns = _cutoff_nanoseconds(cutoff_local)
    after_cutoff = time_ns > cutoff_ns

    market_day = local_day + pd.to_timedelta(after_cutoff.astype(np.int64), unit="D")
    weekday = market_day.dt.weekday
    weekend_shift = np.where(weekday == 5, 2, np.where(weekday == 6, 1, 0))
    market_day = market_day + pd.to_timedelta(weekend_shift, unit="D")

    after_close = (after_cutoff | (local_dt.dt.weekday >= 5)).astype(np.int64)
    return market_day, after_close

