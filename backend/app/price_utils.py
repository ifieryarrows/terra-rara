"""Shared price validation used at every persistence and inference boundary."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import logging
import math
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import PriceBar

logger = logging.getLogger(__name__)

_UTC_TZ = timezone.utc
_EXCHANGE_TZ_CACHE: dict[str, ZoneInfo] = {}
_SUFFIX_TIMEZONES: tuple[tuple[str, str], ...] = (
    (".HK", "Asia/Hong_Kong"),
    (".L", "Europe/London"),
    (".TO", "America/Toronto"),
    (".V", "America/Toronto"),
    (".CN", "America/Toronto"),
    (".NE", "America/Toronto"),
    (".AX", "Australia/Sydney"),
    (".SS", "Asia/Shanghai"),
    (".SZ", "Asia/Shanghai"),
    (".KS", "Asia/Seoul"),
    (".KQ", "Asia/Seoul"),
    (".T", "Asia/Tokyo"),
    (".PA", "Europe/Paris"),
    (".DE", "Europe/Berlin"),
    (".F", "Europe/Berlin"),
    (".AS", "Europe/Amsterdam"),
    (".BR", "Europe/Brussels"),
    (".MI", "Europe/Rome"),
    (".MC", "Europe/Madrid"),
)


def _zoneinfo(tz_name: str) -> ZoneInfo:
    cached = _EXCHANGE_TZ_CACHE.get(tz_name)
    if cached is None:
        cached = ZoneInfo(tz_name)
        _EXCHANGE_TZ_CACHE[tz_name] = cached
    return cached


def exchange_timezone(symbol: Optional[str] = None) -> ZoneInfo:
    """Return the primary exchange ZoneInfo for a ticker symbol."""
    upper = str(symbol or "").strip().upper()
    for suffix, tz_name in _SUFFIX_TIMEZONES:
        if upper.endswith(suffix):
            return _zoneinfo(tz_name)
    return _zoneinfo("America/New_York")


def _parse_timestamp_value(ts: Any) -> Optional[datetime]:
    """Parse datetime-like values without losing explicit non-UTC offsets."""
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, date):
        return datetime(ts.year, ts.month, ts.day)
    if isinstance(ts, str):
        raw = ts.strip()
        if not raw:
            return None
        iso_candidate = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
        try:
            return datetime.fromisoformat(iso_candidate)
        except ValueError:
            parsed = pd.to_datetime(raw, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime()
    parsed = pd.to_datetime(ts, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def is_canonical_utc_midnight(ts: Any) -> bool:
    """Return True when ``ts`` is exact 00:00:00 in UTC or tz-naive."""
    dt = _parse_timestamp_value(ts)
    if dt is None:
        return False
    offset = dt.utcoffset() if dt.tzinfo is not None else None
    if offset is not None and offset != timedelta(0):
        return False
    return dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0


def exchange_local_date(ts: Any, symbol: Optional[str] = None) -> Optional[date]:
    """Resolve the exchange-local trading calendar date for a bar timestamp.

    Canonical ``00:00:00`` UTC (or naive) bars already represent normalized
    calendar midnights from ingestion and stay on that exact calendar date.
    Explicit non-UTC timestamps preserve their local offset date. Non-midnight
    UTC timestamps (e.g. legacy ``04:00/05:00 UTC`` US bars, ``16:00 UTC`` HK
    bars, ``23:00 UTC`` London BST or Sunday US futures bars) are converted to
    the symbol's exchange timezone before extracting ``.date()``.
    """
    if isinstance(ts, date) and not isinstance(ts, datetime):
        return ts
    dt = _parse_timestamp_value(ts)
    if dt is None:
        return None

    offset = dt.utcoffset() if dt.tzinfo is not None else None
    if offset is not None and offset != timedelta(0):
        return dt.date()

    if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.date()

    utc_dt = dt.replace(tzinfo=_UTC_TZ) if dt.tzinfo is None else dt.astimezone(_UTC_TZ)
    return utc_dt.astimezone(exchange_timezone(symbol)).date()


def is_weekend_bar(ts: Any, symbol: Optional[str] = None) -> bool:
    """Return True if ``ts`` resolves to Sat/Sun on the symbol's exchange calendar."""
    local_day = exchange_local_date(ts, symbol)
    return local_day is None or local_day.weekday() >= 5


def sanitize_close_spikes(
    close_series: pd.Series,
    symbol: str = "target",
    *,
    max_abs_log_jump: float = 0.35,
    confirm_bars: int = 2,
    extreme_log_jump: float = 1.0,
    extreme_confirm_bars: int = 3,
    ffill_invalid: bool = True,
) -> pd.Series:
    """Replace transient corrupt price spikes causally without flatlining persistent shifts."""
    close = close_series.astype(float).where(lambda s: np.isfinite(s) & (s > 0.0))
    if ffill_invalid:
        close = close.ffill()
    if len(close) <= 1:
        return close

    close_vals = close.to_numpy(dtype=float, copy=False)
    valid = np.isfinite(close_vals) & (close_vals > 0.0)
    log_vals = np.zeros_like(close_vals)
    log_vals[valid] = np.log(close_vals[valid])
    has_jumps = bool((valid[1:] & valid[:-1] & (np.abs(np.diff(log_vals)) > max_abs_log_jump)).any())
    if not has_jumps:
        return close

    close_vals = close_vals.copy()
    repaired = 0
    pending_anchor: Optional[float] = None
    pending_run = 0
    last_valid_idx: Optional[int] = None

    for idx in range(len(close_vals)):
        cur_val = close_vals[idx]
        if not (np.isfinite(cur_val) and cur_val > 0.0):
            pending_anchor = None
            pending_run = 0
            continue
        if last_valid_idx is None:
            last_valid_idx = idx
            continue
        prev_val = close_vals[last_valid_idx]
        log_jump = abs(np.log(cur_val / prev_val))
        if log_jump > max_abs_log_jump:
            needed = extreme_confirm_bars if log_jump > extreme_log_jump else confirm_bars
            if (
                pending_anchor is not None
                and abs(np.log(cur_val / pending_anchor)) <= max_abs_log_jump
            ):
                pending_run += 1
                pending_anchor = cur_val
            else:
                pending_anchor = cur_val
                pending_run = 1
            if pending_run >= needed:
                pending_anchor = None
                pending_run = 0
                last_valid_idx = idx
            else:
                close_vals[idx] = prev_val
                repaired += 1
                last_valid_idx = idx
        else:
            pending_anchor = None
            pending_run = 0
            last_valid_idx = idx

    if repaired:
        logger.warning(
            "Repaired %d corrupt single-day close spike(s) (> %.2f log jump) for %s",
            repaired,
            max_abs_log_jump,
            symbol,
        )
    return pd.Series(close_vals, index=close_series.index, name=close_series.name)


def finite_positive_price(value: Any) -> Optional[float]:
    """Return a finite, positive float or ``None`` for unusable price data."""
    try:
        price = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(price) or price <= 0.0:
        return None
    return price


def finite_number(value: Any) -> Optional[float]:
    """Return a finite float, allowing zero/negative values for non-price fields."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def price_from_simple_return(baseline_price: Any, predicted_return: Any) -> float:
    """Apply the XGBoost next-day simple-return contract exactly once."""
    baseline = finite_positive_price(baseline_price)
    simple_return = finite_number(predicted_return)
    if baseline is None:
        raise ValueError("A finite positive baseline price is required")
    if simple_return is None:
        raise ValueError("A finite predicted return is required")
    predicted = baseline * (1.0 + simple_return)
    if finite_positive_price(predicted) is None:
        raise ValueError("Simple-return conversion produced an invalid price")
    return predicted


def latest_finite_price_bar(session: Session, symbol: str) -> Optional[PriceBar]:
    """Find the newest usable weekday close without trusting database NaN semantics."""
    # PostgreSQL treats NaN as a comparable numeric value, and SQLite may store
    # it as NULL. A bounded Python validation works consistently across both.
    rows = (
        session.query(PriceBar)
        .filter(PriceBar.symbol == symbol, PriceBar.close.isnot(None))
        .order_by(PriceBar.date.desc())
        .limit(64)
        .all()
    )
    candidates: list[tuple[date, PriceBar]] = []
    for row in rows:
        if finite_positive_price(row.close) is None:
            continue
        local_day = exchange_local_date(row.date, symbol)
        if local_day is None or local_day.weekday() >= 5:
            continue
        candidates.append((local_day, row))
    if not candidates:
        return None
    newest_day = max(day for day, _ in candidates)
    day_rows = [row for day, row in candidates if day == newest_day]

    def _pref_key(row: PriceBar) -> tuple:
        has_adj = 1 if finite_positive_price(getattr(row, "adj_close", None)) is not None else 0
        utc_mid = 1 if is_canonical_utc_midnight(row.date) else 0
        fetched = getattr(row, "fetched_at", None)
        if fetched is not None and getattr(fetched, "tzinfo", None) is not None:
            fetched = fetched.astimezone(_UTC_TZ).replace(tzinfo=None)
        raw_dt = getattr(row, "date", None)
        if raw_dt is not None and getattr(raw_dt, "tzinfo", None) is not None:
            raw_dt = raw_dt.astimezone(_UTC_TZ).replace(tzinfo=None)
        return (
            has_adj,
            utc_mid,
            fetched or datetime.min,
            raw_dt or datetime.min,
            int(getattr(row, "id", 0) or 0),
        )

    return max(day_rows, key=_pref_key)


def latest_finite_price_dates(session: Session, symbols: list[str]) -> dict[str, datetime]:
    """Retrieve the newest usable price bar dates across multiple symbols in a single query.
    
    Uses a partitioned window function to find the most recent bars per symbol,
    then applies Python-level finite price validation to match the NaN/NULL safety
    of ``latest_finite_price_bar``.
    """
    return _boundary_finite_price_dates(session, symbols, descending=True)


def earliest_finite_price_dates(session: Session, symbols: list[str]) -> dict[str, datetime]:
    """Retrieve the oldest usable price bar dates across multiple symbols in a single query."""
    return _boundary_finite_price_dates(session, symbols, descending=False)


def _boundary_finite_price_dates(
    session: Session,
    symbols: list[str],
    *,
    descending: bool,
) -> dict[str, datetime]:
    """Retrieve boundary (newest or oldest) usable weekday price bar dates per symbol."""
    if not symbols:
        return {}
    try:
        order_col = PriceBar.date.desc() if descending else PriceBar.date.asc()
        subq = (
            session.query(
                PriceBar.symbol,
                PriceBar.date,
                PriceBar.close,
                func.row_number().over(
                    partition_by=PriceBar.symbol,
                    order_by=order_col,
                ).label("rn"),
            )
            .filter(PriceBar.symbol.in_(symbols), PriceBar.close.isnot(None))
            .subquery()
        )
        rows = (
            session.query(subq.c.symbol, subq.c.date, subq.c.close)
            .filter(subq.c.rn <= 20)
            .order_by(subq.c.symbol, subq.c.rn)
            .all()
        )
        boundary_dates: dict[str, datetime] = {}
        for sym, dt, close in rows:
            if sym in boundary_dates:
                continue
            if finite_positive_price(close) is not None and not is_weekend_bar(dt, sym):
                boundary_dates[sym] = dt
        return boundary_dates
    except Exception:
        # Fallback if window function or subquery is unsupported on unusual engine
        return {}

