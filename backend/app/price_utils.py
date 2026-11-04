"""Shared price validation used at every persistence and inference boundary."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import PriceBar


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
    """Find the newest usable close without trusting database NaN semantics."""
    # PostgreSQL treats NaN as a comparable numeric value, and SQLite may store
    # it as NULL. A bounded Python validation works consistently across both.
    rows = (
        session.query(PriceBar)
        .filter(PriceBar.symbol == symbol, PriceBar.close.isnot(None))
        .order_by(PriceBar.date.desc())
        .limit(64)
        .all()
    )
    return next((row for row in rows if finite_positive_price(row.close) is not None), None)


def latest_finite_price_dates(session: Session, symbols: list[str]) -> dict[str, datetime]:
    """Retrieve the newest usable price bar dates across multiple symbols in a single query.
    
    Uses a partitioned window function to find the most recent bars per symbol,
    then applies Python-level finite price validation to match the NaN/NULL safety
    of ``latest_finite_price_bar``.
    """
    if not symbols:
        return {}
    try:
        subq = (
            session.query(
                PriceBar.symbol,
                PriceBar.date,
                PriceBar.close,
                func.row_number().over(
                    partition_by=PriceBar.symbol,
                    order_by=PriceBar.date.desc(),
                ).label("rn"),
            )
            .filter(PriceBar.symbol.in_(symbols), PriceBar.close.isnot(None))
            .subquery()
        )
        rows = (
            session.query(subq.c.symbol, subq.c.date, subq.c.close)
            .filter(subq.c.rn <= 5)
            .order_by(subq.c.symbol, subq.c.rn)
            .all()
        )
        latest_dates: dict[str, datetime] = {}
        for sym, dt, close in rows:
            if sym in latest_dates:
                continue
            if finite_positive_price(close) is not None:
                latest_dates[sym] = dt
        return latest_dates
    except Exception:
        # Fallback if window function or subquery is unsupported on unusual engine
        return {}

