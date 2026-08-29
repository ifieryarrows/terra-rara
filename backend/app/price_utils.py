"""Shared price validation used at every persistence and inference boundary."""

from __future__ import annotations

import math
from typing import Any, Optional

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
