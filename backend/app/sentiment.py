"""Sentiment classification — one authoritative balance band.

The dashboard renders the sentiment *label* (Bullish / Neutral / Bearish) next
to the raw index. Every consumer that needs a directional read must use that
classification, never the sign of the raw number: an index inside the balance
band carries no directional information, so reading its sign would invent a
market view out of noise.

This module is the single source of truth for the band. The API layer
(`/api/sentiment/summary`), the inference layer and the commentary/companion
layer all classify through it, so the number rendered next to the label and the
read the companion acts on can never disagree.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = [
    "SENTIMENT_BALANCE_BAND",
    "BULLISH",
    "NEUTRAL",
    "BEARISH",
    "classify_sentiment",
    "sentiment_direction",
    "is_balanced",
]

#: An index with |value| <= SENTIMENT_BALANCE_BAND is "Neutral" / balanced.
SENTIMENT_BALANCE_BAND = 0.10

BULLISH = "Bullish"
NEUTRAL = "Neutral"
BEARISH = "Bearish"


def _finite(value: Optional[float]) -> Optional[float]:
    """Return the value as a finite float, or None when it is not usable."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def classify_sentiment(sentiment_index: Optional[float]) -> str:
    """Classify a sentiment index as Bullish / Neutral / Bearish.

    A missing or unusable index is Neutral: no readable tone is no tone.
    """
    number = _finite(sentiment_index)
    if number is None:
        return NEUTRAL
    if number > SENTIMENT_BALANCE_BAND:
        return BULLISH
    if number < -SENTIMENT_BALANCE_BAND:
        return BEARISH
    return NEUTRAL


def sentiment_direction(sentiment_index: Optional[float]) -> int:
    """Directional read of a sentiment index: +1 bullish, 0 balanced, -1 bearish."""
    label = classify_sentiment(sentiment_index)
    if label == BULLISH:
        return 1
    if label == BEARISH:
        return -1
    return 0


def is_balanced(sentiment_index: Optional[float]) -> bool:
    """True when the index sits inside the balance band (no directional read)."""
    return sentiment_direction(sentiment_index) == 0
