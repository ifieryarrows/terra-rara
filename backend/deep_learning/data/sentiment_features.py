"""
Advanced Sentiment Feature Engineering.

Computes second-order derivatives of sentiment signals that capture
market-moving dynamics the raw sentiment_index cannot:

- Sentiment Momentum (SMA/EMA rate-of-change)
- Sentiment Surprise (Z-score anomaly detection)
- Volume-Weighted Sentiment
- Event-Type Intensity (supply_disruption counts, etc.)
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from deep_learning.config import SentimentFeatureConfig, get_tft_config

logger = logging.getLogger(__name__)


def compute_sentiment_momentum(
    sentiment_index: pd.Series,
    windows: Sequence[int] = (5, 10, 30),
) -> pd.DataFrame:
    """
    Sentiment momentum: how fast and in which direction market mood is shifting.

    For each window *w* the feature set contains:
    - sent_momentum_{w}d  : current index minus SMA(w)
    - sent_ema_{w}d       : EMA(w) of the index (smoothed trend)
    - sent_roc_{w}d       : rate-of-change over w days
    """
    features = pd.DataFrame(index=sentiment_index.index)

    for w in windows:
        sma = sentiment_index.rolling(window=w, min_periods=1).mean()
        features[f"sent_momentum_{w}d"] = sentiment_index - sma
        features[f"sent_ema_{w}d"] = sentiment_index.ewm(span=w, adjust=False).mean()
        features[f"sent_roc_{w}d"] = sentiment_index.diff(w)

    return features


def compute_sentiment_surprise(
    sentiment_index: pd.Series,
    lookback: int = 30,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Sentiment Surprise: Z-score of today's sentiment relative to recent history.

    When |z| >= threshold the market received an "unexpected" signal that
    historically triggers outsized price moves.

    Features:
    - sent_surprise_z        : rolling Z-score
    - sent_surprise_flag     : binary flag (|z| >= threshold)
    - sent_surprise_signed   : z * sign (directional surprise magnitude)
    """
    roll_mean = sentiment_index.rolling(window=lookback, min_periods=5).mean()
    roll_std = sentiment_index.rolling(window=lookback, min_periods=5).std()
    roll_std = roll_std.replace(0, np.nan)

    z_score = (sentiment_index - roll_mean) / roll_std

    features = pd.DataFrame(index=sentiment_index.index)
    features["sent_surprise_z"] = z_score
    features["sent_surprise_flag"] = (z_score.abs() >= threshold).astype(np.float32)
    features["sent_surprise_signed"] = z_score * np.sign(sentiment_index)

    return features


def compute_volume_weighted_sentiment(
    sentiment_index: pd.Series,
    news_count: pd.Series,
) -> pd.DataFrame:
    """
    Weight sentiment by news volume: high-volume days carry stronger signal.

    Features:
    - sent_vol_weighted     : sentiment * log(1 + news_count)
    - sent_vol_zscore       : Z-score of volume-weighted series (30-day)
    - news_count_zscore     : Z-score of news volume itself
    """
    log_count = np.log1p(news_count.fillna(0))

    vol_weighted = sentiment_index * log_count
    vol_roll_mean = vol_weighted.rolling(30, min_periods=5).mean()
    vol_roll_std = vol_weighted.rolling(30, min_periods=5).std().replace(0, np.nan)

    nc_roll_mean = news_count.rolling(30, min_periods=5).mean()
    nc_roll_std = news_count.rolling(30, min_periods=5).std().replace(0, np.nan)

    features = pd.DataFrame(index=sentiment_index.index)
    features["sent_vol_weighted"] = vol_weighted
    features["sent_vol_zscore"] = (vol_weighted - vol_roll_mean) / vol_roll_std
    features["news_count_zscore"] = (news_count - nc_roll_mean) / nc_roll_std

    return features


def compute_event_type_intensity(
    event_counts: pd.DataFrame,
    event_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Daily counts per event-type category from NewsSentimentV2.

    *event_counts* is expected to have date index and one column per event_type
    with daily occurrence counts.

    Features (per event type):
    - evt_{type}_count     : raw daily count
    - evt_{type}_ma5       : 5-day moving average
    - evt_{type}_spike     : flag when count > 2 * MA5
    """
    if event_types is None:
        cfg = get_tft_config()
        event_types = list(cfg.sentiment.event_types)

    features = pd.DataFrame(index=event_counts.index)

    for etype in event_types:
        col = etype if etype in event_counts.columns else None
        if col is None:
            features[f"evt_{etype}_count"] = 0.0
            features[f"evt_{etype}_ma5"] = 0.0
            features[f"evt_{etype}_spike"] = 0.0
            continue

        counts = event_counts[col].fillna(0).astype(float)
        ma5 = counts.rolling(5, min_periods=1).mean()

        features[f"evt_{etype}_count"] = counts
        features[f"evt_{etype}_ma5"] = ma5
        features[f"evt_{etype}_spike"] = (counts > 2.0 * ma5.clip(lower=0.5)).astype(np.float32)

    return features


def build_event_counts_from_db(
    session,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Query NewsSentimentV2 to build a (date x event_type) count matrix.
    """
    from sqlalchemy import func as sa_func
    from app.models import NewsSentimentV2, NewsProcessed, NewsRaw

    rows = (
        session.query(
            sa_func.date(NewsRaw.published_at).label("date"),
            NewsSentimentV2.event_type,
            sa_func.count(NewsSentimentV2.id).label("cnt"),
        )
        .join(NewsProcessed, NewsSentimentV2.news_processed_id == NewsProcessed.id)
        .join(NewsRaw, NewsProcessed.raw_id == NewsRaw.id)
        .filter(NewsRaw.published_at >= start_date, NewsRaw.published_at <= end_date)
        .group_by(sa_func.date(NewsRaw.published_at), NewsSentimentV2.event_type)
        .all()
    )

    if not rows:
        return pd.DataFrame()

    records = [{"date": r.date, "event_type": r.event_type, "count": r.cnt} for r in rows]
    df = pd.DataFrame(records)
    pivot = df.pivot_table(index="date", columns="event_type", values="count", fill_value=0)
    pivot.index = pd.to_datetime(pivot.index)
    return pivot


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------

def build_all_sentiment_features(
    daily_sentiment: pd.DataFrame,
    event_counts: Optional[pd.DataFrame] = None,
    cfg: Optional[SentimentFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Build the complete sentiment feature set from daily_sentiment DataFrame.

    Expected columns in *daily_sentiment*:
        - sentiment_index (float)
        - news_count (int)
    """
    if cfg is None:
        cfg = get_tft_config().sentiment

    si = daily_sentiment["sentiment_index"]
    nc = daily_sentiment["news_count"]

    parts: list[pd.DataFrame] = [
        compute_sentiment_momentum(si, windows=cfg.momentum_windows),
        compute_sentiment_surprise(si, lookback=cfg.surprise_lookback, threshold=cfg.surprise_threshold),
        compute_volume_weighted_sentiment(si, nc),
    ]

    if event_counts is not None and not event_counts.empty:
        evt_aligned = event_counts.reindex(daily_sentiment.index).fillna(0)
        parts.append(compute_event_type_intensity(evt_aligned, event_types=cfg.event_types))

    combined = pd.concat(parts, axis=1)
    return combined
