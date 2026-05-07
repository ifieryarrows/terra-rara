"""Leakage-safe market-date sentiment aggregation for TFT features."""

from __future__ import annotations

import pandas as pd

from app.models import NewsProcessed, NewsRaw, NewsSentimentV2
from pipelines.market_calendar import assign_market_date, is_after_close_news


MATERIAL_RELEVANCE_MIN = 0.60
MATERIAL_CONFIDENCE_MIN = 0.55


def build_market_date_sentiment_frame(session, start_date, end_date) -> pd.DataFrame:
    """Return daily sentiment indexed by market date, not publication date."""
    columns = [
        "sentiment_index",
        "news_count",
        "material_news_count",
        "after_close_news_count",
        "days_since_last_material_news",
        "stale_sentiment_flag",
    ]
    if not hasattr(session, "query"):
        return pd.DataFrame(columns=columns)

    rows = (
        session.query(
            NewsRaw.published_at,
            NewsRaw.fetched_at,
            NewsSentimentV2.final_score,
            NewsSentimentV2.confidence_calibrated,
            NewsSentimentV2.relevance_score,
            NewsSentimentV2.event_type,
        )
        .join(NewsProcessed, NewsProcessed.raw_id == NewsRaw.id)
        .join(NewsSentimentV2, NewsSentimentV2.news_processed_id == NewsProcessed.id)
        .filter(NewsRaw.published_at >= start_date, NewsRaw.published_at <= end_date)
        .all()
    )

    if not rows:
        return pd.DataFrame(columns=columns)

    records = []
    for r in rows:
        market_date = assign_market_date(r.published_at)
        relevance = float(r.relevance_score or 0.0)
        confidence = float(r.confidence_calibrated or 0.0)
        material = relevance >= MATERIAL_RELEVANCE_MIN and confidence >= MATERIAL_CONFIDENCE_MIN
        weight = max(relevance, 0.0) * max(confidence, 0.0)
        records.append(
            {
                "market_date": market_date,
                "score": float(r.final_score or 0.0),
                "weight": weight,
                "material": int(material),
                "after_close": int(is_after_close_news(r.published_at)),
            }
        )

    raw = pd.DataFrame(records)

    def _weighted_sentiment(g: pd.DataFrame) -> float:
        denom = g["weight"].sum()
        if denom <= 1e-9:
            return float(g["score"].mean())
        return float((g["score"] * g["weight"]).sum() / denom)

    daily = raw.groupby("market_date").apply(
        lambda g: pd.Series(
            {
                "sentiment_index": _weighted_sentiment(g),
                "news_count": float(len(g)),
                "material_news_count": float(g["material"].sum()),
                "after_close_news_count": float(g["after_close"].sum()),
            }
        )
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    days_since = []
    last_material = None
    for d in daily.index:
        if float(daily.loc[d, "material_news_count"]) > 0:
            last_material = d
            days_since.append(0)
        elif last_material is None:
            days_since.append(999)
        else:
            days_since.append(int((d - last_material).days))

    daily["days_since_last_material_news"] = days_since
    daily["stale_sentiment_flag"] = (daily["days_since_last_material_news"] > 3).astype(float)
    return daily[columns].astype("float32")


def build_market_date_event_counts_from_db(session, start_date, end_date) -> pd.DataFrame:
    """Return event_type count matrix indexed by leakage-safe market date."""
    if not hasattr(session, "query"):
        return pd.DataFrame()

    rows = (
        session.query(
            NewsRaw.published_at,
            NewsSentimentV2.event_type,
        )
        .join(NewsProcessed, NewsSentimentV2.news_processed_id == NewsProcessed.id)
        .join(NewsRaw, NewsProcessed.raw_id == NewsRaw.id)
        .filter(NewsRaw.published_at >= start_date, NewsRaw.published_at <= end_date)
        .all()
    )
    if not rows:
        return pd.DataFrame()

    records = [
        {
            "market_date": assign_market_date(r.published_at),
            "event_type": r.event_type,
            "count": 1,
        }
        for r in rows
    ]
    df = pd.DataFrame(records)
    pivot = df.pivot_table(index="market_date", columns="event_type", values="count", aggfunc="sum", fill_value=0)
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.sort_index()
