"""Leakage-safe market-date sentiment aggregation for TFT features."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session as SASession

from app.models import NewsProcessed, NewsRaw, NewsSentimentV2
from pipelines.market_calendar import assign_market_date, is_after_close_news


MATERIAL_RELEVANCE_MIN = 0.60
MATERIAL_CONFIDENCE_MIN = 0.55


def _effective_available_at(row) -> object:
    return (
        getattr(row, "available_at", None)
        or getattr(row, "fetched_at", None)
        or getattr(row, "published_at", None)
    )


def _effective_market_date(row):
    """Map news to the later of publication market date and availability date."""
    published_market_date = assign_market_date(row.published_at)
    available_at = _effective_available_at(row)
    if available_at is None:
        return published_market_date
    available_market_date = assign_market_date(available_at)
    return max(published_market_date, available_market_date)


def _within_window(market_date, start_date, end_date) -> bool:
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    return start <= market_date <= end


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

    available_expr = func.coalesce(
        NewsSentimentV2.available_at,
        NewsRaw.fetched_at,
        NewsRaw.published_at,
    )
    query = (
        session.query(
            NewsRaw.published_at,
            NewsRaw.fetched_at,
            NewsSentimentV2.available_at,
            NewsSentimentV2.final_score,
            NewsSentimentV2.confidence_calibrated,
            NewsSentimentV2.relevance_score,
            NewsSentimentV2.event_type,
        )
        .join(NewsProcessed, NewsProcessed.raw_id == NewsRaw.id)
        .join(NewsSentimentV2, NewsSentimentV2.news_processed_id == NewsProcessed.id)
        .filter(
            NewsRaw.published_at <= end_date,
            available_expr <= end_date,
        )
    )
    if isinstance(session, SASession) and start_date is not None:
        lower_bound = pd.Timestamp(start_date).to_pydatetime() - timedelta(days=7)
        query = query.filter(available_expr >= lower_bound)

    rows = query.all()

    if not rows:
        return pd.DataFrame(columns=columns)

    records = []
    for r in rows:
        market_date = _effective_market_date(r)
        if not _within_window(market_date, start_date, end_date):
            continue
        relevance = float(r.relevance_score or 0.0)
        confidence = float(r.confidence_calibrated or 0.0)
        material = relevance >= MATERIAL_RELEVANCE_MIN and confidence >= MATERIAL_CONFIDENCE_MIN
        weight = max(relevance, 0.0) * max(confidence, 0.0)
        score = float(r.final_score or 0.0)
        records.append(
            {
                "market_date": market_date,
                "score": score,
                "weight": weight,
                "weighted_score": score * weight,
                "material": int(material),
                "after_close": int(is_after_close_news(r.published_at)),
            }
        )

    raw = pd.DataFrame(records)
    if raw.empty:
        return pd.DataFrame(columns=columns)

    grouped = raw.groupby("market_date", sort=True).agg(
        score_mean=("score", "mean"),
        weight_sum=("weight", "sum"),
        weighted_score_sum=("weighted_score", "sum"),
        news_count=("score", "count"),
        material_news_count=("material", "sum"),
        after_close_news_count=("after_close", "sum"),
    )
    sentiment_idx = np.where(
        grouped["weight_sum"] <= 1e-9,
        grouped["score_mean"],
        grouped["weighted_score_sum"] / grouped["weight_sum"],
    )
    daily = pd.DataFrame(
        {
            "sentiment_index": sentiment_idx.astype(float),
            "news_count": grouped["news_count"].astype(float),
            "material_news_count": grouped["material_news_count"].astype(float),
            "after_close_news_count": grouped["after_close_news_count"].astype(float),
        },
        index=pd.to_datetime(grouped.index),
    ).sort_index()

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

    available_expr = func.coalesce(
        NewsSentimentV2.available_at,
        NewsRaw.fetched_at,
        NewsRaw.published_at,
    )
    query = (
        session.query(
            NewsRaw.published_at,
            NewsRaw.fetched_at,
            NewsSentimentV2.available_at,
            NewsSentimentV2.event_type,
        )
        .join(NewsProcessed, NewsSentimentV2.news_processed_id == NewsProcessed.id)
        .join(NewsRaw, NewsProcessed.raw_id == NewsRaw.id)
        .filter(
            NewsRaw.published_at <= end_date,
            available_expr <= end_date,
        )
    )
    if isinstance(session, SASession) and start_date is not None:
        lower_bound = pd.Timestamp(start_date).to_pydatetime() - timedelta(days=7)
        query = query.filter(available_expr >= lower_bound)

    rows = query.all()
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        md = _effective_market_date(r)
        if _within_window(md, start_date, end_date):
            records.append(
                {
                    "market_date": md,
                    "event_type": r.event_type,
                    "count": 1,
                }
            )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    pivot = df.pivot_table(index="market_date", columns="event_type", values="count", aggfunc="sum", fill_value=0)
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.sort_index()
