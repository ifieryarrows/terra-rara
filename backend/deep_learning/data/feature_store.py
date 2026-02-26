"""
Centralised Feature Store for TFT-ASRO.

Fuses all heterogeneous data sources (price, sentiment, embeddings, LME,
calendar) into a single long-format DataFrame suitable for
pytorch_forecasting.TimeSeriesDataSet.

TFT data categories:
    1. time_varying_unknown_reals  - observed only in the past
    2. time_varying_known_reals    - known into the future (calendar, etc.)
    3. static_reals / static_categoricals - time-invariant per group
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from deep_learning.config import TFTASROConfig, get_tft_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Screener bridge: load correlated symbols from active.json / screener output
# ---------------------------------------------------------------------------

def load_training_symbols() -> list[str]:
    """
    Load the active symbol set from ``config/symbol_sets/active.json``.

    Falls back to settings.training_symbols if the file cannot be read.
    This bridges the screener's challenger/champion pipeline with the
    TFT feature store so that the same statistically validated symbols
    feed both XGBoost and TFT.
    """
    import json
    from pathlib import Path

    backend_root = Path(__file__).resolve().parent.parent.parent
    active_path = backend_root / "config" / "symbol_sets" / "active.json"

    if active_path.exists():
        try:
            data = json.loads(active_path.read_text(encoding="utf-8"))
            symbols = data.get("symbols", [])
            if symbols:
                logger.info(
                    "Loaded %d training symbols from %s (v%s)",
                    len(symbols), active_path.name, data.get("version", "?"),
                )
                return symbols
        except Exception as exc:
            logger.warning("Failed to read %s: %s", active_path, exc)

    try:
        from app.settings import get_settings
        return get_settings().training_symbols
    except Exception:
        return ["HG=F", "DX-Y.NYB", "CL=F", "FXI"]


def load_screener_selected_symbols(
    artifacts_dir: str = "artifacts/runs/latest",
) -> list[dict]:
    """
    Read the screener's ``selected_symbols.json`` to get the full audit-trail
    entries including IS/OOS Pearson, category, and lead-lag information.

    Returns a list of dicts (one per selected symbol).
    """
    import json
    from pathlib import Path

    backend_root = Path(__file__).resolve().parent.parent.parent
    selected_path = backend_root / artifacts_dir / "selected_symbols.json"

    if not selected_path.exists():
        logger.info("No screener selected_symbols.json found at %s", selected_path)
        return []

    try:
        data = json.loads(selected_path.read_text(encoding="utf-8"))
        selected = data.get("selected", [])
        logger.info(
            "Loaded %d screener-selected symbols (rules v%s, run %s)",
            len(selected),
            data.get("selection_rules_version", "?"),
            data.get("screener_run_id", "?"),
        )
        return selected
    except Exception as exc:
        logger.warning("Failed to read screener output: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Calendar / known-future features
# ---------------------------------------------------------------------------

def _build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic features known at any future date."""
    cal = pd.DataFrame(index=index)

    cal["day_of_week"] = index.dayofweek.astype(np.float32) / 6.0
    cal["day_of_month"] = index.day.astype(np.float32) / 31.0
    cal["month"] = index.month.astype(np.float32) / 12.0

    day_frac = 2 * np.pi * index.dayofyear / 365.25
    cal["cal_sin_day"] = np.sin(day_frac).astype(np.float32)
    cal["cal_cos_day"] = np.cos(day_frac).astype(np.float32)

    month_frac = 2 * np.pi * index.month / 12.0
    cal["cal_sin_month"] = np.sin(month_frac).astype(np.float32)
    cal["cal_cos_month"] = np.cos(month_frac).astype(np.float32)

    cal["is_monday"] = (index.dayofweek == 0).astype(np.float32)
    cal["is_friday"] = (index.dayofweek == 4).astype(np.float32)

    cal["is_month_start"] = index.is_month_start.astype(np.float32)
    cal["is_month_end"] = index.is_month_end.astype(np.float32)
    cal["is_quarter_end"] = index.is_quarter_end.astype(np.float32)

    return cal


# ---------------------------------------------------------------------------
# Price / technical features (reuses existing helpers)
# ---------------------------------------------------------------------------

def _build_price_features(
    session,
    symbol: str,
    start_date,
    end_date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load price data and compute technical features for *symbol*.

    Returns (raw_price_df, features_df).
    """
    from app.features import load_price_data, generate_symbol_features

    price_df = load_price_data(session, symbol, start_date, end_date)
    if price_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    features = generate_symbol_features(price_df, symbol)
    return price_df, features


# ---------------------------------------------------------------------------
# Embedding features (daily aggregated PCA vectors)
# ---------------------------------------------------------------------------

def _build_daily_embedding_features(
    session,
    index: pd.DatetimeIndex,
    pca_dim: int = 32,
) -> pd.DataFrame:
    """
    Load PCA-reduced FinBERT embeddings, aggregate to daily level,
    and reindex onto the trading calendar.
    """
    from sqlalchemy import func as sa_func
    from app.models import NewsEmbedding, NewsProcessed, NewsRaw
    from deep_learning.data.embeddings import bytes_to_embedding, aggregate_daily_embeddings

    rows = (
        session.query(
            sa_func.date(NewsRaw.published_at).label("date"),
            NewsEmbedding.embedding_pca,
        )
        .join(NewsProcessed, NewsEmbedding.news_processed_id == NewsProcessed.id)
        .join(NewsRaw, NewsProcessed.raw_id == NewsRaw.id)
        .order_by(NewsRaw.published_at.asc())
        .all()
    )

    if not rows:
        cols = [f"emb_pca_{i}" for i in range(pca_dim)]
        return pd.DataFrame(0.0, index=index, columns=cols)

    date_groups: dict[str, list[np.ndarray]] = {}
    for r in rows:
        d = str(r.date)
        vec = bytes_to_embedding(r.embedding_pca, dim=pca_dim)
        date_groups.setdefault(d, []).append(vec)

    records = []
    for d, vecs in date_groups.items():
        agg = aggregate_daily_embeddings(np.stack(vecs))
        record = {"date": pd.Timestamp(d)}
        for i, v in enumerate(agg):
            record[f"emb_pca_{i}"] = float(v)
        records.append(record)

    emb_df = pd.DataFrame(records).set_index("date").sort_index()
    emb_df.index = pd.to_datetime(emb_df.index)

    emb_aligned = emb_df.reindex(index).ffill(limit=3).fillna(0.0)
    return emb_aligned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tft_dataframe(
    session,
    cfg: Optional[TFTASROConfig] = None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Build the master DataFrame for TFT training / inference.

    Returns:
        (df, time_varying_unknown_reals, time_varying_known_reals, target_cols)

    The returned df has:
        - "time_idx"  : integer time index (required by pytorch_forecasting)
        - "group_id"  : constant "copper" (single series)
        - "target"    : next-day simple return
        - columns for all three TFT feature categories
    """
    if cfg is None:
        cfg = get_tft_config()

    target_symbol = cfg.feature_store.target_symbol
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=cfg.training.lookback_days)

    # ---- 1. Price & technical indicators ----
    # Use screener-validated symbols from active.json
    training_symbols = load_training_symbols()
    logger.info("Building features with %d symbols: %s", len(training_symbols), training_symbols[:5])

    price_df, price_features = _build_price_features(session, target_symbol, start_date, end_date)
    if price_df.empty:
        raise ValueError(f"No price data for {target_symbol}")

    # Add correlated symbols' features (screener-validated)
    from app.features import load_price_data, generate_symbol_features, align_to_target_calendar

    other_dfs = {}
    for sym in training_symbols:
        if sym == target_symbol:
            continue
        sym_df = load_price_data(session, sym, start_date, end_date)
        if not sym_df.empty:
            other_dfs[sym] = sym_df

    if other_dfs:
        aligned = align_to_target_calendar(price_df, other_dfs, max_ffill=cfg.feature_store.max_ffill)
        for sym, df in aligned.items():
            if not df.empty:
                sym_feats = generate_symbol_features(df, sym)
                price_features = price_features.join(sym_feats, how="left")
        logger.info("Added features from %d correlated symbols", len(aligned))

    target_index = price_df.index
    logger.info("Price data: %d bars for %s", len(target_index), target_symbol)

    # ---- 2. Sentiment features ----
    from app.features import load_sentiment_data
    from deep_learning.data.sentiment_features import (
        build_all_sentiment_features,
        build_event_counts_from_db,
    )

    sent_df = load_sentiment_data(session, start_date, end_date)
    if not sent_df.empty:
        sent_aligned = sent_df.reindex(target_index).ffill(limit=cfg.feature_store.max_ffill)
        sent_aligned["sentiment_index"] = sent_aligned["sentiment_index"].fillna(0.0)
        sent_aligned["news_count"] = sent_aligned["news_count"].fillna(0)

        event_counts = build_event_counts_from_db(session, start_date, end_date)
        advanced_sent = build_all_sentiment_features(sent_aligned, event_counts=event_counts, cfg=cfg.sentiment)
    else:
        sent_aligned = pd.DataFrame(
            {"sentiment_index": 0.0, "news_count": 0},
            index=target_index,
        )
        advanced_sent = pd.DataFrame(index=target_index)

    # ---- 3. Embedding features ----
    emb_features = _build_daily_embedding_features(session, target_index, pca_dim=cfg.embedding.pca_dim)

    # ---- 4. LME / physical market features ----
    from deep_learning.data.lme_warehouse import fetch_lme_data, compute_lme_features, compute_proxy_lme_features
    from deep_learning.data.futures_curve import build_futures_features_from_yfinance

    lme_raw = fetch_lme_data(cfg.lme)
    if not lme_raw.empty:
        lme_features = compute_lme_features(lme_raw, windows=cfg.lme.stock_change_windows)
        lme_features = lme_features.reindex(target_index).ffill(limit=cfg.lme.max_ffill_days)
    else:
        lme_features = compute_proxy_lme_features(price_df)

    futures_features = build_futures_features_from_yfinance(session, target_symbol, cfg.training.lookback_days)
    if not futures_features.empty:
        futures_features = futures_features.reindex(target_index).ffill(limit=3)
    else:
        futures_features = pd.DataFrame(index=target_index)

    # ---- 5. Calendar (known future) ----
    calendar_features = _build_calendar_features(target_index)

    # ---- 6. Target: next-day simple return ----
    close = price_df["close"]
    target_ret = close.pct_change().shift(-1)
    target_ret.name = "target"

    # ---- Assemble master DataFrame ----
    parts = [
        price_features,
        sent_aligned[["sentiment_index", "news_count"]],
        advanced_sent,
        emb_features,
        lme_features,
        futures_features,
        calendar_features,
        target_ret.to_frame(),
    ]

    master = pd.concat(parts, axis=1)
    master = master.loc[target_index]

    valid_mask = master["target"].notna()
    master = master[valid_mask].copy()

    master = master.fillna(0.0)

    master["time_idx"] = np.arange(len(master))
    master["group_id"] = "copper"

    # Categorise columns
    calendar_cols = list(calendar_features.columns)
    target_cols = ["target"]

    all_feature_cols = [c for c in master.columns if c not in ("time_idx", "group_id", "target")]
    time_varying_known = [c for c in calendar_cols if c in master.columns]
    time_varying_unknown = [c for c in all_feature_cols if c not in time_varying_known]

    logger.info(
        "Feature store built: %d rows, %d unknown features, %d known features, %d embedding dims",
        len(master),
        len(time_varying_unknown),
        len(time_varying_known),
        len([c for c in master.columns if c.startswith("emb_pca_")]),
    )

    return master, time_varying_unknown, time_varying_known, target_cols
