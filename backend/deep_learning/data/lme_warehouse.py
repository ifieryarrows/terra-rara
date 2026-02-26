"""
LME Warehouse Stock Data Pipeline.

Fetches London Metal Exchange copper warehouse inventory data and computes
physical-market features that capture supply/demand dynamics invisible to
pure price-based indicators:

- Total stock levels and change velocity
- Cancelled warrant ratio (leading indicator of physical demand)
- Inventory depletion rate

Data sources (in priority order):
    1. Nasdaq Data Link (formerly Quandl) - LME/PR_CU dataset
    2. Fallback: synthetic features derived from price-volume patterns
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from deep_learning.config import LMEConfig, get_tft_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_lme_from_nasdaq(
    api_key: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch LME copper warehouse stock data from Nasdaq Data Link.

    Returns a DataFrame with columns:
        date, total_stock_tonnes, cancelled_warrants_tonnes,
        on_warrant_tonnes, cancelled_ratio
    """
    try:
        import nasdaqdatalink
    except ImportError:
        try:
            import quandl as nasdaqdatalink
        except ImportError:
            logger.error("Neither nasdaqdatalink nor quandl package installed")
            return pd.DataFrame()

    nasdaqdatalink.ApiConfig.api_key = api_key

    try:
        df = nasdaqdatalink.get(
            "LME/PR_CU",
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        logger.warning("Nasdaq Data Link fetch failed: %s", exc)
        return pd.DataFrame()

    if df.empty:
        return df

    df = df.reset_index()
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "date" in cl:
            col_map[c] = "date"
        elif "stock" in cl or "total" in cl:
            col_map[c] = "total_stock_tonnes"
        elif "cancel" in cl:
            col_map[c] = "cancelled_warrants_tonnes"

    df = df.rename(columns=col_map)

    if "date" not in df.columns:
        df["date"] = df.index

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.set_index("date").sort_index()

    if "total_stock_tonnes" not in df.columns:
        if len(df.columns) >= 1:
            df = df.rename(columns={df.columns[0]: "total_stock_tonnes"})

    if "cancelled_warrants_tonnes" in df.columns and "total_stock_tonnes" in df.columns:
        df["on_warrant_tonnes"] = df["total_stock_tonnes"] - df["cancelled_warrants_tonnes"]
        stock = df["total_stock_tonnes"].replace(0, np.nan)
        df["cancelled_ratio"] = df["cancelled_warrants_tonnes"] / stock
    else:
        for col in ["cancelled_warrants_tonnes", "on_warrant_tonnes", "cancelled_ratio"]:
            if col not in df.columns:
                df[col] = np.nan

    return df


def fetch_lme_data(
    cfg: Optional[LMEConfig] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Unified LME data loader with automatic source selection.
    """
    if cfg is None:
        cfg = get_tft_config().lme

    api_key = os.environ.get(cfg.nasdaq_api_key_env, "")

    if api_key:
        df = fetch_lme_from_nasdaq(api_key, start_date=start_date, end_date=end_date)
        if not df.empty:
            logger.info("Loaded %d LME records from Nasdaq Data Link", len(df))
            return df

    logger.info("LME API unavailable - generating proxy features from price data")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_lme_features(
    lme_data: pd.DataFrame,
    windows: Sequence[int] = (1, 5, 10, 20),
    depletion_window: int = 20,
) -> pd.DataFrame:
    """
    Derive physical-market features from raw LME warehouse data.

    Features:
    - lme_stock_total           : raw stock level (normalised)
    - lme_stock_change_{w}d     : w-day stock change (tonnes)
    - lme_stock_pct_change_{w}d : w-day % change
    - lme_cancelled_ratio       : ratio of cancelled warrants
    - lme_cancelled_ratio_ma5   : 5-day MA of cancelled ratio
    - lme_depletion_rate        : avg daily stock loss over window
    - lme_stock_zscore          : 60-day Z-score of stock levels
    """
    features = pd.DataFrame(index=lme_data.index)

    stock = lme_data.get("total_stock_tonnes")
    if stock is None:
        return features

    stock_mean = stock.rolling(60, min_periods=10).mean()
    stock_std = stock.rolling(60, min_periods=10).std().replace(0, np.nan)

    features["lme_stock_total"] = stock
    features["lme_stock_zscore"] = (stock - stock_mean) / stock_std

    for w in windows:
        change = stock.diff(w)
        features[f"lme_stock_change_{w}d"] = change
        pct = stock.pct_change(w)
        features[f"lme_stock_pct_change_{w}d"] = pct

    features["lme_depletion_rate"] = stock.diff(depletion_window) / depletion_window

    cancelled = lme_data.get("cancelled_ratio")
    if cancelled is not None:
        features["lme_cancelled_ratio"] = cancelled
        features["lme_cancelled_ratio_ma5"] = cancelled.rolling(5, min_periods=1).mean()
        cr_mean = cancelled.rolling(30, min_periods=5).mean()
        cr_std = cancelled.rolling(30, min_periods=5).std().replace(0, np.nan)
        features["lme_cancelled_ratio_zscore"] = (cancelled - cr_mean) / cr_std

    return features


def compute_proxy_lme_features(
    price_df: pd.DataFrame,
    volume_col: str = "volume",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    When real LME data is unavailable, derive proxy physical-market signals
    from price and volume patterns.

    Rationale: sharp volume spikes with price increases often correlate with
    physical demand surges / inventory draws.
    """
    features = pd.DataFrame(index=price_df.index)

    vol = price_df.get(volume_col)
    close = price_df.get(close_col)

    if vol is None or close is None:
        return features

    vol = vol.fillna(0).astype(float)
    close = close.astype(float)

    vol_ma20 = vol.rolling(20, min_periods=1).mean()
    vol_std20 = vol.rolling(20, min_periods=1).std().replace(0, np.nan)

    features["proxy_vol_zscore"] = (vol - vol_ma20) / vol_std20
    features["proxy_vol_spike"] = (features["proxy_vol_zscore"] > 2.0).astype(np.float32)
    features["proxy_vol_price_interaction"] = features["proxy_vol_zscore"] * close.pct_change()

    spread_5_20 = close.rolling(5).mean() - close.rolling(20).mean()
    features["proxy_momentum_spread"] = spread_5_20 / close.rolling(20).std().replace(0, np.nan)

    return features


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

def persist_lme_data(lme_df: pd.DataFrame) -> int:
    """
    Upsert LME warehouse data into the database.
    """
    from app.db import SessionLocal
    from app.models import LMEWarehouseData

    if lme_df.empty:
        return 0

    count = 0
    with SessionLocal() as session:
        for date_idx, row in lme_df.iterrows():
            date_val = pd.Timestamp(date_idx)
            if date_val.tzinfo is None:
                date_val = date_val.tz_localize("UTC")

            existing = session.query(LMEWarehouseData).filter(
                LMEWarehouseData.date == date_val
            ).first()

            total = float(row.get("total_stock_tonnes", 0))

            if existing:
                existing.total_stock_tonnes = total
                existing.cancelled_warrants_tonnes = row.get("cancelled_warrants_tonnes")
                existing.on_warrant_tonnes = row.get("on_warrant_tonnes")
                existing.cancelled_ratio = row.get("cancelled_ratio")
            else:
                session.add(LMEWarehouseData(
                    date=date_val,
                    total_stock_tonnes=total,
                    cancelled_warrants_tonnes=row.get("cancelled_warrants_tonnes"),
                    on_warrant_tonnes=row.get("on_warrant_tonnes"),
                    cancelled_ratio=row.get("cancelled_ratio"),
                ))

            count += 1

        session.commit()

    logger.info("Persisted %d LME warehouse records", count)
    return count
