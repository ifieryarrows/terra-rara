"""
Futures Curve Analysis.

Computes contango/backwardation signals from copper futures term structure.
The spread between near-month and deferred contracts is a leading indicator
of physical supply tightness.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_futures_spread(
    front_month: pd.Series,
    deferred: pd.Series,
) -> pd.DataFrame:
    """
    Compute spread features between front-month and deferred futures.

    Features:
    - futures_spread_raw        : deferred - front_month (positive = contango)
    - futures_spread_pct        : spread as percentage of front_month
    - futures_spread_zscore     : 60-day Z-score of percentage spread
    - backwardation_flag        : 1 when market is in backwardation (spread < 0)
    - contango_flag             : 1 when market is in contango (spread > 0)
    """
    features = pd.DataFrame(index=front_month.index)

    spread_raw = deferred - front_month
    spread_pct = spread_raw / front_month.replace(0, np.nan)

    roll_mean = spread_pct.rolling(60, min_periods=10).mean()
    roll_std = spread_pct.rolling(60, min_periods=10).std().replace(0, np.nan)

    features["futures_spread_raw"] = spread_raw
    features["futures_spread_pct"] = spread_pct
    features["futures_spread_zscore"] = (spread_pct - roll_mean) / roll_std
    features["backwardation_flag"] = (spread_raw < 0).astype(np.float32)
    features["contango_flag"] = (spread_raw > 0).astype(np.float32)

    return features


def compute_curve_slope(
    prices_by_maturity: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Compute the slope of the futures curve from multiple maturities.

    *prices_by_maturity* maps labels (e.g. "3m", "6m", "12m") to price series.
    The slope is the linear-regression coefficient of price vs maturity rank.
    """
    if len(prices_by_maturity) < 2:
        return pd.DataFrame()

    sorted_keys = sorted(prices_by_maturity.keys())
    combined = pd.DataFrame({k: prices_by_maturity[k] for k in sorted_keys})
    combined = combined.dropna(how="all")

    if combined.empty:
        return pd.DataFrame()

    ranks = np.arange(len(sorted_keys), dtype=np.float64)
    rank_mean = ranks.mean()
    rank_var = ((ranks - rank_mean) ** 2).sum()

    features = pd.DataFrame(index=combined.index)

    def _slope(row):
        vals = row.values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 2:
            return np.nan
        v = vals[valid]
        r = ranks[valid]
        rm = r.mean()
        return ((r - rm) * (v - v.mean())).sum() / ((r - rm) ** 2).sum()

    features["futures_curve_slope"] = combined.apply(_slope, axis=1)
    features["futures_curve_slope_ma5"] = features["futures_curve_slope"].rolling(5, min_periods=1).mean()

    return features


def build_futures_features_from_yfinance(
    session,
    target_symbol: str = "HG=F",
    lookback_days: int = 730,
) -> pd.DataFrame:
    """
    Build futures-curve features using available yfinance price data.

    Since yfinance provides the front-month HG=F contract, we approximate
    the term structure by comparing short- and long-term moving averages
    as a proxy for the futures curve slope.
    """
    from datetime import timedelta, timezone as tz
    from app.features import load_price_data

    end_date = pd.Timestamp.now(tz=tz.utc)
    start_date = end_date - timedelta(days=lookback_days)

    df = load_price_data(session, target_symbol, start_date, end_date)
    if df.empty:
        return pd.DataFrame()

    close = df["close"]

    features = pd.DataFrame(index=df.index)

    ma_5 = close.rolling(5, min_periods=1).mean()
    ma_20 = close.rolling(20, min_periods=1).mean()
    ma_60 = close.rolling(60, min_periods=10).mean()

    spread_5_20 = ma_5 - ma_20
    spread_5_60 = ma_5 - ma_60
    spread_20_60 = ma_20 - ma_60

    features["futures_spread_pct"] = spread_5_20 / ma_20.replace(0, np.nan)
    features["futures_spread_long"] = spread_5_60 / ma_60.replace(0, np.nan)
    features["backwardation_flag"] = (spread_5_20 < 0).astype(np.float32)
    features["contango_flag"] = (spread_5_20 > 0).astype(np.float32)

    zscore_mean = features["futures_spread_pct"].rolling(60, min_periods=10).mean()
    zscore_std = features["futures_spread_pct"].rolling(60, min_periods=10).std().replace(0, np.nan)
    features["futures_spread_zscore"] = (features["futures_spread_pct"] - zscore_mean) / zscore_std

    features["futures_curve_slope"] = (spread_20_60 / ma_60.replace(0, np.nan)).rolling(5, min_periods=1).mean()

    return features
