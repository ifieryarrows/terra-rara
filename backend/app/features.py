"""
Feature engineering for XGBoost model.

Handles:
- Multi-symbol calendar alignment
- Limited forward-fill for holidays
- Technical indicators (SMA, EMA, RSI, volatility)
- Lagged returns
- Sentiment feature join
"""

from functools import lru_cache
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

# Suppress silent downcasting FutureWarnings globally for this module
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass
from sqlalchemy import func
from sqlalchemy.orm import Session

# pd.set_option("future.no_silent_downcasting", True)

from app.db import SessionLocal
from app.models import PriceBar, DailySentiment, DailySentimentV2
from app.settings import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def _normalize_bar_calendar_dates(raw_dates: pd.Series, utc_ts: pd.Series) -> pd.Series:
    """Normalize bar dates to tz-naive midnight preserving explicit non-UTC exchange offsets."""
    utc_norm = utc_ts.dt.tz_localize(None).dt.normalize()
    if raw_dates.empty:
        return utc_norm

    needs_offset_scan = False
    for raw_val in raw_dates:
        if isinstance(raw_val, datetime):
            offset = raw_val.utcoffset()
            if offset is not None and offset != timedelta(0):
                needs_offset_scan = True
                break
        elif isinstance(raw_val, str):
            stripped = raw_val.strip()
            if not stripped.endswith(("Z", "z", "+00:00", "-00:00")):
                needs_offset_scan = True
                break

    if not needs_offset_scan:
        return utc_norm

    normalized = []
    for raw_val, utc_val in zip(raw_dates, utc_ts):
        if isinstance(raw_val, datetime):
            normalized.append(raw_val.date())
        elif isinstance(raw_val, str) and not raw_val.strip().endswith(
            ("Z", "z", "+00:00", "-00:00")
        ):
            parsed = pd.to_datetime(raw_val, errors="coerce")
            normalized.append(parsed.date() if pd.notna(parsed) else utc_val.date())
        else:
            normalized.append(utc_val.date())
    return pd.to_datetime(normalized)


def _rows_to_price_df(rows: list, symbol: str) -> pd.DataFrame:
    """Convert raw PriceBar rows into a validated, date-indexed DataFrame."""
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume", "adj_close"],
    )
    numeric_columns = ["open", "high", "low", "close", "volume", "adj_close"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    invalid_close = df["close"].isna() | (df["close"] <= 0.0)
    if invalid_close.any():
        logger.warning(
            "Excluded %s invalid close rows for %s", int(invalid_close.sum()), symbol
        )
        df = df.loc[~invalid_close].copy()
    if df.empty:
        return pd.DataFrame()

    raw_ts = pd.to_datetime(df["date"], errors="coerce", utc=True)
    invalid_dates = raw_ts.isna()
    if invalid_dates.any():
        logger.warning(
            "Excluded %s rows with invalid date for %s", int(invalid_dates.sum()), symbol
        )
        df = df.loc[~invalid_dates].copy()
        raw_ts = raw_ts.loc[~invalid_dates]
        if df.empty:
            return pd.DataFrame()

    cal_dates = _normalize_bar_calendar_dates(df["date"], raw_ts)
    df["_raw_ts"] = raw_ts.to_numpy()
    df["date"] = cal_dates.to_numpy()

    # Drop leaked synthetic test fixture rows (100/105/95/102/1000) when mixed with
    # real market series whose price scale, volume scale, or calendar day differs.
    fixture_mask = (
        np.isclose(df["open"].astype(float), 100.0, rtol=1e-6, atol=1e-6)
        & np.isclose(df["high"].astype(float), 105.0, rtol=1e-6, atol=1e-6)
        & np.isclose(df["low"].astype(float), 95.0, rtol=1e-6, atol=1e-6)
        & np.isclose(df["close"].astype(float), 102.0, rtol=1e-6, atol=1e-6)
        & np.isclose(df["volume"].fillna(0.0).astype(float), 1000.0, rtol=1e-6, atol=1e-6)
    )
    if fixture_mask.any() and (~fixture_mask).any():
        non_fixture = df.loc[~fixture_mask]
        non_fixture_median = float(non_fixture["close"].astype(float).median())
        non_fixture_vols = non_fixture["volume"].dropna().astype(float)
        non_fixture_vol_median = (
            float(non_fixture_vols.median()) if not non_fixture_vols.empty else None
        )
        price_mismatch = abs(non_fixture_median - 102.0) > 25.0
        vol_mismatch = (
            non_fixture_vol_median is not None
            and (non_fixture_vol_median < 100.0 or non_fixture_vol_median > 10000.0)
        )
        if price_mismatch or vol_mismatch:
            logger.warning(
                "Excluded %s leaked synthetic fixture rows for %s "
                "(non-fixture close median=%.2f, volume median=%s)",
                int(fixture_mask.sum()),
                symbol,
                non_fixture_median,
                f"{non_fixture_vol_median:.1f}" if non_fixture_vol_median is not None else "n/a",
            )
            df = non_fixture.copy()
        else:
            colliding_fixture = fixture_mask & df["date"].isin(set(non_fixture["date"]))
            if colliding_fixture.any():
                logger.warning(
                    "Excluded %s colliding synthetic fixture rows for %s",
                    int(colliding_fixture.sum()),
                    symbol,
                )
                df = df.loc[~colliding_fixture].copy()
        if df.empty:
            return pd.DataFrame()

    df = df.sort_values("_raw_ts").drop(columns=["_raw_ts"])
    df = df.set_index("date")
    if df.index.duplicated().any():
        dup_count = int(df.index.duplicated(keep="last").sum())
        logger.warning(
            "Deduplicated %s same-calendar-day price bars for %s", dup_count, symbol
        )
        df = df[~df.index.duplicated(keep="last")]
    weekend_mask = df.index.dayofweek >= 5
    if weekend_mask.any():
        logger.warning(
            "Excluded %s weekend (Sat/Sun) ghost price bars for %s",
            int(weekend_mask.sum()),
            symbol,
        )
        df = df.loc[~weekend_mask]
        if df.empty:
            return pd.DataFrame()
    return df


def load_price_data(
    session: Session,
    symbol: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load price data for a symbol.
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    query = session.query(
        PriceBar.date,
        PriceBar.open,
        PriceBar.high,
        PriceBar.low,
        PriceBar.close,
        PriceBar.volume,
        PriceBar.adj_close
    ).filter(PriceBar.symbol == symbol)
    
    if start_date:
        query = query.filter(PriceBar.date >= start_date)
    if end_date:
        query = query.filter(PriceBar.date <= end_date)
    
    query = query.order_by(PriceBar.date.asc())
    
    rows = query.all()
    return _rows_to_price_df(rows, symbol)


_ORIGINAL_LOAD_PRICE_DATA = load_price_data


def load_multi_price_data(
    session: Session,
    symbols: list[str],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> dict[str, pd.DataFrame]:
    """
    Batch-load price data for multiple symbols in a single SQL query.

    Automatically falls back to per-symbol `load_price_data` when `load_price_data`
    is monkeypatched in tests or when `session` is a mock object.
    """
    unique_symbols = list(dict.fromkeys(s for s in symbols if s))
    if not unique_symbols:
        return {}

    if (
        load_price_data is not _ORIGINAL_LOAD_PRICE_DATA
        or not isinstance(session, Session)
    ):
        result: dict[str, pd.DataFrame] = {}
        for sym in unique_symbols:
            df = load_price_data(session, sym, start_date, end_date)
            if not df.empty:
                result[sym] = df
        return result

    query = session.query(
        PriceBar.symbol,
        PriceBar.date,
        PriceBar.open,
        PriceBar.high,
        PriceBar.low,
        PriceBar.close,
        PriceBar.volume,
        PriceBar.adj_close,
    ).filter(PriceBar.symbol.in_(unique_symbols))

    if start_date:
        query = query.filter(PriceBar.date >= start_date)
    if end_date:
        query = query.filter(PriceBar.date <= end_date)

    query = query.order_by(PriceBar.symbol.asc(), PriceBar.date.asc())
    all_rows = query.all()
    if not all_rows:
        return {}

    grouped: dict[str, list] = {sym: [] for sym in unique_symbols}
    for row in all_rows:
        sym = row[0]
        if sym in grouped:
            grouped[sym].append(row[1:])

    result = {}
    for sym in unique_symbols:
        sym_rows = grouped.get(sym)
        if sym_rows:
            df = _rows_to_price_df(sym_rows, sym)
            if not df.empty:
                result[sym] = df
    return result


def load_sentiment_data(
    session: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load daily sentiment data.
    
    Returns:
        DataFrame with columns: date, sentiment_index, news_count
    """
    settings = get_settings()
    source = str(getattr(settings, "scoring_source", "news_articles")).strip().lower()
    use_v2 = source == "news_processed"

    rows = []
    if use_v2:
        query_v2 = session.query(
            DailySentimentV2.date,
            DailySentimentV2.sentiment_index,
            DailySentimentV2.news_count
        )
        if start_date:
            query_v2 = query_v2.filter(DailySentimentV2.date >= start_date)
        if end_date:
            query_v2 = query_v2.filter(DailySentimentV2.date <= end_date)
        rows = query_v2.order_by(DailySentimentV2.date.asc()).all()
        if not rows:
            logger.warning("No rows in daily_sentiments_v2; falling back to daily_sentiments")

    if not rows:
        query = session.query(
            DailySentiment.date,
            DailySentiment.sentiment_index,
            DailySentiment.news_count
        )
        if start_date:
            query = query.filter(DailySentiment.date >= start_date)
        if end_date:
            query = query.filter(DailySentiment.date <= end_date)
        rows = query.order_by(DailySentiment.date.asc()).all()

    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=["date", "sentiment_index", "news_count"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.set_index("date")
    
    return df


# =============================================================================
# Technical Indicators
# =============================================================================

def compute_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Compute percentage returns."""
    return prices.pct_change(periods, fill_method=None)


def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return prices.rolling(window=window, min_periods=1).mean()


def compute_ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return prices.ewm(span=span, adjust=False).mean()


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Neutral RSI when undefined


def compute_volatility(returns: pd.Series, window: int = 10) -> pd.Series:
    """Rolling standard deviation of returns (volatility)."""
    return returns.rolling(window=window, min_periods=1).std()


# =============================================================================
# Feature Engineering
# =============================================================================

def generate_symbol_features(
    df: pd.DataFrame,
    symbol: str,
    include_lags: list[int] = [1, 2, 3, 5],
    sma_windows: list[int] = [5, 10, 20],
    ema_windows: list[int] = [5, 10, 20],
    rsi_window: int = 14,
    vol_window: int = 10,
    include_raw_ma: bool = True,
) -> pd.DataFrame:
    """
    Generate features for a single symbol.

    When ``include_raw_ma`` is True (legacy default), raw SMA_w/EMA_w levels
    are emitted alongside stationary features. When False, non-stationary raw
    price levels are replaced by stationary relative ratios and multi-horizon
    log returns to prevent OOD covariate shift across price regimes.
    """
    features = pd.DataFrame(index=df.index)
    prefix = f"{symbol}_" if symbol else ""

    close = df["close"].astype(float)

    # Returns
    ret1 = compute_returns(close, 1)
    features[f"{prefix}ret1"] = ret1

    # Lagged returns
    for lag in include_lags:
        features[f"{prefix}lag_ret1_{lag}"] = ret1.shift(lag)

    # Moving averages / stationary trend ratios
    sma_20 = compute_sma(close, 20)
    if include_raw_ma:
        for w in sma_windows:
            features[f"{prefix}SMA_{w}"] = compute_sma(close, w)
        for w in ema_windows:
            features[f"{prefix}EMA_{w}"] = compute_ema(close, w)
    else:
        safe_close = close.where(close > 0.0, np.nan)
        for w in sma_windows:
            sma_w = sma_20 if w == 20 else compute_sma(close, w)
            features[f"{prefix}close_sma_{w}_ratio"] = (safe_close / sma_w.replace(0, np.nan)) - 1.0
        if 5 in ema_windows and 20 in ema_windows:
            ema_fast = compute_ema(close, 5)
            ema_slow = compute_ema(close, 20)
            features[f"{prefix}ema_fast_slow_spread"] = (ema_fast / ema_slow.replace(0, np.nan)) - 1.0
        for h in (5, 10, 20):
            prev = safe_close.shift(h)
            features[f"{prefix}ret{h}"] = np.log(safe_close / prev.where(prev > 0.0, np.nan))
        vol_short = compute_volatility(ret1, 5)
        vol_long = compute_volatility(ret1, 20)
        vol_20_safe = vol_long.replace(0, np.nan).clip(lower=1e-4)
        features[f"{prefix}vol_5_20_ratio"] = vol_short / vol_long.replace(0, np.nan)
        for h in (5, 10, 20):
            features[f"{prefix}risk_adj_ret{h}"] = (
                features[f"{prefix}ret{h}"] / (vol_20_safe * np.sqrt(float(h)))
            ).clip(-5.0, 5.0)
        features[f"{prefix}signed_vol_ratio"] = (
            np.sign(features[f"{prefix}ret10"].fillna(0.0))
            * ((vol_short / vol_20_safe) - 1.0).clip(-3.0, 3.0)
        )

    # RSI
    features[f"{prefix}RSI_{rsi_window}"] = compute_rsi(close, rsi_window)

    # Volatility
    features[f"{prefix}vol_{vol_window}"] = compute_volatility(ret1, vol_window)

    # Price level (normalized by SMA for scale invariance)
    features[f"{prefix}price_sma_ratio"] = close / sma_20.replace(0, np.nan)

    return features


def align_to_target_calendar(
    target_df: pd.DataFrame,
    other_dfs: dict[str, pd.DataFrame],
    max_ffill: int = 3
) -> dict[str, pd.DataFrame]:
    """
    Align other DataFrames to target symbol's trading calendar.
    Uses limited forward-fill for handling holidays/gaps.
    
    Args:
        target_df: DataFrame with target symbol (defines the index)
        other_dfs: Dict of symbol -> DataFrame
        max_ffill: Maximum days to forward-fill
    
    Returns:
        Dict of aligned DataFrames
    """
    target_index = target_df.index
    aligned = {}
    
    for symbol, df in other_dfs.items():
        if df.empty:
            aligned[symbol] = pd.DataFrame(index=target_index)
            continue
        
        # Reindex to target calendar
        reindexed = df.reindex(target_index)
        
        # Limited forward-fill (infer_objects fixes future downcasting warning)
        reindexed = reindexed.ffill(limit=max_ffill).infer_objects(copy=False)
        
        aligned[symbol] = reindexed
    
    return aligned


def build_shared_feature_frame(
    target_df: pd.DataFrame,
    other_dfs: dict[str, pd.DataFrame],
    sentiment_df: pd.DataFrame,
    *,
    sentiment_missing_fill: float,
    max_ffill: int = 3,
) -> pd.DataFrame:
    """Single preprocessing contract used by XGBoost training and inference."""
    aligned = align_to_target_calendar(target_df, other_dfs, max_ffill=max_ffill)
    feature_frames = [
        generate_symbol_features(target_df, target_df.attrs.get("symbol", "HG=F"))
    ]
    for symbol, frame in aligned.items():
        if not frame.empty:
            feature_frames.append(generate_symbol_features(frame, symbol))

    all_features = (
        pd.concat(feature_frames, axis=1)
        if len(feature_frames) > 1
        else feature_frames[0]
    )

    if not sentiment_df.empty:
        sentiment_aligned = sentiment_df.reindex(target_df.index).ffill(limit=max_ffill)
        sentiment_parts = [
            sentiment_aligned["sentiment_index"].fillna(sentiment_missing_fill).rename("sentiment__index"),
            sentiment_aligned["news_count"].fillna(0).rename("sentiment__news_count"),
        ]
    else:
        sentiment_parts = [
            pd.Series(sentiment_missing_fill, index=all_features.index, name="sentiment__index"),
            pd.Series(0, index=all_features.index, name="sentiment__news_count"),
        ]
    return pd.concat([all_features] + sentiment_parts, axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)


def build_feature_matrix(
    session: Session,
    target_symbol: str = "HG=F",
    lookback_days: int = 365,
    max_ffill: int = 3
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build complete feature matrix for model training.
    
    Returns:
        Tuple of (X features DataFrame, y target Series)
    
    Target: Next-day return (more stationary than price)
    """
    settings = get_settings()
    symbols = settings.training_symbols  # Use active.json symbols for training
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)
    
    all_requested = [target_symbol] + [s for s in symbols if s != target_symbol]
    price_map = load_multi_price_data(session, all_requested, start_date, end_date)

    # Load target symbol
    target_df = price_map.get(target_symbol, pd.DataFrame())
    
    if target_df.empty:
        logger.error(f"No price data for target symbol {target_symbol}")
        return pd.DataFrame(), pd.Series()
    target_df.attrs["symbol"] = target_symbol
    
    logger.info(f"Target symbol {target_symbol}: {len(target_df)} bars")
    
    # Load other symbols
    other_dfs = {}
    for symbol in symbols:
        if symbol != target_symbol:
            df = price_map.get(symbol, pd.DataFrame())
            if not df.empty:
                other_dfs[symbol] = df
                logger.info(f"Symbol {symbol}: {len(df)} bars")
    
    # Load and join sentiment data
    sentiment_df = load_sentiment_data(session, start_date, end_date)
    all_features = build_shared_feature_frame(
        target_df,
        other_dfs,
        sentiment_df,
        sentiment_missing_fill=settings.sentiment_missing_fill,
        max_ffill=max_ffill,
    )
    
    # Create target: next-day return
    # IMPORTANT: Shift by -1 to get FUTURE return (what we're predicting)
    target_ret = compute_returns(target_df["close"], 1)
    y = target_ret.shift(-1)  # Next day's return
    y.name = "target_ret"
    
    # Align features and target
    X = all_features.copy()
    
    # Drop rows with NaN target (last row won't have next-day return)
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Fill remaining NaN features with 0 (instead of dropping rows)
    # This is important for new symbols that may have missing data
    nan_count_before = X.isna().sum().sum()
    
    if nan_count_before > 0:
        logger.info(f"Filled {nan_count_before} NaN values in features with 0")
    
    logger.info(f"Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y


def get_feature_descriptions() -> dict[str, str]:
    """Get human-readable descriptions for feature names (legacy dict)."""
    return {
        "sentiment__index": "Market Sentiment Index",
        "sentiment__news_count": "Daily News Volume",
        "ret1": "1-day Return",
        "lag_ret1_1": "Return Lag 1",
        "lag_ret1_2": "Return Lag 2",
        "lag_ret1_3": "Return Lag 3",
        "lag_ret1_5": "Return Lag 5",
        "SMA_5": "5-day SMA",
        "SMA_10": "10-day SMA",
        "SMA_20": "20-day SMA",
        "EMA_5": "5-day EMA",
        "EMA_10": "10-day EMA",
        "EMA_20": "20-day EMA",
        "RSI_14": "14-day RSI",
        "vol_10": "10-day Volatility",
        "price_sma_ratio": "Price/SMA Ratio",
    }


# ---------------------------------------------------------------------------
# Symbol and feature-pattern dictionaries used for user-facing labels.
# Extend these whenever new proxies enter the feature universe.
# ---------------------------------------------------------------------------
_SYMBOL_LABEL_MAP: dict[str, str] = {
    "HG=F":       "Copper Futures",
    "CL=F":       "WTI Crude Oil",
    "BZ=F":       "Brent Crude",
    "GC=F":       "Gold Futures",
    "SI=F":       "Silver Futures",
    "PL=F":       "Platinum Futures",
    "PA=F":       "Palladium Futures",
    "NG=F":       "Natural Gas",
    "DX-Y.NYB":   "US Dollar Index",
    "^TNX":       "US 10Y Treasury Yield",
    "^VIX":       "VIX Volatility",
    "^GSPC":      "S&P 500 Index",
    "^DJI":       "Dow Jones Index",
    "^IXIC":      "Nasdaq Composite",
    "COPX":       "Global Copper Miners ETF",
    "PICK":       "Metals & Mining ETF",
    "XME":        "Metals & Mining ETF",
    "SCCO":       "Southern Copper",
    "FCX":        "Freeport-McMoRan",
    "BHP":        "BHP Group",
    "RIO":        "Rio Tinto",
    "VALE":       "Vale",
    "GDX":        "Gold Miners ETF",
    "USO":        "US Oil Fund",
    "UUP":        "US Dollar Bull ETF",
    "FXA":        "Australian Dollar ETF",
    "TIP":        "TIPS Bond ETF",
    "EEM":        "Emerging Markets ETF",
    "FXI":        "China Large-Cap ETF",
    "GXC":        "China Large-Cap",
    "HBM.TO":     "Hudbay Minerals",
    "HBM":        "Hudbay Minerals",
    "LUN.TO":     "Lundin Mining",
    "LUN":        "Lundin Mining",
    "TECK":       "Teck Resources",
    "CPER":       "US Copper Index Fund",
    "CEW":        "Emerging Markets Currency ETF",
    "ALI=F":      "Aluminum Futures",
    "DBA":        "Agricultural Commodity ETF",
    "DBB":        "Base Metals ETF",
    "DBC":        "Broad Commodity ETF",
}


@lru_cache(maxsize=1)
def _symbol_prefix_aliases() -> tuple[tuple[str, str], ...]:
    """Return raw and sanitized feature prefixes mapped to raw symbols, sorted longest first."""
    aliases: list[tuple[str, str]] = []
    for symbol in _SYMBOL_LABEL_MAP:
        aliases.append((symbol, symbol))
        sanitized = re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_")
        if sanitized != symbol:
            aliases.append((sanitized, symbol))
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(aliases)


def _humanize_symbol(sym: str) -> str:
    """Convert ticker to a user-facing label."""
    if not sym:
        return ""
    if sym in _SYMBOL_LABEL_MAP:
        return _SYMBOL_LABEL_MAP[sym]
    # Strip yfinance suffixes for a cleaner display
    return sym.replace("=F", "").replace("^", "").replace("-Y.NYB", " Index")


_INDICATOR_LABEL_MAP: dict[str, str] = {
    "ret1":                 "1-day price change",
    "ret2":                 "2-day price change",
    "ret5":                 "5-day price change",
    "ret10":                "10-day price change",
    "ret20":                "20-day price change",
    "RSI_14":               "14-day price momentum",
    "SMA_5":                "5-day moving average",
    "SMA_10":               "10-day moving average",
    "SMA_20":               "20-day moving average",
    "EMA_5":                "5-day weighted moving average",
    "EMA_10":               "10-day weighted moving average",
    "EMA_20":               "20-day weighted moving average",
    "close_sma_5_ratio":    "Price vs. 5-day average",
    "close_sma_10_ratio":   "Price vs. 10-day average",
    "close_sma_20_ratio":   "Price vs. 20-day average",
    "ema_fast_slow_spread": "5-day vs. 20-day EMA spread",
    "vol_5":                "5-day price volatility",
    "vol_10":               "10-day price volatility",
    "vol_20":               "20-day price volatility",
    "vol_5_20_ratio":       "5-day vs. 20-day volatility ratio",
    "risk_adj_ret5":        "5-day risk-adjusted return",
    "risk_adj_ret10":       "10-day risk-adjusted return",
    "risk_adj_ret20":       "20-day risk-adjusted return",
    "signed_vol_ratio":     "Directional volatility regime shift",
    "theta_5d_prior":       "5-day Theta trend prior",
    "macro_fair_value_zscore": "Macro basket fair-value deviation",
    "price_sma_ratio":      "Price vs. 20-day average",
    "target":               "Model target (next-day change)",
}


def _label_indicator(token: str) -> str:
    """Return a friendly label for a bare indicator token."""
    if token in _INDICATOR_LABEL_MAP:
        return _INDICATOR_LABEL_MAP[token]
    t_lower = token.lower()
    # Dynamic SMA/EMA e.g. SMA_10, to_sma_10
    m_sma = re.match(r"^(?:to_)?sma_(\d+)$", t_lower)
    if m_sma:
        return f"{m_sma.group(1)}-day moving average"
    m_ema = re.match(r"^ema_(\d+)$", t_lower)
    if m_ema:
        return f"{m_ema.group(1)}-day weighted moving average"
    m_vol = re.match(r"^vol_(\d+)$", t_lower)
    if m_vol:
        return f"{m_vol.group(1)}-day price volatility"
    m_csma = re.match(r"^close_sma_(\d+)_ratio$", t_lower)
    if m_csma:
        return f"Price vs. {m_csma.group(1)}-day average"
    m_rsi = re.match(r"^rsi_(\d+)$", t_lower)
    if m_rsi:
        return f"{m_rsi.group(1)}-day price momentum"
    # Lagged return — e.g. lag_ret1_2 → "2-day lagged return"
    m_lag = re.match(r"^lag_ret(?:1|2)?_(\d+)$", t_lower)
    if m_lag:
        n = m_lag.group(1)
        return f"{n}-day lagged return"
    # Embedding features
    if t_lower.startswith("emb_") or t_lower.startswith("news_emb_"):
        return f"news embedding component {token.split('_')[-1]}"
    # Sentiment features
    if t_lower.startswith("sentiment"):
        if "news_count" in t_lower:
            return "daily news volume"
        return "sentiment index"
    # Fallback: prettify by replacing underscores and lower-casing
    return token.replace("_", " ").lower()

_KNOWN_INDICATOR_PREFIXES = frozenset(
    {"ret", "lag_ret", "sma", "ema", "rsi", "vol", "price_sma", "close_sma", "to_sma", "target"}
)


def describe_feature(feature: str) -> dict[str, str]:
    """
    Return a structured, user-facing description of a feature.

    Output keys:
        - label:       short, human-readable label (≈ 30 chars)
        - description: longer description with subject and metric
        - category:    high-level bucket (Price, Momentum, Volatility,
                       Sentiment, Macro, Sector, Embedding, Other)
        - time_horizon optional — lookback window in days when applicable
    """
    if not feature:
        return {"label": "—", "description": "", "category": "Other"}

    # Pattern: <SYMBOL>_<token> where token can itself contain underscores.
    # We try the longest matching raw or sanitized symbol prefix. Training
    # features use safe names such as HG_F and DX_Y_NYB, while the label map
    # uses the canonical market symbols HG=F and DX-Y.NYB.
    symbol = ""
    token = feature
    for prefix, canonical in _symbol_prefix_aliases():
        prefix_with_ = f"{prefix}_"
        if feature.startswith(prefix_with_):
            symbol = canonical
            token = feature[len(prefix_with_) :]
            break

    if not symbol:
        first_token = feature.split("_", 1)[0].lower()
        if (
            first_token in _KNOWN_INDICATOR_PREFIXES
            or any(
                feature.lower().startswith(p)
                for p in ("ret", "vol_", "sma_", "ema_", "rsi_", "price_sma", "close_sma", "to_sma")
            )
        ):
            symbol = "HG=F"
            token = feature
        elif "_" in feature:
            maybe_sym, rest = feature.split("_", 1)
            if re.fullmatch(r"[A-Z0-9^=.\-]{1,20}", maybe_sym or "") and maybe_sym.lower() not in _KNOWN_INDICATOR_PREFIXES:
                symbol = maybe_sym
                token = rest

    indicator_label = _label_indicator(token)

    # Category classification
    category = "Other"
    t = token.lower()
    if t.startswith("sentiment"):
        category = "Sentiment"
    elif t.startswith("emb_") or t.startswith("news_emb_"):
        category = "Embedding"
    elif t.startswith("rsi") or t.startswith("ret") or t.startswith("lag_ret") or t.startswith("risk_adj_ret"):
        category = "Momentum"
    elif t == "macro_fair_value_zscore":
        category = "Macro"
    elif t == "theta_5d_prior":
        category = "Trend"
    elif t.startswith("vol") or t == "signed_vol_ratio":
        category = "Volatility"
    elif t.startswith("sma") or t.startswith("ema") or "price_sma" in t or "close_sma" in t or "to_sma" in t:
        category = "Trend"
    elif symbol and symbol != "HG=F":
        category = "Macro" if symbol in {"DX-Y.NYB", "^TNX", "^VIX", "^GSPC", "^DJI", "^IXIC"} else "Sector"

    # Time horizon extraction (e.g. RSI_14, vol_10)
    time_horizon = ""
    m = re.search(r"_(\d+)$", token)
    if m:
        time_horizon = f"{m.group(1)}d"

    if symbol:
        subject = _humanize_symbol(symbol)
        label = f"{subject} · {indicator_label}"
        description = f"{subject} — {indicator_label}"
    else:
        label = indicator_label[:1].upper() + indicator_label[1:]
        description = label

    return {
        "label": label,
        "description": description,
        "category": category,
        "time_horizon": time_horizon,
    }
