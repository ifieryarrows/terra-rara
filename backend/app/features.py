"""
Feature engineering for XGBoost model.

Handles:
- Multi-symbol calendar alignment
- Limited forward-fill for holidays
- Technical indicators (SMA, EMA, RSI, volatility)
- Lagged returns
- Sentiment feature join
"""

import logging
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
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "adj_close"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.set_index("date")
    
    return df


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
    vol_window: int = 10
) -> pd.DataFrame:
    """
    Generate features for a single symbol.
    
    Features:
    - ret1: 1-day return
    - lag_ret1_k: lagged returns
    - SMA_w: simple moving averages
    - EMA_w: exponential moving averages
    - RSI_14: relative strength index
    - vol_10: rolling volatility
    """
    features = pd.DataFrame(index=df.index)
    prefix = f"{symbol}_" if symbol else ""
    
    close = df["close"]
    
    # Returns
    ret1 = compute_returns(close, 1)
    features[f"{prefix}ret1"] = ret1
    
    # Lagged returns
    for lag in include_lags:
        features[f"{prefix}lag_ret1_{lag}"] = ret1.shift(lag)
    
    # SMA
    for w in sma_windows:
        features[f"{prefix}SMA_{w}"] = compute_sma(close, w)
    
    # EMA
    for w in ema_windows:
        features[f"{prefix}EMA_{w}"] = compute_ema(close, w)
    
    # RSI
    features[f"{prefix}RSI_{rsi_window}"] = compute_rsi(close, rsi_window)
    
    # Volatility
    features[f"{prefix}vol_{vol_window}"] = compute_volatility(ret1, vol_window)
    
    # Price level (normalized by SMA for scale invariance)
    sma_20 = compute_sma(close, 20)
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
    
    # Load target symbol
    target_df = load_price_data(session, target_symbol, start_date, end_date)
    
    if target_df.empty:
        logger.error(f"No price data for target symbol {target_symbol}")
        return pd.DataFrame(), pd.Series()
    
    logger.info(f"Target symbol {target_symbol}: {len(target_df)} bars")
    
    # Load other symbols
    other_dfs = {}
    for symbol in symbols:
        if symbol != target_symbol:
            df = load_price_data(session, symbol, start_date, end_date)
            if not df.empty:
                other_dfs[symbol] = df
                logger.info(f"Symbol {symbol}: {len(df)} bars")
    
    # Align to target calendar
    aligned = align_to_target_calendar(target_df, other_dfs, max_ffill=max_ffill)
    
    # Generate features for target
    all_features = generate_symbol_features(target_df, target_symbol)
    
    # Generate features for other symbols
    for symbol, df in aligned.items():
        if not df.empty:
            symbol_features = generate_symbol_features(df, symbol)
            all_features = all_features.join(symbol_features, how="left")
    
    # Load and join sentiment data
    sentiment_df = load_sentiment_data(session, start_date, end_date)
    
    # Build sentiment features as separate Series, then concat (avoids fragmentation warning)
    sentiment_parts = []
    
    if not sentiment_df.empty:
        # Reindex sentiment to target calendar
        sentiment_aligned = sentiment_df.reindex(target_df.index)
        sentiment_aligned = sentiment_aligned.ffill(limit=max_ffill)
        
        sentiment_parts.append(
            sentiment_aligned["sentiment_index"].fillna(settings.sentiment_missing_fill).rename("sentiment__index")
        )
        sentiment_parts.append(
            sentiment_aligned["news_count"].fillna(0).rename("sentiment__news_count")
        )
        
        logger.info(f"Sentiment data joined: {sentiment_df.shape[0]} daily records")
    else:
        # No sentiment data - use defaults
        sentiment_parts.append(
            pd.Series(settings.sentiment_missing_fill, index=all_features.index, name="sentiment__index")
        )
        sentiment_parts.append(
            pd.Series(0, index=all_features.index, name="sentiment__news_count")
        )
        logger.warning("No sentiment data available, using default values")
    
    # Concat all at once to avoid fragmentation
    all_features = pd.concat([all_features] + sentiment_parts, axis=1)
    
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
    X = X.fillna(0)
    
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
    "DBA":        "Agricultural Commodity ETF",
    "DBB":        "Base Metals ETF",
    "DBC":        "Broad Commodity ETF",
}


def _humanize_symbol(sym: str) -> str:
    """Convert ticker to a user-facing label."""
    if not sym:
        return ""
    if sym in _SYMBOL_LABEL_MAP:
        return _SYMBOL_LABEL_MAP[sym]
    # Strip yfinance suffixes for a cleaner display
    return sym.replace("=F", "").replace("^", "").replace("-Y.NYB", " Index")


_INDICATOR_LABEL_MAP: dict[str, str] = {
    "ret1":            "1-day return",
    "ret2":            "2-day return",
    "ret5":            "5-day return",
    "ret10":           "10-day return",
    "RSI_14":          "14-day RSI (momentum)",
    "SMA_5":           "5-day moving average",
    "SMA_10":          "10-day moving average",
    "SMA_20":          "20-day moving average",
    "EMA_5":           "5-day exponential MA",
    "EMA_10":          "10-day exponential MA",
    "EMA_20":          "20-day exponential MA",
    "vol_5":           "5-day volatility",
    "vol_10":          "10-day volatility",
    "vol_20":          "20-day volatility",
    "price_sma_ratio": "Price vs. 20-day MA",
    "target":          "Model target (next-day return)",
}


def _label_indicator(token: str) -> str:
    """Return a friendly label for a bare indicator token."""
    if token in _INDICATOR_LABEL_MAP:
        return _INDICATOR_LABEL_MAP[token]
    # Lagged return — e.g. lag_ret1_2 → "return lag 2 days"
    if token.startswith("lag_ret1_"):
        n = token.replace("lag_ret1_", "")
        return f"return lag {n} days"
    if token.startswith("lag_ret2_"):
        n = token.replace("lag_ret2_", "")
        return f"2-day return lag {n}"
    # Embedding features
    if token.startswith("emb_") or token.startswith("news_emb_"):
        return f"news embedding component {token.split('_')[-1]}"
    # Sentiment features
    if token.startswith("sentiment"):
        if "news_count" in token:
            return "daily news volume"
        return "sentiment index"
    # Fallback: prettify by replacing underscores and lower-casing
    return token.replace("_", " ").lower()


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
    # We try the longest matching symbol prefix.
    symbol = ""
    token = feature
    for known in sorted(_SYMBOL_LABEL_MAP.keys(), key=len, reverse=True):
        prefix = f"{known}_"
        if feature.startswith(prefix):
            symbol = known
            token = feature[len(prefix) :]
            break

    if not symbol:
        # Fallback heuristic: features are often serialized as
        # "<TICKER>_<INDICATOR...>". If the prefix looks like a ticker we
        # treat it as the subject even when it's not in _SYMBOL_LABEL_MAP yet.
        import re
        if "_" in feature:
            maybe_sym, rest = feature.split("_", 1)
            if re.fullmatch(r"[A-Z0-9^=.\-]{1,20}", maybe_sym or ""):
                symbol = maybe_sym
                token = rest
        indicator_label = _label_indicator(token)
    else:
        indicator_label = _label_indicator(token)

    # Category classification
    category = "Other"
    t = token.lower()
    if t.startswith("sentiment"):
        category = "Sentiment"
    elif t.startswith("emb_") or t.startswith("news_emb_"):
        category = "Embedding"
    elif t.startswith("rsi") or t.startswith("ret") or t.startswith("lag_ret"):
        category = "Momentum"
    elif t.startswith("vol"):
        category = "Volatility"
    elif t.startswith("sma") or t.startswith("ema") or "price_sma" in t:
        category = "Trend"
    elif symbol and symbol != "HG=F":
        category = "Macro" if symbol in {"DX-Y.NYB", "^TNX", "^VIX", "^GSPC", "^DJI", "^IXIC"} else "Sector"

    # Time horizon extraction (e.g. RSI_14, vol_10)
    time_horizon = ""
    import re
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

    # Keep label concise
    if len(label) > 48:
        label = label[:45] + "…"

    return {
        "label": label,
        "description": description,
        "category": category,
        "time_horizon": time_horizon,
    }

