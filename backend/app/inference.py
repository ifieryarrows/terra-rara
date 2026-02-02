"""
Inference module: Live prediction and snapshot generation.

Handles:
- Loading trained model
- Running inference on current data
- Generating analysis report
- Saving snapshots for caching
"""

import json
import logging
import re

# Suppress httpx request logging to prevent API keys in URLs from appearing in logs
logging.getLogger("httpx").setLevel(logging.WARNING)
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import PriceBar, DailySentiment, AnalysisSnapshot, NewsArticle, NewsSentiment
from app.settings import get_settings
from app.features import (
    load_price_data,
    load_sentiment_data,
    generate_symbol_features,
    align_to_target_calendar,
    get_feature_descriptions,
)
from app.ai_engine import load_model, load_model_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Feature Alignment Helpers (Train/Inference compatibility)
# =============================================================================

def _sanitize_symbol(sym: str) -> str:
    """Convert symbol to safe column prefix (HG=F -> HG_F)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", sym).strip("_")


def _rename_sanitized_to_raw(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """
    Rename sanitized column prefixes back to raw symbol names.
    Example: HG_F_ret1 -> HG=F_ret1
    """
    rename_map = {}
    cols = list(df.columns)
    
    for sym in symbols:
        sanitized = _sanitize_symbol(sym)
        if sanitized == sym:
            continue  # No change needed
        
        sanitized_prefix = sanitized + "_"
        raw_prefix = sym + "_"
        
        for col in cols:
            if col.startswith(sanitized_prefix):
                new_name = raw_prefix + col[len(sanitized_prefix):]
                rename_map[col] = new_name
    
    if rename_map:
        logger.debug(f"Renaming {len(rename_map)} columns from sanitized to raw")
        return df.rename(columns=rename_map)
    return df


def _align_features_to_model(df: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    """
    Align DataFrame columns to match model's expected feature names.
    - Missing features are filled with 0.0
    - Extra features are dropped
    - Column order matches expected_features
    """
    if not expected_features:
        logger.warning("No expected features provided; skipping alignment")
        return df
    
    present = set(df.columns)
    expected = set(expected_features)
    
    missing = expected - present
    extra = present - expected
    
    if missing or extra:
        logger.info(
            f"Feature alignment: expected={len(expected_features)} present={len(df.columns)} "
            f"missing={len(missing)} extra={len(extra)}"
        )
        if missing:
            logger.debug(f"Missing features (first 10): {list(missing)[:10]}")
        if extra:
            logger.debug(f"Extra features (first 10): {list(extra)[:10]}")
    
    return df.reindex(columns=expected_features, fill_value=0.0)


def get_current_price(session: Session, symbol: str) -> Optional[float]:
    """
    Get the current price for a symbol.
    
    Priority:
    1. Twelve Data API (most reliable, no rate limit issues)
    2. yfinance live data (15-min delayed)
    3. Database fallback
    """
    import httpx
    import yfinance as yf
    from app.settings import get_settings
    
    settings = get_settings()
    
    # Try Twelve Data first (for XCU/USD copper)
    if settings.twelvedata_api_key:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    "https://api.twelvedata.com/price",
                    params={
                        "symbol": "XCU/USD",
                        "apikey": settings.twelvedata_api_key,
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    price = data.get("price")
                    if price:
                        logger.info(f"Using Twelve Data price for copper: ${float(price):.4f}")
                        return float(price)
        except Exception as e:
            from app.settings import mask_api_key
            logger.debug(f"Twelve Data price fetch failed: {mask_api_key(str(e))}")
    
    # Try yfinance as fallback
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        live_price = info.get('regularMarketPrice') or info.get('currentPrice')
        if live_price is not None:
            logger.info(f"Using yfinance price for {symbol}: ${live_price:.4f}")
            return float(live_price)
    except Exception as e:
        logger.debug(f"yfinance price fetch failed for {symbol}: {e}")
    
    # Final fallback to database
    latest = session.query(PriceBar).filter(
        PriceBar.symbol == symbol
    ).order_by(PriceBar.date.desc()).first()
    
    if latest:
        logger.info(f"Using DB price for {symbol}: ${latest.close:.4f}")
        return latest.close
    
    return None


def get_current_sentiment(session: Session) -> Optional[float]:
    """Get the most recent daily sentiment index."""
    latest = session.query(DailySentiment).order_by(
        DailySentiment.date.desc()
    ).first()
    
    return latest.sentiment_index if latest else None


def get_data_quality_stats(
    session: Session,
    symbol: str,
    days: int = 7
) -> dict:
    """Get data quality statistics for the report."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    
    # News count
    news_count = session.query(func.count(NewsArticle.id)).filter(
        NewsArticle.published_at >= cutoff
    ).scalar()
    
    # Scored news count
    scored_count = session.query(func.count(NewsSentiment.id)).join(
        NewsArticle,
        NewsSentiment.news_article_id == NewsArticle.id
    ).filter(NewsArticle.published_at >= cutoff).scalar()
    
    # Price bar coverage
    expected_days = days
    actual_bars = session.query(func.count(PriceBar.id)).filter(
        PriceBar.symbol == symbol,
        PriceBar.date >= cutoff
    ).scalar()
    
    # Account for weekends (roughly 5/7 of days should have bars)
    expected_trading_days = int(expected_days * 5 / 7)
    coverage_pct = min(100, int(actual_bars / max(1, expected_trading_days) * 100))
    
    # Missing days calculation
    missing_days = max(0, expected_trading_days - actual_bars)
    
    return {
        "news_count_7d": news_count,
        "scored_count_7d": scored_count,
        "missing_days": missing_days,
        "coverage_pct": coverage_pct,
    }


def calculate_confidence_band(
    session: Session,
    symbol: str,
    predicted_price: float,
    lookback_days: int = 30
) -> tuple[float, float]:
    """
    Calculate confidence band based on historical prediction errors.
    Simple approach: use historical return volatility.
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Load recent prices
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    
    prices = session.query(PriceBar.close).filter(
        PriceBar.symbol == symbol,
        PriceBar.date >= cutoff
    ).order_by(PriceBar.date.asc()).all()
    
    if len(prices) < 10:
        # Not enough data, use 5% band
        return predicted_price * 0.95, predicted_price * 1.05
    
    closes = [p[0] for p in prices]
    returns = pd.Series(closes).pct_change().dropna()
    
    # 1 standard deviation of daily returns
    std_ret = returns.std()
    
    # Confidence band: ±1 std
    lower = predicted_price * (1 - std_ret)
    upper = predicted_price * (1 + std_ret)
    
    return lower, upper


def get_sentiment_label(sentiment_index: float) -> str:
    """Convert sentiment index to label."""
    if sentiment_index > 0.1:
        return "Bullish"
    elif sentiment_index < -0.1:
        return "Bearish"
    else:
        return "Neutral"


def build_features_for_prediction(
    session: Session,
    target_symbol: str,
    feature_names: list[str]
) -> Optional[pd.DataFrame]:
    """
    Build feature vector for live prediction.
    Uses the most recent available data.
    MUST use training_symbols to match the model's training data.
    
    Includes robust alignment to handle:
    - Sanitized vs raw symbol name differences (HG_F vs HG=F)
    - Missing/extra features between training and inference
    """
    settings = get_settings()
    # Use training_symbols (not symbols_list) to match model training
    symbols = settings.training_symbols
    
    # Load recent data (need enough for feature calculation)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=60)  # Need history for indicators
    
    # Load target
    target_df = load_price_data(session, target_symbol, start_date, end_date)
    
    if target_df.empty:
        logger.error(f"No price data for {target_symbol}")
        return None
    
    # Load other symbols
    other_dfs = {}
    for symbol in symbols:
        if symbol != target_symbol:
            df = load_price_data(session, symbol, start_date, end_date)
            if not df.empty:
                other_dfs[symbol] = df
    
    # Align
    aligned = align_to_target_calendar(target_df, other_dfs, max_ffill=3)
    
    # Generate features
    all_features = generate_symbol_features(target_df, target_symbol)
    
    for symbol, df in aligned.items():
        if not df.empty:
            symbol_features = generate_symbol_features(df, symbol)
            all_features = all_features.join(symbol_features, how="left")
    
    # Add sentiment (use concat to avoid fragmentation warning)
    sentiment_df = load_sentiment_data(session, start_date, end_date)
    sentiment_parts = []
    
    if not sentiment_df.empty:
        sentiment_aligned = sentiment_df.reindex(target_df.index).ffill(limit=3)
        sentiment_parts.append(
            sentiment_aligned["sentiment_index"].fillna(settings.sentiment_missing_fill).rename("sentiment__index")
        )
        sentiment_parts.append(
            sentiment_aligned["news_count"].fillna(0).rename("sentiment__news_count")
        )
    else:
        sentiment_parts.append(
            pd.Series(settings.sentiment_missing_fill, index=all_features.index, name="sentiment__index")
        )
        sentiment_parts.append(
            pd.Series(0, index=all_features.index, name="sentiment__news_count")
        )
    
    all_features = pd.concat([all_features] + sentiment_parts, axis=1)
    
    # Get latest row
    latest = all_features.iloc[[-1]].copy()
    
    # STEP 1: Rename sanitized prefixes to raw symbol names if needed
    # This handles cases where feature generation used sanitized names (HG_F)
    # but model was trained with raw names (HG=F)
    all_symbols = [target_symbol] + list(symbols)
    latest = _rename_sanitized_to_raw(latest, all_symbols)
    
    # STEP 2: Align to model's expected features
    # - Missing features get 0.0 (same as missing data handling in training)
    # - Extra features are dropped
    # - Column order matches expected feature_names
    latest = _align_features_to_model(latest, feature_names)
    
    # Ensure float dtype for XGBoost
    latest = latest.astype(float)
    
    return latest



def generate_analysis_report(
    session: Session,
    target_symbol: str = "HG=F"
) -> Optional[dict]:
    """
    Generate a full analysis report.
    
    Returns:
        Dict with analysis data matching the API schema
    """
    settings = get_settings()
    
    # Load model
    model = load_model(target_symbol)
    if model is None:
        logger.error(f"No model found for {target_symbol}")
        return None
    
    # Load metadata
    metadata = load_model_metadata(target_symbol)
    features = metadata.get("features", [])
    importance = metadata.get("importance", [])
    metrics = metadata.get("metrics", {})
    
    if not features:
        logger.error("No feature list found for model")
        return None
    
    # CRITICAL: Verify target_type is explicitly set
    # Do NOT guess - wrong interpretation inverts prediction meaning
    target_type = metrics.get("target_type")
    if target_type not in ("simple_return", "log_return", "price"):
        logger.error(f"Invalid or missing target_type in model metadata: {target_type}")
        logger.error("Model must be retrained with explicit target_type. Cannot generate forecast.")
        return None
    
    # Get current price (for display - may be live yfinance or DB fallback)
    current_price = get_current_price(session, target_symbol)
    price_source = "yfinance_live"  # Default assumption
    
    if current_price is None:
        logger.error(f"No price data for {target_symbol}")
        return None
    
    # Get latest DB close price for prediction base (baseline_price)
    # Model predicts based on historical closes, not intraday prices
    latest_bar = session.query(PriceBar).filter(
        PriceBar.symbol == target_symbol
    ).order_by(PriceBar.date.desc()).first()
    
    if latest_bar:
        baseline_price = latest_bar.close
        baseline_price_date = latest_bar.date.strftime("%Y-%m-%d") if latest_bar.date else None
        price_source = "yfinance_db_close"
    else:
        baseline_price = current_price
        baseline_price_date = None
        price_source = "yfinance_live_fallback"
    
    # Get current sentiment
    current_sentiment = get_current_sentiment(session)
    if current_sentiment is None:
        current_sentiment = 0.0
    
    # Build features for prediction
    X = build_features_for_prediction(session, target_symbol, features)
    if X is None or X.empty:
        logger.error("Could not build features for prediction")
        return None
    
    # Make prediction
    dmatrix = xgb.DMatrix(X, feature_names=features)
    model_output = float(model.predict(dmatrix)[0])
    
    logger.info(f"Model prediction: raw_output={model_output:.6f}, target_type={target_type}")
    
    # Compute predicted_return and predicted_price based on target_type
    if target_type == "simple_return":
        predicted_return = model_output
        predicted_price = baseline_price * (1 + predicted_return)
    elif target_type == "log_return":
        import math
        predicted_return = math.exp(model_output) - 1
        predicted_price = baseline_price * math.exp(model_output)
    elif target_type == "price":
        predicted_price = model_output
        predicted_return = (predicted_price / baseline_price) - 1 if baseline_price > 0 else 0
    
    # Validate prediction (do not clamp by default - expose issues)
    prediction_invalid = False
    if predicted_return < -1.0:
        logger.error(f"Invalid prediction: return {predicted_return:.4f} < -100%")
        prediction_invalid = True
    if predicted_price <= 0:
        logger.error(f"Invalid prediction: price {predicted_price:.4f} <= 0")
        prediction_invalid = True
    
    if prediction_invalid:
        return None
    
    # Calculate confidence band
    conf_lower, conf_upper = calculate_confidence_band(
        session, target_symbol, predicted_price
    )
    
    # Get data quality
    data_quality = get_data_quality_stats(session, target_symbol)
    
    # Build influencer descriptions
    descriptions = get_feature_descriptions()
    top_influencers = []
    
    for item in importance[:10]:
        feat = item["feature"]
        # Try to find description
        desc = None
        for key, value in descriptions.items():
            if key in feat:
                desc = value
                break
        
        if desc is None:
            # Build from feature name
            desc = feat.replace("_", " ").replace("  ", " ").title()
        
        top_influencers.append({
            "feature": feat,
            "importance": item["importance"],
            "description": desc,
        })
    
    # Build report with explicit baseline_price and target_type
    report = {
        "symbol": target_symbol,
        "current_price": round(current_price, 4),
        "baseline_price": round(baseline_price, 4),
        "baseline_price_date": baseline_price_date,
        "predicted_return": round(predicted_return, 6),
        "predicted_return_pct": round(predicted_return * 100, 2),
        "predicted_price": round(predicted_price, 4),
        "target_type": target_type,
        "price_source": price_source,
        "confidence_lower": round(conf_lower, 4),
        "confidence_upper": round(conf_upper, 4),
        "sentiment_index": round(current_sentiment, 4),
        "sentiment_label": get_sentiment_label(current_sentiment),
        "top_influencers": top_influencers,
        "data_quality": data_quality,
        "training_symbols_hash": settings.training_symbols_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    return report


def save_analysis_snapshot(
    session: Session,
    report: dict,
    symbol: str
) -> AnalysisSnapshot:
    """Save analysis report as a snapshot."""
    now = datetime.now(timezone.utc)
    
    # Check for existing snapshot today
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    existing = session.query(AnalysisSnapshot).filter(
        AnalysisSnapshot.symbol == symbol,
        AnalysisSnapshot.as_of_date >= today_start
    ).first()
    
    if existing:
        # Update existing
        existing.report_json = report
        existing.generated_at = now
        snapshot = existing
    else:
        # Create new
        snapshot = AnalysisSnapshot(
            symbol=symbol,
            as_of_date=now,
            report_json=report,
            generated_at=now,
        )
        session.add(snapshot)
    
    session.commit()
    logger.info(f"Snapshot saved for {symbol}")
    
    return snapshot


def get_latest_snapshot(
    session: Session,
    symbol: str,
    max_age_minutes: int = 30
) -> Optional[dict]:
    """
    Get the latest snapshot if it's fresh enough.
    
    Returns:
        Report dict if fresh snapshot exists, None otherwise
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    
    snapshot = session.query(AnalysisSnapshot).filter(
        AnalysisSnapshot.symbol == symbol,
        AnalysisSnapshot.generated_at >= cutoff
    ).order_by(AnalysisSnapshot.generated_at.desc()).first()
    
    if snapshot:
        return snapshot.report_json
    
    return None


def get_any_snapshot(
    session: Session,
    symbol: str
) -> Optional[dict]:
    """Get the most recent snapshot regardless of age."""
    snapshot = session.query(AnalysisSnapshot).filter(
        AnalysisSnapshot.symbol == symbol
    ).order_by(AnalysisSnapshot.generated_at.desc()).first()
    
    if snapshot:
        return snapshot.report_json
    
    return None

