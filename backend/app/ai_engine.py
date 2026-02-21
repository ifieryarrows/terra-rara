"""
AI Engine: LLM sentiment scoring (with FinBERT fallback) + XGBoost training.

Sentiment Analysis:
    Primary: Gemini LLM with copper-specific context (1M token batch)
    Fallback: FinBERT for generic financial sentiment

Usage:
    python -m app.ai_engine --run-all --target-symbol HG=F
    python -m app.ai_engine --score-only
    python -m app.ai_engine --train-only --target-symbol HG=F
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Suppress httpx request logging to prevent API keys in URLs from appearing in logs
logging.getLogger("httpx").setLevel(logging.WARNING)

import httpx

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.db import SessionLocal, init_db
from app.models import NewsArticle, NewsSentiment, DailySentiment, PriceBar
from app.settings import get_settings
from app.features import build_feature_matrix, get_feature_descriptions
from app.lock import pipeline_lock
from app.async_bridge import run_async_from_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_FINBERT_OUTPUT_LOGGED = False


# =============================================================================
# FinBERT Sentiment Scoring
# =============================================================================


def _neutral_finbert_score() -> dict:
    """Neutral fallback score used when FinBERT output is invalid or unavailable."""
    return {
        "prob_positive": 0.33,
        "prob_neutral": 0.34,
        "prob_negative": 0.33,
        "score": 0.0,
    }


def _normalize_finbert_output(raw_output: Any) -> list[dict]:
    """
    Normalize FinBERT output into a flat ``list[dict]``.

    Supported raw formats:
    - list[dict]
    - list[list[dict]]
    - dict
    - JSON string of any of the above
    """
    output = raw_output
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ValueError("FinBERT output is not valid JSON") from exc

    if isinstance(output, dict):
        output = [output]

    if not isinstance(output, list):
        raise TypeError(f"Unsupported FinBERT output type: {type(output).__name__}")

    normalized: list[dict] = []
    for item in output:
        if isinstance(item, dict):
            normalized.append(item)
            continue

        if isinstance(item, list):
            normalized.extend([child for child in item if isinstance(child, dict)])
            continue

        logger.debug("Skipping unsupported FinBERT output item type: %s", type(item).__name__)

    return normalized


def _log_finbert_output_once(raw_output: Any) -> None:
    """Log one representative FinBERT output shape for debugging parser mismatches."""
    global _FINBERT_OUTPUT_LOGGED
    if _FINBERT_OUTPUT_LOGGED:
        return

    first_item = raw_output
    if isinstance(raw_output, list) and raw_output:
        first_item = raw_output[0]

    preview = repr(first_item)
    if len(preview) > 220:
        preview = f"{preview[:220]}..."

    logger.info(
        "FinBERT output sample: type=%s first_item_type=%s first_item=%s",
        type(raw_output).__name__,
        type(first_item).__name__,
        preview,
    )
    _FINBERT_OUTPUT_LOGGED = True

def get_finbert_pipeline():
    """
    Load FinBERT model pipeline.
    Lazy loading to avoid import overhead when not needed.
    """
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    
    model_name = "ProsusAI/finbert"
    
    logger.info(f"Loading FinBERT model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        max_length=512,
        truncation=True
    )
    
    logger.info("FinBERT model loaded")
    return pipe


def score_text_with_finbert(
    pipe,
    text: str
) -> dict:
    """
    Score a single text with FinBERT.
    
    Returns:
        Dict with prob_positive, prob_neutral, prob_negative, score
    """
    if not text or len(text.strip()) < 10:
        return _neutral_finbert_score()
    
    # Truncate long text
    text = text[:1000]
    
    try:
        raw_output = pipe(text)
        _log_finbert_output_once(raw_output)
        results = _normalize_finbert_output(raw_output)
        if not results:
            logger.warning("FinBERT output normalized to empty list, using neutral fallback")
            return _neutral_finbert_score()

        probs: dict[str, float] = {}
        for row in results:
            label = row.get("label")
            score = row.get("score")
            if not isinstance(label, str):
                continue
            try:
                probs[label.lower()] = float(score)
            except (TypeError, ValueError):
                continue

        required_labels = {"positive", "neutral", "negative"}
        if not required_labels.issubset(probs):
            logger.warning(
                "FinBERT output missing labels. found=%s expected=%s",
                sorted(probs.keys()),
                sorted(required_labels),
            )
            return _neutral_finbert_score()

        prob_pos = probs["positive"]
        prob_neu = probs["neutral"]
        prob_neg = probs["negative"]
        
        # Derived score: positive - negative (range: -1 to 1)
        score = prob_pos - prob_neg
        
        return {
            "prob_positive": prob_pos,
            "prob_neutral": prob_neu,
            "prob_negative": prob_neg,
            "score": score
        }
        
    except Exception as e:
        logger.warning(f"FinBERT scoring error: {e}")
        return _neutral_finbert_score()


# =============================================================================
# LLM Sentiment Scoring (Primary - Gemini)
# =============================================================================

# Copper-specific system prompt for LLM sentiment analysis
LLM_SENTIMENT_SYSTEM_PROMPT = """You are an independent, neutral Copper (HG=F) market intelligence analyst. You do not represent a buy-side or sell-side desk, you do not take positions, and you do not provide trade recommendations. Your sole objective is to quantify the immediate directional news impulse on COMEX Copper Futures (HG=F) over the next 1–5 trading days, based strictly on the text provided.

TASK
You will receive a JSON array of news items, e.g.:
[{"id": 1, "headline": "..."}]
Items may include additional fields (e.g., "summary", "body", "source", "timestamp"). Use only the text present in each item. Do not browse, do not rely on outside knowledge, and do not assume missing details.

WHAT TO SCORE
For each item, output a single net impact score representing the DIRECT effect on copper futures price:
- +1.0 = very strongly bullish impulse
-  0.0 = neutral / unclear / not copper-relevant
- -1.0 = very strongly bearish impulse
Score must be a float in [-1.0, 1.0].

EVALUATION FRAMEWORK (only if explicitly implied by the item text)
1) Supply availability (typically bullish when constrained)
   - Bullish: strikes, accidents, shutdowns, flooding, power shortages, permitting delays, export bans, lower ore grades/grade decline, smelter/refinery outages, logistics constraints reducing supply.
   - Bearish: ramp-ups, new capacity, higher output guidance, disruptions resolved, easing constraints increasing supply.
2) Demand outlook (bullish when boosted; bearish when destroyed)
   - Bullish: explicit China demand support (property/infrastructure/grid stimulus, EV/grid buildout) with clear copper linkage.
   - Bearish: recession risk, manufacturing contraction, construction/property weakness, credit tightening explicitly reducing activity.
3) Inventories / physical tightness (high signal if specific)
   - Bullish: LME/COMEX/SHFE drawdowns; explicit "tightness"/backwardation tied to copper.
   - Bearish: inventory builds; explicit "glut"/surplus/contango tied to copper.
4) Macro FX / rates (only when explicitly stated)
   - Bullish: USD weakness / DXY down explicitly cited as supportive for commodities.
   - Bearish: USD strength / DXY up explicitly cited; restrictive rates explicitly hurting metals demand.
5) Substitution / policy (only if clearly connected to copper usage)
   - Bearish: explicit substitution from copper to aluminum with meaningful scale.
   - Bullish: policy/capex explicitly increasing copper intensity (grid/electrification) per the text.

CONFLICTS AND AMBIGUITY
- If bullish and bearish elements coexist, output ONE net score reflecting the more direct, immediate, copper-specific channel.
- If it's company-specific but not clearly linked to copper supply/demand/inventories/USD, keep the score near 0.
- If details are vague (no scale, timing, location), reduce magnitude.
- If it clearly states resolution of a prior disruption, treat as bearish versus prior tightness.

MAGNITUDE CALIBRATION (symmetric for negatives)
- 0.05–0.20: weak/indirect/uncertain linkage; generic market chatter; minimal specifics
- 0.25–0.45: moderately copper-relevant with some specificity
- 0.50–0.70: direct driver with clear mechanism and meaningful scale
- 0.75–1.00: major, explicit, time-sensitive shock (large supply cut/strike escalation, sharp stock move, strong demand policy)

OUTPUT FORMAT (STRICT)
Return ONLY a raw JSON array. No markdown, no extra text.
Each input item must yield exactly one output object with the matching id, in the same order:
{
  "id": <MATCHING_INPUT_ID>,
  "score": <FLOAT_BETWEEN_-1_AND_1>,
  "reasoning": "<MAX_15_WORDS, plain English, mechanism-based>"
}
Rules:
- Do not skip any IDs.
- Do not add extra keys.
- Reasoning must be ≤15 words, English only, single concise mechanism statement.
- Use standard decimals (e.g., -0.4, 0.15, 1.0); no NaN, no scientific notation."""


async def score_batch_with_llm(
    articles: list[dict],
) -> list[dict]:
    """
    Score a batch of articles using LLM (Gemini via OpenRouter).
    
    Args:
        articles: List of dicts with 'id', 'title', 'description'
        
    Returns:
        List of dicts with 'id', 'score', 'reasoning', 'prob_positive', 'prob_neutral', 'prob_negative'
        
    Raises:
        Exception on API error or JSON parse failure
    """
    settings = get_settings()
    
    if not settings.openrouter_api_key:
        raise RuntimeError("OpenRouter API key not configured")
    
    # Build articles text for prompt
    articles_text = "\n".join([
        f"{i+1}. [ID:{a['id']}] {a['title']}" + (f" - {a['description'][:200]}" if a.get('description') else "")
        for i, a in enumerate(articles)
    ])
    
    user_prompt = f"""Score these {len(articles)} news articles for copper market sentiment.

Articles:
{articles_text}

Return ONLY a valid JSON array with this exact structure (no markdown code blocks):
[
  {{"id": <article_id>, "score": <float from -1.0 to 1.0>, "reasoning": "<brief explanation>"}},
  ...
]

Rules:
- score: -1.0 (very bearish) to +1.0 (very bullish), 0 = neutral
- reasoning: 1 sentence max explaining the copper market impact
- Include ALL {len(articles)} articles in your response"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://copper-mind.vercel.app",
                "X-Title": "CopperMind Sentiment Analysis",
            },
            json={
                "model": settings.llm_sentiment_model,
                "messages": [
                    {"role": "system", "content": LLM_SENTIMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 2000,
                "temperature": 0.3,  # Lower temperature for consistent scoring
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if not content:
            raise RuntimeError("Empty response from LLM")
        
        # Clean up response - remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            # Remove ```json and ``` markers
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        # Parse JSON
        try:
            results = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"LLM JSON parse error: {e}\nContent: {content[:500]}")
            raise
        
        # Validate and enrich results
        enriched = []
        for item in results:
            score = float(item.get("score", 0))
            # Clamp score to [-1, 1]
            score = max(-1.0, min(1.0, score))
            
            # Derive probabilities from score
            # score = prob_positive - prob_negative
            # Assume prob_neutral is inverse of confidence
            confidence = abs(score)
            if score > 0:
                prob_positive = 0.33 + (confidence * 0.67)
                prob_negative = 0.33 - (confidence * 0.33)
                prob_neutral = 1.0 - prob_positive - prob_negative
            elif score < 0:
                prob_negative = 0.33 + (confidence * 0.67)
                prob_positive = 0.33 - (confidence * 0.33)
                prob_neutral = 1.0 - prob_positive - prob_negative
            else:
                prob_positive = 0.33
                prob_neutral = 0.34
                prob_negative = 0.33
            
            enriched.append({
                "id": item.get("id"),
                "score": score,
                "reasoning": item.get("reasoning", ""),
                "prob_positive": round(prob_positive, 4),
                "prob_neutral": round(prob_neutral, 4),
                "prob_negative": round(prob_negative, 4),
            })
        
        return enriched


def score_batch_with_finbert(articles: list) -> list[dict]:
    """
    Score articles with FinBERT (fallback when LLM fails).
    
    Args:
        articles: List of NewsArticle ORM objects
        
    Returns:
        List of dicts with scoring results
    """
    pipe = get_finbert_pipeline()
    results = []
    
    for article in articles:
        text = f"{article.title} {article.description or ''}"
        scores = score_text_with_finbert(pipe, text)
        
        results.append({
            "id": article.id,
            "score": scores["score"],
            "reasoning": None,  # FinBERT doesn't provide reasoning
            "prob_positive": scores["prob_positive"],
            "prob_neutral": scores["prob_neutral"],
            "prob_negative": scores["prob_negative"],
            "model_name": "ProsusAI/finbert",
        })
    
    return results


def score_unscored_articles(
    session: Session,
    chunk_size: int = 20
) -> int:
    """
    Score all articles that don't have sentiment scores yet.
    
    Strategy:
    - Primary: LLM (Gemini) with copper-specific context
    - Fallback: FinBERT per chunk if LLM fails
    - Chunk size: 20 articles for error isolation
    - Rate limiting: 2 second delay between chunks
    
    Returns:
        Number of articles scored
    """
    settings = get_settings()
    
    # Find unscored articles
    unscored = session.query(NewsArticle).outerjoin(
        NewsSentiment,
        NewsArticle.id == NewsSentiment.news_article_id
    ).filter(NewsSentiment.id.is_(None)).all()
    
    if not unscored:
        logger.info("No unscored articles found")
        return 0
    
    logger.info(f"Found {len(unscored)} unscored articles")
    
    scored_count = 0
    total_chunks = (len(unscored) + chunk_size - 1) // chunk_size
    
    # Process in chunks
    for chunk_idx in range(0, len(unscored), chunk_size):
        chunk = unscored[chunk_idx:chunk_idx + chunk_size]
        chunk_num = chunk_idx // chunk_size + 1
        
        logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} articles)")
        
        # Prepare articles for LLM
        articles_data = [
            {"id": a.id, "title": a.title, "description": a.description}
            for a in chunk
        ]
        
        results = None
        used_model = settings.llm_sentiment_model
        
        # Try LLM first
        if settings.openrouter_api_key:
            try:
                # Bridge async scoring into sync callers without nested-loop errors.
                results = run_async_from_sync(score_batch_with_llm, articles_data)
                logger.info(f"LLM scored chunk {chunk_num} successfully")
            except Exception as e:
                logger.warning(f"LLM scoring failed for chunk {chunk_num}, falling back to FinBERT: {e}")
                results = None
        
        # Fallback to FinBERT if LLM failed or not configured
        if results is None:
            logger.info(f"Using FinBERT fallback for chunk {chunk_num}")
            results = score_batch_with_finbert(chunk)
            used_model = "ProsusAI/finbert"
        
        # Create a lookup for results
        results_by_id = {r["id"]: r for r in results}
        
        # Save to database
        for article in chunk:
            result = results_by_id.get(article.id)
            if not result:
                # If article not in results (shouldn't happen), use neutral
                logger.warning(f"No result for article {article.id}, using neutral")
                result = {
                    "score": 0.0,
                    "reasoning": "Missing from LLM response",
                    "prob_positive": 0.33,
                    "prob_neutral": 0.34,
                    "prob_negative": 0.33,
                }
            
            sentiment = NewsSentiment(
                news_article_id=article.id,
                prob_positive=result["prob_positive"],
                prob_neutral=result["prob_neutral"],
                prob_negative=result["prob_negative"],
                score=result["score"],
                reasoning=result.get("reasoning"),
                model_name=result.get("model_name", used_model),
                scored_at=datetime.now(timezone.utc)
            )
            
            session.add(sentiment)
            scored_count += 1
        
        # Commit after each chunk
        session.commit()
        logger.info(f"Committed chunk {chunk_num}: {len(chunk)} articles")
        
        # Rate limiting: 2 second delay between chunks (except last)
        if chunk_idx + chunk_size < len(unscored):
            logger.debug("Rate limit delay: 2 seconds")
            time.sleep(2)
    
    logger.info(f"Total articles scored: {scored_count}")
    return scored_count


# =============================================================================
# Daily Sentiment Aggregation
# =============================================================================

def aggregate_daily_sentiment(
    session: Session,
    tau_hours: float = 12.0
) -> int:
    """
    Aggregate sentiment scores by day with recency weighting.
    
    Weighting formula: w = exp(-(hours_since_publish) / tau)
    
    Returns:
        Number of days aggregated
    """
    settings = get_settings()
    tau_hours = tau_hours or settings.sentiment_tau_hours
    
    # Get all scored articles with their sentiment
    scored_articles = session.query(
        NewsArticle.published_at,
        NewsSentiment.score,
        NewsSentiment.prob_positive,
        NewsSentiment.prob_neutral,
        NewsSentiment.prob_negative
    ).join(
        NewsSentiment,
        NewsArticle.id == NewsSentiment.news_article_id
    ).all()
    
    if not scored_articles:
        logger.info("No scored articles for aggregation")
        return 0
    
    # Convert to DataFrame
    df = pd.DataFrame(scored_articles, columns=[
        "published_at", "score", "prob_positive", "prob_neutral", "prob_negative"
    ])
    
    # Extract date
    df["date"] = pd.to_datetime(df["published_at"]).dt.normalize()
    
    # Calculate recency weight within each day
    # Higher weight for articles later in the day (closer to market close)
    def calc_weights(group):
        # Hours since start of day
        hours = (group["published_at"] - group["date"]).dt.total_seconds() / 3600
        # Exponential weighting: later = higher weight
        weights = np.exp(hours / tau_hours)
        return weights / weights.sum()  # Normalize
    
    # Group by date and aggregate
    daily_data = []
    
    for date, group in df.groupby("date"):
        weights = calc_weights(group)
        
        # Convert numpy types to native Python types for database compatibility
        daily_data.append({
            "date": date,
            "sentiment_index": float((group["score"] * weights).sum()),
            "news_count": int(len(group)),
            "avg_positive": float(group["prob_positive"].mean()),
            "avg_neutral": float(group["prob_neutral"].mean()),
            "avg_negative": float(group["prob_negative"].mean()),
        })
    
    # Upsert daily sentiments
    count = 0
    for row in daily_data:
        date_dt = row["date"].to_pydatetime()
        if date_dt.tzinfo is None:
            date_dt = date_dt.replace(tzinfo=timezone.utc)
        
        # Check if exists
        existing = session.query(DailySentiment).filter(
            func.date(DailySentiment.date) == func.date(date_dt)
        ).first()
        
        if existing:
            # Update
            existing.sentiment_index = row["sentiment_index"]
            existing.news_count = row["news_count"]
            existing.avg_positive = row["avg_positive"]
            existing.avg_neutral = row["avg_neutral"]
            existing.avg_negative = row["avg_negative"]
            existing.aggregated_at = datetime.now(timezone.utc)
        else:
            # Insert
            daily = DailySentiment(
                date=date_dt,
                sentiment_index=row["sentiment_index"],
                news_count=row["news_count"],
                avg_positive=row["avg_positive"],
                avg_neutral=row["avg_neutral"],
                avg_negative=row["avg_negative"],
                weighting_method="recency_exponential",
                aggregated_at=datetime.now(timezone.utc)
            )
            session.add(daily)
        
        count += 1
    
    session.commit()
    logger.info(f"Aggregated sentiment for {count} days")
    
    return count


# =============================================================================
# XGBoost Model Training
# =============================================================================

def train_xgboost_model(
    session: Session,
    target_symbol: str = "HG=F",
    lookback_days: int = 365,
    validation_days: int = 30,
    early_stopping_rounds: int = 10
) -> Optional[dict]:
    """
    Train XGBoost model for price prediction.
    
    Target: Next-day return (more stationary, avoids direct price level issues)
    
    Returns:
        Dict with model path, metrics, and feature importance
    """
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Build feature matrix
    X, y = build_feature_matrix(
        session,
        target_symbol=target_symbol,
        lookback_days=lookback_days
    )
    
    if X.empty or len(X) < 50:
        logger.error(f"Insufficient data for training: {len(X)} samples")
        return None
    
    # Time-series split: last N days for validation
    split_date = X.index.max() - timedelta(days=validation_days)
    
    train_mask = X.index <= split_date
    val_mask = X.index > split_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    if len(X_train) < 30 or len(X_val) < 5:
        logger.error("Not enough samples for train/val split")
        return None
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # XGBoost parameters - tuned for overfitting prevention
    # With 250 samples / 198 features, we need strong regularization
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 4,              # Shallower trees = less memorization
        "learning_rate": 0.05,       # Slower learning = better generalization
        "subsample": 0.8,
        "colsample_bytree": 0.6,     # Use fewer features per tree
        "min_child_weight": 5,       # Require more samples per leaf
        "reg_alpha": 0.5,            # L1 regularization (sparsity)
        "reg_lambda": 2.0,           # L2 regularization (smoothness)
        "seed": 42,
    }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    # Train with early stopping
    evals = [(dtrain, "train"), (dval, "val")]
    
    logger.info("Training XGBoost model...")
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10  # More frequent logging
    )
    
    # Evaluate
    y_pred_train = model.predict(dtrain)
    y_pred_val = model.predict(dval)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    
    logger.info(f"Training MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}")
    logger.info(f"Validation MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
    
    # Feature importance
    importance = model.get_score(importance_type="gain")
    
    # Sort by importance
    sorted_importance = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Normalize importance
    total_importance = sum(v for _, v in sorted_importance)
    normalized_importance = [
        {"feature": k, "importance": v / total_importance}
        for k, v in sorted_importance
    ]
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_{timestamp}.json"
    model.save_model(str(model_path))
    
    # Save latest symlink/copy
    latest_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.json"
    model.save_model(str(latest_path))
    
    # Save metrics (including training symbols audit)
    # TARGET_TYPE: "simple_return" means model predicts next-day return, not price
    # This MUST be read by inference to correctly compute predicted_price
    metrics = {
        "target_symbol": target_symbol,
        # Target definition audit (prevents semantic confusion)
        "target_type": "simple_return",  # Model predicts: close(t+1)/close(t) - 1
        "target_shift_days": 1,  # Predict 1 day ahead
        "target_definition": "simple_return(close,1).shift(-1)",  # Exact pandas formula
        "baseline_price_source": "yfinance_close",  # Which close normalizes returns
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "best_iteration": model.best_iteration,
        "feature_count": len(feature_names),
        # Audit: which symbols were used for training
        "symbol_set_name": settings.symbol_set,
        "training_symbols": settings.training_symbols,
        "training_symbols_hash": settings.training_symbols_hash,
        "training_symbols_source": settings.training_symbols_source,
    }
    
    metrics_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature names
    features_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.features.json"
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    
    # Save importance (Overwrite to reflect the latest model training)
    importance_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.importance.json"
    with open(importance_path, "w") as f:
        json.dump(normalized_importance, f, indent=2)
    
    logger.info(f"Model saved to: {model_path}")
    
    # Log top influencers
    logger.info("Top 10 feature influencers:")
    descriptions = get_feature_descriptions()
    for item in normalized_importance[:10]:
        feat = item["feature"]
        imp = item["importance"]
        desc = descriptions.get(feat, feat)
        logger.info(f"  {feat}: {imp:.4f} ({desc})")
    
    # Save metadata to database for persistence across HF Space restarts
    try:
        from app.db import SessionLocal
        with SessionLocal() as session:
            save_model_metadata_to_db(
                session=session,
                symbol=target_symbol,
                importance=normalized_importance,
                features=feature_names,
                metrics=metrics,
            )
    except Exception as e:
        logger.warning(f"Could not save model metadata to DB: {e}")
    
    return {
        "model_path": str(model_path),
        "metrics": metrics,
        "top_influencers": normalized_importance[:10],
        "all_features": feature_names,
    }


def load_model(target_symbol: str = "HG=F") -> Optional[xgb.Booster]:
    """Load the latest trained model for a symbol."""
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    
    model_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.json"
    
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None
    
    model = xgb.Booster()
    model.load_model(str(model_path))
    
    return model


def save_model_metadata_to_db(
    session,
    symbol: str,
    importance: list,
    features: list,
    metrics: dict
) -> None:
    """
    Save model metadata to database for persistence across restarts.
    Called after train_model=True pipeline runs.
    """
    from .models import ModelMetadata
    from datetime import datetime
    
    # Try to find existing record
    existing = session.query(ModelMetadata).filter(ModelMetadata.symbol == symbol).first()
    
    if existing:
        existing.importance_json = json.dumps(importance)
        existing.features_json = json.dumps(features)
        existing.metrics_json = json.dumps(metrics)
        existing.trained_at = datetime.utcnow()
        logger.info(f"Updated model metadata in DB for {symbol}")
    else:
        new_record = ModelMetadata(
            symbol=symbol,
            importance_json=json.dumps(importance),
            features_json=json.dumps(features),
            metrics_json=json.dumps(metrics),
        )
        session.add(new_record)
        logger.info(f"Saved new model metadata to DB for {symbol}")
    
    session.commit()


def load_model_metadata_from_db(session, symbol: str) -> dict:
    """
    Load model metadata from database.
    Returns dict with importance, features, metrics or None values if not found.
    """
    from .models import ModelMetadata
    
    metadata = {
        "metrics": None,
        "features": None,
        "importance": None,
    }
    
    record = session.query(ModelMetadata).filter(ModelMetadata.symbol == symbol).first()
    
    if record:
        try:
            if record.importance_json:
                metadata["importance"] = json.loads(record.importance_json)
            if record.features_json:
                metadata["features"] = json.loads(record.features_json)
            if record.metrics_json:
                metadata["metrics"] = json.loads(record.metrics_json)
            logger.info(f"Loaded model metadata from DB for {symbol}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse model metadata from DB: {e}")
    
    return metadata


def load_model_metadata(target_symbol: str = "HG=F") -> dict:
    """
    Load metrics and feature info for a model.
    
    Priority:
    1. Database (survives HF Space restarts)
    2. Local JSON files (fallback for development)
    """
    from app.db import SessionLocal
    
    # Try database first
    try:
        with SessionLocal() as session:
            db_metadata = load_model_metadata_from_db(session, target_symbol)
            if db_metadata.get("importance") and db_metadata.get("features"):
                return db_metadata
    except Exception as e:
        logger.debug(f"Could not load metadata from DB: {e}")
    
    # Fallback to local files
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    
    prefix = f"xgb_{target_symbol.replace('=', '_')}_latest"
    
    metadata = {
        "metrics": None,
        "features": None,
        "importance": None,
    }
    
    # Load metrics
    metrics_path = model_dir / f"{prefix}.metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metadata["metrics"] = json.load(f)
    
    # Load features
    features_path = model_dir / f"{prefix}.features.json"
    if features_path.exists():
        with open(features_path) as f:
            metadata["features"] = json.load(f)
    
    # Load importance
    importance_path = model_dir / f"{prefix}.importance.json"
    if importance_path.exists():
        with open(importance_path) as f:
            metadata["importance"] = json.load(f)
    
    return metadata


# =============================================================================
# Main Entry Point
# =============================================================================

def run_full_pipeline(
    target_symbol: str = "HG=F",
    score_sentiment: bool = True,
    aggregate_sentiment: bool = True,
    train_model: bool = True
) -> dict:
    """
    Run the full AI pipeline.
    
    Returns:
        Dict with results from each stage
    """
    results = {
        "scored_articles": 0,
        "aggregated_days": 0,
        "model_result": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    with SessionLocal() as session:
        if score_sentiment:
            results["scored_articles"] = score_unscored_articles(session)
        
        if aggregate_sentiment:
            results["aggregated_days"] = aggregate_daily_sentiment(session)
        
        if train_model:
            results["model_result"] = train_xgboost_model(
                session,
                target_symbol=target_symbol
            )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run AI pipeline: LLM sentiment scoring (with FinBERT fallback) and XGBoost training"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run full pipeline (score + aggregate + train)"
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Only run sentiment scoring (LLM primary, FinBERT fallback)"
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only run sentiment aggregation"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only run XGBoost training"
    )
    parser.add_argument(
        "--target-symbol",
        type=str,
        default="HG=F",
        help="Target symbol for training (default: HG=F)"
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip pipeline lock (for testing)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine what to run
    score = args.run_all or args.score_only
    aggregate = args.run_all or args.aggregate_only
    train = args.run_all or args.train_only
    
    if not (score or aggregate or train):
        parser.print_help()
        return
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Run pipeline
    def do_run():
        return run_full_pipeline(
            target_symbol=args.target_symbol,
            score_sentiment=score,
            aggregate_sentiment=aggregate,
            train_model=train
        )
    
    if args.no_lock:
        results = do_run()
    else:
        try:
            with pipeline_lock():
                results = do_run()
        except RuntimeError as e:
            logger.error(f"Could not acquire lock: {e}")
            return
    
    # Print summary
    print("\n" + "=" * 50)
    print("AI PIPELINE SUMMARY")
    print("=" * 50)
    
    if score:
        print(f"\nSentiment Scoring: {results['scored_articles']} articles")
    
    if aggregate:
        print(f"Daily Aggregation: {results['aggregated_days']} days")
    
    if train and results.get("model_result"):
        mr = results["model_result"]
        metrics = mr.get("metrics", {})
        print(f"\nModel Training:")
        print(f"  - Validation MAE: {metrics.get('val_mae', 'N/A'):.6f}")
        print(f"  - Validation RMSE: {metrics.get('val_rmse', 'N/A'):.6f}")
        print(f"  - Model saved to: {mr.get('model_path', 'N/A')}")
        
        print("\nTop Influencers:")
        for item in mr.get("top_influencers", [])[:5]:
            print(f"  - {item['feature']}: {item['importance']:.4f}")
    
    print(f"\nTimestamp: {results.get('timestamp', 'N/A')}")


if __name__ == "__main__":
    main()

