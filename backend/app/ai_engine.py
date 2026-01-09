"""
AI Engine: FinBERT sentiment scoring + XGBoost training.

Usage:
    python -m app.ai_engine --run-all --target-symbol HG=F
    python -m app.ai_engine --score-only
    python -m app.ai_engine --train-only --target-symbol HG=F
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# FinBERT Sentiment Scoring
# =============================================================================

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
        return {
            "prob_positive": 0.33,
            "prob_neutral": 0.34,
            "prob_negative": 0.33,
            "score": 0.0
        }
    
    # Truncate long text
    text = text[:1000]
    
    try:
        results = pipe(text)[0]
        
        # Extract probabilities
        probs = {r["label"].lower(): r["score"] for r in results}
        
        prob_pos = probs.get("positive", 0.33)
        prob_neu = probs.get("neutral", 0.34)
        prob_neg = probs.get("negative", 0.33)
        
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
        return {
            "prob_positive": 0.33,
            "prob_neutral": 0.34,
            "prob_negative": 0.33,
            "score": 0.0
        }


def score_unscored_articles(
    session: Session,
    batch_size: int = 32
) -> int:
    """
    Score all articles that don't have sentiment scores yet.
    Idempotent: only scores articles without existing NewsSentiment records.
    
    Returns:
        Number of articles scored
    """
    # Find unscored articles
    unscored = session.query(NewsArticle).outerjoin(
        NewsSentiment,
        NewsArticle.id == NewsSentiment.news_article_id
    ).filter(NewsSentiment.id.is_(None)).all()
    
    if not unscored:
        logger.info("No unscored articles found")
        return 0
    
    logger.info(f"Found {len(unscored)} unscored articles")
    
    # Load model
    pipe = get_finbert_pipeline()
    
    scored_count = 0
    
    # Process in batches
    for i in range(0, len(unscored), batch_size):
        batch = unscored[i:i + batch_size]
        
        for article in batch:
            # Use title + description for scoring
            text = f"{article.title} {article.description or ''}"
            
            scores = score_text_with_finbert(pipe, text)
            
            sentiment = NewsSentiment(
                news_article_id=article.id,
                prob_positive=scores["prob_positive"],
                prob_neutral=scores["prob_neutral"],
                prob_negative=scores["prob_negative"],
                score=scores["score"],
                model_name="ProsusAI/finbert",
                scored_at=datetime.now(timezone.utc)
            )
            
            session.add(sentiment)
            scored_count += 1
        
        # Commit batch
        session.commit()
        logger.info(f"Scored batch {i // batch_size + 1}: {len(batch)} articles")
    
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
    
    # Save metrics
    metrics = {
        "target_symbol": target_symbol,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "best_iteration": model.best_iteration,
        "feature_count": len(feature_names),
    }
    
    metrics_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature names
    features_path = model_dir / f"xgb_{target_symbol.replace('=', '_')}_latest.features.json"
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    
    # Save importance
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


def load_model_metadata(target_symbol: str = "HG=F") -> dict:
    """Load metrics and feature info for a model."""
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
        description="Run AI pipeline: FinBERT scoring and XGBoost training"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run full pipeline (score + aggregate + train)"
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Only run FinBERT scoring"
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

