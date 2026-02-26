"""
SQLAlchemy ORM models for CopperMind.

Tables:
- NewsArticle: Raw news articles with dedup
- PriceBar: OHLCV price data per symbol/date
- NewsSentiment: FinBERT scores per article
- DailySentiment: Aggregated daily sentiment index
- AnalysisSnapshot: Cached analysis reports
"""

from datetime import datetime, timezone
from typing import Optional


def _utcnow() -> datetime:
    """Timezone-aware UTC now, replacing deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc)

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    Index,
    LargeBinary,
    UniqueConstraint,
    JSON,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db import Base


class NewsArticle(Base):
    """
    Raw news articles collected from various sources.
    Dedup key prevents duplicate articles.
    """
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Deduplication
    dedup_key = Column(String(64), unique=True, nullable=False, index=True)
    
    # Content
    title = Column(String(500), nullable=False)
    canonical_title = Column(String(500), nullable=True, index=True)  # For fuzzy dedup
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    url = Column(String(2000), nullable=True)
    
    # Metadata
    source = Column(String(200), nullable=True)
    author = Column(String(200), nullable=True)
    language = Column(String(10), nullable=True, default="en")
    
    # Timestamps
    published_at = Column(DateTime(timezone=True), nullable=False, index=True)
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    
    # Relationships
    sentiment = relationship("NewsSentiment", back_populates="article", uselist=False)
    
    def __repr__(self):
        return f"<NewsArticle(id={self.id}, title='{self.title[:30]}...')>"


class PriceBar(Base):
    """
    Daily OHLCV price data for tracked symbols.
    Unique constraint on (symbol, date) prevents duplicates.
    """
    __tablename__ = "price_bars"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # OHLCV
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    
    # Adjusted close (for splits/dividends)
    adj_close = Column(Float, nullable=True)
    
    # When this record was fetched
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_price_symbol_date"),
        Index("ix_price_symbol_date", "symbol", "date"),
    )
    
    def __repr__(self):
        return f"<PriceBar(symbol={self.symbol}, date={self.date}, close={self.close})>"


class NewsSentiment(Base):
    """
    Sentiment scores for each news article.
    Primary: LLM (OpenRouter structured outputs) with copper-specific context
    Fallback: FinBERT for generic financial sentiment
    One-to-one relationship with NewsArticle.
    """
    __tablename__ = "news_sentiments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    news_article_id = Column(
        Integer,
        ForeignKey("news_articles.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True
    )
    
    # Sentiment probabilities (LLM derives these from score)
    prob_positive = Column(Float, nullable=False)
    prob_neutral = Column(Float, nullable=False)
    prob_negative = Column(Float, nullable=False)
    
    # Sentiment score: -1 (bearish) to +1 (bullish)
    score = Column(Float, nullable=False, index=True)
    
    # LLM reasoning for the score (debug + future UI display)
    reasoning = Column(Text, nullable=True)
    
    # Model info (LLM model or "ProsusAI/finbert" for fallback)
    model_name = Column(String(100), default="google/gemini-2.0-flash-exp:free")
    
    # When scored
    scored_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    
    # Relationship
    article = relationship("NewsArticle", back_populates="sentiment")
    
    def __repr__(self):
        return f"<NewsSentiment(article_id={self.news_article_id}, score={self.score:.3f})>"


class DailySentiment(Base):
    """
    Aggregated daily sentiment index.
    One row per date with weighted average sentiment.
    """
    __tablename__ = "daily_sentiments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    date = Column(DateTime(timezone=True), nullable=False, unique=True, index=True)
    
    # Aggregated sentiment
    sentiment_index = Column(Float, nullable=False)
    
    # Statistics
    news_count = Column(Integer, nullable=False, default=0)
    avg_positive = Column(Float, nullable=True)
    avg_neutral = Column(Float, nullable=True)
    avg_negative = Column(Float, nullable=True)
    
    # Weighting method used
    weighting_method = Column(String(50), default="recency_exponential")
    
    # When aggregated
    aggregated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    
    def __repr__(self):
        return f"<DailySentiment(date={self.date}, index={self.sentiment_index:.3f}, news={self.news_count})>"


class AnalysisSnapshot(Base):
    """
    Cached analysis reports for API responses.
    Enables TTL-based caching and stable responses during pipeline runs.
    """
    __tablename__ = "analysis_snapshots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    symbol = Column(String(20), nullable=False, index=True)
    as_of_date = Column(DateTime(timezone=True), nullable=False)
    
    # Full analysis report as JSON
    report_json = Column(JSON, nullable=False)
    
    # When this snapshot was generated
    generated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
    
    # Model version used
    model_version = Column(String(100), nullable=True)
    
    __table_args__ = (
        UniqueConstraint("symbol", "as_of_date", name="uq_snapshot_symbol_date"),
        Index("ix_snapshot_symbol_generated", "symbol", "generated_at"),
    )
    
    def __repr__(self):
        return f"<AnalysisSnapshot(symbol={self.symbol}, as_of={self.as_of_date})>"


class AICommentary(Base):
    """
    Cached AI commentary generated after pipeline runs.
    One row per symbol, updated after each pipeline execution.
    """
    __tablename__ = "ai_commentaries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    
    # The generated commentary text
    commentary = Column(Text, nullable=False)
    
    # Input data used to generate (for debugging)
    current_price = Column(Float, nullable=True)
    predicted_price = Column(Float, nullable=True)
    predicted_return = Column(Float, nullable=True)
    sentiment_label = Column(String(20), nullable=True)
    
    # AI-determined market stance (BULLISH/NEUTRAL/BEARISH)
    # Generated by having LLM analyze its own commentary
    ai_stance = Column(String(20), nullable=True, default="NEUTRAL")
    
    # When generated
    generated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
    
    # Model used
    model_name = Column(String(100), nullable=True)
    
    def __repr__(self):
        return f"<AICommentary(symbol={self.symbol}, generated_at={self.generated_at})>"


class ModelMetadata(Base):
    """
    Persisted XGBoost model metadata.
    Stores feature importance, features list, and metrics in database
    so they survive HF Space restarts.
    One row per symbol, updated after each model training (train_model=True).
    """
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    
    # Feature importance as JSON [{feature, importance}, ...]
    importance_json = Column(Text, nullable=True)
    
    # Feature names list as JSON ["feature1", "feature2", ...]
    features_json = Column(Text, nullable=True)
    
    # Training metrics as JSON {train_mae, val_mae, etc}
    metrics_json = Column(Text, nullable=True)
    
    # When the model was trained
    trained_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
    
    def __repr__(self):
        return f"<ModelMetadata(symbol={self.symbol}, trained_at={self.trained_at})>"


class PipelineRunMetrics(Base):
    """
    Metrics captured after each pipeline run for monitoring.
    Enables tracking of:
    - Symbol fetch success/failure rates
    - Model training metrics over time
    - Pipeline duration trends
    - Data quality indicators
    """
    __tablename__ = "pipeline_run_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Run identification
    run_id = Column(String(64), nullable=False, unique=True, index=True)
    run_started_at = Column(DateTime(timezone=True), nullable=False, index=True)
    run_completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Duration
    duration_seconds = Column(Float, nullable=True)
    
    # Symbol set info
    symbol_set_name = Column(String(50), nullable=True)  # active/champion/challenger
    symbols_requested = Column(Integer, nullable=True)
    symbols_fetched_ok = Column(Integer, nullable=True)
    symbols_failed = Column(Integer, nullable=True)
    failed_symbols_list = Column(Text, nullable=True)  # JSON array
    
    # Training metrics
    train_mae = Column(Float, nullable=True)
    val_mae = Column(Float, nullable=True)
    train_rmse = Column(Float, nullable=True)
    val_rmse = Column(Float, nullable=True)
    feature_count = Column(Integer, nullable=True)
    train_samples = Column(Integer, nullable=True)
    val_samples = Column(Integer, nullable=True)
    
    # Data quality (legacy - news_articles table)
    news_imported = Column(Integer, nullable=True)
    news_duplicates = Column(Integer, nullable=True)
    price_bars_updated = Column(Integer, nullable=True)
    missing_price_days = Column(Integer, nullable=True)
    
    # Faz 2: Reproducible news pipeline stats
    news_raw_inserted = Column(Integer, nullable=True)
    news_raw_duplicates = Column(Integer, nullable=True)
    news_processed_inserted = Column(Integer, nullable=True)
    news_processed_duplicates = Column(Integer, nullable=True)
    articles_scored_v2 = Column(Integer, nullable=True)
    llm_parse_fail_count = Column(Integer, nullable=True)
    escalation_count = Column(Integer, nullable=True)
    fallback_count = Column(Integer, nullable=True)
    
    # Snapshot info
    snapshot_generated = Column(Boolean, default=False)
    commentary_generated = Column(Boolean, default=False)

    # TFT-ASRO deep learning pipeline stats
    tft_embeddings_computed = Column(Integer, nullable=True)
    tft_trained = Column(Boolean, default=False)
    tft_val_loss = Column(Float, nullable=True)
    tft_sharpe = Column(Float, nullable=True)
    tft_directional_accuracy = Column(Float, nullable=True)
    tft_snapshot_generated = Column(Boolean, default=False)
    
    # Faz 2: News cut-off time
    news_cutoff_time = Column(DateTime(timezone=True), nullable=True)
    
    # Quality state for degraded runs
    quality_state = Column(String(20), nullable=True, default="ok")  # ok/stale/degraded/failed
    
    # Status
    status = Column(String(20), nullable=False, default="running")  # running/success/failed
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<PipelineRunMetrics(run_id={self.run_id}, status={self.status})>"


# =============================================================================
# Faz 2: Reproducible News Pipeline
# =============================================================================

class NewsRaw(Base):
    """
    Ham haber verisi - RSS/API'den geldiği gibi saklanır.
    
    Faz 2: Reproducibility için "golden source".
    
    Dedup stratejisi:
    - url_hash: nullable + partial unique index (WHERE url_hash IS NOT NULL)
    - URL eksikse title-based fallback processed seviyesinde yapılır
    """
    __tablename__ = "news_raw"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # URL (nullable - RSS'te eksik olabilir)
    url = Column(String(2000), nullable=True)
    url_hash = Column(String(64), nullable=True, index=True)  # sha256, partial unique
    
    # Content
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    
    # Metadata
    source = Column(String(200), nullable=True)  # "google_news", "newsapi"
    source_feed = Column(String(500), nullable=True)  # Exact RSS URL or query
    published_at = Column(DateTime(timezone=True), nullable=False, index=True)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Pipeline run tracking (UUID)
    run_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Raw payload (debug/audit)
    raw_payload = Column(JSONB, nullable=True)
    
    # Relationship
    processed_items = relationship("NewsProcessed", back_populates="raw")
    
    def __repr__(self):
        return f"<NewsRaw(id={self.id}, title='{self.title[:30]}...')>"


class NewsProcessed(Base):
    """
    İşlenmiş haber - dedup, cleaning, language filter sonrası.
    
    Faz 2: Sentiment scoring için input.
    
    Dedup stratejisi:
    - dedup_key: NOT NULL + UNIQUE - asıl dedup otoritesi
    - Öncelik: url_hash varsa kullan, yoksa sha256(source + canonical_title_hash)
    """
    __tablename__ = "news_processed"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # FK to raw (RESTRICT - raw silinirse processed da silinmemeli)
    raw_id = Column(
        BigInteger, 
        ForeignKey("news_raw.id", ondelete="RESTRICT"), 
        nullable=False, 
        index=True
    )
    
    # Canonical content
    canonical_title = Column(String(500), nullable=False)
    canonical_title_hash = Column(String(64), nullable=False, index=True)  # sha256
    cleaned_text = Column(Text, nullable=True)  # title + description, cleaned
    
    # Dedup key - ASIL OTORİTE
    dedup_key = Column(String(64), unique=True, nullable=False, index=True)  # sha256
    
    # Language
    language = Column(String(10), nullable=True, default="en")
    language_confidence = Column(Float, nullable=True)
    
    # Processing metadata
    processed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    run_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Future: Tone/Impact scores (Faz 3)
    # tone_score = Column(Float, nullable=True)
    # impact_direction = Column(String(20), nullable=True)  # bullish/bearish/neutral
    
    # Relationship
    raw = relationship("NewsRaw", back_populates="processed_items")
    sentiment_v2_items = relationship("NewsSentimentV2", back_populates="processed")
    
    def __repr__(self):
        return f"<NewsProcessed(id={self.id}, dedup_key='{self.dedup_key[:16]}...')>"


class NewsSentimentV2(Base):
    """
    Commodity-aware sentiment scores generated from news_processed records.
    """

    __tablename__ = "news_sentiments_v2"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    news_processed_id = Column(
        BigInteger,
        ForeignKey("news_processed.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    horizon_days = Column(Integer, nullable=False, default=5)

    label = Column(String(20), nullable=False, index=True)
    impact_score_llm = Column(Float, nullable=False)
    confidence_llm = Column(Float, nullable=False)
    confidence_calibrated = Column(Float, nullable=False, index=True)
    relevance_score = Column(Float, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    rule_sign = Column(Integer, nullable=False)
    final_score = Column(Float, nullable=False, index=True)

    finbert_pos = Column(Float, nullable=False)
    finbert_neu = Column(Float, nullable=False)
    finbert_neg = Column(Float, nullable=False)

    reasoning_json = Column(Text, nullable=True)
    model_fast = Column(String(100), nullable=True)
    model_reliable = Column(String(100), nullable=True)
    scored_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)

    processed = relationship("NewsProcessed", back_populates="sentiment_v2_items")

    __table_args__ = (
        UniqueConstraint("news_processed_id", "horizon_days", name="uq_news_sentiments_v2_processed_horizon"),
        Index("ix_news_sentiments_v2_processed_scored", "news_processed_id", "scored_at"),
    )

    def __repr__(self):
        return (
            "<NewsSentimentV2(processed_id="
            f"{self.news_processed_id}, horizon_days={self.horizon_days}, final_score={self.final_score:.3f})>"
        )


class DailySentimentV2(Base):
    """
    Daily aggregate sentiment generated from NewsSentimentV2.
    """

    __tablename__ = "daily_sentiments_v2"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(DateTime(timezone=True), nullable=False, unique=True, index=True)

    sentiment_index = Column(Float, nullable=False, index=True)
    news_count = Column(Integer, nullable=False, default=0)
    avg_confidence = Column(Float, nullable=True)
    avg_relevance = Column(Float, nullable=True)
    source_version = Column(String(20), nullable=False, default="v2")
    aggregated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)

    def __repr__(self):
        return (
            "<DailySentimentV2(date="
            f"{self.date}, sentiment_index={self.sentiment_index:.3f}, news_count={self.news_count})>"
        )


# =============================================================================
# TFT-ASRO: Deep Learning Pipeline Tables
# =============================================================================


class NewsEmbedding(Base):
    """
    FinBERT CLS token embeddings for news articles.

    Stores both the full 768-dim vector and PCA-reduced representation
    used by the Temporal Fusion Transformer.
    """

    __tablename__ = "news_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    news_processed_id = Column(
        BigInteger,
        ForeignKey("news_processed.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    embedding_full = Column(LargeBinary, nullable=True)
    embedding_pca = Column(LargeBinary, nullable=False)
    pca_version = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    processed = relationship("NewsProcessed")

    def __repr__(self):
        return f"<NewsEmbedding(processed_id={self.news_processed_id}, pca={self.pca_version})>"


class LMEWarehouseData(Base):
    """
    LME copper warehouse stock data: total stocks, cancelled warrants,
    and derived ratios used as physical-market features for the TFT.
    """

    __tablename__ = "lme_warehouse_data"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(DateTime(timezone=True), unique=True, nullable=False, index=True)

    total_stock_tonnes = Column(Float, nullable=False)
    cancelled_warrants_tonnes = Column(Float, nullable=True)
    on_warrant_tonnes = Column(Float, nullable=True)
    cancelled_ratio = Column(Float, nullable=True)

    fetched_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    def __repr__(self):
        return f"<LMEWarehouseData(date={self.date}, stock={self.total_stock_tonnes})>"


class TFTModelMetadata(Base):
    """
    Persisted TFT-ASRO model metadata (parallel to XGBoost ModelMetadata).
    """

    __tablename__ = "tft_model_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    config_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    checkpoint_path = Column(String(500), nullable=True)
    trained_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)

    def __repr__(self):
        return f"<TFTModelMetadata(symbol={self.symbol}, trained_at={self.trained_at})>"
