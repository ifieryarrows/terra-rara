"""
SQLAlchemy ORM models for CopperMind.

Tables:
- NewsArticle: Raw news articles with dedup
- PriceBar: OHLCV price data per symbol/date
- NewsSentiment: FinBERT scores per article
- DailySentiment: Aggregated daily sentiment index
- AnalysisSnapshot: Cached analysis reports
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    Index,
    UniqueConstraint,
    JSON,
)
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
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
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
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_price_symbol_date"),
        Index("ix_price_symbol_date", "symbol", "date"),
    )
    
    def __repr__(self):
        return f"<PriceBar(symbol={self.symbol}, date={self.date}, close={self.close})>"


class NewsSentiment(Base):
    """
    FinBERT sentiment scores for each news article.
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
    
    # FinBERT probabilities
    prob_positive = Column(Float, nullable=False)
    prob_neutral = Column(Float, nullable=False)
    prob_negative = Column(Float, nullable=False)
    
    # Derived score: prob_positive - prob_negative
    # Range: [-1, 1], positive means bullish
    score = Column(Float, nullable=False, index=True)
    
    # Model info
    model_name = Column(String(100), default="ProsusAI/finbert")
    
    # When scored
    scored_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
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
    aggregated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
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
    generated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    
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
    generated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    
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
    trained_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<ModelMetadata(symbol={self.symbol}, trained_at={self.trained_at})>"
