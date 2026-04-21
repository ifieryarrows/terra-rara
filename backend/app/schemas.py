"""
Pydantic schemas for API request/response validation.
These schemas define the contract between backend and frontend.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class Influencer(BaseModel):
    """Top feature influencer in the model."""
    feature: str = Field(..., description="Raw feature identifier (e.g. CL=F_lag_ret1_2)")
    importance: float = Field(..., ge=0, le=1, description="Normalized importance score")
    label: Optional[str] = Field(None, description="Short, user-facing label")
    description: Optional[str] = Field(None, description="Longer human-readable description")
    category: Optional[str] = Field(None, description="High-level bucket (Momentum, Macro, ...)")
    time_horizon: Optional[str] = Field(None, description="Lookback horizon, e.g. '14d'")


class DataQuality(BaseModel):
    """Data quality metrics for transparency."""
    news_count_7d: int = Field(..., ge=0, description="Number of news articles in last 7 days")
    missing_days: int = Field(..., ge=0, description="Number of missing trading days")
    coverage_pct: int = Field(..., ge=0, le=100, description="Data coverage percentage")
    language_filtered: Optional[int] = Field(None, description="Articles filtered by language")


class AnalysisReport(BaseModel):
    """
    Full analysis report returned by /api/analysis.
    
    Faz 1: Snapshot-only mode - fields may be null in degraded states.
    Check quality_state to determine data freshness.
    """
    symbol: str = Field(..., description="Trading symbol (e.g., HG=F)")
    
    # Core prediction data (nullable for degraded modes)
    current_price: Optional[float] = Field(0.0, description="Most recent closing price")
    predicted_return: Optional[float] = Field(0.0, description="Predicted next-day return")
    raw_predicted_return: Optional[float] = Field(
        None, description="Raw model output converted to return before sentiment adjustment"
    )
    sentiment_multiplier: Optional[float] = Field(
        None, description="Sentiment-driven multiplier applied to raw predicted return"
    )
    sentiment_adjustment_applied: Optional[bool] = Field(
        None, description="Whether sentiment adjustment layer altered predicted return"
    )
    predicted_return_capped: Optional[bool] = Field(
        None, description="Whether final predicted return was clipped by safety cap"
    )
    predicted_price: Optional[float] = Field(0.0, description="Predicted next-day price")
    confidence_lower: Optional[float] = Field(0.0, description="Lower bound of confidence interval")
    confidence_upper: Optional[float] = Field(0.0, description="Upper bound of confidence interval")
    sentiment_index: Optional[float] = Field(0.0, description="Current sentiment index (-1 to 1)")
    sentiment_label: Optional[str] = Field("Neutral", description="Sentiment label: Bullish, Bearish, or Neutral")
    
    # Feature influencers (may be empty)
    top_influencers: list[Influencer] = Field(default_factory=list, description="Top feature influencers")
    
    # Data quality (always present)
    data_quality: Optional[DataQuality] = Field(None, description="Data quality metrics")
    
    # Timestamps
    generated_at: Optional[str] = Field(None, description="ISO timestamp of report generation")
    
    # Faz 1: Quality state fields
    quality_state: Optional[str] = Field("ok", description="Snapshot quality: ok, stale, missing")
    model_state: Optional[str] = Field("ok", description="Model status: ok, degraded, offline")
    snapshot_age_hours: Optional[float] = Field(None, description="Hours since snapshot was generated")
    message: Optional[str] = Field(None, description="Human-readable status message")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "HG=F",
                "current_price": 4.2534,
                "predicted_return": 0.0123,
                "predicted_price": 4.3057,
                "confidence_lower": 4.1892,
                "confidence_upper": 4.4222,
                "sentiment_index": 0.1523,
                "sentiment_label": "Bullish",
                "top_influencers": [
                    {"feature": "DX-Y.NYB_ret1", "importance": 0.23, "description": "US Dollar Index Return"},
                    {"feature": "sentiment__index", "importance": 0.18, "description": "Market Sentiment Index"}
                ],
                "data_quality": {
                    "news_count_7d": 45,
                    "missing_days": 0,
                    "coverage_pct": 100
                },
                "generated_at": "2026-01-02T09:00:00Z",
                "quality_state": "ok",
                "model_state": "ok",
                "snapshot_age_hours": 2.5
            }
        }


class HistoryDataPoint(BaseModel):
    """Single data point in history response."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    price: float = Field(..., description="Closing price")
    sentiment_index: Optional[float] = Field(None, description="Sentiment index (can be 0.0)")
    sentiment_news_count: Optional[int] = Field(None, description="Number of news articles")


class HistoryResponse(BaseModel):
    """Historical price and sentiment data for charts."""
    symbol: str = Field(..., description="Trading symbol")
    data: list[HistoryDataPoint] = Field(..., description="Historical data points")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "HG=F",
                "data": [
                    {"date": "2025-12-01", "price": 4.15, "sentiment_index": 0.05, "sentiment_news_count": 12},
                    {"date": "2025-12-02", "price": 4.18, "sentiment_index": -0.02, "sentiment_news_count": 8},
                ]
            }
        }


class HealthResponse(BaseModel):
    """System health check response."""
    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    db_type: str = Field(..., description="Database type (sqlite, postgresql, etc.)")
    models_found: int = Field(..., ge=0, description="Number of trained models found")
    pipeline_locked: bool = Field(..., description="Whether pipeline is currently locked")
    timestamp: str = Field(..., description="Current server timestamp")
    news_count: Optional[int] = Field(None, description="Total news articles in database")
    price_bars_count: Optional[int] = Field(None, description="Total price bars in database")
    
    # Faz 1: Queue and snapshot observability
    redis_ok: Optional[bool] = Field(None, description="Redis queue connectivity")
    last_snapshot_age_seconds: Optional[int] = Field(None, description="Age of last analysis snapshot in seconds")

    # Added 2026-04: ingestion + pipeline freshness so the System page can
    # surface "why is the forecast stale?" at a glance.
    #
    # Naming is deliberate — each timestamp answers a different question and
    # they are NOT interchangeable:
    #   * last_pipeline_run_at       → when the worker actually completed
    #   * last_snapshot_generated_at → when /api/analysis was last refreshed
    #   * last_tft_prediction_at     → when the TFT snapshot was produced
    #   * tft_model_trained_at       → when the TFT checkpoint was (re)trained
    #   * tft_reference_price_date   → close date the latest forecast is anchored to
    #   * price_bar_latest_date      → most recent HG=F OHLC bar we have ingested
    last_pipeline_run_at: Optional[str] = Field(
        None,
        description=(
            "ISO timestamp of the most recent pipeline run that actually "
            "completed on the worker (from pipeline_run_metrics)."
        ),
    )
    last_pipeline_status: Optional[str] = Field(
        None, description="Outcome of the most recent pipeline run (ok, failed, running)"
    )
    last_snapshot_generated_at: Optional[str] = Field(
        None,
        description="ISO timestamp of the latest XGBoost analysis snapshot (AnalysisSnapshot.generated_at).",
    )
    last_tft_prediction_at: Optional[str] = Field(
        None,
        description="ISO timestamp of the latest persisted TFT prediction snapshot.",
    )
    tft_model_trained_at: Optional[str] = Field(
        None,
        description="ISO timestamp of the most recent TFT checkpoint training completion.",
    )
    tft_reference_price_date: Optional[str] = Field(
        None,
        description="Close date (YYYY-MM-DD) that the latest TFT forecast is anchored to.",
    )
    price_bar_latest_date: Optional[str] = Field(
        None, description="Date (YYYY-MM-DD) of the most recent PriceBar for the target"
    )
    price_bar_staleness_days: Optional[int] = Field(
        None, description="Calendar days between today and the latest PriceBar"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "db_type": "postgresql",
                "models_found": 1,
                "pipeline_locked": False,
                "timestamp": "2026-01-02T10:00:00Z",
                "news_count": 1250,
                "price_bars_count": 1460,
                "redis_ok": True,
                "last_snapshot_age_seconds": 3600
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Model not found. Please run 'make train' first.",
                "error_code": "MODEL_NOT_FOUND"
            }
        }


class ConsensusSignal(BaseModel):
    """Consensus signal combining XGBoost and TFT."""
    consensus_direction: str = Field(..., description="BULLISH, BEARISH, or NEUTRAL")
    confidence: str = Field(..., description="HIGH, MEDIUM, or LOW")
    position_scale: float = Field(..., description="0.0 to 1.0 scaling factor for position sizing")
    blended_return: float = Field(..., description="Blended expected return from both models")
    xgb_return_raw: float = Field(..., description="Raw XGBoost predicted return")
    xgb_return_adjusted: float = Field(..., description="Debiased XGBoost predicted return")
    tft_return: float = Field(..., description="TFT-ASRO median predicted return")
    xgb_direction: int = Field(..., description="-1, 0, or 1")
    tft_direction: int = Field(..., description="-1, 0, or 1")


class QualityGateResponse(BaseModel):
    """Quality gate results for TFT-ASRO."""
    passed: bool = Field(..., description="Whether the model passed the quality gate")
    reasons: List[str] = Field(default_factory=list, description="Reasons for failure, if any")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Key metrics evaluated (DA, Sharpe, VR)")


class VariableImportance(BaseModel):
    """TFT variable importance."""
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Normalized importance score")


class TFTModelSummaryResponse(BaseModel):
    """TFT model training and evaluation summary."""
    symbol: str = Field(..., description="Target symbol")
    trained_at: Optional[str] = Field(None, description="ISO timestamp of training completion")
    checkpoint_path: str = Field(..., description="Path to the best model checkpoint")
    config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration parameters")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Test metrics from training")
    variable_importance: List[VariableImportance] = Field(default_factory=list, description="Top feature importance scores")
    quality_gate: Optional[QualityGateResponse] = Field(None, description="Quality gate evaluation results")


class BacktestReportResponse(BaseModel):
    """Latest backtest report results."""
    report_date: str = Field(..., description="ISO timestamp of report generation")
    summary_metrics: Dict[str, Any] = Field(default_factory=dict, description="Overall backtest metrics")
    window_metrics: List[Dict[str, Any]] = Field(default_factory=list, description="Metrics per rolling window")
    theta_comparison: Optional[Dict[str, Any]] = Field(None, description="Comparison with Theta baseline")
    verdict: Optional[str] = Field(None, description="TFT_SUPERIOR, THETA_SUPERIOR, or MIXED")


# =============================================================================
# News Intelligence schemas
# =============================================================================


class NewsFinbertProbs(BaseModel):
    """FinBERT class probability triplet for a news article."""
    pos: float = Field(..., ge=0, le=1)
    neu: float = Field(..., ge=0, le=1)
    neg: float = Field(..., ge=0, le=1)


class NewsSentimentBlock(BaseModel):
    """Per-article sentiment payload shipped to the frontend feed."""
    label: Optional[str] = Field(None, description="BULLISH | BEARISH | NEUTRAL")
    final_score: Optional[float] = Field(None, description="Ensemble score in [-1, 1]")
    impact_score_llm: Optional[float] = Field(None, description="LLM-only impact in [-1, 1]")
    confidence: Optional[float] = Field(None, description="Calibrated confidence in [0, 1]")
    relevance: Optional[float] = Field(None, description="Relevance to copper market in [0, 1]")
    event_type: Optional[str] = Field(None, description="LLM event type bucket")
    finbert: Optional[NewsFinbertProbs] = Field(None, description="FinBERT probability triplet")
    reasoning: Optional[str] = Field(None, description="Short textual rationale from the LLM")
    scored_at: Optional[str] = Field(None, description="ISO timestamp when the score was written")


class NewsItem(BaseModel):
    """Single article row in the news feed."""
    id: int = Field(..., description="news_processed id (stable frontend key)")
    raw_id: Optional[int] = Field(None, description="news_raw id for debugging")
    title: str
    description: Optional[str] = None
    url: Optional[str] = None
    channel: str = Field(
        ..., description="Ingestion channel (google_news, newsapi, ...)"
    )
    publisher: Optional[str] = Field(
        None, description="Original publisher extracted from raw_payload.source"
    )
    source_feed: Optional[str] = Field(None, description="RSS query / feed identifier")
    published_at: Optional[str] = Field(None, description="ISO timestamp")
    fetched_at: Optional[str] = Field(None, description="ISO timestamp")
    language: Optional[str] = None
    sentiment: Optional[NewsSentimentBlock] = None


class NewsListResponse(BaseModel):
    """Paginated news feed response."""
    items: List[NewsItem] = Field(default_factory=list)
    total: int = Field(..., description="Total rows matching filters (for pagination)")
    limit: int = Field(...)
    offset: int = Field(...)
    has_more: bool = Field(...)
    generated_at: str = Field(..., description="ISO timestamp the response was built")
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Echo of the filter args applied server-side",
    )


class NewsStatsResponse(BaseModel):
    """Aggregate stats for the news intelligence sidebar header."""
    window_hours: int = Field(..., description="Rolling window used for aggregation")
    total_articles: int = Field(..., ge=0)
    scored_articles: int = Field(..., ge=0)
    label_distribution: Dict[str, int] = Field(
        default_factory=dict, description="BULLISH/BEARISH/NEUTRAL counts"
    )
    event_type_distribution: Dict[str, int] = Field(default_factory=dict)
    channel_distribution: Dict[str, int] = Field(
        default_factory=dict, description="google_news / newsapi counts"
    )
    top_publishers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="[{publisher, count, avg_final_score}]",
    )
    avg_final_score: Optional[float] = None
    avg_confidence: Optional[float] = None
    avg_relevance: Optional[float] = None
    generated_at: str = Field(...)

