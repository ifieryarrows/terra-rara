"""
Pydantic schemas for API request/response validation.
These schemas define the contract between backend and frontend.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Influencer(BaseModel):
    """Top feature influencer in the model."""
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., ge=0, le=1, description="Normalized importance score")
    description: Optional[str] = Field(None, description="Human-readable description")


class DataQuality(BaseModel):
    """Data quality metrics for transparency."""
    news_count_7d: int = Field(..., ge=0, description="Number of news articles in last 7 days")
    missing_days: int = Field(..., ge=0, description="Number of missing trading days")
    coverage_pct: int = Field(..., ge=0, le=100, description="Data coverage percentage")
    language_filtered: Optional[int] = Field(None, description="Articles filtered by language")


class AnalysisReport(BaseModel):
    """Full analysis report returned by /api/analysis."""
    symbol: str = Field(..., description="Trading symbol (e.g., HG=F)")
    current_price: float = Field(..., description="Most recent closing price")
    predicted_return: float = Field(..., description="Predicted next-day return")
    predicted_price: float = Field(..., description="Predicted next-day price")
    confidence_lower: float = Field(..., description="Lower bound of confidence interval")
    confidence_upper: float = Field(..., description="Upper bound of confidence interval")
    sentiment_index: float = Field(..., description="Current sentiment index (-1 to 1)")
    sentiment_label: str = Field(..., description="Sentiment label: Bullish, Bearish, or Neutral")
    top_influencers: list[Influencer] = Field(..., description="Top feature influencers")
    data_quality: DataQuality = Field(..., description="Data quality metrics")
    generated_at: str = Field(..., description="ISO timestamp of report generation")

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
                "generated_at": "2026-01-02T09:00:00Z"
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

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "db_type": "sqlite",
                "models_found": 1,
                "pipeline_locked": False,
                "timestamp": "2026-01-02T10:00:00Z",
                "news_count": 1250,
                "price_bars_count": 1460
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

