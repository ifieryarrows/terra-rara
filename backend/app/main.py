"""
FastAPI application with /api prefix for all endpoints.

Endpoints:
- GET /api/analysis: Current analysis report
- GET /api/history: Historical price and sentiment data
- GET /api/health: System health check
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func

from app.db import init_db, SessionLocal, get_db_type
from app.models import NewsArticle, PriceBar, DailySentiment
from app.settings import get_settings
from app.lock import is_pipeline_locked
from app.inference import (
    generate_analysis_report,
    save_analysis_snapshot,
    get_latest_snapshot,
    get_any_snapshot,
)
from app.schemas import (
    AnalysisReport,
    HistoryResponse,
    HistoryDataPoint,
    HealthResponse,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info("Starting CopperMind API...")
    init_db()
    logger.info("Database initialized")
    
    # Start scheduler if enabled
    settings = get_settings()
    if settings.scheduler_enabled:
        from app.scheduler import start_scheduler
        start_scheduler()
        logger.info("Scheduler started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CopperMind API...")
    if settings.scheduler_enabled:
        from app.scheduler import stop_scheduler
        stop_scheduler()
        logger.info("Scheduler stopped")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="CopperMind API",
    description="Copper market sentiment analysis and price prediction API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get(
    "/api/analysis",
    response_model=AnalysisReport,
    responses={
        404: {"model": ErrorResponse, "description": "Model or data not found"},
        503: {"model": ErrorResponse, "description": "Pipeline locked, snapshot unavailable"},
    },
    summary="Get current analysis report",
    description="Returns the latest analysis report with predictions, sentiment, and influencers."
)
async def get_analysis(
    symbol: str = Query(default="HG=F", description="Trading symbol")
):
    """
    Get current analysis report.
    
    Behavior:
    - If fresh snapshot exists (within TTL), return it
    - If pipeline is not locked, generate fresh report
    - If pipeline is locked, return stale snapshot or 503
    """
    settings = get_settings()
    
    with SessionLocal() as session:
        # Check for fresh snapshot first
        cached = get_latest_snapshot(
            session,
            symbol,
            max_age_minutes=settings.analysis_ttl_minutes
        )
        
        if cached:
            logger.debug(f"Returning cached snapshot for {symbol}")
            return cached
        
        # Check if pipeline is locked
        if is_pipeline_locked():
            # Try to return stale snapshot
            stale = get_any_snapshot(session, symbol)
            if stale:
                logger.info(f"Pipeline locked, returning stale snapshot for {symbol}")
                return stale
            
            raise HTTPException(
                status_code=503,
                detail="Pipeline is currently running. No cached snapshot available. Please try again later."
            )
        
        # Generate fresh report
        try:
            report = generate_analysis_report(session, symbol)
            
            if report is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Could not generate analysis for {symbol}. "
                           "Please ensure data has been fetched (make seed) and model trained (make train)."
                )
            
            # Save as snapshot
            save_analysis_snapshot(session, report, symbol)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            
            # Try stale snapshot as fallback
            stale = get_any_snapshot(session, symbol)
            if stale:
                logger.info(f"Error in fresh generation, returning stale snapshot")
                return stale
            
            raise HTTPException(
                status_code=500,
                detail=f"Error generating analysis: {str(e)}"
            )


@app.get(
    "/api/history",
    response_model=HistoryResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No data found for symbol"},
    },
    summary="Get historical price and sentiment data",
    description="Returns historical data for charting, including prices and sentiment."
)
async def get_history(
    symbol: str = Query(default="HG=F", description="Trading symbol"),
    days: int = Query(default=180, ge=7, le=730, description="Number of days of history")
):
    """
    Get historical price and sentiment data.
    
    IMPORTANT: sentiment_index of 0.0 is a valid value (neutral sentiment),
    not the same as missing data. We return explicit 0.0 values.
    """
    with SessionLocal() as session:
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Query prices
        prices = session.query(
            PriceBar.date,
            PriceBar.close
        ).filter(
            PriceBar.symbol == symbol,
            PriceBar.date >= start_date
        ).order_by(PriceBar.date.asc()).all()
        
        if not prices:
            raise HTTPException(
                status_code=404,
                detail=f"No price data found for {symbol}"
            )
        
        # Query sentiment
        sentiments = session.query(
            DailySentiment.date,
            DailySentiment.sentiment_index,
            DailySentiment.news_count
        ).filter(
            DailySentiment.date >= start_date
        ).order_by(DailySentiment.date.asc()).all()
        
        # Create sentiment lookup (by date string for easy matching)
        sentiment_lookup = {}
        for s in sentiments:
            date_str = s.date.strftime("%Y-%m-%d") if hasattr(s.date, 'strftime') else str(s.date)[:10]
            sentiment_lookup[date_str] = {
                "sentiment_index": s.sentiment_index,
                "news_count": s.news_count
            }
        
        # Build response data
        data_points = []
        for price in prices:
            date_str = price.date.strftime("%Y-%m-%d") if hasattr(price.date, 'strftime') else str(price.date)[:10]
            
            sent = sentiment_lookup.get(date_str)
            
            # IMPORTANT: Use explicit values, don't convert 0.0 to None
            sentiment_idx = sent["sentiment_index"] if sent is not None else None
            news_count = sent["news_count"] if sent is not None else None
            
            data_points.append(HistoryDataPoint(
                date=date_str,
                price=round(price.close, 4),
                sentiment_index=sentiment_idx,
                sentiment_news_count=news_count
            ))
        
        return HistoryResponse(
            symbol=symbol,
            data=data_points
        )


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Returns system status including database, models, and pipeline lock state."
)
async def health_check():
    """
    Perform system health check.
    
    Returns status information useful for monitoring and debugging.
    """
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    
    # Count models
    models_found = 0
    if model_dir.exists():
        models_found = len(list(model_dir.glob("xgb_*_latest.json")))
    
    # Get counts
    news_count = None
    price_count = None
    
    try:
        with SessionLocal() as session:
            news_count = session.query(func.count(NewsArticle.id)).scalar()
            price_count = session.query(func.count(PriceBar.id)).scalar()
    except Exception as e:
        logger.error(f"Error getting counts: {e}")
    
    # Determine status
    pipeline_locked = is_pipeline_locked()
    
    if models_found == 0:
        status = "degraded"
    elif pipeline_locked:
        status = "degraded"
    else:
        status = "healthy"
    
    return HealthResponse(
        status=status,
        db_type=get_db_type(),
        models_found=models_found,
        pipeline_locked=pipeline_locked,
        timestamp=datetime.now(timezone.utc).isoformat(),
        news_count=news_count,
        price_bars_count=price_count
    )


# =============================================================================
# Root redirect (optional convenience)
# =============================================================================

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/docs")


@app.get("/api", include_in_schema=False)
async def api_root():
    """API root information."""
    return {
        "name": "CopperMind API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }

