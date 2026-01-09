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
            # Update current_price with live data before returning
            import yfinance as yf
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                live_price = info.get('regularMarketPrice') or info.get('currentPrice')
                if live_price is not None:
                    cached['current_price'] = round(float(live_price), 4)
                    # Recalculate predicted_return based on live price
                    if cached.get('predicted_price'):
                        cached['predicted_return'] = round(
                            (cached['predicted_price'] - cached['current_price']) / cached['current_price'],
                            6
                        )
                    logger.info(f"Updated cached snapshot with live price: ${live_price:.4f}")
            except Exception as e:
                logger.debug(f"Could not update live price in cached snapshot: {e}")
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


@app.get(
    "/api/market-prices",
    summary="Get live market prices for all symbols",
    description="Returns live price and daily change for all tracked symbols (15-min delayed)."
)
async def get_market_prices():
    """
    Get live prices and daily changes for all tracked symbols.
    
    Uses yfinance for real-time data (15-minute delayed).
    Used by the Market Intelligence Map component.
    """
    import yfinance as yf
    
    settings = get_settings()
    symbols = settings.symbols_list
    
    result = {}
    
    try:
        # Fetch all tickers at once for efficiency
        tickers = yf.Tickers(' '.join(symbols))
        
        for symbol in symbols:
            try:
                ticker = tickers.tickers.get(symbol)
                if not ticker:
                    result[symbol] = {"price": None, "change": None}
                    continue
                    
                info = ticker.info
                
                # Get current price and change
                current_price = info.get('regularMarketPrice') or info.get('currentPrice')
                change_pct = info.get('regularMarketChangePercent')
                
                if current_price is not None:
                    result[symbol] = {
                        "price": round(current_price, 4),
                        "change": round(change_pct, 2) if change_pct else 0,
                    }
                else:
                    result[symbol] = {"price": None, "change": None}
                    
            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                result[symbol] = {"price": None, "change": None}
                
    except Exception as e:
        logger.error(f"Error fetching market prices: {e}")
        return {"error": str(e), "symbols": {}}
    
    return {"symbols": result}


# =============================================================================
# AI Commentary Endpoint
# =============================================================================

@app.get(
    "/api/commentary",
    summary="AI-generated market commentary",
    description="Returns the AI-generated analysis stored after pipeline completion."
)
async def get_commentary(
    symbol: str = Query(default="HG=F", description="Symbol to get commentary for")
):
    """
    Get AI commentary for the specified symbol.
    
    Commentary is generated once after each pipeline run and stored in the database.
    This endpoint simply returns the stored commentary without making new API calls.
    """
    from app.commentary import get_commentary_from_db
    
    with SessionLocal() as session:
        result = get_commentary_from_db(session, symbol)
    
    if result:
        return {
            "symbol": symbol,
            "commentary": result["commentary"],
            "error": None,
            "generated_at": result["generated_at"],
        }
    else:
        return {
            "symbol": symbol,
            "commentary": None,
            "error": "No commentary available. Commentary is generated after pipeline runs.",
            "generated_at": None,
        }


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


# =============================================================================
# Pipeline Management Endpoints
# =============================================================================

@app.post(
    "/api/pipeline/trigger",
    summary="Trigger data pipeline",
    description="Manually trigger data fetch and AI pipeline. Use with caution - this is a heavy operation.",
    responses={
        200: {"description": "Pipeline triggered successfully"},
        409: {"description": "Pipeline already running"},
    },
)
async def trigger_pipeline(
    fetch_data: bool = Query(default=True, description="Fetch new data from sources"),
    train_model: bool = Query(default=True, description="Train/retrain XGBoost model"),
):
    """
    Manually trigger the pipeline.
    
    This will:
    1. Fetch new news and price data (if fetch_data=True)
    2. Run sentiment scoring
    3. Train XGBoost model (if train_model=True)
    """
    from threading import Thread
    
    if is_pipeline_locked():
        raise HTTPException(
            status_code=409,
            detail="Pipeline is already running. Please wait for it to complete."
        )
    
    def run_pipeline():
        try:
            from app.lock import PipelineLock
            from app.inference import generate_analysis_report, save_analysis_snapshot
            from app.db import SessionLocal
            
            lock = PipelineLock(timeout=0)
            if not lock.acquire():
                logger.error("Could not acquire pipeline lock")
                return
            
            try:
                settings = get_settings()
                
                if fetch_data:
                    logger.info("Step 1: Fetching data...")
                    from app.data_manager import fetch_all
                    fetch_all(news=True, prices=True)
                    logger.info("Data fetch complete")
                
                logger.info("Step 2: Running AI pipeline...")
                from app.ai_engine import run_full_pipeline
                run_full_pipeline(
                    target_symbol="HG=F",
                    score_sentiment=True,
                    aggregate_sentiment=True,
                    train_model=train_model
                )
                logger.info("AI pipeline complete")
                
                # Step 3: Generate snapshot
                logger.info("Step 3: Generating analysis snapshot...")
                with SessionLocal() as session:
                    report = generate_analysis_report(session, settings.target_symbol)
                    if report:
                        save_analysis_snapshot(session, report, settings.target_symbol)
                        logger.info(f"Snapshot generated")
                        
                        # Step 4: Generate AI Commentary
                        logger.info("Step 4: Generating AI commentary...")
                        try:
                            import asyncio
                            from app.commentary import generate_and_save_commentary
                            from sqlalchemy import func
                            from app.models import NewsArticle
                            from datetime import timedelta
                            
                            # Get news count for last 7 days
                            week_ago = datetime.now() - timedelta(days=7)
                            news_count = session.query(func.count(NewsArticle.id)).filter(
                                NewsArticle.published_at >= week_ago
                            ).scalar() or 0
                            
                            # Run async function in sync context
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                commentary = loop.run_until_complete(
                                    generate_and_save_commentary(
                                        session=session,
                                        symbol=settings.target_symbol,
                                        current_price=report.get('current_price', 0),
                                        predicted_price=report.get('predicted_price', 0),
                                        predicted_return=report.get('predicted_return', 0),
                                        sentiment_index=report.get('sentiment_index', 0),
                                        sentiment_label=report.get('sentiment_label', 'Neutral'),
                                        top_influencers=report.get('top_influencers', []),
                                        news_count=news_count,
                                    )
                                )
                                if commentary:
                                    logger.info("AI commentary generated and saved")
                                else:
                                    logger.warning("AI commentary skipped (API key not configured or failed)")
                            finally:
                                loop.close()
                        except Exception as ce:
                            logger.error(f"AI commentary generation failed: {ce}")
                    else:
                        logger.warning("Could not generate analysis snapshot")
                
            finally:
                lock.release()
                
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
    
    # Run in background thread
    thread = Thread(target=run_pipeline, daemon=True)
    thread.start()
    
    return {
        "status": "triggered",
        "message": "Pipeline started in background. Check /api/health for status.",
        "fetch_data": fetch_data,
        "train_model": train_model
    }

