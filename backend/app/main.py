"""
FastAPI application with /api prefix for all endpoints.

Endpoints:
- GET /api/analysis: Current analysis report
- GET /api/history: Historical price and sentiment data
- GET /api/health: System health check
"""

import logging

# Suppress httpx request logging to prevent API keys in URLs from appearing in logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func

from app.db import init_db, SessionLocal, get_db_type
from app.models import NewsArticle, PriceBar, DailySentiment, AnalysisSnapshot
from app.settings import get_settings
from app.lock import is_pipeline_locked
# NOTE: Faz 1 - API is snapshot-only, no report generation
# generate_analysis_report and save_analysis_snapshot are now worker-only
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
    
    # NOTE: Scheduler is NO LONGER started here.
    # Pipeline scheduling is now external (GitHub Actions cron).
    # This API only reads data and enqueues jobs.
    
    yield
    
    # Shutdown
    logger.info("Shutting down CopperMind API...")
    # Close Redis pool if initialized
    try:
        from adapters.queue.redis import close_redis_pool
        import asyncio
        asyncio.create_task(close_redis_pool())
    except ImportError:
        pass


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
        200: {"description": "Analysis report (may include quality_state for degraded modes)"},
        404: {"model": ErrorResponse, "description": "No snapshot available"},
    },
    summary="Get current analysis report (snapshot-only)",
    description="Returns the latest cached analysis snapshot. No live computation - all heavy work is done by the worker."
)
async def get_analysis(
    symbol: str = Query(default="HG=F", description="Trading symbol")
):
    """
    Get current analysis report.
    
    SNAPSHOT-ONLY MODE (Faz 1):
    - Reads the latest snapshot from database
    - NO yfinance calls
    - NO model loading
    - NO feature building
    - All heavy computation is done by the worker pipeline
    
    Response includes quality_state:
    - "ok": Fresh snapshot available
    - "stale": Snapshot older than 36 hours
    - "missing": No snapshot found
    """
    STALE_THRESHOLD_HOURS = 36
    
    with SessionLocal() as session:
        # Get latest snapshot - any age
        snapshot = session.query(AnalysisSnapshot).filter(
            AnalysisSnapshot.symbol == symbol
        ).order_by(AnalysisSnapshot.generated_at.desc()).first()
        
        if snapshot is None:
            # No snapshot at all - return minimal response for UI compatibility
            logger.warning(f"No snapshot found for {symbol}")
            return {
                "symbol": symbol,
                "quality_state": "missing",
                "model_state": "offline",
                "current_price": 0.0,
                "predicted_return": 0.0,
                "predicted_price": 0.0,
                "confidence_lower": 0.0,
                "confidence_upper": 0.0,
                "sentiment_index": 0.0,
                "sentiment_label": "Neutral",
                "top_influencers": [],
                "data_quality": {
                    "news_count_7d": 0,
                    "missing_days": 0,
                    "coverage_pct": 0,
                },
                "generated_at": None,
                "message": "No analysis available. Pipeline may not have run yet.",
            }
        
        # Calculate snapshot age
        now = datetime.now(timezone.utc)
        generated_at = snapshot.generated_at
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        
        age_hours = (now - generated_at).total_seconds() / 3600
        
        # Determine quality state
        if age_hours > STALE_THRESHOLD_HOURS:
            quality_state = "stale"
        else:
            quality_state = "ok"
        
        # Build response from snapshot
        report = snapshot.report_json.copy() if snapshot.report_json else {}
        
        # Add/override metadata
        report["quality_state"] = quality_state
        report["model_state"] = "ok" if quality_state == "ok" else "degraded"
        report["snapshot_age_hours"] = round(age_hours, 1)
        report["generated_at"] = generated_at.isoformat()
        
        # Ensure required fields exist (backward compatibility)
        if "symbol" not in report:
            report["symbol"] = symbol
        if "data_quality" not in report:
            report["data_quality"] = {
                "news_count_7d": 0,
                "missing_days": 0,
                "coverage_pct": 0,
            }
        if "top_influencers" not in report:
            report["top_influencers"] = []
        
        logger.info(f"Returning snapshot for {symbol}: age={age_hours:.1f}h, state={quality_state}")
        
        return report


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
    description="Returns system status including database, Redis queue, models, and pipeline lock state."
)
async def health_check():
    """
    Perform system health check.
    
    Returns status information useful for monitoring and debugging.
    Includes Redis queue status and snapshot age for Faz 1 observability.
    """
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    
    # Count models
    models_found = 0
    if model_dir.exists():
        models_found = len(list(model_dir.glob("xgb_*_latest.json")))
    
    # Get counts and snapshot age
    news_count = None
    price_count = None
    last_snapshot_age = None
    
    try:
        with SessionLocal() as session:
            news_count = session.query(func.count(NewsArticle.id)).scalar()
            price_count = session.query(func.count(PriceBar.id)).scalar()
            
            # Get latest snapshot age
            from app.models import AnalysisSnapshot
            latest_snapshot = session.query(AnalysisSnapshot).order_by(
                AnalysisSnapshot.generated_at.desc()
            ).first()
            
            if latest_snapshot and latest_snapshot.generated_at:
                age = datetime.now(timezone.utc) - latest_snapshot.generated_at.replace(tzinfo=timezone.utc)
                last_snapshot_age = int(age.total_seconds())
                
    except Exception as e:
        logger.error(f"Error getting counts: {e}")
    
    # Check Redis connectivity
    redis_ok = None
    try:
        from adapters.queue.redis import redis_healthcheck
        redis_result = await redis_healthcheck()
        redis_ok = redis_result.get("ok", False)
    except ImportError:
        # Redis adapter not available yet
        redis_ok = None
    except Exception as e:
        logger.warning(f"Redis healthcheck failed: {e}")
        redis_ok = False
    
    # Determine status
    pipeline_locked = is_pipeline_locked()
    
    if models_found == 0:
        status = "degraded"
    elif pipeline_locked:
        status = "degraded"
    elif redis_ok is False:
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
        price_bars_count=price_count,
        redis_ok=redis_ok,
        last_snapshot_age_seconds=last_snapshot_age,
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
# Live Price Endpoint (Twelve Data - Real-time)
# =============================================================================

@app.get(
    "/api/live-price",
    summary="Get real-time copper price from Twelve Data",
    description="Returns live XCU/USD price for header display. Uses Twelve Data API for reliability."
)
async def get_live_price():
    """
    Get real-time copper price from Twelve Data.
    
    Used for the header price display. Separate from yfinance to avoid rate limits.
    """
    import httpx
    
    settings = get_settings()
    
    if not settings.twelvedata_api_key:
        logger.warning("Twelve Data API key not configured")
        return {"price": None, "error": "API key not configured"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
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
                    return {
                        "symbol": "XCU/USD",
                        "price": round(float(price), 4),
                        "error": None,
                    }
                else:
                    return {"price": None, "error": data.get("message", "No price data")}
            else:
                return {"price": None, "error": f"API error: {response.status_code}"}
                
    except Exception as e:
        from app.settings import mask_api_key
        logger.error(f"Twelve Data API error: {mask_api_key(str(e))}")
        return {"price": None, "error": "API error"}


# =============================================================================
# WebSocket Live Price Streaming (Twelve Data)
# =============================================================================

@app.websocket("/ws/live-price")
async def websocket_live_price(websocket: WebSocket):
    """
    WebSocket endpoint for real-time copper price streaming.
    
    Connects to Twelve Data WebSocket and relays price events to the client.
    """
    import websockets
    import asyncio
    import json
    
    await websocket.accept()
    settings = get_settings()
    
    if not settings.twelvedata_api_key:
        await websocket.send_json({"error": "API key not configured"})
        await websocket.close()
        return
    
    td_ws_url = f"wss://ws.twelvedata.com/v1/quotes?apikey={settings.twelvedata_api_key}"
    
    try:
        async with websockets.connect(td_ws_url) as td_ws:
            # Subscribe to BTC/USD first (for testing Basic plan support)
            # If BTC works but XCU doesn't, it means commodities need Pro plan
            subscribe_msg = json.dumps({
                "action": "subscribe",
                "params": {"symbols": "BTC/USD"}
            })
            await td_ws.send(subscribe_msg)
            logger.info("Subscribed to BTC/USD via Twelve Data WebSocket (testing)")
            
            # Heartbeat task to keep connection alive
            async def send_heartbeat():
                while True:
                    await asyncio.sleep(10)
                    try:
                        await td_ws.send(json.dumps({"action": "heartbeat"}))
                    except Exception:
                        break
            
            heartbeat_task = asyncio.create_task(send_heartbeat())
            
            try:
                # Relay messages from Twelve Data to client
                async for message in td_ws:
                    data = json.loads(message)
                    
                    if data.get("event") == "price":
                        await websocket.send_json({
                            "symbol": data.get("symbol"),
                            "price": data.get("price"),
                            "timestamp": data.get("timestamp"),
                        })
                    elif data.get("event") == "subscribe-status":
                        logger.info(f"Subscription status: {data.get('status')}")
                        if data.get("fails"):
                            logger.warning(f"Subscription failures: {data.get('fails')}")
                    
            except WebSocketDisconnect:
                logger.info("Client disconnected from live-price WebSocket")
            finally:
                heartbeat_task.cancel()
                
    except Exception as e:
        # Mask potential API keys in error messages
        from app.settings import mask_api_key
        safe_error = mask_api_key(str(e))
        logger.error(f"WebSocket error: {safe_error}")
        try:
            await websocket.send_json({"error": "Connection error"})  # Don't expose details
        except Exception:
            pass


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
            "ai_stance": result.get("ai_stance", "NEUTRAL"),
        }
    else:
        return {
            "symbol": symbol,
            "commentary": None,
            "error": "No commentary available. Commentary is generated after pipeline runs.",
            "generated_at": None,
            "ai_stance": "NEUTRAL",
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


def verify_pipeline_secret(authorization: Optional[str] = Header(None)) -> None:
    """
    Verify the pipeline trigger secret from Authorization header.
    
    Expected format: Authorization: Bearer <PIPELINE_TRIGGER_SECRET>
    """
    settings = get_settings()
    
    # If no secret is configured, reject all requests (fail secure)
    if not settings.pipeline_trigger_secret:
        logger.warning("Pipeline trigger attempted but PIPELINE_TRIGGER_SECRET not configured")
        raise HTTPException(
            status_code=401,
            detail="Pipeline trigger authentication not configured. Set PIPELINE_TRIGGER_SECRET."
        )
    
    # Check Authorization header
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Expected: Bearer <token>"
        )
    
    # Parse Bearer token
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization format. Expected: Bearer <token>"
        )
    
    token = parts[1]
    
    # Constant-time comparison to prevent timing attacks
    import secrets
    if not secrets.compare_digest(token, settings.pipeline_trigger_secret):
        logger.warning("Pipeline trigger attempted with invalid token")
        raise HTTPException(
            status_code=401,
            detail="Invalid pipeline trigger token"
        )
    
    logger.info("Pipeline trigger authorized successfully")


@app.post(
    "/api/pipeline/trigger",
    summary="Enqueue pipeline job (requires authentication)",
    description="Enqueue a pipeline job to Redis queue. Worker executes the job. Requires Authorization: Bearer <PIPELINE_TRIGGER_SECRET> header.",
    responses={
        200: {"description": "Pipeline job enqueued successfully"},
        401: {"description": "Unauthorized - missing or invalid token"},
        409: {"description": "Pipeline already running"},
        503: {"description": "Redis queue unavailable"},
    },
)
async def trigger_pipeline(
    train_model: bool = Query(default=False, description="Train/retrain XGBoost model"),
    trigger_source: str = Query(default="api", description="Source of trigger (api, cron, manual)"),
    _auth: None = Depends(verify_pipeline_secret),
):
    """
    Enqueue a pipeline job to Redis queue.
    
    This endpoint does NOT run the pipeline - it only enqueues a job.
    The worker service consumes and executes the job.
    
    Returns:
        run_id: UUID for tracking this pipeline run
        enqueued: True if job was enqueued successfully
    """
    # Check if pipeline is already running (advisory lock check)
    # Note: This is a weak check - the worker will do the authoritative lock check
    if is_pipeline_locked():
        raise HTTPException(
            status_code=409,
            detail="Pipeline is already running. Please wait for it to complete."
        )
    
    try:
        from adapters.queue.jobs import enqueue_pipeline_job
        
        result = await enqueue_pipeline_job(
            train_model=train_model,
            trigger_source=trigger_source,
        )
        
        logger.info(f"Pipeline job enqueued: run_id={result['run_id']}, trigger={trigger_source}")
        
        return {
            "status": "enqueued",
            "message": "Pipeline job enqueued. Worker will execute. Check /api/health for status.",
            "run_id": result["run_id"],
            "job_id": result["job_id"],
            "train_model": train_model,
            "trigger_source": trigger_source,
        }
        
    except Exception as e:
        logger.error(f"Failed to enqueue pipeline job: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to enqueue job. Redis may be unavailable: {str(e)}"
        )

