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

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func

from app.db import init_db, SessionLocal, get_db_type
from app.models import NewsArticle, PriceBar, DailySentiment, DailySentimentV2, AnalysisSnapshot
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
    ConsensusSignal,
    TFTModelSummaryResponse,
    BacktestReportResponse,
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
    settings = get_settings()
    source = str(getattr(settings, "scoring_source", "news_articles")).strip().lower()

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
        
        # Query sentiment (prefer V2 when scoring source is news_processed)
        sentiments = []
        if source == "news_processed":
            sentiments = session.query(
                DailySentimentV2.date,
                DailySentimentV2.sentiment_index,
                DailySentimentV2.news_count
            ).filter(
                DailySentimentV2.date >= start_date
            ).order_by(DailySentimentV2.date.asc()).all()

            if not sentiments:
                logger.warning("No rows in daily_sentiments_v2 for history; falling back to daily_sentiments")

        if not sentiments:
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
# Market Heatmap Endpoint
# =============================================================================

@app.get(
    "/api/market-heatmap",
    summary="Get 15-min delayed market heatmap",
    description="Returns a hierarchical treemap payload for the Finviz-style heatmap UI. Uses stale-while-revalidate caching."
)
async def get_market_heatmap(background_tasks: BackgroundTasks):
    from app.models import HeatmapCache
    from app.heatmap import refresh_market_heatmap
    
    with SessionLocal() as session:
        cache = session.query(HeatmapCache).first()
        now = datetime.now(timezone.utc)
        
        # If no cache or completely empty payload
        if not cache or not cache.payload_json:
            # We don't have stale data to return. We must block or trigger background and return a 503/empty state.
            # To be safe for frontend, return an empty hierarchy and start refresh
            background_tasks.add_task(refresh_market_heatmap)
            return {
                "name": "Market",
                "children": [],
                "_meta": {
                    "is_stale": True,
                    "refresh_in_progress": True,
                    "last_updated_at": None,
                    "next_refresh_at": None,
                    "source_delay_minutes": 15
                }
            }
            
        # Check if stale
        is_stale = now > cache.expires_at
        refresh_in_progress = cache.refresh_started_at is not None
        
        if is_stale and not refresh_in_progress:
            # Trigger background refresh
            background_tasks.add_task(refresh_market_heatmap)
            # Mark it as refreshing immediately so we don't trigger multiple times
            cache.refresh_started_at = now
            session.commit()
            refresh_in_progress = True
            
        payload = cache.payload_json
        if isinstance(payload, dict):
            payload["_meta"] = {
                "is_stale": is_stale,
                "refresh_in_progress": refresh_in_progress,
                "last_updated_at": cache.cached_at.isoformat() if cache.cached_at else None,
                "next_refresh_at": cache.expires_at.isoformat() if cache.expires_at else None,
                "source_delay_minutes": 15
            }
            
        return payload



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

_tft_cache: dict = {}
_TFT_CACHE_TTL_S = 300  # 5 minutes


@app.get(
    "/api/analysis/tft/{symbol}",
    summary="Get TFT-ASRO deep learning analysis",
    description="Returns probabilistic multi-quantile prediction from the Temporal Fusion Transformer model.",
    responses={
        200: {"description": "TFT-ASRO analysis with quantile predictions"},
        404: {"description": "TFT model not available"},
        500: {"description": "Prediction failed"},
    },
)
async def get_tft_analysis(symbol: str = "HG=F"):
    """
    Get TFT-ASRO analysis for the given symbol.

    Results are cached for 5 minutes to avoid rebuilding the full feature
    store on every frontend auto-refresh (~60 s polling).
    """
    now = datetime.now(timezone.utc)
    cached = _tft_cache.get(symbol)
    if cached:
        age = (now - cached["ts"]).total_seconds()
        if age < _TFT_CACHE_TTL_S:
            return cached["data"]

    try:
        from deep_learning.inference.predictor import generate_tft_analysis

        with SessionLocal() as session:
            result = generate_tft_analysis(session, symbol)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        _tft_cache[symbol] = {"data": result, "ts": now}
        return result

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="TFT-ASRO model not trained yet. Run training pipeline first.",
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"TFT-ASRO module not available: {exc}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("TFT analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


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


# =============================================================================
# New User-Facing Endpoints
# =============================================================================

@app.get(
    "/api/analysis/consensus",
    response_model=ConsensusSignal,
    summary="Get consensus signal",
    description="Combines XGBoost and TFT-ASRO signals into a directional consensus."
)
async def get_consensus(
    symbol: str = Query(default="HG=F", description="Trading symbol")
):
    from deep_learning.inference.predictor import ensemble_directional_vote, generate_tft_analysis
    
    # 1. Get TFT analysis
    try:
        with SessionLocal() as session:
            tft_result = generate_tft_analysis(session, symbol)
            
        if "error" in tft_result:
            raise HTTPException(status_code=500, detail=tft_result["error"])
            
        tft_return = tft_result.get("prediction", {}).get("predicted_return_median", 0.0)
    except Exception as e:
        logger.error(f"Failed to get TFT analysis for consensus: {e}")
        tft_return = 0.0

    # 2. Get XGBoost analysis (latest snapshot)
    xgb_return = 0.0
    try:
        with SessionLocal() as session:
            snapshot = session.query(AnalysisSnapshot).filter(
                AnalysisSnapshot.symbol == symbol
            ).order_by(AnalysisSnapshot.generated_at.desc()).first()
            if snapshot and snapshot.report_json:
                xgb_return = snapshot.report_json.get("predicted_return", 0.0)
    except Exception as e:
        logger.error(f"Failed to get XGBoost analysis for consensus: {e}")

    # 3. Calculate consensus
    xgb_bias_correction = 0.001 # Hardcoded small bias correction for now
    result = ensemble_directional_vote(xgb_return, tft_return, xgb_bias_correction)
    return result


@app.get(
    "/api/models/tft/summary",
    response_model=TFTModelSummaryResponse,
    summary="Get TFT model training summary",
    description="Returns training metrics, quality gate results, and feature importance."
)
async def get_tft_summary(
    symbol: str = Query(default="HG=F", description="Target symbol")
):
    from app.models import TFTModelMetadata
    from scripts.tft_quality_gate import evaluate_quality_gate
    import json
    
    with SessionLocal() as session:
        meta = session.query(TFTModelMetadata).filter(
            TFTModelMetadata.symbol == symbol
        ).order_by(TFTModelMetadata.trained_at.desc()).first()
        
        if not meta:
            raise HTTPException(status_code=404, detail=f"No TFT model metadata found for {symbol}")

        config = json.loads(meta.config_json) if meta.config_json else {}
        metrics = json.loads(meta.metrics_json) if meta.metrics_json else {}
        
        # Variable importance not directly in TFTModelMetadata yet, extract from latest artifacts if available
        # But we can try to find it in the artifacts folder
        variable_importance = []
        try:
            import pathlib
            # Use the artifact dir from config if present, or guess
            artifact_dir = pathlib.Path(config.get("feature_store", {}).get("artifact_dir", "artifacts/feature_store"))
            mrmr_path = artifact_dir / "latest" / "mrmr_results.json"
            if mrmr_path.exists():
                mrmr_data = json.loads(mrmr_path.read_text(encoding="utf-8"))
                for feat, imp in mrmr_data.get("scores", {}).items():
                    variable_importance.append({"feature": feat, "importance": float(imp)})
                # Sort and take top 20
                variable_importance.sort(key=lambda x: x["importance"], reverse=True)
                variable_importance = variable_importance[:20]
        except Exception as e:
            logger.warning(f"Could not load variable importance: {e}")

        da = metrics.get("directional_accuracy", 0.5)
        sharpe = metrics.get("sharpe_ratio", 0.0)
        vr = metrics.get("variance_ratio", 1.0)
        
        passed, reasons = evaluate_quality_gate(da, sharpe, vr)
        
        return {
            "symbol": symbol,
            "trained_at": meta.trained_at.isoformat() if meta.trained_at else None,
            "checkpoint_path": meta.checkpoint_path,
            "config": config,
            "metrics": metrics,
            "variable_importance": variable_importance,
            "quality_gate": {
                "passed": passed,
                "reasons": reasons,
                "metrics": {"da": da, "sharpe": sharpe, "vr": vr}
            }
        }


@app.get(
    "/api/models/tft/backtest/latest",
    response_model=BacktestReportResponse,
    summary="Get latest backtest report",
    description="Returns the latest walk-forward backtest results and Theta comparison."
)
async def get_latest_backtest():
    import pathlib
    import json
    
    backtest_dir = pathlib.Path("artifacts/backtest")
    if not backtest_dir.exists():
        raise HTTPException(status_code=404, detail="No backtest reports found")
        
    reports = list(backtest_dir.glob("backtest_*.json"))
    if not reports:
        raise HTTPException(status_code=404, detail="No backtest reports found")
        
    latest_report_path = max(reports, key=lambda p: p.stat().st_mtime)
    
    try:
        data = json.loads(latest_report_path.read_text(encoding="utf-8"))
        
        tft_bt = data.get("tft_backtest", {})
        comp = data.get("baseline_comparison", {})
        
        return {
            "report_date": data.get("timestamp", ""),
            "summary_metrics": tft_bt.get("summary", {}),
            "window_metrics": tft_bt.get("windows", []),
            "theta_comparison": comp,
            "verdict": comp.get("verdict")
        }
    except Exception as e:
        logger.error(f"Error reading backtest report: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse backtest report")


