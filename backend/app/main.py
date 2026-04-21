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
from app.models import NewsArticle, PriceBar, DailySentiment, DailySentimentV2, AnalysisSnapshot, NewsSentimentV2, NewsProcessed, NewsRaw
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
    NewsItem,
    NewsListResponse,
    NewsStatsResponse,
    NewsFinbertProbs,
    NewsSentimentBlock,
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

        # Re-label cached influencers so snapshots written before the
        # describe_feature() rollout also render human-readable names in the
        # UI. Non-destructive: pre-existing rich fields (label/description/
        # category/time_horizon) are preserved; missing ones are back-filled.
        try:
            from app.features import describe_feature

            rebuilt: list[dict] = []
            for infl in report.get("top_influencers", []) or []:
                if not isinstance(infl, dict):
                    continue
                feature_key = infl.get("feature") or infl.get("name") or ""
                if not feature_key:
                    rebuilt.append(infl)
                    continue
                meta = describe_feature(str(feature_key))
                enriched = {
                    **infl,
                    "feature": feature_key,
                    "label": infl.get("label") or meta.get("label") or feature_key,
                    "description": infl.get("description") or meta.get("description") or "",
                    "category": infl.get("category") or meta.get("category") or "technical",
                    "time_horizon": (
                        infl.get("time_horizon")
                        or meta.get("time_horizon")
                        or "intraday"
                    ),
                }
                rebuilt.append(enriched)
            report["top_influencers"] = rebuilt
        except Exception as label_err:
            logger.warning(f"Influencer re-label skipped: {label_err}")

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

    # Freshness metadata for the System page. Each field answers a distinct
    # question — see HealthResponse for the exact definitions.
    last_pipeline_run_at: Optional[str] = None
    last_pipeline_status: Optional[str] = None
    last_snapshot_generated_at: Optional[str] = None
    last_tft_prediction_at: Optional[str] = None
    tft_model_trained_at: Optional[str] = None
    tft_reference_price_date: Optional[str] = None
    price_bar_latest_date: Optional[str] = None
    price_bar_staleness_days: Optional[int] = None

    def _iso(dt):
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    try:
        with SessionLocal() as session:
            news_count = session.query(func.count(NewsArticle.id)).scalar()
            price_count = session.query(func.count(PriceBar.id)).scalar()

            from app.models import (
                AnalysisSnapshot,
                PipelineRunMetrics,
                TFTModelMetadata,
                TFTPredictionSnapshot,
            )

            # --- Authoritative pipeline run timestamp ------------------------
            # Read from the actual worker metrics table, not the snapshot
            # table. Snapshots are only ONE artifact of a pipeline run; a
            # failed run still records a row here.
            latest_run = (
                session.query(PipelineRunMetrics)
                .order_by(PipelineRunMetrics.run_started_at.desc())
                .first()
            )
            if latest_run is not None:
                ended = latest_run.run_completed_at or latest_run.run_started_at
                last_pipeline_run_at = _iso(ended)
                # Map internal run.status → external pipeline_status.
                #   running  → running
                #   success  → ok
                #   failed   → failed
                raw_status = (latest_run.status or "").lower()
                if raw_status == "success":
                    last_pipeline_status = "ok"
                elif raw_status in {"running", "failed"}:
                    last_pipeline_status = raw_status
                else:
                    last_pipeline_status = raw_status or None

            # --- XGBoost snapshot age ---------------------------------------
            latest_snapshot = (
                session.query(AnalysisSnapshot)
                .order_by(AnalysisSnapshot.generated_at.desc())
                .first()
            )
            if latest_snapshot and latest_snapshot.generated_at:
                snap_at = latest_snapshot.generated_at
                if snap_at.tzinfo is None:
                    snap_at = snap_at.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - snap_at
                last_snapshot_age = int(age.total_seconds())
                last_snapshot_generated_at = snap_at.isoformat()
                # If PipelineRunMetrics has no rows yet (fresh DB) fall back
                # to snapshot-derived status so older deployments don't go
                # blank.
                if last_pipeline_run_at is None:
                    last_pipeline_run_at = last_snapshot_generated_at
                if last_pipeline_status is None:
                    last_pipeline_status = (
                        "ok" if last_snapshot_age < 36 * 3600 else "stale"
                    )

            # --- Latest persisted TFT snapshot ------------------------------
            latest_tft = (
                session.query(TFTPredictionSnapshot)
                .filter(TFTPredictionSnapshot.symbol == "HG=F")
                .order_by(TFTPredictionSnapshot.generated_at.desc())
                .first()
            )
            if latest_tft is not None:
                last_tft_prediction_at = _iso(latest_tft.generated_at)
                tft_reference_price_date = latest_tft.reference_price_date

            # --- Latest TFT training timestamp ------------------------------
            latest_tft_model = (
                session.query(TFTModelMetadata)
                .filter(TFTModelMetadata.symbol == "HG=F")
                .order_by(TFTModelMetadata.trained_at.desc())
                .first()
            )
            if latest_tft_model is not None:
                tft_model_trained_at = _iso(latest_tft_model.trained_at)

            # --- PriceBar freshness -----------------------------------------
            target = "HG=F"
            latest_bar = (
                session.query(PriceBar.date)
                .filter(PriceBar.symbol == target)
                .order_by(PriceBar.date.desc())
                .first()
            )
            if latest_bar and latest_bar[0]:
                bar_date = latest_bar[0]
                if bar_date.tzinfo is None:
                    bar_date = bar_date.replace(tzinfo=timezone.utc)
                price_bar_latest_date = bar_date.strftime("%Y-%m-%d")
                price_bar_staleness_days = max(
                    int((datetime.now(timezone.utc) - bar_date).days), 0
                )

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
        last_pipeline_run_at=last_pipeline_run_at,
        last_pipeline_status=last_pipeline_status,
        last_snapshot_generated_at=last_snapshot_generated_at,
        last_tft_prediction_at=last_tft_prediction_at,
        tft_model_trained_at=tft_model_trained_at,
        tft_reference_price_date=tft_reference_price_date,
        price_bar_latest_date=price_bar_latest_date,
        price_bar_staleness_days=price_bar_staleness_days,
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
    summary="Get CopperMind universe heatmap (15-min cache)",
    description=(
        "Returns a group->subgroup->symbol treemap payload sourced exclusively from the "
        "CopperMind project universe (broad_universe.csv). Uses stale-while-revalidate "
        "caching with a 15-minute TTL. No general market indices are included."
    )
)
async def get_market_heatmap(background_tasks: BackgroundTasks):
    from app.models import HeatmapCache
    from app.heatmap import refresh_market_heatmap

    # Stuck refresh safety: if a refresh has been "in progress" for longer than
    # this, assume the worker crashed and allow a fresh background refresh to
    # be kicked off. yfinance batch fetch for the full universe finishes in
    # under ~2 minutes under normal conditions.
    STUCK_REFRESH_SECONDS = 180

    with SessionLocal() as session:
        cache = session.query(HeatmapCache).first()
        now = datetime.now(timezone.utc)

        def _payload_count(payload) -> int:
            if not isinstance(payload, dict):
                return 0
            total = 0
            for grp in payload.get("children", []) or []:
                for sub in grp.get("children", []) or []:
                    total += len(sub.get("children", []) or [])
            return total

        # If no cache or completely empty payload — trigger background refresh
        if not cache or not cache.payload_json or _payload_count(cache.payload_json) == 0:
            # Clear any stale "in progress" flag so we don't deadlock.
            if cache and cache.refresh_started_at is not None:
                started = cache.refresh_started_at
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                age = (now - started).total_seconds()
                if age > STUCK_REFRESH_SECONDS:
                    cache.refresh_started_at = None
                    session.commit()
            background_tasks.add_task(refresh_market_heatmap)
            return {
                "name": "CopperMind Universe",
                "children": [],
                "_meta": {
                    "is_stale": True,
                    "refresh_in_progress": True,
                    "last_updated_at": None,
                    "next_refresh_at": None,
                    "source_delay_minutes": 15,
                    "payload_count": 0,
                    "refresh_error": cache.refresh_error if cache else None,
                    "cache_state": "empty",
                },
            }

        # Check if stale
        expires_at = cache.expires_at
        if expires_at and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        is_stale = now > expires_at if expires_at else True

        refresh_in_progress = cache.refresh_started_at is not None
        # Recover from stuck "in progress" flags
        if refresh_in_progress:
            started = cache.refresh_started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if (now - started).total_seconds() > STUCK_REFRESH_SECONDS:
                logger.warning(
                    "Heatmap refresh appears stuck (started %.0fs ago) — clearing flag",
                    (now - started).total_seconds(),
                )
                cache.refresh_started_at = None
                session.commit()
                refresh_in_progress = False

        if is_stale and not refresh_in_progress:
            background_tasks.add_task(refresh_market_heatmap)
            cache.refresh_started_at = now
            session.commit()
            refresh_in_progress = True

        payload = cache.payload_json
        payload_count = _payload_count(payload)
        cache_state = "fresh"
        if is_stale:
            cache_state = "stale"
        if refresh_in_progress:
            cache_state = "refreshing"

        if isinstance(payload, dict):
            payload["_meta"] = {
                "is_stale": is_stale,
                "refresh_in_progress": refresh_in_progress,
                "last_updated_at": cache.cached_at.isoformat() if cache.cached_at else None,
                "next_refresh_at": cache.expires_at.isoformat() if cache.expires_at else None,
                "source_delay_minutes": 15,
                "payload_count": payload_count,
                "refresh_error": cache.refresh_error,
                "cache_state": cache_state,
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
    description=(
        "Returns probabilistic multi-quantile prediction from the Temporal "
        "Fusion Transformer model. By default reads the latest TFT snapshot "
        "produced by the daily pipeline worker (persistent, cheap). Pass "
        "`source=live` to force a fresh inference run — useful for diagnostics."
    ),
    responses={
        200: {"description": "TFT-ASRO analysis with quantile predictions"},
        404: {"description": "TFT model not available"},
        500: {"description": "Prediction failed"},
    },
)
async def get_tft_analysis(
    symbol: str = "HG=F",
    source: str = "snapshot",
):
    """
    Get TFT-ASRO analysis for the given symbol.

    `source` semantics:
      * `snapshot` (default) — serve the latest persisted TFTPredictionSnapshot
        written by the worker. If none exists, transparently fall back to live.
      * `live`                — always run a fresh inference. In-memory cached
        for 5 minutes to protect the worker against the 60s polling loop.
    """
    source = (source or "snapshot").strip().lower()
    if source not in {"snapshot", "live"}:
        raise HTTPException(
            status_code=400,
            detail="source must be one of: snapshot, live",
        )

    # --- 1. Try persisted snapshot ------------------------------------------
    if source == "snapshot":
        try:
            from app.models import TFTPredictionSnapshot

            with SessionLocal() as session:
                latest = (
                    session.query(TFTPredictionSnapshot)
                    .filter(TFTPredictionSnapshot.symbol == symbol)
                    .order_by(TFTPredictionSnapshot.generated_at.desc())
                    .first()
                )
                if latest is not None and isinstance(latest.payload_json, dict):
                    payload = dict(latest.payload_json)
                    gen_at = latest.generated_at
                    if gen_at and gen_at.tzinfo is None:
                        gen_at = gen_at.replace(tzinfo=timezone.utc)
                    payload["source"] = "snapshot"
                    payload["snapshot_generated_at"] = (
                        gen_at.isoformat() if gen_at else None
                    )
                    return payload
        except Exception as exc:
            logger.warning(
                "TFT snapshot read failed, falling back to live inference: %s",
                exc,
            )
        # No snapshot yet — silently fall through to live inference so the
        # UI can still show something on first deployment.

    # --- 2. Live inference (explicit request or snapshot miss) --------------
    now = datetime.now(timezone.utc)
    cache_key = f"{symbol}:live"
    cached = _tft_cache.get(cache_key)
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

        result = dict(result)
        result["source"] = "live"
        _tft_cache[cache_key] = {"data": result, "ts": now}
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
    from app.quality_gate import evaluate_quality_gate
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
            from .features import describe_feature

            artifact_dir = pathlib.Path(config.get("feature_store", {}).get("artifact_dir", "artifacts/feature_store"))
            mrmr_path = artifact_dir / "latest" / "mrmr_results.json"
            if mrmr_path.exists():
                mrmr_data = json.loads(mrmr_path.read_text(encoding="utf-8"))
                for feat, imp in mrmr_data.get("scores", {}).items():
                    meta_desc = describe_feature(feat)
                    variable_importance.append({
                        "feature": feat,
                        "importance": float(imp),
                        "label": meta_desc["label"],
                        "description": meta_desc["description"],
                        "category": meta_desc["category"],
                        "time_horizon": meta_desc.get("time_horizon", ""),
                    })
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
    summary="Get latest backtest report",
    description=(
        "Returns the latest walk-forward backtest results and Theta "
        "comparison. Prefers DB-persisted reports; falls back to "
        "filesystem artifacts. Returns a structured empty state (HTTP 200 "
        "with available=false) when no report has been produced yet, so "
        "the frontend can render a clean zero-state instead of surfacing "
        "a 404 error."
    ),
)
async def get_latest_backtest(symbol: str = Query(default="HG=F", description="Target symbol")):
    import pathlib
    import json as _json

    from app.models import BacktestReport

    empty_payload = {
        "available": False,
        "message": "No backtest runs yet. Run `python -m backend.backtest.runner` to generate one.",
        "report_date": None,
        "summary_metrics": {},
        "window_metrics": [],
        "theta_comparison": {},
        "verdict": None,
    }

    # 1. Prefer DB-persisted row (production-friendly across container restarts)
    try:
        with SessionLocal() as session:
            row = (
                session.query(BacktestReport)
                .filter(BacktestReport.symbol == symbol)
                .order_by(BacktestReport.generated_at.desc())
                .first()
            )
            if row is not None:
                return {
                    "available": True,
                    "report_date": row.generated_at.isoformat() if row.generated_at else None,
                    "summary_metrics": row.summary_json or {},
                    "window_metrics": row.windows_json or [],
                    "theta_comparison": row.theta_comparison_json or {},
                    "verdict": row.verdict,
                }
    except Exception as e:
        logger.warning(f"BacktestReport table read failed, falling back to FS: {e}")

    # 2. Fallback: legacy filesystem artifact (local dev)
    backtest_dir = pathlib.Path("artifacts/backtest")
    if backtest_dir.exists():
        reports = list(backtest_dir.glob("backtest_*.json"))
        if reports:
            latest_report_path = max(reports, key=lambda p: p.stat().st_mtime)
            try:
                data = _json.loads(latest_report_path.read_text(encoding="utf-8"))
                tft_bt = data.get("tft_backtest", {})
                comp = data.get("baseline_comparison", {})
                return {
                    "available": True,
                    "report_date": data.get("timestamp") or data.get("generated_at"),
                    "summary_metrics": tft_bt.get("summary", {}),
                    "window_metrics": tft_bt.get("windows", []),
                    "theta_comparison": comp,
                    "verdict": comp.get("verdict"),
                }
            except Exception as e:
                logger.error(f"Error reading backtest report: {e}")

    # 3. Empty state (no 404, no error)
    return empty_payload


# =============================================================================
# Sentiment Summary — Stable, DB-backed, NO LLM on the hot path.
# =============================================================================
#
# Architecture contract (frontend should depend on this shape forever):
#   - `index`:          blended daily sentiment in [-1, +1]
#   - `label`:          Bullish / Neutral / Bearish (derived from `index`)
#   - `source`:         which aggregate layer produced the value
#                       ("daily_v2" | "rolling_v2" | "legacy_v1" | "none")
#   - `components`:     breakdown of LLM vs FinBERT vs rule_sign contributions
#   - `trend_7d`:       list of {date, index, news_count} for sparkline
#   - `recent_articles`: a small sample of latest processed headlines
#   - `data_freshness`: {oldest, newest, age_hours, article_count_24h}
#
# This endpoint NEVER calls an LLM. Commentary generation (which does use
# OpenRouter) is pipeline-driven and cached in `AICommentary`.
# =============================================================================

@app.get(
    "/api/sentiment/summary",
    summary="Stable sentiment summary (DB-backed, no LLM on hot path)",
    description=(
        "Returns a hybrid sentiment summary that blends FinBERT, rule-based "
        "commodity heuristics and cached LLM impact scores. Falls back "
        "gracefully when individual sources are missing."
    ),
)
async def get_sentiment_summary(
    days: int = Query(default=7, ge=1, le=30, description="Trend window in days"),
    recent_limit: int = Query(default=6, ge=1, le=20, description="Recent headlines to include"),
):
    from sqlalchemy import func, desc

    def _label(idx: float) -> str:
        if idx > 0.10:
            return "Bullish"
        if idx < -0.10:
            return "Bearish"
        return "Neutral"

    with SessionLocal() as session:
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(days=days)

        # ---- 1) Preferred source: DailySentimentV2 (commodity-aware) ----
        v2_rows = (
            session.query(DailySentimentV2)
            .filter(DailySentimentV2.date >= window_start)
            .order_by(DailySentimentV2.date.asc())
            .all()
        )

        # ---- 2) Component breakdown from NewsSentimentV2 (same window) ----
        # Published date lives on NewsRaw, so we join processed → raw.
        component_rows = (
            session.query(
                func.avg(NewsSentimentV2.impact_score_llm).label("avg_llm"),
                func.avg(NewsSentimentV2.finbert_pos - NewsSentimentV2.finbert_neg).label("avg_finbert"),
                func.avg(NewsSentimentV2.rule_sign).label("avg_rule"),
                func.avg(NewsSentimentV2.confidence_calibrated).label("avg_conf"),
                func.avg(NewsSentimentV2.relevance_score).label("avg_rel"),
                func.count(NewsSentimentV2.id).label("n"),
            )
            .join(NewsProcessed, NewsProcessed.id == NewsSentimentV2.news_processed_id)
            .join(NewsRaw, NewsRaw.id == NewsProcessed.raw_id)
            .filter(NewsRaw.published_at >= window_start)
            .one()
        )

        # ---- 3) Pick the freshest possible index ----
        index_val: float = 0.0
        source = "none"
        avg_confidence: Optional[float] = None

        if v2_rows:
            latest_v2 = v2_rows[-1]
            index_val = float(latest_v2.sentiment_index or 0.0)
            avg_confidence = float(latest_v2.avg_confidence or 0.0) if latest_v2.avg_confidence is not None else None
            source = "daily_v2"
        elif component_rows and component_rows.n and component_rows.n > 0:
            # No daily aggregate yet — fall back to rolling per-article avg
            llm = float(component_rows.avg_llm or 0.0)
            fb = float(component_rows.avg_finbert or 0.0)
            rule = float(component_rows.avg_rule or 0.0)
            index_val = 0.5 * llm + 0.3 * fb + 0.2 * rule
            avg_confidence = float(component_rows.avg_conf or 0.0)
            source = "rolling_v2"
        else:
            # Last-ditch fallback: legacy DailySentiment
            legacy = (
                session.query(DailySentiment)
                .order_by(DailySentiment.date.desc())
                .first()
            )
            if legacy is not None:
                index_val = float(legacy.sentiment_index or 0.0)
                source = "legacy_v1"

        # ---- 4) Build trend series for sparkline ----
        trend_7d = [
            {
                "date": r.date.isoformat() if r.date else None,
                "index": float(r.sentiment_index or 0.0),
                "news_count": int(r.news_count or 0),
            }
            for r in v2_rows
        ]

        # ---- 5) Recent articles (hybrid: raw news + processed + V2 score) ----
        recent_q = (
            session.query(NewsRaw, NewsProcessed, NewsSentimentV2)
            .join(NewsProcessed, NewsProcessed.raw_id == NewsRaw.id)
            .outerjoin(
                NewsSentimentV2,
                NewsSentimentV2.news_processed_id == NewsProcessed.id,
            )
            .order_by(desc(NewsRaw.published_at))
            .limit(recent_limit)
            .all()
        )
        recent_articles = []
        for raw, proc, score in recent_q:
            recent_articles.append({
                "title": getattr(raw, "title", None) or getattr(proc, "canonical_title", None) or "",
                "source": getattr(raw, "source", None),
                "url": getattr(raw, "url", None),
                "published_at": raw.published_at.isoformat() if getattr(raw, "published_at", None) else None,
                "sentiment": {
                    "label": score.label if score else None,
                    "final_score": float(score.final_score) if score else None,
                    "relevance": float(score.relevance_score) if score else None,
                    "confidence": float(score.confidence_calibrated) if score else None,
                    "event_type": score.event_type if score else None,
                } if score else None,
            })

        # ---- 6) Data freshness (lives on NewsRaw, not Processed) ----
        freshness_q = session.query(
            func.min(NewsRaw.published_at).label("oldest"),
            func.max(NewsRaw.published_at).label("newest"),
            func.count(NewsRaw.id).label("n_total"),
        ).filter(NewsRaw.published_at >= (now - timedelta(hours=24))).one()

        newest = freshness_q.newest
        age_hours = ((now - newest).total_seconds() / 3600.0) if newest else None

        return {
            "index": round(float(index_val), 4),
            "label": _label(index_val),
            "source": source,
            "components": {
                "llm_impact_avg": float(component_rows.avg_llm) if component_rows.avg_llm is not None else None,
                "finbert_pn_avg": float(component_rows.avg_finbert) if component_rows.avg_finbert is not None else None,
                "rule_sign_avg": float(component_rows.avg_rule) if component_rows.avg_rule is not None else None,
                "avg_confidence": avg_confidence,
                "avg_relevance": float(component_rows.avg_rel) if component_rows.avg_rel is not None else None,
                "sample_size": int(component_rows.n or 0),
            },
            "trend": trend_7d,
            "recent_articles": recent_articles,
            "data_freshness": {
                "newest": newest.isoformat() if newest else None,
                "oldest": freshness_q.oldest.isoformat() if freshness_q.oldest else None,
                "age_hours": round(age_hours, 2) if age_hours is not None else None,
                "article_count_24h": int(freshness_q.n_total or 0),
            },
            "generated_at": now.isoformat(),
        }


# =============================================================================
# News intelligence endpoints
# =============================================================================
#
# Serves the Overview right-sidebar news feed. Reads from the news_raw/
# news_processed/news_sentiments_v2 pipeline the daily worker already fills —
# no LLM is invoked on the hot path.
#
# Source taxonomy:
#   * channel   = ingestion channel (NewsRaw.source): "google_news" | "newsapi"
#   * publisher = original publisher (raw_payload.source): Reuters, Mining.com…
# =============================================================================

_news_list_cache: dict[tuple, tuple[float, dict]] = {}
_news_stats_cache: dict[int, tuple[float, dict]] = {}
_NEWS_LIST_TTL_S = 60.0
_NEWS_STATS_TTL_S = 120.0
_VALID_LABELS = {"BULLISH", "BEARISH", "NEUTRAL"}


def _extract_publisher(raw_payload) -> Optional[str]:
    """Pull the original publisher name out of a NewsRaw.raw_payload blob."""
    if not raw_payload:
        return None
    if isinstance(raw_payload, str):
        try:
            import json as _json
            raw_payload = _json.loads(raw_payload)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw_payload, dict):
        return None
    src = raw_payload.get("source")
    if isinstance(src, dict):
        name = src.get("name") or src.get("title")
        return str(name) if name else None
    if isinstance(src, str) and src.strip():
        return src.strip()
    name = raw_payload.get("publisher") or raw_payload.get("author")
    return str(name) if name else None


def _build_news_sentiment_block(sent: Optional[NewsSentimentV2]) -> Optional[NewsSentimentBlock]:
    if sent is None:
        return None
    return NewsSentimentBlock(
        label=sent.label,
        final_score=float(sent.final_score) if sent.final_score is not None else None,
        impact_score_llm=float(sent.impact_score_llm) if sent.impact_score_llm is not None else None,
        confidence=float(sent.confidence_calibrated) if sent.confidence_calibrated is not None else None,
        relevance=float(sent.relevance_score) if sent.relevance_score is not None else None,
        event_type=sent.event_type,
        finbert=NewsFinbertProbs(
            pos=float(sent.finbert_pos or 0.0),
            neu=float(sent.finbert_neu or 0.0),
            neg=float(sent.finbert_neg or 0.0),
        ),
        reasoning=_extract_reasoning_text(sent.reasoning_json),
        scored_at=sent.scored_at.isoformat() if sent.scored_at else None,
    )


def _extract_reasoning_text(reasoning_json: Optional[str]) -> Optional[str]:
    """Pull a short human-readable rationale out of the cached JSON blob."""
    if not reasoning_json:
        return None
    try:
        import json as _json
        blob = _json.loads(reasoning_json)
    except (ValueError, TypeError):
        return str(reasoning_json)[:500] if reasoning_json else None
    if isinstance(blob, dict):
        for key in ("reasoning", "rationale", "summary", "explanation"):
            val = blob.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:500]
        return None
    if isinstance(blob, str):
        return blob[:500]
    return None


@app.get(
    "/api/news",
    response_model=NewsListResponse,
    summary="Paginated news feed with sentiment annotations",
)
async def get_news_feed(
    limit: int = Query(default=20, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    since_hours: int = Query(default=48, ge=1, le=168),
    label: str = Query(default="all"),
    event_type: str = Query(default="all"),
    min_relevance: float = Query(default=0.0, ge=0.0, le=1.0),
    channel: str = Query(default="all"),
    publisher: Optional[str] = Query(default=None, max_length=200),
    search: Optional[str] = Query(default=None, max_length=200),
):
    from sqlalchemy import desc as _desc

    filters_echo = {
        "limit": limit,
        "offset": offset,
        "since_hours": since_hours,
        "label": label,
        "event_type": event_type,
        "min_relevance": min_relevance,
        "channel": channel,
        "publisher": publisher,
        "search": search,
    }
    cache_key = tuple(sorted(filters_echo.items()))
    now_ts = datetime.now(timezone.utc).timestamp()
    cached = _news_list_cache.get(cache_key)
    if cached and (now_ts - cached[0]) < _NEWS_LIST_TTL_S:
        return cached[1]

    label_upper = label.upper()
    if label_upper != "ALL" and label_upper not in _VALID_LABELS:
        raise HTTPException(status_code=400, detail=f"Invalid label '{label}'")

    with SessionLocal() as session:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=since_hours)

        q = (
            session.query(NewsRaw, NewsProcessed, NewsSentimentV2)
            .join(NewsProcessed, NewsProcessed.raw_id == NewsRaw.id)
            .outerjoin(
                NewsSentimentV2,
                NewsSentimentV2.news_processed_id == NewsProcessed.id,
            )
            .filter(NewsRaw.published_at >= cutoff)
        )

        if channel.lower() != "all":
            q = q.filter(NewsRaw.source == channel)
        if event_type.lower() != "all":
            q = q.filter(NewsSentimentV2.event_type == event_type)
        if label_upper != "ALL":
            q = q.filter(NewsSentimentV2.label == label_upper)
        if min_relevance > 0:
            q = q.filter(NewsSentimentV2.relevance_score >= min_relevance)
        if search:
            q = q.filter(NewsRaw.title.ilike(f"%{search}%"))

        q = q.order_by(_desc(NewsRaw.published_at))

        publisher_needle = publisher.strip().lower() if publisher and publisher.strip() else None

        if publisher_needle:
            # Publisher filter requires JSON extraction; do it in Python to
            # remain backend-agnostic (sqlite/postgres) and keep the endpoint
            # simple. Scope is bounded by the time window filter above.
            rows = q.limit(500).all()
            filtered = [
                triple for triple in rows
                if (
                    _extract_publisher(triple[0].raw_payload) or ""
                ).lower().find(publisher_needle) >= 0
            ]
            total = len(filtered)
            page_rows = filtered[offset: offset + limit]
        else:
            total = q.count()
            page_rows = q.offset(offset).limit(limit).all()

        items: list[NewsItem] = []
        for raw, processed, sentiment in page_rows:
            items.append(
                NewsItem(
                    id=int(processed.id),
                    raw_id=int(raw.id),
                    title=str(raw.title or ""),
                    description=str(raw.description or "") or None,
                    url=str(raw.url or "") or None,
                    channel=str(raw.source or "unknown"),
                    publisher=_extract_publisher(raw.raw_payload),
                    source_feed=str(raw.source_feed or "") or None,
                    published_at=raw.published_at.isoformat() if raw.published_at else None,
                    fetched_at=raw.fetched_at.isoformat() if raw.fetched_at else None,
                    language=str(processed.language or "") or None,
                    sentiment=_build_news_sentiment_block(sentiment),
                )
            )

        response = NewsListResponse(
            items=items,
            total=int(total),
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < int(total),
            generated_at=now.isoformat(),
            filters=filters_echo,
        )

    payload = response.model_dump()
    _news_list_cache[cache_key] = (now_ts, payload)
    # Trim cache to avoid unbounded growth.
    if len(_news_list_cache) > 128:
        oldest = sorted(_news_list_cache.items(), key=lambda kv: kv[1][0])[: len(_news_list_cache) - 128]
        for k, _ in oldest:
            _news_list_cache.pop(k, None)
    return payload


@app.get(
    "/api/news/stats",
    response_model=NewsStatsResponse,
    summary="Aggregate stats for the news sidebar header",
)
async def get_news_stats(
    since_hours: int = Query(default=24, ge=1, le=168),
):
    now_ts = datetime.now(timezone.utc).timestamp()
    cached = _news_stats_cache.get(since_hours)
    if cached and (now_ts - cached[0]) < _NEWS_STATS_TTL_S:
        return cached[1]

    with SessionLocal() as session:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=since_hours)

        rows = (
            session.query(NewsRaw, NewsProcessed, NewsSentimentV2)
            .join(NewsProcessed, NewsProcessed.raw_id == NewsRaw.id)
            .outerjoin(
                NewsSentimentV2,
                NewsSentimentV2.news_processed_id == NewsProcessed.id,
            )
            .filter(NewsRaw.published_at >= cutoff)
            .all()
        )

        label_dist: dict[str, int] = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
        event_dist: dict[str, int] = {}
        channel_dist: dict[str, int] = {}
        publisher_acc: dict[str, dict[str, float]] = {}
        score_sum = 0.0
        conf_sum = 0.0
        rel_sum = 0.0
        scored_count = 0
        total = len(rows)

        for raw, _processed, sent in rows:
            ch = str(raw.source or "unknown")
            channel_dist[ch] = channel_dist.get(ch, 0) + 1
            pub = _extract_publisher(raw.raw_payload)
            if pub:
                acc = publisher_acc.setdefault(pub, {"count": 0, "score_sum": 0.0})
                acc["count"] += 1
                if sent is not None and sent.final_score is not None:
                    acc["score_sum"] += float(sent.final_score)
            if sent is None:
                continue
            scored_count += 1
            if sent.label in label_dist:
                label_dist[sent.label] += 1
            else:
                label_dist[sent.label] = label_dist.get(sent.label, 0) + 1
            etype = sent.event_type or "unknown"
            event_dist[etype] = event_dist.get(etype, 0) + 1
            if sent.final_score is not None:
                score_sum += float(sent.final_score)
            if sent.confidence_calibrated is not None:
                conf_sum += float(sent.confidence_calibrated)
            if sent.relevance_score is not None:
                rel_sum += float(sent.relevance_score)

        top_publishers = sorted(
            (
                {
                    "publisher": name,
                    "count": int(data["count"]),
                    "avg_final_score": (
                        round(float(data["score_sum"]) / float(data["count"]), 4)
                        if data["count"] > 0
                        else 0.0
                    ),
                }
                for name, data in publisher_acc.items()
            ),
            key=lambda item: item["count"],
            reverse=True,
        )[:5]

        response = NewsStatsResponse(
            window_hours=since_hours,
            total_articles=total,
            scored_articles=scored_count,
            label_distribution=label_dist,
            event_type_distribution=event_dist,
            channel_distribution=channel_dist,
            top_publishers=top_publishers,
            avg_final_score=(score_sum / scored_count) if scored_count else None,
            avg_confidence=(conf_sum / scored_count) if scored_count else None,
            avg_relevance=(rel_sum / scored_count) if scored_count else None,
            generated_at=now.isoformat(),
        )

    payload = response.model_dump()
    _news_stats_cache[since_hours] = (now_ts, payload)
    return payload


@app.get(
    "/api/news/{processed_id}",
    response_model=NewsItem,
    summary="Full detail for a single news article",
)
async def get_news_item(processed_id: int):
    with SessionLocal() as session:
        row = (
            session.query(NewsRaw, NewsProcessed, NewsSentimentV2)
            .join(NewsProcessed, NewsProcessed.raw_id == NewsRaw.id)
            .outerjoin(
                NewsSentimentV2,
                NewsSentimentV2.news_processed_id == NewsProcessed.id,
            )
            .filter(NewsProcessed.id == processed_id)
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Article not found")
        raw, processed, sentiment = row
        return NewsItem(
            id=int(processed.id),
            raw_id=int(raw.id),
            title=str(raw.title or ""),
            description=str(raw.description or "") or None,
            url=str(raw.url or "") or None,
            channel=str(raw.source or "unknown"),
            publisher=_extract_publisher(raw.raw_payload),
            source_feed=str(raw.source_feed or "") or None,
            published_at=raw.published_at.isoformat() if raw.published_at else None,
            fetched_at=raw.fetched_at.isoformat() if raw.fetched_at else None,
            language=str(processed.language or "") or None,
            sentiment=_build_news_sentiment_block(sentiment),
        )


