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

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
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
            logger.debug(f"Cached snapshot exists, but running live prediction for accuracy")
            import yfinance as yf
            import xgboost as xgb
            
            try:
                # Get live price from yfinance
                ticker = yf.Ticker(symbol)
                info = ticker.info
                live_price = info.get('regularMarketPrice') or info.get('currentPrice')
                
                if live_price is not None:
                    cached['current_price'] = round(float(live_price), 4)
                    
                    # Get latest DB close price for prediction base
                    # Model predicts based on historical closes, not intraday prices
                    latest_bar = session.query(PriceBar).filter(
                        PriceBar.symbol == symbol
                    ).order_by(PriceBar.date.desc()).first()
                    if live_price is not None:
                        # Prioritize live price for prediction base
                        prediction_base = float(live_price)
                    elif latest_bar:
                        # Fallback to DB close
                        prediction_base = latest_bar.close
                    else:
                        prediction_base = 0.0
                    
                    # Run LIVE model prediction
                    from app.ai_engine import load_model, load_model_metadata
                    from app.inference import build_features_for_prediction
                    
                    model = load_model(symbol)
                    metadata = load_model_metadata(symbol)
                    features = metadata.get("features", [])
                    
                    if model and features:
                        # Build features and predict
                        X = build_features_for_prediction(session, symbol, features)
                        if X is not None and not X.empty:
                            dmatrix = xgb.DMatrix(X, feature_names=features)
                            predicted_return = float(model.predict(dmatrix)[0])
                            
                            # Update with live prediction
                            # Use DB close as prediction base (model trained on closes)
                            cached['predicted_return'] = round(predicted_return, 6)
                            cached['predicted_price'] = round(
                                float(prediction_base) * (1 + predicted_return),
                                4
                            )
                            
                            # Update confidence bounds (based on prediction base)
                            std_mult = 1.0  # 1 standard deviation
                            cached['confidence_lower'] = round(float(prediction_base) * (1 - std_mult * abs(predicted_return)), 4)
                            cached['confidence_upper'] = round(float(prediction_base) * (1 + std_mult * abs(predicted_return) * 2), 4)
                            
                            logger.info(f"LIVE prediction: close=${prediction_base:.4f}, predicted=${cached['predicted_price']:.4f} ({predicted_return*100:.2f}%)")
                    
            except Exception as e:
                logger.error(f"Live prediction failed, using cached: {e}")
            
            # Update top_influencers from current model metadata
            try:
                from app.ai_engine import load_model_metadata
                from app.features import get_feature_descriptions
                
                metadata = load_model_metadata(symbol)
                importance = metadata.get("importance", [])
                
                if importance:
                    descriptions = get_feature_descriptions()
                    top_influencers = []
                    
                    for item in importance[:10]:
                        feat = item["feature"]
                        desc = None
                        for key, value in descriptions.items():
                            if key in feat:
                                desc = value
                                break
                        if desc is None:
                            desc = feat.replace("_", " ").replace("  ", " ").title()
                        
                        top_influencers.append({
                            "feature": feat,
                            "importance": item["importance"],
                            "description": desc,
                        })
                    
                    cached['top_influencers'] = top_influencers
                    logger.info(f"Updated cached snapshot with fresh influencers from model")
            except Exception as e:
                logger.debug(f"Could not update influencers in cached snapshot: {e}")
            
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
        logger.error(f"Twelve Data API error: {e}")
        return {"price": None, "error": str(e)}


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
    
    td_ws_url = f"wss://ws.twelvedata.com/v1/price?apikey={settings.twelvedata_api_key}"
    
    try:
        async with websockets.connect(td_ws_url) as td_ws:
            # Subscribe to XCU/USD (Copper spot)
            subscribe_msg = json.dumps({
                "action": "subscribe",
                "params": {"symbols": "XCU/USD"}
            })
            await td_ws.send(subscribe_msg)
            logger.info("Subscribed to XCU/USD via Twelve Data WebSocket")
            
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
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
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
                
                logger.info(f"Step 2: Running AI pipeline (train_model={train_model})...")
                from app.ai_engine import run_full_pipeline
                ai_result = run_full_pipeline(
                    target_symbol="HG=F",
                    score_sentiment=True,
                    aggregate_sentiment=True,
                    train_model=train_model
                )
                logger.info(f"AI pipeline complete: scored={ai_result.get('scored_articles', 0)}, aggregated={ai_result.get('aggregated_days', 0)}")
                
                # Log model training result specifically
                if train_model:
                    model_result = ai_result.get('model_result')
                    if model_result:
                        logger.info(f"Model training SUCCESS: {model_result.get('model_path')}")
                        logger.info(f"Top influencers updated: {[i['feature'] for i in model_result.get('top_influencers', [])[:3]]}")
                    else:
                        logger.warning("Model training returned None - check for errors above")
                
                # Step 3: Generate snapshot
                logger.info("Step 3: Generating analysis snapshot...")
                with SessionLocal() as session:
                    # Clear old snapshots for this symbol to ensure fresh data
                    from app.models import AnalysisSnapshot
                    deleted = session.query(AnalysisSnapshot).filter(
                        AnalysisSnapshot.symbol == settings.target_symbol
                    ).delete()
                    if deleted:
                        session.commit()
                        logger.info(f"Cleared {deleted} old snapshot(s) for {settings.target_symbol}")
                    
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

