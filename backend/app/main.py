"""
FastAPI application with /api prefix for all endpoints.

Endpoints:
- GET /api/analysis: Current analysis report
- GET /api/history: Historical price and sentiment data
- GET /api/health: System health check
"""

import logging
import json
import time
from collections import defaultdict
from dataclasses import dataclass

# Suppress httpx request logging to prevent API keys in URLs from appearing in logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends, Header, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy import func, desc

from app.db import init_db, SessionLocal, get_db_type
from app.models import NewsArticle, PriceBar, DailySentiment, DailySentimentV2, AnalysisSnapshot, NewsSentimentV2, NewsProcessed, NewsRaw, PipelineRunMetrics
from app.settings import get_settings
from app.lock import is_pipeline_locked
from app.instruments import (
    TARGET_SYMBOL,
    canonicalize_instrument_symbol,
    resolve_provider_symbol,
)
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

def _resolve_cors_origins() -> list[str]:
    settings = get_settings()
    origins = settings.cors_allowed_origins_list
    if "*" in origins and settings.environment.lower() in {"prod", "production"}:
        raise RuntimeError("CORS wildcard is forbidden in production")
    return origins


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)


# =============================================================================
# API Endpoints
# =============================================================================

def _load_cached_xgb_importance(session, symbol: str) -> list[dict]:
    """Load feature importance without running inference.

    The analysis endpoint is snapshot-only, but a snapshot can predate the
    feature-importance metadata or contain an empty list after a failed model
    metadata write. Prefer the DB copy and fall back to the model artifact
    written by the worker so the UI can still render the drivers.
    """
    from app.models import ModelMetadata

    try:
        metadata = (
            session.query(ModelMetadata)
            .filter(ModelMetadata.symbol == symbol)
            .first()
        )
        if metadata and metadata.importance_json:
            importance = json.loads(metadata.importance_json)
            if isinstance(importance, list) and importance:
                return importance[:10]
    except Exception as exc:
        logger.warning("Could not load XGBoost importance from DB: %s", exc)

    try:
        model_dir = Path(get_settings().model_dir)
        importance_path = model_dir / (
            f"xgb_{symbol.replace('=', '_')}_latest.importance.json"
        )
        if importance_path.exists():
            importance = json.loads(importance_path.read_text(encoding="utf-8"))
            if isinstance(importance, list) and importance:
                return importance[:10]
    except Exception as exc:
        logger.warning("Could not load XGBoost importance artifact: %s", exc)

    return []

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
    symbol: str = Query(default=TARGET_SYMBOL, description="Trading symbol")
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
        if not report["top_influencers"]:
            recovered_importance = _load_cached_xgb_importance(session, symbol)
            if recovered_importance:
                report["top_influencers"] = recovered_importance

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
    symbol: str = Query(default=TARGET_SYMBOL, description="Trading symbol"),
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
        raw_prices = session.query(
            PriceBar.date,
            PriceBar.close
        ).filter(
            PriceBar.symbol == symbol,
            PriceBar.date >= start_date
        ).order_by(PriceBar.date.asc()).all()
        
        from app.price_utils import finite_positive_price

        prices = [
            (row.date, finite_positive_price(row.close))
            for row in raw_prices
            if finite_positive_price(row.close) is not None
        ]

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
        for price_date, close in prices:
            date_str = price_date.strftime("%Y-%m-%d") if hasattr(price_date, 'strftime') else str(price_date)[:10]
            
            sent = sentiment_lookup.get(date_str)
            
            # IMPORTANT: Use explicit values, don't convert 0.0 to None
            sentiment_idx = sent["sentiment_index"] if sent is not None else None
            news_count = sent["news_count"] if sent is not None else None
            
            data_points.append(HistoryDataPoint(
                date=date_str,
                price=round(close, 4),
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
                .filter(TFTPredictionSnapshot.symbol == TARGET_SYMBOL)
                .order_by(TFTPredictionSnapshot.generated_at.desc())
                .first()
            )
            if latest_tft is not None:
                last_tft_prediction_at = _iso(latest_tft.generated_at)
                tft_reference_price_date = latest_tft.reference_price_date

            # --- Latest TFT training timestamp ------------------------------
            latest_tft_model = (
                session.query(TFTModelMetadata)
                .filter(
                    TFTModelMetadata.symbol == TARGET_SYMBOL,
                    TFTModelMetadata.quality_gate_passed.is_(True),
                )
                .order_by(TFTModelMetadata.trained_at.desc())
                .first()
            )
            if latest_tft_model is not None:
                tft_model_trained_at = _iso(latest_tft_model.trained_at)

            # --- PriceBar freshness -----------------------------------------
            from app.price_utils import latest_finite_price_bar

            latest_bar = latest_finite_price_bar(session, TARGET_SYMBOL)
            if latest_bar is not None and latest_bar.date:
                bar_date = latest_bar.date
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


def _render_heatmap_http_response(
    *,
    request: Request,
    view: str,
    persisted: dict,
    cached_at: Optional[datetime],
    expires_at: Optional[datetime],
    refresh_error: Optional[str],
    payload_count: int,
    is_stale: bool,
    refresh_in_progress: bool,
    db_ms: float,
    request_started: float,
    memo_state: str,
) -> Response:
    """Serialize one snapshot once and attach cache/performance diagnostics."""
    from app.heatmap import hierarchy_for_view

    now = datetime.now(timezone.utc)
    hierarchy_started = time.perf_counter()
    payload = hierarchy_for_view(persisted, view)
    hierarchy_ms = (time.perf_counter() - hierarchy_started) * 1000
    cache_state = (
        "empty" if payload_count == 0 else
        "refreshing" if refresh_in_progress else
        "stale" if is_stale else "fresh"
    )
    if cached_at and cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=timezone.utc)
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    age = max(0, int((now - cached_at).total_seconds())) if cached_at else 0
    remaining = max(0, int((expires_at - now).total_seconds())) if expires_at else 0
    payload["_meta"] = {
        "view": view,
        "is_stale": is_stale or payload_count == 0,
        "refresh_in_progress": refresh_in_progress,
        "last_updated_at": cached_at.isoformat() if cached_at else None,
        "next_refresh_at": expires_at.isoformat() if expires_at else None,
        "source_delay_minutes": 15,
        "payload_count": payload_count,
        "refresh_error": refresh_error,
        "cache_state": cache_state,
        "cache_age_seconds": age,
    }
    import hashlib
    etag_seed = f"{cached_at.isoformat() if cached_at else 'empty'}:{view}:{payload_count}:{refresh_error or ''}"
    etag = f'"{hashlib.sha256(etag_seed.encode()).hexdigest()[:24]}"'
    serialize_started = time.perf_counter()
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    serialize_ms = (time.perf_counter() - serialize_started) * 1000
    headers = {
        "ETag": etag,
        "Cache-Control": f"public, max-age={remaining if cache_state == 'fresh' else 0}, stale-while-revalidate={60 if cache_state == 'fresh' else 30}",
        "X-Heatmap-Cache": cache_state,
        "X-Heatmap-Cache-Age": str(age),
        "X-Heatmap-Memo": memo_state,
        "Server-Timing": (
            f"db;dur={db_ms:.2f}, hierarchy;dur={hierarchy_ms:.2f}, "
            f"serialize;dur={serialize_ms:.2f}, total;dur={(time.perf_counter() - request_started) * 1000:.2f}"
        ),
    }
    if request.headers.get("if-none-match") == etag and payload_count > 0:
        return Response(status_code=304, headers=headers)
    return Response(content=body, media_type="application/json", headers=headers)


@app.get(
    "/api/market-heatmap",
    summary="Get CopperMind universe heatmap (15-min cache)",
    description="Returns a backward-compatible theme tree or a dynamic market hierarchy.",
)
async def get_market_heatmap(
    background_tasks: BackgroundTasks,
    request: Request,
    view: str = Query(default="themes", pattern="^(themes|market)$"),
):
    from app.models import HeatmapCache
    from app.heatmap import (
        flatten_heatmap_leaves,
        get_heatmap_snapshot_memo,
        refresh_market_heatmap,
        store_heatmap_snapshot_memo,
    )

    request_started = time.perf_counter()
    now = datetime.now(timezone.utc)
    memo = get_heatmap_snapshot_memo(now)
    if memo is not None:
        memo_payload = memo["payload"]
        return _render_heatmap_http_response(
            request=request,
            view=view,
            persisted=memo_payload,
            cached_at=memo["cached_at"],
            expires_at=memo["expires_at"],
            refresh_error=memo["refresh_error"],
            payload_count=len(flatten_heatmap_leaves(memo_payload)),
            is_stale=False,
            refresh_in_progress=False,
            db_ms=0.0,
            request_started=request_started,
            memo_state="hit",
        )
    with SessionLocal() as session:
        db_started = time.perf_counter()
        cache = session.query(HeatmapCache).first()
        if cache is None:
            cache = HeatmapCache(payload_json={}, cached_at=now, expires_at=now)
            session.add(cache)
            session.flush()

        persisted = cache.payload_json if isinstance(cache.payload_json, dict) else {}
        persisted_leaves = flatten_heatmap_leaves(persisted)
        payload_count = len(persisted_leaves)
        needs_enrichment = any(not leaf.get("instrumentType") for leaf in persisted_leaves)
        refresh_in_progress = cache.refresh_started_at is not None
        if refresh_in_progress:
            refresh_started = cache.refresh_started_at
            if refresh_started.tzinfo is None:
                refresh_started = refresh_started.replace(tzinfo=timezone.utc)
            if (now - refresh_started).total_seconds() > 180:
                logger.warning("Heatmap refresh appears stuck; clearing in-flight marker")
                cache.refresh_started_at = None
                refresh_in_progress = False

        expires_at = cache.expires_at
        if expires_at and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        retry_backoff = bool(cache.refresh_error) and bool(expires_at) and now < expires_at
        is_stale = not expires_at or now >= expires_at or needs_enrichment or bool(cache.refresh_error)
        if (payload_count == 0 or is_stale) and not refresh_in_progress and not retry_backoff:
            # Persist before enqueueing so concurrent empty-cache requests do
            # not each schedule a provider refresh.
            cache.refresh_started_at = now
            session.commit()
            background_tasks.add_task(refresh_market_heatmap)
            refresh_in_progress = True
        elif session.dirty or session.new:
            session.commit()
        db_ms = (time.perf_counter() - db_started) * 1000
        cached_at = cache.cached_at
        if cached_at and cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        if not is_stale and not refresh_in_progress and cached_at and expires_at:
            store_heatmap_snapshot_memo(persisted, cached_at, expires_at, cache.refresh_error)
        return _render_heatmap_http_response(
            request=request,
            view=view,
            persisted=persisted,
            cached_at=cached_at,
            expires_at=expires_at,
            refresh_error=cache.refresh_error,
            payload_count=payload_count,
            is_stale=is_stale,
            refresh_in_progress=refresh_in_progress,
            db_ms=db_ms,
            request_started=request_started,
            memo_state="miss",
        )


@app.get("/api/market-heatmap/context", summary="Get cached news context for a heatmap category")
async def get_market_heatmap_context(
    response: Response,
    category_id: str = Query(min_length=4, max_length=64),
    view: str = Query(default="market", pattern="^(themes|market)$"),
):
    """Match seven-day cached news to a valid category without LLM work."""
    from app.heatmap import find_category, flatten_heatmap_leaves, hierarchy_for_view, news_match_scores
    from app.models import HeatmapCache

    with SessionLocal() as session:
        cache = session.query(HeatmapCache).first()
        if not cache or not isinstance(cache.payload_json, dict):
            raise HTTPException(status_code=404, detail="Heatmap snapshot is unavailable")
        category = find_category(hierarchy_for_view(cache.payload_json, view), category_id)
        if category is None:
            raise HTTPException(status_code=404, detail="Unknown heatmap category")
        leaves = flatten_heatmap_leaves(category)
        rows = (
            _news_projection_query(session)
            .filter(NewsRaw.published_at >= datetime.now(timezone.utc) - timedelta(days=7))
            .order_by(desc(NewsRaw.published_at))
            .limit(250)
            .all()
        )
        best = None
        best_score = 0
        stock_best: dict[str, tuple[int, object, object, object]] = {}
        for row in rows:
            raw, processed, sentiment = _unpack_news_projection_row(row)
            title = str(raw.title or "")
            description = str(raw.description or "")
            scores = news_match_scores(title, description, leaves)
            score = sum(scores.values())
            if score > best_score:
                best_score = score
                best = (raw, processed, sentiment)

            for symbol, stock_score in scores.items():
                previous = stock_best.get(symbol)
                if stock_score > 0 and (previous is None or stock_score > previous[0]):
                    stock_best[symbol] = (stock_score, raw, processed, sentiment)

        def serialize_news(match):
            if match is None:
                return None
            raw, processed, sentiment = match
            summary = str(raw.description or "").strip()[:360]
            if not summary and sentiment is not None:
                summary = _extract_reasoning_text(sentiment.reasoning_json) or ""
            return {
                "id": int(processed.id), "title": str(raw.title or ""),
                "summary": summary or None, "url": str(raw.url or "") or None,
                "publisher": getattr(raw, "publisher", None) or _extract_publisher(raw.raw_payload),
                "publishedAt": raw.published_at.isoformat() if raw.published_at else None,
                "sentiment": getattr(sentiment, "label", None) if sentiment else None,
            }
        news = serialize_news(best)
        stock_news = {
            symbol: serialize_news((raw, processed, sentiment))
            for symbol, (_score, raw, processed, sentiment) in stock_best.items()
        }
        response.headers["Cache-Control"] = "private, max-age=300"
        return {
            "categoryId": category_id,
            "categoryName": category.get("name"),
            "symbolCount": len(leaves),
            "news": news,
            "stockNews": stock_news,
        }



# =============================================================================
# Live Price Endpoint (Twelve Data - Real-time)
# =============================================================================

@app.get(
    "/api/live-price",
    summary="Get canonical copper futures price",
    description=(
        "Returns the canonical CopperMind target price. The project standard is "
        "COMEX copper futures via HG=F; no spot XCU/USD substitution is applied."
    )
)
async def get_live_price(
    symbol: str = Query(default=TARGET_SYMBOL, description="Canonical CopperMind symbol")
):
    """
    Get the current canonical target price.

    The helper enforces exact-instrument lookup: HG=F uses yfinance first and
    falls back to the latest DB close. Spot XCU/USD is no longer used by the UI.
    """
    try:
        from app.inference import get_current_price

        with SessionLocal() as session:
            price = get_current_price(session, symbol)
        return {
            "symbol": symbol,
            "price": round(float(price), 4) if price is not None else None,
            "error": None if price is not None else "No price data",
        }
    except Exception as e:
        logger.error("Canonical live price error: %s", e)
        return {"price": None, "error": "API error"}


# =============================================================================
# WebSocket Live Price Streaming (Yahoo Finance)
# =============================================================================

@app.websocket("/ws/live-price")
async def websocket_live_price(
    websocket: WebSocket,
    symbol: str = Query(default=TARGET_SYMBOL, description="Canonical CopperMind symbol"),
):
    """
    WebSocket endpoint for real-time copper price streaming.

    Streams the canonical CopperMind instrument from Yahoo Finance's websocket.
    TradingView uses COMEX:HG1! for the same futures contract, but Yahoo uses
    HG=F; provider-specific mapping is handled before subscribing.
    """
    import asyncio
    import json
    import websockets
    import yfinance as yf

    await websocket.accept()
    canonical_symbol = canonicalize_instrument_symbol(symbol)
    yahoo_symbol = resolve_provider_symbol(canonical_symbol, "yahoo_websocket")
    yahoo_ws_url = "wss://streamer.finance.yahoo.com/?version=2"

    def _as_float(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _as_int(value):
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    async def send_payload(payload: dict):
        await websocket.send_json(
            {
                "symbol": canonical_symbol,
                "provider_symbol": yahoo_symbol,
                "provider": "yahoo_finance",
                **payload,
            }
        )

    async def send_initial_snapshot():
        from app.inference import get_current_price

        def read_price():
            with SessionLocal() as session:
                return get_current_price(session, canonical_symbol)

        price = await asyncio.to_thread(read_price)
        if price is not None:
            await send_payload(
                {
                    "price": round(float(price), 4),
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                    "received_at": datetime.now(timezone.utc).isoformat(),
                    "source": "yfinance_snapshot",
                    "error": None,
                }
            )

    async def watch_client_disconnect():
        while True:
            await websocket.receive_text()

    async def stream_yahoo_prices():
        decoder = yf.AsyncWebSocket(verbose=False)

        while True:
            try:
                async with websockets.connect(
                    yahoo_ws_url,
                    ping_interval=15,
                    ping_timeout=10,
                    close_timeout=5,
                ) as yahoo_ws:
                    await yahoo_ws.send(json.dumps({"subscribe": [yahoo_symbol]}))
                    await send_payload(
                        {
                            "status": "subscribed",
                            "source": "yahoo_finance_websocket",
                            "error": None,
                        }
                    )

                    async for message in yahoo_ws:
                        raw = json.loads(message)
                        encoded = raw.get("message")
                        if not encoded:
                            continue

                        decoded = decoder._decode_message(encoded)
                        if decoded.get("id") not in {None, yahoo_symbol}:
                            continue

                        price = _as_float(decoded.get("price"))
                        if price is None:
                            continue

                        await send_payload(
                            {
                                "price": price,
                                "timestamp": _as_int(decoded.get("time")),
                                "received_at": datetime.now(timezone.utc).isoformat(),
                                "source": "yahoo_finance_websocket",
                                "change": _as_float(decoded.get("change")),
                                "change_percent": _as_float(decoded.get("change_percent")),
                                "day_volume": _as_float(decoded.get("day_volume")),
                                "day_high": _as_float(decoded.get("day_high")),
                                "day_low": _as_float(decoded.get("day_low")),
                                "market_hours": decoded.get("market_hours"),
                                "error": None,
                            }
                        )

            except asyncio.CancelledError:
                raise
            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.warning("Yahoo websocket stream interrupted for %s: %s", yahoo_symbol, e)
                try:
                    await send_payload(
                        {
                            "status": "reconnecting",
                            "source": "yahoo_finance_websocket",
                            "error": "stream reconnecting",
                        }
                    )
                except Exception:
                    raise WebSocketDisconnect()
                await asyncio.sleep(5)

    disconnect_task = None
    stream_task = None

    try:
        await send_initial_snapshot()
        disconnect_task = asyncio.create_task(watch_client_disconnect())
        stream_task = asyncio.create_task(stream_yahoo_prices())

        done, pending = await asyncio.wait(
            {disconnect_task, stream_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
        for task in done:
            task.result()
    except WebSocketDisconnect:
        logger.info("Client disconnected from live-price WebSocket")
    except Exception as e:
        logger.error("Yahoo live-price WebSocket error: %s", e)
        try:
            await websocket.send_json({"error": "Connection error"})
        except Exception:
            pass
    finally:
        for task in (disconnect_task, stream_task):
            if task is not None and not task.done():
                task.cancel()


# =============================================================================
# AI Commentary Endpoint
# =============================================================================

@app.get(
    "/api/commentary",
    summary="AI-generated market commentary",
    description="Returns the AI-generated analysis stored after pipeline completion."
)
async def get_commentary(
    symbol: str = Query(default=TARGET_SYMBOL, description="Symbol to get commentary for")
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
            "generation_mode": result.get("generation_mode", "unknown"),
            "model_name": result.get("model_name"),
            "fallback_reason": result.get("fallback_reason"),
        }
    else:
        return {
            "symbol": symbol,
            "commentary": None,
            "error": "No commentary available. Commentary is generated after pipeline runs.",
            "generated_at": None,
            "ai_stance": "NEUTRAL",
            "generation_mode": "unavailable",
            "model_name": None,
            "fallback_reason": "not_generated",
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
    symbol: str = TARGET_SYMBOL,
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
            from app.models import TFTPredictionSnapshot, TFTModelMetadata
            from datetime import date

            def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
                if dt is None:
                    return None
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt

            def _parse_ref_date(value: Optional[str]) -> Optional[date]:
                if not value:
                    return None
                try:
                    return datetime.strptime(value[:10], "%Y-%m-%d").date()
                except Exception:
                    return None

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
                    gen_at = _as_utc(gen_at)
                    # Only a quality-gate-passed model should invalidate the
                    # snapshot. A failed model being newer must NOT force live
                    # inference, since that would serve a rejected checkpoint.
                    model_meta = (
                        session.query(TFTModelMetadata)
                        .filter(
                            TFTModelMetadata.symbol == symbol,
                            TFTModelMetadata.quality_gate_passed.is_(True),
                        )
                        .order_by(TFTModelMetadata.trained_at.desc())
                        .first()
                    )
                    trained_at = _as_utc(model_meta.trained_at) if model_meta else None

                    # Decide whether snapshot is still valid.
                    should_fallback_live = False
                    fallback_reasons: list[str] = []

                    if trained_at and gen_at and trained_at > gen_at:
                        should_fallback_live = True
                        fallback_reasons.append(
                            "model trained after snapshot "
                            f"(trained_at={trained_at.isoformat()} > snapshot_at={gen_at.isoformat()})"
                        )

                    prediction = payload.get("prediction") if isinstance(payload.get("prediction"), dict) else {}
                    reference_price_date = (
                        latest.reference_price_date
                        or prediction.get("reference_price_date")
                    )
                    ref_date = _parse_ref_date(reference_price_date)
                    if ref_date is not None:
                        staleness_days = (datetime.now(timezone.utc).date() - ref_date).days
                        if staleness_days >= 3:
                            should_fallback_live = True
                            fallback_reasons.append(
                                f"reference_price_date stale ({reference_price_date}, {staleness_days}d)"
                            )

                    if not should_fallback_live:
                        if isinstance(prediction, dict):
                            payload.setdefault("primary_horizon", "5D")
                            payload.setdefault("primary_forecast_return", prediction.get("weekly_return"))
                            payload.setdefault("primary_forecast_q10", prediction.get("weekly_return_q10_calibrated"))
                            payload.setdefault("primary_forecast_q90", prediction.get("weekly_return_q90_calibrated"))
                            payload.setdefault("t1_return", prediction.get("predicted_return_median"))
                            payload.setdefault("t1_impulse", payload.get("t1_impulse"))
                            payload.setdefault("return_space", prediction.get("return_space"))
                        payload["source"] = "snapshot"
                        payload["snapshot_generated_at"] = (
                            gen_at.isoformat() if gen_at else None
                        )
                        return payload

                    logger.info(
                        "TFT snapshot bypassed for %s; using live inference (%s)",
                        symbol,
                        "; ".join(fallback_reasons),
                    )
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


_PIPELINE_AUTH_FAILURES: dict[str, list[datetime]] = defaultdict(list)


def _pipeline_auth_key(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _record_pipeline_auth_failure(key: str) -> None:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=10)
    recent = [ts for ts in _PIPELINE_AUTH_FAILURES[key] if ts >= cutoff]
    recent.append(now)
    _PIPELINE_AUTH_FAILURES[key] = recent
    if len(recent) > 5:
        raise HTTPException(
            status_code=429,
            detail="Too many invalid pipeline trigger attempts",
        )


def verify_pipeline_secret(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> None:
    """
    Verify the pipeline trigger secret from Authorization header.
    
    Expected format: Authorization: Bearer <PIPELINE_TRIGGER_SECRET>
    """
    settings = get_settings()
    
    auth_key = _pipeline_auth_key(request)

    # If no secret is configured, reject all requests (fail secure)
    if not settings.pipeline_trigger_secret:
        logger.warning("Pipeline trigger attempted but PIPELINE_TRIGGER_SECRET not configured")
        _record_pipeline_auth_failure(auth_key)
        raise HTTPException(
            status_code=401,
            detail="Pipeline trigger authentication not configured. Set PIPELINE_TRIGGER_SECRET."
        )
    
    # Check Authorization header
    if not authorization:
        _record_pipeline_auth_failure(auth_key)
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Expected: Bearer <token>"
        )
    
    # Parse Bearer token
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        _record_pipeline_auth_failure(auth_key)
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization format. Expected: Bearer <token>"
        )
    
    token = parts[1]
    
    # Constant-time comparison to prevent timing attacks
    import secrets
    if not secrets.compare_digest(token, settings.pipeline_trigger_secret):
        logger.warning("Pipeline trigger attempted with invalid token")
        _record_pipeline_auth_failure(auth_key)
        raise HTTPException(
            status_code=401,
            detail="Invalid pipeline trigger token"
        )

    _PIPELINE_AUTH_FAILURES.pop(auth_key, None)
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
    trigger_source: str = Query(
        default="api",
        max_length=32,
        pattern="^(api|cron|manual|github-actions)$",
        description="Source of trigger (api, cron, manual, github-actions)",
    ),
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
        import uuid

        now = datetime.now(timezone.utc)
        run_id = str(uuid.uuid4())
        orphan_cutoff = now - timedelta(minutes=get_settings().pipeline_orphan_timeout_minutes)
        with SessionLocal() as session:
            orphans = session.query(PipelineRunMetrics).filter(
                PipelineRunMetrics.status.in_(("queued", "running")),
                PipelineRunMetrics.run_started_at < orphan_cutoff,
            ).all()
            for orphan in orphans:
                orphan.status = "failed"
                orphan.quality_state = "failed"
                orphan.error_message = "worker_interrupted"
                orphan.run_completed_at = now
            session.add(
                PipelineRunMetrics(
                    run_id=run_id,
                    run_started_at=now,
                    enqueued_at=now,
                    trigger_source=trigger_source,
                    train_model_requested=train_model,
                    status="queued",
                    quality_state="queued",
                )
            )
            session.commit()

        result = await enqueue_pipeline_job(
            train_model=train_model,
            trigger_source=trigger_source,
            run_id=run_id,
        )
        with SessionLocal() as session:
            queued = session.query(PipelineRunMetrics).filter(PipelineRunMetrics.run_id == run_id).first()
            if queued is not None:
                queued.job_id = result["job_id"]
                session.commit()
        
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
        if "run_id" in locals():
            with SessionLocal() as session:
                queued = session.query(PipelineRunMetrics).filter(PipelineRunMetrics.run_id == run_id).first()
                if queued is not None:
                    queued.status = "failed"
                    queued.quality_state = "failed"
                    queued.error_message = f"enqueue_failed: {str(e)[:800]}"
                    queued.run_completed_at = datetime.now(timezone.utc)
                    session.commit()
        raise HTTPException(
            status_code=503,
            detail=f"Failed to enqueue job. Redis may be unavailable: {str(e)}"
        )


@app.get(
    "/api/pipeline/runs/{run_id}",
    summary="Get terminal-aware pipeline run status (requires authentication)",
)
async def get_pipeline_run_status(
    run_id: str,
    _auth: None = Depends(verify_pipeline_secret),
):
    with SessionLocal() as session:
        record = session.query(PipelineRunMetrics).filter(PipelineRunMetrics.run_id == run_id).first()
        if record is None:
            raise HTTPException(status_code=404, detail="Pipeline run not found")
        try:
            stages = json.loads(record.stage_results_json or "{}")
        except (TypeError, ValueError):
            stages = {}
        return {
            "run_id": record.run_id,
            "status": record.status,
            "quality_state": record.quality_state,
            "enqueued_at": record.enqueued_at.isoformat() if record.enqueued_at else None,
            "worker_started_at": record.worker_started_at.isoformat() if record.worker_started_at else None,
            "completed_at": record.run_completed_at.isoformat() if record.run_completed_at else None,
            "error_message": record.error_message,
            "stage_results": stages,
            "fallback_counts": {
                "llm_success": record.llm_success_count or 0,
                "operational": record.operational_fallback_count or 0,
                "policy": record.policy_fallback_count or 0,
            },
            "commentary_generation_mode": record.commentary_generation_mode,
            "artifact_version": record.artifact_version,
            "promoted_artifact_version": record.promoted_artifact_version,
            "train_model_requested": bool(record.train_model_requested),
        }


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
    symbol: str = Query(default=TARGET_SYMBOL, description="Trading symbol")
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
    symbol: str = Query(default=TARGET_SYMBOL, description="Target symbol")
):
    from app.models import TFTModelMetadata
    from app.quality_gate import (
        evaluate_quality_gate_metric_warnings,
        evaluate_quality_gate_metrics,
    )
    import json
    
    import math

    def _safe_json_load(raw: Optional[str]) -> dict:
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _numeric_map(data: dict) -> dict:
        out: dict[str, float] = {}
        for k, v in data.items():
            try:
                num = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(num) or math.isinf(num):
                continue
            out[k] = num
        return out

    with SessionLocal() as session:
        # This endpoint describes the active/promoted model. Rejected candidate
        # metrics remain in the GitHub run artifact and must never replace the
        # last known-good DB row shown by the production UI.
        meta = (
            session.query(TFTModelMetadata)
            .filter(
                TFTModelMetadata.symbol == symbol,
                TFTModelMetadata.quality_gate_passed.is_(True),
            )
            .order_by(TFTModelMetadata.trained_at.desc())
            .first()
        )
        if not meta:
            raise HTTPException(
                status_code=404,
                detail=f"No quality-gate-passed TFT model metadata found for {symbol}",
            )

        config = _safe_json_load(meta.config_json)
        metrics_raw = _safe_json_load(meta.metrics_json)
        metrics = _numeric_map(metrics_raw)
        
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

        da = float(metrics.get("directional_accuracy", 0.5))
        sharpe = float(metrics.get("sharpe_ratio", 0.0))
        vr = float(metrics.get("variance_ratio", 1.0))
        tail_capture = metrics.get("tail_capture_rate")
        quantile_crossing = metrics.get("quantile_crossing_rate")
        median_gap_max = metrics.get("median_sort_gap_max")
        weekly_da = metrics.get("weekly_directional_accuracy")
        weekly_mr = metrics.get("weekly_magnitude_ratio")
        weekly_tail = metrics.get("weekly_tail_capture_rate")
        weekly_pi80 = metrics.get("weekly_pi80_coverage")
        weekly_pi80_width_ratio = metrics.get("weekly_pi80_width_ratio")
        weekly_pi96 = metrics.get("weekly_pi96_coverage")
        weekly_pi96_width_ratio = metrics.get("weekly_pi96_width_ratio")
        weekly_qcross = metrics.get("weekly_quantile_crossing_rate")
        weekly_sorted_qcross = metrics.get("weekly_sorted_quantile_crossing_rate")
        weekly_gap = metrics.get("weekly_median_sort_gap_max")
        weekly_samples = metrics.get("weekly_sample_count")
        weekly_pred_pos = metrics.get("weekly_pred_positive_rate")
        weekly_actual_pos = metrics.get("weekly_actual_positive_rate")
        weekly_raw_mr = metrics.get("weekly_raw_magnitude_ratio")
        weekly_bound_rate = metrics.get("weekly_median_bound_applied_rate")
        weekly_cap = metrics.get("weekly_median_cap")
        weekly_sharpe = metrics.get("weekly_sharpe_ratio")
        weekly_sortino = metrics.get("weekly_sortino_ratio")
        
        passed, reasons = evaluate_quality_gate_metrics(metrics)
        warnings = evaluate_quality_gate_metric_warnings(metrics)
        
        gate_metrics = {
            "da": da,
            "sharpe": sharpe,
            "vr": vr,
        }
        if tail_capture is not None:
            gate_metrics["tail_capture"] = float(tail_capture)
        if quantile_crossing is not None:
            gate_metrics["quantile_crossing_rate"] = float(quantile_crossing)
        if median_gap_max is not None:
            gate_metrics["median_sort_gap_max"] = float(median_gap_max)
        for name, value in {
            "weekly_directional_accuracy": weekly_da,
            "weekly_magnitude_ratio": weekly_mr,
            "weekly_tail_capture_rate": weekly_tail,
            "weekly_pi80_coverage": weekly_pi80,
            "weekly_pi80_width_ratio": weekly_pi80_width_ratio,
            "weekly_pi96_coverage": weekly_pi96,
            "weekly_pi96_width_ratio": weekly_pi96_width_ratio,
            "weekly_quantile_crossing_rate": weekly_qcross,
            "weekly_sorted_quantile_crossing_rate": weekly_sorted_qcross,
            "weekly_median_sort_gap_max": weekly_gap,
            "weekly_sample_count": weekly_samples,
            "weekly_raw_magnitude_ratio": weekly_raw_mr,
            "weekly_median_bound_applied_rate": weekly_bound_rate,
            "weekly_median_cap": weekly_cap,
            "weekly_sharpe_ratio": weekly_sharpe,
            "weekly_sortino_ratio": weekly_sortino,
        }.items():
            if value is not None:
                gate_metrics[name] = float(value)

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
                "warnings": warnings,
                "metrics": gate_metrics,
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
async def get_latest_backtest(symbol: str = Query(default=TARGET_SYMBOL, description="Target symbol")):
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
#   - `data_freshness`: {oldest, newest, age_hours, article_count_24h,
#                         window_start, window_days, article_count_window}
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
            session.query(
                DailySentimentV2.date,
                DailySentimentV2.sentiment_index,
                DailySentimentV2.news_count,
                DailySentimentV2.avg_confidence,
            )
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
            .filter(
                NewsRaw.published_at >= window_start,
                NewsProcessed.duplicate_of_id.is_(None),
                NewsSentimentV2.horizon_days == max(1, int(get_settings().sentiment_horizon_days)),
            )
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
            _news_projection_query(session)
            .filter(NewsRaw.published_at >= window_start)
            .order_by(desc(NewsRaw.published_at))
            .limit(recent_limit)
            .all()
        )
        recent_articles = []
        for row in recent_q:
            raw, proc, score = _unpack_news_projection_row(row)
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

        window_freshness_q = session.query(
            func.count(NewsRaw.id).label("n_total"),
        ).filter(NewsRaw.published_at >= window_start).one()

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
                "window_start": window_start.isoformat(),
                "window_days": int(days),
                "article_count_window": int(window_freshness_q.n_total or 0),
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
_news_stats_cache: dict[tuple, tuple[float, dict]] = {}
_NEWS_LIST_TTL_S = 60.0
_NEWS_STATS_TTL_S = 120.0
_VALID_LABELS = {"BULLISH", "BEARISH", "NEUTRAL"}


@dataclass(frozen=True)
class _NewsSentimentProjection:
    label: Optional[str]
    final_score: Optional[float]
    impact_score_llm: Optional[float]
    confidence_calibrated: Optional[float]
    relevance_score: Optional[float]
    event_type: Optional[str]
    finbert_pos: Optional[float]
    finbert_neu: Optional[float]
    finbert_neg: Optional[float]
    reasoning_json: Optional[str]
    scored_at: Optional[datetime]


def _news_projection_query(session, horizon_days: Optional[int] = None):
    """
    Build a backward-compatible query for news + sentiment.

    We intentionally project only stable legacy sentiment columns so this
    endpoint keeps working even before weekly-contract migrations are applied
    on older databases.
    """
    if horizon_days is None:
        horizon_days = max(1, int(get_settings().sentiment_horizon_days))
    return (
        session.query(
            NewsRaw,
            NewsProcessed,
            NewsSentimentV2.id.label("sent_id"),
            NewsSentimentV2.label.label("sent_label"),
            NewsSentimentV2.final_score.label("sent_final_score"),
            NewsSentimentV2.impact_score_llm.label("sent_impact_score_llm"),
            NewsSentimentV2.confidence_calibrated.label("sent_confidence_calibrated"),
            NewsSentimentV2.relevance_score.label("sent_relevance_score"),
            NewsSentimentV2.event_type.label("sent_event_type"),
            NewsSentimentV2.finbert_pos.label("sent_finbert_pos"),
            NewsSentimentV2.finbert_neu.label("sent_finbert_neu"),
            NewsSentimentV2.finbert_neg.label("sent_finbert_neg"),
            NewsSentimentV2.reasoning_json.label("sent_reasoning_json"),
            NewsSentimentV2.scored_at.label("sent_scored_at"),
        )
        .join(NewsProcessed, NewsProcessed.raw_id == NewsRaw.id)
        .outerjoin(
            NewsSentimentV2,
            (NewsSentimentV2.news_processed_id == NewsProcessed.id)
            & (NewsSentimentV2.horizon_days == horizon_days),
        )
        .filter(NewsProcessed.duplicate_of_id.is_(None))
    )


def _unpack_news_projection_row(row):
    if len(row) == 3:
        return row

    raw = row[0]
    processed = row[1]
    sent_id = row[2]
    sentiment = None
    if sent_id is not None:
        sentiment = _NewsSentimentProjection(
            label=row[3],
            final_score=float(row[4]) if row[4] is not None else None,
            impact_score_llm=float(row[5]) if row[5] is not None else None,
            confidence_calibrated=float(row[6]) if row[6] is not None else None,
            relevance_score=float(row[7]) if row[7] is not None else None,
            event_type=row[8],
            finbert_pos=float(row[9]) if row[9] is not None else None,
            finbert_neu=float(row[10]) if row[10] is not None else None,
            finbert_neg=float(row[11]) if row[11] is not None else None,
            reasoning_json=row[12],
            scored_at=row[13],
        )
    return raw, processed, sentiment


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


def _build_news_sentiment_block(sent: Optional[_NewsSentimentProjection]) -> Optional[NewsSentimentBlock]:
    if sent is None:
        return None
    metadata: dict = {}
    try:
        metadata = json.loads(sent.reasoning_json or "{}")
        if not isinstance(metadata, dict):
            metadata = {}
    except (TypeError, ValueError):
        metadata = {}
    finbert_available = metadata.get("finbert_available", True)
    return NewsSentimentBlock(
        label=sent.label,
        final_score=float(sent.final_score) if sent.final_score is not None else None,
        impact_score_llm=float(sent.impact_score_llm) if sent.impact_score_llm is not None else None,
        confidence=float(sent.confidence_calibrated) if sent.confidence_calibrated is not None else None,
        relevance=float(sent.relevance_score) if sent.relevance_score is not None else None,
        event_type=sent.event_type,
        finbert=(
            NewsFinbertProbs(
                pos=float(sent.finbert_pos),
                neu=float(sent.finbert_neu),
                neg=float(sent.finbert_neg),
            )
            if finbert_available and all(v is not None for v in (sent.finbert_pos, sent.finbert_neu, sent.finbert_neg))
            else None
        ),
        reasoning=_extract_reasoning_text(sent.reasoning_json),
        scored_at=sent.scored_at.isoformat() if sent.scored_at else None,
        scoring_mode="deterministic_fallback" if metadata.get("fallback_used") else "llm",
        model_name=metadata.get("llm_model"),
        fallback_reason=metadata.get("fallback_reason"),
    )


def _completed_pipeline_cache_version(session) -> str:
    try:
        latest = (
            session.query(PipelineRunMetrics.run_id)
            .filter(PipelineRunMetrics.status.in_(("success", "degraded")))
            .order_by(PipelineRunMetrics.run_completed_at.desc())
            .first()
        )
        return str(latest[0]) if latest else "no-completed-run"
    except Exception:
        return "no-completed-run"


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
    as_of: Optional[datetime] = Query(default=None, description="Stable pagination cutoff from the first page"),
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
    label_upper = label.upper()
    if label_upper != "ALL" and label_upper not in _VALID_LABELS:
        raise HTTPException(status_code=400, detail=f"Invalid label '{label}'")

    with SessionLocal() as session:
        now = datetime.now(timezone.utc)
        stable_as_of = as_of or now.replace(microsecond=0)
        if stable_as_of.tzinfo is None:
            stable_as_of = stable_as_of.replace(tzinfo=timezone.utc)
        cutoff = stable_as_of - timedelta(hours=since_hours)
        cache_key = tuple(sorted({**filters_echo, "as_of": stable_as_of.isoformat(), "pipeline": _completed_pipeline_cache_version(session)}.items()))
        now_ts = now.timestamp()
        cached = _news_list_cache.get(cache_key)
        if cached and (now_ts - cached[0]) < _NEWS_LIST_TTL_S:
            return cached[1]

        q = _news_projection_query(session).filter(
            NewsRaw.published_at >= cutoff,
            NewsRaw.fetched_at <= stable_as_of,
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
        if publisher and publisher.strip():
            q = q.filter(NewsRaw.publisher.ilike(f"%{publisher.strip()}%"))

        data_as_of_value = q.order_by(None).with_entities(
            func.max(NewsSentimentV2.scored_at)
        ).scalar()
        q = q.order_by(_desc(NewsRaw.published_at), _desc(NewsProcessed.id))
        total = q.count()
        page_rows = q.offset(offset).limit(limit).all()

        items: list[NewsItem] = []
        for row in page_rows:
            raw, processed, sentiment = _unpack_news_projection_row(row)
            items.append(
                NewsItem(
                    id=int(processed.id),
                    raw_id=int(raw.id),
                    title=str(raw.title or ""),
                    description=str(raw.description or "") or None,
                    url=str(raw.url or "") or None,
                    channel=str(raw.source or "unknown"),
                    publisher=getattr(raw, "publisher", None) or _extract_publisher(raw.raw_payload),
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
            as_of=stable_as_of.isoformat(),
            data_as_of=data_as_of_value.isoformat() if data_as_of_value else None,
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
    label: str = Query(default="all"),
    event_type: str = Query(default="all"),
    min_relevance: float = Query(default=0.0, ge=0.0, le=1.0),
    channel: str = Query(default="all"),
    publisher: Optional[str] = Query(default=None, max_length=200),
    search: Optional[str] = Query(default=None, max_length=200),
):
    from sqlalchemy import desc as _desc

    filters_echo = {
        "since_hours": since_hours,
        "label": label,
        "event_type": event_type,
        "min_relevance": min_relevance,
        "channel": channel,
        "publisher": publisher,
        "search": search,
    }
    label_upper = label.upper()
    if label_upper != "ALL" and label_upper not in _VALID_LABELS:
        raise HTTPException(status_code=400, detail=f"Invalid label '{label}'")

    with SessionLocal() as session:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=since_hours)
        cache_key = tuple(sorted({**filters_echo, "pipeline": _completed_pipeline_cache_version(session)}.items()))
        now_ts = now.timestamp()
        cached = _news_stats_cache.get(cache_key)
        if cached and (now_ts - cached[0]) < _NEWS_STATS_TTL_S:
            return cached[1]

        q = _news_projection_query(session).filter(NewsRaw.published_at >= cutoff)

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
        if publisher and publisher.strip():
            q = q.filter(NewsRaw.publisher.ilike(f"%{publisher.strip()}%"))

        q = q.order_by(_desc(NewsRaw.published_at))

        rows = q.all()

        label_dist: dict[str, int] = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
        event_dist: dict[str, int] = {}
        channel_dist: dict[str, int] = {}
        publisher_acc: dict[str, dict[str, float]] = {}
        score_sum = 0.0
        conf_sum = 0.0
        rel_sum = 0.0
        scored_count = 0
        total = len(rows)

        for row in rows:
            raw, _processed, sent = _unpack_news_projection_row(row)
            ch = str(raw.source or "unknown")
            channel_dist[ch] = channel_dist.get(ch, 0) + 1
            pub = getattr(raw, "publisher", None) or _extract_publisher(raw.raw_payload)
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
    _news_stats_cache[cache_key] = (now_ts, payload)
    return payload


@app.get(
    "/api/news/{processed_id}",
    response_model=NewsItem,
    summary="Full detail for a single news article",
)
async def get_news_item(processed_id: int):
    with SessionLocal() as session:
        row = (
            _news_projection_query(session)
            .filter(NewsProcessed.id == processed_id)
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Article not found")
        raw, processed, sentiment = _unpack_news_projection_row(row)
        return NewsItem(
            id=int(processed.id),
            raw_id=int(raw.id),
            title=str(raw.title or ""),
            description=str(raw.description or "") or None,
            url=str(raw.url or "") or None,
            channel=str(raw.source or "unknown"),
            publisher=getattr(raw, "publisher", None) or _extract_publisher(raw.raw_payload),
            source_feed=str(raw.source_feed or "") or None,
            published_at=raw.published_at.isoformat() if raw.published_at else None,
            fetched_at=raw.fetched_at.isoformat() if raw.fetched_at else None,
            language=str(processed.language or "") or None,
            sentiment=_build_news_sentiment_block(sentiment),
        )


