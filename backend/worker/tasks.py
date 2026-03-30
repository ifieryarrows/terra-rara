"""
Worker tasks for arq.

This module defines the tasks that the worker executes.
The main task is `run_pipeline` which orchestrates the entire pipeline.

Faz 2: Integrated news_raw/news_processed pipeline with proper
       commit boundaries, metrics tracking, and degraded mode handling.
"""

import logging
import os
import socket
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

# These imports will be updated as we refactor
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import SessionLocal, init_db, get_db_type
from app.settings import get_settings
from app.models import PipelineRunMetrics
from adapters.db.lock import (
    PIPELINE_LOCK_KEY,
    try_acquire_lock,
    release_lock,
    write_lock_visibility,
    clear_lock_visibility,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper functions for metrics tracking
# =============================================================================

def create_run_metrics(
    session: Session,
    run_id: str,
    started_at: datetime,
) -> PipelineRunMetrics:
    """Create initial pipeline_run_metrics record."""
    metrics = PipelineRunMetrics(
        run_id=run_id,
        run_started_at=started_at,
        status="running",
    )
    session.add(metrics)
    session.flush()
    return metrics


def update_run_metrics(
    session: Session,
    run_id: str,
    **kwargs,
) -> None:
    """Update pipeline_run_metrics with new values."""
    metrics = session.query(PipelineRunMetrics).filter(
        PipelineRunMetrics.run_id == run_id
    ).first()
    
    if metrics:
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        session.flush()


def finalize_run_metrics(
    session: Session,
    run_id: str,
    status: str,
    quality_state: str = "ok",
    error_message: Optional[str] = None,
) -> None:
    """Finalize run metrics with completion status."""
    completed_at = datetime.now(timezone.utc)
    
    metrics = session.query(PipelineRunMetrics).filter(
        PipelineRunMetrics.run_id == run_id
    ).first()
    
    if metrics:
        metrics.run_completed_at = completed_at
        metrics.status = status
        metrics.quality_state = quality_state
        if metrics.run_started_at:
            metrics.duration_seconds = (completed_at - metrics.run_started_at).total_seconds()
        if error_message:
            metrics.error_message = error_message
        session.flush()


# =============================================================================
# Main pipeline task
# =============================================================================

async def run_pipeline(
    ctx: dict,
    run_id: str,
    train_model: bool = False,
    trigger_source: str = "unknown",
    enqueued_at: str = None,
) -> dict:
    """
    Main pipeline task - executed by arq worker.
    
    This is the ONLY entrypoint for pipeline execution.
    
    Faz 2 Flow:
        Stage 1a: News ingestion → news_raw
        Stage 1b: Raw processing → news_processed
        Stage 1c: Cut-off calculation
        Stage 1d: Price ingestion
        Stage 2: Sentiment scoring
        Stage 3: Sentiment aggregation
        Stage 3.5: FinBERT embedding extraction
        Stage 4: XGBoost training (optional)
        Stage 5: XGBoost snapshot
        Stage 5.5: TFT-ASRO inference (downloads checkpoint from HF Hub)
        Stage 6: Commentary generation

    Note: TFT-ASRO full training is handled exclusively by the weekly
    tft-training.yml GitHub workflow. The daily pipeline only runs inference.

    Args:
        ctx: arq context (contains redis connection)
        run_id: Unique identifier for this run
        train_model: Whether to train the XGBoost model
        trigger_source: Where the trigger came from (cron, manual, api)
        enqueued_at: ISO timestamp when job was enqueued
        
    Returns:
        dict with run results
    """
    started_at = datetime.now(timezone.utc)
    holder_id = f"{socket.gethostname()}:{os.getpid()}"
    run_uuid = uuid.UUID(run_id) if isinstance(run_id, str) else run_id
    
    logger.info(f"[run_id={run_id}] Pipeline starting: trigger={trigger_source}, train_model={train_model}")
    
    # Initialize database
    init_db()
    
    # Get a dedicated session for this pipeline run
    # IMPORTANT: This session holds the advisory lock
    session: Session = SessionLocal()
    quality_state = "ok"
    result = {}
    
    try:
        # 0. Create run metrics record
        create_run_metrics(session, run_id, started_at)
        session.commit()
        
        # 1. Acquire distributed lock
        if not try_acquire_lock(session, PIPELINE_LOCK_KEY):
            logger.warning(f"[run_id={run_id}] Pipeline skipped: lock held by another process")
            finalize_run_metrics(session, run_id, status="skipped_locked", quality_state="skipped")
            session.commit()
            return {
                "run_id": run_id,
                "status": "skipped_locked",
                "message": "Another pipeline is running",
            }
        
        # Write lock visibility (best-effort)
        write_lock_visibility(session, PIPELINE_LOCK_KEY, run_id, holder_id)
        session.commit()
        
        logger.info(f"[run_id={run_id}] Lock acquired, executing pipeline...")
        
        # 2. Execute pipeline stages with proper commit boundaries
        result = await _execute_pipeline_stages_v2(
            session=session,
            run_id=run_id,
            run_uuid=run_uuid,
            train_model=train_model,
        )
        
        # Determine quality state from result
        # More nuanced logic to avoid false alarms
        raw_inserted = result.get("news_raw_inserted", 0)
        proc_inserted = result.get("news_processed_inserted", 0)
        raw_error = result.get("news_raw_error")
        proc_error = result.get("news_processed_error")
        
        if raw_error or proc_error:
            # Actual errors during ingestion/processing
            quality_state = "degraded"
            result["message"] = f"Pipeline errors: {raw_error or ''} {proc_error or ''}".strip()
        elif raw_inserted == 0 and proc_inserted == 0:
            # No new data at all - could be dedup working or sources haven't updated
            quality_state = "stale"
            result["message"] = "No new articles - sources may not have updated"
        elif raw_inserted > 0 and proc_inserted == 0:
            # Got raw but nothing processed - potential dedup anomaly
            quality_state = "ok"  # This is actually fine - all duplicates
            result["message"] = f"All {raw_inserted} articles were duplicates"
        else:
            quality_state = "ok"
        
        # 3. Record success
        finished_at = datetime.now(timezone.utc)
        duration = (finished_at - started_at).total_seconds()
        
        finalize_run_metrics(
            session, run_id, 
            status="success", 
            quality_state=quality_state,
        )
        session.commit()
        
        logger.info(f"[run_id={run_id}] Pipeline completed in {duration:.1f}s")
        
        return {
            "run_id": run_id,
            "status": "success",
            "quality_state": quality_state,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
            "train_model": train_model,
            **result,
        }
        
    except Exception as e:
        logger.error(f"[run_id={run_id}] Pipeline failed: {e}", exc_info=True)
        
        try:
            finalize_run_metrics(
                session, run_id,
                status="failed",
                quality_state="failed",
                error_message=str(e)[:1000],
            )
            session.commit()
        except Exception:
            session.rollback()
        
        return {
            "run_id": run_id,
            "status": "failed",
            "error": str(e),
        }
        
    finally:
        # Always release lock and cleanup
        try:
            release_lock(session, PIPELINE_LOCK_KEY)
            clear_lock_visibility(session, PIPELINE_LOCK_KEY)
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()


async def _execute_pipeline_stages_v2(
    session: Session,
    run_id: str,
    run_uuid: uuid.UUID,
    train_model: bool,
) -> dict:
    """
    Execute pipeline stages with Faz 2 news pipeline integration.
    
    Each stage has proper commit boundaries and metrics updates.
    """
    from app.settings import get_settings
    
    settings = get_settings()
    result = {}
    
    # -------------------------------------------------------------------------
    # Stage 1a: News ingestion → news_raw (FAZ 2)
    # -------------------------------------------------------------------------
    logger.info(f"[run_id={run_id}] Stage 1a: News ingestion → news_raw")
    try:
        from pipelines.ingestion.news import ingest_news_to_raw
        
        raw_stats = ingest_news_to_raw(
            session=session,
            run_id=run_uuid,
        )
        session.commit()
        
        result["news_raw_inserted"] = raw_stats.get("inserted", 0)
        result["news_raw_duplicates"] = raw_stats.get("duplicates", 0)
        
        update_run_metrics(
            session, run_id,
            news_raw_inserted=raw_stats.get("inserted", 0),
            news_raw_duplicates=raw_stats.get("duplicates", 0),
        )
        session.commit()
        
        logger.info(f"[run_id={run_id}] news_raw: {raw_stats.get('inserted', 0)} inserted")
        
    except Exception as e:
        logger.error(f"[run_id={run_id}] Stage 1a failed: {e}")
        result["news_raw_error"] = str(e)
        session.rollback()
    
    # -------------------------------------------------------------------------
    # Stage 1b: Raw → Processed (FAZ 2)
    # -------------------------------------------------------------------------
    logger.info(f"[run_id={run_id}] Stage 1b: news_raw → news_processed")
    try:
        from pipelines.processing.news import process_raw_to_processed
        
        proc_stats = process_raw_to_processed(
            session=session,
            run_id=run_uuid,
            batch_size=200,
        )
        session.commit()
        
        result["news_processed_inserted"] = proc_stats.get("inserted", 0)
        result["news_processed_duplicates"] = proc_stats.get("duplicates", 0)
        
        update_run_metrics(
            session, run_id,
            news_processed_inserted=proc_stats.get("inserted", 0),
            news_processed_duplicates=proc_stats.get("duplicates", 0),
        )
        session.commit()
        
        logger.info(f"[run_id={run_id}] news_processed: {proc_stats.get('inserted', 0)} inserted")
        
    except Exception as e:
        logger.error(f"[run_id={run_id}] Stage 1b failed: {e}")
        result["news_processed_error"] = str(e)
        session.rollback()
    
    # -------------------------------------------------------------------------
    # Stage 1c: Cut-off calculation (FAZ 2)
    # -------------------------------------------------------------------------
    logger.info(f"[run_id={run_id}] Stage 1c: Computing news cut-off")
    try:
        from pipelines.cutoff import compute_news_cutoff
        
        cutoff_dt = compute_news_cutoff(
            run_datetime=datetime.now(timezone.utc),
            market_tz=settings.market_timezone,
            market_close=settings.market_close_time,
            buffer_minutes=settings.cutoff_buffer_minutes,
        )
        
        result["news_cutoff_time"] = cutoff_dt.isoformat()
        
        update_run_metrics(session, run_id, news_cutoff_time=cutoff_dt)
        session.commit()
        
        logger.info(f"[run_id={run_id}] Cut-off: {cutoff_dt.isoformat()}")
        
    except Exception as e:
        logger.error(f"[run_id={run_id}] Stage 1c failed: {e}")
        result["cutoff_error"] = str(e)
    
    # -------------------------------------------------------------------------
    # Stage 1d: Price ingestion (existing)
    # -------------------------------------------------------------------------
    logger.info(f"[run_id={run_id}] Stage 1d: Price ingestion")
    try:
        from app.data_manager import ingest_prices
        
        price_stats = ingest_prices(session)
        session.commit()
        
        result["symbols_fetched"] = len(price_stats)
        result["price_bars_updated"] = sum(
            s.get("imported", 0) for s in price_stats.values()
        )
        
        update_run_metrics(
            session, run_id,
            price_bars_updated=result["price_bars_updated"],
        )
        session.commit()
        
    except Exception as e:
        logger.error(f"[run_id={run_id}] Stage 1d failed: {e}")
        result["price_error"] = str(e)
        session.rollback()
    
    # -------------------------------------------------------------------------
    # Stage 2: Sentiment scoring (V2 - news_processed based)
    # -------------------------------------------------------------------------
    logger.info(f"[run_id={run_id}] Stage 2: Sentiment scoring")
    try:
        from app.ai_engine import score_unscored_processed_articles

        scoring_stats = score_unscored_processed_articles(session)
        session.commit()

        result["articles_scored"] = int(scoring_stats.get("scored_count", 0))
        result["articles_scored_v2"] = int(scoring_stats.get("scored_count", 0))
        result["llm_parse_fail_count"] = int(scoring_stats.get("parse_fail_count", 0))
        result["escalation_count"] = int(scoring_stats.get("escalation_count", 0))
        result["fallback_count"] = int(scoring_stats.get("fallback_count", 0))

        update_run_metrics(
            session,
            run_id,
            articles_scored_v2=result["articles_scored_v2"],
            llm_parse_fail_count=result["llm_parse_fail_count"],
            escalation_count=result["escalation_count"],
            fallback_count=result["fallback_count"],
        )
        session.commit()
        
    except Exception as e:
        logger.error(f"[run_id={run_id}] Stage 2 failed: {e}")
        result["scoring_error"] = str(e)
        session.rollback()
    
    # -------------------------------------------------------------------------
    # Stage 3: Sentiment aggregation (existing)
    # -------------------------------------------------------------------------
    logger.info(f"[run_id={run_id}] Stage 3: Sentiment aggregation")
    try:
        from app.ai_engine import aggregate_daily_sentiment_v2

        days_aggregated_v2 = aggregate_daily_sentiment_v2(session)
        session.commit()
        
        result["days_aggregated_v2"] = days_aggregated_v2
        
    except Exception as e:
        logger.error(f"[run_id={run_id}] Stage 3 failed: {e}")
        result["aggregation_error"] = str(e)
        session.rollback()
    
    # -------------------------------------------------------------------------
    # Stage 3.5: FinBERT embedding extraction (TFT-ASRO)
    # -------------------------------------------------------------------------
    logger.info(f"[run_id={run_id}] Stage 3.5: FinBERT embedding extraction")
    try:
        from deep_learning.data.embeddings import backfill_embeddings

        emb_stats = backfill_embeddings(days=30, pca_dim=32, batch_size=64)
        session.commit()

        result["tft_embeddings_computed"] = emb_stats.get("embedded", 0)
        result["tft_embeddings_skipped"] = emb_stats.get("skipped", 0)
        result["tft_pca_fitted"] = emb_stats.get("pca_fitted", False)

        update_run_metrics(
            session, run_id,
            tft_embeddings_computed=emb_stats.get("embedded", 0),
        )
        session.commit()

        logger.info(
            f"[run_id={run_id}] FinBERT embeddings: "
            f"{emb_stats.get('embedded', 0)} computed, {emb_stats.get('skipped', 0)} skipped"
        )

    except ImportError:
        logger.info(f"[run_id={run_id}] Stage 3.5 skipped: deep_learning module not available")
    except Exception as e:
        logger.warning(f"[run_id={run_id}] Stage 3.5 failed (non-critical): {e}")
        result["tft_embedding_error"] = str(e)
        session.rollback()

    # -------------------------------------------------------------------------
    # Stage 4: Model training (optional)
    # -------------------------------------------------------------------------
    if train_model:
        logger.info(f"[run_id={run_id}] Stage 4: Model training")
        try:
            from app.ai_engine import train_xgboost_model, save_model_metadata_to_db
            
            train_result = train_xgboost_model(session)
            save_model_metadata_to_db(
                session,
                symbol="HG=F",
                importance=train_result.get("importance", []),
                features=train_result.get("features", []),
                metrics=train_result.get("metrics", {}),
            )
            session.commit()
            
            result["model_trained"] = True
            result["model_metrics"] = train_result.get("metrics", {})
            
            update_run_metrics(
                session, run_id,
                train_mae=train_result.get("metrics", {}).get("mae"),
                val_mae=train_result.get("metrics", {}).get("val_mae"),
            )
            session.commit()
            
        except Exception as e:
            logger.error(f"[run_id={run_id}] Stage 4 failed: {e}")
            result["training_error"] = str(e)
            result["model_trained"] = False
            session.rollback()
    else:
        result["model_trained"] = False

    # -------------------------------------------------------------------------
    # Stage 4.5: TFT-ASRO — inference only (training handled by weekly
    # tft-training.yml workflow; daily pipeline never retrains TFT)
    # -------------------------------------------------------------------------
    result["tft_trained"] = False

    # -------------------------------------------------------------------------
    # Stage 5: Generate snapshot
    # -------------------------------------------------------------------------
    logger.info(f"[run_id={run_id}] Stage 5: Generate snapshot")
    snapshot_report = None  # Will be used by Stage 6
    try:
        from app.inference import generate_analysis_report, save_analysis_snapshot
        
        report = generate_analysis_report(session, "HG=F")
        
        if report:
            # Add Faz 2 metadata
            report["quality_state"] = "ok"
            if result.get("news_processed_inserted", 0) == 0:
                report["quality_state"] = "degraded"
                report["message"] = "No fresh news data"
            
            save_analysis_snapshot(session, report, "HG=F")
            session.commit()
            
            result["snapshot_generated"] = True
            snapshot_report = report  # Save for Stage 6
            update_run_metrics(session, run_id, snapshot_generated=True)
            session.commit()
        else:
            result["snapshot_generated"] = False
            
    except Exception as e:
        logger.error(f"[run_id={run_id}] Stage 5 failed: {e}")
        result["snapshot_error"] = str(e)
        result["snapshot_generated"] = False
        session.rollback()
    
    # -------------------------------------------------------------------------
    # Stage 5.5: TFT-ASRO snapshot (parallel to XGBoost snapshot)
    # -------------------------------------------------------------------------
    logger.info(f"[run_id={run_id}] Stage 5.5: TFT-ASRO snapshot")
    try:
        from deep_learning.inference.predictor import generate_tft_analysis
        from deep_learning.config import get_tft_config
        from pathlib import Path

        tft_cfg = get_tft_config()
        ckpt = Path(tft_cfg.training.best_model_path)

        # If checkpoint is not cached locally, try to pull from HF Hub first
        if not ckpt.exists():
            try:
                from deep_learning.models.hub import download_tft_artifacts
                logger.info(f"[run_id={run_id}] TFT checkpoint not found locally – attempting HF Hub download")
                download_tft_artifacts(
                    local_dir=ckpt.parent,
                    repo_id=tft_cfg.training.hf_model_repo,
                )
            except Exception as hub_exc:
                logger.warning(f"[run_id={run_id}] HF Hub download failed: {hub_exc}")

        if ckpt.exists():
            tft_report = generate_tft_analysis(session, "HG=F")

            if "error" not in tft_report:
                result["tft_snapshot_generated"] = True
                update_run_metrics(session, run_id, tft_snapshot_generated=True)
                session.commit()
                logger.info(f"[run_id={run_id}] TFT-ASRO snapshot generated")
                
                # --- NEW PIPELINE BRIDGE ---
                # Fallback to TFT data for commentary if XGBoost failed or wasn't generated
                if not snapshot_report:
                    snapshot_report = tft_report
                # ---------------------------
            else:
                result["tft_snapshot_generated"] = False
                logger.warning(f"[run_id={run_id}] TFT prediction error: {tft_report.get('error')}")
        else:
            result["tft_snapshot_generated"] = False
            logger.info(f"[run_id={run_id}] Stage 5.5 skipped: no TFT checkpoint found (train-tft workflow has not run yet)")

    except ImportError:
        result["tft_snapshot_generated"] = False
    except Exception as e:
        logger.warning(f"[run_id={run_id}] Stage 5.5 failed (non-critical): {e}")
        result["tft_snapshot_generated"] = False
        session.rollback()

    # -------------------------------------------------------------------------
    # Stage 6: Generate commentary (if any snapshot was generated)
    # -------------------------------------------------------------------------
    has_xgb_snapshot = result.get("snapshot_generated") and snapshot_report
    has_tft_snapshot = result.get("tft_snapshot_generated")

    if has_xgb_snapshot or has_tft_snapshot:
        logger.info(f"[run_id={run_id}] Stage 6: Generate commentary")
        try:
            from app.commentary import generate_and_save_commentary

            report = snapshot_report or {}
            
            # Default XGBoost Variable Extraction
            current_price = report.get("current_price", 0.0)
            predicted_price = report.get("predicted_price", 0.0)
            predicted_return = report.get("predicted_return", 0.0)
            sentiment_index = report.get("sentiment_index", 0.0)
            sentiment_label = report.get("sentiment_label", "Neutral")
            top_influencers = report.get("top_influencers", [])
            news_count = report.get("data_quality", {}).get("news_count_7d", 0)

            # --- NEW: TFT Fallback Scheme ---
            is_tft = report.get("model_type") == "TFT-ASRO"
            if is_tft:
                prediction = report.get("prediction", {})
                predicted_price = prediction.get("predicted_price_median", 0.0)
                predicted_return = prediction.get("predicted_return_median", 0.0)
                
                try:
                    from app.inference import get_current_price
                    from app.models import AnalysisSnapshot
                    
                    # Guarantee current price isn't zero
                    fetched_price = get_current_price(session, "HG=F")
                    if fetched_price:
                        current_price = fetched_price
                    
                    # Graft missing context from the most recent successful XGBoost run
                    last_xgb = session.query(AnalysisSnapshot).filter(
                        AnalysisSnapshot.symbol == "HG=F"
                    ).order_by(AnalysisSnapshot.generated_at.desc()).first()
                    
                    if last_xgb:
                        sentiment_index = last_xgb.sentiment_index or 0.0
                        sentiment_label = last_xgb.sentiment_label or "Neutral"
                        top_influencers = last_xgb.top_influencers or []
                        news_count = last_xgb.data_quality.get("news_count_7d", 0) if isinstance(last_xgb.data_quality, dict) else 0
                except Exception as fallback_exc:
                    logger.warning(f"Error mapping TFT values in Stage 6: {fallback_exc}")
            # --------------------------------

            # --- None-safety guard: f-string formatters crash on None ---
            current_price = float(current_price or 0.0)
            predicted_price = float(predicted_price or 0.0)
            predicted_return = float(predicted_return or 0.0)
            sentiment_index = float(sentiment_index or 0.0)
            sentiment_label = sentiment_label or "Neutral"
            top_influencers = top_influencers or []
            news_count = int(news_count or 0)
            # -------------------------------------------------------------

            await generate_and_save_commentary(
                session=session,
                symbol="HG=F",
                current_price=current_price,
                predicted_price=predicted_price,
                predicted_return=predicted_return,
                sentiment_index=sentiment_index,
                sentiment_label=sentiment_label,
                top_influencers=top_influencers,
                news_count=news_count,
            )
            session.commit()

            result["commentary_generated"] = True
            update_run_metrics(session, run_id, commentary_generated=True)
            session.commit()

        except Exception as e:
            logger.warning(f"[run_id={run_id}] Stage 6 failed: {e}")
            result["commentary_generated"] = False
    else:
        logger.warning(f"[run_id={run_id}] Stage 6 skipped: no snapshot generated")
        result["commentary_generated"] = False
    
    return result


# =============================================================================
# arq worker lifecycle
# =============================================================================

async def startup(ctx: dict) -> None:
    """Called when worker starts."""
    logger.info("Worker starting up...")
    init_db()


async def shutdown(ctx: dict) -> None:
    """Called when worker shuts down."""
    logger.info("Worker shutting down...")
