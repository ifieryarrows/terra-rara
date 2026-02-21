"""
APScheduler-based daily automation.

Runs the data fetch + AI pipeline at a configured time each day.
"""

import logging
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from app.settings import get_settings
from app.lock import PipelineLock
from app.db import init_db

logger = logging.getLogger(__name__)

# Global scheduler instance
_scheduler: Optional[BackgroundScheduler] = None


def run_daily_pipeline():
    """
    Execute the daily pipeline:
    1. Fetch news and prices
    2. Score sentiment
    3. Aggregate daily sentiment
    4. Generate fresh analysis snapshot
    
    Uses pipeline lock to prevent concurrent runs.
    Records metrics to PipelineRunMetrics table for monitoring.
    """
    import json
    import uuid
    from datetime import timezone as tz
    
    logger.info("Starting daily pipeline run...")
    
    # Generate run ID and start time
    run_id = f"run-{datetime.now(tz.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_started_at = datetime.now(tz.utc)
    
    # Initialize metrics
    metrics = {
        "run_id": run_id,
        "run_started_at": run_started_at,
        "status": "running",
        "symbols_requested": 0,
        "symbols_fetched_ok": 0,
        "symbols_failed": 0,
        "failed_symbols_list": [],
        "news_imported": 0,
        "news_duplicates": 0,
        "price_bars_updated": 0,
        "snapshot_generated": False,
        "commentary_generated": False,
    }
    
    lock = PipelineLock(timeout=0)
    
    if not lock.acquire():
        logger.warning("Could not acquire lock - another pipeline may be running")
        return
    
    try:
        # Import here to avoid circular imports
        from app.data_manager import fetch_all
        from app.ai_engine import run_full_pipeline
        from app.inference import generate_analysis_report, save_analysis_snapshot
        from app.db import SessionLocal
        from app.models import PipelineRunMetrics
        
        settings = get_settings()
        
        # Record symbol set info
        metrics["symbol_set_name"] = settings.symbol_set
        metrics["symbols_requested"] = len(settings.training_symbols)
        
        # Step 1: Fetch data
        logger.info("Step 1/3: Fetching data...")
        fetch_results = fetch_all(news=True, prices=True)
        
        # Track news stats
        news_stats = fetch_results.get('news', {})
        metrics["news_imported"] = news_stats.get('imported', 0)
        metrics["news_duplicates"] = news_stats.get('duplicates', 0)
        
        # Track price stats
        price_stats = fetch_results.get('prices', {})
        total_bars = 0
        failed_symbols = []
        for symbol, stats in price_stats.items():
            if stats.get('error'):
                failed_symbols.append(symbol)
            else:
                total_bars += stats.get('imported', 0)
        
        metrics["price_bars_updated"] = total_bars
        metrics["symbols_fetched_ok"] = len(price_stats) - len(failed_symbols)
        metrics["symbols_failed"] = len(failed_symbols)
        metrics["failed_symbols_list"] = failed_symbols
        
        logger.info(f"Data fetch complete: {metrics['news_imported']} news, {total_bars} price bars")
        if failed_symbols:
            logger.warning(f"Failed symbols: {failed_symbols}")
        
        # Step 2: Run AI pipeline (score + aggregate only, no retraining by default)
        logger.info("Step 2/3: Running AI pipeline (sentiment scoring)...")
        ai_results = run_full_pipeline(
            target_symbol=settings.target_symbol,
            score_sentiment=True,
            aggregate_sentiment=True,
            train_model=False  # Don't retrain daily, just refresh sentiment
        )
        logger.info(f"AI pipeline complete: {ai_results.get('scored_articles', 0)} articles scored, "
                   f"{ai_results.get('aggregated_days', 0)} days aggregated")
        
        # Step 3: Generate fresh snapshot
        logger.info("Step 3/4: Generating analysis snapshot...")
        with SessionLocal() as session:
            report = generate_analysis_report(session, settings.target_symbol)
            if report:
                save_analysis_snapshot(session, report, settings.target_symbol)
                metrics["snapshot_generated"] = True
                logger.info(f"Snapshot generated: predicted return {report.get('predicted_return', 'N/A')}")
                
                # Step 4: Generate AI Commentary
                logger.info("Step 4/4: Generating AI commentary...")
                try:
                    from app.commentary import generate_and_save_commentary
                    from app.async_bridge import run_async_from_sync
                    from sqlalchemy import func
                    from app.models import NewsArticle
                    from datetime import timedelta
                    
                    # Get news count for last 7 days
                    week_ago = datetime.now() - timedelta(days=7)
                    news_count = session.query(func.count(NewsArticle.id)).filter(
                        NewsArticle.published_at >= week_ago
                    ).scalar() or 0
                    
                    commentary = run_async_from_sync(
                        generate_and_save_commentary,
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
                    if commentary:
                        metrics["commentary_generated"] = True
                        logger.info("AI commentary generated and saved")
                    else:
                        logger.warning("AI commentary generation skipped (API key not configured or failed)")
                except Exception as ce:
                    logger.error(f"AI commentary generation failed: {ce}")
            else:
                logger.warning("Could not generate analysis snapshot")
        
        metrics["status"] = "success"
        logger.info("Daily pipeline complete!")
        
    except Exception as e:
        metrics["status"] = "failed"
        metrics["error_message"] = str(e)[:500]
        logger.error(f"Daily pipeline failed: {e}", exc_info=True)
    
    finally:
        lock.release()
        
        # Save metrics to database
        try:
            from app.db import SessionLocal
            from app.models import PipelineRunMetrics
            
            run_completed_at = datetime.now(tz.utc)
            duration = (run_completed_at - run_started_at).total_seconds()
            
            with SessionLocal() as session:
                metrics_record = PipelineRunMetrics(
                    run_id=metrics["run_id"],
                    run_started_at=metrics["run_started_at"],
                    run_completed_at=run_completed_at,
                    duration_seconds=duration,
                    symbol_set_name=metrics.get("symbol_set_name"),
                    symbols_requested=metrics.get("symbols_requested"),
                    symbols_fetched_ok=metrics.get("symbols_fetched_ok"),
                    symbols_failed=metrics.get("symbols_failed"),
                    failed_symbols_list=json.dumps(metrics.get("failed_symbols_list", [])),
                    news_imported=metrics.get("news_imported"),
                    news_duplicates=metrics.get("news_duplicates"),
                    price_bars_updated=metrics.get("price_bars_updated"),
                    snapshot_generated=metrics.get("snapshot_generated", False),
                    commentary_generated=metrics.get("commentary_generated", False),
                    status=metrics.get("status", "unknown"),
                    error_message=metrics.get("error_message"),
                )
                session.add(metrics_record)
                session.commit()
                logger.info(f"Pipeline metrics saved: {run_id} ({metrics['status']}, {duration:.1f}s)")
        except Exception as me:
            logger.warning(f"Could not save pipeline metrics: {me}")


def parse_schedule_time(time_str: str) -> tuple[int, int]:
    """Parse HH:MM time string to (hour, minute) tuple."""
    try:
        parts = time_str.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        return hour, minute
    except (ValueError, IndexError):
        logger.warning(f"Invalid schedule time '{time_str}', defaulting to 09:00")
        return 9, 0


def start_scheduler():
    """Start the background scheduler."""
    global _scheduler
    
    if _scheduler is not None:
        logger.warning("Scheduler already running")
        return
    
    settings = get_settings()
    
    if not settings.scheduler_enabled:
        logger.info("Scheduler is disabled by configuration")
        return
    
    # Parse schedule time
    hour, minute = parse_schedule_time(settings.schedule_time)
    
    # Get timezone
    try:
        tz = pytz.timezone(settings.tz)
    except Exception:
        logger.warning(f"Invalid timezone '{settings.tz}', using UTC")
        tz = pytz.UTC
    
    # Create scheduler
    _scheduler = BackgroundScheduler(
        timezone=tz,
        job_defaults={
            "coalesce": True,  # Combine missed runs
            "max_instances": 1,  # Only one instance at a time
            "misfire_grace_time": 3600,  # 1 hour grace for misfires
        }
    )
    
    # Add daily job
    trigger = CronTrigger(
        hour=hour,
        minute=minute,
        timezone=tz
    )
    
    _scheduler.add_job(
        run_daily_pipeline,
        trigger=trigger,
        id="daily_pipeline",
        name="Daily Data + AI Pipeline",
        replace_existing=True
    )
    
    # Start
    _scheduler.start()
    
    logger.info(f"Scheduler started - daily pipeline at {hour:02d}:{minute:02d} {settings.tz}")
    
    # Log next run time
    job = _scheduler.get_job("daily_pipeline")
    if job and job.next_run_time:
        logger.info(f"Next scheduled run: {job.next_run_time}")


def stop_scheduler():
    """Stop the background scheduler gracefully."""
    global _scheduler
    
    if _scheduler is None:
        return
    
    try:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")
    finally:
        _scheduler = None


def get_scheduler_status() -> dict:
    """Get current scheduler status."""
    global _scheduler
    
    if _scheduler is None:
        return {
            "running": False,
            "next_run": None,
            "job_count": 0
        }
    
    jobs = _scheduler.get_jobs()
    next_run = None
    
    for job in jobs:
        if job.next_run_time:
            if next_run is None or job.next_run_time < next_run:
                next_run = job.next_run_time
    
    return {
        "running": _scheduler.running,
        "next_run": next_run.isoformat() if next_run else None,
        "job_count": len(jobs)
    }


def trigger_pipeline_now():
    """
    Manually trigger the pipeline immediately.
    For CLI or administrative use.
    """
    logger.info("Manual pipeline trigger requested")
    run_daily_pipeline()

