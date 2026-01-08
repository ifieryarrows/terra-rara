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
    """
    logger.info("Starting daily pipeline run...")
    
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
        
        settings = get_settings()
        
        # Step 1: Fetch data
        logger.info("Step 1/3: Fetching data...")
        fetch_results = fetch_all(news=True, prices=True)
        logger.info(f"Data fetch complete: {fetch_results.get('news', {}).get('imported', 0)} news, "
                   f"{sum(s.get('imported', 0) for s in fetch_results.get('prices', {}).values())} price bars")
        
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
                logger.info(f"Snapshot generated: predicted return {report.get('predicted_return', 'N/A')}")
                
                # Step 4: Generate AI Commentary
                logger.info("Step 4/4: Generating AI commentary...")
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
                            logger.warning("AI commentary generation skipped (API key not configured or failed)")
                    finally:
                        loop.close()
                except Exception as ce:
                    logger.error(f"AI commentary generation failed: {ce}")
            else:
                logger.warning("Could not generate analysis snapshot")
        
        logger.info("Daily pipeline complete!")
        
    except Exception as e:
        logger.error(f"Daily pipeline failed: {e}", exc_info=True)
    
    finally:
        lock.release()


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

