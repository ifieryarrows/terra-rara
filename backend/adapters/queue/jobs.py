"""
Job enqueue/dequeue functions for pipeline tasks.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from arq import create_pool

from adapters.queue.redis import get_redis_settings

logger = logging.getLogger(__name__)


async def enqueue_pipeline_job(
    train_model: bool = False,
    trigger_source: str = "manual",
    run_id: Optional[str] = None,
) -> dict:
    """
    Enqueue a pipeline job to Redis.
    
    Args:
        train_model: Whether to train/retrain the XGBoost model
        trigger_source: Source of trigger (manual, cron, api)
        run_id: Optional run ID, generated if not provided
        
    Returns:
        dict with run_id and job_id
    """
    if run_id is None:
        run_id = str(uuid4())
    
    try:
        redis = await create_pool(get_redis_settings())
        
        job = await redis.enqueue_job(
            "run_pipeline",
            run_id=run_id,
            train_model=train_model,
            trigger_source=trigger_source,
            enqueued_at=datetime.now(timezone.utc).isoformat(),
        )
        
        await redis.close()
        
        logger.info(f"Pipeline job enqueued: run_id={run_id}, job_id={job.job_id}")
        
        return {
            "run_id": run_id,
            "job_id": job.job_id,
            "enqueued": True,
            "trigger_source": trigger_source,
            "train_model": train_model,
        }
        
    except Exception as e:
        logger.error(f"Failed to enqueue pipeline job: {e}")
        raise


async def get_job_status(job_id: str) -> Optional[dict]:
    """
    Get status of a queued job.
    
    Returns:
        dict with job status or None if not found
    """
    try:
        redis = await create_pool(get_redis_settings())
        job = await redis.job(job_id)
        await redis.close()
        
        if job is None:
            return None
            
        return {
            "job_id": job_id,
            "status": job.status,
        }
        
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return None
