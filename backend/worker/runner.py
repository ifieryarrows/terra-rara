"""
arq Worker Runner.

This is the entrypoint for the worker process.
Run with: python -m worker.runner

The worker:
- Consumes jobs from Redis queue
- Executes pipeline tasks
- Has NO scheduler - scheduling is external (GitHub Actions, cron, etc.)
"""

import logging
import os
import sys

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arq import run_worker

from adapters.queue.redis import get_redis_settings
from worker.tasks import run_pipeline, startup, shutdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - [worker] - %(message)s"
)
logger = logging.getLogger(__name__)


class WorkerSettings:
    """
    arq worker settings.
    
    This class is discovered by arq when running:
        arq worker.runner.WorkerSettings
    """
    
    # Redis connection
    redis_settings = get_redis_settings()
    
    # Task functions
    functions = [run_pipeline]
    
    # Lifecycle hooks
    on_startup = startup
    on_shutdown = shutdown
    
    # Job settings
    max_jobs = 1  # Only one pipeline at a time per worker
    job_timeout = 3600  # 1 hour max
    max_tries = 1  # No automatic retries - cron will retry next cycle
    
    # Health check
    health_check_interval = 30


def main():
    """Run the worker."""
    logger.info("Starting Terra Rara worker...")
    logger.info(f"Redis: {WorkerSettings.redis_settings.host}:{WorkerSettings.redis_settings.port}")
    
    # Run worker (blocking)
    run_worker(WorkerSettings)


if __name__ == "__main__":
    main()
