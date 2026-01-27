"""
Redis connection and settings for arq queue.
"""

import logging
from typing import Optional

from arq.connections import RedisSettings as ArqRedisSettings
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# Module-level pool cache
_redis_pool: Optional[Redis] = None


def get_redis_settings() -> ArqRedisSettings:
    """
    Get Redis settings for arq worker.
    
    Reads from environment:
        REDIS_URL: Full Redis URL (redis://host:port/db)
        
    Falls back to localhost:6379 for development.
    """
    import os
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Parse URL for arq settings
    # Format: redis://[user:password@]host:port/db
    from urllib.parse import urlparse
    parsed = urlparse(redis_url)
    
    return ArqRedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or 0),
        password=parsed.password,
    )


async def get_redis_pool(max_retries: int = 5, retry_delay: float = 1.0) -> Redis:
    """
    Get async Redis connection pool.
    Lazy initialization, cached at module level.
    
    Includes retry logic for HF Spaces where Redis might start
    slightly after API/Worker due to supervisord startup order.
    """
    global _redis_pool
    
    if _redis_pool is None:
        import os
        import asyncio
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        for attempt in range(max_retries):
            try:
                pool = Redis.from_url(
                    redis_url, 
                    decode_responses=True,
                    socket_connect_timeout=5.0,
                    socket_timeout=5.0,
                )
                # Test connection
                await pool.ping()
                _redis_pool = pool
                logger.info(f"Redis pool created: {redis_url.split('@')[-1]}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Redis connection attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Redis connection failed after {max_retries} attempts: {e}")
                    raise
    
    return _redis_pool


async def close_redis_pool():
    """Close Redis pool on shutdown."""
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.close()
        _redis_pool = None
        logger.info("Redis pool closed")


async def redis_healthcheck() -> dict:
    """
    Check Redis connectivity.
    
    Returns:
        dict with 'ok' bool and 'latency_ms' float
    """
    import time
    
    try:
        pool = await get_redis_pool()
        start = time.monotonic()
        await pool.ping()
        latency = (time.monotonic() - start) * 1000
        
        return {"ok": True, "latency_ms": round(latency, 2)}
    except Exception as e:
        logger.warning(f"Redis healthcheck failed: {e}")
        return {"ok": False, "error": str(e)}


# Re-export for convenience
RedisSettings = ArqRedisSettings
