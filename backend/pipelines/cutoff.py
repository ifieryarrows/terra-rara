"""
Market cut-off calculation for news aggregation.

Faz 2: Defines which news articles belong to "today's" sentiment.
Uses market close time with buffer to determine cut-off.
"""

import logging
from datetime import datetime, time, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python < 3.9

from app.settings import get_settings

logger = logging.getLogger(__name__)


def compute_news_cutoff(
    run_datetime: Optional[datetime] = None,
    market_tz: Optional[str] = None,
    market_close: Optional[str] = None,
    buffer_minutes: Optional[int] = None,
) -> datetime:
    """
    Compute news cut-off datetime for a pipeline run.
    
    Logic:
        1. Convert run_datetime to market timezone
        2. Calculate today's close + buffer
        3. If run is before today's close+buffer, use yesterday's close+buffer
        4. If run is on weekend, roll back to Friday
    
    Args:
        run_datetime: When pipeline started (UTC). Defaults to now.
        market_tz: Market timezone (e.g., "America/New_York"). Defaults to settings.
        market_close: Market close time "HH:MM". Defaults to settings.
        buffer_minutes: Minutes after close to allow. Defaults to settings.
        
    Returns:
        Cut-off datetime in UTC
        
    Example:
        Pipeline runs at 2026-01-28 10:00 UTC (05:00 ET)
        → Before 16:30 ET → use 2026-01-27 16:30 ET → 2026-01-27 21:30 UTC
        
        Pipeline runs at 2026-01-28 22:00 UTC (17:00 ET)
        → After 16:30 ET → use 2026-01-28 16:30 ET → 2026-01-28 21:30 UTC
    """
    settings = get_settings()
    
    # Defaults from settings
    if run_datetime is None:
        run_datetime = datetime.now(timezone.utc)
    if market_tz is None:
        market_tz = settings.market_timezone
    if market_close is None:
        market_close = settings.market_close_time
    if buffer_minutes is None:
        buffer_minutes = settings.cutoff_buffer_minutes
    
    # Parse market close time
    close_hour, close_minute = map(int, market_close.split(":"))
    buffer = timedelta(minutes=buffer_minutes)
    
    # Get timezone
    tz = ZoneInfo(market_tz)
    
    # Convert run_datetime to market timezone
    if run_datetime.tzinfo is None:
        run_datetime = run_datetime.replace(tzinfo=timezone.utc)
    run_local = run_datetime.astimezone(tz)
    
    # Today's close + buffer in market timezone
    today_close = run_local.replace(
        hour=close_hour,
        minute=close_minute,
        second=0,
        microsecond=0,
    ) + buffer
    
    # Determine which day's close to use
    if run_local >= today_close:
        # After today's close+buffer → use today's close
        cutoff_local = today_close
    else:
        # Before today's close+buffer → use yesterday's close
        yesterday_close = today_close - timedelta(days=1)
        cutoff_local = yesterday_close
    
    # Weekend guard: roll back to Friday if cutoff falls on weekend
    cutoff_local = _adjust_for_weekend(cutoff_local)
    
    # Convert back to UTC
    cutoff_utc = cutoff_local.astimezone(timezone.utc)
    
    logger.debug(
        f"Cut-off computed: run={run_datetime.isoformat()}, "
        f"cutoff={cutoff_utc.isoformat()} (local: {cutoff_local.isoformat()})"
    )
    
    return cutoff_utc


def _adjust_for_weekend(dt: datetime) -> datetime:
    """
    Adjust datetime to Friday if it falls on weekend.
    
    Args:
        dt: Datetime to adjust
        
    Returns:
        Adjusted datetime (Friday if input was Sat/Sun)
    """
    weekday = dt.weekday()  # 0=Mon, 5=Sat, 6=Sun
    
    if weekday == 5:  # Saturday
        return dt - timedelta(days=1)  # Roll back to Friday
    elif weekday == 6:  # Sunday
        return dt - timedelta(days=2)  # Roll back to Friday
    
    return dt


def get_news_window(
    cutoff_dt: datetime,
    lookback_days: int = 7,
) -> tuple[datetime, datetime]:
    """
    Get the time window for news aggregation.
    
    Args:
        cutoff_dt: Cut-off datetime (latest news to include)
        lookback_days: How many days back to look
        
    Returns:
        Tuple of (start_dt, end_dt) for news query
    """
    end_dt = cutoff_dt
    start_dt = cutoff_dt - timedelta(days=lookback_days)
    
    return (start_dt, end_dt)


def is_market_open(
    dt: Optional[datetime] = None,
    market_tz: Optional[str] = None,
) -> bool:
    """
    Check if market is currently open (approximate).
    
    Note: Does not account for holidays, just weekdays 9:30-16:00 ET.
    
    Args:
        dt: Datetime to check. Defaults to now.
        market_tz: Market timezone. Defaults to settings.
        
    Returns:
        True if market is likely open
    """
    settings = get_settings()
    
    if dt is None:
        dt = datetime.now(timezone.utc)
    if market_tz is None:
        market_tz = settings.market_timezone
    
    tz = ZoneInfo(market_tz)
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(tz)
    
    # Weekend
    if local.weekday() >= 5:
        return False
    
    # Market hours (approximate: 9:30 - 16:00)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    current_time = local.time()
    return market_open <= current_time <= market_close
