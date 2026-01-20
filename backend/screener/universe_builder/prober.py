"""
yfinance prober for validating tickers and fetching metadata.

Validates that tickers are tradeable and have sufficient history.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for yfinance calls.
    
    Ensures we don't exceed API rate limits.
    """
    
    def __init__(self, calls_per_hour: int = 1800):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_hour: Maximum calls allowed per hour
        """
        self.min_interval = 3600 / calls_per_hour
        self.last_call = 0.0
    
    def wait(self):
        """Wait if needed before making next call."""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_call = time.time()


# Global rate limiter instance
_rate_limiter = RateLimiter(calls_per_hour=1800)


class ProbeResult:
    """Result of probing a single ticker."""
    
    def __init__(
        self,
        ticker: str,
        canonical_ticker: str,
        valid: bool = False,
        first_date: Optional[str] = None,
        last_date: Optional[str] = None,
        total_weeks: int = 0,
        coverage_pct: float = 0.0,
        error: Optional[str] = None
    ):
        self.ticker = ticker
        self.canonical_ticker = canonical_ticker
        self.valid = valid
        self.first_date = first_date
        self.last_date = last_date
        self.total_weeks = total_weeks
        self.coverage_pct = coverage_pct
        self.error = error


def probe_ticker(
    ticker: str,
    frequency: str = "W-FRI",
    min_history_days: int = 730,
    max_retries: int = 2,
    retry_delay: int = 30
) -> ProbeResult:
    """
    Probe a single ticker to validate it and get metadata.
    
    Args:
        ticker: Ticker symbol to probe
        frequency: Resample frequency (e.g., "W-FRI")
        min_history_days: Minimum required history
        max_retries: Number of retries on rate limit
        retry_delay: Base delay between retries
        
    Returns:
        ProbeResult with validation status and metadata
    """
    result = ProbeResult(ticker=ticker, canonical_ticker=ticker)
    
    for attempt in range(max_retries + 1):
        _rate_limiter.wait()
        
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # Fetch historical data
            # Use max period to get all available data
            df = yf_ticker.history(period="max", interval="1d")
            
            if df.empty:
                result.error = "no_data_returned"
                return result
            
            # Check for minimum history
            if len(df) < min_history_days:
                result.error = f"insufficient_history_{len(df)}_days"
                return result
            
            # Resample to weekly
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            weekly = df["Close"].resample(frequency).last().dropna()
            
            if len(weekly) < 52:  # Minimum 1 year of weekly data
                result.error = f"insufficient_weekly_data_{len(weekly)}_weeks"
                return result
            
            # Calculate coverage
            first_date = weekly.index.min()
            last_date = weekly.index.max()
            expected_weeks = (last_date - first_date).days // 7 + 1
            actual_weeks = len(weekly)
            coverage_pct = (actual_weeks / expected_weeks * 100) if expected_weeks > 0 else 0
            
            result.valid = True
            result.first_date = first_date.strftime("%Y-%m-%d")
            result.last_date = last_date.strftime("%Y-%m-%d")
            result.total_weeks = actual_weeks
            result.coverage_pct = round(coverage_pct, 2)
            
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "429" in error_str or "rate" in error_str:
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(f"{ticker}: Rate limited, waiting {delay}s (attempt {attempt + 1})")
                    time.sleep(delay)
                    continue
            
            result.error = f"error_{type(e).__name__}"
            logger.debug(f"{ticker}: Probe failed - {e}")
            return result
    
    result.error = "max_retries_exceeded"
    return result


def probe_batch(
    tickers: list[dict],
    frequency: str = "W-FRI",
    min_history_days: int = 730,
    progress_callback=None
) -> list[ProbeResult]:
    """
    Probe multiple tickers.
    
    Args:
        tickers: List of ticker dicts with 'canonical_ticker' key
        frequency: Resample frequency
        min_history_days: Minimum required history
        progress_callback: Optional callback(current, total, ticker)
        
    Returns:
        List of ProbeResult objects
    """
    results = []
    total = len(tickers)
    
    for i, ticker_info in enumerate(tickers):
        ticker = ticker_info.get("canonical_ticker", ticker_info.get("ticker", ""))
        
        if progress_callback:
            progress_callback(i + 1, total, ticker)
        
        result = probe_ticker(
            ticker=ticker,
            frequency=frequency,
            min_history_days=min_history_days
        )
        
        results.append(result)
        
        if (i + 1) % 50 == 0:
            logger.info(f"Probed {i + 1}/{total} tickers...")
    
    valid_count = sum(1 for r in results if r.valid)
    logger.info(f"Probe complete: {valid_count}/{total} valid tickers")
    
    return results
