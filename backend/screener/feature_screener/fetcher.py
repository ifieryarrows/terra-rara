"""
Price fetcher with rate limiting, retry logic, and caching.

Fetches weekly price data from yfinance with robust error handling.
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from screener.core.cache import PriceCache
from screener.core.config import FetcherConfig

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, calls_per_hour: int = 1800):
        self.min_interval = 3600 / calls_per_hour
        self.last_call = 0.0
    
    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class PriceFetcher:
    """
    Fetches and caches weekly price data from yfinance.
    
    Uses collision-proof cache keys via FetchParams fingerprinting.
    """
    
    def __init__(
        self,
        config: FetcherConfig,
        cache_dir: Optional[str | Path] = None,
        cache_namespace: Optional[str] = None
    ):
        """
        Initialize fetcher.
        
        Args:
            config: Fetcher configuration
            cache_dir: Directory for price cache (None = no caching)
            cache_namespace: Optional namespace for cache isolation (e.g., config_hash[:8])
        """
        self.config = config
        self.rate_limiter = RateLimiter(config.calls_per_hour)
        
        self.cache: Optional[PriceCache] = None
        if config.cache_enabled and cache_dir:
            self.cache = PriceCache(cache_dir, namespace=cache_namespace)
        
        self.fetch_stats = {
            "fetched": 0,
            "cached": 0,
            "failed": 0,
            "rate_limited": 0
        }
    
    def fetch_weekly_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        frequency: str = "W-FRI"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch weekly prices for a symbol.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), default = today
            frequency: Resample frequency
            
        Returns:
            DataFrame with columns: date (index), close, returns
            or None if fetch failed
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Build collision-proof fetch params
        from screener.core.cache import FetchParams
        fetch_params = FetchParams(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            resample_rule=frequency,
            price_field=self.config.price_field,
            auto_adjust=True,
            actions=True
        )
        
        # Check cache first
        if self.cache:
            cached = self.cache.get(fetch_params)
            if cached is not None:
                self.fetch_stats["cached"] += 1
                logger.debug(f"{symbol}: Cache hit")
                return cached
        
        # Fetch from yfinance with retry
        df = self._fetch_with_retry(symbol, start_date, end_date)
        
        if df is None or df.empty:
            self.fetch_stats["failed"] += 1
            return None
        
        # Process: resample to weekly
        df = self._process_to_weekly(df, frequency)
        
        if df is None or df.empty:
            self.fetch_stats["failed"] += 1
            return None
        
        # Cache result with full params fingerprint
        if self.cache:
            self.cache.put(fetch_params, df)
        
        self.fetch_stats["fetched"] += 1
        return df
    
    def _fetch_with_retry(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch raw daily data with retry on rate limit."""
        
        for attempt in range(self.config.max_retries + 1):
            self.rate_limiter.wait()
            
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval="1d")
                
                if df.empty:
                    logger.warning(f"{symbol}: No data returned")
                    return None
                
                return df
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "429" in error_str or "rate" in error_str:
                    self.fetch_stats["rate_limited"] += 1
                    
                    if attempt < self.config.max_retries:
                        delay = self.config.base_retry_delay_seconds * (2 ** attempt)
                        logger.warning(f"{symbol}: Rate limited, waiting {delay}s")
                        time.sleep(delay)
                        continue
                
                logger.error(f"{symbol}: Fetch failed - {e}")
                return None
        
        return None
    
    def _process_to_weekly(
        self,
        df: pd.DataFrame,
        frequency: str
    ) -> Optional[pd.DataFrame]:
        """Process daily data to weekly frequency."""
        
        try:
            # Ensure datetime index without timezone
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Get price field
            price_col = self.config.price_field
            if price_col not in df.columns:
                # Fallback to Close
                price_col = "Close"
                if price_col not in df.columns:
                    logger.warning(f"No price column found")
                    return None
            
            # Resample to weekly (last observation)
            weekly_close = df[price_col].resample(frequency).last()
            weekly_close = weekly_close.dropna()
            
            if len(weekly_close) < 10:
                return None
            
            # Build result DataFrame
            result = pd.DataFrame({
                "close": weekly_close
            })
            
            # Calculate returns
            result["returns"] = result["close"].pct_change()
            
            # Convert index to date only
            result.index = result.index.date
            result.index.name = "date"
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return None
    
    def fetch_batch(
        self,
        symbols: list[str],
        start_date: str,
        end_date: Optional[str] = None,
        frequency: str = "W-FRI",
        progress_callback=None
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch prices for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            frequency: Resample frequency
            progress_callback: Optional callback(current, total, symbol)
            
        Returns:
            Dict mapping symbol to DataFrame
        """
        results = {}
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(i + 1, total, symbol)
            
            df = self.fetch_weekly_prices(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency
            )
            
            if df is not None:
                results[symbol] = df
            
            if (i + 1) % 50 == 0:
                logger.info(f"Fetched {i + 1}/{total} symbols...")
        
        logger.info(
            f"Fetch complete: {len(results)}/{total} successful, "
            f"{self.fetch_stats['cached']} cached, "
            f"{self.fetch_stats['failed']} failed"
        )
        
        return results
    
    def get_stats(self) -> dict:
        """Get fetch statistics."""
        return self.fetch_stats.copy()
