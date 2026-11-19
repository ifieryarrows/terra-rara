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


def _evaluate_history_df(
    ticker: str,
    canonical_ticker: str,
    df: Optional[pd.DataFrame],
    *,
    frequency: str = "W-FRI",
    min_history_days: int = 730,
) -> ProbeResult:
    """Validate an OHLCV history DataFrame and populate ProbeResult metadata."""
    result = ProbeResult(ticker=ticker, canonical_ticker=canonical_ticker)
    if df is None or df.empty or "Close" not in df.columns:
        result.error = "no_data_returned"
        return result

    try:
        clean_df = df.dropna(subset=["Close"]).copy()
        if clean_df.empty:
            result.error = "no_data_returned"
            return result

        if len(clean_df) < min_history_days:
            result.error = f"insufficient_history_{len(clean_df)}_days"
            return result

        clean_df.index = pd.to_datetime(clean_df.index)
        if clean_df.index.tz is not None:
            clean_df.index = clean_df.index.tz_localize(None)

        weekly = clean_df["Close"].resample(frequency).last().dropna()
        if len(weekly) < 52:
            result.error = f"insufficient_weekly_data_{len(weekly)}_weeks"
            return result

        first_date = weekly.index.min()
        last_date = weekly.index.max()
        expected_weeks = (last_date - first_date).days // 7 + 1
        actual_weeks = len(weekly)
        coverage_pct = (actual_weeks / expected_weeks * 100) if expected_weeks > 0 else 0.0

        result.valid = True
        result.first_date = first_date.strftime("%Y-%m-%d")
        result.last_date = last_date.strftime("%Y-%m-%d")
        result.total_weeks = actual_weeks
        result.coverage_pct = round(coverage_pct, 2)
        return result
    except Exception as exc:
        result.error = f"error_{type(exc).__name__}"
        logger.debug("%s: History evaluation failed - %s", ticker, exc)
        return result


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
            return _evaluate_history_df(
                ticker,
                ticker,
                df,
                frequency=frequency,
                min_history_days=min_history_days,
            )
            
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


_DEFAULT_PROBE_TICKER = probe_ticker


def probe_batch(
    tickers: list[dict],
    frequency: str = "W-FRI",
    min_history_days: int = 730,
    progress_callback=None,
    chunk_size: int = 25,
) -> list[ProbeResult]:
    """
    Probe multiple tickers using chunked yfinance batch prefetch with per-ticker fallback.
    
    Args:
        tickers: List of ticker dicts with 'canonical_ticker' key
        frequency: Resample frequency
        min_history_days: Minimum required history
        progress_callback: Optional callback(current, total, ticker)
        chunk_size: Number of tickers per batch yfinance download call
        
    Returns:
        List of ProbeResult objects in input order
    """
    total = len(tickers)
    items: list[tuple[int, str, str]] = [
        (
            idx,
            str(ticker_info.get("ticker", ticker_info.get("canonical_ticker", ""))),
            str(ticker_info.get("canonical_ticker", ticker_info.get("ticker", ""))),
        )
        for idx, ticker_info in enumerate(tickers)
    ]
    results_by_idx: dict[int, ProbeResult] = {}

    use_batch = (probe_ticker is _DEFAULT_PROBE_TICKER) and hasattr(yf, "download")
    if use_batch and total > 1:
        for start in range(0, total, max(1, int(chunk_size))):
            chunk_items = items[start : start + max(1, int(chunk_size))]
            if len(chunk_items) <= 1:
                continue
            chunk_symbols = list(dict.fromkeys(canon for _i, _raw, canon in chunk_items if canon))
            if not chunk_symbols:
                continue
            _rate_limiter.wait()
            try:
                batch_df = yf.download(
                    chunk_symbols,
                    period="max",
                    interval="1d",
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
            except Exception as exc:
                logger.debug("Batch ticker probe prefetch failed, using per-ticker fallback: %s", exc)
                continue

            if batch_df is None or batch_df.empty:
                continue

            has_multi = isinstance(batch_df.columns, pd.MultiIndex)
            level0 = set(batch_df.columns.get_level_values(0)) if has_multi else set()
            for idx, raw_ticker, canon in chunk_items:
                sub = None
                if has_multi and canon in level0:
                    sub = batch_df[canon]
                elif not has_multi and len(chunk_symbols) == 1 and "Close" in batch_df.columns:
                    sub = batch_df
                if sub is None or sub.dropna(how="all").empty:
                    continue
                evaluated = _evaluate_history_df(
                    canon,
                    canon,
                    sub,
                    frequency=frequency,
                    min_history_days=min_history_days,
                )
                evaluated.ticker = canon
                results_by_idx[idx] = evaluated

    results: list[ProbeResult] = []
    for idx, _raw_ticker, canon in items:
        if progress_callback:
            progress_callback(idx + 1, total, canon)

        result = results_by_idx.get(idx)
        if result is None:
            result = probe_ticker(
                ticker=canon,
                frequency=frequency,
                min_history_days=min_history_days,
            )

        results.append(result)

        if (idx + 1) % 50 == 0:
            logger.info(f"Probed {idx + 1}/{total} tickers...")

    valid_count = sum(1 for r in results if r.valid)
    logger.info(f"Probe complete: {valid_count}/{total} valid tickers")

    return results
