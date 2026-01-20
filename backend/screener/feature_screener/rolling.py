"""
Rolling correlation analysis with sign flip detection.

Computes rolling correlation windows and detects instability.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RollingResult:
    """Result of rolling correlation analysis."""
    
    def __init__(
        self,
        ticker: str,
        rolling_corr_mean: Optional[float] = None,
        rolling_corr_std: Optional[float] = None,
        rolling_corr_min: Optional[float] = None,
        rolling_corr_max: Optional[float] = None,
        sign_flip_count: int = 0,
        n_windows: int = 0,
        error: Optional[str] = None
    ):
        self.ticker = ticker
        self.rolling_corr_mean = rolling_corr_mean
        self.rolling_corr_std = rolling_corr_std
        self.rolling_corr_min = rolling_corr_min
        self.rolling_corr_max = rolling_corr_max
        self.sign_flip_count = sign_flip_count
        self.n_windows = n_windows
        self.error = error
    
    @property
    def valid(self) -> bool:
        return self.error is None and self.rolling_corr_mean is not None
    
    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "rolling_corr_mean": self.rolling_corr_mean,
            "rolling_corr_std": self.rolling_corr_std,
            "rolling_corr_min": self.rolling_corr_min,
            "rolling_corr_max": self.rolling_corr_max,
            "sign_flip_count": self.sign_flip_count,
            "n_windows": self.n_windows,
            "error": self.error
        }


def compute_rolling_correlation(
    target: pd.Series,
    candidate: pd.Series,
    ticker: str,
    window: int = 26,
    epsilon: float = 0.05,
    min_periods: Optional[int] = None
) -> RollingResult:
    """
    Compute rolling correlation and sign flip count.
    
    Args:
        target: Target series (returns)
        candidate: Candidate series (returns)
        ticker: Candidate ticker name
        window: Rolling window size (weeks)
        epsilon: Threshold for near-zero (ignored in sign flip)
        min_periods: Minimum periods for rolling window
        
    Returns:
        RollingResult with rolling statistics
    """
    result = RollingResult(ticker=ticker)
    
    if min_periods is None:
        min_periods = window // 2
    
    try:
        # Align series
        aligned = pd.concat(
            [target, candidate],
            axis=1,
            keys=["target", "candidate"]
        ).dropna()
        
        if len(aligned) < window:
            result.error = f"insufficient_data_{len(aligned)}_need_{window}"
            return result
        
        # Compute rolling correlation
        rolling_corr = aligned["target"].rolling(
            window=window,
            min_periods=min_periods
        ).corr(aligned["candidate"])
        
        # Drop NaN values
        rolling_corr = rolling_corr.dropna()
        
        if len(rolling_corr) < 10:
            result.error = f"insufficient_windows_{len(rolling_corr)}"
            return result
        
        result.n_windows = len(rolling_corr)
        result.rolling_corr_mean = float(rolling_corr.mean())
        result.rolling_corr_std = float(rolling_corr.std())
        result.rolling_corr_min = float(rolling_corr.min())
        result.rolling_corr_max = float(rolling_corr.max())
        
        # Count sign flips (ignoring epsilon band)
        result.sign_flip_count = _count_sign_flips(rolling_corr, epsilon)
        
        return result
        
    except Exception as e:
        result.error = f"error_{type(e).__name__}"
        logger.debug(f"{ticker}: Rolling correlation failed - {e}")
        return result


def _count_sign_flips(series: pd.Series, epsilon: float = 0.05) -> int:
    """
    Count sign flips in a series, ignoring values in [-epsilon, +epsilon].
    
    Args:
        series: Series of correlation values
        epsilon: Threshold for near-zero zone
        
    Returns:
        Number of sign flips
    """
    # Extract values outside epsilon band
    significant_values = []
    
    for val in series:
        if val > epsilon:
            significant_values.append(1)   # Positive
        elif val < -epsilon:
            significant_values.append(-1)  # Negative
        # Values in [-epsilon, +epsilon] are ignored
    
    if len(significant_values) < 2:
        return 0
    
    # Count transitions
    flips = 0
    for i in range(1, len(significant_values)):
        if significant_values[i] != significant_values[i-1]:
            flips += 1
    
    return flips


def compute_rolling_batch(
    target: pd.Series,
    candidates: dict[str, pd.Series],
    window: int = 26,
    epsilon: float = 0.05
) -> dict[str, RollingResult]:
    """
    Compute rolling correlations for multiple candidates.
    
    Args:
        target: Target series
        candidates: Dict mapping ticker to series
        window: Rolling window size
        epsilon: Sign flip epsilon
        
    Returns:
        Dict mapping ticker to RollingResult
    """
    results = {}
    
    for ticker, candidate_series in candidates.items():
        result = compute_rolling_correlation(
            target=target,
            candidate=candidate_series,
            ticker=ticker,
            window=window,
            epsilon=epsilon
        )
        results[ticker] = result
    
    valid_count = sum(1 for r in results.values() if r.valid)
    logger.info(f"Computed rolling stats: {valid_count}/{len(results)} valid")
    
    return results
