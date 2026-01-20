"""
Lead-lag discovery for correlation analysis.

Discovers optimal lead-lag relationship in IS period,
then freezes lag for OOS evaluation.

LEAD-LAG CONVENTION (CRITICAL):
    lag = +k  =>  candidate LEADS target by k periods
                  candidate[t] is correlated with target[t+k]
                  Use: candidate.shift(k) aligns with target
    
    lag = 0   =>  contemporaneous relationship
    
    lag = -k  =>  candidate LAGS target by k periods
                  target[t] is correlated with candidate[t+abs(k)]

    Example: If FCX leads HG=F by 2 weeks (FCX moves, copper follows),
    best_lag = +2 means we should use FCX from 2 weeks ago to predict
    today's copper movement.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LeadLagResult:
    """Result of lead-lag analysis."""
    
    def __init__(
        self,
        ticker: str,
        best_lag: int = 0,
        best_corr: Optional[float] = None,
        all_lags: Optional[dict[int, float]] = None,
        error: Optional[str] = None
    ):
        self.ticker = ticker
        self.best_lag = best_lag
        self.best_corr = best_corr
        self.all_lags = all_lags or {}
        self.error = error
    
    @property
    def valid(self) -> bool:
        return self.error is None and self.best_corr is not None
    
    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "best_lag": self.best_lag,
            "best_corr": self.best_corr,
            "all_lags": self.all_lags,
            "error": self.error
        }


def discover_lead_lag(
    target: pd.Series,
    candidate: pd.Series,
    ticker: str,
    max_lag: int = 4,
    min_obs: int = 50
) -> LeadLagResult:
    """
    Discover optimal lead-lag relationship.
    
    CONVENTION:
        lag = +k => candidate LEADS target by k periods
                    candidate.shift(k) aligns with target
                    Interpretation: candidate moves first, target follows
        
        lag = 0  => contemporaneous (same-week correlation)
        
        lag = -k => candidate LAGS target by k periods
                    Interpretation: target moves first, candidate follows
    
    The best_lag is chosen by maximum absolute correlation.
    
    Args:
        target: Target series (returns) - e.g., HG=F
        candidate: Candidate series (returns) - e.g., FCX
        ticker: Candidate ticker name
        max_lag: Maximum lag periods to test (±max_lag)
        min_obs: Minimum observations for valid correlation
        
    Returns:
        LeadLagResult with best lag and all tested lags
    """
    result = LeadLagResult(ticker=ticker)
    lag_correlations = {}
    
    try:
        # Test range of lags
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                shifted_candidate = candidate
            else:
                shifted_candidate = candidate.shift(lag)
            
            # Align and compute correlation
            aligned = pd.concat(
                [target, shifted_candidate],
                axis=1,
                keys=["target", "candidate"]
            ).dropna()
            
            if len(aligned) < min_obs:
                continue
            
            corr = aligned["target"].corr(aligned["candidate"])
            
            if not pd.isna(corr):
                lag_correlations[lag] = float(corr)
        
        if not lag_correlations:
            result.error = "no_valid_lags"
            return result
        
        result.all_lags = lag_correlations
        
        # Find best lag (highest absolute correlation)
        best_lag = max(lag_correlations, key=lambda k: abs(lag_correlations[k]))
        result.best_lag = best_lag
        result.best_corr = lag_correlations[best_lag]
        
        return result
        
    except Exception as e:
        result.error = f"error_{type(e).__name__}"
        logger.debug(f"{ticker}: Lead-lag discovery failed - {e}")
        return result


def apply_frozen_lag(
    target: pd.Series,
    candidate: pd.Series,
    ticker: str,
    frozen_lag: int,
    min_obs: int = 20
) -> dict:
    """
    Apply a frozen lag (from IS) to OOS data.
    
    Args:
        target: Target series
        candidate: Candidate series
        ticker: Candidate ticker name
        frozen_lag: Lag to apply (discovered in IS)
        min_obs: Minimum observations
        
    Returns:
        Dict with lag_corr_at_frozen and metadata
    """
    try:
        if frozen_lag == 0:
            shifted_candidate = candidate
        else:
            shifted_candidate = candidate.shift(frozen_lag)
        
        aligned = pd.concat(
            [target, shifted_candidate],
            axis=1,
            keys=["target", "candidate"]
        ).dropna()
        
        if len(aligned) < min_obs:
            return {
                "frozen_lag": frozen_lag,
                "lag_corr_at_frozen": None,
                "n_obs": len(aligned),
                "error": f"insufficient_obs_{len(aligned)}"
            }
        
        corr = aligned["target"].corr(aligned["candidate"])
        
        return {
            "frozen_lag": frozen_lag,
            "lag_corr_at_frozen": float(corr) if not pd.isna(corr) else None,
            "n_obs": len(aligned),
            "error": None
        }
        
    except Exception as e:
        return {
            "frozen_lag": frozen_lag,
            "lag_corr_at_frozen": None,
            "error": f"error_{type(e).__name__}"
        }


def discover_lead_lag_batch(
    target: pd.Series,
    candidates: dict[str, pd.Series],
    max_lag: int = 4,
    min_obs: int = 50
) -> dict[str, LeadLagResult]:
    """
    Discover lead-lag for multiple candidates.
    
    Args:
        target: Target series
        candidates: Dict mapping ticker to series
        max_lag: Maximum lag periods
        min_obs: Minimum observations
        
    Returns:
        Dict mapping ticker to LeadLagResult
    """
    results = {}
    
    for ticker, candidate_series in candidates.items():
        result = discover_lead_lag(
            target=target,
            candidate=candidate_series,
            ticker=ticker,
            max_lag=max_lag,
            min_obs=min_obs
        )
        results[ticker] = result
    
    # Log summary
    non_zero_lags = sum(1 for r in results.values() if r.valid and r.best_lag != 0)
    logger.info(f"Lead-lag discovery: {non_zero_lags}/{len(results)} have non-zero optimal lag")
    
    return results
