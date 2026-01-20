"""
Pairwise correlation computation.

Computes correlation metrics on pairwise intersection of two series,
avoiding global dropna that would reduce sample size.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PairwiseResult:
    """Result of pairwise correlation analysis."""
    
    def __init__(
        self,
        target_ticker: str,
        candidate_ticker: str,
        n_obs: int = 0,
        first_date: Optional[str] = None,
        last_date: Optional[str] = None,
        pearson: Optional[float] = None,
        spearman: Optional[float] = None,
        error: Optional[str] = None
    ):
        self.target_ticker = target_ticker
        self.candidate_ticker = candidate_ticker
        self.n_obs = n_obs
        self.first_date = first_date
        self.last_date = last_date
        self.pearson = pearson
        self.spearman = spearman
        self.error = error
    
    @property
    def valid(self) -> bool:
        return self.error is None and self.pearson is not None
    
    def to_dict(self) -> dict:
        return {
            "target_ticker": self.target_ticker,
            "candidate_ticker": self.candidate_ticker,
            "n_obs": self.n_obs,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "error": self.error
        }


def compute_pairwise_correlation(
    target: pd.Series,
    candidate: pd.Series,
    target_ticker: str = "target",
    candidate_ticker: str = "candidate",
    min_obs: int = 104
) -> PairwiseResult:
    """
    Compute Pearson and Spearman correlation on pairwise intersection.
    
    IMPORTANT: Does NOT use global dropna. Only drops rows where
    BOTH series are missing (pairwise complete observations).
    
    Args:
        target: Target series (e.g., HG=F returns)
        candidate: Candidate series returns
        target_ticker: Target ticker name for logging
        candidate_ticker: Candidate ticker name for logging
        min_obs: Minimum observations required
        
    Returns:
        PairwiseResult with correlation metrics
    """
    result = PairwiseResult(
        target_ticker=target_ticker,
        candidate_ticker=candidate_ticker
    )
    
    try:
        # Align series by index
        aligned = pd.concat(
            [target, candidate],
            axis=1,
            keys=["target", "candidate"]
        )
        
        # Drop only rows where BOTH are NaN (pairwise complete)
        # This is different from dropna() which drops any NaN
        aligned = aligned.dropna()
        
        if len(aligned) < min_obs:
            result.error = f"insufficient_observations_{len(aligned)}"
            result.n_obs = len(aligned)
            return result
        
        result.n_obs = len(aligned)
        result.first_date = str(aligned.index.min())
        result.last_date = str(aligned.index.max())
        
        # Compute correlations
        target_vals = aligned["target"]
        candidate_vals = aligned["candidate"]
        
        # Pearson
        pearson = target_vals.corr(candidate_vals, method="pearson")
        if pd.isna(pearson):
            result.error = "pearson_nan"
            return result
        result.pearson = float(pearson)
        
        # Spearman
        spearman = target_vals.corr(candidate_vals, method="spearman")
        if pd.isna(spearman):
            result.spearman = None  # Non-fatal
        else:
            result.spearman = float(spearman)
        
        return result
        
    except Exception as e:
        result.error = f"error_{type(e).__name__}"
        logger.debug(f"{candidate_ticker}: Correlation failed - {e}")
        return result


def compute_pairwise_batch(
    target: pd.Series,
    candidates: dict[str, pd.Series],
    target_ticker: str = "HG=F",
    min_obs: int = 104
) -> dict[str, PairwiseResult]:
    """
    Compute pairwise correlations for multiple candidates.
    
    Args:
        target: Target series
        candidates: Dict mapping ticker to series
        target_ticker: Target ticker name
        min_obs: Minimum observations
        
    Returns:
        Dict mapping ticker to PairwiseResult
    """
    results = {}
    
    for ticker, candidate_series in candidates.items():
        result = compute_pairwise_correlation(
            target=target,
            candidate=candidate_series,
            target_ticker=target_ticker,
            candidate_ticker=ticker,
            min_obs=min_obs
        )
        results[ticker] = result
    
    valid_count = sum(1 for r in results.values() if r.valid)
    logger.info(f"Computed correlations: {valid_count}/{len(results)} valid")
    
    return results
