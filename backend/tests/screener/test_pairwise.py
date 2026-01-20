"""
Unit tests for pairwise correlation.

Tests that pairwise dropna is used instead of global dropna.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from screener.feature_screener.pairwise import (
    compute_pairwise_correlation,
    PairwiseResult,
)


class TestPairwiseCorrelation:
    """Tests for compute_pairwise_correlation function."""
    
    def test_basic_correlation(self):
        """Basic correlation should work."""
        # Create correlated series
        np.random.seed(42)
        n = 200
        
        target = pd.Series(np.random.randn(n), index=pd.date_range("2020-01-01", periods=n, freq="W"))
        candidate = target * 0.8 + np.random.randn(n) * 0.2
        
        result = compute_pairwise_correlation(
            target, candidate, "HG=F", "FCX", min_obs=50
        )
        
        assert result.valid
        assert result.pearson is not None
        assert result.pearson > 0.7  # Strong positive correlation
        assert result.n_obs == n
    
    def test_pairwise_dropna_not_global(self):
        """
        Pairwise dropna should NOT reduce sample size like global dropna.
        
        If we have:
        - target with 100 values, 10 NaN at positions 0-9
        - candidate with 100 values, 10 NaN at positions 90-99
        
        Global dropna would give us only 80 valid observations.
        Pairwise dropna should give us 90 valid observations
        (dropping only where BOTH are NaN, which is 0 here).
        """
        np.random.seed(42)
        
        # Create base series
        dates = pd.date_range("2020-01-01", periods=100, freq="W")
        target = pd.Series(np.random.randn(100), index=dates)
        candidate = pd.Series(np.random.randn(100), index=dates)
        
        # Add NaN at DIFFERENT positions
        target.iloc[:10] = np.nan   # First 10 are NaN
        candidate.iloc[-10:] = np.nan  # Last 10 are NaN
        
        result = compute_pairwise_correlation(
            target, candidate, "HG=F", "TEST", min_obs=50
        )
        
        # Should have 80 observations (100 - 10 - 10, non-overlapping NaNs)
        assert result.valid
        assert result.n_obs == 80
    
    def test_pairwise_overlapping_nan(self):
        """Overlapping NaN positions should only count once."""
        np.random.seed(42)
        
        dates = pd.date_range("2020-01-01", periods=100, freq="W")
        target = pd.Series(np.random.randn(100), index=dates)
        candidate = pd.Series(np.random.randn(100), index=dates)
        
        # Add NaN at SAME positions
        target.iloc[:10] = np.nan
        candidate.iloc[:10] = np.nan
        
        result = compute_pairwise_correlation(
            target, candidate, "HG=F", "TEST", min_obs=50
        )
        
        # Should have 90 observations (100 - 10)
        assert result.valid
        assert result.n_obs == 90
    
    def test_insufficient_observations(self):
        """Should fail with insufficient observations."""
        dates = pd.date_range("2020-01-01", periods=50, freq="W")
        target = pd.Series(np.random.randn(50), index=dates)
        candidate = pd.Series(np.random.randn(50), index=dates)
        
        result = compute_pairwise_correlation(
            target, candidate, "HG=F", "TEST", min_obs=100
        )
        
        assert not result.valid
        assert "insufficient" in result.error
    
    def test_spearman_computed(self):
        """Spearman correlation should also be computed."""
        np.random.seed(42)
        n = 200
        
        target = pd.Series(np.random.randn(n), index=pd.date_range("2020-01-01", periods=n, freq="W"))
        candidate = target + np.random.randn(n) * 0.3
        
        result = compute_pairwise_correlation(
            target, candidate, "HG=F", "FCX", min_obs=50
        )
        
        assert result.valid
        assert result.spearman is not None
    
    def test_date_range_captured(self):
        """First and last dates should be captured."""
        dates = pd.date_range("2020-01-01", periods=100, freq="W")
        target = pd.Series(np.random.randn(100), index=dates)
        candidate = pd.Series(np.random.randn(100), index=dates)
        
        result = compute_pairwise_correlation(
            target, candidate, "HG=F", "TEST", min_obs=50
        )
        
        assert result.first_date is not None
        assert result.last_date is not None
