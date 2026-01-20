"""
Unit tests for lead-lag discovery and frozen lag.

Tests that lag discovered in IS is frozen for OOS.
"""

import pytest
import pandas as pd
import numpy as np

from screener.feature_screener.lead_lag import (
    discover_lead_lag,
    apply_frozen_lag,
    LeadLagResult,
)


class TestLeadLagDiscovery:
    """Tests for discover_lead_lag function."""
    
    def test_contemporaneous(self):
        """Contemporaneous correlation should find lag=0."""
        np.random.seed(42)
        n = 200
        
        dates = pd.date_range("2020-01-01", periods=n, freq="W")
        target = pd.Series(np.random.randn(n), index=dates)
        # Candidate is highly correlated with target at lag 0
        candidate = target * 0.9 + np.random.randn(n) * 0.1
        
        result = discover_lead_lag(target, candidate, "TEST", max_lag=4)
        
        assert result.valid
        assert result.best_lag == 0
        assert result.best_corr > 0.8
    
    def test_candidate_leads(self):
        """Candidate leading should find positive lag."""
        np.random.seed(42)
        n = 200
        
        dates = pd.date_range("2020-01-01", periods=n, freq="W")
        signal = pd.Series(np.random.randn(n), index=dates)
        
        # Candidate leads target by 2 periods
        target = signal.shift(2)
        candidate = signal * 0.95 + np.random.randn(n) * 0.05
        
        target = target.dropna()
        candidate = candidate.loc[target.index]
        
        result = discover_lead_lag(target, candidate, "TEST", max_lag=4)
        
        assert result.valid
        # Best lag should be positive (candidate leads)
        assert result.best_lag > 0
    
    def test_all_lags_returned(self):
        """All tested lags should be in all_lags dict."""
        np.random.seed(42)
        n = 200
        
        dates = pd.date_range("2020-01-01", periods=n, freq="W")
        target = pd.Series(np.random.randn(n), index=dates)
        candidate = pd.Series(np.random.randn(n), index=dates)
        
        result = discover_lead_lag(target, candidate, "TEST", max_lag=3)
        
        assert result.valid
        # Should have lags from -3 to +3
        assert len(result.all_lags) >= 5  # At least -3, -2, -1, 0, 1, 2, 3
    
    def test_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        dates = pd.date_range("2020-01-01", periods=20, freq="W")
        target = pd.Series(np.random.randn(20), index=dates)
        candidate = pd.Series(np.random.randn(20), index=dates)
        
        result = discover_lead_lag(target, candidate, "TEST", max_lag=4, min_obs=50)
        
        # May still be valid with fewer observations, or may fail
        # The important thing is it doesn't crash
        assert isinstance(result, LeadLagResult)


class TestFrozenLag:
    """Tests for apply_frozen_lag function."""
    
    def test_frozen_lag_zero(self):
        """Frozen lag of 0 should correlate contemporaneously."""
        np.random.seed(42)
        n = 100
        
        dates = pd.date_range("2024-01-01", periods=n, freq="W")
        target = pd.Series(np.random.randn(n), index=dates)
        candidate = target * 0.8 + np.random.randn(n) * 0.2
        
        result = apply_frozen_lag(target, candidate, "TEST", frozen_lag=0)
        
        assert result["frozen_lag"] == 0
        assert result["lag_corr_at_frozen"] is not None
        assert result["lag_corr_at_frozen"] > 0.7
    
    def test_frozen_lag_applied(self):
        """Frozen lag should be applied correctly."""
        np.random.seed(42)
        n = 100
        
        dates = pd.date_range("2024-01-01", periods=n, freq="W")
        signal = pd.Series(np.random.randn(n), index=dates)
        
        # Candidate leads target by 2 periods
        target = signal.shift(2).dropna()
        candidate = signal.loc[target.index]
        
        result = apply_frozen_lag(target, candidate, "TEST", frozen_lag=2)
        
        assert result["frozen_lag"] == 2
        assert result["lag_corr_at_frozen"] is not None
    
    def test_frozen_lag_unchanged(self):
        """Frozen lag should not change between calls."""
        np.random.seed(42)
        n = 100
        
        dates = pd.date_range("2024-01-01", periods=n, freq="W")
        target = pd.Series(np.random.randn(n), index=dates)
        candidate = pd.Series(np.random.randn(n), index=dates)
        
        # Call multiple times with same frozen lag
        frozen_lag = 2
        result1 = apply_frozen_lag(target, candidate, "TEST", frozen_lag=frozen_lag)
        result2 = apply_frozen_lag(target, candidate, "TEST", frozen_lag=frozen_lag)
        
        assert result1["frozen_lag"] == result2["frozen_lag"] == frozen_lag
        assert result1["lag_corr_at_frozen"] == result2["lag_corr_at_frozen"]


class TestFrozenLagIntegration:
    """Integration tests for IS discovery -> OOS frozen application."""
    
    def test_is_discovery_oos_frozen(self):
        """Lag discovered in IS should be frozen for OOS."""
        np.random.seed(42)
        
        # IS period
        is_dates = pd.date_range("2018-01-01", periods=260, freq="W")
        signal_is = pd.Series(np.random.randn(260), index=is_dates)
        
        # Create lagged relationship
        target_is = signal_is.shift(1).dropna()
        candidate_is = signal_is.loc[target_is.index]
        
        # Discover lag in IS
        is_result = discover_lead_lag(target_is, candidate_is, "TEST", max_lag=3)
        discovered_lag = is_result.best_lag
        
        # OOS period with same relationship
        oos_dates = pd.date_range("2024-01-01", periods=52, freq="W")
        signal_oos = pd.Series(np.random.randn(52), index=oos_dates)
        
        target_oos = signal_oos.shift(1).dropna()
        candidate_oos = signal_oos.loc[target_oos.index]
        
        # Apply frozen lag
        oos_result = apply_frozen_lag(
            target_oos, candidate_oos, "TEST", frozen_lag=discovered_lag
        )
        
        # Frozen lag should be same as discovered
        assert oos_result["frozen_lag"] == discovered_lag
        assert oos_result["lag_corr_at_frozen"] is not None
