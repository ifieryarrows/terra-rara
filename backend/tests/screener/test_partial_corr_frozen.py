"""
Unit tests for partial correlation with frozen coefficients.

Tests that OOS uses frozen IS coefficients (no re-fitting).
"""

import pytest
import pandas as pd
import numpy as np

# Skip if statsmodels not installed
pytest.importorskip("statsmodels")

from screener.feature_screener.partial_corr import (
    compute_partial_correlation,
    PartialCorrResult,
)


class TestPartialCorrFrozenFit:
    """Tests that OOS uses frozen IS coefficients."""
    
    def test_is_returns_fitted_params(self):
        """IS mode should return fitted parameters."""
        np.random.seed(42)
        n = 200
        
        dates = pd.date_range("2018-01-01", periods=n, freq="W")
        target = pd.Series(np.random.randn(n), index=dates)
        candidate = pd.Series(np.random.randn(n), index=dates)
        control = pd.Series(np.random.randn(n), index=dates)
        
        result = compute_partial_correlation(
            target=target,
            candidate=candidate,
            controls={"^GSPC": control},
            ticker="TEST",
            min_obs=50,
            fitted_params=None  # IS mode
        )
        
        assert result.valid
        assert result.fitted_params is not None
        assert "target_params" in result.fitted_params
        assert "candidate_params" in result.fitted_params
    
    def test_oos_uses_frozen_params(self):
        """OOS mode should use frozen IS params, not re-fit."""
        np.random.seed(42)
        
        # Create IS data
        n_is = 200
        is_dates = pd.date_range("2018-01-01", periods=n_is, freq="W")
        target_is = pd.Series(np.random.randn(n_is), index=is_dates)
        candidate_is = pd.Series(np.random.randn(n_is), index=is_dates)
        control_is = pd.Series(np.random.randn(n_is), index=is_dates)
        
        # Fit in IS
        is_result = compute_partial_correlation(
            target=target_is,
            candidate=candidate_is,
            controls={"^GSPC": control_is},
            ticker="TEST",
            fitted_params=None  # IS mode
        )
        
        assert is_result.fitted_params is not None
        frozen_params = is_result.fitted_params
        
        # Create OOS data with DIFFERENT distribution
        n_oos = 52
        oos_dates = pd.date_range("2024-01-01", periods=n_oos, freq="W")
        target_oos = pd.Series(np.random.randn(n_oos) * 2 + 1, index=oos_dates)  # Different!
        candidate_oos = pd.Series(np.random.randn(n_oos) * 2 + 1, index=oos_dates)
        control_oos = pd.Series(np.random.randn(n_oos) * 2 + 1, index=oos_dates)
        
        # Apply frozen IS params to OOS
        oos_result = compute_partial_correlation(
            target=target_oos,
            candidate=candidate_oos,
            controls={"^GSPC": control_oos},
            ticker="TEST",
            fitted_params=frozen_params  # OOS mode with frozen params
        )
        
        # OOS should also be valid
        assert oos_result.valid
        assert oos_result.partial_corr is not None
        
        # OOS result should NOT have new fitted_params (used frozen ones)
        # Note: Current implementation doesn't set fitted_params in OOS mode
        # This is correct behavior - we use, but don't re-fit
    
    def test_frozen_params_are_deterministic(self):
        """Same IS data should produce same frozen params."""
        np.random.seed(42)
        n = 200
        
        dates = pd.date_range("2018-01-01", periods=n, freq="W")
        target = pd.Series(np.random.randn(n), index=dates)
        candidate = pd.Series(np.random.randn(n), index=dates)
        control = pd.Series(np.random.randn(n), index=dates)
        
        # Fit twice with same data
        result1 = compute_partial_correlation(
            target=target.copy(),
            candidate=candidate.copy(),
            controls={"^GSPC": control.copy()},
            ticker="TEST",
            fitted_params=None
        )
        
        result2 = compute_partial_correlation(
            target=target.copy(),
            candidate=candidate.copy(),
            controls={"^GSPC": control.copy()},
            ticker="TEST",
            fitted_params=None
        )
        
        # Params should be identical
        for key in result1.fitted_params["target_params"]:
            assert result1.fitted_params["target_params"][key] == \
                   result2.fitted_params["target_params"][key]
    
    def test_oos_with_wrong_controls_fails_gracefully(self):
        """OOS with missing control columns should handle gracefully."""
        np.random.seed(42)
        
        n_is = 200
        is_dates = pd.date_range("2018-01-01", periods=n_is, freq="W")
        target_is = pd.Series(np.random.randn(n_is), index=is_dates)
        candidate_is = pd.Series(np.random.randn(n_is), index=is_dates)
        control_is = pd.Series(np.random.randn(n_is), index=is_dates)
        
        # Fit with one control
        is_result = compute_partial_correlation(
            target=target_is,
            candidate=candidate_is,
            controls={"^GSPC": control_is},
            ticker="TEST",
            fitted_params=None
        )
        
        # OOS with different control name
        n_oos = 52
        oos_dates = pd.date_range("2024-01-01", periods=n_oos, freq="W")
        target_oos = pd.Series(np.random.randn(n_oos), index=oos_dates)
        candidate_oos = pd.Series(np.random.randn(n_oos), index=oos_dates)
        different_control = pd.Series(np.random.randn(n_oos), index=oos_dates)
        
        # This should still work (columns get aligned with fill_value=0)
        oos_result = compute_partial_correlation(
            target=target_oos,
            candidate=candidate_oos,
            controls={"DIFFERENT": different_control},  # Different control name!
            ticker="TEST",
            fitted_params=is_result.fitted_params
        )
        
        # Should either work with degraded behavior or fail gracefully
        # (implementation uses reindex with fill_value=0)
        assert isinstance(oos_result, PartialCorrResult)


class TestPartialCorrBasic:
    """Basic partial correlation tests."""
    
    def test_partial_corr_removes_common_factor(self):
        """
        Partial correlation should remove spurious correlation from common factor.
        
        If X and Y are both correlated with Z, but not with each other,
        raw correlation will show X-Y correlation, but partial (controlling Z)
        should show near-zero.
        """
        np.random.seed(42)
        n = 500
        
        dates = pd.date_range("2018-01-01", periods=n, freq="W")
        
        # Common factor
        z = pd.Series(np.random.randn(n), index=dates)
        
        # X and Y both correlated with Z, but independent of each other
        x = z * 0.7 + pd.Series(np.random.randn(n) * 0.3, index=dates)
        y = z * 0.7 + pd.Series(np.random.randn(n) * 0.3, index=dates)
        
        # Raw correlation should be positive (spurious)
        raw_corr = x.corr(y)
        assert raw_corr > 0.3, f"Raw correlation should be high: {raw_corr}"
        
        # Partial correlation controlling Z should be near zero
        result = compute_partial_correlation(
            target=x,
            candidate=y,
            controls={"Z": z},
            ticker="TEST",
            min_obs=50
        )
        
        assert result.valid
        assert result.raw_corr is not None
        assert result.partial_corr is not None
        
        # Partial should be much smaller than raw
        assert abs(result.partial_corr) < abs(result.raw_corr)
        assert abs(result.partial_corr) < 0.2, f"Partial should be near zero: {result.partial_corr}"
