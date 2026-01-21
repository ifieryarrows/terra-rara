"""
Unit tests for target redundancy filter.

Verifies that candidates with:
- abs(oos_pearson) > 0.95 AND frozen_lag = 0
are excluded as "target copies" (oos_target_redundant).

Candidates with high correlation but lag != 0 should NOT be excluded
(they may be leading signals, not copies).
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Optional


@dataclass
class MockISMetrics:
    """Mock IS metrics for testing."""
    pearson: float
    n_obs: int
    best_lead_lag: int


@dataclass
class MockOOSMetrics:
    """Mock OOS metrics for testing."""
    pearson: float
    n_obs: int
    rolling_corr_std: float
    partial_corr: Optional[float]
    frozen_lag: int
    lag_corr_at_frozen: Optional[float] = None


@dataclass
class MockDecision:
    """Mock decision for testing."""
    rank: int
    score_composite: float
    include: bool
    reason: str


@dataclass
class MockCandidate:
    """Mock candidate for testing."""
    ticker: str
    category: str
    is_metrics: MockISMetrics
    oos_metrics: MockOOSMetrics
    decision: MockDecision


def apply_target_redundancy_filter(candidate: MockCandidate, max_target_corr: float = 0.95) -> bool:
    """
    Apply target redundancy filter logic.
    Returns True if candidate PASSES (should be included).
    Returns False if candidate is REDUNDANT (should be excluded).
    """
    oos = candidate.oos_metrics
    if oos is None:
        return True  # Can't filter without OOS metrics
    
    frozen_lag = oos.frozen_lag if oos.frozen_lag is not None else candidate.is_metrics.best_lead_lag
    
    # High correlation AT LAG=0 means "target copy", not "leading signal"
    if frozen_lag == 0 and abs(oos.pearson) > max_target_corr:
        return False  # REDUNDANT
    
    return True  # PASSES


class TestTargetRedundancyFilter:
    """Tests for the target redundancy filter."""
    
    def test_cper_like_candidate_excluded(self):
        """CPER-like candidate with lag=0 and corr>0.95 should be excluded."""
        cper = MockCandidate(
            ticker="CPER",
            category="etf_copper",
            is_metrics=MockISMetrics(pearson=0.98, n_obs=300, best_lead_lag=0),
            oos_metrics=MockOOSMetrics(
                pearson=0.99,  # Very high - target copy
                n_obs=100,
                rolling_corr_std=0.02,
                partial_corr=0.85,
                frozen_lag=0   # Contemporaneous
            ),
            decision=MockDecision(rank=1, score_composite=0.95, include=True, reason="is_retained")
        )
        
        result = apply_target_redundancy_filter(cper)
        assert result is False, "CPER with lag=0 and corr=0.99 should be excluded as redundant"
    
    def test_high_corr_with_lead_not_excluded(self):
        """High correlation with lag>0 should NOT be excluded (potential leading signal)."""
        leading_signal = MockCandidate(
            ticker="LEAD",
            category="macro_china",
            is_metrics=MockISMetrics(pearson=0.96, n_obs=300, best_lead_lag=2),
            oos_metrics=MockOOSMetrics(
                pearson=0.96,  # High but with lag
                n_obs=100,
                rolling_corr_std=0.05,
                partial_corr=0.70,
                frozen_lag=2   # LEADING signal, not contemporaneous
            ),
            decision=MockDecision(rank=2, score_composite=0.90, include=True, reason="is_retained")
        )
        
        result = apply_target_redundancy_filter(leading_signal)
        assert result is True, "High corr with lag>0 should NOT be excluded (may be leading signal)"
    
    def test_moderate_corr_at_lag_zero_not_excluded(self):
        """Moderate correlation at lag=0 should not be excluded."""
        moderate = MockCandidate(
            ticker="PICK",
            category="etf_miners",
            is_metrics=MockISMetrics(pearson=0.65, n_obs=300, best_lead_lag=0),
            oos_metrics=MockOOSMetrics(
                pearson=0.68,  # Below threshold
                n_obs=100,
                rolling_corr_std=0.08,
                partial_corr=0.55,
                frozen_lag=0
            ),
            decision=MockDecision(rank=3, score_composite=0.75, include=True, reason="is_retained")
        )
        
        result = apply_target_redundancy_filter(moderate)
        assert result is True, "Moderate corr at lag=0 should NOT be excluded"
    
    def test_exactly_at_threshold_not_excluded(self):
        """Correlation exactly at 0.95 should NOT be excluded (threshold is exclusive)."""
        at_threshold = MockCandidate(
            ticker="EDGE",
            category="miner_major",
            is_metrics=MockISMetrics(pearson=0.95, n_obs=300, best_lead_lag=0),
            oos_metrics=MockOOSMetrics(
                pearson=0.95,  # Exactly at threshold
                n_obs=100,
                rolling_corr_std=0.05,
                partial_corr=0.60,
                frozen_lag=0
            ),
            decision=MockDecision(rank=4, score_composite=0.80, include=True, reason="is_retained")
        )
        
        result = apply_target_redundancy_filter(at_threshold)
        assert result is True, "Correlation exactly at 0.95 should NOT be excluded"
    
    def test_negative_high_corr_at_lag_zero_excluded(self):
        """Negative high correlation at lag=0 should also be excluded (inverse copy)."""
        inverse_copy = MockCandidate(
            ticker="INV",
            category="index_vol",
            is_metrics=MockISMetrics(pearson=-0.97, n_obs=300, best_lead_lag=0),
            oos_metrics=MockOOSMetrics(
                pearson=-0.97,  # High negative
                n_obs=100,
                rolling_corr_std=0.04,
                partial_corr=-0.80,
                frozen_lag=0   # Contemporaneous
            ),
            decision=MockDecision(rank=5, score_composite=0.85, include=True, reason="is_retained")
        )
        
        result = apply_target_redundancy_filter(inverse_copy)
        assert result is False, "High negative corr at lag=0 should be excluded as redundant"
    
    def test_fallback_to_is_best_lag(self):
        """If frozen_lag is None, should fallback to is_metrics.best_lead_lag."""
        fallback = MockCandidate(
            ticker="FALLBACK",
            category="miner_major",
            is_metrics=MockISMetrics(pearson=0.98, n_obs=300, best_lead_lag=0),
            oos_metrics=MockOOSMetrics(
                pearson=0.98,
                n_obs=100,
                rolling_corr_std=0.03,
                partial_corr=0.70,
                frozen_lag=None  # Will fallback to is_metrics.best_lead_lag=0
            ),
            decision=MockDecision(rank=6, score_composite=0.88, include=True, reason="is_retained")
        )
        # Patch frozen_lag to be None (dataclass doesn't allow None by default in our mock)
        fallback.oos_metrics.frozen_lag = None
        
        # The filter logic should use best_lead_lag=0 as fallback
        oos = fallback.oos_metrics
        frozen_lag = oos.frozen_lag if oos.frozen_lag is not None else fallback.is_metrics.best_lead_lag
        assert frozen_lag == 0, "Should fallback to best_lead_lag when frozen_lag is None"
        
        # Then should be excluded
        result = apply_target_redundancy_filter(fallback)
        assert result is False, "Should be excluded when falling back to lag=0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
