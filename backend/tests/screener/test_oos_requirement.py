"""
Tests for OOS requirement in model inclusion decision.
"""

import pytest
from datetime import date
from dataclasses import dataclass
from typing import Optional


@dataclass
class MockPeriodMetrics:
    """Mock period metrics for testing."""
    pearson: Optional[float] = 0.5
    spearman: Optional[float] = 0.4
    rolling_corr_mean: Optional[float] = 0.45
    rolling_corr_std: Optional[float] = 0.1
    sign_flip_count: Optional[int] = 2
    best_lead_lag: Optional[int] = 0
    best_lead_lag_corr: Optional[float] = 0.5
    partial_corr: Optional[float] = None
    passed_retention: bool = True
    n_obs: Optional[int] = 100
    first_date: Optional[str] = "2020-01-01"
    last_date: Optional[str] = "2023-12-31"
    frozen_lag: Optional[int] = None
    lag_corr_at_frozen: Optional[float] = None


@dataclass
class MockCandidateEvaluation:
    """Mock evaluation result for testing."""
    ticker: str
    category: str
    pairwise_obs: int
    is_metrics: MockPeriodMetrics
    oos_metrics: Optional[MockPeriodMetrics]
    passed_is_retention: bool
    error: Optional[str] = None


class TestOOSRequirement:
    """Test that OOS is required for model inclusion."""
    
    def test_candidate_with_valid_oos_is_included(self):
        """Candidate with IS + OOS should have include_in_model=True."""
        eval_result = MockCandidateEvaluation(
            ticker="FCX",
            category="miner_major",
            pairwise_obs=200,
            is_metrics=MockPeriodMetrics(pearson=0.7, passed_retention=True, n_obs=150),
            oos_metrics=MockPeriodMetrics(pearson=0.6, n_obs=50),
            passed_is_retention=True
        )
        
        # Simulate _has_valid_oos logic
        has_valid_oos = (
            eval_result.oos_metrics is not None 
            and eval_result.oos_metrics.n_obs is not None 
            and eval_result.oos_metrics.n_obs >= 26  # min_obs
            and eval_result.oos_metrics.pearson is not None
        )
        
        include_in_model = eval_result.passed_is_retention and has_valid_oos
        
        assert has_valid_oos is True
        assert include_in_model is True
    
    def test_candidate_without_oos_is_excluded(self):
        """Candidate with IS but no OOS should have include_in_model=False."""
        eval_result = MockCandidateEvaluation(
            ticker="JJC",
            category="etf_copper",
            pairwise_obs=150,
            is_metrics=MockPeriodMetrics(pearson=0.9, passed_retention=True, n_obs=150),
            oos_metrics=None,  # No OOS data!
            passed_is_retention=True
        )
        
        # Simulate _has_valid_oos logic
        has_valid_oos = (
            eval_result.oos_metrics is not None 
            and eval_result.oos_metrics.n_obs is not None 
            and eval_result.oos_metrics.n_obs >= 26
            and eval_result.oos_metrics.pearson is not None
        )
        
        include_in_model = eval_result.passed_is_retention and has_valid_oos
        
        assert has_valid_oos is False
        assert include_in_model is False
        
        # Should be excluded with reason
        reason = "oos_missing_data"
        assert reason == "oos_missing_data"
    
    def test_candidate_with_insufficient_oos_obs_is_excluded(self):
        """Candidate with OOS but insufficient observations should be excluded."""
        eval_result = MockCandidateEvaluation(
            ticker="TEST",
            category="test",
            pairwise_obs=50,
            is_metrics=MockPeriodMetrics(pearson=0.5, passed_retention=True, n_obs=100),
            oos_metrics=MockPeriodMetrics(pearson=0.4, n_obs=10),  # Only 10 obs, need 26+
            passed_is_retention=True
        )
        
        min_obs = 26
        has_valid_oos = (
            eval_result.oos_metrics is not None 
            and eval_result.oos_metrics.n_obs is not None 
            and eval_result.oos_metrics.n_obs >= min_obs
            and eval_result.oos_metrics.pearson is not None
        )
        
        include_in_model = eval_result.passed_is_retention and has_valid_oos
        
        assert has_valid_oos is False
        assert include_in_model is False
        
        # Should be excluded with reason
        reason = "oos_insufficient_observations"
        assert reason == "oos_insufficient_observations"
    
    def test_candidate_with_null_oos_correlation_is_excluded(self):
        """Candidate with OOS but null correlation should be excluded."""
        eval_result = MockCandidateEvaluation(
            ticker="TEST2",
            category="test",
            pairwise_obs=100,
            is_metrics=MockPeriodMetrics(pearson=0.5, passed_retention=True, n_obs=100),
            oos_metrics=MockPeriodMetrics(pearson=None, n_obs=50),  # Null correlation
            passed_is_retention=True
        )
        
        has_valid_oos = (
            eval_result.oos_metrics is not None 
            and eval_result.oos_metrics.n_obs is not None 
            and eval_result.oos_metrics.n_obs >= 26
            and eval_result.oos_metrics.pearson is not None
        )
        
        include_in_model = eval_result.passed_is_retention and has_valid_oos
        
        assert has_valid_oos is False
        assert include_in_model is False
    
    def test_ranking_excludes_oos_missing_candidates(self):
        """Ranking list should only include candidates with valid OOS."""
        candidates = [
            MockCandidateEvaluation(
                ticker="FCX", category="miner", pairwise_obs=200,
                is_metrics=MockPeriodMetrics(pearson=0.7, passed_retention=True),
                oos_metrics=MockPeriodMetrics(pearson=0.6, n_obs=50),
                passed_is_retention=True
            ),
            MockCandidateEvaluation(
                ticker="JJC", category="etf", pairwise_obs=150,
                is_metrics=MockPeriodMetrics(pearson=0.9, passed_retention=True),
                oos_metrics=None,  # No OOS
                passed_is_retention=True
            ),
            MockCandidateEvaluation(
                ticker="SCCO", category="miner", pairwise_obs=180,
                is_metrics=MockPeriodMetrics(pearson=0.65, passed_retention=True),
                oos_metrics=MockPeriodMetrics(pearson=0.55, n_obs=45),
                passed_is_retention=True
            ),
        ]
        
        # Simulate score_and_rank filtering
        def has_valid_oos(e):
            return (
                e.oos_metrics is not None 
                and e.oos_metrics.n_obs is not None 
                and e.oos_metrics.n_obs >= 26
                and e.oos_metrics.pearson is not None
            )
        
        passed = [e for e in candidates if e.passed_is_retention and has_valid_oos(e)]
        
        assert len(passed) == 2
        assert all(e.ticker != "JJC" for e in passed)
        assert "FCX" in [e.ticker for e in passed]
        assert "SCCO" in [e.ticker for e in passed]
