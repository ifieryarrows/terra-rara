"""
IS/OOS evaluator for correlation analysis.

Splits data into In-Sample and Out-of-Sample periods,
computes metrics for each, and applies retention filters.
"""

import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd

from screener.feature_screener.pairwise import compute_pairwise_correlation, PairwiseResult
from screener.feature_screener.rolling import compute_rolling_correlation, RollingResult
from screener.feature_screener.lead_lag import discover_lead_lag, apply_frozen_lag, LeadLagResult
from screener.feature_screener.partial_corr import compute_partial_correlation, PartialCorrResult

logger = logging.getLogger(__name__)


class PeriodMetrics:
    """Metrics computed for a single period (IS or OOS)."""
    
    def __init__(self):
        self.pearson: Optional[float] = None
        self.spearman: Optional[float] = None
        self.rolling_corr_mean: Optional[float] = None
        self.rolling_corr_std: Optional[float] = None
        self.sign_flip_count: Optional[int] = None
        self.best_lead_lag: Optional[int] = None
        self.best_lead_lag_corr: Optional[float] = None
        self.frozen_lag: Optional[int] = None
        self.lag_corr_at_frozen: Optional[float] = None
        self.partial_corr: Optional[float] = None
        self.passed_retention: Optional[bool] = None
        self.n_obs: Optional[int] = None
        self.first_date: Optional[str] = None
        self.last_date: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "pearson": self.pearson,
            "spearman": self.spearman,
            "rolling_corr_mean": self.rolling_corr_mean,
            "rolling_corr_std": self.rolling_corr_std,
            "sign_flip_count": self.sign_flip_count,
            "best_lead_lag": self.best_lead_lag,
            "best_lead_lag_corr": self.best_lead_lag_corr,
            "frozen_lag": self.frozen_lag,
            "lag_corr_at_frozen": self.lag_corr_at_frozen,
            "partial_corr": self.partial_corr,
            "passed_retention": self.passed_retention,
            "n_obs": self.n_obs,
            "first_date": self.first_date,
            "last_date": self.last_date
        }


class CandidateEvaluation:
    """Complete evaluation of a candidate symbol."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.category: Optional[str] = None
        self.pairwise_obs: int = 0
        self.is_metrics = PeriodMetrics()
        self.oos_metrics: Optional[PeriodMetrics] = None
        self.fitted_partial_params: Optional[dict] = None
        self.error: Optional[str] = None
    
    @property
    def passed_is_retention(self) -> bool:
        """Check if candidate passed IS retention threshold."""
        return self.is_metrics.passed_retention == True
    
    def to_dict(self) -> dict:
        result = {
            "ticker": self.ticker,
            "category": self.category,
            "pairwise_obs": self.pairwise_obs,
            "is": self.is_metrics.to_dict(),
            "error": self.error
        }
        if self.oos_metrics:
            result["oos"] = self.oos_metrics.to_dict()
        return result


def split_by_date(
    series: pd.Series,
    is_start: date,
    is_end: date,
    oos_start: date,
    oos_end: Optional[date] = None
) -> tuple[pd.Series, pd.Series]:
    """
    Split series into IS and OOS periods.
    
    Args:
        series: Input series with date index
        is_start: IS period start
        is_end: IS period end (inclusive)
        oos_start: OOS period start
        oos_end: OOS period end (None = all remaining data)
        
    Returns:
        Tuple of (is_series, oos_series)
    """
    # Ensure index is datetime-like
    if not isinstance(series.index.dtype, (pd.DatetimeTZDtype,)):
        series.index = pd.to_datetime(series.index)
    
    # Convert dates to datetime if needed
    is_start_dt = pd.Timestamp(is_start)
    is_end_dt = pd.Timestamp(is_end)
    oos_start_dt = pd.Timestamp(oos_start)
    oos_end_dt = pd.Timestamp(oos_end) if oos_end else None
    
    # Split
    is_mask = (series.index >= is_start_dt) & (series.index <= is_end_dt)
    is_series = series[is_mask]
    
    if oos_end_dt:
        oos_mask = (series.index >= oos_start_dt) & (series.index <= oos_end_dt)
    else:
        oos_mask = series.index >= oos_start_dt
    oos_series = series[oos_mask]
    
    return is_series, oos_series


class Evaluator:
    """
    Evaluates candidates against target with IS/OOS split.
    """
    
    def __init__(
        self,
        target_returns: pd.Series,
        is_start: date,
        is_end: date,
        oos_start: date,
        oos_end: Optional[date] = None,
        rolling_window: int = 26,
        lead_lag_max: int = 4,
        sign_flip_epsilon: float = 0.05,
        min_is_corr_threshold: float = 0.5,
        min_obs: int = 104,
        controls: Optional[dict[str, pd.Series]] = None,
        enable_partial_corr: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            target_returns: Target symbol returns series
            is_start: IS period start date
            is_end: IS period end date
            oos_start: OOS period start date
            oos_end: OOS period end date (None = run date)
            rolling_window: Window for rolling correlation
            lead_lag_max: Maximum lead-lag periods to test
            sign_flip_epsilon: Epsilon for sign flip detection
            min_is_corr_threshold: Minimum IS correlation for retention
            min_obs: Minimum observations for valid correlation
            controls: Control series for partial correlation
            enable_partial_corr: Whether to compute partial correlation
        """
        self.target_returns = target_returns
        self.is_start = is_start
        self.is_end = is_end
        self.oos_start = oos_start
        self.oos_end = oos_end
        self.rolling_window = rolling_window
        self.lead_lag_max = lead_lag_max
        self.sign_flip_epsilon = sign_flip_epsilon
        self.min_is_corr_threshold = min_is_corr_threshold
        self.min_obs = min_obs
        self.controls = controls or {}
        self.enable_partial_corr = enable_partial_corr
        
        # Split target
        self.target_is, self.target_oos = split_by_date(
            target_returns, is_start, is_end, oos_start, oos_end
        )
        
        # Split controls
        self.controls_is = {}
        self.controls_oos = {}
        for name, series in self.controls.items():
            ctrl_is, ctrl_oos = split_by_date(
                series, is_start, is_end, oos_start, oos_end
            )
            self.controls_is[name] = ctrl_is
            self.controls_oos[name] = ctrl_oos
        
        logger.info(
            f"Evaluator initialized: IS={len(self.target_is)} obs, "
            f"OOS={len(self.target_oos)} obs"
        )
    
    def evaluate(
        self,
        candidate_returns: pd.Series,
        ticker: str,
        category: Optional[str] = None
    ) -> CandidateEvaluation:
        """
        Evaluate a single candidate.
        
        Args:
            candidate_returns: Candidate returns series
            ticker: Ticker symbol
            category: Category for this ticker
            
        Returns:
            CandidateEvaluation with IS and optionally OOS metrics
        """
        eval_result = CandidateEvaluation(ticker)
        eval_result.category = category
        
        # Split candidate
        candidate_is, candidate_oos = split_by_date(
            candidate_returns,
            self.is_start, self.is_end,
            self.oos_start, self.oos_end
        )
        
        # ========== IS Analysis ==========
        
        # Pairwise correlation
        pairwise_is = compute_pairwise_correlation(
            target=self.target_is,
            candidate=candidate_is,
            target_ticker="HG=F",
            candidate_ticker=ticker,
            min_obs=self.min_obs // 2  # IS can have fewer obs
        )
        
        if not pairwise_is.valid:
            eval_result.error = f"is_pairwise_failed_{pairwise_is.error}"
            return eval_result
        
        eval_result.pairwise_obs = pairwise_is.n_obs
        eval_result.is_metrics.pearson = pairwise_is.pearson
        eval_result.is_metrics.spearman = pairwise_is.spearman
        eval_result.is_metrics.n_obs = pairwise_is.n_obs
        eval_result.is_metrics.first_date = pairwise_is.first_date
        eval_result.is_metrics.last_date = pairwise_is.last_date
        
        # Rolling correlation
        rolling_is = compute_rolling_correlation(
            target=self.target_is,
            candidate=candidate_is,
            ticker=ticker,
            window=self.rolling_window,
            epsilon=self.sign_flip_epsilon
        )
        
        if rolling_is.valid:
            eval_result.is_metrics.rolling_corr_mean = rolling_is.rolling_corr_mean
            eval_result.is_metrics.rolling_corr_std = rolling_is.rolling_corr_std
            eval_result.is_metrics.sign_flip_count = rolling_is.sign_flip_count
        
        # Lead-lag discovery
        lead_lag_is = discover_lead_lag(
            target=self.target_is,
            candidate=candidate_is,
            ticker=ticker,
            max_lag=self.lead_lag_max
        )
        
        if lead_lag_is.valid:
            eval_result.is_metrics.best_lead_lag = lead_lag_is.best_lag
            eval_result.is_metrics.best_lead_lag_corr = lead_lag_is.best_corr
        
        # Partial correlation
        if self.enable_partial_corr and self.controls_is:
            partial_is = compute_partial_correlation(
                target=self.target_is,
                candidate=candidate_is,
                controls=self.controls_is,
                ticker=ticker
            )
            if partial_is.valid:
                eval_result.is_metrics.partial_corr = partial_is.partial_corr
                eval_result.fitted_partial_params = partial_is.fitted_params
        
        # Retention check
        if pairwise_is.pearson is not None:
            eval_result.is_metrics.passed_retention = (
                abs(pairwise_is.pearson) >= self.min_is_corr_threshold
            )
        else:
            eval_result.is_metrics.passed_retention = False
        
        # ========== OOS Analysis (only if passed IS) ==========
        
        if not eval_result.passed_is_retention:
            return eval_result
        
        if len(candidate_oos) < 10:
            return eval_result
        
        eval_result.oos_metrics = PeriodMetrics()
        
        # Pairwise correlation
        pairwise_oos = compute_pairwise_correlation(
            target=self.target_oos,
            candidate=candidate_oos,
            target_ticker="HG=F",
            candidate_ticker=ticker,
            min_obs=10  # OOS can be shorter
        )
        
        if pairwise_oos.valid:
            eval_result.oos_metrics.pearson = pairwise_oos.pearson
            eval_result.oos_metrics.spearman = pairwise_oos.spearman
            eval_result.oos_metrics.n_obs = pairwise_oos.n_obs
            eval_result.oos_metrics.first_date = pairwise_oos.first_date
            eval_result.oos_metrics.last_date = pairwise_oos.last_date
        
        # Rolling correlation
        rolling_oos = compute_rolling_correlation(
            target=self.target_oos,
            candidate=candidate_oos,
            ticker=ticker,
            window=min(self.rolling_window, len(candidate_oos) // 2),
            epsilon=self.sign_flip_epsilon
        )
        
        if rolling_oos.valid:
            eval_result.oos_metrics.rolling_corr_mean = rolling_oos.rolling_corr_mean
            eval_result.oos_metrics.rolling_corr_std = rolling_oos.rolling_corr_std
            eval_result.oos_metrics.sign_flip_count = rolling_oos.sign_flip_count
        
        # Apply frozen lag
        frozen_lag = lead_lag_is.best_lag if lead_lag_is.valid else 0
        eval_result.oos_metrics.frozen_lag = frozen_lag
        
        frozen_result = apply_frozen_lag(
            target=self.target_oos,
            candidate=candidate_oos,
            ticker=ticker,
            frozen_lag=frozen_lag
        )
        eval_result.oos_metrics.lag_corr_at_frozen = frozen_result.get("lag_corr_at_frozen")
        
        # Partial correlation (with frozen params)
        if self.enable_partial_corr and self.controls_oos and eval_result.fitted_partial_params:
            partial_oos = compute_partial_correlation(
                target=self.target_oos,
                candidate=candidate_oos,
                controls=self.controls_oos,
                ticker=ticker,
                fitted_params=eval_result.fitted_partial_params
            )
            if partial_oos.valid:
                eval_result.oos_metrics.partial_corr = partial_oos.partial_corr
        
        return eval_result
    
    def evaluate_batch(
        self,
        candidates: dict[str, tuple[pd.Series, Optional[str]]],
        progress_callback=None
    ) -> list[CandidateEvaluation]:
        """
        Evaluate multiple candidates.
        
        Args:
            candidates: Dict of ticker -> (returns_series, category)
            progress_callback: Optional callback(current, total, ticker)
            
        Returns:
            List of CandidateEvaluation objects
        """
        results = []
        total = len(candidates)
        
        for i, (ticker, (returns, category)) in enumerate(candidates.items()):
            if progress_callback:
                progress_callback(i + 1, total, ticker)
            
            result = self.evaluate(returns, ticker, category)
            results.append(result)
            
            if (i + 1) % 50 == 0:
                passed = sum(1 for r in results if r.passed_is_retention)
                logger.info(f"Evaluated {i + 1}/{total} ({passed} passed retention)")
        
        passed_count = sum(1 for r in results if r.passed_is_retention)
        logger.info(f"Evaluation complete: {passed_count}/{len(results)} passed IS retention")
        
        return results
