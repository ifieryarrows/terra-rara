"""
Partial correlation computation.

Computes residual correlation after regressing out control variables.
IS fit is frozen and applied to OOS.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PartialCorrResult:
    """Result of partial correlation analysis."""
    
    def __init__(
        self,
        ticker: str,
        partial_corr: Optional[float] = None,
        raw_corr: Optional[float] = None,
        controls_used: Optional[list[str]] = None,
        n_obs: int = 0,
        fitted_params: Optional[dict] = None,
        error: Optional[str] = None
    ):
        self.ticker = ticker
        self.partial_corr = partial_corr
        self.raw_corr = raw_corr
        self.controls_used = controls_used or []
        self.n_obs = n_obs
        self.fitted_params = fitted_params
        self.error = error
    
    @property
    def valid(self) -> bool:
        return self.error is None and self.partial_corr is not None
    
    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "partial_corr": self.partial_corr,
            "raw_corr": self.raw_corr,
            "controls_used": self.controls_used,
            "n_obs": self.n_obs,
            "error": self.error
        }


def compute_partial_correlation(
    target: pd.Series,
    candidate: pd.Series,
    controls: dict[str, pd.Series],
    ticker: str,
    min_obs: int = 50,
    fitted_params: Optional[dict] = None
) -> PartialCorrResult:
    """
    Compute partial correlation by regressing out controls.
    
    Uses OLS to regress both target and candidate on controls,
    then correlates the residuals.
    
    If fitted_params is provided (OOS mode), uses frozen coefficients
    from IS instead of fitting new model.
    
    Args:
        target: Target series (returns)
        candidate: Candidate series (returns)
        controls: Dict mapping control name to series
        ticker: Candidate ticker name
        min_obs: Minimum observations
        fitted_params: Pre-fitted OLS params from IS (for OOS)
        
    Returns:
        PartialCorrResult with partial correlation
    """
    result = PartialCorrResult(ticker=ticker)
    
    # Check if statsmodels is available
    try:
        import statsmodels.api as sm
    except ImportError:
        result.error = "statsmodels_not_installed"
        logger.warning("statsmodels not installed, skipping partial correlation")
        return result
    
    try:
        # Build aligned DataFrame with target, candidate, and all controls
        data_dict = {"target": target, "candidate": candidate}
        data_dict.update(controls)
        
        aligned = pd.DataFrame(data_dict).dropna()
        
        if len(aligned) < min_obs:
            result.error = f"insufficient_observations_{len(aligned)}"
            result.n_obs = len(aligned)
            return result
        
        result.n_obs = len(aligned)
        result.controls_used = list(controls.keys())
        
        # Extract variables
        y_target = aligned["target"]
        y_candidate = aligned["candidate"]
        X_controls = aligned[list(controls.keys())]
        X_controls = sm.add_constant(X_controls)  # Add intercept
        
        # Compute raw correlation for reference
        raw_corr = y_target.corr(y_candidate)
        result.raw_corr = float(raw_corr) if not pd.isna(raw_corr) else None
        
        if fitted_params is None:
            # IS mode: fit new models
            model_target = sm.OLS(y_target, X_controls).fit()
            model_candidate = sm.OLS(y_candidate, X_controls).fit()
            
            resid_target = model_target.resid
            resid_candidate = model_candidate.resid
            
            # Store fitted parameters
            result.fitted_params = {
                "target_params": model_target.params.to_dict(),
                "candidate_params": model_candidate.params.to_dict()
            }
        else:
            # OOS mode: apply frozen coefficients
            try:
                target_params = pd.Series(fitted_params["target_params"])
                candidate_params = pd.Series(fitted_params["candidate_params"])
                
                # Ensure column alignment
                target_params = target_params.reindex(X_controls.columns, fill_value=0)
                candidate_params = candidate_params.reindex(X_controls.columns, fill_value=0)
                
                predicted_target = (X_controls * target_params).sum(axis=1)
                predicted_candidate = (X_controls * candidate_params).sum(axis=1)
                
                resid_target = y_target - predicted_target
                resid_candidate = y_candidate - predicted_candidate
                
            except Exception as e:
                result.error = f"frozen_params_error_{type(e).__name__}"
                return result
        
        # Compute partial correlation on residuals
        partial_corr = resid_target.corr(resid_candidate)
        
        if pd.isna(partial_corr):
            result.error = "partial_corr_nan"
            return result
        
        result.partial_corr = float(partial_corr)
        
        return result
        
    except Exception as e:
        result.error = f"error_{type(e).__name__}"
        logger.debug(f"{ticker}: Partial correlation failed - {e}")
        return result


def compute_partial_batch(
    target: pd.Series,
    candidates: dict[str, pd.Series],
    controls: dict[str, pd.Series],
    min_obs: int = 50,
    fitted_params_map: Optional[dict[str, dict]] = None
) -> dict[str, PartialCorrResult]:
    """
    Compute partial correlations for multiple candidates.
    
    Args:
        target: Target series
        candidates: Dict mapping ticker to series
        controls: Dict mapping control name to series
        min_obs: Minimum observations
        fitted_params_map: Optional dict of ticker -> fitted_params (for OOS)
        
    Returns:
        Dict mapping ticker to PartialCorrResult
    """
    results = {}
    
    for ticker, candidate_series in candidates.items():
        fitted_params = None
        if fitted_params_map:
            fitted_params = fitted_params_map.get(ticker)
        
        result = compute_partial_correlation(
            target=target,
            candidate=candidate_series,
            controls=controls,
            ticker=ticker,
            min_obs=min_obs,
            fitted_params=fitted_params
        )
        results[ticker] = result
    
    valid_count = sum(1 for r in results.values() if r.valid)
    logger.info(f"Computed partial correlations: {valid_count}/{len(results)} valid")
    
    return results
