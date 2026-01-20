"""
Pydantic models for Feature Screener output contract.

Defines the structure of screener results with full validation.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class ScreenerMeta(BaseModel):
    """Metadata for screener run."""
    
    generated_at: str = Field(description="ISO 8601 timestamp")
    run_id: str = Field(pattern=r"^scr-[0-9]{8}-[a-z0-9]+$")
    git_commit: Optional[str] = None
    config_hash: str
    lib_versions: dict[str, str]
    universe_version: str
    universe_fingerprint: str
    data_provider_notes: list[str] = Field(default_factory=list)


class TargetInfo(BaseModel):
    """Information about the target symbol (HG=F)."""
    
    ticker: str
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    total_weeks: Optional[int] = None


class AnalysisParameters(BaseModel):
    """Parameters used for correlation analysis."""
    
    is_start: str  # YYYY-MM-DD
    is_end: str    # YYYY-MM-DD
    oos_start: str # YYYY-MM-DD
    oos_end: Optional[str] = None  # YYYY-MM-DD or null (= run_date)
    rolling_window_weeks: int
    lead_lag_max_periods: int
    sign_flip_epsilon: float
    min_is_corr_threshold: float
    min_obs: int
    controls: list[str]
    
    # CONVENTION: lag = +k means candidate LEADS target by k periods
    lead_lag_convention: str = Field(
        default="positive_lag_means_candidate_leads",
        description="lag=+k => candidate.shift(k) aligns with target"
    )


class PeriodMetrics(BaseModel):
    """Correlation metrics for a single period (IS or OOS)."""
    
    pearson: Optional[float] = None
    spearman: Optional[float] = None
    rolling_corr_mean: Optional[float] = None
    rolling_corr_std: Optional[float] = None
    sign_flip_count: Optional[int] = None
    
    # Lead-lag (IS: discovery, OOS: frozen)
    best_lead_lag: Optional[int] = None
    best_lead_lag_corr: Optional[float] = None
    frozen_lag: Optional[int] = None  # OOS only
    lag_corr_at_frozen: Optional[float] = None  # OOS only
    
    # Partial correlation
    partial_corr: Optional[float] = None
    
    # Retention check
    passed_retention: Optional[bool] = None
    
    # Additional info
    n_obs: Optional[int] = None
    first_date: Optional[str] = None
    last_date: Optional[str] = None


class CandidateDecision(BaseModel):
    """Decision information for a candidate symbol."""
    
    rank: Optional[int] = None
    score_composite: Optional[float] = None
    include_in_model: bool = False
    notes: list[str] = Field(default_factory=list)


class ScreenerCandidate(BaseModel):
    """A fully analyzed candidate symbol."""
    
    ticker: str
    category: Optional[str] = None
    pairwise_obs: int
    
    # Period-specific metrics
    is_metrics: PeriodMetrics = Field(alias="is")
    oos_metrics: Optional[PeriodMetrics] = Field(default=None, alias="oos")
    
    # Overrides that were applied
    overrides_applied: list[str] = Field(default_factory=list)
    
    # Final decision
    decision: CandidateDecision
    
    class Config:
        populate_by_name = True


class ExcludedCandidate(BaseModel):
    """A candidate that was excluded from analysis."""
    
    ticker: str
    reason: str
    is_pearson: Optional[float] = None
    details: Optional[str] = None


class ArtifactReference(BaseModel):
    """Reference to a stored artifact file with full audit info."""
    
    name: str
    sha256: str
    path: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: Optional[str] = None
    content_type: Optional[str] = None  # e.g., "application/json", "application/parquet"


class ScreenerOutput(BaseModel):
    """
    Complete output contract for Feature Screener.
    
    This is the schema for screener_output.json files.
    
    FINGERPRINT POLICY:
        - content_fingerprint: Hash of analysis content only (deterministic).
          Same inputs + same config = same content_fingerprint.
          Excludes: meta.run_id, meta.generated_at, artifacts[].created_at/size_bytes.
          
        - output_fingerprint: Hash of full output envelope.
          Changes with each run (includes run_id, timestamps).
    """
    
    meta: ScreenerMeta
    target: TargetInfo
    analysis_parameters: AnalysisParameters
    candidates: list[ScreenerCandidate]
    excluded: list[ExcludedCandidate] = Field(default_factory=list)
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    
    # Dual fingerprints for audit + determinism
    content_fingerprint: str = Field(
        description="Hash of content only (deterministic)"
    )
    output_fingerprint: str = Field(
        description="Hash of full output (includes meta)"
    )
    
    # Deprecated: use content_fingerprint instead
    fingerprint: Optional[str] = Field(default=None, deprecated=True)
    
    def get_top_candidates(self, n: int = 10) -> list[ScreenerCandidate]:
        """Get top N candidates by composite score."""
        ranked = [c for c in self.candidates if c.decision.rank is not None]
        ranked.sort(key=lambda x: x.decision.rank or 999)
        return ranked[:n]
    
    def get_included_for_model(self) -> list[ScreenerCandidate]:
        """Get candidates marked for model inclusion."""
        return [c for c in self.candidates if c.decision.include_in_model]
