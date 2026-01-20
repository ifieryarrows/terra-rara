"""
Pydantic models for Universe Builder output contract.

Defines the structure of universe.json with full validation.
"""

from datetime import date, datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class LibVersions(BaseModel):
    """Library version information."""
    python: str
    yfinance: str = "not_installed"
    pandas: str = "not_installed"
    numpy: str = "not_installed"
    pydantic: str = "not_installed"
    
    class Config:
        extra = "allow"


class UniverseMeta(BaseModel):
    """Metadata for universe generation run."""
    
    generated_at: str = Field(description="ISO 8601 timestamp")
    run_id: str = Field(pattern=r"^univ-[0-9]{8}-[a-z0-9]+$")
    git_commit: Optional[str] = None
    config_hash: str = Field(pattern=r"^sha256:[a-f0-9]+$")
    lib_versions: LibVersions
    data_provider_notes: list[str] = Field(default_factory=list)


class SourceInfo(BaseModel):
    """Information about a seed source."""
    
    type: Literal["seed_csv", "seed_json", "etf_holdings", "macro_peers"]
    path: Optional[str] = None
    sha256: Optional[str] = None
    embedded: bool = False


class FilterParameters(BaseModel):
    """Parameters used for universe filtering."""
    
    min_history_days: int = Field(ge=1)
    min_coverage_pct: float = Field(ge=0, le=100)
    frequency: str


class UniverseSymbol(BaseModel):
    """A single symbol in the universe."""
    
    ticker: str
    canonical_ticker: str
    category: Optional[str] = None
    first_date: Optional[str] = None  # YYYY-MM-DD
    last_date: Optional[str] = None   # YYYY-MM-DD
    total_weeks: Optional[int] = None
    coverage_pct: Optional[float] = None
    status: Literal["included", "excluded"]
    exclusion_reason: Optional[str] = None
    
    # PROVENANCE: Full list of sources where this ticker appeared
    # Format: ["seed:smoke_test.csv", "etf:copx_holdings.json", "macro_peers"]
    sources: list[str] = Field(default_factory=list)
    
    # Deprecated: use 'sources' instead
    source_tag: Optional[str] = None


class UniverseSummary(BaseModel):
    """Summary statistics for universe."""
    
    total_candidates: int
    included: int
    excluded: int


class UniverseOutput(BaseModel):
    """
    Complete output contract for Universe Builder.
    
    This is the schema for universe.json files.
    
    FINGERPRINT POLICY:
        - content_fingerprint: Hash of analysis content only (deterministic).
          Same inputs + same config = same content_fingerprint.
          Excludes: meta.run_id, meta.generated_at, envelope fields.
          
        - output_fingerprint: Hash of full output envelope.
          Changes with each run (includes run_id, timestamps).
    """
    
    meta: UniverseMeta
    sources: list[SourceInfo]
    filter_parameters: FilterParameters
    universe: list[UniverseSymbol]
    summary: UniverseSummary
    
    # Dual fingerprints for audit + determinism
    content_fingerprint: str = Field(
        pattern=r"^sha256:[a-f0-9]+$",
        description="Hash of content only (deterministic)"
    )
    output_fingerprint: str = Field(
        pattern=r"^sha256:[a-f0-9]+$",
        description="Hash of full output (includes meta)"
    )
    
    # Deprecated: use content_fingerprint instead
    fingerprint: Optional[str] = Field(default=None, deprecated=True)
    
    def get_included_tickers(self) -> list[str]:
        """Get list of included ticker symbols."""
        return [s.canonical_ticker for s in self.universe if s.status == "included"]
    
    def get_by_category(self, category: str) -> list[UniverseSymbol]:
        """Get symbols by category."""
        return [s for s in self.universe if s.category == category and s.status == "included"]
