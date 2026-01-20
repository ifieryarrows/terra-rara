"""
Configuration management for screener module.

Loads YAML config with validation via Pydantic models.
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import date

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class UniverseFilterConfig(BaseModel):
    """Configuration for universe filtering."""
    
    min_history_days: int = Field(default=730, ge=30)
    min_coverage_pct: float = Field(default=80.0, ge=0, le=100)
    frequency: str = Field(default="W-FRI")


class AnalysisConfig(BaseModel):
    """Configuration for correlation analysis."""
    
    # IS/OOS split dates
    is_start: date = Field(default=date(2018, 1, 1))
    is_end: date = Field(default=date(2023, 12, 31))
    oos_start: date = Field(default=date(2024, 1, 1))
    oos_end: Optional[date] = Field(default=None)  # None = run_date
    
    # Rolling correlation
    rolling_window_weeks: int = Field(default=26, ge=4)
    
    # Lead-lag discovery
    lead_lag_max_periods: int = Field(default=4, ge=0)
    
    # Sign flip detection
    sign_flip_epsilon: float = Field(default=0.05, ge=0, le=1)
    
    # Retention threshold (note: 0.5 is very strict for weekly data; 0.3 is more reasonable)
    min_is_corr_threshold: float = Field(default=0.3, ge=0, le=1)
    
    # Minimum observations for valid correlation
    min_obs: int = Field(default=104, ge=10)
    
    # Partial correlation controls
    controls: list[str] = Field(default=["^GSPC", "UUP"])
    enable_partial_corr: bool = Field(default=True)


class FetcherConfig(BaseModel):
    """Configuration for price fetching."""
    
    calls_per_hour: int = Field(default=1800, ge=100)
    max_retries: int = Field(default=3, ge=1)
    base_retry_delay_seconds: int = Field(default=30, ge=1)
    cache_enabled: bool = Field(default=True)
    price_field: str = Field(default="Close")  # or "Adj Close"


class SeedSourceConfig(BaseModel):
    """Configuration for a single seed source."""
    
    type: str  # seed_csv, seed_json, etf_holdings, macro_peers
    path: Optional[str] = None
    embedded: bool = False


class ScreenerConfig(BaseModel):
    """Root configuration for screener module."""
    
    # Target symbol
    target: str = Field(default="HG=F")
    
    # Seed sources
    seed_sources: list[SeedSourceConfig] = Field(default_factory=list)
    
    # Sub-configs
    filter: UniverseFilterConfig = Field(default_factory=UniverseFilterConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    fetcher: FetcherConfig = Field(default_factory=FetcherConfig)
    
    # Artifact storage
    artifact_base_dir: str = Field(default="artifacts")
    
    # Logging
    log_level: str = Field(default="INFO")
    
    @field_validator("seed_sources", mode="before")
    @classmethod
    def parse_seed_sources(cls, v):
        if isinstance(v, list):
            return [
                SeedSourceConfig(**item) if isinstance(item, dict) else item
                for item in v
            ]
        return v


def load_config(config_path: str | Path) -> ScreenerConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        ScreenerConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raw_config = {}
    
    logger.info(f"Loaded config from {path}")
    
    return ScreenerConfig(**raw_config)


def compute_config_hash(config: ScreenerConfig) -> str:
    """
    Compute deterministic hash of configuration.
    
    Uses JSON serialization with sorted keys for consistency.
    """
    from screener.core.fingerprint import compute_fingerprint
    
    # Convert to dict, handling date objects
    config_dict = config.model_dump(mode="json")
    
    return compute_fingerprint(config_dict)
