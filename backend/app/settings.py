"""
Configuration management using pydantic-settings.
All settings are loaded from environment variables.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Database
    database_url: str = "sqlite:////data/app.db"
    
    # Pipeline lock & model storage
    pipeline_lock_file: str = "/data/pipeline.lock"
    model_dir: str = "/data/models"
    
    # News sources
    newsapi_key: Optional[str] = None
    news_query: str = "copper OR copper price OR copper futures OR copper mining"
    news_language: str = "en"
    
    # Price data (yfinance)
    yfinance_symbols: str = "HG=F,DX-Y.NYB,CL=F,FXI"
    lookback_days: int = 365
    
    # Fuzzy deduplication
    fuzzy_dedup_threshold: int = 85
    fuzzy_dedup_window_hours: int = 48
    
    # Sentiment aggregation
    sentiment_tau_hours: float = 12.0
    sentiment_missing_fill: float = 0.0
    
    # API settings
    analysis_ttl_minutes: int = 30
    log_level: str = "INFO"
    
    # Scheduler
    schedule_time: str = "09:00"
    tz: str = "Europe/Istanbul"
    scheduler_enabled: bool = True
    
    @property
    def symbols_list(self) -> list[str]:
        """Parse comma-separated symbols into a list."""
        return [s.strip() for s in self.yfinance_symbols.split(",") if s.strip()]
    
    @property
    def target_symbol(self) -> str:
        """Primary symbol for predictions (first in list)."""
        symbols = self.symbols_list
        return symbols[0] if symbols else "HG=F"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

