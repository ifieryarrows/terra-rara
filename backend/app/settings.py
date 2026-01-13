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
    
    # Price data (yfinance) - Expanded for 2026 market intelligence
    # Core: HG=F (copper), DX-Y.NYB (dollar), CL=F (oil)
    # ETFs: COPX (global miners), COPJ (junior miners)
    # Titans: BHP, FCX, SCCO, RIO
    # Regional: TECK, IVN.TO
    # Juniors: LUN.TO (FIL.TO removed - delisted)
    # China: 2899.HK (Zijin), FXI (China Large-Cap ETF)
    yfinance_symbols: str = "HG=F,DX-Y.NYB,CL=F,FXI,COPX,COPJ,BHP,FCX,SCCO,RIO,TECK,LUN.TO,IVN.TO,2899.HK"
    lookback_days: int = 730  # 2 years for better pattern learning
    
    # Fuzzy deduplication
    fuzzy_dedup_threshold: int = 85
    fuzzy_dedup_window_hours: int = 48
    
    # Sentiment aggregation
    sentiment_tau_hours: float = 12.0
    sentiment_missing_fill: float = 0.0
    
    # API settings
    analysis_ttl_minutes: int = 30
    log_level: str = "INFO"
    
    # Futures vs Spot adjustment factor
    # HG=F (futures) is ~1.5% higher than XCU/USD (spot)
    futures_spot_adjustment: float = 0.985  # Multiply HG=F by this to get XCU/USD
    
    # Scheduler
    schedule_time: str = "02:00"
    tz: str = "Europe/Istanbul"
    scheduler_enabled: bool = True
    
    # OpenRouter AI Commentary
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "xiaomi/mimo-v2-flash:free"
    
    # Twelve Data (Live Price)
    twelvedata_api_key: Optional[str] = None
    
    # LLM Sentiment Analysis (replaces FinBERT)
    llm_sentiment_model: str = "google/gemini-2.0-flash-exp:free"
    
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

