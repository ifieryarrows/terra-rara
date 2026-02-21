"""
Configuration management using pydantic-settings.
All settings are loaded from environment variables.
"""

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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
    
    # Symbol set configuration
    symbol_set: str = "active"  # active | champion | challenger
    
    # Price data (yfinance) - Dashboard symbols (backward compatible)
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
    futures_spot_adjustment: float = 0.985
    
    # Scheduler (DEPRECATED in API - external scheduler only)
    # These are kept for backward compatibility but scheduler no longer runs in API
    schedule_time: str = "02:00"
    tz: str = "Europe/Istanbul"
    scheduler_enabled: bool = False  # Default to False - scheduler is external now
    
    # Redis Queue (for worker)
    redis_url: str = "redis://localhost:6379/0"
    
    # OpenRouter AI Commentary
    openrouter_api_key: Optional[str] = None
    # Deprecated - kept for backward compatibility
    openrouter_model: str = "openai/gpt-oss-120b:free"
    # New primary config
    openrouter_model_scoring: str = "stepfun/step-3.5-flash:free"
    openrouter_model_commentary: str = "stepfun/step-3.5-flash:free"
    openrouter_rpm: int = 18
    openrouter_max_retries: int = 3
    max_llm_articles_per_run: int = 200
    openrouter_fallback_models: Optional[str] = None
    
    # Twelve Data (Live Price)
    twelvedata_api_key: Optional[str] = None
    
    # LLM Sentiment Analysis
    # Deprecated - kept for backward compatibility
    llm_sentiment_model: str = "openai/gpt-oss-120b:free"
    
    # Pipeline trigger authentication
    pipeline_trigger_secret: Optional[str] = None
    
    # Faz 2: Market cut-off for news aggregation
    # Defines when "today's news" ends for sentiment calculation
    market_timezone: str = "America/New_York"  # NYSE timezone
    market_close_time: str = "16:00"  # 4 PM ET
    cutoff_buffer_minutes: int = 30  # Allow 30 min after close for late news
    
    def _load_symbol_set_file(self, set_name: str) -> Optional[dict]:
        """Load symbol set from JSON file. Returns None on error."""
        try:
            # Path relative to backend root
            backend_root = Path(__file__).resolve().parent.parent
            symbol_file = backend_root / "config" / "symbol_sets" / f"{set_name}.json"
            
            if not symbol_file.exists():
                logger.warning(f"Symbol set file not found: {symbol_file}")
                return None
            
            with open(symbol_file) as f:
                data = json.load(f)
            
            symbols = data.get("symbols", [])
            if not symbols:
                logger.warning(f"Symbol set {set_name} has empty symbols list")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading symbol set {set_name}: {e}")
            return None
    
    def _compute_symbols_hash(self, symbols: list[str]) -> str:
        """Compute deterministic hash of symbol list."""
        canonical = json.dumps(sorted(symbols), sort_keys=True)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"
    
    @property
    def training_symbols(self) -> list[str]:
        """
        Symbols for ML training - loaded from symbol set file.
        Falls back to dashboard symbols on error.
        """
        data = self._load_symbol_set_file(self.symbol_set)
        if data:
            symbols = data.get("symbols", [])
            logger.info(f"Loaded training symbols from file: {self.symbol_set}.json ({len(symbols)}) hash={self._compute_symbols_hash(symbols)}")
            return symbols
        
        # Fallback to env variable
        logger.warning(f"Falling back to YFINANCE_SYMBOLS for training")
        return self.symbols_list
    
    @property
    def training_symbols_source(self) -> str:
        """Source of training symbols for audit."""
        data = self._load_symbol_set_file(self.symbol_set)
        if data:
            return f"file:{self.symbol_set}.json"
        return "env:YFINANCE_SYMBOLS"
    
    @property
    def training_symbols_hash(self) -> str:
        """Hash of training symbols for audit."""
        return self._compute_symbols_hash(self.training_symbols)
    
    @property
    def symbols_list(self) -> list[str]:
        """
        Dashboard symbols - backward compatible with frontend.
        Always uses env variable (14 symbols).
        """
        return [s.strip() for s in self.yfinance_symbols.split(",") if s.strip()]
    
    @property
    def target_symbol(self) -> str:
        """Primary symbol for predictions (first in list)."""
        symbols = self.symbols_list
        return symbols[0] if symbols else "HG=F"

    @staticmethod
    def _first_non_empty(*values: Optional[str]) -> Optional[str]:
        """Return first non-empty string value."""
        for value in values:
            if value and value.strip():
                return value.strip()
        return None

    @property
    def resolved_scoring_model(self) -> str:
        """Preferred scoring model with backward-compatible fallback chain."""
        return (
            self._first_non_empty(
                self.openrouter_model_scoring,
                self.llm_sentiment_model,
                self.openrouter_model,
            )
            or "stepfun/step-3.5-flash:free"
        )

    @property
    def resolved_commentary_model(self) -> str:
        """Preferred commentary model with backward-compatible fallback chain."""
        return (
            self._first_non_empty(
                self.openrouter_model_commentary,
                self.openrouter_model,
                self.llm_sentiment_model,
            )
            or "stepfun/step-3.5-flash:free"
        )

    @property
    def openrouter_fallback_models_list(self) -> list[str]:
        """
        Parse comma-separated fallback models.
        Empty/whitespace items are ignored.
        """
        if not self.openrouter_fallback_models:
            return []
        return [m.strip() for m in self.openrouter_fallback_models.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def mask_api_key(text: str, settings: Settings = None) -> str:
    """
    Mask API keys in text to prevent leaking in logs.
    Replaces known API key patterns with masked versions.
    """
    import re
    
    if settings is None:
        settings = get_settings()
    
    result = text
    
    # Mask known API keys
    keys_to_mask = [
        settings.twelvedata_api_key,
        settings.openrouter_api_key,
        settings.newsapi_key,
        settings.pipeline_trigger_secret,
    ]
    
    for key in keys_to_mask:
        if key and len(key) > 8:
            masked = f"{key[:4]}...{key[-4:]}"
            result = result.replace(key, masked)
    
    # Also mask any apikey= query params
    result = re.sub(r'apikey=[a-zA-Z0-9_-]+', 'apikey=***MASKED***', result)
    
    return result
