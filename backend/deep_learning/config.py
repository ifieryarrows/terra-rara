"""
Central configuration for the TFT-ASRO deep learning pipeline.

All hyperparameters, feature dimensions, and training settings live here
so every module draws from a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "ProsusAI/finbert"
    full_dim: int = 768
    pca_dim: int = 32
    max_token_length: int = 512
    batch_size: int = 64
    pca_model_path: str = "models/tft/pca_finbert.joblib"


@dataclass(frozen=True)
class SentimentFeatureConfig:
    momentum_windows: tuple[int, ...] = (5, 10, 30)
    surprise_lookback: int = 30
    surprise_threshold: float = 2.0
    event_types: tuple[str, ...] = (
        "supply_disruption",
        "supply_expansion",
        "demand_increase",
        "demand_decrease",
        "inventory_draw",
        "inventory_build",
        "policy_support",
        "policy_drag",
        "macro_usd_up",
        "macro_usd_down",
        "cost_push",
    )


@dataclass(frozen=True)
class LMEConfig:
    nasdaq_api_key_env: str = "NASDAQ_DATA_LINK_API_KEY"
    quandl_dataset: str = "LME/PR_CU"
    stock_change_windows: tuple[int, ...] = (1, 5, 10, 20)
    depletion_window: int = 20
    futures_symbols: tuple[str, ...] = ("HG=F",)
    futures_months_ahead: tuple[int, ...] = (3, 6, 12)
    max_ffill_days: int = 5


@dataclass(frozen=True)
class TFTModelConfig:
    max_encoder_length: int = 60
    max_prediction_length: int = 5
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 32
    quantiles: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)
    learning_rate: float = 1e-3
    reduce_on_plateau_patience: int = 4
    gradient_clip_val: float = 0.5


@dataclass(frozen=True)
class ASROConfig:
    lambda_vol: float = 0.3
    lambda_quantile: float = 0.2
    risk_free_rate: float = 0.0
    sharpe_window: int = 20


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 100
    early_stopping_patience: int = 10
    batch_size: int = 64
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    lookback_days: int = 730
    seed: int = 42
    num_workers: int = 0
    optuna_n_trials: int = 50
    checkpoint_dir: str = "models/tft/checkpoints"
    best_model_path: str = "models/tft/best_tft_asro.ckpt"


@dataclass(frozen=True)
class FeatureStoreConfig:
    target_symbol: str = "HG=F"
    max_ffill: int = 3
    calendar_features: bool = True
    macro_event_features: bool = True


@dataclass
class TFTASROConfig:
    """Top-level config aggregating all sub-configs."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    sentiment: SentimentFeatureConfig = field(default_factory=SentimentFeatureConfig)
    lme: LMEConfig = field(default_factory=LMEConfig)
    model: TFTModelConfig = field(default_factory=TFTModelConfig)
    asro: ASROConfig = field(default_factory=ASROConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)

    @property
    def model_root(self) -> Path:
        return Path(self.training.checkpoint_dir).parent


def get_tft_config() -> TFTASROConfig:
    """Return the default TFT-ASRO configuration."""
    return TFTASROConfig()
