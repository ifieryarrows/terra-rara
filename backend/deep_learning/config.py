"""
Central configuration for the TFT-ASRO deep learning pipeline.

All hyperparameters, feature dimensions, and training settings live here
so every module draws from a single source of truth.

Model paths honour the MODEL_DIR environment variable so they work both
locally (``data/models``) and inside the HF Space container
(``/data/models``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _model_dir() -> str:
    """Resolve the base model directory from env (same as app.settings)."""
    return os.environ.get("MODEL_DIR", "/data/models")


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "ProsusAI/finbert"
    full_dim: int = 768
    pca_dim: int = 32
    max_token_length: int = 512
    batch_size: int = 64
    pca_model_path: str = ""


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
    # hidden_size 64→32: VSN encoder had 3.2M params for only 313 training
    # samples (344 features × hidden_size × hidden_continuous_size).
    # Reducing halves the dominant layer while keeping expressiveness.
    hidden_size: int = 32
    # attention_head_size 4→2: fewer heads for a small, single-series dataset.
    attention_head_size: int = 2
    # dropout 0.1→0.3: 313 samples / ~900K params still demands heavy regularisation.
    dropout: float = 0.3
    hidden_continuous_size: int = 16   # was 32; paired reduction with hidden_size
    quantiles: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)
    # lr 1e-3→3e-4: smaller batches produce noisier gradients; conservative LR
    # reduces the risk of overshooting the narrow-loss landscape.
    learning_rate: float = 3e-4
    reduce_on_plateau_patience: int = 4
    # clip 0.5→1.0: tanh-based Sharpe gradients are inherently bounded;
    # relaxing the clip lets the model escape flat regions more aggressively.
    gradient_clip_val: float = 1.0


@dataclass(frozen=True)
class ASROConfig:
    # lambda_vol 0.3→0.2: with tanh signal the Sharpe term is now on the same
    # scale as actual_std (~0.024); slightly reduce vol weight to give Sharpe
    # more room to drive directional learning.
    lambda_vol: float = 0.2
    # lambda_quantile 0.2→0.3: acts as a regulariser preventing the model from
    # ignoring tail coverage while chasing directional Sharpe.
    lambda_quantile: float = 0.3
    risk_free_rate: float = 0.0
    sharpe_window: int = 20


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 100
    # patience 10→15: with 19 batches/epoch (vs 4 before) each epoch carries
    # more information; give the model more time to converge.
    early_stopping_patience: int = 15
    # batch_size 64→16: 313 samples / 64 = 4 batches/epoch → noisy gradients.
    # 313 / 16 ≈ 19 batches/epoch gives stable, consistent gradient estimates.
    batch_size: int = 16
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    lookback_days: int = 730
    seed: int = 42
    num_workers: int = 0
    optuna_n_trials: int = 50
    checkpoint_dir: str = ""
    best_model_path: str = ""
    hf_model_repo: str = "ifieryarrows/copper-mind-tft"


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
    """
    Return the default TFT-ASRO configuration with paths resolved from
    MODEL_DIR (``/data/models`` on HF Space, configurable locally).
    """
    base = Path(_model_dir()) / "tft"
    return TFTASROConfig(
        embedding=EmbeddingConfig(
            pca_model_path=str(base / "pca_finbert.joblib"),
        ),
        training=TrainingConfig(
            checkpoint_dir=str(base / "checkpoints"),
            best_model_path=str(base / "best_tft_asro.ckpt"),
        ),
    )
