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
    # pca_dim 32→8: ~375 training samples cannot support 32 embedding
    # dimensions without overfitting.  8 retains the dominant semantic axes
    # while drastically improving the signal-to-noise ratio.
    pca_dim: int = 8
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
    # It.4 known-good default: post-MRMR feature count supports 48 hidden units
    # and produced the first quality-gate passing TFT-ASRO run.
    hidden_size: int = 48
    # attention_head_size 4→2: fewer heads for a small, single-series dataset.
    attention_head_size: int = 2
    # It.4 used 0.30: enough regularisation for the small sample regime without
    # collapsing the output range.
    dropout: float = 0.30
    hidden_continuous_size: int = 16   # was 32; paired reduction with hidden_size
    quantiles: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)
    # It.4 known-good learning rate.
    learning_rate: float = 2e-4
    reduce_on_plateau_patience: int = 4
    # clip 0.5→1.0: tanh-based Sharpe gradients are inherently bounded;
    # relaxing the clip lets the model escape flat regions more aggressively.
    gradient_clip_val: float = 1.0
    # L2 regularisation via AdamW; prevents large weight growth on small datasets.
    weight_decay: float = 5e-5


@dataclass(frozen=True)
class ASROConfig:
    # Total loss = lambda_quantile * calibration + (1-lambda_quantile) * sharpe
    #
    # lambda_quantile is the EXPLICIT weight of the quantile calibration bundle:
    #   calibration = q_loss + lambda_vol * vol_loss
    # w_sharpe = 1 - lambda_quantile (the complementary directional weight)
    #
    # This normalised (sum-to-1) formulation makes both components interpretable
    # and prevents either from silently dominating across loss-magnitude regimes.
    #
    # It.4 split: 25% calibration, 75% directional learning.
    lambda_quantile: float = 0.25
    # lambda_vol is a sub-weight within the calibration bundle only.
    # It controls how much the Q90-Q10 spread tracks 2× actual σ.
    # It.4 known-good volatility calibration sub-weight.
    lambda_vol: float = 0.30
    # MADL (Mean Absolute Directional Loss) weight.
    # Directly rewards correct directional predictions scaled by |actual_return|.
    # Ref: Kisiel & Gorse (2023) "Mean Absolute Directional Loss"
    lambda_madl: float = 0.40
    # Quantile monotonicity guard.  This is a structural constraint, not a
    # calibration/direction trade-off knob: adjacent quantile inversions are
    # mathematically invalid and must be penalised during training.
    lambda_crossing: float = 1.0
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
    # 730→1095: 3 years of history yields ~500 training samples (up from ~313).
    # Improves the feature-to-sample ratio from ~200:313 to ~60:500 (after
    # MRMR pruning).  Walk-forward CV ensures older regime shifts don't
    # pollute validation metrics.
    lookback_days: int = 1095
    seed: int = 42
    num_workers: int = 0
    # 25→15: CI budget fix. 15 trials × 3 folds × 25 epochs ≈ 108 min;
    # final trainer adds ~40-50 min → total ~155 min < 180 min limit.
    optuna_n_trials: int = 15
    # Walk-Forward temporal CV folds for hyperopt (REG-2026-001 P2).
    # Set to 1 to disable CV and fall back to single-split behaviour.
    cv_n_folds: int = 3
    checkpoint_dir: str = ""
    best_model_path: str = ""
    hf_model_repo: str = "ifieryarrows/copper-mind-tft"


@dataclass(frozen=True)
class FeatureStoreConfig:
    target_symbol: str = "HG=F"
    max_ffill: int = 3
    calendar_features: bool = True
    macro_event_features: bool = True
    # MRMR pre-filter: reduce unknown features to top-K by mutual information
    # relevance minus pairwise redundancy.  Set to 0 to disable.
    mrmr_top_k: int = 80


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
