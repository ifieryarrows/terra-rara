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

from deep_learning.contract import TARGET_RETURN_TYPE


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
    max_encoder_length: int = 10
    max_prediction_length: int = 5
    # Compact capacity with differential head LR prevents gradient ping-pong on
    # raw daily log-return targets (~17k params vs 510k previously).
    hidden_size: int = 12
    attention_head_size: int = 2
    dropout: float = 0.35
    hidden_continuous_size: int = 6
    quantiles: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)
    learning_rate: float = 5e-4
    output_head_lr_mult: float = 0.05
    reduce_on_plateau_patience: int = 4
    gradient_clip_val: float = 0.5
    # L2 regularisation via AdamW; prevents weight growth on small financial series.
    weight_decay: float = 1e-2


@dataclass(frozen=True)
class ForecastContractConfig:
    primary_horizon_days: int = 5
    auxiliary_horizon_days: int = 1
    target_return_type: str = TARGET_RETURN_TYPE
    primary_target_col: str = "target_5d_log_return"
    auxiliary_target_col: str = "target_1d_log_return"
    model_daily_target_col: str = "target"
    material_move_vol_multiple: float = 0.75
    weekly_direction_threshold: float = 0.005
    t1_direction_threshold: float = 0.002


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
class WeeklyLossConfig:
    # Proper-scoring core: scale-normalized weekly quantile pinball + auxiliary
    # T+1 quantile pinball + gentle bias guard (directional surrogate default 0.0
    # avoids cross-regime threshold flip instability on compact samples).
    # Legacy heuristic batch penalties default to 0.0 (fields kept for backward compat).
    lambda_weekly_quantile: float = 1.00
    lambda_t1_quantile: float = 0.25
    lambda_t1_directional: float = 0.0
    lambda_dispersion: float = 0.0
    lambda_magnitude: float = 0.0
    lambda_naive: float = 0.0
    lambda_bias: float = 0.0
    lambda_drift_prior: float = 1.0
    lambda_directional: float = 0.0
    lambda_saturation: float = 0.0
    lambda_positive_rate: float = 0.0
    lambda_interval: float = 0.0
    # Safety anomaly ceiling only — never clip realistic forecasts during normal operation.
    weekly_median_cap_abs_median_multiple: float = 5.0
    weekly_median_cap_mean_abs_multiple: float = 4.5
    weekly_median_cap_std_multiple: float = 3.5
    weekly_median_cap: Optional[float] = None


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 60
    early_stopping_patience: int = 10
    batch_size: int = 32
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    lookback_days: int = 1825
    seed: int = 42
    num_workers: int = 0
    optuna_n_trials: int = 30
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
    mrmr_top_k: int = 12


@dataclass
class TFTASROConfig:
    """Top-level config aggregating all sub-configs."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    sentiment: SentimentFeatureConfig = field(default_factory=SentimentFeatureConfig)
    lme: LMEConfig = field(default_factory=LMEConfig)
    model: TFTModelConfig = field(default_factory=TFTModelConfig)
    forecast: ForecastContractConfig = field(default_factory=ForecastContractConfig)
    asro: ASROConfig = field(default_factory=ASROConfig)
    weekly_loss: WeeklyLossConfig = field(default_factory=WeeklyLossConfig)
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
