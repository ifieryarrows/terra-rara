"""Tests for TFT-ASRO configuration."""

from deep_learning.config import (
    TFTASROConfig,
    EmbeddingConfig,
    TFTModelConfig,
    ASROConfig,
    TrainingConfig,
    get_tft_config,
)
from deep_learning.training.trainer import KNOWN_GOOD_CONFIG, _overlay_training_config


def test_default_config_creates_valid_instance():
    cfg = get_tft_config()
    assert isinstance(cfg, TFTASROConfig)
    assert cfg.embedding.full_dim == 768
    assert cfg.embedding.pca_dim == 8
    assert cfg.model.hidden_size == 48
    assert len(cfg.model.quantiles) == 7


def test_quantiles_are_sorted_and_include_median():
    cfg = get_tft_config()
    q = cfg.model.quantiles
    assert q == tuple(sorted(q))
    assert 0.50 in q


def test_asro_lambda_defaults():
    cfg = get_tft_config()
    assert cfg.asro.lambda_quantile == 0.25
    assert cfg.asro.lambda_vol == 0.30
    assert cfg.asro.lambda_madl == 0.40
    assert 0 < cfg.asro.lambda_vol <= 1.0
    assert 0 < cfg.asro.lambda_quantile <= 1.0
    assert 0 < cfg.asro.lambda_madl <= 1.0
    assert cfg.asro.lambda_crossing > 0


def test_model_defaults_match_known_good_config():
    cfg = get_tft_config()
    assert cfg.model.dropout == 0.30
    assert cfg.model.learning_rate == 2e-4
    assert cfg.model.weight_decay == 5e-5


def test_known_good_overlay_includes_batch_size_fallback():
    cfg = get_tft_config()
    updated = _overlay_training_config(cfg, KNOWN_GOOD_CONFIG)

    assert updated.model.hidden_size == 48
    assert updated.asro.lambda_quantile == 0.25
    assert updated.asro.lambda_madl == 0.40
    assert updated.weekly_loss.lambda_weekly_quantile == 0.70
    assert updated.weekly_loss.lambda_t1_quantile == 0.20
    assert updated.weekly_loss.lambda_dispersion == 0.35
    assert updated.weekly_loss.lambda_magnitude == 0.55
    assert updated.weekly_loss.lambda_naive == 0.40
    assert updated.weekly_loss.lambda_bias == 0.19
    assert updated.weekly_loss.lambda_directional == 0.06
    assert updated.weekly_loss.lambda_saturation == 0.25
    assert updated.weekly_loss.lambda_positive_rate == 0.20
    assert updated.weekly_loss.lambda_interval == 0.15
    assert updated.training.batch_size == 32


def test_weekly_loss_defaults_use_structural_monotonic_recovery_terms():
    cfg = get_tft_config()
    assert cfg.weekly_loss.lambda_weekly_quantile == 0.70
    assert cfg.weekly_loss.lambda_t1_quantile == 0.20
    assert cfg.weekly_loss.lambda_dispersion == 0.35
    assert cfg.weekly_loss.lambda_magnitude == 0.55
    assert cfg.weekly_loss.lambda_naive == 0.40
    assert cfg.weekly_loss.lambda_bias == 0.19
    assert cfg.weekly_loss.lambda_directional == 0.06
    assert cfg.weekly_loss.lambda_saturation == 0.25
    assert cfg.weekly_loss.lambda_positive_rate == 0.20
    assert cfg.weekly_loss.lambda_interval == 0.15
    assert cfg.weekly_loss.weekly_median_cap_abs_median_multiple == 2.0
    assert cfg.weekly_loss.weekly_median_cap_mean_abs_multiple == 1.6
    assert cfg.weekly_loss.weekly_median_cap_std_multiple == 1.2
    assert not hasattr(cfg.weekly_loss, "lambda_crossing")
    assert not hasattr(cfg.weekly_loss, "lambda_width")
    assert not hasattr(cfg.weekly_loss, "lambda_tail_width")
    assert not hasattr(cfg.weekly_loss, "lambda_sanity")
    assert not hasattr(cfg.weekly_loss, "lambda_vol")


def test_lookback_days_is_3_years():
    cfg = get_tft_config()
    assert cfg.training.lookback_days == 1095


def test_mrmr_top_k_positive():
    cfg = get_tft_config()
    assert cfg.feature_store.mrmr_top_k >= 0


def test_weight_decay_positive():
    cfg = get_tft_config()
    assert cfg.model.weight_decay > 0


def test_training_splits_sum_to_less_than_one():
    cfg = get_tft_config()
    assert cfg.training.val_ratio + cfg.training.test_ratio < 1.0


def test_model_root_property():
    cfg = get_tft_config()
    root = cfg.model_root
    assert root is not None
    assert "tft" in str(root).lower() or "models" in str(root).lower()
