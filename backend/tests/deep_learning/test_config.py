"""Tests for TFT-ASRO configuration."""

from deep_learning.config import (
    TFTASROConfig,
    EmbeddingConfig,
    TFTModelConfig,
    ASROConfig,
    TrainingConfig,
    get_tft_config,
)


def test_default_config_creates_valid_instance():
    cfg = get_tft_config()
    assert isinstance(cfg, TFTASROConfig)
    assert cfg.embedding.full_dim == 768
    assert cfg.embedding.pca_dim == 32
    assert cfg.model.hidden_size == 32
    assert len(cfg.model.quantiles) == 7


def test_quantiles_are_sorted_and_include_median():
    cfg = get_tft_config()
    q = cfg.model.quantiles
    assert q == tuple(sorted(q))
    assert 0.50 in q


def test_asro_lambda_defaults():
    cfg = get_tft_config()
    assert 0 < cfg.asro.lambda_vol <= 1.0
    assert 0 < cfg.asro.lambda_quantile <= 1.0


def test_training_splits_sum_to_less_than_one():
    cfg = get_tft_config()
    assert cfg.training.val_ratio + cfg.training.test_ratio < 1.0


def test_model_root_property():
    cfg = get_tft_config()
    root = cfg.model_root
    assert root is not None
    assert "tft" in str(root).lower() or "models" in str(root).lower()
