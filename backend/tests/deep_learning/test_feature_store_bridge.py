"""Tests for screener-TFT bridge and calendar features."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from deep_learning.data.feature_store import (
    _build_calendar_features,
    build_tft_dataframe,
    load_training_symbols,
    load_screener_selected_symbols,
)
from deep_learning.config import (
    EmbeddingConfig,
    FeatureStoreConfig,
    TFTASROConfig,
    TrainingConfig,
)
from deep_learning.data import feature_store


def test_calendar_features_shape():
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    cal = _build_calendar_features(dates)
    assert len(cal) == 60
    assert "day_of_week" in cal.columns
    assert "cal_sin_day" in cal.columns
    assert "is_friday" in cal.columns
    assert "is_quarter_end" in cal.columns


def test_calendar_sinusoidal_range():
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    cal = _build_calendar_features(dates)
    assert cal["cal_sin_day"].min() >= -1.0
    assert cal["cal_sin_day"].max() <= 1.0
    assert cal["cal_cos_month"].min() >= -1.0


def test_load_training_symbols_returns_list():
    symbols = load_training_symbols()
    assert isinstance(symbols, list)
    assert len(symbols) > 0
    assert all(isinstance(s, str) for s in symbols)


def test_load_training_symbols_includes_mandatory():
    symbols = load_training_symbols()
    assert "DX-Y.NYB" in symbols or "CL=F" in symbols or len(symbols) >= 4


def test_load_screener_selected_symbols_missing_file():
    result = load_screener_selected_symbols(artifacts_dir="nonexistent/path")
    assert result == []


def test_calendar_is_monday_friday_exclusive():
    dates = pd.date_range("2025-01-06", periods=5, freq="B")
    cal = _build_calendar_features(dates)
    monday_row = cal.iloc[0]
    assert monday_row["is_monday"] == 1.0
    assert monday_row["is_friday"] == 0.0
    friday_row = cal.iloc[4]
    assert friday_row["is_friday"] == 1.0
    assert friday_row["is_monday"] == 0.0


def _minimal_tft_cfg() -> TFTASROConfig:
    return TFTASROConfig(
        embedding=EmbeddingConfig(pca_dim=0),
        training=TrainingConfig(lookback_days=30),
        feature_store=FeatureStoreConfig(
            target_symbol="HG=F",
            max_ffill=3,
            mrmr_top_k=0,
        ),
    )


def _patch_feature_store_sources(monkeypatch, price_df: pd.DataFrame) -> None:
    price_features = pd.DataFrame(
        {"close_lag_1": price_df["close"].shift(1).bfill()},
        index=price_df.index,
    )

    monkeypatch.setattr(feature_store, "load_training_symbols", lambda: ["HG=F"])
    monkeypatch.setattr(
        feature_store,
        "_build_price_features",
        lambda *_args, **_kwargs: (price_df, price_features),
    )
    monkeypatch.setattr(
        feature_store,
        "_build_daily_embedding_features",
        lambda _session, index, pca_dim=0: pd.DataFrame(index=index),
    )
    monkeypatch.setattr(
        "app.features.load_sentiment_data",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "deep_learning.data.lme_warehouse.fetch_lme_data",
        lambda _cfg: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "deep_learning.data.lme_warehouse.compute_proxy_lme_features",
        lambda df: pd.DataFrame({"lme_proxy": 0.0}, index=df.index),
    )
    monkeypatch.setattr(
        "deep_learning.data.futures_curve.build_futures_features_from_yfinance",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )


def test_build_tft_dataframe_drops_missing_target_by_default(monkeypatch):
    dates = pd.date_range("2026-04-20", periods=5, freq="B")
    price_df = pd.DataFrame(
        {"close": [6.00, 6.05, 6.10, 6.02, 6.01]},
        index=dates,
    )
    _patch_feature_store_sources(monkeypatch, price_df)

    master, *_ = build_tft_dataframe(object(), _minimal_tft_cfg())

    assert master.index.max() == dates[-2]
    assert dates[-1] not in master.index
    assert master["target"].isna().sum() == 0


def test_build_tft_dataframe_keeps_latest_bar_for_inference(monkeypatch):
    dates = pd.date_range("2026-04-20", periods=5, freq="B")
    price_df = pd.DataFrame(
        {"close": [6.00, 6.05, 6.10, 6.02, 6.01]},
        index=dates,
    )
    _patch_feature_store_sources(monkeypatch, price_df)

    master, *_ = build_tft_dataframe(
        object(),
        _minimal_tft_cfg(),
        drop_missing_target=False,
    )

    assert master.index.max() == dates[-1]
    assert master.loc[dates[-1], "target"] == 0.0
    assert master["target"].isna().sum() == 0
