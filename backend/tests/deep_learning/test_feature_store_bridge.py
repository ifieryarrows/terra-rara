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
        lambda _cfg, **_kwargs: pd.DataFrame(),
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
    dates = pd.date_range("2026-04-20", periods=10, freq="B")
    price_df = pd.DataFrame(
        {"close": [6.00, 6.05, 6.10, 6.02, 6.01, 6.08, 6.11, 6.09, 6.15, 6.20]},
        index=dates,
    )
    _patch_feature_store_sources(monkeypatch, price_df)

    master, *_ = build_tft_dataframe(object(), _minimal_tft_cfg())

    # Only row 0 (where causal target = log_close.diff() is NaN) is dropped;
    # the last 5 bars stay present so TimeSeriesDataSet can form the last OOS 5D window.
    assert dates[0] not in master.index
    assert master.index.max() == dates[-1]
    assert dates[-5] in master.index
    assert master["target"].isna().sum() == 0
    assert master["target_5d_log_return"].isna().sum() == 0
    assert master.loc[dates[-6], "target_5d_log_return"] == pytest.approx(float(np.log(6.20 / 6.01)))


def test_build_tft_dataframe_trims_warmup_excludes_proxies_and_includes_ret1(monkeypatch, tmp_path):
    from dataclasses import replace
    from app.features import generate_symbol_features
    from deep_learning.config import get_tft_config

    monkeypatch.setenv("TFT_FEATURE_SNAPSHOT_PATH", str(tmp_path / "should_not_exist.pkl"))
    dates = pd.bdate_range("2025-01-01", periods=170)
    rng = np.random.default_rng(123)

    def _px(base: float, scale: float) -> pd.DataFrame:
        close = pd.Series(base * np.exp(np.cumsum(rng.normal(0.0, scale, len(dates)))), index=dates)
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adj_close": close,
                "volume": np.full(len(dates), 10000.0),
            },
            index=dates,
        )

    hg_px = _px(4.5, 0.012)
    dxy_px = _px(100.0, 0.007)
    bhp_px = _px(55.0, 0.015)
    hg_feats = generate_symbol_features(hg_px, "HG=F", include_raw_ma=False)

    monkeypatch.setattr(feature_store, "load_training_symbols", lambda: ["DX-Y.NYB", "BHP"])
    monkeypatch.setattr(feature_store, "_build_price_features", lambda *_a, **_kw: (hg_px, hg_feats))
    monkeypatch.setattr(
        "app.features.load_multi_price_data",
        lambda *_a, **_kw: {"DX-Y.NYB": dxy_px, "BHP": bhp_px},
    )
    monkeypatch.setattr("deep_learning.data.lme_warehouse.fetch_lme_data", lambda *_a, **_kw: pd.DataFrame())

    cfg = replace(
        get_tft_config(),
        training=replace(get_tft_config().training, lookback_days=260),
    )

    # 1. Pre-selection frame for CV does NOT overwrite TFT_FEATURE_SNAPSHOT_PATH
    raw_master, raw_unk, _raw_kn, _, _ = build_tft_dataframe(
        object(), cfg, apply_mrmr=False, use_snapshot=False
    )
    assert not (tmp_path / "should_not_exist.pkl").exists()
    assert len(raw_master) == len(dates) - 60  # 1 row dropped for target.notna() + 59 warmup rows
    assert raw_master.index.max() == dates[-1]  # last 5 OOS days preserved
    assert "HG=F_ret1" in raw_unk
    for bad in (
        "backwardation_flag",
        "contango_flag",
        "futures_curve_slope",
        "proxy_vol_spike",
        "regime_inventory_tightness",
        "HG=F_price_sma_ratio",
        "HG=F_vol_10",
    ):
        assert bad not in raw_unk

    # 2. Full MRMR selection drops forbidden/proxy/unsigned_vol cols from master too
    monkeypatch.delenv("TFT_FEATURE_SNAPSHOT_PATH", raising=False)
    master, tv_unknown, tv_known, _, _ = build_tft_dataframe(object(), cfg)
    assert tv_known == ["day_of_week"]
    assert "HG=F_ret1" in tv_unknown
    assert "HG=F_ret5" in tv_unknown
    assert "HG=F_close_sma_20_ratio" in tv_unknown
    assert "HG=F_signed_vol_ratio" in tv_unknown
    assert "HG=F_theta_5d_prior" in tv_unknown
    assert "macro_fair_value_zscore" in tv_unknown
    assert "backwardation_flag" not in master.columns
    assert "futures_curve_slope" not in master.columns
    assert "proxy_vol_spike" not in master.columns
    assert "cal_sin_day" not in master.columns
    assert "HG=F_vol_10" not in master.columns
    assert "HG=F_price_sma_ratio" not in master.columns


def test_build_tft_dataframe_loads_real_lme_from_db_and_excludes_raw_tonnage(monkeypatch):
    from dataclasses import replace
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import Base
    from app.features import generate_symbol_features
    from app.models import LMEWarehouseData, PriceBar
    from deep_learning.config import get_tft_config

    monkeypatch.delenv("TFT_FEATURE_SNAPSHOT_PATH", raising=False)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    dates = pd.bdate_range("2025-01-01", periods=170)
    rng = np.random.default_rng(7)
    hg_closes = 4.5 * np.exp(np.cumsum(rng.normal(0.0, 0.012, len(dates))))
    lme_stock = 180000.0 + np.cumsum(rng.normal(-200.0, 1500.0, len(dates)))

    with Session() as session:
        for dt, px, stk in zip(dates, hg_closes, lme_stock):
            dt_utc = dt.to_pydatetime().replace(tzinfo=pd.Timestamp.utcnow().tz)
            session.add(PriceBar(symbol="HG=F", date=dt_utc, close=float(px), adj_close=float(px), volume=10000.0))
            session.add(
                LMEWarehouseData(
                    date=dt_utc,
                    total_stock_tonnes=float(stk),
                    cancelled_warrants_tonnes=float(stk * 0.12),
                    on_warrant_tonnes=float(stk * 0.88),
                    cancelled_ratio=0.12 + float(rng.normal(0.0, 0.01)),
                )
            )
        session.commit()

        monkeypatch.setattr(feature_store, "load_training_symbols", lambda: [])
        cfg = replace(
            get_tft_config(),
            training=replace(get_tft_config().training, lookback_days=600),
            feature_store=replace(get_tft_config().feature_store, mrmr_top_k=0),
        )
        master, tv_unknown, _, _, _ = build_tft_dataframe(session, cfg)
        assert "lme_stock_zscore" in tv_unknown
        assert "lme_cancelled_ratio" in tv_unknown
        assert "lme_stock_total" not in tv_unknown
        assert "lme_stock_change_5d" not in tv_unknown
        assert "lme_depletion_rate" not in tv_unknown
    engine.dispose()


def test_build_tft_dataframe_keeps_latest_bar_for_inference(monkeypatch):
    dates = pd.date_range("2026-04-20", periods=10, freq="B")
    price_df = pd.DataFrame(
        {"close": [6.00, 6.05, 6.10, 6.02, 6.01, 6.08, 6.11, 6.09, 6.15, 6.20]},
        index=dates,
    )
    _patch_feature_store_sources(monkeypatch, price_df)

    master, *_ = build_tft_dataframe(
        object(),
        _minimal_tft_cfg(),
        drop_missing_target=False,
    )

    assert master.index.max() == dates[-1]
    assert master.loc[dates[-1], "target"] == pytest.approx(float(np.log(6.20 / 6.15)))
    assert master.loc[dates[-1], "target_5d_log_return"] == 0.0
    assert master["target"].isna().sum() == 0


def test_build_tft_dataframe_honors_replay_cutoff(monkeypatch):
    dates = pd.date_range("2026-04-20", periods=10, freq="B")
    price_df = pd.DataFrame(
        {"close": [6.00, 6.05, 6.10, 6.02, 6.01, 6.08, 6.11, 6.09, 6.15, 6.20]},
        index=dates,
    )
    _patch_feature_store_sources(monkeypatch, price_df)
    captured = {}
    original_builder = feature_store._build_price_features

    def capture_end_date(session, symbol, start_date, end_date):
        captured["end_date"] = end_date
        return original_builder(session, symbol, start_date, end_date)

    monkeypatch.setattr(feature_store, "_build_price_features", capture_end_date)
    monkeypatch.setenv("TFT_DATA_AS_OF", "2026-04-29T12:34:56Z")

    build_tft_dataframe(object(), _minimal_tft_cfg())

    assert captured["end_date"].isoformat() == "2026-04-29T12:34:56+00:00"


def test_build_tft_dataframe_replays_immutable_feature_snapshot(monkeypatch, tmp_path):
    dates = pd.date_range("2026-04-20", periods=10, freq="B")
    price_df = pd.DataFrame(
        {"close": [6.00, 6.05, 6.10, 6.02, 6.01, 6.08, 6.11, 6.09, 6.15, 6.20]},
        index=dates,
    )
    _patch_feature_store_sources(monkeypatch, price_df)
    snapshot_path = tmp_path / "feature_snapshot.pkl"
    monkeypatch.setenv("TFT_FEATURE_SNAPSHOT_PATH", str(snapshot_path))
    monkeypatch.setenv("TFT_DATA_AS_OF", "2026-04-29T12:34:56Z")

    first = build_tft_dataframe(object(), _minimal_tft_cfg())
    assert snapshot_path.exists()
    assert snapshot_path.with_suffix(".pkl.json").exists()

    def fail_if_live_data_is_loaded(*_args, **_kwargs):
        raise AssertionError("replay unexpectedly rebuilt the live feature store")

    monkeypatch.setattr(feature_store, "_build_price_features", fail_if_live_data_is_loaded)
    second = build_tft_dataframe(object(), _minimal_tft_cfg())

    pd.testing.assert_frame_equal(first[0], second[0])
    assert first[1:] == second[1:]
