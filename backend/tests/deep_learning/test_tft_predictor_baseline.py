"""Regression tests for TFT inference baseline freshness."""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import PriceBar
from deep_learning.config import (
    FeatureStoreConfig,
    TFTASROConfig,
    TFTModelConfig,
    TrainingConfig,
)
from deep_learning.inference.predictor import TFTPredictor


class _FakeTimeSeriesDataSet:
    def __init__(self, data, **_kwargs):
        self.data = data

    def to_dataloader(self, **_kwargs):
        return ["fake-batch"]


class _FakeModel:
    def predict(self, _dl, mode=None):
        assert mode == "quantiles"
        return np.array(
            [[[-0.01, -0.005, -0.002, 0.01, 0.012, 0.015, 0.02]]],
            dtype=float,
        )


@pytest.fixture
def price_session():
    engine = create_engine("sqlite:///:memory:")
    PriceBar.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        session.add_all(
            [
                PriceBar(
                    symbol="HG=F",
                    date=datetime(2026, 4, 24, 4, 0, tzinfo=timezone.utc),
                    close=6.0235,
                ),
                PriceBar(
                    symbol="HG=F",
                    date=datetime(2026, 4, 27, 4, 0, tzinfo=timezone.utc),
                    close=6.0180,
                ),
            ]
        )
        session.commit()
        yield session
    finally:
        session.close()
        engine.dispose()


def test_predict_uses_latest_price_bar_for_reference_date(monkeypatch, price_session):
    fake_pf = types.ModuleType("pytorch_forecasting")
    fake_pf.TimeSeriesDataSet = _FakeTimeSeriesDataSet
    monkeypatch.setitem(sys.modules, "pytorch_forecasting", fake_pf)

    import deep_learning.data.feature_store as feature_store
    import deep_learning.data.dataset as dataset_mod
    monkeypatch.setattr(dataset_mod, "_identity_target_normalizer", lambda: None)

    def fake_build_tft_dataframe(_session, _cfg, *, drop_missing_target=True):
        assert drop_missing_target is False
        index = pd.to_datetime(["2026-04-22", "2026-04-23", "2026-04-24"])
        master = pd.DataFrame(
            {
                "feat": [1.0, 1.1, 1.2],
                "target": [0.001, -0.002, 0.0],
                "target_1d_log_return": [0.001, -0.002, 0.0],
                "target_5d_log_return": [0.01, 0.02, 0.0],
                "realized_vol_20d": [0.01, 0.01, 0.0],
                "material_move_5d": [0.0, 1.0, 0.0],
                "group_id": ["copper", "copper", "copper"],
                "time_idx": [0, 1, 2],
            },
            index=index,
        )
        return master, ["feat"], [], ["target"], 6.0235

    monkeypatch.setattr(
        feature_store,
        "build_tft_dataframe",
        fake_build_tft_dataframe,
    )

    cfg = TFTASROConfig(
        model=TFTModelConfig(max_encoder_length=2, max_prediction_length=1),
        training=TrainingConfig(best_model_path="unused.ckpt"),
        feature_store=FeatureStoreConfig(target_symbol="HG=F", mrmr_top_k=0),
    )
    predictor = TFTPredictor(cfg=cfg)
    predictor._model = _FakeModel()
    monkeypatch.setattr(
        predictor,
        "_check_price_freshness",
        lambda _session, _symbol: (1, False),
    )

    result = predictor.predict(price_session, "HG=F")

    assert "error" not in result
    assert result["reference_price"] == pytest.approx(6.0180)
    assert result["reference_price_date"] == "2026-04-27"
    assert result["predicted_price_median"] == pytest.approx(6.0180 * np.exp(0.01))
    assert result["return_basis"] == "daily_log_return_path"


def test_incompatible_checkpoint_metadata_returns_degraded_payload(tmp_path):
    ckpt = tmp_path / "old.ckpt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    predictor = TFTPredictor(checkpoint_path=str(ckpt))
    result = None
    try:
        _ = predictor.model
    except Exception as exc:
        result = predictor._degraded_retrain_required(str(exc))
    assert result is not None
    assert result["model_state"] == "retrain_required"
    assert result["quality_state"] == "degraded"
