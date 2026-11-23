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
    last_parameters = None
    last_data = None

    def __init__(self, data, **_kwargs):
        self.data = data

    @classmethod
    def from_parameters(cls, parameters, data, **_kwargs):
        cls.last_parameters = parameters
        cls.last_data = data.copy()
        return cls(data)

    def to_dataloader(self, **_kwargs):
        return ["fake-batch"]


class _FakeModel:
    dataset_parameters = {
        "time_varying_unknown_reals": ["feat"],
        "time_varying_known_reals": [],
        "max_encoder_length": 2,
        "max_prediction_length": 1,
    }

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
    def fake_build_tft_dataframe(_session, _cfg, *, drop_missing_target=True):
        assert drop_missing_target is False
        assert _cfg.feature_store.mrmr_top_k == 0
        index = pd.to_datetime(["2026-04-22", "2026-04-23", "2026-04-24"])
        master = pd.DataFrame(
            {
                "feat": [1.0, 1.1, 1.2],
                "rolling_window_pick": [2.0, 2.1, 2.2],
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
        # Simulate rolling-window MRMR choosing a different feature than the
        # checkpoint. Inference must retain the fitted checkpoint order.
        return master, ["rolling_window_pick"], [], ["target"], 6.0235

    monkeypatch.setattr(
        feature_store,
        "build_tft_dataframe",
        fake_build_tft_dataframe,
    )

    cfg = TFTASROConfig(
        # Deliberately differ from the fitted checkpoint contract. A config
        # change after training must not reshape live inference tensors.
        model=TFTModelConfig(max_encoder_length=60, max_prediction_length=5),
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
    assert result["daily_forecasts"][0]["forecast_date"] == "2026-04-28"
    assert result["model_info"]["feature_contract_source"] == "checkpoint_dataset_parameters"
    assert result["model_info"]["encoder_length"] == 2
    assert result["model_info"]["prediction_length"] == 1
    assert _FakeTimeSeriesDataSet.last_parameters == _FakeModel.dataset_parameters
    assert "feat" in _FakeTimeSeriesDataSet.last_data.columns


def test_predict_returns_degraded_payload_when_checkpoint_feature_is_missing(
    monkeypatch,
    price_session,
):
    fake_pf = types.ModuleType("pytorch_forecasting")
    fake_pf.TimeSeriesDataSet = _FakeTimeSeriesDataSet
    monkeypatch.setitem(sys.modules, "pytorch_forecasting", fake_pf)

    import deep_learning.data.feature_store as feature_store

    index = pd.to_datetime(["2026-04-22", "2026-04-23", "2026-04-24"])
    master = pd.DataFrame(
        {
            "different_feat": [1.0, 1.1, 1.2],
            "target": [0.001, -0.002, 0.0],
            "group_id": ["copper", "copper", "copper"],
            "time_idx": [0, 1, 2],
        },
        index=index,
    )
    monkeypatch.setattr(
        feature_store,
        "build_tft_dataframe",
        lambda *_args, **_kwargs: (
            master,
            ["different_feat"],
            [],
            ["target"],
            6.0235,
        ),
    )

    cfg = TFTASROConfig(
        model=TFTModelConfig(max_encoder_length=2, max_prediction_length=1),
        training=TrainingConfig(best_model_path="unused.ckpt"),
        feature_store=FeatureStoreConfig(target_symbol="HG=F", mrmr_top_k=80),
    )
    predictor = TFTPredictor(cfg=cfg)
    predictor._model = _FakeModel()
    monkeypatch.setattr(
        predictor,
        "_check_price_freshness",
        lambda _session, _symbol: (1, False),
    )

    result = predictor.predict(price_session, "HG=F")

    assert result["model_state"] == "retrain_required"
    assert result["quality_state"] == "degraded"
    assert result["weekly_forecast"] is None
    assert "checkpoint features are missing (feat)" in result["message"]


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
    assert result["is_forecast_healthy"] is False
    assert result["primary_forecast_return"] is None
    assert result["reference_price_date"] is None


def test_predict_staleness_preserves_healthy_forecast_without_lazy_ingest(monkeypatch, price_session):
    fake_pf = types.ModuleType("pytorch_forecasting")
    fake_pf.TimeSeriesDataSet = _FakeTimeSeriesDataSet
    monkeypatch.setitem(sys.modules, "pytorch_forecasting", fake_pf)

    def fake_build_tft_dataframe(*_args, **_kwargs):
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
        "deep_learning.data.feature_store.build_tft_dataframe",
        fake_build_tft_dataframe,
    )

    cfg = TFTASROConfig(
        model=TFTModelConfig(max_encoder_length=60, max_prediction_length=5),
        training=TrainingConfig(best_model_path="unused.ckpt"),
        feature_store=FeatureStoreConfig(target_symbol="HG=F", mrmr_top_k=0),
    )
    predictor = TFTPredictor(cfg=cfg)
    predictor._model = _FakeModel()
    monkeypatch.setattr(
        predictor,
        "_check_price_freshness",
        lambda _session, _symbol: (5, True),
    )

    result = predictor.predict(price_session, "HG=F")

    assert "error" not in result
    assert result["is_forecast_healthy"] is True
    assert result["is_stale"] is True
    assert result["baseline_staleness_days"] == 5
    assert result["lazy_ingest_triggered"] is False
    assert not hasattr(predictor, "_trigger_lazy_ingest")


def test_predict_loads_and_applies_weekly_magnitude_scale_and_cap(monkeypatch, price_session, tmp_path):
    import json
    from deep_learning.contract import FORECAST_CONTRACT_VERSION, RETURN_SPACE, TARGET_RETURN_TYPE

    fake_pf = types.ModuleType("pytorch_forecasting")
    fake_pf.TimeSeriesDataSet = _FakeTimeSeriesDataSet
    monkeypatch.setitem(sys.modules, "pytorch_forecasting", fake_pf)

    class _WeeklyFakeModel:
        dataset_parameters = {
            "time_varying_unknown_reals": ["feat"],
            "time_varying_known_reals": [],
            "max_encoder_length": 2,
            "max_prediction_length": 5,
        }

        def predict(self, _dl, mode=None):
            assert mode == "quantiles"
            # Raw weekly median = 5 * 0.02 = 0.10
            base = np.full((1, 5, 7), 0.02, dtype=float)
            offsets = np.array([-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015])
            return base + offsets.reshape(1, 1, -1)

    import deep_learning.data.feature_store as feature_store

    def fake_build_tft_dataframe(_session, _cfg, *, drop_missing_target=True):
        index = pd.to_datetime(["2026-04-22", "2026-04-23", "2026-04-24"])
        master = pd.DataFrame(
            {
                "feat": [1.0, 1.1, 1.2],
                "target": [0.001, -0.002, 0.0],
                "target_1d_log_return": [0.001, -0.002, 0.0],
                "target_5d_log_return": [0.01, 0.02, 0.0],
                "realized_vol_20d": [0.01, 0.01, 0.01],
                "material_move_5d": [0.0, 1.0, 0.0],
                "group_id": ["copper", "copper", "copper"],
                "time_idx": [0, 1, 2],
            },
            index=index,
        )
        return master, ["feat"], [], ["target"], 6.0180

    monkeypatch.setattr(feature_store, "build_tft_dataframe", fake_build_tft_dataframe)

    ckpt_path = tmp_path / "best_tft_asro.ckpt"
    ckpt_path.write_bytes(b"stub")
    meta_path = tmp_path / "tft_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "forecast_contract_version": FORECAST_CONTRACT_VERSION,
                "target_return_type": TARGET_RETURN_TYPE,
                "primary_horizon_days": 5,
                "return_space": RETURN_SPACE,
                "config": {"weekly_median_cap": 0.030},
                "direction_calibration": {
                    "direction_sign_multiplier": 1,
                    "weekly_sign_threshold": 0.0,
                    "weekly_magnitude_scale": 0.50,
                },
                "magnitude_calibration": {"scale": 0.50},
                "interval_calibration": {"weekly_interval_scale": 1.0},
            }
        ),
        encoding="utf-8",
    )

    import deep_learning.models.hub as hub_module

    monkeypatch.setattr(hub_module, "validate_artifact_manifest", lambda _dir: True)

    cfg = TFTASROConfig(
        model=TFTModelConfig(max_encoder_length=2, max_prediction_length=5),
        training=TrainingConfig(best_model_path=str(ckpt_path)),
        feature_store=FeatureStoreConfig(target_symbol="HG=F", mrmr_top_k=0),
    )
    predictor = TFTPredictor(checkpoint_path=str(ckpt_path), cfg=cfg)
    predictor._validate_checkpoint_contract()
    predictor._model = _WeeklyFakeModel()
    monkeypatch.setattr(
        predictor,
        "_check_price_freshness",
        lambda _session, _symbol: (1, False),
    )

    assert predictor._weekly_magnitude_scale == pytest.approx(0.50)
    assert predictor._weekly_median_cap == pytest.approx(0.030)

    result = predictor.predict(price_session, "HG=F")

    assert "error" not in result
    # 0.10 * 0.50 = 0.05 -> then capped at weekly_median_cap=0.030 log return!
    expected_weekly_simple = float(np.exp(0.030) - 1.0)
    assert result["weekly_log_return"] == pytest.approx(0.030, rel=1e-4)
    assert result["weekly_return"] == pytest.approx(expected_weekly_simple, rel=1e-4)
    assert result["model_info"]["weekly_magnitude_scale"] == pytest.approx(0.50)
    assert result["model_info"]["weekly_median_cap"] == pytest.approx(0.030)


def test_predict_ensemble_averages_in_ordered_quantile_space_and_fails_on_missing_seed(
    monkeypatch, price_session, tmp_path
):
    import json
    from deep_learning.contract import FORECAST_CONTRACT_VERSION, RETURN_SPACE, TARGET_RETURN_TYPE
    from deep_learning.training.metrics import monotonic_quantiles_np

    fake_pf = types.ModuleType("pytorch_forecasting")
    fake_pf.TimeSeriesDataSet = _FakeTimeSeriesDataSet
    monkeypatch.setitem(sys.modules, "pytorch_forecasting", fake_pf)

    # Distinct raw logit outer channels so nonlinear softplus(mean(logits)) != mean(ordered_quantiles)
    logits_1 = np.tile(np.array([[-4.0, -3.0, -2.0, 0.002, -2.0, -3.0, -4.0]]), (1, 5, 1))
    logits_2 = np.tile(np.array([[2.0, 1.5, 1.0, 0.004, 1.0, 1.5, 2.0]]), (1, 5, 1))

    class _SeedModel:
        dataset_parameters = {
            "time_varying_unknown_reals": ["feat"],
            "time_varying_known_reals": [],
            "max_encoder_length": 2,
            "max_prediction_length": 5,
        }

        def __init__(self, out):
            self._out = out

        def predict(self, _dl, mode=None):
            assert mode == "quantiles"
            return self._out

    import deep_learning.data.feature_store as feature_store
    import deep_learning.models.hub as hub_module
    import deep_learning.models.tft_copper as tft_copper

    def fake_build_tft_dataframe(_session, _cfg, *, drop_missing_target=True):
        index = pd.to_datetime(["2026-04-22", "2026-04-23", "2026-04-24"])
        master = pd.DataFrame(
            {
                "feat": [1.0, 1.1, 1.2],
                "target": [0.001, -0.002, 0.0],
                "target_1d_log_return": [0.001, -0.002, 0.0],
                "target_5d_log_return": [0.01, 0.02, 0.0],
                "realized_vol_20d": [0.01, 0.01, 0.01],
                "material_move_5d": [0.0, 1.0, 0.0],
                "group_id": ["copper", "copper", "copper"],
                "time_idx": [0, 1, 2],
            },
            index=index,
        )
        return master, ["feat"], [], ["target"], 6.0180

    monkeypatch.setattr(feature_store, "build_tft_dataframe", fake_build_tft_dataframe)
    monkeypatch.setattr(hub_module, "validate_artifact_manifest", lambda _dir: True)

    ckpt_1 = tmp_path / "best_tft_asro.ckpt"
    ckpt_2 = tmp_path / "best_tft_asro_seed1.ckpt"
    ckpt_1.write_bytes(b"seed1")
    ckpt_2.write_bytes(b"seed2")
    (tmp_path / "tft_metadata.json").write_text(
        json.dumps(
            {
                "forecast_contract_version": FORECAST_CONTRACT_VERSION,
                "target_return_type": TARGET_RETURN_TYPE,
                "primary_horizon_days": 5,
                "return_space": RETURN_SPACE,
                "monotonic_quantile_transform": True,
                "ensemble_checkpoints": [ckpt_1.name, ckpt_2.name],
                "test_metrics": {"ensemble_size": 2},
            }
        ),
        encoding="utf-8",
    )

    m1 = _SeedModel(logits_1)
    m2 = _SeedModel(logits_2)
    monkeypatch.setattr(tft_copper, "load_tft_model", lambda p: m2 if "seed1" in str(p) else m1)

    cfg = TFTASROConfig(
        model=TFTModelConfig(max_encoder_length=2, max_prediction_length=5),
        training=TrainingConfig(best_model_path=str(ckpt_1)),
        feature_store=FeatureStoreConfig(target_symbol="HG=F", mrmr_top_k=0),
    )
    predictor = TFTPredictor(checkpoint_path=str(ckpt_1), cfg=cfg)
    predictor._validate_checkpoint_contract()
    predictor._model = m1
    monkeypatch.setattr(predictor, "_check_price_freshness", lambda _s, _sym: (1, False))

    captured = {}
    orig_fmt = tft_copper.format_prediction

    def spy_format(pred_for_format, **kwargs):
        captured["pred_for_format"] = pred_for_format.copy()
        return orig_fmt(pred_for_format, **kwargs)

    monkeypatch.setattr(tft_copper, "format_prediction", spy_format)

    res = predictor.predict(price_session, "HG=F")
    assert res["model_info"]["ensemble_size"] == 2
    expected_ordered_mean = 0.5 * (
        monotonic_quantiles_np(logits_1, median_idx=3)[0]
        + monotonic_quantiles_np(logits_2, median_idx=3)[0]
    )
    np.testing.assert_allclose(captured["pred_for_format"], expected_ordered_mean, rtol=1e-6)

    # Removing secondary ensemble checkpoint must degrade cleanly (no silent partial ensemble)
    ckpt_2.unlink()
    predictor._ensemble_models = []
    degraded = predictor.predict(price_session, "HG=F")
    assert degraded["model_state"] == "retrain_required"
    assert degraded["is_forecast_healthy"] is False


def test_generate_tft_analysis_populates_weekly_regime_and_filters_weekend_bars(
    monkeypatch, price_session
):
    import deep_learning.inference.predictor as predictor_mod

    # Add weekend ghost bar later than Friday price bar; weekday lookup must skip it
    price_session.add(
        PriceBar(symbol="HG=F", date=pd.Timestamp("2026-04-25 04:00:00+00:00").to_pydatetime(), close=9.99)
    )
    price_session.commit()

    class _StubPredictor:
        cfg = TFTASROConfig()

        def predict(self, _s, _sym, include_explain=False):
            return {
                "weekly_return": 0.032,
                "weekly_return_q10_calibrated": 0.005,
                "weekly_return_q90_calibrated": 0.055,
                "predicted_return_median": 0.006,
                "volatility_estimate": 0.008,
                "weekly_interval_calibrated": True,
                "is_forecast_healthy": True,
                "is_stale": False,
            }

        def get_model_metadata(self, _s):
            return None

    monkeypatch.setattr(predictor_mod, "get_tft_predictor", lambda: _StubPredictor())
    report = predictor_mod.generate_tft_analysis(price_session, "HG=F")
    assert report["weekly_forecast"]["regime"] == "trending_bull"

    latest_bar = TFTPredictor._latest_weekday_price_bar(price_session, "HG=F")
    assert latest_bar is not None
    assert latest_bar.date.weekday() < 5
    assert latest_bar.close == pytest.approx(6.0180)


def test_get_model_metadata_orders_newest_first_and_atomic_refresh_guards_cache(
    monkeypatch, price_session, tmp_path
):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.models import TFTModelMetadata
    import deep_learning.data.feature_store as feature_store
    import deep_learning.models.hub as hub_module
    import inspect

    # 1. Verify get_model_metadata filters quality_gate_passed and orders by trained_at.desc()
    src = inspect.getsource(TFTPredictor.get_model_metadata)
    assert "trained_at.desc()" in src
    assert "quality_gate_passed" in src

    engine = create_engine("sqlite:///:memory:")
    TFTModelMetadata.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(
            TFTModelMetadata(
                symbol="HG=F",
                checkpoint_path="new.ckpt",
                trained_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
                quality_gate_passed=True,
                config_json="{}",
                metrics_json="{}",
            )
        )
        s.commit()
        predictor = TFTPredictor(cfg=TFTASROConfig())
        meta = predictor.get_model_metadata(s)
        assert meta is not None
        assert meta["checkpoint_path"] == "new.ckpt"
    engine.dispose()

    # 2. Atomic HF refresh must preserve existing cached files when staged download fails validation
    tft_dir = tmp_path / "tft"
    tft_dir.mkdir()
    cached_ckpt = tft_dir / "best_tft_asro.ckpt"
    cached_ckpt.write_bytes(b"original-valid-cache")

    fake_pf = types.ModuleType("pytorch_forecasting")
    fake_pf.TimeSeriesDataSet = _FakeTimeSeriesDataSet
    monkeypatch.setitem(sys.modules, "pytorch_forecasting", fake_pf)
    monkeypatch.setattr(
        feature_store,
        "build_tft_dataframe",
        lambda _s, _c, *, drop_missing_target=True: (
            pd.DataFrame({"other_feat": [1.0, 2.0, 3.0], "target": [0.0, 0.0, 0.0]}, index=pd.to_datetime(["2026-04-22", "2026-04-23", "2026-04-24"])),
            ["other_feat"],
            [],
            ["target"],
            6.0,
        ),
    )

    def fake_bad_download(*, local_dir, repo_id, force_all=False):
        (local_dir / "best_tft_asro.ckpt").write_bytes(b"corrupt-partial")
        return True

    monkeypatch.setattr(hub_module, "download_tft_artifacts", fake_bad_download)
    monkeypatch.setattr(hub_module, "validate_tft_artifact_set", lambda _d: False)

    cfg = TFTASROConfig(
        model=TFTModelConfig(max_encoder_length=2, max_prediction_length=1),
        training=TrainingConfig(best_model_path=str(cached_ckpt)),
        feature_store=FeatureStoreConfig(target_symbol="HG=F", mrmr_top_k=0),
    )
    p2 = TFTPredictor(checkpoint_path=str(cached_ckpt), cfg=cfg)
    p2._model = _FakeModel()  # requires "feat", absent -> triggers HF refresh
    monkeypatch.setattr(p2, "_check_price_freshness", lambda _s, _sym: (1, False))

    degraded = p2.predict(price_session, "HG=F")
    assert degraded["model_state"] == "retrain_required"
    assert cached_ckpt.read_bytes() == b"original-valid-cache"

