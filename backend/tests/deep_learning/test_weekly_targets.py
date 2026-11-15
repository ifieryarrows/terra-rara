import numpy as np
import pandas as pd
import pytest

from deep_learning.data.validation import validate_weekly_target_contract


def test_log_return_formula():
    close = pd.Series([100, 101, 103, 102, 104, 110], dtype=float)
    log_close = np.log(close)
    target_5d = log_close.shift(-5) - log_close
    assert np.isclose(target_5d.iloc[0], np.log(110 / 100))


def test_validation_train_requires_real_weekly_labels():
    df = pd.DataFrame(
        {
            "target": [0.01, 0.02],
            "target_1d_log_return": [0.01, 0.02],
            "target_5d_log_return": [0.05, np.nan],
            "realized_vol_20d": [0.01, 0.01],
            "material_move_5d": [0.0, 1.0],
            "time_idx": [0, 1],
            "group_id": ["copper", "copper"],
        }
    )
    with pytest.raises(ValueError, match="target_5d_log_return contains NaN"):
        validate_weekly_target_contract(df, mode="train")


def test_validation_inference_allows_dummy_decoder_labels():
    df = pd.DataFrame(
        {
            "target": [0.01, 0.0],
            "target_1d_log_return": [0.01, 0.0],
            "target_5d_log_return": [0.05, 0.0],
            "realized_vol_20d": [0.01, 0.0],
            "material_move_5d": [0.0, 0.0],
            "time_idx": [0, 1],
            "group_id": ["copper", "copper"],
        }
    )
    validate_weekly_target_contract(df, mode="inference")


def test_validation_train_rejects_corrupt_single_day_log_return():
    df = pd.DataFrame(
        {
            "target": [0.01, 3.20],
            "target_1d_log_return": [0.01, 3.20],
            "target_5d_log_return": [0.05, 3.10],
            "realized_vol_20d": [0.01, 0.02],
            "material_move_5d": [0.0, 1.0],
            "time_idx": [0, 1],
            "group_id": ["copper", "copper"],
        }
    )
    with pytest.raises(ValueError, match="Corrupt single-day log return"):
        validate_weekly_target_contract(df, mode="train")


def test_sanitize_close_spikes_repairs_extreme_jumps():
    from deep_learning.data.feature_store import _sanitize_close_spikes

    dates = pd.date_range("2025-01-20", periods=5, freq="B")
    close = pd.Series([4.20, 4.22, 102.0, 4.25, 4.28], index=dates)
    repaired = _sanitize_close_spikes(close, symbol="HG=F")

    assert repaired.iloc[2] == pytest.approx(4.22)
    assert repaired.iloc[3] == pytest.approx(4.25)
    assert np.max(np.abs(np.log(repaired).diff().dropna())) < 0.05


def test_sanitize_close_spikes_does_not_bfill_or_flatline_persistent_shifts():
    from deep_learning.data.feature_store import _sanitize_close_spikes

    dates = pd.date_range("2025-01-20", periods=7, freq="B")
    # Leading invalid/zero prices must remain NaN (no future bfill), and a persistent
    # regime shift (6.50 -> 6.55 -> 6.60) must resume instead of flatlining forever.
    close = pd.Series([0.0, 4.20, 4.22, 6.50, 6.55, 6.60, 6.65], index=dates)
    repaired = _sanitize_close_spikes(close, symbol="TEST")

    assert pd.isna(repaired.iloc[0])
    assert repaired.iloc[1] == pytest.approx(4.20)
    assert repaired.iloc[2] == pytest.approx(4.22)
    assert repaired.iloc[4] == pytest.approx(6.55)
    assert repaired.iloc[6] == pytest.approx(6.65)


def test_realized_vol_20d_is_strictly_causal(monkeypatch):
    from deep_learning.config import (
        EmbeddingConfig,
        FeatureStoreConfig,
        TFTASROConfig,
        TrainingConfig,
    )
    from deep_learning.data import feature_store

    dates = pd.date_range("2026-01-05", periods=30, freq="B")
    base_closes = np.linspace(4.0, 4.5, len(dates))
    alt_closes = base_closes.copy()
    alt_closes[-1] = 5.6  # future close moves sharply on final bar

    def _run_with_closes(closes: np.ndarray) -> pd.DataFrame:
        price_df = pd.DataFrame(
            {
                "open": closes,
                "high": closes * 1.01,
                "low": closes * 0.99,
                "close": closes,
                "volume": 10000.0,
            },
            index=dates,
        )
        price_features = pd.DataFrame(
            {"close_lag_1": price_df["close"].shift(1).bfill()},
            index=dates,
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
        monkeypatch.setattr("app.features.load_sentiment_data", lambda *_a, **_k: pd.DataFrame())
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
            lambda *_a, **_k: pd.DataFrame(),
        )
        cfg = TFTASROConfig(
            embedding=EmbeddingConfig(pca_dim=0),
            training=TrainingConfig(lookback_days=60),
            feature_store=FeatureStoreConfig(target_symbol="HG=F", max_ffill=3, mrmr_top_k=0),
        )
        master, *_ = feature_store.build_tft_dataframe(object(), cfg, drop_missing_target=False)
        return master

    master_base = _run_with_closes(base_closes)
    master_alt = _run_with_closes(alt_closes)

    # All prior bars' realized_vol_20d must remain bit-identical regardless of future close[-1]
    pd.testing.assert_series_equal(
        master_base["realized_vol_20d"].iloc[:-1],
        master_alt["realized_vol_20d"].iloc[:-1],
    )
    # Initial warmup period (< min_periods=5) uses causal 0.01 constant instead of future bfill
    assert np.allclose(master_base["realized_vol_20d"].iloc[:4].to_numpy(), 0.01)
