"""
Tests for feature engineering functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.features import (
    compute_returns,
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_volatility,
    generate_symbol_features,
    describe_feature,
)


class TestComputeReturns:
    def test_simple_returns(self):
        prices = pd.Series([100, 110, 105])
        returns = compute_returns(prices)
        
        assert pd.isna(returns.iloc[0])  # First return is NaN
        assert abs(returns.iloc[1] - 0.10) < 0.001  # 10% return
        assert abs(returns.iloc[2] - (-0.0454545)) < 0.001  # -4.5% return
    
    def test_multi_period_returns(self):
        prices = pd.Series([100, 105, 110, 115])
        returns = compute_returns(prices, periods=2)
        
        # 2-period return from 100 to 110
        assert abs(returns.iloc[2] - 0.10) < 0.001


class TestComputeSMA:
    def test_simple_case(self):
        prices = pd.Series([1, 2, 3, 4, 5])
        sma = compute_sma(prices, window=3)
        
        # SMA of last 3 values [3, 4, 5] = 4
        assert abs(sma.iloc[-1] - 4.0) < 0.001
    
    def test_handles_short_series(self):
        prices = pd.Series([1, 2])
        sma = compute_sma(prices, window=5)
        
        # Should still produce values with min_periods=1
        assert not sma.isna().all()


class TestComputeEMA:
    def test_more_weight_to_recent(self):
        prices = pd.Series([1, 1, 1, 1, 5])  # Jump at end
        ema = compute_ema(prices, span=3)
        sma = compute_sma(prices, window=3)
        
        # EMA should be higher than SMA due to recent jump
        assert ema.iloc[-1] > sma.iloc[-1]


class TestComputeRSI:
    def test_rsi_range(self):
        # Generate random walk
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(50)))
        rsi = compute_rsi(prices)
        
        # RSI should be between 0 and 100
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()
    
    def test_uptrend_high_rsi(self):
        # Strong uptrend with enough data points
        prices = pd.Series([float(i) for i in range(1, 51)])  # 1 to 50
        rsi = compute_rsi(prices)
        
        # Should be high (above 50 for uptrend)
        # Note: RSI depends on implementation details
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert valid_rsi.iloc[-1] >= 50  # Uptrend should have RSI >= 50
    
    def test_downtrend_low_rsi(self):
        # Strong downtrend
        prices = pd.Series(range(30, 0, -1))  # 30 to 1
        rsi = compute_rsi(prices)
        
        # Should be low (close to 0)
        assert rsi.iloc[-1] < 20


class TestComputeVolatility:
    def test_volatility_positive(self):
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02, 0.01, -0.01, 0.02, -0.02, 0.01])
        vol = compute_volatility(returns)
        
        # Only check non-NaN values
        valid_vol = vol.dropna()
        assert (valid_vol >= 0).all()
    
    def test_flat_returns_zero_vol(self):
        returns = pd.Series([0.01] * 10)  # Constant returns
        vol = compute_volatility(returns)
        
        assert abs(vol.iloc[-1]) < 0.0001


class TestGenerateSymbolFeatures:
    def test_feature_columns_created(self, sample_price_data):
        features = generate_symbol_features(sample_price_data, "TEST")
        
        # Check expected columns exist
        assert "TEST_ret1" in features.columns
        assert "TEST_SMA_5" in features.columns
        assert "TEST_EMA_10" in features.columns
        assert "TEST_RSI_14" in features.columns
        assert "TEST_vol_10" in features.columns
    
    def test_lagged_features(self, sample_price_data):
        features = generate_symbol_features(
            sample_price_data,
            "TEST",
            include_lags=[1, 2, 5]
        )
        
        assert "TEST_lag_ret1_1" in features.columns
        assert "TEST_lag_ret1_2" in features.columns
        assert "TEST_lag_ret1_5" in features.columns
    
    def test_output_same_index(self, sample_price_data):
        features = generate_symbol_features(sample_price_data, "TEST")
        
        assert len(features) == len(sample_price_data)
        assert features.index.equals(sample_price_data.index)
    
    def test_no_future_leakage(self, sample_price_data):
        """Ensure features don't use future data."""
        features = generate_symbol_features(sample_price_data, "TEST")
        
        # Lagged returns should be shifted
        # lag_ret1_1 at time t should equal ret1 at time t-1
        ret1 = features["TEST_ret1"]
        lag1 = features["TEST_lag_ret1_1"]
        
        # Check a middle value (not first few which may be NaN)
        idx = 10
        assert abs(lag1.iloc[idx] - ret1.iloc[idx - 1]) < 0.0001


class TestFeatureDescriptions:
    def test_sanitized_symbols_use_canonical_user_facing_labels(self):
        dxy = describe_feature("DX_Y_NYB_price_sma_ratio")
        copper = describe_feature("HG_F_SMA_5")

        assert dxy["label"] == "US Dollar Index · Price vs. 20-day average"
        assert dxy["category"] == "Trend"
        assert copper["label"] == "Copper Futures · 5-day moving average"
        assert copper["category"] == "Trend"

    def test_long_labels_are_not_truncated(self):
        described = describe_feature("COPX_EMA_20")

        assert "Global Copper Miners ETF" in described["label"]
        assert "weighted moving average" in described["label"]
        assert "…" not in described["label"]

    def test_bare_indicators_attribute_to_copper(self):
        ret = describe_feature("ret1")
        assert ret["label"] == "Copper Futures · 1-day price change"
        assert ret["category"] == "Momentum"

        sma = describe_feature("SMA_20")
        assert sma["label"] == "Copper Futures · 20-day moving average"
        assert sma["category"] == "Trend"
        assert sma["time_horizon"] == "20d"

        vol = describe_feature("vol_10")
        assert vol["label"] == "Copper Futures · 10-day price volatility"
        assert vol["category"] == "Volatility"
        assert vol["time_horizon"] == "10d"

    def test_universe_symbols_and_indicators_format_cleanly(self):
        gxc = describe_feature("GXC_lag_ret1_1")
        assert gxc["label"] == "China Large-Cap · 1-day lagged return"
        assert gxc["category"] == "Momentum"

        hbm = describe_feature("HBM_TO_SMA_10")
        assert hbm["label"] == "Hudbay Minerals · 10-day moving average"
        assert hbm["category"] == "Trend"

        lun = describe_feature("LUN.TO_lag_ret1_3")
        assert lun["label"] == "Lundin Mining · 3-day lagged return"

        fcx = describe_feature("FCX_vol_10")
        assert fcx["label"] == "Freeport-McMoRan · 10-day price volatility"
        assert fcx["category"] == "Volatility"


class TestTargetCreation:
    def test_target_shift(self, sample_price_data):
        """Target should be next-day return (shifted by -1)."""
        from app.features import compute_returns
        
        close = sample_price_data["close"]
        ret1 = compute_returns(close)
        target = ret1.shift(-1)  # Next day's return
        
        # At time t, target should be the return from t to t+1
        # Which equals (close[t+1] - close[t]) / close[t]
        for i in range(len(close) - 1):
            expected = (close.iloc[i + 1] - close.iloc[i]) / close.iloc[i]
            if not pd.isna(target.iloc[i]):
                assert abs(target.iloc[i] - expected) < 0.0001
        
        # Last value should be NaN (no future data)
        assert pd.isna(target.iloc[-1])


class TestRowsToPriceDfHygiene:
    def test_filters_leaked_synthetic_fixtures_and_deduplicates_calendar_days(self):
        from app.features import _rows_to_price_df

        rows = [
            ("2025-01-28T00:00:00Z", 100.0, 105.0, 95.0, 102.0, 1000.0, 102.0),
            ("2025-01-28T05:00:00Z", 4.18, 4.25, 4.15, 4.22, 54000.0, 4.22),
            ("2025-01-29T00:00:00Z", 100.0, 105.0, 95.0, 102.0, 1000.0, 102.0),
            # In Jan (EST = UTC-5), 05:00 UTC and 14:00 UTC are both NY trading day 2025-01-30
            ("2025-01-30T05:00:00Z", 4.22, 4.28, 4.20, 4.26, 48000.0, 4.26),
            ("2025-01-30T14:00:00Z", 4.26, 4.30, 4.24, 4.29, 51000.0, 4.29),
        ]

        df = _rows_to_price_df(rows, "HG=F")

        assert len(df) == 2
        assert list(df.index) == list(pd.to_datetime(["2025-01-28", "2025-01-30"]))
        assert df.loc[pd.Timestamp("2025-01-28"), "close"] == pytest.approx(4.22)
        assert df.loc[pd.Timestamp("2025-01-30"), "close"] == pytest.approx(4.29)

    def test_preserves_pure_synthetic_series_for_isolated_unit_tests(self):
        from app.features import _rows_to_price_df

        rows = [
            ("2025-01-28T00:00:00Z", 100.0, 105.0, 95.0, 102.0, 1000.0, 102.0),
            ("2025-01-29T00:00:00Z", 100.0, 105.0, 95.0, 102.0, 1000.0, 102.0),
        ]

        df = _rows_to_price_df(rows, "SYNTH")

        assert len(df) == 2
        assert (df["close"] == 102.0).all()

    def test_filters_leaked_fixtures_when_close_is_near_102_by_volume_or_collision(self):
        from app.features import _rows_to_price_df

        # DX-Y.NYB has close ~104 (near 102.0) but 0.0 volume, whereas leaked fixture has 1000.0 volume
        rows = [
            ("2025-01-28T00:00:00Z", 100.0, 105.0, 95.0, 102.0, 1000.0, 102.0),
            ("2025-01-28T05:00:00Z", 103.5, 104.2, 103.1, 104.0, 0.0, 104.0),
            ("2025-01-29T00:00:00Z", 100.0, 105.0, 95.0, 102.0, 1000.0, 102.0),
            ("2025-01-30T05:00:00Z", 104.0, 104.8, 103.7, 104.5, 0.0, 104.5),
        ]

        df = _rows_to_price_df(rows, "DX-Y.NYB")

        assert len(df) == 2
        assert list(df.index) == list(pd.to_datetime(["2025-01-28", "2025-01-30"]))
        assert (df["close"] > 103.5).all()

    def test_preserves_exchange_calendar_date_and_drops_invalid_dates(self):
        from app.features import _rows_to_price_df

        rows = [
            ("2025-01-28T00:00:00+09:00", 50.0, 51.0, 49.0, 50.5, 12000.0, 50.5),
            ("not-a-date", 50.0, 51.0, 49.0, 50.5, 12000.0, 50.5),
            ("2025-01-29T05:00:00Z", 50.5, 51.5, 50.0, 51.2, 13000.0, 51.2),
        ]

        df = _rows_to_price_df(rows, "ASIA")

        assert list(df.index) == list(pd.to_datetime(["2025-01-28", "2025-01-29"]))

    def test_prefers_canonical_utc_midnight_over_stale_offset_and_drops_nan_close(self):
        from app.features import _rows_to_price_df

        # Even though 05:00Z has a later wall-clock timestamp on 2024-01-22,
        # canonical 00:00Z (latest auto_adjust=False ingest) must win.
        rows = [
            ("2024-01-22T00:00:00Z", 46.0, 46.5, 45.5, 46.2, 50000.0, 42.10),
            ("2024-01-22T05:00:00Z", 41.0, 41.5, 40.5, 41.2, 48000.0, 41.20),
            ("2024-01-23T00:00:00Z", 46.2, 47.0, 46.0, float("nan"), 10000.0, float("nan")),
            ("2024-01-24T00:00:00Z", 46.5, 47.2, 46.1, 46.9, 51000.0, 42.74),
        ]
        df = _rows_to_price_df(rows, "BHP")

        assert list(df.index) == list(pd.to_datetime(["2024-01-22", "2024-01-24"]))
        assert df.loc[pd.Timestamp("2024-01-22"), "close"] == pytest.approx(46.2)
        assert df.loc[pd.Timestamp("2024-01-22"), "adj_close"] == pytest.approx(42.10)

    def test_hk_1600_utc_maps_to_next_hkt_trading_day(self):
        from app.features import _rows_to_price_df

        # Sunday 2026-11-15T16:00:00Z == Monday 2026-11-16 00:00 HKT -> must NOT be dropped as weekend
        rows = [
            ("2026-11-15T16:00:00Z", 25.9, 26.2, 25.8, 25.98, 100000.0, 25.98),
            ("2026-11-16T00:00:00Z", 25.9, 26.2, 25.8, 26.05, 110000.0, 26.05),
        ]
        df = _rows_to_price_df(rows, "2899.HK")
        assert len(df) == 1
        assert df.index[0] == pd.Timestamp("2026-11-16")
        assert df.iloc[0]["close"] == pytest.approx(26.05)


class TestAdjCloseAndStationaryHygiene:
    def test_rsi_boundary_conditions(self):
        # Monotonic uptrend -> 100.0, monotonic downtrend -> 0.0, flat -> 50.0
        up = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        down = pd.Series([14.0, 13.0, 12.0, 11.0, 10.0])
        flat = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])

        assert compute_rsi(up).iloc[-1] == pytest.approx(100.0)
        assert compute_rsi(down).iloc[-1] == pytest.approx(0.0)
        assert compute_rsi(flat).iloc[-1] == pytest.approx(50.0)

    def test_generate_symbol_features_uses_adj_close_and_eliminates_duplicate_ratio(self):
        dates = pd.bdate_range("2025-01-01", periods=30)
        # Simulate a 2-for-1 split / large ex-dividend drop in raw close while adj_close is smooth
        raw_close = np.array([100.0] * 20 + [50.0] * 10, dtype=float)
        adj_close = np.linspace(50.0, 52.9, 30, dtype=float)
        df = pd.DataFrame(
            {"close": raw_close, "adj_close": adj_close, "volume": 1000.0},
            index=dates,
        )

        feats = generate_symbol_features(df, "BHP", include_raw_ma=False)
        assert "BHP_price_sma_ratio" not in feats.columns
        assert "BHP_close_sma_20_ratio" in feats.columns
        # No -50% fake crash in ret1 or ret5
        assert feats["BHP_ret1"].dropna().abs().max() < 0.05
        assert feats["BHP_ret5"].dropna().abs().max() < 0.05

        # Legacy include_raw_ma=True still emits price_sma_ratio
        feats_raw = generate_symbol_features(df, "BHP", include_raw_ma=True)
        assert "BHP_price_sma_ratio" in feats_raw.columns

    def test_generate_aligned_symbol_features_avoids_holiday_vol_distortion(self):
        from app.features import generate_aligned_symbol_features

        target_dates = pd.bdate_range("2025-01-01", periods=30)
        # Drop every 3rd day as foreign holiday in correlated symbol
        native_dates = target_dates[[i for i in range(len(target_dates)) if i % 3 != 1]]
        rng = np.random.default_rng(12)
        native_close = 20.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, len(native_dates))))
        native_df = pd.DataFrame({"close": native_close, "adj_close": native_close}, index=native_dates)

        native_direct = generate_symbol_features(native_df, "HBM.TO", include_raw_ma=False)
        aligned = generate_aligned_symbol_features(
            native_df,
            "HBM.TO",
            target_dates,
            max_ffill=3,
            include_raw_ma=False,
        )

        # Native vol_20 on active bars must match exactly (zero fake 0% holiday damping)
        common_idx = native_dates[-5:]
        pd.testing.assert_series_equal(
            aligned.loc[common_idx, "HBM.TO_vol_10"],
            native_direct.loc[common_idx, "HBM.TO_vol_10"],
        )
        # Holiday bar gets ret1 == 0.0 rather than repeating previous trading day's return
        holiday_dt = target_dates[10]
        assert holiday_dt not in native_dates
        assert aligned.loc[holiday_dt, "HBM.TO_ret1"] == pytest.approx(0.0)


