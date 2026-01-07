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
        # Strong uptrend
        prices = pd.Series(range(1, 31))  # 1 to 30
        rsi = compute_rsi(prices)
        
        # Should be high (close to 100)
        assert rsi.iloc[-1] > 80
    
    def test_downtrend_low_rsi(self):
        # Strong downtrend
        prices = pd.Series(range(30, 0, -1))  # 30 to 1
        rsi = compute_rsi(prices)
        
        # Should be low (close to 0)
        assert rsi.iloc[-1] < 20


class TestComputeVolatility:
    def test_volatility_positive(self):
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        vol = compute_volatility(returns)
        
        assert (vol >= 0).all()
    
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

