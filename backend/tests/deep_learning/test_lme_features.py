"""Tests for LME warehouse and futures curve feature engineering."""

import numpy as np
import pandas as pd
import pytest

from deep_learning.data.lme_warehouse import compute_lme_features, compute_proxy_lme_features
from deep_learning.data.futures_curve import compute_futures_spread, compute_curve_slope


@pytest.fixture
def lme_data():
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    np.random.seed(42)
    stock = 200000 + np.cumsum(np.random.normal(-500, 2000, 60))
    cancelled = stock * np.random.uniform(0.05, 0.15, 60)
    return pd.DataFrame(
        {
            "total_stock_tonnes": stock,
            "cancelled_warrants_tonnes": cancelled,
            "cancelled_ratio": cancelled / stock,
        },
        index=dates,
    )


@pytest.fixture
def price_data():
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    np.random.seed(42)
    close = 4.0 + np.cumsum(np.random.normal(0, 0.05, 60))
    vol = np.random.randint(5000, 50000, 60).astype(float)
    return pd.DataFrame({"close": close, "volume": vol}, index=dates)


def test_lme_features_columns(lme_data):
    features = compute_lme_features(lme_data)
    assert "lme_stock_total" in features.columns
    assert "lme_stock_change_5d" in features.columns
    assert "lme_depletion_rate" in features.columns
    assert "lme_cancelled_ratio" in features.columns
    assert len(features) == len(lme_data)


def test_lme_features_empty():
    features = compute_lme_features(pd.DataFrame())
    assert features.empty


def test_proxy_features(price_data):
    features = compute_proxy_lme_features(price_data)
    assert "proxy_vol_zscore" in features.columns
    assert "proxy_vol_spike" in features.columns
    assert len(features) == len(price_data)


def test_futures_spread():
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    front = pd.Series(4.0 + np.random.normal(0, 0.1, 30), index=dates)
    deferred = front + np.random.normal(0.05, 0.02, 30)
    result = compute_futures_spread(front, deferred)
    assert "futures_spread_raw" in result.columns
    assert "backwardation_flag" in result.columns
    assert "contango_flag" in result.columns


def test_curve_slope():
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    prices = {
        "3m": pd.Series(4.0 + np.arange(30) * 0.01, index=dates),
        "6m": pd.Series(4.1 + np.arange(30) * 0.01, index=dates),
    }
    result = compute_curve_slope(prices)
    assert "futures_curve_slope" in result.columns
    assert len(result) > 0
