"""Tests for advanced sentiment feature engineering."""

import numpy as np
import pandas as pd
import pytest

from deep_learning.data.sentiment_features import (
    compute_sentiment_momentum,
    compute_sentiment_surprise,
    compute_volume_weighted_sentiment,
    compute_event_type_intensity,
    build_all_sentiment_features,
)


@pytest.fixture
def daily_sentiment():
    """60-day synthetic sentiment data."""
    np.random.seed(42)
    dates = pd.date_range("2025-06-01", periods=60, freq="B")
    return pd.DataFrame(
        {
            "sentiment_index": np.random.normal(0, 0.3, 60).cumsum() * 0.01,
            "news_count": np.random.poisson(5, 60),
        },
        index=dates,
    )


def test_momentum_produces_expected_columns(daily_sentiment):
    si = daily_sentiment["sentiment_index"]
    result = compute_sentiment_momentum(si, windows=(5, 10))
    assert "sent_momentum_5d" in result.columns
    assert "sent_ema_10d" in result.columns
    assert "sent_roc_5d" in result.columns
    assert len(result) == len(si)


def test_surprise_zscore_range(daily_sentiment):
    si = daily_sentiment["sentiment_index"]
    result = compute_sentiment_surprise(si, lookback=30, threshold=2.0)
    z = result["sent_surprise_z"].dropna()
    assert z.abs().max() < 10, "Z-scores should be reasonable"
    assert "sent_surprise_flag" in result.columns
    assert result["sent_surprise_flag"].dtype == np.float32


def test_volume_weighted_not_nan(daily_sentiment):
    si = daily_sentiment["sentiment_index"]
    nc = daily_sentiment["news_count"]
    result = compute_volume_weighted_sentiment(si, nc)
    assert "sent_vol_weighted" in result.columns
    assert result["sent_vol_weighted"].notna().sum() > 0


def test_event_type_intensity_with_missing_types():
    dates = pd.date_range("2025-01-01", periods=10, freq="B")
    event_counts = pd.DataFrame(
        {"supply_disruption": [1, 0, 2, 0, 0, 1, 3, 0, 0, 0]},
        index=dates,
    )
    result = compute_event_type_intensity(event_counts, event_types=["supply_disruption", "demand_increase"])
    assert "evt_supply_disruption_count" in result.columns
    assert "evt_demand_increase_count" in result.columns
    assert (result["evt_demand_increase_count"] == 0).all()


def test_build_all_sentiment_features_shape(daily_sentiment):
    result = build_all_sentiment_features(daily_sentiment)
    assert len(result) == len(daily_sentiment)
    assert result.shape[1] > 5
    assert not result.empty
