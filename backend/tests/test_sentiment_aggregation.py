"""
Tests for sentiment aggregation logic.
"""

import pytest
import numpy as np
from datetime import datetime, timezone


def calculate_recency_weights(hours_list: list[float], tau: float = 12.0) -> list[float]:
    """
    Calculate recency weights for testing.
    Mirrors the logic in ai_engine.py
    """
    weights = [np.exp(h / tau) for h in hours_list]
    total = sum(weights)
    return [w / total for w in weights]


class TestRecencyWeighting:
    def test_later_gets_higher_weight(self):
        """Articles later in the day should get higher weight."""
        hours = [9.0, 12.0, 16.0]  # 9am, noon, 4pm
        weights = calculate_recency_weights(hours)
        
        # Later hour should have higher weight
        assert weights[2] > weights[1] > weights[0]
    
    def test_weights_sum_to_one(self):
        """Normalized weights should sum to 1."""
        hours = [9.0, 12.0, 15.0, 18.0]
        weights = calculate_recency_weights(hours)
        
        assert abs(sum(weights) - 1.0) < 0.0001
    
    def test_single_article_weight_one(self):
        """Single article should have weight of 1."""
        weights = calculate_recency_weights([12.0])
        assert weights[0] == 1.0
    
    def test_tau_affects_spread(self):
        """Lower tau should increase weight spread."""
        hours = [9.0, 16.0]
        
        weights_high_tau = calculate_recency_weights(hours, tau=24.0)
        weights_low_tau = calculate_recency_weights(hours, tau=6.0)
        
        # With lower tau, the difference should be larger
        spread_high = weights_high_tau[1] - weights_high_tau[0]
        spread_low = weights_low_tau[1] - weights_low_tau[0]
        
        assert spread_low > spread_high


class TestSentimentIndex:
    def test_weighted_average(self):
        """Test weighted average calculation."""
        scores = [0.5, -0.2, 0.3]
        hours = [9.0, 14.0, 16.0]
        weights = calculate_recency_weights(hours)
        
        weighted_avg = sum(s * w for s, w in zip(scores, weights))
        
        # Should be between min and max scores
        assert min(scores) <= weighted_avg <= max(scores)
    
    def test_all_positive_yields_positive(self):
        """All positive scores should yield positive index."""
        scores = [0.3, 0.5, 0.4]
        hours = [9.0, 12.0, 15.0]
        weights = calculate_recency_weights(hours)
        
        weighted_avg = sum(s * w for s, w in zip(scores, weights))
        assert weighted_avg > 0
    
    def test_all_negative_yields_negative(self):
        """All negative scores should yield negative index."""
        scores = [-0.3, -0.5, -0.4]
        hours = [9.0, 12.0, 15.0]
        weights = calculate_recency_weights(hours)
        
        weighted_avg = sum(s * w for s, w in zip(scores, weights))
        assert weighted_avg < 0
    
    def test_equal_positive_negative_near_zero(self):
        """Equal positive and negative with same timing should be near zero."""
        scores = [0.5, -0.5]
        hours = [12.0, 12.0]  # Same time
        weights = calculate_recency_weights(hours)
        
        weighted_avg = sum(s * w for s, w in zip(scores, weights))
        assert abs(weighted_avg) < 0.1


class TestSentimentLabel:
    def test_bullish_threshold(self):
        """Positive sentiment above 0.1 should be Bullish."""
        from app.inference import get_sentiment_label
        
        assert get_sentiment_label(0.15) == "Bullish"
        assert get_sentiment_label(0.5) == "Bullish"
    
    def test_bearish_threshold(self):
        """Negative sentiment below -0.1 should be Bearish."""
        from app.inference import get_sentiment_label
        
        assert get_sentiment_label(-0.15) == "Bearish"
        assert get_sentiment_label(-0.5) == "Bearish"
    
    def test_neutral_range(self):
        """Sentiment between -0.1 and 0.1 should be Neutral."""
        from app.inference import get_sentiment_label
        
        assert get_sentiment_label(0.0) == "Neutral"
        assert get_sentiment_label(0.05) == "Neutral"
        assert get_sentiment_label(-0.05) == "Neutral"

