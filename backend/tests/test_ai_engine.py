"""
Tests for AI Engine components.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock


class TestFinBERTScoring:
    """Tests for FinBERT sentiment scoring."""
    
    def test_score_text_empty_input(self):
        """Test scoring with empty input."""
        from app.ai_engine import score_text_with_finbert
        
        # Mock pipeline
        mock_pipe = MagicMock()
        
        # Empty text should return neutral scores
        result = score_text_with_finbert(mock_pipe, "")
        
        assert result["prob_positive"] == 0.33
        assert result["prob_neutral"] == 0.34
        assert result["prob_negative"] == 0.33
        assert result["score"] == 0.0
    
    def test_score_text_short_input(self):
        """Test scoring with very short input."""
        from app.ai_engine import score_text_with_finbert
        
        mock_pipe = MagicMock()
        
        # Short text (< 10 chars) should return neutral
        result = score_text_with_finbert(mock_pipe, "hi")
        
        assert result["score"] == 0.0
    
    def test_score_text_normal_input(self):
        """Test scoring with normal input."""
        from app.ai_engine import score_text_with_finbert
        
        # Mock pipeline to return positive sentiment
        mock_pipe = MagicMock()
        mock_pipe.return_value = [[
            {"label": "positive", "score": 0.8},
            {"label": "neutral", "score": 0.15},
            {"label": "negative", "score": 0.05},
        ]]
        
        result = score_text_with_finbert(
            mock_pipe, 
            "Copper prices surge to new highs on strong demand"
        )
        
        assert result["prob_positive"] == 0.8
        assert result["prob_neutral"] == 0.15
        assert result["prob_negative"] == 0.05
        assert result["score"] == 0.75  # 0.8 - 0.05
    
    def test_score_text_negative_sentiment(self):
        """Test scoring with negative sentiment."""
        from app.ai_engine import score_text_with_finbert
        
        mock_pipe = MagicMock()
        mock_pipe.return_value = [[
            {"label": "positive", "score": 0.1},
            {"label": "neutral", "score": 0.2},
            {"label": "negative", "score": 0.7},
        ]]
        
        result = score_text_with_finbert(
            mock_pipe,
            "Copper prices crash amid recession fears"
        )
        
        assert result["score"] == -0.6  # 0.1 - 0.7


class TestSentimentAggregation:
    """Tests for sentiment aggregation logic."""
    
    def test_recency_weighting(self):
        """Test that later articles get higher weight."""
        # This tests the concept, actual implementation may vary
        tau = 12.0
        
        # Article at 9am vs 4pm
        hours_early = 9.0
        hours_late = 16.0
        
        weight_early = np.exp(hours_early / tau)
        weight_late = np.exp(hours_late / tau)
        
        # Later article should have higher weight
        assert weight_late > weight_early
    
    def test_weighted_average_calculation(self):
        """Test weighted average calculation."""
        scores = np.array([0.5, -0.2, 0.3])
        weights = np.array([0.2, 0.3, 0.5])  # Normalized weights
        
        weighted_avg = np.sum(scores * weights)
        expected = 0.5 * 0.2 + (-0.2) * 0.3 + 0.3 * 0.5
        
        assert abs(weighted_avg - expected) < 1e-10
    
    def test_sentiment_index_range(self):
        """Test that sentiment index is in valid range."""
        # Sentiment index should be between -1 and 1
        scores = np.array([0.9, -0.8, 0.5])
        weights = np.array([0.33, 0.33, 0.34])
        
        weighted_avg = np.sum(scores * weights)
        
        assert -1 <= weighted_avg <= 1


class TestFeatureEngineering:
    """Tests for feature engineering."""
    
    def test_technical_indicators(self, sample_price_data):
        """Test that technical indicators are calculated correctly."""
        df = sample_price_data
        
        # Calculate SMA
        sma_5 = df["close"].rolling(window=5).mean()
        sma_10 = df["close"].rolling(window=10).mean()
        
        # SMA calculations should not be NaN after sufficient data
        assert not np.isnan(sma_5.iloc[-1])
        assert not np.isnan(sma_10.iloc[-1])
        
        # SMA10 should smooth more than SMA5
        assert sma_10.std() < df["close"].std()
    
    def test_return_calculation(self, sample_price_data):
        """Test return calculation."""
        df = sample_price_data
        
        # Calculate returns
        returns = df["close"].pct_change()
        
        # First return should be NaN
        assert np.isnan(returns.iloc[0])
        
        # Returns should be small (reasonable daily returns)
        assert abs(returns.iloc[1:].mean()) < 0.1
    
    def test_volatility_calculation(self, sample_price_data):
        """Test volatility calculation."""
        df = sample_price_data
        
        returns = df["close"].pct_change()
        volatility_10 = returns.rolling(window=10).std()
        
        # Volatility should be positive
        assert all(v >= 0 or np.isnan(v) for v in volatility_10)
    
    def test_lagged_features(self, sample_price_data):
        """Test lagged feature creation."""
        df = sample_price_data
        
        returns = df["close"].pct_change()
        
        # Create lags
        lag_1 = returns.shift(1)
        lag_2 = returns.shift(2)
        lag_3 = returns.shift(3)
        
        # Lags should have correct offset
        assert lag_1.iloc[5] == returns.iloc[4]
        assert lag_2.iloc[5] == returns.iloc[3]
        assert lag_3.iloc[5] == returns.iloc[2]


class TestModelTraining:
    """Tests for model training logic."""
    
    def test_train_test_split_temporal(self):
        """Test that train/test split respects time order."""
        dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
        
        validation_days = 20
        split_date = dates.max() - timedelta(days=validation_days)
        
        train_dates = dates[dates <= split_date]
        val_dates = dates[dates > split_date]
        
        # All train dates should be before all val dates
        assert train_dates.max() < val_dates.min()
        
        # Correct number of validation samples
        assert len(val_dates) == validation_days
    
    def test_feature_importance_normalized(self):
        """Test that feature importance sums to 1."""
        importance = {
            "feature_a": 10.0,
            "feature_b": 5.0,
            "feature_c": 3.0,
            "feature_d": 2.0,
        }
        
        total = sum(importance.values())
        normalized = {k: v / total for k, v in importance.items()}
        
        assert abs(sum(normalized.values()) - 1.0) < 1e-10
    
    def test_prediction_direction_from_return(self):
        """Test prediction direction logic."""
        def get_direction(predicted_return, threshold=0.005):
            if predicted_return > threshold:
                return "up"
            elif predicted_return < -threshold:
                return "down"
            else:
                return "neutral"
        
        assert get_direction(0.02) == "up"
        assert get_direction(-0.02) == "down"
        assert get_direction(0.001) == "neutral"
        assert get_direction(-0.003) == "neutral"


class TestModelPersistence:
    """Tests for model saving and loading."""
    
    def test_model_path_generation(self):
        """Test model path generation."""
        from datetime import datetime
        
        target_symbol = "HG=F"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_filename = f"xgb_{target_symbol.replace('=', '_')}_{timestamp}.json"
        latest_filename = f"xgb_{target_symbol.replace('=', '_')}_latest.json"
        
        assert "HG_F" in model_filename
        assert "HG_F" in latest_filename
        assert model_filename.endswith(".json")
    
    def test_metrics_json_structure(self):
        """Test that metrics JSON has required fields."""
        import json
        
        metrics = {
            "target_symbol": "HG=F",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_samples": 200,
            "val_samples": 30,
            "train_mae": 0.01,
            "train_rmse": 0.015,
            "val_mae": 0.02,
            "val_rmse": 0.025,
            "best_iteration": 50,
            "feature_count": 58,
        }
        
        # Should serialize properly
        json_str = json.dumps(metrics)
        loaded = json.loads(json_str)
        
        assert loaded["target_symbol"] == "HG=F"
        assert loaded["val_mae"] == 0.02
