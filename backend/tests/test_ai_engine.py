"""
Tests for AI Engine components.
"""

import asyncio
import sys
import types
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


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

    def test_score_text_list_dict_output(self):
        """Test FinBERT parsing when output is list[dict]."""
        from app.ai_engine import score_text_with_finbert

        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"label": "positive", "score": 0.6},
            {"label": "neutral", "score": 0.3},
            {"label": "negative", "score": 0.1},
        ]

        result = score_text_with_finbert(
            mock_pipe,
            "Copper demand remains strong with supportive macro backdrop",
        )

        assert result["prob_positive"] == 0.6
        assert result["prob_neutral"] == 0.3
        assert result["prob_negative"] == 0.1
        assert result["score"] == 0.5

    def test_score_text_json_string_flat_output(self):
        """Test FinBERT parsing when output is a JSON string of list[dict]."""
        from app.ai_engine import score_text_with_finbert

        mock_pipe = MagicMock()
        mock_pipe.return_value = (
            '[{"label":"positive","score":0.55},'
            '{"label":"neutral","score":0.35},'
            '{"label":"negative","score":0.10}]'
        )

        result = score_text_with_finbert(
            mock_pipe,
            "Refined copper demand outlook improved after industrial utilization data",
        )

        assert result["prob_positive"] == 0.55
        assert result["prob_neutral"] == 0.35
        assert result["prob_negative"] == 0.10
        assert result["score"] == pytest.approx(0.45)

    def test_score_text_json_string_nested_output(self):
        """Test FinBERT parsing when output is a JSON string of list[list[dict]]."""
        from app.ai_engine import score_text_with_finbert

        mock_pipe = MagicMock()
        mock_pipe.return_value = (
            '[[{"label":"positive","score":0.40},'
            '{"label":"neutral","score":0.20},'
            '{"label":"negative","score":0.40}]]'
        )

        result = score_text_with_finbert(
            mock_pipe,
            "Balanced copper news flow kept near-term expectations mostly neutral",
        )

        assert result["prob_positive"] == 0.40
        assert result["prob_neutral"] == 0.20
        assert result["prob_negative"] == 0.40
        assert result["score"] == 0.0

    def test_score_text_invalid_json_string_returns_neutral(self):
        """Invalid JSON string from FinBERT should fall back to neutral."""
        from app.ai_engine import score_text_with_finbert

        mock_pipe = MagicMock()
        mock_pipe.return_value = "not-valid-json"

        result = score_text_with_finbert(
            mock_pipe,
            "Copper markets reacted sharply to mixed macro and inventory signals today",
        )

        assert result["prob_positive"] == 0.33
        assert result["prob_neutral"] == 0.34
        assert result["prob_negative"] == 0.33
        assert result["score"] == 0.0

    def test_score_text_missing_labels_returns_neutral(self):
        """Missing required labels should fall back to neutral."""
        from app.ai_engine import score_text_with_finbert

        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"label": "positive", "score": 0.95}]

        result = score_text_with_finbert(
            mock_pipe,
            "Copper outlook remains constructive with selective demand-side support factors",
        )

        assert result["prob_positive"] == 0.33
        assert result["prob_neutral"] == 0.34
        assert result["prob_negative"] == 0.33
        assert result["score"] == 0.0


class TestAsyncBridge:
    """Tests for async-to-sync bridge utility."""

    def test_run_async_from_sync_without_running_loop(self):
        """Bridge should run async callable directly when no loop is active."""
        from app.async_bridge import run_async_from_sync

        async def add(a, b):
            await asyncio.sleep(0)
            return a + b

        assert run_async_from_sync(add, 2, 3) == 5

    def test_run_async_from_sync_with_running_loop(self):
        """Bridge should work when called from inside an active event loop."""
        from app.async_bridge import run_async_from_sync

        async def double(x):
            await asyncio.sleep(0)
            return x * 2

        async def caller():
            return run_async_from_sync(double, 21)

        assert asyncio.run(caller()) == 42

    def test_run_async_from_sync_propagates_exception(self):
        """Bridge should propagate async exceptions back to sync caller."""
        from app.async_bridge import run_async_from_sync

        async def fail():
            raise RuntimeError("bridge-failure")

        with pytest.raises(RuntimeError, match="bridge-failure"):
            run_async_from_sync(fail)


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


class TestDailyAggregationCopperFilter:
    def test_aggregate_daily_sentiment_filters_non_copper_articles(self, monkeypatch):
        from app import ai_engine
        from app.models import DailySentiment, NewsArticle, NewsSentiment

        engine = create_engine("sqlite:///:memory:")
        TestingSession = sessionmaker(bind=engine)
        NewsArticle.__table__.create(bind=engine)
        NewsSentiment.__table__.create(bind=engine)
        DailySentiment.__table__.create(bind=engine)
        session = TestingSession()

        monkeypatch.setattr(
            ai_engine,
            "get_settings",
            lambda: SimpleNamespace(sentiment_tau_hours=12.0),
        )

        copper_article = NewsArticle(
            dedup_key="copper-1",
            title="Copper supply tightening",
            description="Copper inventories decline",
            content="",
            url="https://example.com/copper",
            source="source",
            author="author",
            language="en",
            published_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            fetched_at=datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc),
        )
        oil_article = NewsArticle(
            dedup_key="oil-1",
            title="Crude oil inventory data",
            description="Energy complex update",
            content="",
            url="https://example.com/oil",
            source="source",
            author="author",
            language="en",
            published_at=datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc),
            fetched_at=datetime(2026, 1, 1, 13, 1, tzinfo=timezone.utc),
        )
        session.add(copper_article)
        session.add(oil_article)
        session.commit()

        session.add(
            NewsSentiment(
                news_article_id=copper_article.id,
                prob_positive=0.8,
                prob_neutral=0.1,
                prob_negative=0.1,
                score=0.7,
                reasoning="{}",
                model_name="hybrid",
                scored_at=datetime.now(timezone.utc),
            )
        )
        session.add(
            NewsSentiment(
                news_article_id=oil_article.id,
                prob_positive=0.2,
                prob_neutral=0.3,
                prob_negative=0.5,
                score=-0.4,
                reasoning="{}",
                model_name="hybrid",
                scored_at=datetime.now(timezone.utc),
            )
        )
        session.commit()

        count = ai_engine.aggregate_daily_sentiment(session=session)

        assert count == 1
        daily = session.query(DailySentiment).first()
        assert daily is not None
        assert daily.news_count == 1
        assert daily.sentiment_index == pytest.approx(0.7)


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


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def outerjoin(self, *_args, **_kwargs):
        return self

    def filter(self, *_args, **_kwargs):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, articles):
        self._articles = articles
        self.added = []
        self.commit_count = 0

    def query(self, *_args, **_kwargs):
        return _FakeQuery(self._articles)

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commit_count += 1


class TestLLMStructuredScoring:
    def test_score_batch_with_llm_structured_output(self, monkeypatch):
        from app import ai_engine

        fake_settings = SimpleNamespace(
            openrouter_api_key="test-key",
            resolved_scoring_model="stepfun/step-3.5-flash:free",
            openrouter_max_retries=3,
            openrouter_rpm=18,
            openrouter_fallback_models_list=[],
        )
        monkeypatch.setattr(ai_engine, "get_settings", lambda: fake_settings)

        async def fake_create_chat_completion(**_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '[{"id": 101, "label": "BULLISH", "confidence": 0.82, "reasoning": "Supply disruption tightens near-term availability."}]'
                        }
                    }
                ]
            }

        monkeypatch.setattr(ai_engine, "create_chat_completion", fake_create_chat_completion)

        async def run_call():
            return await ai_engine.score_batch_with_llm([
                {"id": 101, "title": "Copper output disrupted", "description": "Mine outage reported"},
            ])

        results = asyncio.run(run_call())

        assert len(results) == 1
        assert results[0]["id"] == 101
        assert results[0]["label"] == "BULLISH"
        assert results[0]["llm_confidence"] == pytest.approx(0.82)
        assert results[0]["model_name"] == "hybrid(stepfun/step-3.5-flash:free+ProsusAI/finbert)"

    def test_score_batch_with_llm_retries_relaxed_mode_on_404_provider_mismatch(self, monkeypatch):
        from app import ai_engine
        from app.openrouter_client import OpenRouterError

        fake_settings = SimpleNamespace(
            openrouter_api_key="test-key",
            resolved_scoring_model="stepfun/step-3.5-flash:free",
            openrouter_max_retries=3,
            openrouter_rpm=18,
            openrouter_fallback_models_list=[],
        )
        monkeypatch.setattr(ai_engine, "get_settings", lambda: fake_settings)

        call_count = {"n": 0}

        async def fake_create_chat_completion(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                assert "response_format" in kwargs
                assert "provider" in kwargs
                raise OpenRouterError(
                    "No endpoints found that can handle the requested parameters",
                    status_code=404,
                )
            assert "response_format" not in kwargs
            assert "provider" not in kwargs
            return {
                "choices": [
                    {
                        "message": {
                            "content": "```json\n[{\"id\": 42, \"label\": \"BEARISH\", \"confidence\": 0.4, \"reasoning\": \"Weak demand outlook weighs copper.\"}]\n```"
                        }
                    }
                ]
            }

        monkeypatch.setattr(ai_engine, "create_chat_completion", fake_create_chat_completion)

        async def run_call():
            return await ai_engine.score_batch_with_llm([
                {"id": 42, "title": "Copper demand slows", "description": "Manufacturing weakness"},
            ])

        results = asyncio.run(run_call())

        assert call_count["n"] == 2
        assert len(results) == 1
        assert results[0]["id"] == 42
        assert results[0]["label"] == "BEARISH"
        assert results[0]["llm_confidence"] == pytest.approx(0.4)

    def test_score_batch_with_llm_accepts_missing_confidence_in_relaxed_mode(self, monkeypatch):
        from app import ai_engine
        from app.openrouter_client import OpenRouterError

        fake_settings = SimpleNamespace(
            openrouter_api_key="test-key",
            resolved_scoring_model="stepfun/step-3.5-flash:free",
            openrouter_max_retries=3,
            openrouter_rpm=18,
            openrouter_fallback_models_list=[],
        )
        monkeypatch.setattr(ai_engine, "get_settings", lambda: fake_settings)

        call_count = {"n": 0}

        async def fake_create_chat_completion(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                assert "response_format" in kwargs
                raise OpenRouterError(
                    "No endpoints found that can handle the requested parameters",
                    status_code=404,
                )
            return {
                "choices": [
                    {
                        "message": {
                            "content": '[{"id": 55, "label": "BULLISH", "reasoning": "Supply shock."}]'
                        }
                    }
                ]
            }

        monkeypatch.setattr(ai_engine, "create_chat_completion", fake_create_chat_completion)

        async def run_call():
            return await ai_engine.score_batch_with_llm(
                [{"id": 55, "title": "Copper mine disruption", "description": "supply"}]
            )

        results = asyncio.run(run_call())

        assert call_count["n"] == 2
        assert results[0]["id"] == 55
        assert results[0]["label"] == "BULLISH"
        assert results[0]["llm_confidence"] == pytest.approx(0.5)

    def test_score_batch_with_llm_runs_json_repair_on_parse_failure(self, monkeypatch):
        from app import ai_engine

        fake_settings = SimpleNamespace(
            openrouter_api_key="test-key",
            resolved_scoring_model="stepfun/step-3.5-flash:free",
            openrouter_max_retries=3,
            openrouter_rpm=18,
            openrouter_fallback_models_list=[],
        )
        monkeypatch.setattr(ai_engine, "get_settings", lambda: fake_settings)

        call_count = {"n": 0}

        async def fake_create_chat_completion(**_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '[{"id": 7, "label": "BULLISH", "confidence": 0.7, "reasoning": "bad"'
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "content": '[{"id": 7, "label": "BULLISH", "confidence": 0.7, "reasoning": "Valid repaired json."}]'
                        }
                    }
                ]
            }

        monkeypatch.setattr(ai_engine, "create_chat_completion", fake_create_chat_completion)

        async def run_call():
            return await ai_engine.score_batch_with_llm(
                [{"id": 7, "title": "Copper demand improved", "description": "Strong grids"}]
            )

        results = asyncio.run(run_call())

        assert call_count["n"] == 2
        assert results[0]["id"] == 7
        assert results[0]["label"] == "BULLISH"
        assert results[0]["json_repair_used"] is True

    def test_score_batch_with_llm_retries_with_higher_max_tokens_on_empty_length(self, monkeypatch):
        from app import ai_engine

        fake_settings = SimpleNamespace(
            openrouter_api_key="test-key",
            resolved_scoring_model="stepfun/step-3.5-flash:free",
            openrouter_max_retries=3,
            openrouter_rpm=18,
            openrouter_fallback_models_list=[],
        )
        monkeypatch.setattr(ai_engine, "get_settings", lambda: fake_settings)

        call_count = {"n": 0}
        max_tokens_seen = []

        async def fake_create_chat_completion(**kwargs):
            call_count["n"] += 1
            max_tokens_seen.append(kwargs.get("max_tokens"))
            if call_count["n"] == 1:
                return {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": ""},
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '[{"id": 9, "label": "NEUTRAL", "confidence": 0.2, "reasoning": "Mixed signals."}]'
                        },
                    }
                ]
            }

        monkeypatch.setattr(ai_engine, "create_chat_completion", fake_create_chat_completion)

        async def run_call():
            return await ai_engine.score_batch_with_llm(
                [{"id": 9, "title": "Mixed copper outlook", "description": "mixed"}]
            )

        results = asyncio.run(run_call())

        assert call_count["n"] == 2
        assert max_tokens_seen[0] == ai_engine.LLM_SCORING_MAX_TOKENS_PRIMARY
        assert max_tokens_seen[1] == ai_engine.LLM_SCORING_MAX_TOKENS_RETRY
        assert results[0]["id"] == 9
        assert results[0]["label"] == "NEUTRAL"


class TestScoringFallbackAndBudget:
    def test_score_unscored_articles_falls_back_to_finbert_on_llm_error(self, monkeypatch):
        from app import ai_engine

        articles = [
            SimpleNamespace(id=1, title="A", description="d1"),
            SimpleNamespace(id=2, title="B", description="d2"),
        ]
        session = _FakeSession(articles)

        fake_settings = SimpleNamespace(
            openrouter_api_key="test-key",
            max_llm_articles_per_run=200,
            resolved_scoring_model="stepfun/step-3.5-flash:free",
        )
        monkeypatch.setattr(ai_engine, "get_settings", lambda: fake_settings)

        def fail_llm(*_args, **_kwargs):
            raise ai_engine.LLMStructuredOutputError("LLM JSON parse error")

        def fake_finbert(batch):
            return [
                {
                    "id": article.id,
                    "score": 0.3,
                    "prob_positive": 0.51,
                    "prob_neutral": 0.34,
                    "prob_negative": 0.15,
                    "finbert_strength": 0.36,
                }
                for article in batch
            ]

        monkeypatch.setattr(ai_engine, "run_async_from_sync", fail_llm)
        monkeypatch.setattr(ai_engine, "score_batch_with_finbert", fake_finbert)

        scored = ai_engine.score_unscored_articles(session=session, chunk_size=2)

        assert scored == 2
        assert len(session.added) == 2
        assert all(obj.model_name == "hybrid_fallback_parse" for obj in session.added)
        assert all(obj.score == pytest.approx(0.25) for obj in session.added)

    def test_score_unscored_articles_respects_llm_budget(self, monkeypatch):
        from app import ai_engine

        articles = [
            SimpleNamespace(id=1, title="A", description="d1"),
            SimpleNamespace(id=2, title="B", description="d2"),
            SimpleNamespace(id=3, title="C", description="d3"),
            SimpleNamespace(id=4, title="D", description="d4"),
            SimpleNamespace(id=5, title="E", description="d5"),
        ]
        session = _FakeSession(articles)
        llm_calls = []

        fake_settings = SimpleNamespace(
            openrouter_api_key="test-key",
            max_llm_articles_per_run=3,
            resolved_scoring_model="stepfun/step-3.5-flash:free",
        )
        monkeypatch.setattr(ai_engine, "get_settings", lambda: fake_settings)

        def fake_run_async(_fn, articles_data):
            llm_calls.append([item["id"] for item in articles_data])
            return [
                {
                    "id": item["id"],
                    "label": "BULLISH",
                    "llm_confidence": 0.8,
                    "llm_reasoning": "LLM scored",
                    "llm_model": "stepfun/step-3.5-flash:free",
                    "model_name": "hybrid(stepfun/step-3.5-flash:free+ProsusAI/finbert)",
                    "json_repair_used": False,
                }
                for item in articles_data
            ]

        def fake_finbert(batch):
            return [
                {
                    "id": article.id,
                    "score": 0.6,
                    "prob_positive": 0.8,
                    "prob_neutral": 0.3,
                    "prob_negative": 0.2,
                    "finbert_strength": 0.6,
                }
                for article in batch
            ]

        monkeypatch.setattr(ai_engine, "run_async_from_sync", fake_run_async)
        monkeypatch.setattr(ai_engine, "score_batch_with_finbert", fake_finbert)

        scored = ai_engine.score_unscored_articles(session=session, chunk_size=2)

        assert scored == 5
        assert llm_calls == [[1, 2], [3]]
        boosted_llm_scores = [obj for obj in session.added if obj.score >= 0.99]
        soft_neutral_scores = [obj for obj in session.added if obj.score == pytest.approx(0.25)]
        assert len(boosted_llm_scores) == 3
        assert len(soft_neutral_scores) == 2

    def test_score_unscored_articles_uses_neutral_429_fallback(self, monkeypatch):
        from app import ai_engine
        from app.openrouter_client import OpenRouterRateLimitError

        articles = [
            SimpleNamespace(id=1, title="A", description="d1"),
            SimpleNamespace(id=2, title="B", description="d2"),
        ]
        session = _FakeSession(articles)

        fake_settings = SimpleNamespace(
            openrouter_api_key="test-key",
            max_llm_articles_per_run=200,
            resolved_scoring_model="stepfun/step-3.5-flash:free",
        )
        monkeypatch.setattr(ai_engine, "get_settings", lambda: fake_settings)

        def fail_llm(*_args, **_kwargs):
            raise OpenRouterRateLimitError("429", status_code=429)

        def fake_finbert(batch):
            return [
                {
                    "id": article.id,
                    "score": 0.2,
                    "prob_positive": 0.6,
                    "prob_neutral": 0.2,
                    "prob_negative": 0.2,
                    "finbert_strength": 0.4,
                }
                for article in batch
            ]

        monkeypatch.setattr(ai_engine, "run_async_from_sync", fail_llm)
        monkeypatch.setattr(ai_engine, "score_batch_with_finbert", fake_finbert)

        scored = ai_engine.score_unscored_articles(session=session, chunk_size=2)

        assert scored == 2
        assert all(obj.model_name == "hybrid_fallback_429" for obj in session.added)
        assert all(obj.score == pytest.approx(0.25) for obj in session.added)


class TestHybridFormula:
    def test_compute_hybrid_score_soft_neutral_below_threshold_returns_zero(self):
        from app.ai_engine import _compute_hybrid_score

        score = _compute_hybrid_score(
            label="NEUTRAL",
            llm_confidence=0.9,
            finbert_strength=0.2,
            finbert_polarity=0.05,
        )
        assert score == pytest.approx(0.0)

    def test_compute_hybrid_score_soft_neutral_above_threshold_emits_directional_score(self):
        from app.ai_engine import _compute_hybrid_score

        score = _compute_hybrid_score(
            label="NEUTRAL",
            llm_confidence=0.0,
            finbert_strength=0.4,
            finbert_polarity=-0.4,
        )
        assert score == pytest.approx(-0.25)

    def test_compute_hybrid_score_bullish_boosted_magnitude(self):
        from app.ai_engine import _compute_hybrid_score

        score = _compute_hybrid_score(label="BULLISH", llm_confidence=0.7, finbert_strength=0.4)
        assert score == pytest.approx(0.8235)


class TestFinbertPipelineCaching:
    def test_get_finbert_pipeline_is_cached(self, monkeypatch):
        from app import ai_engine

        ai_engine.get_finbert_pipeline.cache_clear()
        calls = {"tokenizer": 0, "model": 0, "pipeline": 0}

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, _model_name):
                calls["tokenizer"] += 1
                return "tokenizer"

        class FakeModel:
            @classmethod
            def from_pretrained(cls, _model_name):
                calls["model"] += 1
                return "model"

        def fake_pipeline(*_args, **_kwargs):
            calls["pipeline"] += 1
            return object()

        fake_transformers_module = types.SimpleNamespace(
            pipeline=fake_pipeline,
            AutoModelForSequenceClassification=FakeModel,
            AutoTokenizer=FakeTokenizer,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers_module)

        first = ai_engine.get_finbert_pipeline()
        second = ai_engine.get_finbert_pipeline()

        assert first is second
        assert calls["tokenizer"] == 1
        assert calls["model"] == 1
        assert calls["pipeline"] == 1
