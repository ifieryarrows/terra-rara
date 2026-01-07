"""
Tests for API endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""
    
    def test_health_response_structure(self):
        """Test that health response has required fields."""
        from app.schemas import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            db_type="postgresql",
            models_found=1,
            pipeline_locked=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            news_count=100,
            price_bars_count=500
        )
        
        assert response.status == "healthy"
        assert response.db_type == "postgresql"
        assert response.models_found == 1
        assert response.pipeline_locked is False
        assert response.news_count == 100
        assert response.price_bars_count == 500
    
    def test_health_status_degraded_no_models(self):
        """Test degraded status when no models found."""
        from app.schemas import HealthResponse
        
        response = HealthResponse(
            status="degraded",
            db_type="postgresql",
            models_found=0,
            pipeline_locked=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        assert response.status == "degraded"
        assert response.models_found == 0


class TestAnalysisSchema:
    """Tests for analysis report schema."""
    
    def test_analysis_report_structure(self):
        """Test AnalysisReport schema validation."""
        from app.schemas import AnalysisReport, Influencer
        
        influencers = [
            Influencer(feature="HG=F_EMA_10", importance=0.15, description="Test"),
            Influencer(feature="DX-Y.NYB_ret1", importance=0.10, description="Test"),
        ]
        
        report = AnalysisReport(
            symbol="HG=F",
            prediction_direction="up",
            confidence_score=0.75,
            current_price=4.25,
            predicted_return=0.015,
            sentiment_index=0.35,
            news_count_24h=15,
            model_metrics={
                "val_mae": 0.02,
                "val_rmse": 0.025,
            },
            top_influencers=influencers,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        
        assert report.symbol == "HG=F"
        assert report.prediction_direction == "up"
        assert report.confidence_score == 0.75
        assert len(report.top_influencers) == 2
    
    def test_prediction_direction_values(self):
        """Test valid prediction directions."""
        from app.schemas import AnalysisReport
        
        for direction in ["up", "down", "neutral"]:
            report = AnalysisReport(
                symbol="HG=F",
                prediction_direction=direction,
                confidence_score=0.5,
                current_price=4.0,
                predicted_return=0.0,
                sentiment_index=0.0,
                news_count_24h=0,
                model_metrics={},
                top_influencers=[],
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            assert report.prediction_direction == direction


class TestHistorySchema:
    """Tests for history response schema."""
    
    def test_history_data_point(self):
        """Test HistoryDataPoint schema."""
        from app.schemas import HistoryDataPoint
        
        point = HistoryDataPoint(
            date="2026-01-01",
            price=4.25,
            sentiment_index=0.35,
            sentiment_news_count=10,
        )
        
        assert point.date == "2026-01-01"
        assert point.price == 4.25
        assert point.sentiment_index == 0.35
        assert point.sentiment_news_count == 10
    
    def test_history_data_point_nullable_sentiment(self):
        """Test that sentiment can be None."""
        from app.schemas import HistoryDataPoint
        
        point = HistoryDataPoint(
            date="2026-01-01",
            price=4.25,
            sentiment_index=None,
            sentiment_news_count=None,
        )
        
        assert point.sentiment_index is None
        assert point.sentiment_news_count is None
    
    def test_history_response(self):
        """Test HistoryResponse schema."""
        from app.schemas import HistoryResponse, HistoryDataPoint
        
        data = [
            HistoryDataPoint(date="2026-01-01", price=4.20),
            HistoryDataPoint(date="2026-01-02", price=4.25),
        ]
        
        response = HistoryResponse(symbol="HG=F", data=data)
        
        assert response.symbol == "HG=F"
        assert len(response.data) == 2


class TestPipelineLock:
    """Tests for pipeline lock mechanism."""
    
    def test_lock_acquire_release(self, tmp_path):
        """Test acquiring and releasing lock."""
        from app.lock import PipelineLock
        
        lock_file = tmp_path / "test.lock"
        lock = PipelineLock(lock_file=str(lock_file), timeout=0)
        
        # Should acquire
        assert lock.acquire() is True
        assert lock_file.exists()
        
        # Should release
        lock.release()
        assert not lock_file.exists()
    
    def test_lock_already_held(self, tmp_path):
        """Test that second acquire fails when lock is held."""
        from app.lock import PipelineLock
        
        lock_file = tmp_path / "test.lock"
        lock1 = PipelineLock(lock_file=str(lock_file), timeout=0)
        lock2 = PipelineLock(lock_file=str(lock_file), timeout=0)
        
        # First lock should succeed
        assert lock1.acquire() is True
        
        # Second lock should fail
        assert lock2.acquire() is False
        
        # Cleanup
        lock1.release()
    
    def test_is_pipeline_locked(self, tmp_path):
        """Test is_pipeline_locked helper."""
        from app.lock import PipelineLock
        
        lock_file = tmp_path / "test.lock"
        
        with patch("app.lock.get_settings") as mock_settings:
            mock_settings.return_value.pipeline_lock_file = str(lock_file)
            
            from app.lock import is_pipeline_locked
            
            # Initially not locked
            assert is_pipeline_locked() is False
            
            # Create lock
            lock_file.write_text("locked")
            assert is_pipeline_locked() is True
            
            # Remove lock
            lock_file.unlink()
            assert is_pipeline_locked() is False


class TestDataNormalization:
    """Tests for URL and text normalization."""
    
    def test_normalize_url(self):
        """Test URL normalization."""
        from app.utils import normalize_url
        
        # Should remove tracking params
        url = "https://example.com/article?id=123&utm_source=google&utm_medium=cpc"
        normalized = normalize_url(url)
        
        assert "utm_source" not in normalized
        assert "utm_medium" not in normalized
        assert "id=123" in normalized
    
    def test_generate_dedup_key(self):
        """Test dedup key generation."""
        from app.utils import generate_dedup_key
        
        key1 = generate_dedup_key("Copper prices rise", "https://example.com/a")
        key2 = generate_dedup_key("Copper prices rise", "https://example.com/a")
        key3 = generate_dedup_key("Different title", "https://example.com/a")
        
        # Same input should give same key
        assert key1 == key2
        
        # Different input should give different key
        assert key1 != key3
    
    def test_truncate_text(self):
        """Test text truncation."""
        from app.utils import truncate_text
        
        long_text = "a" * 1000
        truncated = truncate_text(long_text, max_length=100)
        
        assert len(truncated) == 100
        
        short_text = "hello"
        not_truncated = truncate_text(short_text, max_length=100)
        
        assert not_truncated == "hello"
