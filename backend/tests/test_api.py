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
        from app.schemas import AnalysisReport, Influencer, DataQuality
        
        influencers = [
            Influencer(feature="HG=F_EMA_10", importance=0.15, description="Test"),
            Influencer(feature="DX-Y.NYB_ret1", importance=0.10, description="Test"),
        ]
        
        data_quality = DataQuality(
            news_count_7d=45,
            missing_days=0,
            coverage_pct=100
        )
        
        report = AnalysisReport(
            symbol="HG=F",
            current_price=4.25,
            predicted_return=0.015,
            predicted_price=4.3137,
            confidence_lower=4.20,
            confidence_upper=4.35,
            sentiment_index=0.35,
            sentiment_label="Bullish",
            top_influencers=influencers,
            data_quality=data_quality,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        
        assert report.symbol == "HG=F"
        assert report.predicted_price == 4.3137
        assert report.sentiment_label == "Bullish"
        assert len(report.top_influencers) == 2
    
    def test_sentiment_labels(self):
        """Test valid sentiment labels."""
        from app.schemas import AnalysisReport, DataQuality
        
        for label in ["Bullish", "Bearish", "Neutral"]:
            data_quality = DataQuality(
                news_count_7d=10,
                missing_days=0,
                coverage_pct=100
            )
            
            report = AnalysisReport(
                symbol="HG=F",
                current_price=4.0,
                predicted_return=0.0,
                predicted_price=4.0,
                confidence_lower=3.9,
                confidence_upper=4.1,
                sentiment_index=0.0,
                sentiment_label=label,
                top_influencers=[],
                data_quality=data_quality,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            assert report.sentiment_label == label


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
    
    def test_lock_file_creation(self, tmp_path):
        """Test that lock file is created on acquire."""
        from app.lock import PipelineLock
        
        lock_file = tmp_path / "test.lock"
        lock = PipelineLock(lock_file=str(lock_file), timeout=0)
        
        # Should acquire
        assert lock.acquire() is True
        assert lock_file.exists()
        
        # Cleanup - release doesn't delete file immediately in some implementations
        lock.release()
    
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


class TestInfluencer:
    """Tests for Influencer schema."""
    
    def test_influencer_valid(self):
        """Test valid influencer."""
        from app.schemas import Influencer
        
        inf = Influencer(
            feature="HG=F_EMA_10",
            importance=0.15,
            description="10-day EMA"
        )
        
        assert inf.feature == "HG=F_EMA_10"
        assert inf.importance == 0.15
    
    def test_influencer_importance_bounds(self):
        """Test that importance is bounded 0-1."""
        from app.schemas import Influencer
        
        # Valid bounds
        inf_low = Influencer(feature="test", importance=0.0)
        inf_high = Influencer(feature="test", importance=1.0)
        
        assert inf_low.importance == 0.0
        assert inf_high.importance == 1.0


class TestDataQuality:
    """Tests for DataQuality schema."""
    
    def test_data_quality_valid(self):
        """Test valid data quality metrics."""
        from app.schemas import DataQuality
        
        dq = DataQuality(
            news_count_7d=50,
            missing_days=2,
            coverage_pct=95
        )
        
        assert dq.news_count_7d == 50
        assert dq.missing_days == 2
        assert dq.coverage_pct == 95
    
    def test_data_quality_coverage_bounds(self):
        """Test coverage percentage bounds."""
        from app.schemas import DataQuality
        
        dq_low = DataQuality(news_count_7d=0, missing_days=0, coverage_pct=0)
        dq_high = DataQuality(news_count_7d=100, missing_days=0, coverage_pct=100)
        
        assert dq_low.coverage_pct == 0
        assert dq_high.coverage_pct == 100
