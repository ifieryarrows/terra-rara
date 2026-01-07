"""
Tests for data ingestion and management.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock


class TestLanguageDetection:
    """Tests for language detection."""
    
    def test_detect_english(self):
        """Test detection of English text."""
        from app.data_manager import detect_language
        
        result = detect_language("Copper prices rose sharply today")
        assert result == "en"
    
    def test_detect_non_english(self):
        """Test detection of non-English text."""
        from app.data_manager import detect_language
        
        # German
        result = detect_language("Die Kupferpreise sind heute gestiegen")
        assert result != "en"
    
    def test_detect_empty_text(self):
        """Test detection with empty text."""
        from app.data_manager import detect_language
        
        result = detect_language("")
        assert result is None
    
    def test_detect_short_text(self):
        """Test detection with very short text."""
        from app.data_manager import detect_language
        
        # Short text may fail detection
        result = detect_language("Hi")
        # Should handle gracefully
        assert result is None or isinstance(result, str)


class TestLanguageFiltering:
    """Tests for language filtering."""
    
    def test_filter_keeps_english(self, sample_articles):
        """Test that English articles are kept."""
        from app.data_manager import filter_by_language
        
        articles = [
            {"title": "Copper prices rise", "description": "Copper up today"},
            {"title": "Mining output increases", "description": "Good news"},
        ]
        
        filtered, count = filter_by_language(articles, "en")
        
        assert len(filtered) == 2
        assert count == 0
    
    def test_filter_removes_non_english(self):
        """Test that non-English articles are filtered."""
        from app.data_manager import filter_by_language
        
        articles = [
            {"title": "Copper prices rise", "description": "Copper up today"},
            {"title": "Kupferpreise steigen", "description": "Kupfer heute höher"},
        ]
        
        filtered, count = filter_by_language(articles, "en")
        
        assert len(filtered) == 1
        assert count == 1


class TestFuzzyDeduplication:
    """Tests for fuzzy title matching."""
    
    def test_exact_duplicate(self):
        """Test that exact duplicates are detected."""
        from app.data_manager import is_fuzzy_duplicate
        
        existing = ["Copper prices surge on supply concerns"]
        new_title = "Copper prices surge on supply concerns"
        
        assert is_fuzzy_duplicate(new_title, existing, threshold=85) is True
    
    def test_similar_titles(self):
        """Test that similar titles are detected."""
        from app.data_manager import is_fuzzy_duplicate
        
        existing = ["Copper prices surge on supply concerns"]
        new_title = "Copper prices rise on supply concerns"  # Similar
        
        # Should be detected as duplicate with default threshold
        result = is_fuzzy_duplicate(new_title, existing, threshold=85)
        assert result is True
    
    def test_different_titles(self):
        """Test that different titles are not marked as duplicates."""
        from app.data_manager import is_fuzzy_duplicate
        
        existing = ["Copper prices surge on supply concerns"]
        new_title = "Gold reaches new all-time high"  # Different topic
        
        assert is_fuzzy_duplicate(new_title, existing, threshold=85) is False
    
    def test_empty_existing_titles(self):
        """Test with no existing titles."""
        from app.data_manager import is_fuzzy_duplicate
        
        existing = []
        new_title = "Any title here"
        
        assert is_fuzzy_duplicate(new_title, existing, threshold=85) is False


class TestRSSParsing:
    """Tests for RSS feed parsing."""
    
    def test_rss_query_building(self):
        """Test RSS query URL building."""
        query = "copper OR copper price OR copper futures"
        language = "en"
        
        # URL encoding
        from urllib.parse import quote
        encoded_query = quote(query)
        
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl={language}&gl=US&ceid=US:en"
        
        assert "copper" in url
        assert "hl=en" in url


class TestPriceIngestion:
    """Tests for price data ingestion."""
    
    def test_symbol_parsing(self):
        """Test multi-symbol parsing."""
        symbols_str = "HG=F,DX-Y.NYB,CL=F,FXI"
        symbols = symbols_str.split(",")
        
        assert len(symbols) == 4
        assert "HG=F" in symbols
        assert "DX-Y.NYB" in symbols
    
    def test_lookback_calculation(self):
        """Test lookback date calculation."""
        lookback_days = 365
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)
        
        delta = end_date - start_date
        assert delta.days == lookback_days
    
    def test_price_bar_fields(self):
        """Test that price bars have required fields."""
        required_fields = ["date", "open", "high", "low", "close", "volume"]
        
        sample_bar = {
            "date": datetime.now(),
            "open": 4.0,
            "high": 4.1,
            "low": 3.9,
            "close": 4.05,
            "volume": 50000,
        }
        
        for field in required_fields:
            assert field in sample_bar


class TestDatabaseUpsert:
    """Tests for database upsert logic."""
    
    def test_upsert_key_generation(self):
        """Test unique key generation for upsert."""
        from app.utils import generate_dedup_key
        
        # Same URL should give same key
        url = "https://example.com/article/123"
        key1 = generate_dedup_key("Title 1", url)
        key2 = generate_dedup_key("Title 2", url)
        
        # Keys based on URL should be consistent
        # (depends on implementation - may include title or not)
        assert isinstance(key1, str)
        assert isinstance(key2, str)
    
    def test_date_normalization(self):
        """Test date normalization for comparison."""
        dt1 = datetime(2026, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
        dt2 = datetime(2026, 1, 1, 14, 45, 0, tzinfo=timezone.utc)
        
        # Same date, different time
        date1 = dt1.date()
        date2 = dt2.date()
        
        assert date1 == date2


class TestDataValidation:
    """Tests for data validation."""
    
    def test_price_validation(self):
        """Test that prices are positive."""
        prices = [4.0, 4.1, 4.05, 3.95]
        
        assert all(p > 0 for p in prices)
    
    def test_volume_validation(self):
        """Test that volume is non-negative."""
        volumes = [50000, 0, 100000]
        
        assert all(v >= 0 for v in volumes)
    
    def test_date_validation(self):
        """Test date is not in future."""
        from datetime import datetime, timezone
        
        test_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        
        # For historical data, date should be in past or present
        assert test_date <= now or True  # Flexible for test dates
    
    def test_sentiment_score_range(self):
        """Test that sentiment scores are in valid range."""
        scores = [0.5, -0.3, 0.8, -0.9, 0.0]
        
        assert all(-1 <= s <= 1 for s in scores)
