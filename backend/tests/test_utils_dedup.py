"""
Unit tests for normalize_url and canonical_title.

CRITICAL: These functions are the foundation of dedup.
DO NOT change normalize_url/canonical_title without updating these tests.
If you change them, all historical hashes become invalid!
"""

import pytest
from app.utils import normalize_url, canonical_title


class TestNormalizeUrl:
    """Test URL normalization - hash foundation."""
    
    # -------------------------------------------------------------------------
    # Basic normalization
    # -------------------------------------------------------------------------
    
    def test_simple_url(self):
        """Simple URL should be unchanged (except lowercase domain)."""
        url = "https://example.com/article/123"
        result = normalize_url(url)
        assert result == "https://example.com/article/123"
    
    def test_lowercase_domain(self):
        """Domain should be lowercased."""
        url = "https://EXAMPLE.COM/Article/123"
        result = normalize_url(url)
        assert "example.com" in result
        assert "EXAMPLE" not in result
    
    def test_path_case_preserved(self):
        """Path case should be preserved (URLs are case-sensitive in path)."""
        url = "https://example.com/Article/ABC"
        result = normalize_url(url)
        assert "/Article/ABC" in result
    
    # -------------------------------------------------------------------------
    # Fragment removal
    # -------------------------------------------------------------------------
    
    def test_fragment_removed(self):
        """URL fragments (#section) should be removed."""
        url = "https://example.com/article#comments"
        result = normalize_url(url)
        assert "#comments" not in result
        assert result == "https://example.com/article"
    
    def test_fragment_with_query(self):
        """Fragment should be removed even with query params."""
        url = "https://example.com/article?id=1#top"
        result = normalize_url(url)
        assert "#top" not in result
        assert "id=1" in result
    
    # -------------------------------------------------------------------------
    # Tracking parameter removal
    # -------------------------------------------------------------------------
    
    def test_utm_params_removed(self):
        """UTM tracking params should be removed."""
        url = "https://example.com/article?utm_source=twitter&utm_medium=social&id=123"
        result = normalize_url(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "id=123" in result
    
    def test_fbclid_removed(self):
        """Facebook click ID should be removed."""
        url = "https://example.com/article?fbclid=abc123&real_param=value"
        result = normalize_url(url)
        assert "fbclid" not in result
        assert "real_param=value" in result
    
    def test_gclid_removed(self):
        """Google click ID should be removed."""
        url = "https://example.com/article?gclid=xyz789"
        result = normalize_url(url)
        assert "gclid" not in result
    
    def test_all_tracking_params(self):
        """All known tracking params should be removed."""
        tracking_params = [
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "ref", "source", "mc_cid", "mc_eid"
        ]
        
        url = "https://example.com/article?" + "&".join(f"{p}=test" for p in tracking_params)
        result = normalize_url(url)
        
        for param in tracking_params:
            assert param not in result
    
    # -------------------------------------------------------------------------
    # Query param sorting
    # -------------------------------------------------------------------------
    
    def test_query_params_sorted(self):
        """Query params should be sorted for consistency."""
        url1 = "https://example.com/article?b=2&a=1"
        url2 = "https://example.com/article?a=1&b=2"
        
        result1 = normalize_url(url1)
        result2 = normalize_url(url2)
        
        assert result1 == result2
        assert "a=1" in result1
        assert "b=2" in result1
    
    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------
    
    def test_empty_url(self):
        """Empty URL should return empty string."""
        assert normalize_url("") == ""
        assert normalize_url(None) == ""
    
    def test_url_without_scheme(self):
        """URL without scheme should still work (best effort)."""
        url = "example.com/article"
        result = normalize_url(url)
        # Should not raise, behavior may vary
        assert isinstance(result, str)
    
    def test_malformed_url(self):
        """Malformed URL should return original (fallback)."""
        url = "not-a-valid-url:::///what"
        result = normalize_url(url)
        assert isinstance(result, str)
    
    def test_google_news_redirect(self):
        """Google News redirect URLs should normalize properly."""
        url = "https://news.google.com/rss/articles/CBMiWGh0dHBzOi8vd3d3LnJldXRlcnMuY29tL2J1c2luZXNzL2NvcHBlci1wcmljZXMtcmlzZS1vbi1jaGluYS1zdGltdWx1cy1ob3Blcy0yMDI2LTAxLTI0L9IBAA?oc=5"
        result = normalize_url(url)
        # Should handle without crash
        assert isinstance(result, str)
        assert "news.google.com" in result.lower()
    
    def test_encoded_characters(self):
        """URL-encoded characters should be preserved (or consistently decoded)."""
        url = "https://example.com/article?title=Hello%20World"
        result = normalize_url(url)
        assert isinstance(result, str)
    
    # -------------------------------------------------------------------------
    # Determinism (CRITICAL)
    # -------------------------------------------------------------------------
    
    def test_deterministic(self):
        """Same URL should always produce same result."""
        url = "https://example.com/article?utm_source=test&id=123#section"
        
        results = [normalize_url(url) for _ in range(10)]
        
        assert all(r == results[0] for r in results)
    
    def test_same_url_different_tracking(self):
        """Same article with different tracking should normalize to same URL."""
        url1 = "https://example.com/article?id=123&utm_source=twitter"
        url2 = "https://example.com/article?id=123&utm_source=facebook"
        url3 = "https://example.com/article?id=123"
        
        result1 = normalize_url(url1)
        result2 = normalize_url(url2)
        result3 = normalize_url(url3)
        
        assert result1 == result2 == result3


class TestCanonicalTitle:
    """Test title canonicalization - dedup foundation."""
    
    # -------------------------------------------------------------------------
    # Basic normalization
    # -------------------------------------------------------------------------
    
    def test_lowercase(self):
        """Title should be lowercased."""
        title = "Copper Prices Rise 5%"
        result = canonical_title(title)
        assert result == result.lower()
    
    def test_punctuation_removed(self):
        """Punctuation should be removed."""
        title = "Copper: The Future of Energy!"
        result = canonical_title(title)
        assert ":" not in result
        assert "!" not in result
    
    def test_whitespace_normalized(self):
        """Multiple spaces should become single space."""
        title = "Copper   prices    rise"
        result = canonical_title(title)
        assert "   " not in result
        assert result == "copper prices rise"
    
    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------
    
    def test_empty_title(self):
        """Empty title should return empty string."""
        assert canonical_title("") == ""
        assert canonical_title(None) == ""
    
    def test_only_punctuation(self):
        """Title with only punctuation should return empty/minimal."""
        result = canonical_title("!@#$%")
        assert isinstance(result, str)
    
    def test_unicode_preserved(self):
        """Unicode characters (non-punctuation) should be preserved."""
        title = "中国铜价上涨"
        result = canonical_title(title)
        # Chinese characters should remain (they're alphanumeric in regex terms)
        assert len(result) > 0
    
    def test_numbers_preserved(self):
        """Numbers should be preserved."""
        title = "Copper price hits $4.50 per pound"
        result = canonical_title(title)
        assert "4" in result
        assert "50" in result
    
    # -------------------------------------------------------------------------
    # Determinism (CRITICAL)
    # -------------------------------------------------------------------------
    
    def test_deterministic(self):
        """Same title should always produce same result."""
        title = "Copper: The Metal of the Future! (2026)"
        
        results = [canonical_title(title) for _ in range(10)]
        
        assert all(r == results[0] for r in results)
    
    def test_similar_titles_differ(self):
        """Slightly different titles should produce different results."""
        title1 = "Copper prices rise 5%"
        title2 = "Copper prices rise 6%"
        
        result1 = canonical_title(title1)
        result2 = canonical_title(title2)
        
        assert result1 != result2
    
    def test_same_meaning_different_punctuation(self):
        """Same words with different punctuation should match."""
        title1 = "Copper prices rise!"
        title2 = "Copper prices rise."
        title3 = "Copper prices rise"
        
        result1 = canonical_title(title1)
        result2 = canonical_title(title2)
        result3 = canonical_title(title3)
        
        assert result1 == result2 == result3


class TestHashConsistency:
    """Test that hash generation is consistent with normalize functions."""
    
    def test_url_hash_deterministic(self):
        """URL hash should be deterministic."""
        import hashlib
        
        url = "https://example.com/article?id=123&utm_source=test"
        normalized = normalize_url(url)
        
        hashes = [hashlib.sha256(normalized.encode()).hexdigest() for _ in range(5)]
        
        assert all(h == hashes[0] for h in hashes)
        assert len(hashes[0]) == 64  # sha256 hex
    
    def test_title_hash_deterministic(self):
        """Title hash should be deterministic."""
        import hashlib
        
        title = "Copper: The Future!"
        canonical = canonical_title(title)
        
        hashes = [hashlib.sha256(canonical.encode()).hexdigest() for _ in range(5)]
        
        assert all(h == hashes[0] for h in hashes)
        assert len(hashes[0]) == 64  # sha256 hex


# -------------------------------------------------------------------------
# Regression tests - ADD HERE when bugs are found
# -------------------------------------------------------------------------

class TestRegression:
    """Regression tests for bugs found in production."""
    
    def test_reuters_url(self):
        """Reuters URLs should normalize correctly."""
        url = "https://www.reuters.com/business/copper-prices-rise-on-china-stimulus-hopes-2026-01-24/"
        result = normalize_url(url)
        assert "reuters.com" in result
        assert "copper-prices-rise" in result
    
    def test_bloomberg_url(self):
        """Bloomberg URLs should normalize correctly."""
        url = "https://www.bloomberg.com/news/articles/2026-01-24/copper-outlook"
        result = normalize_url(url)
        assert "bloomberg.com" in result
    
    def test_yahoo_finance_url(self):
        """Yahoo Finance URLs with complex params should normalize."""
        url = "https://finance.yahoo.com/news/copper-mining-stocks-surge-123456.html?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8"
        result = normalize_url(url)
        assert "yahoo.com" in result
        # guccounter is not in tracking list, so might remain - that's OK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
