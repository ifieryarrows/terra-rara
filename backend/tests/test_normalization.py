"""
Tests for data normalization and deduplication utilities.
"""

import pytest
from datetime import datetime, timezone

from app.utils import (
    normalize_whitespace,
    strip_html,
    clean_text,
    canonical_title,
    normalize_url,
    generate_dedup_key,
    truncate_text,
    safe_parse_date,
)


class TestNormalizeWhitespace:
    def test_multiple_spaces(self):
        assert normalize_whitespace("hello   world") == "hello world"
    
    def test_tabs_and_newlines(self):
        assert normalize_whitespace("hello\t\nworld") == "hello world"
    
    def test_leading_trailing(self):
        assert normalize_whitespace("  hello world  ") == "hello world"
    
    def test_empty_string(self):
        assert normalize_whitespace("") == ""
    
    def test_none_safe(self):
        assert normalize_whitespace(None) == ""


class TestStripHtml:
    def test_simple_tags(self):
        assert strip_html("<p>Hello</p>") == "Hello"
    
    def test_nested_tags(self):
        assert strip_html("<div><p>Hello <b>World</b></p></div>") == "Hello World"
    
    def test_no_html(self):
        assert strip_html("Plain text") == "Plain text"
    
    def test_empty(self):
        assert strip_html("") == ""


class TestCleanText:
    def test_combined_cleaning(self):
        result = clean_text("<p>Hello    World</p>")
        assert result == "Hello World"
    
    def test_preserves_content(self):
        result = clean_text("Copper prices rose 3.5% today")
        assert "3.5%" in result


class TestCanonicalTitle:
    def test_lowercase(self):
        assert canonical_title("HELLO World") == "hello world"
    
    def test_punctuation_removed(self):
        result = canonical_title("Hello, World! How's it going?")
        assert "," not in result
        assert "!" not in result
        assert "'" not in result
    
    def test_whitespace_normalized(self):
        result = canonical_title("Hello   World")
        assert result == "hello world"


class TestNormalizeUrl:
    def test_removes_tracking_params(self):
        url = "https://example.com/news?utm_source=twitter&id=123"
        result = normalize_url(url)
        assert "utm_source" not in result
        assert "id=123" in result
    
    def test_removes_fragment(self):
        url = "https://example.com/page#section"
        result = normalize_url(url)
        assert "#section" not in result
    
    def test_lowercase_domain(self):
        url = "https://EXAMPLE.COM/Page"
        result = normalize_url(url)
        assert "example.com" in result
    
    def test_empty_url(self):
        assert normalize_url("") == ""


class TestGenerateDedupKey:
    def test_url_based_key(self):
        key1 = generate_dedup_key(url="https://example.com/news/1")
        key2 = generate_dedup_key(url="https://example.com/news/1")
        assert key1 == key2
    
    def test_different_urls_different_keys(self):
        key1 = generate_dedup_key(url="https://example.com/news/1")
        key2 = generate_dedup_key(url="https://example.com/news/2")
        assert key1 != key2
    
    def test_content_based_fallback(self):
        key = generate_dedup_key(
            title="Copper prices rise",
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            source="Reuters"
        )
        assert len(key) == 32
    
    def test_key_length(self):
        key = generate_dedup_key(url="https://example.com/test")
        assert len(key) == 32


class TestTruncateText:
    def test_short_text_unchanged(self):
        text = "Short text"
        assert truncate_text(text, 100) == text
    
    def test_long_text_truncated(self):
        text = "This is a very long text that should be truncated"
        result = truncate_text(text, 20)
        assert len(result) <= 20
        assert result.endswith("...")
    
    def test_empty_text(self):
        assert truncate_text("", 10) == ""
    
    def test_none_text(self):
        assert truncate_text(None, 10) == ""


class TestSafeParseDate:
    def test_iso_format(self):
        result = safe_parse_date("2026-01-15T10:30:00Z")
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 15
    
    def test_date_only(self):
        result = safe_parse_date("2026-01-15")
        assert result.year == 2026
    
    def test_invalid_returns_none(self):
        result = safe_parse_date("not a date")
        assert result is None
    
    def test_empty_returns_none(self):
        result = safe_parse_date("")
        assert result is None

