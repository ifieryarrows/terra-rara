"""
Unit tests for ticker canonicalization.

Tests determinism and alias resolution.
"""

import pytest
from screener.universe_builder.canonicalize import (
    canonicalize_ticker,
    extract_base_ticker,
    is_valid_ticker_format,
    get_ticker_category_hint,
)


class TestCanonicalize:
    """Tests for canonicalize_ticker function."""
    
    def test_basic_uppercase(self):
        """Tickers should be uppercased."""
        assert canonicalize_ticker("fcx") == "FCX"
        assert canonicalize_ticker("bhp") == "BHP"
    
    def test_whitespace_stripped(self):
        """Whitespace should be stripped."""
        assert canonicalize_ticker("  FCX  ") == "FCX"
        assert canonicalize_ticker("\tBHP\n") == "BHP"
    
    def test_deterministic(self):
        """Same input should always produce same output."""
        inputs = ["fcx", "FCX", "Fcx", "  fcx  ", "FCX "]
        results = [canonicalize_ticker(i) for i in inputs]
        assert all(r == "FCX" for r in results)
    
    def test_aliases(self):
        """Known aliases should be resolved."""
        assert canonicalize_ticker("COPPER") == "HG=F"
        assert canonicalize_ticker("copper") == "HG=F"
        assert canonicalize_ticker("DXY") == "DX-Y.NYB"
        assert canonicalize_ticker("SPX") == "^GSPC"
        assert canonicalize_ticker("VIX") == "^VIX"
    
    def test_exchange_suffixes(self):
        """Exchange suffixes should be preserved."""
        assert canonicalize_ticker("lun.to") == "LUN.TO"
        assert canonicalize_ticker("2899.hk") == "2899.HK"
        assert canonicalize_ticker("glen.l") == "GLEN.L"
    
    def test_futures(self):
        """Futures symbols should be normalized."""
        assert canonicalize_ticker("hg=f") == "HG=F"
        assert canonicalize_ticker("cl=F") == "CL=F"
    
    def test_indices(self):
        """Index symbols should preserve ^ prefix."""
        assert canonicalize_ticker("^gspc") == "^GSPC"
        assert canonicalize_ticker("^DJI") == "^DJI"
    
    def test_empty_invalid(self):
        """Empty/invalid inputs should return empty string."""
        assert canonicalize_ticker("") == ""
        assert canonicalize_ticker("   ") == ""
        assert canonicalize_ticker(None) == ""


class TestExtractBaseTicker:
    """Tests for extract_base_ticker function."""
    
    def test_simple(self):
        assert extract_base_ticker("FCX") == "FCX"
    
    def test_exchange_suffix(self):
        assert extract_base_ticker("LUN.TO") == "LUN"
        assert extract_base_ticker("2899.HK") == "2899"
    
    def test_futures(self):
        assert extract_base_ticker("HG=F") == "HG"
        assert extract_base_ticker("CL=F") == "CL"
    
    def test_index(self):
        assert extract_base_ticker("^GSPC") == "GSPC"


class TestIsValidTickerFormat:
    """Tests for is_valid_ticker_format function."""
    
    def test_valid(self):
        assert is_valid_ticker_format("FCX") == True
        assert is_valid_ticker_format("HG=F") == True
        assert is_valid_ticker_format("^GSPC") == True
        assert is_valid_ticker_format("2899.HK") == True
    
    def test_invalid(self):
        assert is_valid_ticker_format("") == False
        assert is_valid_ticker_format(None) == False
        assert is_valid_ticker_format("A" * 25) == False  # Too long
        assert is_valid_ticker_format("test<script>") == False
        assert is_valid_ticker_format("has space") == False


class TestGetTickerCategoryHint:
    """Tests for get_ticker_category_hint function."""
    
    def test_copper_futures(self):
        assert get_ticker_category_hint("HG=F") == "commodity_copper"
    
    def test_energy_futures(self):
        assert get_ticker_category_hint("CL=F") == "commodity_energy"
        assert get_ticker_category_hint("NG=F") == "commodity_energy"
    
    def test_precious(self):
        assert get_ticker_category_hint("GC=F") == "commodity_precious"
    
    def test_index(self):
        assert get_ticker_category_hint("^GSPC") == "index_equity"
    
    def test_regional(self):
        assert get_ticker_category_hint("LUN.TO") == "equity_canada"
        assert get_ticker_category_hint("2899.HK") == "equity_hongkong"
    
    def test_unknown(self):
        assert get_ticker_category_hint("FCX") is None
