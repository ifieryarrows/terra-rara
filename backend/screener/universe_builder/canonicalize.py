"""
Ticker canonicalization for consistent symbol handling.

Normalizes tickers, handles aliases, and preserves exchange suffixes.
"""

import re
from typing import Optional


# Known aliases mapping raw inputs to canonical tickers
#
# USD PROXY POLICY:
#   - DXY/DOLLAR aliases map to DX-Y.NYB (Dollar Index futures)
#   - Default control variable is UUP (Dollar ETF) for better liquidity/coverage
#   - Use DX-Y.NYB for direct dollar index exposure if needed
#   - Both are valid USD proxies; UUP preferred for daily tradeable proxy
#
TICKER_ALIASES = {
    # Copper
    "COPPER": "HG=F",
    "COMEX_COPPER": "HG=F",
    "COMEX COPPER": "HG=F",
    "HGF": "HG=F",
    
    # Dollar Index (see USD PROXY POLICY above)
    "DXY": "DX-Y.NYB",
    "DOLLAR": "DX-Y.NYB",
    "USD_INDEX": "DX-Y.NYB",
    
    # Gold
    "GOLD": "GC=F",
    "XAU": "GC=F",
    
    # Silver
    "SILVER": "SI=F",
    "XAG": "SI=F",
    
    # Oil
    "WTI": "CL=F",
    "CRUDE": "CL=F",
    "OIL": "CL=F",
    "BRENT": "BZ=F",
    
    # Indices
    "SPX": "^GSPC",
    "SP500": "^GSPC",
    "S&P500": "^GSPC",
    "S&P 500": "^GSPC",
    "DOW": "^DJI",
    "DJIA": "^DJI",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
    
    # Common ETFs
    "SPY": "SPY",
    "QQQ": "QQQ",
}

# Known exchange suffixes to preserve
EXCHANGE_SUFFIXES = [
    ".TO",   # Toronto
    ".HK",   # Hong Kong
    ".L",    # London
    ".AX",   # Australia
    ".SI",   # Singapore
    ".T",    # Tokyo
    ".KS",   # Korea
    ".TW",   # Taiwan
    ".NS",   # India NSE
    ".BO",   # India BSE
    ".F",    # Frankfurt
    ".PA",   # Paris
    ".AS",   # Amsterdam
    ".BR",   # Brussels
    ".SW",   # Swiss
    ".MI",   # Milan
    ".MC",   # Madrid
    ".OL",   # Oslo
    ".ST",   # Stockholm
    ".CO",   # Copenhagen
    ".HE",   # Helsinki
]


def canonicalize_ticker(raw: str) -> str:
    """
    Normalize ticker to consistent format.
    
    Rules:
    1. Strip whitespace
    2. Check alias mapping first
    3. Uppercase base symbol, preserve suffix case
    4. Handle futures suffixes (=F)
    5. Handle index prefixes (^)
    
    Args:
        raw: Raw ticker string
        
    Returns:
        Canonicalized ticker string
    """
    if not raw or not isinstance(raw, str):
        return ""
    
    # Strip and basic cleanup
    ticker = raw.strip()
    
    if not ticker:
        return ""
    
    # Check aliases (case-insensitive)
    upper = ticker.upper()
    if upper in TICKER_ALIASES:
        return TICKER_ALIASES[upper]
    
    # Check if it has a known exchange suffix
    for suffix in EXCHANGE_SUFFIXES:
        if ticker.upper().endswith(suffix.upper()):
            base = ticker[:-len(suffix)]
            # Uppercase base, preserve suffix format
            return f"{base.upper()}{suffix}"
    
    # Handle futures (=F suffix)
    if "=F" in ticker.upper():
        parts = ticker.upper().split("=")
        return f"{parts[0]}=F"
    
    # Handle indices (^ prefix)
    if ticker.startswith("^"):
        return "^" + ticker[1:].upper()
    
    # Default: uppercase everything
    return ticker.upper()


def extract_base_ticker(canonical: str) -> str:
    """
    Extract base ticker without exchange suffix or special markers.
    
    Args:
        canonical: Canonicalized ticker
        
    Returns:
        Base ticker (e.g., "FCX" from "FCX", "LUN" from "LUN.TO")
    """
    if not canonical:
        return ""
    
    # Remove exchange suffixes
    for suffix in EXCHANGE_SUFFIXES:
        if canonical.endswith(suffix):
            return canonical[:-len(suffix)]
    
    # Remove futures suffix
    if canonical.endswith("=F"):
        return canonical[:-2]
    
    # Remove index prefix
    if canonical.startswith("^"):
        return canonical[1:]
    
    return canonical


def is_valid_ticker_format(ticker: str) -> bool:
    """
    Check if ticker has valid format (basic validation).
    
    Args:
        ticker: Ticker to validate
        
    Returns:
        True if format is valid
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    ticker = ticker.strip()
    
    if len(ticker) < 1 or len(ticker) > 20:
        return False
    
    # Must start with letter, ^, or digit
    if not re.match(r'^[\^a-zA-Z0-9]', ticker):
        return False
    
    # No invalid characters
    if re.search(r'[<>:"/\\|?*\s]', ticker):
        return False
    
    return True


def get_ticker_category_hint(ticker: str) -> Optional[str]:
    """
    Get category hint based on ticker format.
    
    Args:
        ticker: Canonicalized ticker
        
    Returns:
        Category hint or None
    """
    if not ticker:
        return None
    
    # Futures
    if ticker.endswith("=F"):
        if ticker.startswith("HG"):
            return "commodity_copper"
        if ticker.startswith(("CL", "BZ", "NG", "RB", "HO")):
            return "commodity_energy"
        if ticker.startswith(("GC", "SI", "PL", "PA")):
            return "commodity_precious"
        if ticker.startswith(("ALI", "ZN", "NI", "PB")):
            return "commodity_base"
        return "commodity_futures"
    
    # Indices
    if ticker.startswith("^"):
        return "index_equity"
    
    # Currency ETFs
    if ticker.startswith(("DX", "UUP", "FX")):
        return "macro_currency"
    
    # Regional exchanges
    if ticker.endswith(".TO"):
        return "equity_canada"
    if ticker.endswith(".HK"):
        return "equity_hongkong"
    if ticker.endswith(".L"):
        return "equity_london"
    if ticker.endswith(".AX"):
        return "equity_australia"
    
    return None
