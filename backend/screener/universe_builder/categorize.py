"""
Category assignment for symbols.

Assigns categories based on ticker patterns, source info, and heuristics.
"""

import logging
from typing import Optional

from screener.universe_builder.canonicalize import get_ticker_category_hint

logger = logging.getLogger(__name__)


# Known ticker to category mappings (curated list)
KNOWN_CATEGORIES = {
    # Copper Miners - Major
    "FCX": "miner_major",
    "BHP": "miner_major",
    "SCCO": "miner_major",
    "RIO": "miner_major",
    "TECK": "miner_major",
    "GLEN.L": "miner_major",
    "ANTO.L": "miner_major",
    "1088.HK": "miner_major",
    
    # Copper Miners - Mid
    "LUN.TO": "miner_mid",
    "IVN.TO": "miner_mid",
    "FM.TO": "miner_mid",
    "CS.TO": "miner_mid",
    "HBM.TO": "miner_mid",
    "2899.HK": "miner_mid",
    "ERO": "miner_mid",
    "NEXA": "miner_mid",
    
    # Copper Miners - Junior
    "ARIS": "miner_junior",
    "EDV.TO": "miner_junior",
    "LAC": "miner_junior",
    "NOVR": "miner_junior",
    
    # Diversified Miners
    "VALE": "miner_diversified",
    "NEM": "miner_diversified",
    "GOLD": "miner_diversified",
    "AEM": "miner_diversified",
    "WPM": "miner_diversified",
    
    # Copper ETFs
    "COPX": "etf_copper",
    "COPJ": "etf_copper",
    "CPER": "etf_copper",
    "JJC": "etf_copper",
    
    # Metal/Miner ETFs
    "XME": "etf_metals",
    "PICK": "etf_miners",
    "GDX": "etf_gold",
    "GDXJ": "etf_gold",
    "SLV": "etf_silver",
    
    # Commodity ETFs
    "DBC": "etf_commodity",
    "GSG": "etf_commodity",
    "PDBC": "etf_commodity",
    
    # Sector ETFs
    "XLB": "etf_sector",
    "XLI": "etf_sector",
    "XLF": "etf_sector",
    "XLU": "etf_sector",
    "XHB": "etf_sector",
    
    # Macro - Currency
    "DX-Y.NYB": "macro_currency",
    "UUP": "macro_currency",
    "USDU": "macro_currency",
    "FXA": "macro_currency",
    "FXC": "macro_currency",
    
    # Macro - China
    "FXI": "macro_china",
    "MCHI": "macro_china",
    "KWEB": "macro_china",
    "ASHR": "macro_china",
    "GXC": "macro_china",
    
    # Rates
    "IEF": "rates_proxy",
    "TLT": "rates_proxy",
    "SHY": "rates_proxy",
    "GOVT": "rates_proxy",
    "TIP": "rates_inflation",
    
    # Industrial Equipment
    "CAT": "industrial_equipment",
    "DE": "industrial_equipment",
    "CMI": "industrial_equipment",
    "TEX": "industrial_equipment",
    "PCAR": "industrial_equipment",
    
    # Industrial Conglomerates
    "GE": "industrial_conglom",
    "HON": "industrial_conglom",
    "MMM": "industrial_conglom",
    "EMR": "industrial_conglom",
    
    # Steel
    "X": "materials_steel",
    "NUE": "materials_steel",
    "CLF": "materials_steel",
    "STLD": "materials_steel",
    
    # Aluminum
    "AA": "materials_aluminum",
    "CENX": "materials_aluminum",
    
    # Energy
    "XOM": "energy_major",
    "CVX": "energy_major",
    "COP": "energy_major",
    "SLB": "energy_services",
    
    # Auto/EV
    "TSLA": "auto_ev",
    "RIVN": "auto_ev",
    "NIO": "auto_ev",
    "F": "auto_traditional",
    "GM": "auto_traditional",
    
    # Financial
    "JPM": "financial",
    "BAC": "financial",
    "HYG": "credit_hy",
    "JNK": "credit_hy",
    
    # Transport
    "UNP": "transport_rail",
    "CSX": "transport_rail",
    "FDX": "transport_cargo",
    "UPS": "transport_cargo",
    
    # EM
    "EEM": "em_broad",
    "VWO": "em_broad",
    "EWZ": "em_brazil",
    "EWW": "em_mexico",
    
    # Battery Metals
    "ALB": "lithium",
    "SQM": "lithium",
    "MP": "rare_earth",
    "CCJ": "uranium",
}

# Category priority for ranking/sorting
CATEGORY_PRIORITY = {
    "target": 0,
    "miner_major": 1,
    "miner_mid": 2,
    "miner_junior": 3,
    "miner_diversified": 4,
    "etf_copper": 5,
    "etf_metals": 6,
    "etf_miners": 6,
    "etf_commodity": 7,
    "macro_currency": 10,
    "macro_china": 11,
    "commodity_copper": 15,
    "commodity_energy": 16,
    "commodity_precious": 17,
    "commodity_base": 18,
    "index_equity": 20,
    "rates_proxy": 21,
    "industrial_equipment": 25,
    "materials_steel": 26,
    "materials_aluminum": 27,
    "auto_ev": 30,
    "transport_rail": 35,
    "em_broad": 40,
    "unknown": 99,
}


def assign_category(
    ticker: str,
    source_category: Optional[str] = None,
    source_tag: Optional[str] = None
) -> str:
    """
    Assign category to a ticker.
    
    Priority:
    1. Known category mapping
    2. Source-provided category (if valid)
    3. Ticker pattern heuristics
    4. "unknown"
    
    Args:
        ticker: Canonicalized ticker
        source_category: Category from source file (optional)
        source_tag: Source tag hint (optional)
        
    Returns:
        Category string
    """
    # Check known mappings first
    if ticker in KNOWN_CATEGORIES:
        return KNOWN_CATEGORIES[ticker]
    
    # Use source-provided category if valid
    if source_category and source_category.strip():
        cat = source_category.strip().lower().replace(" ", "_")
        # Validate it's not just noise
        if len(cat) >= 3 and cat not in ("none", "null", "unknown", "other"):
            return cat
    
    # Try ticker pattern heuristics
    hint = get_ticker_category_hint(ticker)
    if hint:
        return hint
    
    # Default
    return "unknown"


def categorize_batch(tickers: list[dict]) -> list[dict]:
    """
    Assign categories to a batch of tickers.
    
    Modifies ticker dicts in place, adding 'category' if missing.
    
    Args:
        tickers: List of ticker dicts
        
    Returns:
        Same list with categories assigned
    """
    for ticker_info in tickers:
        current_category = ticker_info.get("category")
        
        if not current_category:
            ticker = ticker_info.get("canonical_ticker", ticker_info.get("ticker", ""))
            source_tag = ticker_info.get("source_tag")
            
            ticker_info["category"] = assign_category(
                ticker=ticker,
                source_category=None,
                source_tag=source_tag
            )
        else:
            # Normalize existing category
            ticker_info["category"] = current_category.strip().lower().replace(" ", "_")
    
    return tickers


def get_category_priority(category: str) -> int:
    """Get priority for sorting by category."""
    return CATEGORY_PRIORITY.get(category, 99)
