"""
Source loaders for seed files, ETF holdings, and macro peer sets.

Handles parsing and normalization of ticker lists from various input formats.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional

from screener.universe_builder.canonicalize import canonicalize_ticker, is_valid_ticker_format
from screener.core.fingerprint import compute_file_fingerprint

logger = logging.getLogger(__name__)


# Built-in macro/commodity peer set
MACRO_PEERS = [
    # Target
    {"ticker": "HG=F", "category": "target", "source_tag": "macro_peers"},
    
    # Dollar
    {"ticker": "DX-Y.NYB", "category": "macro_currency", "source_tag": "macro_peers"},
    {"ticker": "UUP", "category": "macro_currency", "source_tag": "macro_peers"},
    
    # China
    {"ticker": "FXI", "category": "macro_china", "source_tag": "macro_peers"},
    
    # Commodities
    {"ticker": "CL=F", "category": "commodity_energy", "source_tag": "macro_peers"},
    {"ticker": "GC=F", "category": "commodity_precious", "source_tag": "macro_peers"},
    
    # Indices
    {"ticker": "^GSPC", "category": "index_equity", "source_tag": "macro_peers"},
    
    # Rates
    {"ticker": "IEF", "category": "rates_proxy", "source_tag": "macro_peers"},
]


class SourceLoadResult:
    """Result from loading a single source."""
    
    def __init__(
        self,
        source_type: str,
        path: Optional[str] = None,
        sha256: Optional[str] = None,
        embedded: bool = False,
        tickers: list[dict] = None,
        errors: list[str] = None
    ):
        self.source_type = source_type
        self.path = path
        self.sha256 = sha256
        self.embedded = embedded
        self.tickers = tickers or []
        self.errors = errors or []
    
    def to_source_info(self) -> dict:
        """Convert to SourceInfo format for output contract."""
        info = {"type": self.source_type}
        if self.path:
            info["path"] = self.path
        if self.sha256:
            info["sha256"] = self.sha256
        if self.embedded:
            info["embedded"] = True
        return info


def load_seed_csv(file_path: str | Path) -> SourceLoadResult:
    """
    Load tickers from CSV seed file.
    
    Expected columns: ticker (required), category (optional), source_tag (optional)
    Lines starting with # are treated as comments.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        SourceLoadResult with parsed tickers
    """
    path = Path(file_path)
    result = SourceLoadResult(source_type="seed_csv", path=str(path))
    
    if not path.exists():
        result.errors.append(f"File not found: {path}")
        return result
    
    try:
        result.sha256 = compute_file_fingerprint(path)
        
        with open(path, "r", encoding="utf-8") as f:
            # Filter out comment lines before parsing
            lines = [line for line in f if not line.strip().startswith("#")]
            content = "\n".join(lines)
        
        # Parse CSV
        from io import StringIO
        reader = csv.DictReader(StringIO(content))
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is 1)
            ticker_raw = row.get("ticker", "").strip()
            
            if not ticker_raw:
                continue
            
            if not is_valid_ticker_format(ticker_raw):
                result.errors.append(f"Row {row_num}: Invalid ticker format: {ticker_raw}")
                continue
            
            canonical = canonicalize_ticker(ticker_raw)
            
            result.tickers.append({
                "ticker": ticker_raw,
                "canonical_ticker": canonical,
                "category": row.get("category", "").strip() or None,
                "source_tag": row.get("source_tag", "").strip() or None,
            })
        
        logger.info(f"Loaded {len(result.tickers)} tickers from {path.name}")
        
    except Exception as e:
        result.errors.append(f"Parse error: {e}")
        logger.error(f"Failed to parse {path}: {e}")
    
    return result


def load_seed_json(file_path: str | Path) -> SourceLoadResult:
    """
    Load tickers from JSON seed file.
    
    Expected format:
    {
        "tickers": [
            {"ticker": "FCX", "category": "miner_major"},
            {"ticker": "BHP"}
        ]
    }
    
    Or simple list:
    ["FCX", "BHP", "SCCO"]
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        SourceLoadResult with parsed tickers
    """
    path = Path(file_path)
    result = SourceLoadResult(source_type="seed_json", path=str(path))
    
    if not path.exists():
        result.errors.append(f"File not found: {path}")
        return result
    
    try:
        result.sha256 = compute_file_fingerprint(path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            ticker_list = data
        elif isinstance(data, dict) and "tickers" in data:
            ticker_list = data["tickers"]
        else:
            result.errors.append("Invalid JSON format: expected list or {tickers: [...]}")
            return result
        
        for item in ticker_list:
            if isinstance(item, str):
                ticker_raw = item.strip()
                category = None
                source_tag = None
            elif isinstance(item, dict):
                ticker_raw = item.get("ticker", "").strip()
                category = item.get("category")
                source_tag = item.get("source_tag")
            else:
                result.errors.append(f"Invalid ticker entry: {item}")
                continue
            
            if not ticker_raw:
                continue
            
            if not is_valid_ticker_format(ticker_raw):
                result.errors.append(f"Invalid ticker format: {ticker_raw}")
                continue
            
            canonical = canonicalize_ticker(ticker_raw)
            
            result.tickers.append({
                "ticker": ticker_raw,
                "canonical_ticker": canonical,
                "category": category,
                "source_tag": source_tag,
            })
        
        logger.info(f"Loaded {len(result.tickers)} tickers from {path.name}")
        
    except json.JSONDecodeError as e:
        result.errors.append(f"JSON parse error: {e}")
        logger.error(f"Failed to parse {path}: {e}")
    except Exception as e:
        result.errors.append(f"Error: {e}")
        logger.error(f"Failed to load {path}: {e}")
    
    return result


def load_etf_holdings(file_path: str | Path) -> SourceLoadResult:
    """
    Load tickers from ETF holdings file.
    
    Expected format (JSON):
    {
        "etf": "COPX",
        "holdings": [
            {"ticker": "FCX", "weight": 0.15},
            {"ticker": "SCCO", "weight": 0.12}
        ]
    }
    
    Args:
        file_path: Path to holdings JSON file
        
    Returns:
        SourceLoadResult with parsed tickers
    """
    path = Path(file_path)
    result = SourceLoadResult(source_type="etf_holdings", path=str(path))
    
    if not path.exists():
        result.errors.append(f"File not found: {path}")
        return result
    
    try:
        result.sha256 = compute_file_fingerprint(path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        etf_name = data.get("etf", "unknown_etf")
        holdings = data.get("holdings", [])
        
        for holding in holdings:
            if isinstance(holding, dict):
                ticker_raw = holding.get("ticker", "").strip()
            elif isinstance(holding, str):
                ticker_raw = holding.strip()
            else:
                continue
            
            if not ticker_raw or not is_valid_ticker_format(ticker_raw):
                continue
            
            canonical = canonicalize_ticker(ticker_raw)
            
            result.tickers.append({
                "ticker": ticker_raw,
                "canonical_ticker": canonical,
                "category": f"etf_holding_{etf_name.lower()}",
                "source_tag": f"etf_{etf_name}",
            })
        
        logger.info(f"Loaded {len(result.tickers)} holdings from {etf_name}")
        
    except Exception as e:
        result.errors.append(f"Error: {e}")
        logger.error(f"Failed to load {path}: {e}")
    
    return result


def load_macro_peers() -> SourceLoadResult:
    """
    Load built-in macro peer set.
    
    Returns:
        SourceLoadResult with embedded macro tickers
    """
    result = SourceLoadResult(source_type="macro_peers", embedded=True)
    
    for peer in MACRO_PEERS:
        ticker_raw = peer["ticker"]
        canonical = canonicalize_ticker(ticker_raw)
        
        result.tickers.append({
            "ticker": ticker_raw,
            "canonical_ticker": canonical,
            "category": peer.get("category"),
            "source_tag": peer.get("source_tag", "macro_peers"),
        })
    
    logger.info(f"Loaded {len(result.tickers)} macro peers (embedded)")
    
    return result


def load_source(source_config: dict) -> SourceLoadResult:
    """
    Load tickers from a source based on config.
    
    Args:
        source_config: Dict with 'type', 'path', 'embedded' keys
        
    Returns:
        SourceLoadResult
    """
    source_type = source_config.get("type", "")
    path = source_config.get("path")
    
    if source_type == "seed_csv":
        if not path:
            return SourceLoadResult(
                source_type=source_type,
                errors=["seed_csv requires 'path'"]
            )
        return load_seed_csv(path)
    
    elif source_type == "seed_json":
        if not path:
            return SourceLoadResult(
                source_type=source_type,
                errors=["seed_json requires 'path'"]
            )
        return load_seed_json(path)
    
    elif source_type == "etf_holdings":
        if not path:
            return SourceLoadResult(
                source_type=source_type,
                errors=["etf_holdings requires 'path'"]
            )
        return load_etf_holdings(path)
    
    elif source_type == "macro_peers":
        return load_macro_peers()
    
    else:
        return SourceLoadResult(
            source_type=source_type,
            errors=[f"Unknown source type: {source_type}"]
        )


def merge_sources(results: list[SourceLoadResult]) -> list[dict]:
    """
    Merge tickers from multiple sources, collecting ALL sources per ticker.
    
    PROVENANCE: Each ticker tracks all sources it appeared in.
    Format: sources = ["seed:smoke_test.csv", "etf:copx_holdings.json", "macro_peers"]
    
    First occurrence determines category (first source wins for that).
    
    Args:
        results: List of SourceLoadResult objects
        
    Returns:
        List of unique ticker dicts with 'sources' list
    """
    seen = {}  # canonical_ticker -> ticker dict
    
    for result in results:
        # Determine source identifier
        if result.embedded:
            source_id = result.source_type  # e.g., "macro_peers"
        elif result.path:
            filename = Path(result.path).name
            source_id = f"{result.source_type}:{filename}"  # e.g., "seed_csv:smoke_test.csv"
        else:
            source_id = result.source_type
        
        for ticker_info in result.tickers:
            canonical = ticker_info["canonical_ticker"]
            
            if canonical not in seen:
                # First occurrence - create entry with sources list
                ticker_info["sources"] = [source_id]
                seen[canonical] = ticker_info
            else:
                # Already exists - append to sources list
                if source_id not in seen[canonical].get("sources", []):
                    seen[canonical].setdefault("sources", []).append(source_id)
    
    merged = list(seen.values())
    
    # Log sources distribution
    multi_source = sum(1 for t in merged if len(t.get("sources", [])) > 1)
    logger.info(f"Merged {len(merged)} unique tickers from {len(results)} sources ({multi_source} appear in multiple sources)")
    
    return merged
