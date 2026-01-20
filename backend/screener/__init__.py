"""
Screener Module: Universe Builder + Feature Screener for copper correlation analysis.

This module provides:
- Universe building from seed sources (CSV/JSON, ETF holdings, macro peers)
- Pairwise correlation screening with IS/OOS split
- Audit-first artifact management with full data lineage
- Rate-limited, cached price fetching from yfinance

CLI Commands:
    python -m screener.universe_builder --config config/screener_config.yaml
    python -m screener.feature_screener --universe path/to/universe.json --config config.yaml
"""

__version__ = "0.1.0"
