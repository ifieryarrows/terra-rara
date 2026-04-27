"""Canonical instrument identifiers used across API, training and UI metadata."""

TARGET_SYMBOL = "HG=F"
TARGET_DISPLAY_NAME = "COMEX Copper Futures"
TARGET_PROVIDER = "yfinance"
TARGET_TRADINGVIEW_SYMBOL = "COMEX:HG1!"
TARGET_TRADINGVIEW_EXCHANGE = "COMEX"
TARGET_TRADINGVIEW_TICKER = "HG1!"
TARGET_YAHOO_STREAM_SYMBOL = TARGET_SYMBOL

SPOT_COPPER_SYMBOLS = {"XCU/USD", "XCUUSD", "XCU=X"}
SPOT_COPPER_DISPLAY_NAME = "Copper spot (XCU/USD)"

PROVIDER_SYMBOLS = {
    "canonical": TARGET_SYMBOL,
    "yahoo": TARGET_YAHOO_STREAM_SYMBOL,
    "yfinance": TARGET_YAHOO_STREAM_SYMBOL,
    "yahoo_websocket": TARGET_YAHOO_STREAM_SYMBOL,
    "tradingview": TARGET_TRADINGVIEW_SYMBOL,
    "tradingview_ws": TARGET_TRADINGVIEW_SYMBOL,
    "tradingview_exchange": TARGET_TRADINGVIEW_EXCHANGE,
    "tradingview_ticker": TARGET_TRADINGVIEW_TICKER,
}

_COPPER_ALIASES = {
    TARGET_SYMBOL.upper(): TARGET_SYMBOL,
    TARGET_TRADINGVIEW_SYMBOL.upper(): TARGET_SYMBOL,
    TARGET_TRADINGVIEW_TICKER.upper(): TARGET_SYMBOL,
}


def canonicalize_instrument_symbol(symbol: str | None) -> str:
    """Normalize known copper aliases to the project canonical symbol."""
    if not symbol:
        return TARGET_SYMBOL
    return _COPPER_ALIASES.get(symbol.strip().upper(), symbol.strip())


def resolve_provider_symbol(symbol: str | None, provider: str) -> str:
    """Resolve the requested instrument into the provider-specific symbol format."""
    canonical = canonicalize_instrument_symbol(symbol)
    if canonical == TARGET_SYMBOL:
        return PROVIDER_SYMBOLS.get(provider.lower(), TARGET_SYMBOL)
    return canonical
