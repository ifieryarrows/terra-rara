from app.instruments import (
    TARGET_SYMBOL,
    canonicalize_instrument_symbol,
    resolve_provider_symbol,
)


def test_canonicalize_known_copper_aliases():
    assert canonicalize_instrument_symbol("HG=F") == TARGET_SYMBOL
    assert canonicalize_instrument_symbol("COMEX:HG1!") == TARGET_SYMBOL
    assert canonicalize_instrument_symbol("hg1!") == TARGET_SYMBOL


def test_resolve_provider_symbols_for_canonical_copper():
    assert resolve_provider_symbol("HG=F", "yahoo_websocket") == "HG=F"
    assert resolve_provider_symbol("HG=F", "tradingview_ws") == "COMEX:HG1!"
    assert resolve_provider_symbol("HG=F", "tradingview_exchange") == "COMEX"
    assert resolve_provider_symbol("HG=F", "tradingview_ticker") == "HG1!"
