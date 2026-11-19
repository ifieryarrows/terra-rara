"""
Backtest module for champion/challenger comparison.
"""

from importlib import import_module

_EXPORTS = (
    "BacktestRunner",
    "BacktestConfig",
    "BacktestResult",
    "BacktestMetrics",
    "SymbolSet",
    "load_symbol_set",
)

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Lazily resolve runner exports so `python -m ...runner` avoids runpy warnings."""
    if name in _EXPORTS:
        module = import_module(f"{__name__}.runner")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
