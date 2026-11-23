"""Isolated Long-Term Copper Outlook Engine (20D / 1M & 60D / 1Q).

Provides a parsimonious structural ensemble (Macro-Trend Ridge, Damped-Trend
Theta, Regularized Multinomial Regime Classifier, and Purged Split Conformal
Intervals) completely decoupled from the 5D TFT-ASRO checkpoints.
"""

from long_term.schemas import (
    LongTermDataWindow,
    LongTermDriverItem,
    LongTermEnsembleComponents,
    LongTermFoldMetrics,
    LongTermHorizonForecast,
    LongTermMacroPillar,
    LongTermMacroScorecard,
    LongTermOutlookResponse,
    LongTermPillarMetric,
    LongTermValidationSummary,
)

__all__ = [
    "LongTermDataWindow",
    "LongTermDriverItem",
    "LongTermEnsembleComponents",
    "LongTermFoldMetrics",
    "LongTermHorizonForecast",
    "LongTermMacroPillar",
    "LongTermMacroScorecard",
    "LongTermOutlookResponse",
    "LongTermPillarMetric",
    "LongTermValidationSummary",
]
