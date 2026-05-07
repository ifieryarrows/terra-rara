"""Shared forecasting contract constants and return-space helpers."""

from __future__ import annotations

import numpy as np


FORECAST_CONTRACT_VERSION = "weekly_log_v1"
RETURN_SPACE = "simple_public_log_internal"
PUBLIC_RETURN_SPACE = "simple_return"
TARGET_RETURN_TYPE = "log_return"


def log_to_simple_return(x: float) -> float:
    """Convert a log return to a simple return for public display."""
    return float(np.exp(float(x)) - 1.0)
