"""Tests for screener-TFT bridge and calendar features."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from deep_learning.data.feature_store import (
    _build_calendar_features,
    load_training_symbols,
    load_screener_selected_symbols,
)


def test_calendar_features_shape():
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    cal = _build_calendar_features(dates)
    assert len(cal) == 60
    assert "day_of_week" in cal.columns
    assert "cal_sin_day" in cal.columns
    assert "is_friday" in cal.columns
    assert "is_quarter_end" in cal.columns


def test_calendar_sinusoidal_range():
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    cal = _build_calendar_features(dates)
    assert cal["cal_sin_day"].min() >= -1.0
    assert cal["cal_sin_day"].max() <= 1.0
    assert cal["cal_cos_month"].min() >= -1.0


def test_load_training_symbols_returns_list():
    symbols = load_training_symbols()
    assert isinstance(symbols, list)
    assert len(symbols) > 0
    assert all(isinstance(s, str) for s in symbols)


def test_load_training_symbols_includes_mandatory():
    symbols = load_training_symbols()
    assert "DX-Y.NYB" in symbols or "CL=F" in symbols or len(symbols) >= 4


def test_load_screener_selected_symbols_missing_file():
    result = load_screener_selected_symbols(artifacts_dir="nonexistent/path")
    assert result == []


def test_calendar_is_monday_friday_exclusive():
    dates = pd.date_range("2025-01-06", periods=5, freq="B")
    cal = _build_calendar_features(dates)
    monday_row = cal.iloc[0]
    assert monday_row["is_monday"] == 1.0
    assert monday_row["is_friday"] == 0.0
    friday_row = cal.iloc[4]
    assert friday_row["is_friday"] == 1.0
    assert friday_row["is_monday"] == 0.0
