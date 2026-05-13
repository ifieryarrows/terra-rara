import logging

import numpy as np
import pandas as pd

from deep_learning.data.feature_store import _sentiment_qc


def test_sentiment_qc_skips_constant_sentiment_with_warning(caplog):
    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    sentiment = pd.Series(0.0, index=idx)
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)

    with caplog.at_level(logging.WARNING):
        _sentiment_qc(sentiment, close)

    assert "sentiment has insufficient variance" in caplog.text


def test_sentiment_qc_skips_nonfinite_values_with_warning(caplog):
    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    sentiment = pd.Series(np.linspace(-0.2, 0.2, len(idx)), index=idx)
    sentiment.iloc[10] = np.inf
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)

    with caplog.at_level(logging.WARNING):
        _sentiment_qc(sentiment, close)

    assert "non-finite values detected" in caplog.text
