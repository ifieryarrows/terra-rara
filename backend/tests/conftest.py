"""
Pytest configuration and fixtures.
"""

import os
import sys
import pytest
from datetime import datetime, timezone

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "online: marks tests that require internet access (yfinance, external APIs)"
    )
    config.addinivalue_line(
        "markers", 
        "slow: marks tests that are slow to execute"
    )


@pytest.fixture
def sample_articles():
    """Sample news articles for testing."""
    return [
        {
            "title": "Copper prices surge amid supply concerns",
            "description": "Global copper prices rose 3% today...",
            "url": "https://example.com/news/copper-surge",
            "source": "Financial Times",
            "published_at": datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        },
        {
            "title": "Mining output drops in Chile",
            "description": "Chile's copper production fell...",
            "url": "https://example.com/news/chile-mining",
            "source": "Reuters",
            "published_at": datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc),
        },
        {
            "title": "Copper prices rise on supply concerns",  # Similar to first
            "description": "Copper prices increased today...",
            "url": "https://example.com/news/copper-rise",
            "source": "Bloomberg",
            "published_at": datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc),
        },
    ]


@pytest.fixture
def sample_sentiment_scores():
    """Sample sentiment scores for aggregation testing."""
    return [
        {
            "score": 0.5,
            "published_at": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
        },
        {
            "score": -0.2,
            "published_at": datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc),
        },
        {
            "score": 0.3,
            "published_at": datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc),
        },
    ]


@pytest.fixture
def sample_price_data():
    """Sample price data for feature testing."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start="2025-01-01", periods=30, freq="D")
    
    # Generate realistic price series
    np.random.seed(42)
    base_price = 4.0
    returns = np.random.normal(0, 0.02, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        "high": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        "low": prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        "close": prices,
        "volume": np.random.randint(10000, 100000, len(dates)),
    })
    
    df = df.set_index("date")
    return df

