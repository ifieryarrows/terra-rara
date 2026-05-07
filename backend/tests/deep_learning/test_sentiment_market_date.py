from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

from deep_learning.data.sentiment_market_date import build_market_date_sentiment_frame
from pipelines.market_calendar import assign_market_date


NY = ZoneInfo("America/New_York")


def test_friday_after_close_maps_to_monday_market_date():
    ts = datetime(2026, 5, 1, 18, 0, tzinfo=NY)
    assert assign_market_date(ts).isoformat() == "2026-05-04"


def test_saturday_article_maps_to_monday_market_date():
    ts = datetime(2026, 5, 2, 10, 0, tzinfo=NY)
    assert assign_market_date(ts).isoformat() == "2026-05-04"


def test_weekday_before_close_maps_same_market_date():
    ts = datetime(2026, 5, 5, 10, 0, tzinfo=NY)
    assert assign_market_date(ts).isoformat() == "2026-05-05"


def test_days_since_last_material_news_increments(monkeypatch):
    rows = [
        SimpleNamespace(published_at=datetime(2026, 5, 4, 10, tzinfo=NY), fetched_at=None, final_score=0.5, confidence_calibrated=0.8, relevance_score=0.8, event_type="supply_disruption"),
        SimpleNamespace(published_at=datetime(2026, 5, 5, 10, tzinfo=NY), fetched_at=None, final_score=0.1, confidence_calibrated=0.2, relevance_score=0.2, event_type="mixed_unclear"),
        SimpleNamespace(published_at=datetime(2026, 5, 6, 10, tzinfo=NY), fetched_at=None, final_score=0.1, confidence_calibrated=0.2, relevance_score=0.2, event_type="mixed_unclear"),
    ]

    class Query:
        def join(self, *args, **kwargs):
            return self

        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return rows

    class Session:
        def query(self, *args, **kwargs):
            return Query()

    out = build_market_date_sentiment_frame(Session(), pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-10"))
    assert out["days_since_last_material_news"].tolist() == [0, 1, 2]
