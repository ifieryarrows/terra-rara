"""The analysis report must always carry the ranked drivers.

The API recovers feature importance from a cached copy when a snapshot has none,
but the pipeline report did not: the worker read an empty importance list from
the active artifact, so the snapshot (and every consumer of the report, starting
with the commentary) saw zero drivers and printed placeholder names.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


def _fake_settings():
    return SimpleNamespace(
        training_symbols_hash="sha256:test",
        inference_sentiment_multiplier_max=2.0,
        inference_sentiment_multiplier_min=0.5,
        inference_sentiment_news_ref=30,
        inference_sentiment_power_ref=0.20,
        inference_tiny_signal_threshold=0.0015,
        inference_tiny_signal_floor=0.0025,
        inference_return_cap=0.02,
    )


def _build_report(monkeypatch, inference, bundle_importance):
    monkeypatch.setattr(inference, "get_settings", _fake_settings)

    class FakeModel:
        def predict(self, _dmatrix):
            return np.array([0.001], dtype=float)

    monkeypatch.setattr(
        inference,
        "load_active_model_bundle",
        lambda _symbol: (
            FakeModel(),
            {
                "features": ["f1"],
                "importance": bundle_importance,
                "metrics": {"target_type": "simple_return"},
                "artifact_version": "test-artifact",
            },
        ),
    )
    monkeypatch.setattr(
        inference,
        "build_features_for_prediction",
        lambda *_args, **_kwargs: pd.DataFrame({"f1": [1.0]}),
    )
    monkeypatch.setattr(inference, "get_current_price", lambda *_args, **_kwargs: 100.0)
    monkeypatch.setattr(inference, "get_current_sentiment", lambda *_args, **_kwargs: 0.3)
    monkeypatch.setattr(
        inference,
        "get_data_quality_stats",
        lambda *_args, **_kwargs: {
            "news_count_7d": 15,
            "scored_count_7d": 15,
            "missing_days": 0,
            "coverage_pct": 100,
        },
    )
    monkeypatch.setattr(
        inference, "calculate_confidence_band", lambda *_args, **_kwargs: (99.0, 101.0)
    )
    monkeypatch.setattr(inference, "get_feature_descriptions", lambda: {})
    monkeypatch.setattr(inference.xgb, "DMatrix", lambda X, feature_names=None: X)

    session = MagicMock()
    latest_bar = SimpleNamespace(close=100.0, date=datetime(2026, 1, 1, tzinfo=timezone.utc))
    monkeypatch.setattr(inference, "latest_finite_price_bar", lambda *_args: latest_bar)

    return inference.generate_analysis_report(session=session, target_symbol="HG=F")


class TestReportDrivers:
    def test_report_falls_back_to_cached_importance_when_the_bundle_has_none(self, monkeypatch):
        from app import inference

        monkeypatch.setattr(
            inference,
            "load_cached_xgb_importance",
            lambda _session, _symbol: [{"feature": "HG=F_rsi14", "importance": 0.21}],
            raising=False,
        )

        report = _build_report(monkeypatch, inference, bundle_importance=[])

        assert report is not None
        assert [item["feature"] for item in report["top_influencers"]] == ["HG=F_rsi14"]
        assert report["top_influencers"][0]["importance"] == pytest.approx(0.21)

    def test_report_keeps_the_bundle_importance_when_the_bundle_has_it(self, monkeypatch):
        from app import inference

        monkeypatch.setattr(
            inference,
            "load_cached_xgb_importance",
            lambda _session, _symbol: [{"feature": "HG_F_sma_5", "importance": 0.9}],
            raising=False,
        )

        report = _build_report(
            monkeypatch,
            inference,
            bundle_importance=[{"feature": "HG=F_rsi14", "importance": 0.21}],
        )

        assert report is not None
        assert [item["feature"] for item in report["top_influencers"]] == ["HG=F_rsi14"]


class TestSnapshotContext:
    def test_context_is_read_from_the_snapshot_report_json(self):
        from worker.tasks import _xgb_commentary_context

        payloads = [
            {
                "artifact_version": "xgb-HG_F-20261019T044129Z",
                "sentiment_index": -0.0287,
                "sentiment_label": "Neutral",
                "top_influencers": [{"feature": "HG=F_rsi14"}],
                "data_quality": {"news_count_7d": 271},
            }
        ]

        context = _xgb_commentary_context(payloads)

        assert context["sentiment_index"] == pytest.approx(-0.0287)
        assert context["sentiment_label"] == "Neutral"
        assert context["top_influencers"] == [{"feature": "HG=F_rsi14"}]
        assert context["news_count"] == 271

    def test_context_uses_the_newest_xgb_snapshot(self):
        from worker.tasks import _xgb_commentary_context

        payloads = [
            {"model_type": "TFT-ASRO", "prediction": {}},
            {
                "artifact_version": "xgb-HG_F-20261019T044129Z",
                "sentiment_index": 0.31,
                "sentiment_label": "Bullish",
                "top_influencers": [{"feature": "HG=F_sma_5"}],
                "data_quality": {"news_count_7d": 12},
            },
        ]

        context = _xgb_commentary_context(payloads)

        assert context["sentiment_index"] == pytest.approx(0.31)
        assert context["news_count"] == 12

    def test_context_is_empty_when_no_snapshot_carries_a_report(self):
        from worker.tasks import _xgb_commentary_context

        context = _xgb_commentary_context([{"model_type": "TFT-ASRO", "prediction": {}}])

        assert context["sentiment_index"] == 0.0
        assert context["sentiment_label"] == "Neutral"
        assert context["top_influencers"] == []
        assert context["news_count"] == 0
