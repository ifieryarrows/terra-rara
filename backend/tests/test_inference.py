"""
Tests for inference sentiment adjustment and analysis report fields.
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


class TestSentimentAdjustment:
    def test_multiplier_increases_return_when_direction_aligned(self, monkeypatch):
        from app import inference

        monkeypatch.setattr(inference, "get_settings", _fake_settings)
        adjusted, multiplier, applied, capped = inference._apply_sentiment_adjustment(
            raw_predicted_return=0.01,
            sentiment_index=0.3,
            news_count_7d=30,
        )

        assert multiplier > 1.0
        assert adjusted > 0.01
        assert applied is True
        assert capped is False

    def test_multiplier_decreases_return_when_direction_opposes(self, monkeypatch):
        from app import inference

        monkeypatch.setattr(inference, "get_settings", _fake_settings)
        adjusted, multiplier, applied, capped = inference._apply_sentiment_adjustment(
            raw_predicted_return=0.01,
            sentiment_index=-0.3,
            news_count_7d=30,
        )

        assert multiplier == pytest.approx(0.5)
        assert adjusted == pytest.approx(0.005)
        assert applied is True
        assert capped is False

    def test_tiny_signal_floor_applies_when_raw_prediction_too_small(self, monkeypatch):
        from app import inference

        monkeypatch.setattr(inference, "get_settings", _fake_settings)
        adjusted, multiplier, applied, capped = inference._apply_sentiment_adjustment(
            raw_predicted_return=0.001,
            sentiment_index=-0.25,
            news_count_7d=12,
        )

        assert multiplier <= 2.0
        assert adjusted == pytest.approx(-0.0025)
        assert applied is True
        assert capped is False

    def test_return_is_capped_to_two_percent(self, monkeypatch):
        from app import inference

        monkeypatch.setattr(inference, "get_settings", _fake_settings)
        adjusted, multiplier, applied, capped = inference._apply_sentiment_adjustment(
            raw_predicted_return=0.018,
            sentiment_index=0.35,
            news_count_7d=30,
        )

        assert multiplier > 1.0
        assert adjusted == pytest.approx(0.02)
        assert applied is True
        assert capped is True


class TestAnalysisReportFields:
    def test_generate_analysis_report_includes_sentiment_adjustment_fields(self, monkeypatch):
        from app import inference

        monkeypatch.setattr(inference, "get_settings", _fake_settings)

        class FakeModel:
            def predict(self, _dmatrix):
                return np.array([0.001], dtype=float)

        monkeypatch.setattr(inference, "load_model", lambda _symbol: FakeModel())
        monkeypatch.setattr(
            inference,
            "load_model_metadata",
            lambda _symbol: {
                "features": ["f1"],
                "importance": [{"feature": "f1", "importance": 1.0}],
                "metrics": {"target_type": "simple_return"},
            },
        )
        monkeypatch.setattr(inference, "build_features_for_prediction", lambda *_args, **_kwargs: pd.DataFrame({"f1": [1.0]}))
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
        monkeypatch.setattr(inference, "calculate_confidence_band", lambda *_args, **_kwargs: (99.0, 101.0))
        monkeypatch.setattr(inference, "get_feature_descriptions", lambda: {})
        monkeypatch.setattr(inference.xgb, "DMatrix", lambda X, feature_names=None: X)

        session = MagicMock()
        latest_bar = SimpleNamespace(close=100.0, date=datetime(2026, 1, 1, tzinfo=timezone.utc))
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = latest_bar

        report = inference.generate_analysis_report(session=session, target_symbol="HG=F")

        assert report is not None
        assert report["raw_predicted_return"] == pytest.approx(0.001)
        assert report["predicted_return"] == pytest.approx(0.0025)
        assert report["sentiment_adjustment_applied"] is True
        assert report["predicted_return_capped"] is False
        assert "sentiment_multiplier" in report
