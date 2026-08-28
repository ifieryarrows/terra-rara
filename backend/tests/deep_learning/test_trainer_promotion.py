"""Regression tests for active TFT metadata promotion isolation."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from deep_learning.training.trainer import persist_promoted_tft_metadata


def _candidate_result(*, weekly_sharpe: float = 0.40) -> dict:
    return {
        "trained_at": "2026-08-26T22:45:16+00:00",
        "checkpoint_path": "/tmp/models/tft/best_tft_asro.ckpt",
        "config": {"primary_horizon_days": 5},
        "test_metrics": {
            "directional_accuracy": 0.60,
            "sharpe_ratio": 0.10,
            "variance_ratio": 1.0,
            "tail_capture_rate": 0.50,
            "quantile_crossing_rate": 0.0,
            "median_sort_gap_max": 0.0,
            "weekly_directional_accuracy": 0.55,
            "weekly_magnitude_ratio": 1.0,
            "weekly_tail_capture_rate": 0.50,
            "weekly_pi80_coverage": 0.80,
            "weekly_pi80_width_ratio": 1.0,
            "weekly_pi96_coverage": 0.96,
            "weekly_pi96_width_ratio": 1.0,
            "weekly_quantile_crossing_rate": 0.0,
            "weekly_sorted_quantile_crossing_rate": 0.0,
            "weekly_median_sort_gap_max": 0.0,
            "weekly_sample_count": 120,
            "weekly_pred_positive_rate": 0.55,
            "weekly_actual_positive_rate": 0.55,
            "weekly_sharpe_ratio": weekly_sharpe,
            "weekly_raw_magnitude_ratio": 1.0,
            "weekly_median_bound_applied_rate": 0.10,
        },
    }


def test_rejected_candidate_never_opens_db_session(monkeypatch):
    import app.db as db

    def unexpected_session():
        raise AssertionError("rejected candidate must not touch active metadata")

    monkeypatch.setattr(db, "SessionLocal", unexpected_session)

    with pytest.raises(ValueError, match="Refusing to persist rejected TFT candidate"):
        persist_promoted_tft_metadata(
            "HG=F",
            _candidate_result(weekly_sharpe=-0.45),
        )


def test_passed_candidate_updates_single_active_row(monkeypatch):
    import app.db as db

    existing = SimpleNamespace(
        symbol="HG=F",
        config_json="{}",
        metrics_json="{}",
        checkpoint_path="old.ckpt",
        trained_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
        quality_gate_passed=True,
    )

    class FakeQuery:
        def filter(self, *_args):
            return self

        def first(self):
            return existing

    class FakeSession:
        committed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def query(self, _model):
            return FakeQuery()

        def commit(self):
            self.committed = True

    session = FakeSession()
    monkeypatch.setattr(db, "SessionLocal", lambda: session)
    monkeypatch.setattr(db, "ensure_tft_model_metadata_schema", lambda: None)

    persist_promoted_tft_metadata("HG=F", _candidate_result())

    assert session.committed is True
    assert existing.quality_gate_passed is True
    assert existing.checkpoint_path.endswith("best_tft_asro.ckpt")
    assert existing.trained_at == datetime(2026, 8, 26, 22, 45, 16, tzinfo=timezone.utc)
