"""Tests for the snapshot-only `GET /api/explain/{symbol}` endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.models import PriceBar, TFTModelMetadata, TFTPredictionSnapshot


@pytest.fixture
def tft_explain_session_factory(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in (
        PriceBar.__table__,
        TFTModelMetadata.__table__,
        TFTPredictionSnapshot.__table__,
    ):
        table.create(bind=engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(main, "SessionLocal", Session)
    try:
        yield Session
    finally:
        engine.dispose()


def _forbid_live_inference(monkeypatch) -> None:
    """Make any live-inference attempt fail the test loudly.

    `app/main.py` imports `generate_tft_analysis` from
    `deep_learning.inference.predictor` inside its live path, so patching the
    name on that module intercepts an accidental fallback.
    """
    import deep_learning.inference.predictor as predictor_module

    monkeypatch.setattr(
        predictor_module,
        "generate_tft_analysis",
        lambda *_args, **_kwargs: pytest.fail(
            "live TFT inference must never run for /api/explain"
        ),
    )


def _snapshot_payload(reference_date: str) -> dict:
    return {
        "model_state": "ok",
        "quality_state": "ok",
        "is_forecast_healthy": True,
        "prediction": {
            "reference_price_date": reference_date,
            "weekly_return": 0.01,
            "daily_forecasts": [],
        },
    }


def _explain_payload(reference_date: str = "2026-09-01") -> dict:
    return {
        "schema_version": 1,
        "symbol": "HG=F",
        "reference_price_date": reference_date,
        "generated_at": "2026-09-01T20:05:00+00:00",
        "run_id": "run-2026-09-01",
        "model": {"checkpoint": "best.ckpt", "trained_at": None},
        "horizon": {"label": "5-day cumulative log return (primary)", "days": 5},
        "variable_selection": {
            "method": "variable_selection_network",
            "encoder": [
                {"feature": "HG=F_close", "weight": 0.7},
                {"feature": "HG=F_sentiment_mean", "weight": 0.3},
            ],
            "decoder": [{"feature": "dow_sin", "weight": 1.0}],
            "static": [],
        },
        "attention": {
            "method": "temporal_fusion_transformer",
            "step": {"kind": "first_forecast_day", "horizon_index": 0},
            "window": {"start": "2026-08-04", "end": reference_date, "days": 20},
            "top_days": [{"date": reference_date, "weight": 0.5}],
            "series": [
                {"date": "2026-08-28", "weight": 0.5},
                {"date": reference_date, "weight": 0.5},
            ],
        },
        "notes": "Feature weights reflect model focus, not causal effects.",
    }


def _seed_snapshot_and_price(
    Session,
    *,
    snapshot_date: str,
    price_date: str,
    explain: dict | None = None,
) -> None:
    with Session() as session:
        session.add(
            PriceBar(
                symbol="HG=F",
                date=datetime.fromisoformat(f"{price_date}T04:00:00+00:00"),
                close=6.0,
            )
        )
        session.add(
            TFTPredictionSnapshot(
                symbol="HG=F",
                payload_json=_snapshot_payload(snapshot_date),
                generated_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                reference_price_date=snapshot_date,
                explain_json=explain,
            )
        )
        session.commit()


def test_fresh_snapshot_returns_explain_payload(monkeypatch, tft_explain_session_factory):
    Session = tft_explain_session_factory
    explain = _explain_payload()
    _seed_snapshot_and_price(
        Session,
        snapshot_date="2026-09-01",
        price_date="2026-09-01",
        explain=explain,
    )
    _forbid_live_inference(monkeypatch)

    response = TestClient(main.app).get("/api/explain/HG=F")

    assert response.status_code == 200
    assert response.json() == explain


def test_snapshot_without_explanation_returns_404(monkeypatch, tft_explain_session_factory):
    Session = tft_explain_session_factory
    _seed_snapshot_and_price(
        Session,
        snapshot_date="2026-09-01",
        price_date="2026-09-01",
        explain=None,
    )
    _forbid_live_inference(monkeypatch)

    response = TestClient(main.app).get("/api/explain/HG=F")

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Explanation not available for this snapshot. "
        "Try again after the next pipeline refresh."
    )


def test_new_price_bar_makes_snapshot_stale_without_live_inference(
    monkeypatch,
    tft_explain_session_factory,
):
    Session = tft_explain_session_factory
    _seed_snapshot_and_price(
        Session,
        snapshot_date="2026-09-01",
        price_date="2026-09-02",
        explain=_explain_payload(),
    )
    _forbid_live_inference(monkeypatch)

    response = TestClient(main.app).get("/api/explain/HG=F")

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "TFT snapshot is stale: snapshot is not anchored to latest PriceBar "
        "(snapshot=2026-09-01, latest_price_bar=2026-09-02)"
    )


def test_missing_snapshot_returns_404(monkeypatch, tft_explain_session_factory):
    Session = tft_explain_session_factory
    with Session() as session:
        session.add(
            PriceBar(
                symbol="HG=F",
                date=datetime.fromisoformat("2026-09-02T04:00:00+00:00"),
                close=6.0,
            )
        )
        session.commit()
    _forbid_live_inference(monkeypatch)

    response = TestClient(main.app).get("/api/explain/HG=F")

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "No TFT snapshot for HG=F. Explanation will be available after "
        "the next pipeline refresh."
    )
