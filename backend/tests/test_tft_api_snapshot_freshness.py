"""Regression tests for rolling TFT forecast-vintage selection."""

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
def tft_api_session_factory(monkeypatch):
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
    main._tft_cache.clear()
    try:
        yield Session
    finally:
        main._tft_cache.clear()
        engine.dispose()


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


def _seed_snapshot_and_price(Session, *, snapshot_date: str, price_date: str) -> None:
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
            )
        )
        session.commit()


def test_tft_endpoint_serves_snapshot_and_never_invokes_inline_inference(
    monkeypatch,
    tft_api_session_factory,
):
    Session = tft_api_session_factory
    _seed_snapshot_and_price(
        Session,
        snapshot_date="2026-09-01",
        price_date="2026-09-02",
    )

    calls: list[str] = []

    def fake_generate(_session, symbol):
        calls.append(symbol)
        pytest.fail("Inline live TFT inference must never be called by HTTP handlers")

    import deep_learning.inference.predictor as predictor_module

    monkeypatch.setattr(predictor_module, "generate_tft_analysis", fake_generate)

    response = TestClient(main.app).get("/api/analysis/tft/HG=F")

    assert response.status_code == 200
    assert response.json()["source"] == "snapshot"
    assert response.json()["prediction"]["reference_price_date"] == "2026-09-01"
    assert calls == []

    response_live = TestClient(main.app).get("/api/analysis/tft/HG=F?source=live")
    assert response_live.status_code == 400


def test_matching_market_bar_keeps_snapshot_across_weekend_or_holiday(
    monkeypatch,
    tft_api_session_factory,
):
    Session = tft_api_session_factory
    _seed_snapshot_and_price(
        Session,
        snapshot_date="2020-01-03",
        price_date="2020-01-03",
    )

    import deep_learning.inference.predictor as predictor_module

    monkeypatch.setattr(
        predictor_module,
        "generate_tft_analysis",
        lambda *_args, **_kwargs: pytest.fail("matching snapshot should be reused"),
    )

    response = TestClient(main.app).get("/api/analysis/tft/HG=F")

    assert response.status_code == 200
    assert response.json()["source"] == "snapshot"
    assert response.json()["prediction"]["reference_price_date"] == "2020-01-03"


def test_consensus_reuses_matching_tft_snapshot(monkeypatch, tft_api_session_factory):
    Session = tft_api_session_factory
    _seed_snapshot_and_price(
        Session,
        snapshot_date="2020-01-03",
        price_date="2020-01-03",
    )

    calls: list[str] = []

    def fake_generate(_session, symbol):
        calls.append(symbol)
        return {"prediction": {"predicted_return_median": 0.25}}

    import deep_learning.inference.predictor as predictor_module

    monkeypatch.setattr(predictor_module, "generate_tft_analysis", fake_generate)

    response = TestClient(main.app).get("/api/analysis/consensus?symbol=HG=F")

    assert response.status_code == 200
    body = response.json()
    assert body["tft_return"] == pytest.approx(0.01)
    assert body["tft_horizon"] == "5D"
    assert body["xgb_horizon"] == "1D"
    assert body["tft_degraded"] is False
    assert calls == []


def test_consensus_handles_degraded_none_prediction_and_stale_snapshot_safely(
    tft_api_session_factory,
):
    Session = tft_api_session_factory
    # Case 1: degraded TFT snapshot with prediction=None must return 200 without AttributeError.
    with Session() as session:
        session.add(
            PriceBar(
                symbol="HG=F",
                date=datetime(2026, 9, 4, 4, 0, tzinfo=timezone.utc),
                close=6.0,
            )
        )
        # Also seed a weekend ghost bar later than Friday that must be ignored by _latest_price_bar_date
        session.add(
            PriceBar(
                symbol="HG=F",
                date=datetime(2026, 9, 6, 4, 0, tzinfo=timezone.utc),
                close=6.2,
            )
        )
        session.add(
            TFTPredictionSnapshot(
                symbol="HG=F",
                payload_json={
                    "model_state": "retrain_required",
                    "quality_state": "degraded",
                    "is_forecast_healthy": False,
                    "prediction": None,
                },
                generated_at=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
                reference_price_date="2026-09-04",
            )
        )
        session.commit()

        assert main._latest_price_bar_date(session, "HG=F").isoformat() == "2026-09-04"

    resp_degraded = TestClient(main.app).get("/api/analysis/consensus?symbol=HG=F")
    assert resp_degraded.status_code == 200
    degraded_body = resp_degraded.json()
    assert degraded_body["tft_degraded"] is True
    assert degraded_body["tft_return"] == 0.0
    assert degraded_body["tft_horizon"] == "none"

    # Case 2: healthy snapshot anchored to an older date than latest weekday PriceBar -> excluded as stale
    with Session() as session:
        session.query(TFTPredictionSnapshot).delete()
        session.add(
            TFTPredictionSnapshot(
                symbol="HG=F",
                payload_json=_snapshot_payload("2026-09-02"),
                generated_at=datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc),
                reference_price_date="2026-09-02",
            )
        )
        session.commit()

    resp_stale = TestClient(main.app).get("/api/analysis/consensus?symbol=HG=F")
    assert resp_stale.status_code == 200
    stale_body = resp_stale.json()
    assert stale_body["tft_degraded"] is True
    assert stale_body["tft_return"] == 0.0
    assert stale_body["degraded_reasons"]

    # /api/analysis/tft still returns HTTP 200 snapshot with stale metadata surfaced
    resp_tft = TestClient(main.app).get("/api/analysis/tft/HG=F")
    assert resp_tft.status_code == 200
    assert resp_tft.json()["is_stale"] is True
    assert resp_tft.json()["snapshot_invalid_reasons"]
