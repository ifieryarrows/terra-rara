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


def test_new_price_bar_bypasses_previous_forecast_vintage(
    monkeypatch,
    tft_api_session_factory,
):
    Session = tft_api_session_factory
    _seed_snapshot_and_price(
        Session,
        snapshot_date="2026-09-01",
        price_date="2026-09-02",
    )
    main._tft_cache["HG=F:live:2026-09-01"] = {
        "data": {**_snapshot_payload("2026-09-01"), "source": "live"},
        "ts": datetime.now(timezone.utc),
    }

    calls: list[str] = []

    def fake_generate(_session, symbol):
        calls.append(symbol)
        return {
            **_snapshot_payload("2026-09-02"),
            "prediction": {
                **_snapshot_payload("2026-09-02")["prediction"],
                "reference_price_date": "2026-09-02",
            },
        }

    import deep_learning.inference.predictor as predictor_module

    monkeypatch.setattr(predictor_module, "generate_tft_analysis", fake_generate)

    response = TestClient(main.app).get("/api/analysis/tft/HG=F")

    assert response.status_code == 200
    assert response.json()["source"] == "live"
    assert response.json()["prediction"]["reference_price_date"] == "2026-09-02"
    assert calls == ["HG=F"]
    assert "HG=F:live:2026-09-01" not in main._tft_cache
    assert "HG=F:live:2026-09-02" in main._tft_cache


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
