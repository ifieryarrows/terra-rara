"""Offline tests for the multi-source Exogenous Catalyst & Shock Radar (0-72h)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.catalyst_radar import build_catalyst_radar
from app.models import NewsProcessed, NewsRaw, NewsSentimentV2


def _make_memory_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    NewsRaw.__table__.create(bind=engine)
    NewsProcessed.__table__.create(bind=engine)
    NewsSentimentV2.__table__.create(bind=engine)
    return engine, sessionmaker(bind=engine)


def _insert_scored_article(
    session,
    *,
    idx: int,
    published_at: datetime,
    publisher: str,
    event_type: str,
    impact: float,
    relevance: float = 0.90,
    confidence: float = 0.85,
    horizon_days: int = 5,
):
    raw = NewsRaw(
        url=f"https://example.com/{idx}",
        url_hash=f"uh-{idx:060d}",
        title=f"Copper event {idx}: {event_type}",
        source="newsapi",
        publisher=publisher,
        published_at=published_at,
    )
    session.add(raw)
    session.flush()
    proc = NewsProcessed(
        raw_id=raw.id,
        canonical_title=raw.title,
        canonical_title_hash=f"th-{idx:060d}",
        dedup_key=f"dk-{idx:060d}",
        language="en",
    )
    session.add(proc)
    session.flush()
    sent = NewsSentimentV2(
        news_processed_id=proc.id,
        horizon_days=horizon_days,
        label="BULLISH" if impact > 0 else ("BEARISH" if impact < 0 else "NEUTRAL"),
        impact_score_llm=impact,
        confidence_llm=confidence,
        confidence_calibrated=confidence,
        relevance_score=relevance,
        event_type=event_type,
        rule_sign=1 if impact > 0 else (-1 if impact < 0 else 0),
        final_score=impact * 0.9,
        finbert_pos=0.7 if impact > 0 else 0.1,
        finbert_neu=0.2,
        finbert_neg=0.1 if impact > 0 else 0.7,
    )
    session.add(sent)
    session.flush()


def test_quiet_when_only_low_relevance_or_non_copper():
    engine, Session = _make_memory_db()
    now = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)
    try:
        with Session() as session:
            _insert_scored_article(
                session,
                idx=1,
                published_at=now - timedelta(hours=4),
                publisher="Reuters",
                event_type="non_copper",
                impact=0.90,
                relevance=0.95,
            )
            _insert_scored_article(
                session,
                idx=2,
                published_at=now - timedelta(hours=3),
                publisher="Bloomberg",
                event_type="supply_disruption",
                impact=0.80,
                relevance=0.40,
            )
            session.commit()

            radar = build_catalyst_radar(session, now=now, horizon_days=5)
            assert radar["status"] == "QUIET"
            assert radar["cone_override_warning"] is False
            assert radar["dominant_cluster"] is None
    finally:
        engine.dispose()


def test_single_source_triggers_watch_not_catalyst_active():
    """Anti-false-alarm guard: one publisher alone cannot trigger CATALYST_ACTIVE."""
    engine, Session = _make_memory_db()
    now = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)
    try:
        with Session() as session:
            for idx in (1, 2):
                _insert_scored_article(
                    session,
                    idx=idx,
                    published_at=now - timedelta(hours=idx * 2),
                    publisher="SingleBlog",
                    event_type="supply_disruption",
                    impact=0.85,
                    relevance=0.92,
                )
            session.commit()

            radar = build_catalyst_radar(session, now=now, horizon_days=5)
            assert radar["status"] == "WATCH"
            assert radar["cone_override_warning"] is False
            assert radar["dominant_cluster"] is not None
            assert radar["dominant_cluster"]["distinct_sources"] == 1
            assert radar["dominant_cluster"]["corroborated"] is False
    finally:
        engine.dispose()


def test_multi_source_corroboration_triggers_catalyst_active_and_endpoint():
    engine, Session = _make_memory_db()
    now = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)
    try:
        with Session() as session:
            _insert_scored_article(
                session,
                idx=1,
                published_at=now - timedelta(hours=3),
                publisher="Reuters",
                event_type="policy_drag",
                impact=-0.82,
                relevance=0.94,
            )
            _insert_scored_article(
                session,
                idx=2,
                published_at=now - timedelta(hours=5),
                publisher="Bloomberg",
                event_type="policy_drag",
                impact=-0.76,
                relevance=0.91,
            )
            session.commit()

            radar = build_catalyst_radar(session, now=now, horizon_days=5)
            assert radar["status"] == "CATALYST_ACTIVE"
            assert radar["cone_override_warning"] is True
            assert "corroborated across 2 sources" in (radar["cone_warning_message"] or "")
            dom = radar["dominant_cluster"]
            assert dom["event_type"] == "policy_drag"
            assert dom["direction"] == "BEARISH"
            assert dom["article_count_24h"] == 2
            assert dom["distinct_sources"] == 2
            assert len(dom["top_headlines"]) == 2
    finally:
        engine.dispose()


def test_catalysts_api_endpoint_reads_session(monkeypatch):
    from app import main as main_module

    engine, Session = _make_memory_db()
    now = datetime.now(timezone.utc)
    try:
        with Session() as session:
            _insert_scored_article(
                session,
                idx=10,
                published_at=now - timedelta(hours=2),
                publisher="Mining.com",
                event_type="inventory_draw",
                impact=0.72,
                relevance=0.88,
            )
            session.commit()

        monkeypatch.setattr(main_module, "SessionLocal", Session)
        client = TestClient(main_module.app)
        resp = client.get("/api/news/catalysts?window_hours=72")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "WATCH"
        assert body["dominant_cluster"]["event_type"] == "inventory_draw"
    finally:
        engine.dispose()
