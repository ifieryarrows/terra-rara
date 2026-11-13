"""
Unit tests for bottleneck optimizations:
1. Database & News Ingestion (insert_raw_article RETURNING id, insert_raw_articles_batch)
2. Database & News Processing (batch accumulation, single flush)
3. Database & Price Ingestion (latest_finite_price_dates batch query)
4. Network (fetch_google_news_concurrent with connection pooling)
5. Model Inference (predictor num_workers=0, finbert_v2 torch.inference_mode)
"""

from __future__ import annotations

import datetime
import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import NewsProcessed, NewsRaw, PriceBar


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    NewsRaw.__table__.create(engine)
    NewsProcessed.__table__.create(engine)
    PriceBar.__table__.create(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    session = Session()
    yield session
    session.close()
    engine.dispose()


def test_insert_raw_article_returning_id_and_duplicate(db_session):
    from pipelines.ingestion.news import insert_raw_article

    run_id = uuid.uuid4()
    pub = datetime.datetime(2026, 11, 1, 10, 0, tzinfo=datetime.timezone.utc)

    # First insert
    art_id, status = insert_raw_article(
        session=db_session,
        url="https://example.com/copper-news-1",
        title="Copper Market Update",
        description="Copper prices update",
        source="google_news",
        source_feed="google_news:copper",
        published_at=pub,
        run_id=run_id,
    )
    assert status == "inserted"
    assert art_id is not None
    assert art_id > 0

    # Duplicate insert with same URL
    dup_id, dup_status = insert_raw_article(
        session=db_session,
        url="https://example.com/copper-news-1",
        title="Copper Market Update (Duplicate)",
        description="Duplicate article",
        source="google_news",
        source_feed="google_news:copper",
        published_at=pub,
        run_id=run_id,
    )
    assert dup_status == "duplicate"
    assert dup_id is None


def test_insert_raw_articles_batch(db_session):
    from pipelines.ingestion.news import insert_raw_articles_batch

    run_id = uuid.uuid4()
    pub = datetime.datetime(2026, 11, 1, 10, 0, tzinfo=datetime.timezone.utc)

    articles = [
        {"url": "https://example.com/art-1", "title": "Article 1", "description": "D1"},
        {"url": "https://example.com/art-2", "title": "Article 2", "description": "D2"},
        # Intra-batch duplicate of art-1:
        {"url": "https://example.com/art-1", "title": "Article 1 Duplicate", "description": "D1 dup"},
        {"url": "https://example.com/art-3", "title": "Article 3", "description": "D3"},
    ]

    stats = insert_raw_articles_batch(
        session=db_session,
        articles=articles,
        run_id=run_id,
        default_source="google_news",
    )
    assert stats["inserted"] == 3
    assert stats["duplicates"] == 1
    assert stats["errors"] == 0

    # Second batch with existing DB records:
    articles_2 = [
        {"url": "https://example.com/art-2", "title": "Article 2 DB Dup", "description": "D2 db dup"},
        {"url": "https://example.com/art-4", "title": "Article 4", "description": "D4"},
    ]
    stats_2 = insert_raw_articles_batch(
        session=db_session,
        articles=articles_2,
        run_id=run_id,
        default_source="google_news",
    )
    assert stats_2["inserted"] == 1
    assert stats_2["duplicates"] == 1
    assert stats_2["errors"] == 0

    # Empty batch
    assert insert_raw_articles_batch(db_session, [], run_id)["inserted"] == 0


def test_process_raw_to_processed_batch_accumulation(db_session):
    from pipelines.processing.news import process_raw_to_processed

    run_id = uuid.uuid4()
    pub = datetime.datetime(2026, 11, 1, 12, tzinfo=datetime.timezone.utc)

    db_session.add_all([
        NewsRaw(
            url=f"https://example.com/article-{i}",
            url_hash=f"{i:064x}",
            title=f"Copper market analysis {i}",
            description="Copper market supply analysis and demand trends.",
            source="google_news",
            publisher="Reuters",
            published_at=pub,
            run_id=run_id,
        )
        for i in range(10)
    ])
    # Intra-batch duplicate with same content identity (title, publisher, date)
    db_session.add(
        NewsRaw(
            url="https://example.com/article-dup",
            url_hash="f" * 64,
            title="Copper market analysis 0",
            description="Syndicated identical news from another source.",
            source="google_news",
            publisher="Reuters",
            published_at=pub,
            run_id=run_id,
        )
    )
    db_session.commit()

    stats = process_raw_to_processed(db_session, run_id, batch_size=50)
    assert stats["processed"] == 11
    assert stats["inserted"] == 10
    assert stats["duplicates"] == 1

    processed_rows = db_session.query(NewsProcessed).all()
    assert len(processed_rows) == 11
    dups = [r for r in processed_rows if r.duplicate_of_id is not None]
    assert len(dups) == 1
    assert dups[0].raw_id == 11
    assert dups[0].duplicate_of_id == 1


def test_latest_finite_price_dates_batch(db_session):
    from app.price_utils import latest_finite_price_dates

    d1 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    d2 = datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)
    d3 = datetime.datetime(2026, 1, 3, tzinfo=datetime.timezone.utc)

    db_session.add_all([
        PriceBar(symbol="HG=F", date=d1, close=4.1),
        PriceBar(symbol="HG=F", date=d2, close=4.2),
        PriceBar(symbol="HG=F", date=d3, close=0.0),  # non-positive, should be skipped
        PriceBar(symbol="ALI=F", date=d1, close=2.1),
        PriceBar(symbol="ALI=F", date=d2, close=2.2),
        PriceBar(symbol="ZN=F", date=d1, close=1.5),
    ])
    db_session.commit()

    dates = latest_finite_price_dates(db_session, ["HG=F", "ALI=F", "ZN=F", "MISSING=F"])
    assert "HG=F" in dates
    # d3 had close=0.0 so d2 is latest valid
    assert dates["HG=F"].date() == d2.date()
    assert dates["ALI=F"].date() == d2.date()
    assert dates["ZN=F"].date() == d1.date()
    assert "MISSING=F" not in dates
    assert latest_finite_price_dates(db_session, []) == {}


def test_fetch_google_news_concurrent_pools_requests():
    from app.rss_ingest import fetch_google_news_concurrent

    mock_entries = [
        {"title": "Copper rallies - Reuters", "link": "https://example.com/1", "published": "2026-11-01"},
        {"title": "Mining expansion - Bloomberg", "link": "https://example.com/2", "published": "2026-11-01"},
    ]

    with patch("app.rss_ingest.fetch_google_news") as mock_fetch:
        mock_fetch.return_value = mock_entries
        queries = ["copper deficit", "copper price", "copper mining"]
        results = fetch_google_news_concurrent(queries, max_workers=3)

        assert len(results) == 3
        for q in queries:
            assert len(results[q]) == 2
        assert mock_fetch.call_count == 3


def test_finbert_v2_inference_mode_executes():
    import torch
    from app.ai_engine import score_batch_with_finbert_v2

    articles = [
        {"id": 101, "title": "Copper surge on tight mine supply", "description": "Global deficit widens"},
        {"id": 102, "title": "Smelter capacity expands in China", "description": "Higher inventories observed"},
    ]

    mock_raw_output = [
        [{"label": "positive", "score": 0.8}, {"label": "neutral", "score": 0.15}, {"label": "negative", "score": 0.05}],
        [{"label": "negative", "score": 0.7}, {"label": "neutral", "score": 0.2}, {"label": "positive", "score": 0.1}],
    ]

    inference_mode_active = []

    class FakeTransformersPipeline:
        __module__ = "transformers.pipelines.text_classification"

        def __call__(self, texts, batch_size=16, top_k=None):
            inference_mode_active.append(torch.is_inference_mode_enabled())
            return mock_raw_output

    fake_pipe = FakeTransformersPipeline()
    with patch("app.ai_engine.get_finbert_pipeline", return_value=fake_pipe):
        res = score_batch_with_finbert_v2(articles)

    assert 101 in res
    assert 102 in res
    assert res[101]["prob_positive"] == pytest.approx(0.8)
    assert any(inference_mode_active) is True


def test_predictor_uses_num_workers_zero():
    import inspect
    from deep_learning.inference.predictor import TFTPredictor

    source = inspect.getsource(TFTPredictor.predict)
    assert "to_dataloader(train=False, batch_size=1, num_workers=0)" in source


def test_process_raw_to_processed_fallback_recovers_cleanly_on_flush_error(db_session):
    """Verify that a batch flush failure does not poison canonical_cache with uncommitted IDs."""
    from pipelines.processing.news import process_raw_to_processed

    run_id = uuid.uuid4()
    pub = datetime.datetime(2026, 11, 1, 12, tzinfo=datetime.timezone.utc)

    # 1 canonical and 1 intra-batch duplicate
    r1 = NewsRaw(
        url="https://example.com/art-a",
        url_hash="a" * 64,
        title="Copper Demand Climbs",
        publisher="Reuters",
        published_at=pub,
        run_id=run_id,
    )
    r2 = NewsRaw(
        url="https://example.com/art-b",
        url_hash="b" * 64,
        title="Copper Demand Climbs",
        publisher="Reuters",
        published_at=pub,
        run_id=run_id,
    )
    db_session.add_all([r1, r2])
    db_session.commit()

    # Simulate batch flush failing on the second flush (intra_batch_duplicates)
    orig_flush = db_session.flush
    calls = [0]

    def flush_with_second_fail():
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("Simulated intra-batch flush crash")
        return orig_flush()

    db_session.flush = flush_with_second_fail

    stats = process_raw_to_processed(db_session, run_id, batch_size=50)

    # Must cleanly recover via fallback without poisoning cache
    assert stats["processed"] == 2
    assert stats["inserted"] == 1
    assert stats["duplicates"] == 1

    processed_rows = db_session.query(NewsProcessed).order_by(NewsProcessed.id).all()
    assert len(processed_rows) == 2
    # The first article should be canonical (duplicate_of_id is None)
    assert processed_rows[0].duplicate_of_id is None
    # The second article should be duplicate pointing to the first
    assert processed_rows[1].duplicate_of_id == processed_rows[0].id


def test_process_raw_to_processed_prevents_infinite_loop_on_unhandled_failure(db_session):
    """Verify that unprocessable raw items do not trigger an infinite loop in the batch while loop."""
    from pipelines.processing.news import process_raw_to_processed

    run_id = uuid.uuid4()
    pub = datetime.datetime(2026, 11, 1, 12, tzinfo=datetime.timezone.utc)

    r = NewsRaw(
        url="https://example.com/bad-article",
        url_hash="c" * 64,
        title="Bad Article",
        publisher="Reuters",
        published_at=pub,
        run_id=run_id,
    )
    db_session.add(r)
    db_session.commit()

    # Mock process_single_raw returning None and batch flush failing
    with patch("pipelines.processing.news.process_single_raw", return_value=None):
        with patch.object(db_session, "flush", side_effect=RuntimeError("Batch error")):
            stats = process_raw_to_processed(db_session, run_id, batch_size=10)

    assert stats["processed"] == 1
    assert stats["inserted"] == 0
    assert stats["duplicates"] == 1


def test_get_db_type_with_session_get_bind():
    from app.db import get_db_type

    mock_session = MagicMock()
    mock_session.bind = None
    mock_session.get_bind.return_value.dialect.name = "postgresql"

    assert get_db_type(mock_session) == "postgresql"


def test_vectorized_market_calendar_exact_parity():
    from pipelines.market_calendar import (
        assign_market_date,
        assign_market_date_and_after_close_series,
        is_after_close_news,
    )

    timestamps = [
        datetime.datetime(2026, 3, 6, 22, 29, 59, tzinfo=datetime.timezone.utc),   # EST before 17:30 Fri
        datetime.datetime(2026, 3, 6, 22, 30, 0, tzinfo=datetime.timezone.utc),    # EST exact 17:30 Fri
        datetime.datetime(2026, 3, 6, 22, 30, 1, tzinfo=datetime.timezone.utc),    # EST after 17:30 Fri -> Mon
        datetime.datetime(2026, 3, 7, 14, 0, 0, tzinfo=datetime.timezone.utc),     # Saturday -> Mon
        datetime.datetime(2026, 3, 8, 23, 0, 0, tzinfo=datetime.timezone.utc),     # Sunday EDT transition -> Mon
        datetime.datetime(2026, 7, 15, 21, 30, 0, tzinfo=datetime.timezone.utc),   # EDT exact 17:30 Wed
        datetime.datetime(2026, 7, 15, 21, 30, 1, tzinfo=datetime.timezone.utc),   # EDT after 17:30 Wed -> Thu
        datetime.datetime(2026, 11, 6, 22, 30, 10, tzinfo=datetime.timezone.utc),  # EST after 17:30 Fri -> Mon
    ]
    md_series, ac_series = assign_market_date_and_after_close_series(timestamps)
    for idx, ts in enumerate(timestamps):
        assert md_series.iloc[idx].date() == assign_market_date(ts)
        assert int(ac_series.iloc[idx]) == int(is_after_close_news(ts))


def test_aggregate_daily_sentiment_v2_skips_unchanged_db_updates(db_session):
    from sqlalchemy import event
    from app.ai_engine import aggregate_daily_sentiment_v2
    from app.models import DailySentimentV2, NewsSentimentV2

    DailySentimentV2.__table__.create(bind=db_session.bind, checkfirst=True)
    NewsSentimentV2.__table__.create(bind=db_session.bind, checkfirst=True)

    run_id = uuid.uuid4()
    base_dt = datetime.datetime(2026, 10, 5, 14, 0, tzinfo=datetime.timezone.utc)  # Mon through Fri
    for day_offset in range(5):
        pub_dt = base_dt + datetime.timedelta(days=day_offset)
        raw = NewsRaw(
            url=f"https://example.com/news-{day_offset}",
            url_hash=f"{day_offset + 100:064x}",
            title=f"Copper supply update day {day_offset}",
            publisher="Reuters",
            published_at=pub_dt,
            fetched_at=pub_dt + datetime.timedelta(minutes=5),
            run_id=run_id,
        )
        db_session.add(raw)
        db_session.flush()
        proc = NewsProcessed(
            raw_id=raw.id,
            canonical_title=f"copper supply update day {day_offset}",
            canonical_title_hash=f"{day_offset + 200:064x}",
            cleaned_text="Copper supply tightens.",
            dedup_key=f"{day_offset + 300:064x}",
            dedup_version="content_v2",
            language="en",
            run_id=run_id,
        )
        db_session.add(proc)
        db_session.flush()
        db_session.add(
            NewsSentimentV2(
                news_processed_id=proc.id,
                horizon_days=5,
                label="BULLISH",
                impact_score_llm=0.5,
                confidence_llm=0.8,
                confidence_calibrated=0.75,
                relevance_score=0.85,
                event_type="supply_disruption",
                rule_sign=1,
                final_score=0.45 + 0.01 * day_offset,
                finbert_pos=0.7,
                finbert_neu=0.2,
                finbert_neg=0.1,
                reasoning_json="{}",
            )
        )
    db_session.commit()

    first_count = aggregate_daily_sentiment_v2(db_session)
    assert first_count == 5

    sql_statements: list[str] = []

    def _capture_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        sql_statements.append(statement.strip().upper())

    event.listen(db_session.bind, "before_cursor_execute", _capture_sql)
    try:
        second_count = aggregate_daily_sentiment_v2(db_session)
        assert second_count == 5
        update_statements = [s for s in sql_statements if s.startswith("UPDATE DAILY_SENTIMENTS_V2")]
        assert update_statements == []
    finally:
        event.remove(db_session.bind, "before_cursor_execute", _capture_sql)


def test_v2_scoring_disables_reasoning_and_skips_same_model_conflict_escalation(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from app import ai_engine

    captured_calls = []

    async def _fake_chat_completion(**kwargs):
        captured_calls.append(kwargs)
        return {
            "model": "nvidia/nemotron-3.5-super-vl:free",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '[{"id": 1, "label": "BEARISH", "impact_score": -0.60, '
                            '"confidence": 0.90, "relevance": 0.90, '
                            '"event_type": "supply_disruption", "reasoning": "Contrarian take."}]'
                        )
                    },
                }
            ],
        }

    fake_settings = SimpleNamespace(
        openrouter_api_key="sk-or-v1-test-key",
        openrouter_api_keys=["sk-or-v1-test-key"],
        resolved_scoring_fast_model="nvidia/nemotron-3.5-super-vl:free",
        resolved_scoring_reliable_model="nvidia/nemotron-3.5-super-vl:free",
        openrouter_fallback_models_list=[],
        openrouter_max_retries=1,
        openrouter_rpm=20,
        openrouter_timeout_seconds=30.0,
        openrouter_chain_deadline_seconds=60.0,
        sentiment_escalate_conflict_threshold=0.55,
    )
    monkeypatch.setattr(ai_engine, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(ai_engine, "create_chat_completion", _fake_chat_completion)

    bundle = asyncio.run(
        ai_engine.score_batch_with_llm_v2(
            [{"id": 1, "title": "Copper outage", "description": "Mine halted"}],
            horizon_days=5,
        )
    )
    assert len(captured_calls) == 1
    assert captured_calls[0]["extra_payload"] == {"reasoning": {"enabled": False, "exclude": True}}
    assert bundle["escalation_count"] == 0
    assert len(bundle["results"]) == 1
    assert bundle["results"][0]["id"] == 1


