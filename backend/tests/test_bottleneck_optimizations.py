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
    Session = sessionmaker(bind=engine)
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
