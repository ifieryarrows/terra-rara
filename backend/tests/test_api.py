"""
Tests for API endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from types import SimpleNamespace
import asyncio


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""
    
    def test_health_response_structure(self):
        """Test that health response has required fields."""
        from app.schemas import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            db_type="postgresql",
            models_found=1,
            pipeline_locked=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            news_count=100,
            price_bars_count=500
        )
        
        assert response.status == "healthy"
        assert response.db_type == "postgresql"
        assert response.models_found == 1
        assert response.pipeline_locked is False
        assert response.news_count == 100
        assert response.price_bars_count == 500
    
    def test_health_status_degraded_no_models(self):
        """Test degraded status when no models found."""
        from app.schemas import HealthResponse
        
        response = HealthResponse(
            status="degraded",
            db_type="postgresql",
            models_found=0,
            pipeline_locked=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        assert response.status == "degraded"
        assert response.models_found == 0


class TestAnalysisSchema:
    """Tests for analysis report schema."""
    
    def test_analysis_report_structure(self):
        """Test AnalysisReport schema validation."""
        from app.schemas import AnalysisReport, Influencer, DataQuality
        
        influencers = [
            Influencer(feature="HG=F_EMA_10", importance=0.15, description="Test"),
            Influencer(feature="DX-Y.NYB_ret1", importance=0.10, description="Test"),
        ]
        
        data_quality = DataQuality(
            news_count_7d=45,
            missing_days=0,
            coverage_pct=100
        )
        
        report = AnalysisReport(
            symbol="HG=F",
            current_price=4.25,
            predicted_return=0.015,
            raw_predicted_return=0.011,
            sentiment_multiplier=1.35,
            sentiment_adjustment_applied=True,
            predicted_return_capped=False,
            predicted_price=4.3137,
            confidence_lower=4.20,
            confidence_upper=4.35,
            sentiment_index=0.35,
            sentiment_label="Bullish",
            top_influencers=influencers,
            data_quality=data_quality,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        
        assert report.symbol == "HG=F"
        assert report.predicted_price == 4.3137
        assert report.raw_predicted_return == 0.011
        assert report.sentiment_label == "Bullish"
        assert len(report.top_influencers) == 2
    
    def test_sentiment_labels(self):
        """Test valid sentiment labels."""
        from app.schemas import AnalysisReport, DataQuality
        
        for label in ["Bullish", "Bearish", "Neutral"]:
            data_quality = DataQuality(
                news_count_7d=10,
                missing_days=0,
                coverage_pct=100
            )
            
            report = AnalysisReport(
                symbol="HG=F",
                current_price=4.0,
                predicted_return=0.0,
                predicted_price=4.0,
                confidence_lower=3.9,
                confidence_upper=4.1,
                sentiment_index=0.0,
                sentiment_label=label,
                top_influencers=[],
                data_quality=data_quality,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            assert report.sentiment_label == label


class TestHistorySchema:
    """Tests for history response schema."""
    
    def test_history_data_point(self):
        """Test HistoryDataPoint schema."""
        from app.schemas import HistoryDataPoint
        
        point = HistoryDataPoint(
            date="2026-01-01",
            price=4.25,
            sentiment_index=0.35,
            sentiment_news_count=10,
        )
        
        assert point.date == "2026-01-01"
        assert point.price == 4.25
        assert point.sentiment_index == 0.35
        assert point.sentiment_news_count == 10
    
    def test_history_data_point_nullable_sentiment(self):
        """Test that sentiment can be None."""
        from app.schemas import HistoryDataPoint
        
        point = HistoryDataPoint(
            date="2026-01-01",
            price=4.25,
            sentiment_index=None,
            sentiment_news_count=None,
        )
        
        assert point.sentiment_index is None
        assert point.sentiment_news_count is None
    
    def test_history_response(self):
        """Test HistoryResponse schema."""
        from app.schemas import HistoryResponse, HistoryDataPoint
        
        data = [
            HistoryDataPoint(date="2026-01-01", price=4.20),
            HistoryDataPoint(date="2026-01-02", price=4.25),
        ]
        
        response = HistoryResponse(symbol="HG=F", data=data)
        
        assert response.symbol == "HG=F"
        assert len(response.data) == 2


class TestPipelineLock:
    """Tests for pipeline lock mechanism."""
    
    def test_lock_file_creation(self, tmp_path):
        """Test that lock file is created on acquire."""
        from app.lock import PipelineLock
        
        lock_file = tmp_path / "test.lock"
        lock = PipelineLock(lock_file=str(lock_file), timeout=0)
        
        # Should acquire
        assert lock.acquire() is True
        assert lock_file.exists()
        
        # Cleanup - release doesn't delete file immediately in some implementations
        lock.release()
    
    def test_lock_already_held(self, tmp_path):
        """Test that second acquire fails when lock is held."""
        from app.lock import PipelineLock
        
        lock_file = tmp_path / "test.lock"
        lock1 = PipelineLock(lock_file=str(lock_file), timeout=0)
        lock2 = PipelineLock(lock_file=str(lock_file), timeout=0)
        
        # First lock should succeed
        assert lock1.acquire() is True
        
        # Second lock should fail
        assert lock2.acquire() is False
        
        # Cleanup
        lock1.release()


class TestDataNormalization:
    """Tests for URL and text normalization."""
    
    def test_normalize_url(self):
        """Test URL normalization."""
        from app.utils import normalize_url
        
        # Should remove tracking params
        url = "https://example.com/article?id=123&utm_source=google&utm_medium=cpc"
        normalized = normalize_url(url)
        
        assert "utm_source" not in normalized
        assert "utm_medium" not in normalized
        assert "id=123" in normalized
    
    def test_generate_dedup_key(self):
        """Test dedup key generation."""
        from app.utils import generate_dedup_key
        
        key1 = generate_dedup_key("Copper prices rise", "https://example.com/a")
        key2 = generate_dedup_key("Copper prices rise", "https://example.com/a")
        key3 = generate_dedup_key("Different title", "https://example.com/a")
        
        # Same input should give same key
        assert key1 == key2
        
        # Different input should give different key
        assert key1 != key3
    
    def test_truncate_text(self):
        """Test text truncation."""
        from app.utils import truncate_text
        
        long_text = "a" * 1000
        truncated = truncate_text(long_text, max_length=100)
        
        assert len(truncated) == 100
        
        short_text = "hello"
        not_truncated = truncate_text(short_text, max_length=100)
        
        assert not_truncated == "hello"


class TestInfluencer:
    """Tests for Influencer schema."""
    
    def test_influencer_valid(self):
        """Test valid influencer."""
        from app.schemas import Influencer
        
        inf = Influencer(
            feature="HG=F_EMA_10",
            importance=0.15,
            description="10-day EMA"
        )
        
        assert inf.feature == "HG=F_EMA_10"
        assert inf.importance == 0.15
    
    def test_influencer_importance_bounds(self):
        """Test that importance is bounded 0-1."""
        from app.schemas import Influencer
        
        # Valid bounds
        inf_low = Influencer(feature="test", importance=0.0)
        inf_high = Influencer(feature="test", importance=1.0)
        
        assert inf_low.importance == 0.0
        assert inf_high.importance == 1.0


class TestDataQuality:
    """Tests for DataQuality schema."""
    
    def test_data_quality_valid(self):
        """Test valid data quality metrics."""
        from app.schemas import DataQuality
        
        dq = DataQuality(
            news_count_7d=50,
            missing_days=2,
            coverage_pct=95
        )
        
        assert dq.news_count_7d == 50
        assert dq.missing_days == 2
        assert dq.coverage_pct == 95
    
    def test_data_quality_coverage_bounds(self):
        """Test coverage percentage bounds."""
        from app.schemas import DataQuality
        
        dq_low = DataQuality(news_count_7d=0, missing_days=0, coverage_pct=0)
        dq_high = DataQuality(news_count_7d=100, missing_days=0, coverage_pct=100)
        
        assert dq_low.coverage_pct == 0
        assert dq_high.coverage_pct == 100


class TestNewsSchemas:
    """Shape contracts for the /api/news* endpoints."""

    def test_news_item_round_trip(self):
        from app.schemas import NewsItem, NewsSentimentBlock, NewsFinbertProbs

        item = NewsItem(
            id=1,
            raw_id=42,
            title="Copper supply shock lifts futures",
            description="Chile mine outage reduces near-term availability.",
            url="https://example.com/copper",
            channel="google_news",
            publisher="Reuters",
            source_feed="google_news:copper supply deficit",
            published_at="2026-04-21T10:00:00+00:00",
            fetched_at="2026-04-21T10:05:00+00:00",
            language="en",
            sentiment=NewsSentimentBlock(
                label="BULLISH",
                final_score=0.42,
                impact_score_llm=0.55,
                confidence=0.78,
                relevance=0.66,
                event_type="supply_disruption",
                finbert=NewsFinbertProbs(pos=0.7, neu=0.25, neg=0.05),
                reasoning="Mine disruption tightens supply.",
                scored_at="2026-04-21T10:06:00+00:00",
            ),
        )
        dumped = item.model_dump()
        assert dumped["publisher"] == "Reuters"
        assert dumped["channel"] == "google_news"
        assert dumped["sentiment"]["finbert"]["pos"] == 0.7
        assert dumped["sentiment"]["label"] == "BULLISH"

    def test_news_list_response_defaults(self):
        from app.schemas import NewsListResponse

        payload = NewsListResponse(
            items=[],
            total=0,
            limit=20,
            offset=0,
            has_more=False,
            generated_at="2026-04-21T10:00:00+00:00",
        )
        assert payload.items == []
        assert payload.has_more is False

    def test_news_stats_response_shape(self):
        from app.schemas import NewsStatsResponse

        stats = NewsStatsResponse(
            window_hours=24,
            total_articles=10,
            scored_articles=8,
            label_distribution={"BULLISH": 3, "BEARISH": 2, "NEUTRAL": 3},
            event_type_distribution={"supply_disruption": 3, "demand_increase": 2},
            channel_distribution={"google_news": 8, "newsapi": 2},
            top_publishers=[{"publisher": "Reuters", "count": 4, "avg_final_score": 0.31}],
            avg_final_score=0.12,
            avg_confidence=0.55,
            avg_relevance=0.48,
            generated_at="2026-04-21T10:00:00+00:00",
        )
        assert stats.total_articles == 10
        assert stats.top_publishers[0]["publisher"] == "Reuters"


class TestNewsHelpers:
    """Unit tests for the helpers that back /api/news."""

    def test_extract_publisher_from_dict_source(self):
        from app.main import _extract_publisher

        assert _extract_publisher({"source": "Reuters"}) == "Reuters"

    def test_extract_publisher_from_nested_source(self):
        from app.main import _extract_publisher

        assert _extract_publisher({"source": {"name": "Bloomberg"}}) == "Bloomberg"

    def test_extract_publisher_from_string_json(self):
        from app.main import _extract_publisher

        raw = '{"source": "Mining.com"}'
        assert _extract_publisher(raw) == "Mining.com"

    def test_extract_publisher_handles_missing(self):
        from app.main import _extract_publisher

        assert _extract_publisher(None) is None
        assert _extract_publisher({"unrelated": 1}) is None

    def test_extract_reasoning_unwraps_dict(self):
        from app.main import _extract_reasoning_text

        blob = '{"reasoning": "Supply shock.", "event_type": "supply_disruption"}'
        assert _extract_reasoning_text(blob) == "Supply shock."

    def test_extract_reasoning_handles_invalid_json(self):
        from app.main import _extract_reasoning_text

        assert _extract_reasoning_text("") is None
        assert _extract_reasoning_text("not json") == "not json"


class TestNewsStatsFilters:
    """Regression tests for filtered /api/news/stats semantics."""

    def test_news_stats_respects_min_relevance_and_other_filters(self, monkeypatch):
        from app import main as main_module

        class FakeQuery:
            def __init__(self, rows):
                self.rows = rows
                self.filters = []
                self.limit_n = None

            def join(self, *_args):
                return self

            def outerjoin(self, *_args):
                return self

            def filter(self, *criteria):
                self.filters.extend(criteria)
                return self

            def order_by(self, *_args):
                return self

            def limit(self, n):
                self.limit_n = int(n)
                return self

            def _extract_filter(self, criterion):
                left = getattr(criterion, "left", None)
                key = getattr(left, "key", None)
                right = getattr(criterion, "right", None)
                value = getattr(right, "value", None)
                op = getattr(getattr(criterion, "operator", None), "__name__", "")
                return key, op, value

            def _match(self, triple, key, op, value):
                raw, _processed, sent = triple
                if key == "published_at" and op == "ge":
                    return raw.published_at >= value
                if key == "source" and op == "eq":
                    return (raw.source or "") == value
                if key == "event_type" and op == "eq":
                    return sent is not None and (sent.event_type or "") == value
                if key == "label" and op == "eq":
                    return sent is not None and (sent.label or "") == value
                if key == "relevance_score" and op == "ge":
                    return sent is not None and sent.relevance_score is not None and float(sent.relevance_score) >= float(value)
                if key == "title" and ("like" in op):
                    needle = str(value).replace("%", "").lower()
                    return needle in (raw.title or "").lower()
                return True

            def all(self):
                rows = list(self.rows)
                for criterion in self.filters:
                    key, op, value = self._extract_filter(criterion)
                    rows = [triple for triple in rows if self._match(triple, key, op, value)]
                if self.limit_n is not None:
                    rows = rows[: self.limit_n]
                return rows

        class FakeSession:
            def __init__(self, rows):
                self.rows = rows

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def query(self, *_args):
                return FakeQuery(self.rows)

        now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)

        def make_row(title, source, publisher, label, relevance, event_type):
            raw = SimpleNamespace(
                title=title,
                source=source,
                published_at=now,
                raw_payload={"source": publisher},
            )
            processed = SimpleNamespace(id=1)
            sent = SimpleNamespace(
                label=label,
                event_type=event_type,
                relevance_score=relevance,
                final_score=0.1,
                confidence_calibrated=0.7,
            )
            return (raw, processed, sent)

        rows = [
            make_row("Copper supply squeeze", "google_news", "Reuters", "BULLISH", 0.80, "supply_disruption"),
            make_row("Copper demand softens", "newsapi", "Bloomberg", "BEARISH", 0.40, "demand_softening"),
            make_row("Mining update points to stable output", "google_news", "Mining.com", "NEUTRAL", 0.65, "production_stable"),
        ]

        monkeypatch.setattr(main_module, "SessionLocal", lambda: FakeSession(rows))
        main_module._news_stats_cache.clear()

        relaxed = asyncio.run(
            main_module.get_news_stats(
                since_hours=168,
                label="all",
                event_type="all",
                min_relevance=0.2,
                channel="all",
                publisher=None,
                search=None,
            )
        )
        strict = asyncio.run(
            main_module.get_news_stats(
                since_hours=168,
                label="all",
                event_type="all",
                min_relevance=0.6,
                channel="all",
                publisher=None,
                search=None,
            )
        )
        focused = asyncio.run(
            main_module.get_news_stats(
                since_hours=168,
                label="all",
                event_type="all",
                min_relevance=0.2,
                channel="google_news",
                publisher=None,
                search="mining",
            )
        )

        assert relaxed["total_articles"] == 3
        assert strict["total_articles"] == 2
        assert relaxed["label_distribution"]["BEARISH"] == 1
        assert strict["label_distribution"]["BEARISH"] == 0
        assert focused["total_articles"] == 1
        assert focused["channel_distribution"] == {"google_news": 1}


class TestSentimentSummary:
    """Regression tests for the DB-backed /api/sentiment/summary contract."""

    def test_sentiment_summary_filters_recent_articles_to_window(self, monkeypatch):
        from app import main as main_module

        class FakeQuery:
            def __init__(self, kind, session):
                self.kind = kind
                self.session = session
                self.filters = []

            def filter(self, *criteria):
                self.filters.extend(criteria)
                return self

            def order_by(self, *_args):
                return self

            def join(self, *_args):
                return self

            def outerjoin(self, *_args):
                return self

            def limit(self, *_args):
                return self

            def all(self):
                if self.kind == "daily":
                    return self.session.daily_rows
                if self.kind == "recent":
                    assert self.filters, "recent articles query must be window-filtered"
                    return self.session.recent_rows
                return []

            def one(self):
                if self.kind == "components":
                    return self.session.components
                if self.kind == "freshness_24h":
                    return self.session.freshness_24h
                if self.kind == "freshness_window":
                    return self.session.freshness_window
                raise AssertionError(f"unexpected one() for {self.kind}")

        class FakeSession:
            kinds = ["daily", "components", "recent", "freshness_24h", "freshness_window"]

            def __init__(self):
                now = datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc)
                self.query_index = 0
                self.queries = []
                self.daily_rows = [
                    SimpleNamespace(
                        date=datetime(2026, 4, 28, tzinfo=timezone.utc),
                        sentiment_index=-0.2,
                        news_count=13,
                        avg_confidence=0.79,
                    )
                ]
                self.components = SimpleNamespace(
                    avg_llm=0.1,
                    avg_finbert=-0.05,
                    avg_rule=0.0,
                    avg_conf=0.7,
                    avg_rel=0.6,
                    n=2,
                )
                raw = SimpleNamespace(
                    title="Windowed copper article",
                    source="google_news",
                    url="https://example.com/copper",
                    published_at=now,
                )
                processed = SimpleNamespace(canonical_title="Windowed copper article")
                score = SimpleNamespace(
                    label="BEARISH",
                    final_score=-0.4,
                    impact_score_llm=-0.5,
                    confidence_calibrated=0.8,
                    relevance_score=0.9,
                    event_type="supply_expansion",
                    finbert_pos=0.1,
                    finbert_neu=0.2,
                    finbert_neg=0.7,
                    reasoning_json=None,
                    scored_at=now,
                )
                self.recent_rows = [(raw, processed, score)]
                self.freshness_24h = SimpleNamespace(
                    oldest=now,
                    newest=now,
                    n_total=1,
                )
                self.freshness_window = SimpleNamespace(n_total=2)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def query(self, *_args):
                kind = self.kinds[self.query_index]
                self.query_index += 1
                query = FakeQuery(kind, self)
                self.queries.append(query)
                return query

        fake_session = FakeSession()
        monkeypatch.setattr(main_module, "SessionLocal", lambda: fake_session)

        result = asyncio.run(main_module.get_sentiment_summary(days=7, recent_limit=1))

        assert result["index"] == -0.2
        assert len(result["recent_articles"]) == 1
        assert fake_session.queries[2].kind == "recent"
        assert fake_session.queries[2].filters
        assert result["data_freshness"]["window_days"] == 7
        assert result["data_freshness"]["article_count_window"] == 2

