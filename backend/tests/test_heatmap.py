from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pandas as pd
from fastapi.testclient import TestClient


def _leaf(name: str, **overrides):
    return {
        "name": name,
        "shortName": f"{name} Corporation",
        "price": 10.0,
        "changePercent": 1.0,
        "weight": 100.0,
        "group": "Copper Miners",
        "subgroup": "Major Producers",
        "instrumentType": "equity",
        "sector": "Basic Materials",
        "industry": "Copper",
        **overrides,
    }


def test_dynamic_market_hierarchy_and_stable_ids():
    from app.heatmap import _build_hierarchy

    data = [
        _leaf("FCX"),
        _leaf("LEGACY", sector="Other Equities", industry="Unclassified Equities"),
        _leaf("UNKNOWN", sector=None, industry=None, group=None, subgroup=None),
        _leaf("COPX", instrumentType="etf", subgroup="Copper ETFs"),
        _leaf("HG=F", instrumentType="future", subgroup="Base Metal Futures"),
    ]
    first = _build_hierarchy(data, view="market")
    second = _build_hierarchy(list(reversed(data)), view="market")

    assert [node["id"] for node in first["children"]] == [node["id"] for node in second["children"]]
    groups = {node["name"]: node for node in first["children"]}
    assert "Basic Materials" in groups
    assert "Copper Miners" in groups
    assert "Other Equities" in groups
    assert "Funds & ETFs" in groups
    assert "Commodities & Futures" in groups
    assert groups["Basic Materials"]["children"][0]["name"] == "Copper"
    assert groups["Copper Miners"]["children"][0]["name"] == "Major Producers"


def test_theme_view_and_leaf_fields_remain_backward_compatible():
    from app.heatmap import _build_hierarchy, flatten_heatmap_leaves

    payload = _build_hierarchy([
        _leaf("FCX", sparkline=[100, 101], instrumentType=None, logoTicker=None)
    ])
    leaf = flatten_heatmap_leaves(payload)[0]
    assert payload["name"] == "CopperMind Universe"
    assert leaf["name"] == "FCX"
    assert leaf["sparkline"] == [100, 101]
    assert leaf["logoTicker"] == "FCX"
    assert leaf["instrumentType"] == "equity"


def test_duplicate_instruments_are_removed_from_hierarchy_and_provider_universe():
    from app.heatmap import _build_hierarchy, _deduplicate_universe, flatten_heatmap_leaves

    duplicates = [
        _leaf("EWZ", id="instrument-ewz", group="Macro", subgroup="Currency"),
        _leaf("EWZ", id="instrument-ewz", group="EM & China", subgroup="Commodity EM"),
        _leaf("EWW", id="instrument-eww"),
    ]
    hierarchy = _build_hierarchy(duplicates)
    assert [leaf["name"] for leaf in flatten_heatmap_leaves(hierarchy)] == ["EWW", "EWZ"]

    universe = _deduplicate_universe([
        {"ticker": "EWZ", "category": "macro_currency"},
        {"ticker": "EWW", "category": "macro_currency"},
        {"ticker": "EWZ", "category": "em_brazil"},
    ])
    assert [row["ticker"] for row in universe] == ["EWZ", "EWW"]
    assert universe[0]["category"] == "em_brazil"


def test_low_coverage_is_rejected_to_preserve_last_good_snapshot():
    from app.heatmap import _coverage_is_healthy

    assert _coverage_is_healthy(194, 202, 194)
    assert not _coverage_is_healthy(20, 202, 194)
    assert not _coverage_is_healthy(0, 202, 0)


def test_sparkline_normalization_and_provider_error_fallback():
    from app.heatmap import _history_quotes, _history_sparklines

    index = pd.date_range("2026-01-01", periods=3)
    frame = pd.DataFrame({"Close": [10.0, 11.0, 12.0]}, index=index)
    assert _history_sparklines(frame, ["FCX"])["FCX"] == [100.0, 110.0, 120.0]
    assert _history_quotes(frame, ["FCX"])["FCX"]["price"] == 12.0
    assert round(_history_quotes(frame, ["FCX"])["FCX"]["changePercent"], 3) == 9.091
    assert _history_sparklines(pd.DataFrame(), ["FCX"]) == {}

    quarterly = pd.DataFrame(
        {"Close": [100.0 + index for index in range(65)]},
        index=pd.date_range("2026-01-01", periods=65, freq="B"),
    )
    daily_sparkline = _history_sparklines(quarterly, ["FCX"])["FCX"]
    assert len(daily_sparkline) == 65
    assert daily_sparkline[0] == 100.0
    assert daily_sparkline[-1] == 164.0

    longer_than_quarter = pd.DataFrame(
        {"Close": [100.0 + index for index in range(80)]},
        index=pd.date_range("2026-01-01", periods=80, freq="B"),
    )
    capped_sparkline = _history_sparklines(longer_than_quarter, ["FCX"])["FCX"]
    assert len(capped_sparkline) == 66
    assert capped_sparkline[0] == 100.0
    assert capped_sparkline[-1] == 179.0

    multi = pd.DataFrame(
        [[10.0, 20.0], [11.0, 18.0]],
        index=pd.date_range("2026-01-01", periods=2),
        columns=pd.MultiIndex.from_tuples([("FCX", "Close"), ("SCCO", "Close")]),
    )
    quotes = _history_quotes(multi, ["FCX", "SCCO"])
    assert quotes["FCX"]["price"] == 11.0
    assert round(quotes["SCCO"]["changePercent"], 3) == -10.0


def test_yfinance_cache_is_redirected_to_writable_temp(monkeypatch, tmp_path):
    from app import heatmap

    observed = []
    initialized = []
    cache = SimpleNamespace(
        get_tz_cache=lambda: SimpleNamespace(initialise=lambda: initialized.append("tz")),
        get_cookie_cache=lambda: SimpleNamespace(initialise=lambda: initialized.append("cookie")),
    )
    monkeypatch.setattr(heatmap, "_yfinance_cache_location", None)
    monkeypatch.setattr(heatmap.tempfile, "gettempdir", lambda: str(tmp_path))
    cache_dir = heatmap._configure_yfinance_cache(
        SimpleNamespace(set_tz_cache_location=lambda path: observed.append(path), cache=cache)
    )

    assert cache_dir == tmp_path / "coppermind-yfinance-cache"
    assert cache_dir.is_dir()
    assert observed == [str(cache_dir)]
    assert initialized == ["tz", "cookie"]


def test_metadata_refresh_is_24_hour_incremental_batch():
    from app.heatmap import HEATMAP_METADATA_BATCH_SIZE, _metadata_refresh_symbols

    now = datetime.now(timezone.utc)
    symbols = [f"SYM{index}" for index in range(20)]
    previous = {
        symbol: {"name": symbol, "metadataAsOf": (now - timedelta(hours=25)).isoformat()}
        for symbol in symbols
    }
    first = _metadata_refresh_symbols(symbols, previous, now - timedelta(hours=25), now)
    assert first == set(symbols[:HEATMAP_METADATA_BATCH_SIZE])

    for symbol in first:
        previous[symbol]["metadataAsOf"] = now.isoformat()
    second = _metadata_refresh_symbols(symbols, previous, now - timedelta(hours=25), now)
    assert second == set(symbols[HEATMAP_METADATA_BATCH_SIZE:])

    legacy = {"LEGACY": {"name": "LEGACY"}}
    assert _metadata_refresh_symbols(["LEGACY"], legacy, now, now) == set()
    pending = {"PENDING": {"name": "PENDING", "metadataAsOf": None}}
    assert _metadata_refresh_symbols(["PENDING"], pending, now, now) == {"PENDING"}


def test_category_news_matching_uses_ticker_or_company_name():
    from app.heatmap import news_match_score

    leaves = [_leaf("FCX", shortName="Freeport-McMoRan Inc")]
    assert news_match_score("FCX expands copper output", "", leaves) >= 4
    assert news_match_score("Freeport McMoRan reports results", "", leaves) >= 2
    assert news_match_score("Oil prices are unchanged", "", leaves) == 0


class _FakeQuery:
    def __init__(self, cache):
        self.cache = cache

    def first(self):
        return self.cache


class _FakeSession:
    def __init__(self, cache):
        self.cache = cache
        self.commits = 0
        self.queries = 0
        self.dirty = []
        self.new = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def query(self, _model):
        self.queries += 1
        return _FakeQuery(self.cache)

    def add(self, cache):
        self.cache = cache
        self.new = [cache]

    def flush(self):
        return None

    def commit(self):
        self.commits += 1
        self.new = []

    def rollback(self):
        return None


def test_endpoint_etag_304_gzip_and_cache_headers(monkeypatch):
    from app import main
    from app.heatmap import _build_hierarchy, invalidate_heatmap_snapshot_memo
    from app.models import HeatmapCache

    invalidate_heatmap_snapshot_memo()
    now = datetime.now(timezone.utc)
    payload = _build_hierarchy([_leaf(f"SYM{i}", shortName=f"Company number {i}") for i in range(80)])
    cache = HeatmapCache(payload_json=payload, cached_at=now, expires_at=now + timedelta(minutes=10))
    session = _FakeSession(cache)
    monkeypatch.setattr(main, "SessionLocal", lambda: session)
    client = TestClient(main.app)

    response = client.get("/api/market-heatmap?view=market", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers["x-heatmap-cache"] == "fresh"
    assert response.headers["x-heatmap-memo"] == "miss"
    assert "db;dur=" in response.headers["server-timing"]
    assert response.headers["content-encoding"] == "gzip"
    assert int(response.headers["content-length"]) < len(response.content) * 0.4
    etag = response.headers["etag"]

    not_modified = client.get(
        "/api/market-heatmap?view=market",
        headers={"If-None-Match": etag, "Accept-Encoding": "identity"},
    )
    assert not_modified.status_code == 304
    assert not_modified.content == b""
    assert not_modified.headers["x-heatmap-memo"] == "hit"
    assert session.queries == 1
    invalidate_heatmap_snapshot_memo()


def test_empty_cache_marks_refresh_before_background_execution(monkeypatch):
    from app import heatmap, main
    from app.models import HeatmapCache

    heatmap.invalidate_heatmap_snapshot_memo()
    now = datetime.now(timezone.utc)
    cache = HeatmapCache(payload_json={}, cached_at=now, expires_at=now)
    session = _FakeSession(cache)
    observed = []
    monkeypatch.setattr(main, "SessionLocal", lambda: session)
    monkeypatch.setattr(heatmap, "refresh_market_heatmap", lambda: observed.append(cache.refresh_started_at is not None))

    response = TestClient(main.app).get("/api/market-heatmap?view=market")
    assert response.status_code == 200
    assert response.json()["_meta"]["cache_state"] == "empty"
    assert session.commits == 1
    assert observed == [True]
    heatmap.invalidate_heatmap_snapshot_memo()


def test_stale_cache_exposes_refresh_error_and_revalidates(monkeypatch):
    from app import heatmap, main
    from app.heatmap import _build_hierarchy
    from app.models import HeatmapCache

    heatmap.invalidate_heatmap_snapshot_memo()
    now = datetime.now(timezone.utc)
    payload = _build_hierarchy([_leaf("FCX")])
    cache = HeatmapCache(
        payload_json=payload,
        cached_at=now - timedelta(minutes=20),
        expires_at=now - timedelta(minutes=5),
        refresh_error="provider timeout",
    )
    session = _FakeSession(cache)
    observed = []
    monkeypatch.setattr(main, "SessionLocal", lambda: session)
    monkeypatch.setattr(heatmap, "refresh_market_heatmap", lambda: observed.append(True))

    response = TestClient(main.app).get("/api/market-heatmap?view=market")
    assert response.status_code == 200
    assert response.headers["x-heatmap-cache"] == "refreshing"
    assert response.headers["cache-control"] == "public, max-age=0, stale-while-revalidate=30"
    assert response.json()["_meta"]["refresh_error"] == "provider timeout"
    assert response.json()["_meta"]["is_stale"] is True
    assert observed == [True]
    heatmap.invalidate_heatmap_snapshot_memo()


def test_refresh_preserves_last_good_snapshot_on_unhealthy_coverage(monkeypatch):
    from app import db, heatmap
    from app.heatmap import _build_hierarchy
    from app.models import HeatmapCache

    now = datetime.now(timezone.utc)
    payload = _build_hierarchy([_leaf("FCX")])
    cache = HeatmapCache(payload_json=payload, cached_at=now, expires_at=now + timedelta(minutes=5))
    session = _FakeSession(cache)
    monkeypatch.setattr(db, "SessionLocal", lambda: session)
    monkeypatch.setattr(db, "get_db_type", lambda: "sqlite")
    monkeypatch.setattr(heatmap, "_load_universe", lambda: [
        {"ticker": "FCX", "category": "miner_major", "source_tag": "test"},
        {"ticker": "SCCO", "category": "miner_major", "source_tag": "test"},
    ])
    fake_yfinance = SimpleNamespace(
        set_tz_cache_location=lambda _path: None,
        cache=SimpleNamespace(
            get_tz_cache=lambda: SimpleNamespace(initialise=lambda: None),
            get_cookie_cache=lambda: SimpleNamespace(initialise=lambda: None),
        ),
        download=lambda **_kwargs: pd.DataFrame(),
        Tickers=lambda symbols: SimpleNamespace(
            tickers={symbol: SimpleNamespace(info={}) for symbol in symbols.split()}
        ),
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    heatmap.refresh_market_heatmap()
    assert cache.payload_json == payload
    assert cache.refresh_started_at is None
    assert "unhealthy symbol coverage" in cache.refresh_error
    assert cache.expires_at > datetime.now(timezone.utc) + timedelta(seconds=50)


def test_refresh_uses_batched_history_when_info_has_no_quotes(monkeypatch):
    from app import db, heatmap
    from app.heatmap import _build_hierarchy, flatten_heatmap_leaves
    from app.models import HeatmapCache

    now = datetime.now(timezone.utc)
    payload = _build_hierarchy([_leaf("FCX", weight=500.0, weightLabel="Market Cap")])
    cache = HeatmapCache(payload_json=payload, cached_at=now, expires_at=now + timedelta(minutes=5))
    session = _FakeSession(cache)
    monkeypatch.setattr(db, "SessionLocal", lambda: session)
    monkeypatch.setattr(db, "get_db_type", lambda: "sqlite")
    monkeypatch.setattr(heatmap, "_load_universe", lambda: [
        {"ticker": "FCX", "category": "miner_major", "source_tag": "test"},
        {"ticker": "SCCO", "category": "miner_major", "source_tag": "test"},
    ])
    history = pd.DataFrame(
        [[10.0, 20.0], [11.0, 18.0]],
        index=pd.date_range("2026-01-01", periods=2),
        columns=pd.MultiIndex.from_tuples([("FCX", "Close"), ("SCCO", "Close")]),
    )
    download_args = {}

    def download_history(**kwargs):
        download_args.update(kwargs)
        return history

    fake_yfinance = SimpleNamespace(
        set_tz_cache_location=lambda _path: None,
        cache=SimpleNamespace(
            get_tz_cache=lambda: SimpleNamespace(initialise=lambda: None),
            get_cookie_cache=lambda: SimpleNamespace(initialise=lambda: None),
        ),
        download=download_history,
        Tickers=lambda symbols: SimpleNamespace(
            tickers={symbol: SimpleNamespace(info={}) for symbol in symbols.split()}
        ),
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    heatmap.refresh_market_heatmap()

    leaves = {leaf["name"]: leaf for leaf in flatten_heatmap_leaves(cache.payload_json)}
    assert set(leaves) == {"FCX", "SCCO"}
    assert leaves["FCX"]["price"] == 11.0
    assert leaves["FCX"]["changePercent"] == 10.0
    assert leaves["FCX"]["weight"] == 500.0
    assert leaves["FCX"]["shortName"] == "FCX Corporation"
    assert leaves["SCCO"]["price"] == 18.0
    assert leaves["SCCO"]["sector"] is None
    assert download_args["period"] == "3mo"
    assert download_args["interval"] == "1d"
    assert cache.refresh_error is None


def test_refresh_error_backoff_serves_stale_without_retry_storm(monkeypatch):
    from app import heatmap, main
    from app.heatmap import _build_hierarchy
    from app.models import HeatmapCache

    heatmap.invalidate_heatmap_snapshot_memo()
    now = datetime.now(timezone.utc)
    cache = HeatmapCache(
        payload_json=_build_hierarchy([_leaf("FCX")]),
        cached_at=now - timedelta(minutes=20),
        expires_at=now + timedelta(seconds=55),
        refresh_error="provider unavailable",
    )
    session = _FakeSession(cache)
    observed = []
    monkeypatch.setattr(main, "SessionLocal", lambda: session)
    monkeypatch.setattr(heatmap, "refresh_market_heatmap", lambda: observed.append(True))

    response = TestClient(main.app).get("/api/market-heatmap?view=market")
    assert response.status_code == 200
    assert response.headers["x-heatmap-cache"] == "stale"
    assert response.json()["_meta"]["refresh_in_progress"] is False
    assert response.json()["_meta"]["refresh_error"] == "provider unavailable"
    assert observed == []
    heatmap.invalidate_heatmap_snapshot_memo()


def test_postgres_advisory_lock_deduplicates_refresh(monkeypatch):
    from adapters.db import lock
    from app import db, heatmap

    connection = SimpleNamespace(closed=False)
    connection.close = lambda: setattr(connection, "closed", True)
    monkeypatch.setattr(db, "get_db_type", lambda: "postgresql")
    monkeypatch.setattr(db, "get_engine", lambda: SimpleNamespace(connect=lambda: connection))
    monkeypatch.setattr(db, "SessionLocal", Mock(side_effect=AssertionError("DB refresh should not start")))
    monkeypatch.setattr(lock, "try_acquire_lock", lambda *_args: False)
    release = Mock()
    monkeypatch.setattr(lock, "release_lock", release)

    heatmap.refresh_market_heatmap()
    assert connection.closed is True
    release.assert_not_called()


class _FakeRowsQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *_args):
        return self

    def order_by(self, *_args):
        return self

    def limit(self, *_args):
        return self

    def all(self):
        return self.rows


def test_category_context_accepts_only_cached_ids_and_returns_matching_news(monkeypatch):
    from app import main
    from app.heatmap import _build_hierarchy, hierarchy_for_view
    from app.models import HeatmapCache

    now = datetime.now(timezone.utc)
    payload = _build_hierarchy([
        _leaf("FCX", shortName="Freeport McMoRan"),
        _leaf("NVDA", shortName="NVIDIA Corporation"),
        _leaf("MSFT", shortName="Microsoft Corporation"),
    ])
    category = hierarchy_for_view(payload, "market")["children"][0]
    cache = HeatmapCache(payload_json=payload, cached_at=now, expires_at=now + timedelta(minutes=5))
    session = _FakeSession(cache)
    raw = SimpleNamespace(
        title="FCX expands copper output",
        description="Freeport McMoRan reported higher production.",
        url="https://example.test/fcx",
        publisher="Example News",
        published_at=now,
        raw_payload={},
    )
    processed = SimpleNamespace(id=42)
    raw_nvda = SimpleNamespace(
        title="NVDA reports record data-center revenue",
        description="NVIDIA raised its quarterly outlook.",
        url="https://example.test/nvda",
        publisher="Market Wire",
        published_at=now - timedelta(hours=1),
        raw_payload={},
    )
    processed_nvda = SimpleNamespace(id=43)
    monkeypatch.setattr(main, "SessionLocal", lambda: session)
    monkeypatch.setattr(
        main,
        "_news_projection_query",
        lambda *_args, **_kwargs: _FakeRowsQuery([
            (raw, processed, None),
            (raw_nvda, processed_nvda, None),
        ]),
    )
    client = TestClient(main.app)

    response = client.get(f"/api/market-heatmap/context?view=market&category_id={category['id']}")
    assert response.status_code == 200
    assert response.json()["news"]["id"] == 42
    assert response.json()["symbolCount"] == 3
    assert response.json()["stockNews"]["FCX"]["id"] == 42
    assert response.json()["stockNews"]["NVDA"]["id"] == 43
    assert "MSFT" not in response.json()["stockNews"]
    unknown = client.get("/api/market-heatmap/context?view=market&category_id=hm-unknown")
    assert unknown.status_code == 404
