"""CopperMind heatmap snapshot construction and dynamic hierarchy helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Iterable, Optional

import pandas as pd

from app.models import HeatmapCache

logger = logging.getLogger(__name__)
HEATMAP_TTL = timedelta(minutes=15)
HEATMAP_METADATA_TTL = timedelta(hours=24)
HEATMAP_METADATA_BATCH_SIZE = 12
HEATMAP_REFRESH_LOCK = "heatmap:refresh"
_snapshot_memo_lock = RLock()
_snapshot_memo: Optional[dict] = None
_yfinance_cache_lock = RLock()
_yfinance_cache_location: Optional[Path] = None

# The legacy project taxonomy remains the source of ``view=themes``.
HEATMAP_GROUP_MAP: dict[str, tuple[str, str]] = {
    "miner_major": ("Copper Miners", "Major Producers"),
    "miner_mid": ("Copper Miners", "Mid-Cap"),
    "miner_junior": ("Copper Miners", "Junior / Exploration"),
    "miner_diversified": ("Copper Miners", "Diversified"),
    "etf_copper": ("Copper & Metals ETFs", "Copper ETFs"),
    "etf_metals": ("Copper & Metals ETFs", "Metals ETFs"),
    "etf_miners": ("Copper & Metals ETFs", "Miner ETFs"),
    "etf_commodity": ("Commodity ETFs", "Broad Commodity"),
    "etf_sector": ("Commodity ETFs", "Sector ETFs"),
    "etf_gold": ("Precious Metals", "Gold ETFs"),
    "gold_etf": ("Precious Metals", "Gold ETFs"),
    "etf_silver": ("Precious Metals", "Silver"),
    "etf_platinum": ("Precious Metals", "Platinum & Palladium"),
    "etf_palladium": ("Precious Metals", "Platinum & Palladium"),
    "commodity_precious": ("Precious Metals", "Precious Futures"),
    "lithium": ("Battery Metals", "Lithium"),
    "rare_earth": ("Battery Metals", "Rare Earth"),
    "uranium": ("Battery Metals", "Uranium"),
    "ev_battery": ("Battery Metals", "EV Battery"),
    "auto_ev": ("EV & Auto Demand", "Electric Vehicles"),
    "auto_traditional": ("EV & Auto Demand", "Traditional Auto"),
    "ev_charging": ("EV & Auto Demand", "EV Charging"),
    "industrial_equipment": ("Industrial Demand", "Equipment"),
    "industrial_conglom": ("Industrial Demand", "Conglomerates"),
    "industrial_electrical": ("Industrial Demand", "Electrical"),
    "industrial_construction": ("Industrial Demand", "Construction"),
    "reit_industrial": ("Industrial Demand", "Industrial REITs"),
    "infra": ("Industrial Demand", "Infrastructure"),
    "materials_steel": ("Base & Materials", "Steel"),
    "materials_aluminum": ("Base & Materials", "Aluminum"),
    "materials_specialty": ("Base & Materials", "Specialty Materials"),
    "materials_chemical": ("Base & Materials", "Chemicals"),
    "commodity_base": ("Base & Materials", "Base Metal Futures"),
    "commodity_energy": ("Energy", "Energy Futures"),
    "energy_major": ("Energy", "Energy Majors"),
    "energy_services": ("Energy", "Energy Services"),
    "tech_semi": ("Tech & Semis", "Semiconductors"),
    "tech_semi_equip": ("Tech & Semis", "Semi Equipment"),
    "homebuilder_etf": ("Homebuilders", "Homebuilder ETFs"),
    "homebuilder": ("Homebuilders", "Homebuilders"),
    "commodity_agri": ("Agricultural", "Agri Futures"),
    "macro_currency": ("Macro & Rates", "Currency"),
    "rates_proxy": ("Macro & Rates", "US Rates"),
    "rates_inflation": ("Macro & Rates", "TIPS / Inflation"),
    "macro_china": ("EM & China", "China"),
    "adr_china": ("EM & China", "China ADRs"),
    "macro_em": ("EM & China", "EM Macro"),
    "em_broad": ("EM & China", "Broad EM"),
    "em_brazil": ("EM & China", "Commodity EM"),
    "em_mexico": ("EM & China", "Commodity EM"),
    "em_canada": ("EM & China", "Commodity EM"),
    "em_southafrica": ("EM & China", "Commodity EM"),
    "em_australia": ("EM & China", "Commodity EM"),
    "europe_broad": ("Europe", "Broad Europe"),
    "europe_eurozone": ("Europe", "Eurozone"),
    "europe_germany": ("Europe", "Germany"),
    "europe_uk": ("Europe", "UK"),
    "transport_trucking": ("Transport", "Trucking"),
    "transport_rail": ("Transport", "Rail"),
    "transport_cargo": ("Transport", "Cargo"),
    "transport_shipping": ("Transport", "Shipping"),
    "financial_etf": ("Credit & Financial", "Financial ETFs"),
    "financial_bank": ("Credit & Financial", "Banks"),
    "credit_hy": ("Credit & Financial", "High Yield"),
    "credit_loans": ("Credit & Financial", "Loans"),
    "utility_etf": ("Utilities", "Utility ETFs"),
    "utility_power": ("Utilities", "Power Utilities"),
    "crypto_proxy": ("Alternative", "Crypto Proxy"),
}

EXCLUDED_CATEGORIES = frozenset({"index_equity", "index_global", "index_vol"})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_heatmap_snapshot_memo(now: Optional[datetime] = None) -> Optional[dict]:
    """Return the one fresh process-local DB snapshot, never stale data."""
    current_time = now or _utcnow()
    with _snapshot_memo_lock:
        memo = _snapshot_memo
        if not memo:
            return None
        expires_at = memo["expires_at"]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if current_time >= expires_at:
            return None
        return memo


def store_heatmap_snapshot_memo(
    payload: dict,
    cached_at: datetime,
    expires_at: datetime,
    refresh_error: Optional[str],
) -> None:
    global _snapshot_memo
    with _snapshot_memo_lock:
        _snapshot_memo = {
            "payload": payload,
            "cached_at": cached_at,
            "expires_at": expires_at,
            "refresh_error": refresh_error,
        }


def invalidate_heatmap_snapshot_memo() -> None:
    global _snapshot_memo
    with _snapshot_memo_lock:
        _snapshot_memo = None


def stable_node_id(*path: str) -> str:
    """Return a stable ID for a semantic hierarchy path."""
    canonical = "/".join(str(part).strip().casefold() for part in path)
    return f"hm-{hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:16]}"


def _load_universe() -> list[dict]:
    for candidate in (
        Path("config/seeds/broad_universe.csv"),
        Path(__file__).parent.parent / "config" / "seeds" / "broad_universe.csv",
    ):
        if candidate.exists():
            frame = pd.read_csv(candidate, comment="#").dropna(subset=["ticker"])
            frame["ticker"] = frame["ticker"].astype(str).str.strip()
            frame = frame[(frame["ticker"] != "") & ~frame["category"].isin(EXCLUDED_CATEGORIES)]
            return frame.to_dict("records")
    logger.warning("broad_universe.csv not found; using minimal copper fallback")
    return [
        {"ticker": "FCX", "category": "miner_major", "source_tag": "copper_core"},
        {"ticker": "SCCO", "category": "miner_major", "source_tag": "copper_core"},
        {"ticker": "BHP", "category": "miner_major", "source_tag": "copper_core"},
        {"ticker": "COPX", "category": "etf_copper", "source_tag": "etf_core"},
        {"ticker": "HG=F", "category": "commodity_base", "source_tag": "commodity_base"},
    ]


def _derive_weight(info: dict, price: float) -> tuple[float, str]:
    market_cap = info.get("marketCap")
    if market_cap and market_cap > 0:
        return float(market_cap), "Market Cap"
    average_volume = info.get("averageVolume") or info.get("regularMarketVolume") or 0
    if average_volume and average_volume > 0 and price > 0:
        return float(average_volume) * price, "Dollar Volume"
    return 1.0, "Equal Weight"


def _clean_label(value: object, fallback: str) -> str:
    label = str(value or "").strip()
    return label if label and label.lower() not in {"none", "nan", "n/a"} else fallback


def _equity_metadata_label(value: object, placeholder: str) -> Optional[str]:
    label = _clean_label(value, "")
    return None if not label or label.casefold() == placeholder.casefold() else label


def _instrument_type(info: dict, category: str, ticker: str) -> str:
    quote_type = str(info.get("quoteType") or "").upper()
    supported = {"EQUITY", "ETF", "MUTUALFUND", "FUTURE", "CRYPTOCURRENCY", "CURRENCY"}
    if quote_type in supported:
        return quote_type.lower()
    if ticker.endswith("=F") or category.startswith("commodity_"):
        return "future"
    if ticker.startswith("^"):
        return "index"
    if category.startswith("etf_") or category.endswith("_etf") or category == "gold_etf":
        return "etf"
    if category == "macro_currency":
        return "currency"
    return "equity"


def _market_path(item: dict) -> tuple[str, str]:
    instrument_type = str(item.get("instrumentType") or "equity").lower()
    if instrument_type == "equity":
        sector = _equity_metadata_label(item.get("sector"), "Other Equities")
        industry = _equity_metadata_label(item.get("industry"), "Unclassified Equities")
        return (
            _clean_label(sector or item.get("group"), "Other Equities"),
            _clean_label(industry or item.get("subgroup"), "Unclassified Equities"),
        )
    asset_labels = {
        "etf": "Funds & ETFs",
        "mutualfund": "Funds & ETFs",
        "future": "Commodities & Futures",
        "currency": "Currencies",
        "cryptocurrency": "Digital Assets",
        "index": "Indices",
    }
    return (
        asset_labels.get(instrument_type, "Other Instruments"),
        _clean_label(item.get("subgroup") or item.get("category"), instrument_type.title()),
    )


def _build_hierarchy(symbols: Iterable[dict], *, view: str = "themes") -> dict:
    """Build a stable D3-compatible hierarchy for either public view."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    seen_instrument_ids: set[str] = set()
    for original in symbols:
        item = dict(original)
        if not item.get("instrumentType"):
            item["instrumentType"] = _instrument_type(
                {}, str(item.get("category") or ""), str(item.get("name") or "")
            )
        if (
            not item.get("logoTicker")
            and str(item.get("instrumentType") or "").lower() in {"equity", "etf", "mutualfund"}
        ):
            item["logoTicker"] = str(item.get("name") or "") or None
        if view == "market":
            group, subgroup = _market_path(item)
        else:
            group = _clean_label(item.get("group"), "Other")
            subgroup = _clean_label(item.get("subgroup"), "Uncategorized")
        item["id"] = item.get("id") or stable_node_id("instrument", str(item.get("name", "")))
        instrument_id = str(item["id"])
        if instrument_id in seen_instrument_ids:
            continue
        seen_instrument_ids.add(instrument_id)
        buckets.setdefault((group, subgroup), []).append(item)

    root_name = "Market by Sector" if view == "market" else "CopperMind Universe"
    root: dict = {"id": stable_node_id(view, "root"), "name": root_name, "children": []}
    grouped: dict[str, list[tuple[str, list[dict]]]] = {}
    for (group, subgroup), leaves in buckets.items():
        grouped.setdefault(group, []).append((subgroup, leaves))
    for group in sorted(grouped):
        group_node = {"id": stable_node_id(view, group), "name": group, "children": []}
        for subgroup, leaves in sorted(grouped[group], key=lambda entry: entry[0]):
            group_node["children"].append(
                {
                    "id": stable_node_id(view, group, subgroup),
                    "name": subgroup,
                    "children": sorted(leaves, key=lambda leaf: str(leaf.get("name", ""))),
                }
            )
        root["children"].append(group_node)
    return root


def flatten_heatmap_leaves(payload: object) -> list[dict]:
    leaves: list[dict] = []

    def walk(node: object) -> None:
        if not isinstance(node, dict):
            return
        children = node.get("children")
        if isinstance(children, list) and children:
            for child in children:
                walk(child)
        elif node.get("name") and "price" in node:
            leaves.append(node)

    walk(payload)
    return leaves


def hierarchy_for_view(payload: dict, view: str) -> dict:
    return _build_hierarchy(
        flatten_heatmap_leaves(payload), view="market" if view == "market" else "themes"
    )


def _deduplicate_universe(rows: Iterable[dict]) -> list[dict]:
    """Keep one provider request per ticker while retaining the last taxonomy row."""
    unique: dict[str, dict] = {}
    for row in rows:
        symbol = str(row.get("ticker") or "").strip()
        if symbol:
            unique[symbol] = dict(row)
    return list(unique.values())


def find_category(payload: dict, category_id: str) -> Optional[dict]:
    stack = list(payload.get("children", []) or [])
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if node.get("id") == category_id and node.get("children"):
            return node
        stack.extend(node.get("children", []) or [])
    return None


def _history_close_series(frame: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None
    try:
        if isinstance(frame.columns, pd.MultiIndex):
            if symbol in frame.columns.get_level_values(0):
                return frame[symbol]["Close"]
            if symbol in frame.columns.get_level_values(1):
                return frame["Close"][symbol]
            return None
        return frame["Close"]
    except (KeyError, TypeError):
        return None


def _history_sparklines(frame: pd.DataFrame, symbols: list[str], points: int = 10) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    if frame is None or frame.empty:
        return output
    for symbol in symbols:
        try:
            series = _history_close_series(frame, symbol)
            if series is None:
                continue
            values = [float(value) for value in series.dropna().tail(points).tolist() if math.isfinite(float(value))]
            if len(values) >= 2 and values[0] != 0:
                output[symbol] = [round(value / values[0] * 100.0, 3) for value in values]
        except (KeyError, TypeError, ValueError):
            continue
    return output


def _history_quotes(frame: pd.DataFrame, symbols: list[str]) -> dict[str, dict]:
    """Extract latest price/change from the already-batched chart response."""
    output: dict[str, dict] = {}
    if frame is None or frame.empty:
        return output
    for symbol in symbols:
        series = _history_close_series(frame, symbol)
        if series is None:
            continue
        try:
            finite = series.dropna()
            values = [float(value) for value in finite.tail(2).tolist() if math.isfinite(float(value))]
            if not values or values[-1] <= 0:
                continue
            previous = values[-2] if len(values) >= 2 else None
            change = ((values[-1] / previous) - 1.0) * 100.0 if previous and previous > 0 else 0.0
            as_of = None
            if len(finite.index):
                timestamp = pd.Timestamp(finite.index[-1])
                timestamp = timestamp.tz_localize(timezone.utc) if timestamp.tzinfo is None else timestamp.tz_convert(timezone.utc)
                as_of = timestamp.isoformat()
            output[symbol] = {
                "price": values[-1],
                "changePercent": change,
                "asOf": as_of,
            }
        except (TypeError, ValueError, ZeroDivisionError, OSError):
            continue
    return output


def _finite_float(*values: object) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _utc_timestamp(value: object) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _metadata_refresh_symbols(
    symbols: list[str],
    previous_by_symbol: dict[str, dict],
    previous_cached_at: object,
    now: datetime,
) -> set[str]:
    cached_at = _utc_timestamp(previous_cached_at)
    due: list[str] = []
    for symbol in symbols:
        previous = previous_by_symbol.get(symbol)
        if not previous:
            due.append(symbol)
            continue
        metadata_at = (
            _utc_timestamp(previous.get("metadataAsOf"))
            if "metadataAsOf" in previous
            else cached_at
        )
        if metadata_at is None or now - metadata_at >= HEATMAP_METADATA_TTL:
            due.append(symbol)
    return set(due[:HEATMAP_METADATA_BATCH_SIZE])


def _configure_yfinance_cache(yfinance_module: object) -> Path:
    """Keep yfinance's SQLite caches off read-only runtime filesystems."""
    global _yfinance_cache_location
    cache_dir = Path(tempfile.gettempdir()) / "coppermind-yfinance-cache"
    with _yfinance_cache_lock:
        if _yfinance_cache_location == cache_dir:
            return cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        setter = getattr(yfinance_module, "set_tz_cache_location", None)
        if not callable(setter):
            raise RuntimeError("installed yfinance does not support a custom cache location")
        setter(str(cache_dir))
        # yfinance's first threaded download can race while lazily creating its
        # SQLite timezone/cookie databases. Initialise both caches serially
        # before the 202-symbol batch starts.
        cache_module = getattr(yfinance_module, "cache", None)
        for getter_name in ("get_tz_cache", "get_cookie_cache"):
            getter = getattr(cache_module, getter_name, None)
            if not callable(getter):
                raise RuntimeError(f"installed yfinance is missing cache initializer {getter_name}")
            cache = getter()
            initialise = getattr(cache, "initialise", None)
            if not callable(initialise):
                raise RuntimeError(f"installed yfinance cache {getter_name} cannot be initialized")
            initialise()
        _yfinance_cache_location = cache_dir
    return cache_dir


def _coverage_is_healthy(new_count: int, requested_count: int, previous_count: int) -> bool:
    if new_count <= 0:
        return False
    return new_count >= max(max(1, int(requested_count * 0.50)), max(1, int(previous_count * 0.70)) if previous_count else 1)


def _as_of(info: dict, fallback: datetime) -> str:
    try:
        if info.get("regularMarketTime"):
            return datetime.fromtimestamp(float(info["regularMarketTime"]), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        pass
    return fallback.isoformat()


def refresh_market_heatmap() -> None:
    """Fetch quotes/history and atomically publish a healthy theme snapshot."""
    from adapters.db.lock import release_lock, try_acquire_lock
    from app.db import SessionLocal, get_db_type, get_engine

    started = time.perf_counter()
    stage_ms: dict[str, float] = {}
    lock_connection = get_engine().connect() if get_db_type() == "postgresql" else None
    if lock_connection is not None and not try_acquire_lock(lock_connection, HEATMAP_REFRESH_LOCK):
        lock_connection.close()
        logger.info("Heatmap refresh deduplicated by advisory lock")
        return
    try:
        with SessionLocal() as session:
            cache: Optional[HeatmapCache] = session.query(HeatmapCache).first()
            if not cache:
                now = _utcnow()
                cache = HeatmapCache(payload_json={}, cached_at=now, expires_at=now)
                session.add(cache)
            cache.refresh_started_at = _utcnow()
            cache.refresh_error = None
            session.commit()
            previous_payload = cache.payload_json if isinstance(cache.payload_json, dict) else {}
            previous_count = len(flatten_heatmap_leaves(previous_payload))
            previous_by_symbol = {
                str(leaf.get("name")): leaf for leaf in flatten_heatmap_leaves(previous_payload)
                if leaf.get("name")
            }
            provider_stats = {
                "info_success": 0,
                "info_empty": 0,
                "info_error": 0,
                "info_without_price": 0,
                "history_price_fallback": 0,
                "missing_price": 0,
                "batch_error": 0,
            }
            try:
                stage_start = time.perf_counter()
                universe_rows = _deduplicate_universe(_load_universe())
                symbols = [str(row["ticker"]) for row in universe_rows]
                row_by_symbol = {str(row["ticker"]): row for row in universe_rows}
                stage_ms["universe"] = (time.perf_counter() - stage_start) * 1000

                import yfinance as yf

                _configure_yfinance_cache(yf)

                stage_start = time.perf_counter()
                sparkline_by_symbol: dict[str, list[float]] = {}
                quote_by_symbol: dict[str, dict] = {}
                try:
                    history = yf.download(
                        tickers=" ".join(symbols), period="3mo", interval="1d",
                        group_by="ticker", auto_adjust=False, progress=False, threads=True,
                    )
                    sparkline_by_symbol = _history_sparklines(history, symbols)
                    quote_by_symbol = _history_quotes(history, symbols)
                except Exception as history_error:
                    logger.warning("Heatmap sparkline batch failed; publishing quotes: %s", history_error)
                stage_ms["history"] = (time.perf_counter() - stage_start) * 1000

                stage_start = time.perf_counter()
                all_data: list[dict] = []
                snapshot_time = _utcnow()
                metadata_symbols = _metadata_refresh_symbols(
                    symbols, previous_by_symbol, cache.cached_at, snapshot_time,
                )
                provider_stats["metadata_scheduled"] = len(metadata_symbols)
                provider_stats["info_skipped"] = 0
                for offset in range(0, len(symbols), 50):
                    if offset:
                        time.sleep(1.5)
                    batch = symbols[offset : offset + 50]
                    try:
                        tickers = yf.Tickers(" ".join(batch))
                        for symbol in batch:
                            info: dict = {}
                            try:
                                ticker = tickers.tickers.get(symbol)
                                history_quote = quote_by_symbol.get(symbol, {})
                                should_fetch_info = symbol in metadata_symbols or not history_quote
                                if should_fetch_info:
                                    try:
                                        info = (ticker.info or {}) if ticker else {}
                                        provider_stats["info_success" if info else "info_empty"] += 1
                                    except Exception as info_error:
                                        provider_stats["info_error"] += 1
                                        logger.debug("Heatmap metadata failed for %s: %s", symbol, info_error)
                                else:
                                    provider_stats["info_skipped"] += 1
                                info_price = _finite_float(info.get("regularMarketPrice"), info.get("currentPrice"))
                                if info and info_price is None:
                                    provider_stats["info_without_price"] += 1
                                raw_price = info_price or _finite_float(history_quote.get("price"))
                                if raw_price is None or raw_price <= 0:
                                    provider_stats["missing_price"] += 1
                                    continue
                                if info_price is None:
                                    provider_stats["history_price_fallback"] += 1
                                price = raw_price
                                row = row_by_symbol[symbol]
                                previous = previous_by_symbol.get(symbol, {})
                                category = str(row.get("category") or "")
                                group, subgroup = HEATMAP_GROUP_MAP.get(category, ("Other", category or "Uncategorized"))
                                weight, weight_label = _derive_weight(info, price)
                                if weight_label == "Equal Weight":
                                    previous_weight = _finite_float(previous.get("weight"))
                                    if previous_weight is not None and previous_weight > 0:
                                        weight = previous_weight
                                        weight_label = str(previous.get("weightLabel") or "Previous Snapshot")
                                instrument_type = _instrument_type(info, category, symbol)
                                sector = (
                                    _equity_metadata_label(info.get("sector"), "Other Equities")
                                    or _equity_metadata_label(previous.get("sector"), "Other Equities")
                                )
                                industry = (
                                    _equity_metadata_label(info.get("industry"), "Unclassified Equities")
                                    or _equity_metadata_label(previous.get("industry"), "Unclassified Equities")
                                )
                                change_percent = _finite_float(
                                    info.get("regularMarketChangePercent"), history_quote.get("changePercent"), 0.0,
                                )
                                has_market_time = info.get("regularMarketTime") is not None
                                if should_fetch_info:
                                    metadata_as_of = snapshot_time.isoformat()
                                elif "metadataAsOf" in previous:
                                    metadata_as_of = previous.get("metadataAsOf")
                                elif previous:
                                    metadata_as_of = (_utc_timestamp(cache.cached_at) or snapshot_time).isoformat()
                                else:
                                    metadata_as_of = None
                                all_data.append({
                                    "id": stable_node_id("instrument", symbol),
                                    "name": symbol,
                                    "shortName": info.get("shortName") or info.get("longName") or previous.get("shortName") or symbol,
                                    "price": round(price, 4),
                                    "changePercent": round(change_percent or 0.0, 4),
                                    "weight": round(weight, 2),
                                    "weightLabel": weight_label,
                                    "group": group, "subgroup": subgroup,
                                    "category": category, "sourceTag": row.get("source_tag", ""),
                                    "instrumentType": instrument_type,
                                    "sector": sector if instrument_type == "equity" else None,
                                    "industry": industry if instrument_type == "equity" else None,
                                    "exchange": info.get("exchange") or info.get("fullExchangeName") or previous.get("exchange"),
                                    "logoTicker": symbol if instrument_type in {"equity", "etf", "mutualfund"} else None,
                                    "sparkline": sparkline_by_symbol.get(symbol) or previous.get("sparkline"),
                                    "asOf": _as_of(info, snapshot_time) if has_market_time else history_quote.get("asOf") or snapshot_time.isoformat(),
                                    "metadataAsOf": metadata_as_of,
                                })
                            except Exception as symbol_error:
                                logger.debug("Heatmap quote failed for %s: %s", symbol, symbol_error)
                    except Exception as batch_error:
                        provider_stats["batch_error"] += len(batch)
                        logger.warning("Heatmap quote batch %d failed: %s", offset // 50, batch_error)
                stage_ms["quotes"] = (time.perf_counter() - stage_start) * 1000
                if not _coverage_is_healthy(len(all_data), len(symbols), previous_count):
                    raise RuntimeError(f"unhealthy symbol coverage {len(all_data)}/{len(symbols)}; previous={previous_count}")

                stage_start = time.perf_counter()
                root = _build_hierarchy(all_data, view="themes")
                stage_ms["hierarchy"] = (time.perf_counter() - stage_start) * 1000
                now = _utcnow()
                cache.payload_json = root
                cache.cached_at = now
                cache.expires_at = now + HEATMAP_TTL
                cache.refresh_started_at = None
                cache.refresh_error = None
                session.commit()
                invalidate_heatmap_snapshot_memo()
                logger.info("heatmap_refresh %s", json.dumps({
                    "status": "success", "symbols": len(all_data), "requested": len(symbols),
                    "provider": provider_stats,
                    "stage_ms": {key: round(value, 2) for key, value in stage_ms.items()},
                    "total_ms": round((time.perf_counter() - started) * 1000, 2),
                }, sort_keys=True))
            except Exception as error:
                session.rollback()
                current = session.query(HeatmapCache).first()
                if current:
                    current.refresh_started_at = None
                    current.refresh_error = str(error)[:500]
                    # Keep serving the last healthy payload and bound provider
                    # retries. The API exposes this future instant as
                    # next_refresh_at while still marking the payload stale.
                    current.expires_at = _utcnow() + timedelta(seconds=60)
                    session.commit()
                logger.error("heatmap_refresh %s", json.dumps({
                    "status": "error", "error": str(error)[:500], "preserved_symbols": previous_count,
                    "provider": provider_stats,
                    "stage_ms": {key: round(value, 2) for key, value in stage_ms.items()},
                    "total_ms": round((time.perf_counter() - started) * 1000, 2),
                }, sort_keys=True), exc_info=True)
    finally:
        if lock_connection is not None:
            try:
                release_lock(lock_connection, HEATMAP_REFRESH_LOCK)
            finally:
                lock_connection.close()


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def news_match_scores(title: str, description: str, leaves: Iterable[dict]) -> dict[str, int]:
    """Score every ticker/company in one tokenization pass without LLM work."""
    haystack = f"{title} {description}".casefold()
    tokens = set(_TOKEN_RE.findall(haystack))
    scores: dict[str, int] = {}
    for leaf in leaves:
        symbol = str(leaf.get("name") or "").strip()
        ticker = symbol.casefold()
        score = 0
        normalized = ticker.replace("-", "").replace(".", "")
        if ticker and ((len(normalized) >= 3 and normalized in tokens) or f"${ticker}" in haystack):
            score += 4
        company_tokens = [
            token for token in _TOKEN_RE.findall(str(leaf.get("shortName") or "").casefold())
            if len(token) >= 4 and token not in {"incorporated", "corporation", "company", "limited", "holdings"}
        ]
        if company_tokens and all(token in tokens for token in company_tokens[:2]):
            score += 2
        elif company_tokens and company_tokens[0] in tokens:
            score += 1
        if symbol:
            scores[symbol] = max(score, scores.get(symbol, 0))
    return scores


def news_match_score(title: str, description: str, leaves: Iterable[dict]) -> int:
    """Backward-compatible aggregate score for a category."""
    return sum(news_match_scores(title, description, leaves).values())
