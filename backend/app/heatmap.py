"""
CopperMind Heatmap Backend — Project-Universe Aligned.

Universe source: config/seeds/broad_universe.csv (ticker, category, source_tag).
Hierarchy: group -> subgroup -> symbol (taxonomy derived from CSV, not Yahoo sector).
Weight fallback: marketCap -> dollarVolume (avgVolume*price) -> equalWeight(1.0).
Excluded: broad market indices (index_equity, index_global, index_vol) which are
  screener correlation proxies, not displayable heatmap assets.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.models import HeatmapCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project taxonomy: CSV category -> (display group, display subgroup)
# These are the ONLY groups that appear in the UI.
# ---------------------------------------------------------------------------
HEATMAP_GROUP_MAP: dict[str, tuple[str, str]] = {
    # Miners
    "miner_major":              ("Copper Miners",            "Major Producers"),
    "miner_mid":                ("Copper Miners",            "Mid-Cap"),
    "miner_junior":             ("Copper Miners",            "Junior / Exploration"),
    "miner_diversified":        ("Copper Miners",            "Diversified"),
    # Copper-focused ETFs
    "etf_copper":               ("Copper & Metals ETFs",     "Copper ETFs"),
    "etf_metals":               ("Copper & Metals ETFs",     "Metals ETFs"),
    "etf_miners":               ("Copper & Metals ETFs",     "Miner ETFs"),
    # Commodity ETFs
    "etf_commodity":            ("Commodity ETFs",           "Broad Commodity"),
    "etf_sector":               ("Commodity ETFs",           "Sector ETFs"),
    # Precious Metals
    "etf_gold":                 ("Precious Metals",          "Gold ETFs"),
    "gold_etf":                 ("Precious Metals",          "Gold ETFs"),
    "etf_silver":               ("Precious Metals",          "Silver"),
    "etf_platinum":             ("Precious Metals",          "Platinum & Palladium"),
    "etf_palladium":            ("Precious Metals",          "Platinum & Palladium"),
    "commodity_precious":       ("Precious Metals",          "Precious Futures"),
    # Battery Metals
    "lithium":                  ("Battery Metals",           "Lithium"),
    "rare_earth":               ("Battery Metals",           "Rare Earth"),
    "uranium":                  ("Battery Metals",           "Uranium"),
    "ev_battery":               ("Battery Metals",           "EV Battery"),
    # EV & Auto Demand
    "auto_ev":                  ("EV & Auto Demand",         "Electric Vehicles"),
    "auto_traditional":         ("EV & Auto Demand",         "Traditional Auto"),
    "ev_charging":              ("EV & Auto Demand",         "EV Charging"),
    # Industrial Demand
    "industrial_equipment":     ("Industrial Demand",        "Equipment"),
    "industrial_conglom":       ("Industrial Demand",        "Conglomerates"),
    "industrial_electrical":    ("Industrial Demand",        "Electrical"),
    "industrial_construction":  ("Industrial Demand",        "Construction"),
    "reit_industrial":          ("Industrial Demand",        "Industrial REITs"),
    "infra":                    ("Industrial Demand",        "Infrastructure"),
    # Base & Materials
    "materials_steel":          ("Base & Materials",         "Steel"),
    "materials_aluminum":       ("Base & Materials",         "Aluminum"),
    "materials_specialty":      ("Base & Materials",         "Specialty Materials"),
    "materials_chemical":       ("Base & Materials",         "Chemicals"),
    "commodity_base":           ("Base & Materials",         "Base Metal Futures"),
    # Energy
    "commodity_energy":         ("Energy",                   "Energy Futures"),
    "energy_major":             ("Energy",                   "Energy Majors"),
    "energy_services":          ("Energy",                   "Energy Services"),
    # Tech & Semis Demand
    "tech_semi":                ("Tech & Semis",             "Semiconductors"),
    "tech_semi_equip":          ("Tech & Semis",             "Semi Equipment"),
    # Homebuilders
    "homebuilder_etf":          ("Homebuilders",             "Homebuilder ETFs"),
    "homebuilder":              ("Homebuilders",             "Homebuilders"),
    # Agricultural (Demand Proxy)
    "commodity_agri":           ("Agricultural",             "Agri Futures"),
    # Macro & Rates
    "macro_currency":           ("Macro & Rates",            "Currency"),
    "rates_proxy":              ("Macro & Rates",            "US Rates"),
    "rates_inflation":          ("Macro & Rates",            "TIPS / Inflation"),
    # EM & China
    "macro_china":              ("EM & China",               "China"),
    "adr_china":                ("EM & China",               "China ADRs"),
    "macro_em":                 ("EM & China",               "EM Macro"),
    "em_broad":                 ("EM & China",               "Broad EM"),
    "em_brazil":                ("EM & China",               "Commodity EM"),
    "em_mexico":                ("EM & China",               "Commodity EM"),
    "em_canada":                ("EM & China",               "Commodity EM"),
    "em_southafrica":           ("EM & China",               "Commodity EM"),
    "em_australia":             ("EM & China",               "Commodity EM"),
    # Europe
    "europe_broad":             ("Europe",                   "Broad Europe"),
    "europe_eurozone":          ("Europe",                   "Eurozone"),
    "europe_germany":           ("Europe",                   "Germany"),
    "europe_uk":                ("Europe",                   "UK"),
    # Transport (Demand Proxy)
    "transport_trucking":       ("Transport",                "Trucking"),
    "transport_rail":           ("Transport",                "Rail"),
    "transport_cargo":          ("Transport",                "Cargo"),
    "transport_shipping":       ("Transport",                "Shipping"),
    # Credit & Financial (Risk Proxy)
    "financial_etf":            ("Credit & Financial",       "Financial ETFs"),
    "financial_bank":           ("Credit & Financial",       "Banks"),
    "credit_hy":                ("Credit & Financial",       "High Yield"),
    "credit_loans":             ("Credit & Financial",       "Loans"),
    # Utilities
    "utility_etf":              ("Utilities",                "Utility ETFs"),
    "utility_power":            ("Utilities",                "Power Utilities"),
    # Alternative / Crypto Risk Proxy
    "crypto_proxy":             ("Alternative",              "Crypto Proxy"),
}

# Categories that are broad market indices — kept for screener correlation,
# excluded from the heatmap display universe entirely.
EXCLUDED_CATEGORIES: frozenset[str] = frozenset({
    "index_equity",   # ^GSPC, ^DJI, ^IXIC, ^RUT
    "index_global",   # ^STOXX50E, ^FTSE, ^N225, ^HSI
    "index_vol",      # ^VIX
})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_universe() -> list[dict]:
    """
    Load broad_universe.csv, skip comment lines, and return a list of
    {ticker, category, source_tag} dicts after filtering excluded categories.
    """
    for candidate in [
        Path("config/seeds/broad_universe.csv"),
        Path(__file__).parent.parent / "config" / "seeds" / "broad_universe.csv",
    ]:
        if candidate.exists():
            df = pd.read_csv(candidate, comment="#")
            # Only keep rows that have a valid ticker (not blank / comment artifacts)
            df = df.dropna(subset=["ticker"])
            df["ticker"] = df["ticker"].str.strip()
            df = df[df["ticker"] != ""]
            # Exclude broad market index categories
            df = df[~df["category"].isin(EXCLUDED_CATEGORIES)]
            return df.to_dict("records")

    # Absolute fallback if CSV not found
    logger.warning("broad_universe.csv not found — using minimal copper fallback")
    return [
        {"ticker": "FCX",  "category": "miner_major", "source_tag": "copper_core"},
        {"ticker": "SCCO", "category": "miner_major", "source_tag": "copper_core"},
        {"ticker": "BHP",  "category": "miner_major", "source_tag": "copper_core"},
        {"ticker": "COPX", "category": "etf_copper",  "source_tag": "etf_core"},
        {"ticker": "HG=F", "category": "commodity_base","source_tag": "commodity_base"},
    ]


def _derive_weight(info: dict, price: float) -> tuple[float, str]:
    """
    Derive a sizing weight for the treemap cell.
    Priority: marketCap -> dollarVolume (avgVolume * price) -> equal weight 1.0.
    Returns (weight_value, weight_label).
    """
    mc = info.get("marketCap")
    if mc and mc > 0:
        return float(mc), "Market Cap"

    avg_vol = info.get("averageVolume") or info.get("regularMarketVolume") or 0
    if avg_vol and avg_vol > 0 and price > 0:
        dollar_vol = float(avg_vol) * price
        return dollar_vol, "Dollar Volume"

    return 1.0, "Equal Weight"


def _build_hierarchy(symbols: list[dict]) -> dict:
    """
    Build D3-compatible squarified treemap hierarchy from project taxonomy.
    Structure: root -> group -> subgroup -> leaf (symbol).
    """
    groups: dict[str, dict[str, list]] = {}

    for item in symbols:
        grp = item.get("group", "Other")
        sub = item.get("subgroup", "Other")
        groups.setdefault(grp, {}).setdefault(sub, []).append(item)

    root: dict = {"name": "CopperMind Universe", "children": []}
    for grp_name, subs in groups.items():
        grp_node: dict = {"name": grp_name, "children": []}
        for sub_name, leaves in subs.items():
            sub_node: dict = {"name": sub_name, "children": leaves}
            grp_node["children"].append(sub_node)
        root["children"].append(grp_node)

    return root


def refresh_market_heatmap() -> None:
    """
    Background task: fetch live quotes for the project universe and rebuild
    the heatmap cache. Uses project taxonomy (not Yahoo sector/industry).
    """
    from app.db import SessionLocal

    with SessionLocal() as session:
        cache: Optional[HeatmapCache] = session.query(HeatmapCache).first()
        if not cache:
            cache = HeatmapCache(
                payload_json={},
                cached_at=_utcnow(),
                expires_at=_utcnow(),
            )
            session.add(cache)

        cache.refresh_started_at = _utcnow()
        cache.refresh_error = None
        session.commit()

        try:
            universe_rows = _load_universe()
            symbols_to_fetch = [r["ticker"] for r in universe_rows]
            category_map = {r["ticker"]: r["category"] for r in universe_rows}
            source_tag_map = {r["ticker"]: r.get("source_tag", "") for r in universe_rows}

            logger.info("Heatmap: fetching %d project universe symbols", len(symbols_to_fetch))

            import yfinance as yf

            # Batch fetch with rate-limit protection
            # Yahoo Finance aggressively blocks high-frequency scrapers.
            # Process in batches of 50 (smaller than before) with inter-batch
            # delays to stay under the radar.
            import time

            all_data: list[dict] = []
            batch_size = 50
            for i in range(0, len(symbols_to_fetch), batch_size):
                batch = symbols_to_fetch[i : i + batch_size]

                # Inter-batch delay to avoid Yahoo crumb invalidation
                if i > 0:
                    time.sleep(1.5)

                try:
                    tickers_obj = yf.Tickers(" ".join(batch))
                    for sym in batch:
                        try:
                            ticker = tickers_obj.tickers.get(sym)
                            if not ticker:
                                continue

                            info = ticker.info or {}
                            price = info.get("regularMarketPrice") or info.get("currentPrice")

                            if price is None:
                                # Symbol returned no live price — skip
                                logger.debug("Heatmap: no price for %s, skipping", sym)
                                continue

                            price = float(price)
                            change = float(info.get("regularMarketChangePercent") or 0.0)
                            short_name = info.get("shortName") or info.get("longName") or sym

                            weight, weight_label = _derive_weight(info, price)

                            cat = category_map.get(sym, "")
                            group, subgroup = HEATMAP_GROUP_MAP.get(cat, ("Other", cat or "Uncategorized"))

                            all_data.append({
                                "name": sym,
                                "shortName": short_name,
                                "price": round(price, 4),
                                "changePercent": round(change, 4),
                                "weight": round(weight, 2),
                                "weightLabel": weight_label,
                                "group": group,
                                "subgroup": subgroup,
                                "category": cat,
                                "sourceTag": source_tag_map.get(sym, ""),
                            })
                        except Exception as sym_err:
                            logger.debug("Heatmap: error for %s: %s", sym, sym_err)
                except Exception as batch_err:
                    logger.warning("Heatmap: batch %d failed: %s", i // batch_size, batch_err)

            logger.info("Heatmap: %d symbols with live data (of %d universe)", len(all_data), len(symbols_to_fetch))

            root = _build_hierarchy(all_data)

            now = _utcnow()
            cache.payload_json = root
            cache.cached_at = now
            cache.expires_at = now + timedelta(minutes=15)
            cache.refresh_started_at = None
            cache.refresh_error = None
            session.commit()
            logger.info("Heatmap cache refreshed successfully")

        except Exception as err:
            logger.error("Heatmap refresh failed: %s", err, exc_info=True)
            try:
                c = session.query(HeatmapCache).first()
                if c:
                    c.refresh_started_at = None
                    c.refresh_error = str(err)[:500]
                    session.commit()
            except Exception:
                pass
