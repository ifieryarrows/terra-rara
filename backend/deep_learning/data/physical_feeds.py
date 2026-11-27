"""Free, zero-key dynamic physical & positioning copper data loaders.

Sources:
1. Westmetall public LME daily table (``LME Copper Cash-Settlement``,
   ``LME Copper 3-month``, ``LME Copper stock`` in tonnes) — 2008..present.
2. CFTC Disaggregated Futures-Only Commitments of Traders Socrata JSON API
   (COMEX Copper ``085692``) — 2006..present.

All functions fail open (returning empty DataFrames on network/schema error)
and apply strict date/weekend normalization. Historical completed Westmetall
calendar years (``year < current_year``) are cached cleanly on disk so live
worker runs only fetch the current year.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WESTMETALL_LME_CU_URL = "https://www.westmetall.com/en/markdaten.php"
CFTC_COT_SOCRATA_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
CFTC_COPPER_CONTRACT_CODE = "085692"


def _default_physical_cache_dir() -> Path:
    model_dir = os.environ.get("MODEL_DIR", "").strip()
    candidates = []
    if model_dir:
        candidates.append(Path(model_dir) / "cache" / "physical")
    candidates.append(Path(tempfile.gettempdir()) / "coppermind_cache" / "physical")
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            continue
    return Path(tempfile.gettempdir())


def _parse_numeric_cell(raw: str) -> float:
    text = re.sub(r"<[^>]+>", "", str(raw or "")).replace("\xa0", " ").strip()
    if not text or text in {"-", "—", "n/a", "N/A", "nan", "NaN"}:
        return float("nan")
    cleaned = text.replace(",", "").replace(" ", "")
    try:
        val = float(cleaned)
        return val if np.isfinite(val) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def parse_westmetall_lme_html(html: str) -> pd.DataFrame:
    """Parse Westmetall LME Copper HTML table into a clean daily DataFrame."""
    if not html or "<tr" not in html.lower():
        return pd.DataFrame()

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL)
    records: list[dict[str, Any]] = []
    for row_html in rows:
        cells = [
            re.sub(r"<[^>]+>", "", cell).replace("\xa0", " ").strip()
            for cell in re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.IGNORECASE | re.DOTALL)
        ]
        if len(cells) < 4:
            continue
        dt_str = cells[0].strip()
        if not re.search(r"\d{4}", dt_str):
            continue
        dt = pd.to_datetime(dt_str, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            continue
        cash = _parse_numeric_cell(cells[1])
        m3 = _parse_numeric_cell(cells[2])
        stock = _parse_numeric_cell(cells[3])
        if not np.isfinite(stock) or stock <= 0.0:
            if not (np.isfinite(cash) and np.isfinite(m3)):
                continue
        records.append(
            {
                "date": dt.tz_localize(None).normalize() if getattr(dt, "tzinfo", None) else dt.normalize(),
                "lme_cash_usd_t": cash,
                "lme_3m_usd_t": m3,
                "lme_stock_tonnes": stock,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    df = df.set_index("date").sort_index()
    df = df.loc[df.index.dayofweek < 5]

    cash_s = pd.to_numeric(df["lme_cash_usd_t"], errors="coerce")
    m3_s = pd.to_numeric(df["lme_3m_usd_t"], errors="coerce").where(lambda s: s > 0.0)
    # Positive = backwardation (cash premium over 3M = physical tightness)
    df["lme_cash_3m_spread_pct"] = (cash_s - m3_s) / m3_s
    return df.astype("float64")


def _year_frame_to_cache(df: pd.DataFrame, cache_file: Path) -> None:
    try:
        serializable = df.copy()
        serializable.index = serializable.index.strftime("%Y-%m-%d")
        cache_file.write_text(serializable.to_json(orient="index"), encoding="utf-8")
    except Exception:
        pass


def _year_frame_from_cache(cache_file: Path) -> pd.DataFrame:
    if not cache_file.exists():
        return pd.DataFrame()
    try:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not raw:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(raw, orient="index")
        df.index = pd.to_datetime(df.index, errors="coerce").normalize()
        df = df.loc[df.index.notna()].sort_index()
        return df.astype("float64")
    except Exception:
        return pd.DataFrame()


def fetch_westmetall_lme_copper(
    start_year: int = 2011,
    end_year: Optional[int] = None,
    *,
    timeout_seconds: float = 12.0,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch multi-year LME Copper cash, 3M, and warehouse stock series from Westmetall."""
    import requests

    current_year = datetime.now(timezone.utc).year
    final_year = int(end_year) if end_year is not None else current_year
    first_year = max(2008, min(int(start_year), final_year))
    resolved_cache = Path(cache_dir) if cache_dir is not None else _default_physical_cache_dir()
    try:
        resolved_cache.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    frames: list[pd.DataFrame] = []
    headers = {"User-Agent": "CopperMind/1.0 (Commodity Research)"}

    for year in range(first_year, final_year + 1):
        cache_file = resolved_cache / f"westmetall_lme_cu_{year}.json"
        if year < current_year:
            cached = _year_frame_from_cache(cache_file)
            if not cached.empty:
                frames.append(cached)
                continue

        params: dict[str, Any] = {"action": "table", "field": "LME_Cu_cash", "year": year}
        try:
            resp = requests.get(
                WESTMETALL_LME_CU_URL,
                params=params,
                headers=headers,
                timeout=timeout_seconds,
            )
            if resp.status_code != 200 or not resp.text:
                continue
            yr_df = parse_westmetall_lme_html(resp.text)
            if not yr_df.empty:
                frames.append(yr_df)
                if year < current_year:
                    _year_frame_to_cache(yr_df, cache_file)
        except Exception as exc:
            logger.debug("Westmetall LME fetch skipped for %s: %s", year, exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = combined.loc[combined.index.dayofweek < 5]
    return combined


def _first_numeric(rec: dict[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        if key in rec and rec[key] not in (None, ""):
            try:
                val = float(rec[key])
                if np.isfinite(val):
                    return val
            except (TypeError, ValueError):
                continue
    return float("nan")


def parse_cftc_cot_json(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse CFTC Disaggregated Futures JSON rows into causal COT positioning metrics."""
    if not records:
        return pd.DataFrame()

    parsed: list[dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        raw_dt = rec.get("report_date_as_yyyy_mm_dd") or rec.get("report_date")
        dt = pd.to_datetime(raw_dt, errors="coerce")
        if pd.isna(dt):
            continue
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.tz_localize(None)
        dt = dt.normalize()

        oi = _first_numeric(rec, ("open_interest_all", "open_interest"))
        mm_long = _first_numeric(rec, ("m_money_positions_long_all", "m_money_positions_long"))
        mm_short = _first_numeric(rec, ("m_money_positions_short_all", "m_money_positions_short"))
        pm_long = _first_numeric(rec, ("prod_merc_positions_long_all", "prod_merc_positions_long"))
        pm_short = _first_numeric(rec, ("prod_merc_positions_short_all", "prod_merc_positions_short"))

        if not np.isfinite(oi) or oi <= 0.0:
            continue
        mm_net = (mm_long - mm_short) if (np.isfinite(mm_long) and np.isfinite(mm_short)) else float("nan")
        pm_net = (pm_long - pm_short) if (np.isfinite(pm_long) and np.isfinite(pm_short)) else float("nan")

        parsed.append(
            {
                "report_date": dt,
                "open_interest": oi,
                "cot_mm_net": mm_net,
                "cot_mm_net_pct_oi": mm_net / oi if np.isfinite(mm_net) else float("nan"),
                "cot_prod_net_pct_oi": pm_net / oi if np.isfinite(pm_net) else float("nan"),
            }
        )

    if not parsed:
        return pd.DataFrame()

    df = pd.DataFrame(parsed)
    df = df.dropna(subset=["report_date"]).drop_duplicates(subset=["report_date"], keep="last")
    df = df.set_index("report_date").sort_index()
    return df.astype("float64")


def fetch_cftc_copper_cot(
    *,
    contract_code: str = CFTC_COPPER_CONTRACT_CODE,
    limit: int = 2000,
    timeout_seconds: float = 12.0,
) -> pd.DataFrame:
    """Fetch COMEX Copper Disaggregated COT history from CFTC Socrata API (free, zero-key)."""
    import requests

    headers: dict[str, str] = {"User-Agent": "CopperMind/1.0 (Commodity Research)"}
    app_token = os.environ.get("CFTC_APP_TOKEN", "").strip()
    if app_token:
        headers["X-App-Token"] = app_token

    params = {
        "cftc_contract_market_code": contract_code,
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": int(limit),
    }
    try:
        resp = requests.get(
            CFTC_COT_SOCRATA_URL,
            params=params,
            headers=headers,
            timeout=timeout_seconds,
        )
        if resp.status_code != 200:
            logger.debug("CFTC COT fetch returned HTTP %s", resp.status_code)
            return pd.DataFrame()
        payload = resp.json()
        if not isinstance(payload, list):
            return pd.DataFrame()
        return parse_cftc_cot_json(payload)
    except Exception as exc:
        logger.debug("CFTC COT fetch failed (fail-open): %s", exc)
        return pd.DataFrame()
