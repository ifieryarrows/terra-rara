"""Causal weekly data loading, resampling, and macro feature engineering for 20D/60D outlook."""

from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.instruments import canonicalize_instrument_symbol
from app.models import PriceBar
from long_term.schemas import (
    LongTermMacroPillar,
    LongTermMacroScorecard,
    LongTermPillarMetric,
    StanceType,
)

logger = logging.getLogger(__name__)

LONG_TERM_SYMBOLS: tuple[str, ...] = (
    "HG=F",
    "DX-Y.NYB",
    "CL=F",
    "GXC",
    "COPX",
    "FCX",
    "SCCO",
    "RIO",
    "BHP",
    "FXA",
    "XME",
)

MINER_SYMBOLS: tuple[str, ...] = ("FCX", "SCCO", "RIO", "BHP")

FRED_SERIES_LAG_DAYS: dict[str, int] = {
    "DGS10": 2,
    "T10YIE": 2,
    "DTWEXBGS": 7,
}

# Feature catalog mapping column name -> human-readable metadata and economic pillar.
FEATURE_CATALOG: dict[str, dict[str, str]] = {
    "hg_ret4": {"label": "Copper 4W Momentum", "pillar": "trend_structure"},
    "hg_ret12": {"label": "Copper 12W Return", "pillar": "trend_structure"},
    "hg_ret26": {"label": "Copper 26W Return", "pillar": "trend_structure"},
    "hg_ret52": {"label": "Copper 52W Return", "pillar": "trend_structure"},
    "hg_sma10w_ratio": {"label": "Copper vs 10W Trend", "pillar": "trend_structure"},
    "hg_sma40w_ratio": {"label": "Copper vs 40W Trend", "pillar": "trend_structure"},
    "hg_dd52": {"label": "Copper 52W Drawdown", "pillar": "trend_structure"},
    "hg_vol12": {"label": "12W Realized Volatility", "pillar": "trend_structure"},
    "hg_vol26": {"label": "26W Realized Volatility", "pillar": "trend_structure"},
    "dxy_ret12": {"label": "US Dollar 12W Move", "pillar": "usd_macro"},
    "dxy_ret26": {"label": "US Dollar 26W Trend", "pillar": "usd_macro"},
    "dxy_sma26_ratio": {"label": "US Dollar vs 26W Trend", "pillar": "usd_macro"},
    "real_yield_10y": {"label": "10Y US Real Yield", "pillar": "usd_macro"},
    "real_yield_chg12": {"label": "10Y Real Yield 12W Change", "pillar": "usd_macro"},
    "real_yield_chg26": {"label": "10Y Real Yield 26W Change", "pillar": "usd_macro"},
    "breakeven_10y_z": {"label": "10Y Inflation Breakeven Regime", "pillar": "usd_macro"},
    "dtwex_ret26": {"label": "Broad Trade-Weighted USD 26W", "pillar": "usd_macro"},
    "gxc_ret12": {"label": "China Equities (GXC) 12W", "pillar": "china_industrial_demand"},
    "gxc_ret26": {"label": "China Equities (GXC) 26W", "pillar": "china_industrial_demand"},
    "gxc_sma26_ratio": {"label": "China Equities vs 26W Trend", "pillar": "china_industrial_demand"},
    "copx_ret12": {"label": "Copper Miners ETF (COPX) 12W", "pillar": "china_industrial_demand"},
    "fxa_ret12": {"label": "AUD Commodity FX (FXA) 12W", "pillar": "china_industrial_demand"},
    "fxa_ret26": {"label": "AUD Commodity FX (FXA) 26W", "pillar": "china_industrial_demand"},
    "cl_ret12": {"label": "Crude Oil (WTI) 12W Cost Impulse", "pillar": "china_industrial_demand"},
    "miners_vs_hg_12w": {"label": "Major Miners vs Copper 12W Spread", "pillar": "cross_asset_confirmation"},
    "copx_vs_hg_12w": {"label": "COPX vs Copper 12W Spread", "pillar": "cross_asset_confirmation"},
    "xme_vs_hg_12w": {"label": "Metals & Mining vs Copper 12W Spread", "pillar": "cross_asset_confirmation"},
    "miners_ret12": {"label": "Major Copper Miners 12W Basket", "pillar": "cross_asset_confirmation"},
}


@dataclass(frozen=True)
class LongTermPanelBundle:
    """Container for the weekly feature/target panel and provenance metadata."""

    panel: pd.DataFrame
    target_symbol: str
    feature_columns: list[str]
    as_of_date: str
    start_date: str
    end_date: str
    weekly_bars: int
    years_covered: float
    fred_macro_enabled: bool
    symbols_used: list[str]


def _normalize_bar_calendar_dates(raw_dates: pd.Series, utc_ts: pd.Series) -> pd.Series:
    """Normalize bar timestamps to tz-naive midnight preserving explicit non-UTC offsets."""
    utc_norm = utc_ts.dt.tz_localize(None).dt.normalize()
    if raw_dates.empty:
        return utc_norm

    needs_offset_scan = False
    for raw_val in raw_dates:
        if isinstance(raw_val, datetime):
            offset = raw_val.utcoffset()
            if offset is not None and offset != timedelta(0):
                needs_offset_scan = True
                break
        elif isinstance(raw_val, str):
            stripped = raw_val.strip()
            if not stripped.endswith(("Z", "z", "+00:00", "-00:00")):
                needs_offset_scan = True
                break

    if not needs_offset_scan:
        return utc_norm

    normalized = []
    for raw_val, utc_val in zip(raw_dates, utc_ts):
        if isinstance(raw_val, datetime):
            normalized.append(raw_val.date())
        elif isinstance(raw_val, str) and not raw_val.strip().endswith(
            ("Z", "z", "+00:00", "-00:00")
        ):
            parsed = pd.to_datetime(raw_val, errors="coerce")
            normalized.append(parsed.date() if pd.notna(parsed) else utc_val.date())
        else:
            normalized.append(utc_val.date())
    return pd.to_datetime(normalized)


def clean_price_rows_to_series(rows: Iterable[tuple[Any, ...]], symbol: str) -> pd.Series:
    """Clean PriceBar tuples into a deduplicated, weekday-only adjusted-close series.

    Matches CopperMind's canonical data hygiene:
    - Excludes non-finite or non-positive closes
    - Prefers ``00:00 UTC`` canonical rows over legacy ``04:00/05:00 UTC`` rows on duplicate dates
    - Excludes weekend (Sat/Sun) ghost bars
    - Uses ``adj_close`` when finite and positive, falling back to ``close``
    """
    rows_list = list(rows)
    if not rows_list:
        return pd.Series(dtype="float64", name=symbol)

    df = pd.DataFrame(
        rows_list,
        columns=["date", "open", "high", "low", "close", "volume", "adj_close"],
    )
    numeric_cols = ["open", "high", "low", "close", "volume", "adj_close"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    invalid_close = df["close"].isna() | (df["close"] <= 0.0)
    if invalid_close.any():
        df = df.loc[~invalid_close].copy()
    if df.empty:
        return pd.Series(dtype="float64", name=symbol)

    raw_ts = pd.to_datetime(df["date"], errors="coerce", utc=True)
    invalid_dates = raw_ts.isna()
    if invalid_dates.any():
        df = df.loc[~invalid_dates].copy()
        raw_ts = raw_ts.loc[~invalid_dates]
    if df.empty:
        return pd.Series(dtype="float64", name=symbol)

    cal_dates = _normalize_bar_calendar_dates(df["date"], raw_ts)
    df["_raw_ts"] = raw_ts.to_numpy()
    # Prefer canonical 00:00:00 UTC bars over legacy 04:00/05:00 UTC bars on duplicate calendar days.
    df["_is_midnight"] = (
        (raw_ts.dt.hour == 0)
        & (raw_ts.dt.minute == 0)
        & (raw_ts.dt.second == 0)
    ).astype(int).to_numpy()
    df["date"] = cal_dates.to_numpy()

    # Drop leaked synthetic test fixture rows (100/105/95/102/1000) when mixed with real prices.
    fixture_mask = (
        np.isclose(df["open"].fillna(-1.0).astype(float), 100.0, rtol=1e-6, atol=1e-6)
        & np.isclose(df["high"].fillna(-1.0).astype(float), 105.0, rtol=1e-6, atol=1e-6)
        & np.isclose(df["low"].fillna(-1.0).astype(float), 95.0, rtol=1e-6, atol=1e-6)
        & np.isclose(df["close"].astype(float), 102.0, rtol=1e-6, atol=1e-6)
        & np.isclose(df["volume"].fillna(0.0).astype(float), 1000.0, rtol=1e-6, atol=1e-6)
    )
    if fixture_mask.any() and (~fixture_mask).any():
        non_fixture = df.loc[~fixture_mask]
        non_fixture_median = float(non_fixture["close"].astype(float).median())
        non_fixture_vols = non_fixture["volume"].dropna().astype(float)
        non_fixture_vol_median = (
            float(non_fixture_vols.median()) if not non_fixture_vols.empty else None
        )
        price_mismatch = abs(non_fixture_median - 102.0) > 25.0
        vol_mismatch = (
            non_fixture_vol_median is not None
            and (non_fixture_vol_median < 100.0 or non_fixture_vol_median > 10000.0)
        )
        if price_mismatch or vol_mismatch:
            df = non_fixture.copy()
        else:
            colliding_fixture = fixture_mask & df["date"].isin(set(non_fixture["date"]))
            if colliding_fixture.any():
                df = df.loc[~colliding_fixture].copy()

    df = df.sort_values(["date", "_is_midnight", "_raw_ts"], ascending=[True, True, True])
    df = df[~df["date"].duplicated(keep="last")].copy()

    df = df.set_index("date").sort_index()
    df = df.loc[df.index.dayofweek < 5]
    if df.empty:
        return pd.Series(dtype="float64", name=symbol)

    valid_adj = df["adj_close"].where(df["adj_close"].notna() & (df["adj_close"] > 0.0))
    effective_close = valid_adj.combine_first(df["close"]).astype("float64")
    effective_close.name = symbol
    return effective_close


def _configure_yf_cache(yf_module: Any) -> None:
    try:
        cache_dir = Path(tempfile.gettempdir()) / "py-yfinance-tzcache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf_module, "set_tz_cache_location"):
            yf_module.set_tz_cache_location(str(cache_dir))
    except Exception:
        pass


def fetch_yfinance_daily_panel(
    symbols: Iterable[str] = LONG_TERM_SYMBOLS,
    *,
    lookback_years: int = 15,
) -> pd.DataFrame:
    """Fetch multi-year daily adjusted-close panel via yfinance (fail-open)."""
    symbol_list = list(dict.fromkeys(symbols))
    if not symbol_list:
        return pd.DataFrame()
    try:
        import yfinance as yf

        _configure_yf_cache(yf)
        raw = yf.download(
            symbol_list,
            period=f"{max(5, int(lookback_years))}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = set(raw.columns.get_level_values(0))
            if "Adj Close" in lvl0 and "Close" in lvl0:
                adj = raw["Adj Close"].where(raw["Adj Close"] > 0.0)
                close = raw["Close"].where(raw["Close"] > 0.0)
                panel = adj.combine_first(close)
            elif "Close" in lvl0:
                panel = raw["Close"].where(raw["Close"] > 0.0)
            else:
                return pd.DataFrame()
        else:
            adj = raw["Adj Close"].where(raw["Adj Close"] > 0.0) if "Adj Close" in raw else None
            close = raw["Close"].where(raw["Close"] > 0.0) if "Close" in raw else None
            if adj is not None and close is not None:
                ser = adj.combine_first(close)
            elif close is not None:
                ser = close
            else:
                return pd.DataFrame()
            panel = ser.to_frame(name=symbol_list[0])

        idx = pd.to_datetime(panel.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        panel.index = idx.normalize()
        panel = panel.loc[panel.index.notna()]
        panel = panel[~panel.index.duplicated(keep="last")].sort_index()
        panel = panel.loc[panel.index.dayofweek < 5]
        return panel.astype("float64")
    except Exception as exc:
        logger.warning("yfinance 15Y long-term fetch failed (falling back to DB): %s", exc)
        return pd.DataFrame()


def load_db_daily_panel(
    session: Optional[Session],
    symbols: Iterable[str] = LONG_TERM_SYMBOLS,
) -> pd.DataFrame:
    """Load clean daily adjusted-close bars for tracked symbols from DB PriceBar."""
    if session is None:
        return pd.DataFrame()
    symbol_list = list(dict.fromkeys(symbols))
    if not symbol_list:
        return pd.DataFrame()

    try:
        rows = (
            session.query(
                PriceBar.symbol,
                PriceBar.date,
                PriceBar.open,
                PriceBar.high,
                PriceBar.low,
                PriceBar.close,
                PriceBar.volume,
                PriceBar.adj_close,
            )
            .filter(PriceBar.symbol.in_(symbol_list))
            .order_by(PriceBar.symbol.asc(), PriceBar.date.asc())
            .all()
        )
    except Exception as exc:
        logger.warning("DB PriceBar query failed in long_term loader: %s", exc)
        return pd.DataFrame()

    grouped: dict[str, list[tuple[Any, ...]]] = {sym: [] for sym in symbol_list}
    for row in rows:
        sym = row[0]
        if sym in grouped:
            grouped[sym].append(row[1:])

    series_map: dict[str, pd.Series] = {}
    for sym, sym_rows in grouped.items():
        if sym_rows:
            ser = clean_price_rows_to_series(sym_rows, sym)
            if not ser.empty:
                series_map[sym] = ser
    if not series_map:
        return pd.DataFrame()
    return pd.DataFrame(series_map).sort_index()


def fetch_fred_macro_frames(
    series_ids: Iterable[str] = tuple(FRED_SERIES_LAG_DAYS.keys()),
    *,
    timeout_seconds: float = 8.0,
) -> dict[str, pd.DataFrame]:
    """Fetch optional FRED CSV macro series fail-open."""
    import requests

    frames: dict[str, pd.DataFrame] = {}
    for series_id in series_ids:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            resp = requests.get(url, timeout=timeout_seconds)
            if resp.status_code != 200 or not resp.text:
                continue
            df = pd.read_csv(io.StringIO(resp.text))
            if df.shape[1] < 2:
                continue
            df = df.iloc[:, :2].copy()
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date", "value"]).sort_values("date")
            if not df.empty:
                frames[series_id] = df
        except Exception as exc:
            logger.debug("Optional FRED fetch failed for %s: %s", series_id, exc)
    return frames


def resample_daily_to_weekly_friday(
    daily_panel: pd.DataFrame,
    *,
    target_symbol: str = "HG=F",
) -> tuple[pd.DataFrame, Optional[str]]:
    """Resample clean weekday daily closes to weekly ``W-FRI`` frequency.

    Returns ``(weekly_df, actual_as_of_date)`` where ``actual_as_of_date`` is the
    true latest observed trading bar date (never overstated to Friday on a mid-week run).
    """
    if daily_panel is None or daily_panel.empty:
        return pd.DataFrame(), None

    df = daily_panel.copy()
    idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df.loc[df.index.notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.loc[df.index.dayofweek < 5]
    if df.empty:
        return pd.DataFrame(), None

    canonical_target = canonicalize_instrument_symbol(target_symbol)
    if canonical_target in df.columns:
        valid_target_dates = df.index[df[canonical_target].notna() & (df[canonical_target] > 0.0)]
        actual_last_dt = valid_target_dates.max() if len(valid_target_dates) else df.index.max()
    else:
        actual_last_dt = df.index.max()
    as_of_date = actual_last_dt.strftime("%Y-%m-%d") if pd.notna(actual_last_dt) else None

    weekly = df.resample("W-FRI").last()
    weekly = weekly.dropna(how="all")
    # Limit forward fill strictly to 1 holiday week so missing early years remain NaN.
    weekly = weekly.ffill(limit=1)
    if canonical_target in weekly.columns:
        weekly = weekly.loc[weekly[canonical_target].notna() & (weekly[canonical_target] > 0.0)]
    return weekly, as_of_date


def _align_fred_to_weekly(
    fred_frames: Optional[dict[str, pd.DataFrame]],
    weekly_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Lag FRED observations by publication delay and align causally to W-FRI."""
    if not fred_frames or len(weekly_index) == 0:
        return pd.DataFrame(index=weekly_index)

    aligned: dict[str, pd.Series] = {}
    for series_id, raw_df in fred_frames.items():
        if raw_df is None or raw_df.empty:
            continue
        lag_days = FRED_SERIES_LAG_DAYS.get(series_id, 7)
        if "date" in raw_df.columns and "value" in raw_df.columns:
            dates = pd.to_datetime(raw_df["date"], errors="coerce")
            vals = pd.to_numeric(raw_df["value"], errors="coerce")
        else:
            dates = pd.to_datetime(raw_df.index, errors="coerce")
            vals = pd.to_numeric(raw_df.iloc[:, 0], errors="coerce")

        ser = pd.Series(vals.to_numpy(), index=dates + pd.Timedelta(days=lag_days))
        ser = ser.dropna()
        if ser.empty:
            continue
        if getattr(ser.index, "tz", None) is not None:
            ser.index = ser.index.tz_localize(None)
        ser.index = ser.index.normalize()
        ser = ser[~ser.index.duplicated(keep="last")].sort_index()
        w_ser = ser.resample("W-FRI").last().reindex(weekly_index).ffill(limit=2)
        aligned[series_id] = w_ser

    return pd.DataFrame(aligned, index=weekly_index)


def _safe_log_ret(series: pd.Series, periods: int) -> pd.Series:
    ratio = series / series.shift(periods)
    ratio = ratio.where(ratio > 0.0)
    return np.log(ratio)


def _safe_sma_ratio(series: pd.Series, window: int) -> pd.Series:
    sma = series.rolling(window, min_periods=max(4, window // 2)).mean()
    return (series / sma.where(sma > 0.0)) - 1.0


def build_long_term_feature_panel(
    daily_or_weekly_panel: pd.DataFrame,
    *,
    target_symbol: str = "HG=F",
    fred_frames: Optional[dict[str, pd.DataFrame]] = None,
    as_of_date_override: Optional[str] = None,
    already_weekly: bool = False,
) -> LongTermPanelBundle:
    """Build causal weekly macro/trend features and horizon targets from a price panel."""
    canonical_target = canonicalize_instrument_symbol(target_symbol)
    if daily_or_weekly_panel is None or daily_or_weekly_panel.empty:
        raise ValueError("Price panel is empty")

    if already_weekly:
        weekly = daily_or_weekly_panel.copy()
        idx = pd.to_datetime(weekly.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        weekly.index = idx.normalize()
        weekly = weekly.sort_index()
        as_of_date = as_of_date_override or weekly.index.max().strftime("%Y-%m-%d")
    else:
        weekly, detected_as_of = resample_daily_to_weekly_friday(
            daily_or_weekly_panel,
            target_symbol=canonical_target,
        )
        as_of_date = as_of_date_override or detected_as_of or ""

    if weekly.empty or canonical_target not in weekly.columns:
        raise ValueError(f"Target symbol {canonical_target} missing from weekly panel")

    hg = pd.to_numeric(weekly[canonical_target], errors="coerce").astype("float64")
    hg = hg.where(hg > 0.0)
    weekly = weekly.loc[hg.notna()].copy()
    hg = hg.loc[weekly.index]

    features: dict[str, pd.Series] = {}

    # Pillar 1: Trend structure & realized volatility (strictly causal)
    r1 = _safe_log_ret(hg, 1)
    vol12 = r1.rolling(12, min_periods=8).std()
    vol26 = r1.rolling(26, min_periods=16).std()
    sigma_w = (0.6 * vol26.fillna(vol12) + 0.4 * vol12.fillna(vol26)).clip(lower=0.008)

    features["hg_ret4"] = _safe_log_ret(hg, 4)
    features["hg_ret12"] = _safe_log_ret(hg, 12)
    features["hg_ret26"] = _safe_log_ret(hg, 26)
    features["hg_ret52"] = _safe_log_ret(hg, 52)
    features["hg_sma10w_ratio"] = _safe_sma_ratio(hg, 10)
    features["hg_sma40w_ratio"] = _safe_sma_ratio(hg, 40)
    roll_max_52 = hg.rolling(52, min_periods=16).max()
    features["hg_dd52"] = (hg / roll_max_52.where(roll_max_52 > 0.0)) - 1.0
    features["hg_vol12"] = vol12 * np.sqrt(52.0)
    features["hg_vol26"] = vol26 * np.sqrt(52.0)

    # Pillar 2: USD & Macro Rates
    if "DX-Y.NYB" in weekly.columns:
        dxy = pd.to_numeric(weekly["DX-Y.NYB"], errors="coerce").where(lambda s: s > 0.0)
        features["dxy_ret12"] = _safe_log_ret(dxy, 12)
        features["dxy_ret26"] = _safe_log_ret(dxy, 26)
        features["dxy_sma26_ratio"] = _safe_sma_ratio(dxy, 26)

    fred_df = _align_fred_to_weekly(fred_frames, weekly.index)
    fred_enabled = False
    if not fred_df.empty:
        if {"DGS10", "T10YIE"}.issubset(fred_df.columns):
            real_yield = fred_df["DGS10"] - fred_df["T10YIE"]
            if real_yield.notna().sum() >= 52:
                features["real_yield_10y"] = real_yield
                features["real_yield_chg12"] = real_yield - real_yield.shift(12)
                features["real_yield_chg26"] = real_yield - real_yield.shift(26)
                be = fred_df["T10YIE"]
                be_mean52 = be.rolling(52, min_periods=26).mean()
                be_std52 = be.rolling(52, min_periods=26).std().clip(lower=0.05)
                features["breakeven_10y_z"] = (be - be_mean52) / be_std52
                fred_enabled = True
        if "DTWEXBGS" in fred_df.columns:
            dtwex = fred_df["DTWEXBGS"].where(lambda s: s > 0.0)
            if dtwex.notna().sum() >= 52:
                features["dtwex_ret26"] = _safe_log_ret(dtwex, 26)
                fred_enabled = True

    # Pillar 3: China & Industrial Demand / Commodity FX / Energy Cost
    if "GXC" in weekly.columns:
        gxc = pd.to_numeric(weekly["GXC"], errors="coerce").where(lambda s: s > 0.0)
        features["gxc_ret12"] = _safe_log_ret(gxc, 12)
        features["gxc_ret26"] = _safe_log_ret(gxc, 26)
        features["gxc_sma26_ratio"] = _safe_sma_ratio(gxc, 26)
    if "COPX" in weekly.columns:
        copx = pd.to_numeric(weekly["COPX"], errors="coerce").where(lambda s: s > 0.0)
        features["copx_ret12"] = _safe_log_ret(copx, 12)
        features["copx_vs_hg_12w"] = features["copx_ret12"] - features["hg_ret12"]
    if "FXA" in weekly.columns:
        fxa = pd.to_numeric(weekly["FXA"], errors="coerce").where(lambda s: s > 0.0)
        features["fxa_ret12"] = _safe_log_ret(fxa, 12)
        features["fxa_ret26"] = _safe_log_ret(fxa, 26)
    if "CL=F" in weekly.columns:
        cl = pd.to_numeric(weekly["CL=F"], errors="coerce").where(lambda s: s > 0.0)
        features["cl_ret12"] = _safe_log_ret(cl, 12)

    # Pillar 4: Cross-Asset / Miner Confirmation
    miner_ret12_list = []
    for sym in MINER_SYMBOLS:
        if sym in weekly.columns:
            ser = pd.to_numeric(weekly[sym], errors="coerce").where(lambda s: s > 0.0)
            r12 = _safe_log_ret(ser, 12)
            if r12.notna().sum() >= 26:
                miner_ret12_list.append(r12)
    if miner_ret12_list:
        miners_ret12 = pd.concat(miner_ret12_list, axis=1).mean(axis=1)
        features["miners_ret12"] = miners_ret12
        features["miners_vs_hg_12w"] = miners_ret12 - features["hg_ret12"]

    if "XME" in weekly.columns:
        xme = pd.to_numeric(weekly["XME"], errors="coerce").where(lambda s: s > 0.0)
        features["xme_vs_hg_12w"] = _safe_log_ret(xme, 12) - features["hg_ret12"]

    panel = pd.DataFrame(features, index=weekly.index)
    feature_columns = list(panel.columns)

    # Attach non-feature series (price, causal vol, and forward targets for training/OOS backtest).
    panel["target_price"] = hg
    panel["target_log_price"] = np.log(hg)
    panel["sigma_w"] = sigma_w
    panel["target_log_ret_4w"] = np.log(hg.shift(-4) / hg)
    panel["target_log_ret_12w"] = np.log(hg.shift(-12) / hg)

    start_date = panel.index.min().strftime("%Y-%m-%d")
    end_date = as_of_date or panel.index.max().strftime("%Y-%m-%d")
    years_covered = round(len(panel) / 52.0, 2)
    symbols_used = [sym for sym in LONG_TERM_SYMBOLS if sym in weekly.columns]
    if canonical_target not in symbols_used:
        symbols_used.insert(0, canonical_target)

    return LongTermPanelBundle(
        panel=panel,
        target_symbol=canonical_target,
        feature_columns=feature_columns,
        as_of_date=end_date,
        start_date=start_date,
        end_date=end_date,
        weekly_bars=int(len(panel)),
        years_covered=float(years_covered),
        fred_macro_enabled=fred_enabled,
        symbols_used=symbols_used,
    )


def load_long_term_weekly_panel(
    session: Optional[Session] = None,
    target_symbol: str = "HG=F",
    *,
    allow_network: bool = True,
    lookback_years: int = 15,
) -> LongTermPanelBundle:
    """Load 10-15Y copper/macro weekly feature panel from yfinance + DB + FRED."""
    canonical_target = canonicalize_instrument_symbol(target_symbol)
    symbols = list(dict.fromkeys([canonical_target, *LONG_TERM_SYMBOLS]))

    yf_daily = (
        fetch_yfinance_daily_panel(symbols, lookback_years=lookback_years)
        if allow_network
        else pd.DataFrame()
    )
    db_daily = load_db_daily_panel(session, symbols)

    if not yf_daily.empty and not db_daily.empty:
        # yfinance re-adjusts the whole equity history as of today, whereas DB adj_close rows
        # were adjusted as of their fetch date. Mixing them on overlapping dates would create
        # artificial return seams, so yfinance stays authoritative and DB bars only extend
        # the series on dates strictly after yfinance's last available bar.
        newer_db = db_daily.loc[db_daily.index > yf_daily.index.max()]
        combined = pd.concat([yf_daily, newer_db]).sort_index() if not newer_db.empty else yf_daily
        combined = combined[~combined.index.duplicated(keep="first")]
    elif not yf_daily.empty:
        combined = yf_daily
    elif not db_daily.empty:
        combined = db_daily
    else:
        raise RuntimeError(
            f"No historical price data available for long-term outlook ({canonical_target})"
        )

    fred_frames = fetch_fred_macro_frames() if allow_network else None
    return build_long_term_feature_panel(
        combined,
        target_symbol=canonical_target,
        fred_frames=fred_frames,
    )


def _stance_from_score(score: float, threshold: float = 0.15) -> StanceType:
    if score >= threshold:
        return "BULLISH"
    if score <= -threshold:
        return "BEARISH"
    return "NEUTRAL"


def _fmt_pct(val: Optional[float], signed: bool = True) -> str:
    if val is None or not np.isfinite(val):
        return "n/a"
    pct = val * 100.0
    return f"{pct:+.1f}%" if signed else f"{pct:.1f}%"


def _fmt_bps_pp(val: Optional[float]) -> str:
    if val is None or not np.isfinite(val):
        return "n/a"
    return f"{val:+.2f} pp"


def compute_macro_scorecard(panel: pd.DataFrame) -> LongTermMacroScorecard:
    """Compute the 4-pillar macro factor scorecard from the latest causal feature row."""
    if panel.empty:
        raise ValueError("Cannot build macro scorecard from empty panel")

    row = panel.iloc[-1]

    def _val(col: str) -> Optional[float]:
        v = row.get(col)
        if v is None or pd.isna(v) or not np.isfinite(float(v)):
            return None
        return float(v)

    def _z_tanh(col: str, direction: float = 1.0, fallback_scale: float = 0.08) -> Optional[float]:
        v = _val(col)
        if v is None:
            return None
        hist = pd.to_numeric(panel[col], errors="coerce").dropna().tail(156)
        if len(hist) >= 26:
            std = float(hist.std())
            if std > 1e-6:
                z = (v - float(hist.mean())) / std
                return float(np.tanh((direction * z) / 1.5))
        return float(np.tanh((direction * v) / max(fallback_scale, 1e-4)))

    # 1. Trend Structure (40W/10W trend posture + drawdown)
    sma40 = _val("hg_sma40w_ratio")
    sma10 = _val("hg_sma10w_ratio")
    dd52 = _val("hg_dd52")
    vol26 = _val("hg_vol26")
    # Moderate positive trend is bullish; extreme >20% stretch gets tempered for mean-reversion risk.
    t_parts = [
        _z_tanh("hg_sma40w_ratio", 1.0, 0.10),
        _z_tanh("hg_sma10w_ratio", 1.0, 0.06),
        _z_tanh("hg_dd52", 1.0, 0.10),
    ]
    t_valid = [x for x in t_parts if x is not None]
    trend_score = float(np.clip(np.mean(t_valid) if t_valid else 0.0, -1.0, 1.0))
    trend_stance = _stance_from_score(trend_score)
    trend_metrics = [
        LongTermPillarMetric(
            name="Price vs 40W Trend",
            value=sma40,
            formatted_value=_fmt_pct(sma40),
            signal=_stance_from_score((sma40 or 0.0) / 0.05) if sma40 is not None else "NEUTRAL",
        ),
        LongTermPillarMetric(
            name="Price vs 10W Trend",
            value=sma10,
            formatted_value=_fmt_pct(sma10),
            signal=_stance_from_score((sma10 or 0.0) / 0.03) if sma10 is not None else "NEUTRAL",
        ),
        LongTermPillarMetric(
            name="52W Drawdown",
            value=dd52,
            formatted_value=_fmt_pct(dd52),
            signal="BEARISH" if (dd52 is not None and dd52 < -0.12) else ("BULLISH" if (dd52 is not None and dd52 > -0.04) else "NEUTRAL"),
        ),
        LongTermPillarMetric(
            name="26W Annualized Vol",
            value=vol26,
            formatted_value=_fmt_pct(vol26, signed=False),
            signal="BEARISH" if (vol26 is not None and vol26 > 0.30) else "NEUTRAL",
        ),
    ]
    trend_summary = (
        "Constructive above long-term moving trend"
        if trend_stance == "BULLISH"
        else ("Trading below 40W structural trend" if trend_stance == "BEARISH" else "Range-bound near multi-month equilibrium")
    )

    # 2. USD & Macro (inverse USD 26W trend + inverse real yield tightening)
    dxy26 = _val("dxy_ret26")
    ry_chg12 = _val("real_yield_chg12")
    ry_10y = _val("real_yield_10y")
    u_parts = [
        _z_tanh("dxy_ret26", -1.0, 0.05),
        _z_tanh("dxy_sma26_ratio", -1.0, 0.03),
    ]
    if "real_yield_chg12" in panel.columns:
        u_parts.append(_z_tanh("real_yield_chg12", -1.0, 0.35))
    if "real_yield_chg26" in panel.columns:
        u_parts.append(_z_tanh("real_yield_chg26", -1.0, 0.50))
    u_valid = [x for x in u_parts if x is not None]
    usd_score = float(np.clip(np.mean(u_valid) if u_valid else 0.0, -1.0, 1.0))
    usd_stance = _stance_from_score(usd_score)
    usd_metrics = [
        LongTermPillarMetric(
            name="US Dollar 26W Trend",
            value=dxy26,
            formatted_value=_fmt_pct(dxy26),
            signal=_stance_from_score(-(dxy26 or 0.0) / 0.025) if dxy26 is not None else "NEUTRAL",
        ),
        LongTermPillarMetric(
            name="10Y Real Yield 12W Chg",
            value=ry_chg12,
            formatted_value=_fmt_bps_pp(ry_chg12),
            signal=_stance_from_score(-(ry_chg12 or 0.0) / 0.25) if ry_chg12 is not None else "NEUTRAL",
        ),
        LongTermPillarMetric(
            name="10Y US Real Yield",
            value=ry_10y,
            formatted_value=f"{ry_10y:.2f}%" if ry_10y is not None else "n/a",
            signal="BEARISH" if (ry_10y is not None and ry_10y > 2.2) else ("BULLISH" if (ry_10y is not None and ry_10y < 0.8) else "NEUTRAL"),
        ),
    ]
    usd_summary = (
        "Easing USD and real-rate pressure support copper"
        if usd_stance == "BULLISH"
        else ("Dollar strength / higher real rates weigh on metals" if usd_stance == "BEARISH" else "Dollar and real-rate impulses broadly balanced")
    )

    # 3. China / Industrial Demand (GXC, FXA commodity FX, COPX, WTI impulse)
    gxc12 = _val("gxc_ret12")
    fxa26 = _val("fxa_ret26")
    cl12 = _val("cl_ret12")
    c_parts = [
        _z_tanh("fxa_ret26", 1.0, 0.05) if "fxa_ret26" in panel.columns else None,
        _z_tanh("gxc_ret12", 1.0, 0.08) if "gxc_ret12" in panel.columns else None,
        _z_tanh("copx_ret12", 1.0, 0.12) if "copx_ret12" in panel.columns else None,
    ]
    c_valid = [x for x in c_parts if x is not None]
    china_score = float(np.clip(np.mean(c_valid) if c_valid else 0.0, -1.0, 1.0))
    china_stance = _stance_from_score(china_score)
    china_metrics = [
        LongTermPillarMetric(
            name="AUD Commodity FX 26W",
            value=fxa26,
            formatted_value=_fmt_pct(fxa26),
            signal=_stance_from_score((fxa26 or 0.0) / 0.03) if fxa26 is not None else "NEUTRAL",
        ),
        LongTermPillarMetric(
            name="China Equities (GXC) 12W",
            value=gxc12,
            formatted_value=_fmt_pct(gxc12),
            signal=_stance_from_score((gxc12 or 0.0) / 0.05) if gxc12 is not None else "NEUTRAL",
        ),
        LongTermPillarMetric(
            name="WTI Crude 12W Impulse",
            value=cl12,
            formatted_value=_fmt_pct(cl12),
            signal=_stance_from_score(-(cl12 or 0.0) / 0.12) if cl12 is not None else "NEUTRAL",
        ),
    ]
    china_summary = (
        "China equities and commodity FX signal industrial tailwind"
        if china_stance == "BULLISH"
        else ("Muted China demand and commodity FX drag" if china_stance == "BEARISH" else "Mixed industrial demand signals across China & FX")
    )

    # 4. Cross-Asset Confirmation (Miners & COPX & XME relative strength vs copper)
    miners_vs = _val("miners_vs_hg_12w")
    copx_vs = _val("copx_vs_hg_12w")
    xme_vs = _val("xme_vs_hg_12w")
    x_parts = [
        _z_tanh("miners_vs_hg_12w", 1.0, 0.08) if "miners_vs_hg_12w" in panel.columns else None,
        _z_tanh("copx_vs_hg_12w", 1.0, 0.08) if "copx_vs_hg_12w" in panel.columns else None,
        _z_tanh("xme_vs_hg_12w", 1.0, 0.08) if "xme_vs_hg_12w" in panel.columns else None,
    ]
    x_valid = [x for x in x_parts if x is not None]
    cross_score = float(np.clip(np.mean(x_valid) if x_valid else 0.0, -1.0, 1.0))
    cross_stance = _stance_from_score(cross_score)
    cross_metrics = [
        LongTermPillarMetric(
            name="Major Miners vs HG=F (12W)",
            value=miners_vs,
            formatted_value=_fmt_pct(miners_vs),
            signal=_stance_from_score((miners_vs or 0.0) / 0.04) if miners_vs is not None else "NEUTRAL",
        ),
        LongTermPillarMetric(
            name="COPX vs HG=F (12W)",
            value=copx_vs,
            formatted_value=_fmt_pct(copx_vs),
            signal=_stance_from_score((copx_vs or 0.0) / 0.04) if copx_vs is not None else "NEUTRAL",
        ),
        LongTermPillarMetric(
            name="Metals & Mining vs HG=F (12W)",
            value=xme_vs,
            formatted_value=_fmt_pct(xme_vs),
            signal=_stance_from_score((xme_vs or 0.0) / 0.04) if xme_vs is not None else "NEUTRAL",
        ),
    ]
    cross_summary = (
        "Mining equities leading physical copper higher"
        if cross_stance == "BULLISH"
        else ("Mining equities lagging physical copper" if cross_stance == "BEARISH" else "Miners moving broadly in line with spot copper")
    )

    def _pct(score: float) -> int:
        return int(np.clip(round((score + 1.0) * 50.0), 0, 100))

    pillars = {
        "trend_structure": LongTermMacroPillar(
            key="trend_structure",
            label="Trend Structure",
            score=round(trend_score, 4),
            score_pct=_pct(trend_score),
            stance=trend_stance,
            summary=trend_summary,
            metrics=trend_metrics,
        ),
        "usd_macro": LongTermMacroPillar(
            key="usd_macro",
            label="USD & Macro",
            score=round(usd_score, 4),
            score_pct=_pct(usd_score),
            stance=usd_stance,
            summary=usd_summary,
            metrics=usd_metrics,
        ),
        "china_industrial_demand": LongTermMacroPillar(
            key="china_industrial_demand",
            label="China & Industrial Demand",
            score=round(china_score, 4),
            score_pct=_pct(china_score),
            stance=china_stance,
            summary=china_summary,
            metrics=china_metrics,
        ),
        "cross_asset_confirmation": LongTermMacroPillar(
            key="cross_asset_confirmation",
            label="Cross-Asset Confirmation",
            score=round(cross_score, 4),
            score_pct=_pct(cross_score),
            stance=cross_stance,
            summary=cross_summary,
            metrics=cross_metrics,
        ),
    }
    composite_score = float(np.clip(np.mean([p.score for p in pillars.values()]), -1.0, 1.0))
    return LongTermMacroScorecard(
        composite_score=round(composite_score, 4),
        composite_stance=_stance_from_score(composite_score),
        pillars=pillars,
    )
