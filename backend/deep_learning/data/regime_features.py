"""Regime and event-conditioning features for weekly TFT forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_FEATURES = [
    "regime_risk_on_demand",
    "regime_risk_off_macro",
    "regime_usd_pressure",
    "regime_supply_shock",
    "regime_inventory_tightness",
    "regime_high_vol_chop",
    "event_shock_score",
    "sentiment_x_supply_shock",
    "sentiment_x_usd_pressure",
    "sentiment_x_risk_on",
    "event_shock_x_high_vol",
]

FORCED_TFT_UNKNOWN_FEATURES = [
    "sentiment_index",
    "news_count",
    "material_news_count",
    "after_close_news_count",
    "days_since_last_material_news",
    "stale_sentiment_flag",
    "regime_risk_on_demand",
    "regime_risk_off_macro",
    "regime_usd_pressure",
    "regime_supply_shock",
    "regime_inventory_tightness",
    "regime_high_vol_chop",
    "event_shock_score",
]


def _zero(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index)


def _zscore(s: pd.Series, window: int = 60, min_periods: int = 20) -> pd.Series:
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return ((s - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _first_present_series(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    idx: pd.Index,
) -> pd.Series:
    """Return the first matching numeric column as a clean float Series."""
    for name in candidates:
        if name in df.columns:
            return pd.Series(df[name], index=idx).fillna(0.0).astype(float)
    return _zero(idx)


def build_regime_event_features(master_like: pd.DataFrame) -> pd.DataFrame:
    """
    Build market regime and event conditioning features.

    Missing optional source columns are treated as neutral zero so the function
    remains stable across training, local tests, and production inference.
    """
    idx = master_like.index
    out = pd.DataFrame(index=idx)

    sentiment = pd.Series(master_like.get("sentiment_index", _zero(idx)), index=idx).fillna(0.0).astype(float)
    news_count = pd.Series(master_like.get("news_count", _zero(idx)), index=idx).fillna(0.0).astype(float)

    dxy_ret = _first_present_series(
        master_like,
        ("DX-Y.NYB_ret1", "DX-Y_NYB_ret1", "DX_Y_NYB_ret1"),
        idx,
    )
    china_ret = _first_present_series(
        master_like,
        ("GXC_ret1", "FXI_ret1", "MCHI_ret1", "ASHR_ret1"),
        idx,
    )

    lme_draw = pd.Series(master_like.get("lme_stock_change_5d", _zero(idx)), index=idx).fillna(0.0).astype(float)
    cancelled_ratio = pd.Series(master_like.get("lme_cancelled_ratio", _zero(idx)), index=idx).fillna(0.0).astype(float)
    curve_slope = pd.Series(master_like.get("futures_curve_slope", _zero(idx)), index=idx).fillna(0.0).astype(float)
    proxy_vol_spike = pd.Series(master_like.get("proxy_vol_spike", _zero(idx)), index=idx).fillna(0.0).astype(float)
    supply_count = pd.Series(master_like.get("evt_supply_disruption_count", _zero(idx)), index=idx).fillna(0.0).astype(float)
    inventory_draw_count = pd.Series(master_like.get("evt_inventory_draw_count", _zero(idx)), index=idx).fillna(0.0).astype(float)

    # Regime inputs must be observable at the forecast origin. ``target`` is
    # a forward return label, so it must never participate in a feature even
    # through a rolling statistic. The feature store supplies the historical
    # close-to-close log return under this explicit causal name.
    observed_return = master_like.get("observed_return_1d", _zero(idx))
    realized_vol = pd.Series(observed_return, index=idx).rolling(20, min_periods=10).std().fillna(0.0)

    vol_z = _zscore(realized_vol, 60, 20)
    sent_z = _zscore(sentiment, 60, 20)
    lme_draw_z = _zscore(-lme_draw, 60, 20)
    dxy_5d = dxy_ret.rolling(5, min_periods=1).sum()
    china_5d = china_ret.rolling(5, min_periods=1).sum()

    # Pure macro regime indicators remain defined even across multi-year history
    # where daily news sentiment is unavailable; sentiment interactions live below.
    out["regime_usd_pressure"] = (dxy_5d > 0.01).astype(float)
    out["regime_risk_on_demand"] = ((china_5d > 0.01) & (dxy_5d < 0)).astype(float)
    out["regime_risk_off_macro"] = ((china_5d < -0.01) & (dxy_5d > 0)).astype(float)
    out["regime_supply_shock"] = ((supply_count > 0) | (inventory_draw_count > 0)).astype(float)

    has_lme_feed = bool(lme_draw.abs().max() > 1e-12 or cancelled_ratio.abs().max() > 1e-12)
    if has_lme_feed:
        out["regime_inventory_tightness"] = (
            (lme_draw_z > 1.0)
            | (cancelled_ratio > cancelled_ratio.rolling(60, min_periods=20).mean())
        ).astype(float)
    else:
        slope_z = _zscore(-curve_slope, 60, 20)
        out["regime_inventory_tightness"] = (
            (slope_z > 1.0) | (proxy_vol_spike > 0.0)
        ).astype(float)

    out["regime_high_vol_chop"] = (vol_z > 1.0).astype(float)

    event_importance = (
        1.50 * supply_count
        + 1.35 * inventory_draw_count
        + 1.00 * news_count.clip(upper=5)
    )

    out["event_shock_score"] = (
        sent_z.abs()
        * np.log1p(news_count.clip(lower=0))
        * (1.0 + event_importance)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["sentiment_x_supply_shock"] = sentiment * out["regime_supply_shock"]
    out["sentiment_x_usd_pressure"] = sentiment * out["regime_usd_pressure"]
    out["sentiment_x_risk_on"] = sentiment * out["regime_risk_on_demand"]
    out["event_shock_x_high_vol"] = out["event_shock_score"] * out["regime_high_vol_chop"]

    return out.astype("float32")
