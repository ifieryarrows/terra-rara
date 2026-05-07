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


def build_regime_event_features(master_like: pd.DataFrame) -> pd.DataFrame:
    """
    Build market regime and event conditioning features.

    Missing optional source columns are treated as neutral zero so the function
    remains stable across training, local tests, and production inference.
    """
    idx = master_like.index
    out = pd.DataFrame(index=idx)

    sentiment = master_like.get("sentiment_index", _zero(idx)).astype(float)
    news_count = master_like.get("news_count", _zero(idx)).astype(float)

    dxy_ret = (
        master_like.get("DX-Y_NYB_ret1")
        if "DX-Y_NYB_ret1" in master_like.columns
        else master_like.get("DX_Y_NYB_ret1", _zero(idx))
    )
    dxy_ret = pd.Series(dxy_ret, index=idx).fillna(0.0).astype(float)

    fxi_ret = master_like.get("FXI_ret1", _zero(idx)).fillna(0.0).astype(float)
    crude_ret = master_like.get(
        "CL=F_ret1",
        master_like.get("CL_F_ret1", _zero(idx)),
    ).fillna(0.0).astype(float)
    _ = crude_ret

    lme_draw = master_like.get("lme_stock_change_5d", _zero(idx)).fillna(0.0).astype(float)
    cancelled_ratio = master_like.get("lme_cancelled_ratio", _zero(idx)).fillna(0.0).astype(float)
    supply_count = master_like.get("evt_supply_disruption_count", _zero(idx)).fillna(0.0).astype(float)
    inventory_draw_count = master_like.get("evt_inventory_draw_count", _zero(idx)).fillna(0.0).astype(float)

    if "target" in master_like.columns:
        realized_vol = master_like["target"].rolling(20, min_periods=10).std().fillna(0.0)
    else:
        realized_vol = _zero(idx)

    vol_z = _zscore(realized_vol, 60, 20)
    sent_z = _zscore(sentiment, 60, 20)
    lme_draw_z = _zscore(-lme_draw, 60, 20)
    dxy_5d = dxy_ret.rolling(5, min_periods=1).sum()
    fxi_5d = fxi_ret.rolling(5, min_periods=1).sum()

    out["regime_usd_pressure"] = ((dxy_5d > 0.01) & (sentiment < 0)).astype(float)
    out["regime_risk_on_demand"] = ((fxi_5d > 0.01) & (dxy_5d < 0)).astype(float)
    out["regime_risk_off_macro"] = ((fxi_5d < -0.01) & (dxy_5d > 0)).astype(float)
    out["regime_supply_shock"] = ((supply_count > 0) | (inventory_draw_count > 0)).astype(float)
    out["regime_inventory_tightness"] = (
        (lme_draw_z > 1.0)
        | (cancelled_ratio > cancelled_ratio.rolling(60, min_periods=20).mean())
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
