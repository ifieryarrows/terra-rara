"""Future decoder row construction for TFT inference."""

from __future__ import annotations

from functools import lru_cache
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay

from deep_learning.config import TFTASROConfig
from deep_learning.data.feature_store import _build_calendar_features
from deep_learning.data.regime_features import REGIME_FEATURES


class USMarketHolidayCalendar(AbstractHolidayCalendar):
    """US/NYSE/COMEX exchange holiday calendar for deterministic offline scheduling."""

    rules = [
        Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


@lru_cache(maxsize=1)
def _us_market_business_day() -> CustomBusinessDay:
    return CustomBusinessDay(calendar=USMarketHolidayCalendar())


def _future_business_dates(last_date, horizon: int) -> pd.DatetimeIndex:
    if horizon <= 0:
        return pd.DatetimeIndex([])
    base = pd.Timestamp(last_date)
    if base.tzinfo is not None:
        base = base.tz_localize(None)
    bday = _us_market_business_day()
    start = base.normalize() + bday
    return pd.DatetimeIndex(
        pd.bdate_range(start=start, periods=horizon, freq=bday).to_numpy()
    )


def build_future_decoder_rows(
    history_df: pd.DataFrame,
    horizon: int,
    cfg: TFTASROConfig,
) -> pd.DataFrame:
    """
    Build no-lookahead future decoder rows.

    Known future values are limited to calendar features. Unknown future
    covariates receive deterministic placeholders that never use future price,
    news, embedding, or event information.
    """
    if history_df.empty:
        raise ValueError("history_df must contain at least one encoder row")

    last = history_df.iloc[-1].copy()
    future_index = _future_business_dates(history_df.index.max(), horizon)
    future = pd.DataFrame([last.to_dict()] * horizon, index=future_index)

    for col in future.columns:
        if col == "group_id":
            future[col] = last.get(col, "copper")
        elif col == "time_idx":
            start_idx = int(history_df["time_idx"].iloc[-1]) + 1 if "time_idx" in history_df else len(history_df)
            future[col] = range(start_idx, start_idx + horizon)

    target_cols = {
        cfg.forecast.model_daily_target_col,
        cfg.forecast.auxiliary_target_col,
        cfg.forecast.primary_target_col,
        "realized_vol_20d",
        "material_move_5d",
    }
    for col in target_cols:
        if col in future.columns:
            future[col] = 0.0

    calendar = _build_calendar_features(future_index)
    for col in calendar.columns:
        if col in future.columns:
            future[col] = calendar[col].values

    neutral_exact = {
        "sentiment_index",
        "news_count",
        "material_news_count",
        "after_close_news_count",
        "event_shock_score",
        "sentiment_x_supply_shock",
        "sentiment_x_usd_pressure",
        "sentiment_x_risk_on",
        "event_shock_x_high_vol",
    }
    for col in future.columns:
        lower = col.lower()
        if col in neutral_exact or col.startswith("emb_pca_") or col.startswith("evt_"):
            future[col] = 0.0
        elif "ret" in lower or "roc" in lower or "momentum" in lower:
            future[col] = 0.0

    for col in REGIME_FEATURES:
        if col in future.columns and col != "event_shock_score":
            future[col] = float(last.get(col, 0.0))

    prev_days = float(last.get("days_since_last_material_news", 999.0))
    if "days_since_last_material_news" in future.columns:
        future["days_since_last_material_news"] = [prev_days + i for i in range(1, horizon + 1)]
    if "stale_sentiment_flag" in future.columns:
        future["stale_sentiment_flag"] = (
            future.get("days_since_last_material_news", pd.Series(999.0, index=future.index)) >= 3
        ).astype(float)

    return future[history_df.columns].copy()
