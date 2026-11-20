"""
Theta Model Baseline for Copper Price Forecasting.

The Theta method decomposes a time series into a linear trend (theta=0)
and a curvature-amplified component (theta=2), forecasts each separately
using Simple Exponential Smoothing, then combines.  It won the M3
forecasting competition.

This serves as a simple but strong baseline against which TFT-ASRO
performance is measured.  If TFT cannot beat Theta consistently, the
added complexity is not justified.

Reference:
    Assimakopoulos & Nikolopoulos (2000) "The theta model:
    a decomposition approach to forecasting" (IJF)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def theta_forecast(
    series: pd.Series,
    horizon: int = 5,
    theta: float = 2.0,
) -> dict:
    """
    Generate Theta model forecasts.

    Args:
        series:  Historical closing prices (DatetimeIndex, sorted ascending).
        horizon: Number of days ahead to forecast.
        theta:   Theta parameter (2.0 = standard Theta method).

    Returns:
        Dict with predicted returns, prices, and direction.
    """
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    if len(series) < 30:
        return {"error": f"Theta needs >=30 observations, got {len(series)}"}

    y = series.values.astype(np.float64)
    n = len(y)

    # Linear regression (theta=0 line)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = intercept + slope * x

    # Theta=2 component: amplify curvature
    theta_component = theta * y - (theta - 1) * trend_line

    try:
        ses = SimpleExpSmoothing(theta_component, initialization_method="estimated")
        fit = ses.fit(optimized=True)
        ses_forecast = fit.forecast(horizon)
    except Exception as exc:
        logger.warning("SES fit failed, using last value: %s", exc)
        ses_forecast = np.full(horizon, theta_component[-1])

    # Combine: extrapolate trend + SES forecast / theta
    future_x = np.arange(n, n + horizon)
    trend_forecast = intercept + slope * future_x
    combined = (ses_forecast + (theta - 1) * trend_forecast) / theta

    last_price = float(y[-1])
    daily_returns = np.diff(np.concatenate([[last_price], combined])) / np.concatenate([[last_price], combined[:-1]])
    cumulative_return = (combined[-1] / last_price) - 1.0

    direction = "BULLISH" if cumulative_return > 0.005 else ("BEARISH" if cumulative_return < -0.005 else "NEUTRAL")

    daily_forecasts = []
    for d in range(horizon):
        daily_forecasts.append({
            "day": d + 1,
            "price": float(combined[d]),
            "daily_return": float(daily_returns[d]) if d < len(daily_returns) else 0.0,
        })

    return {
        "model": "Theta",
        "direction": direction,
        "horizon": horizon,
        "predicted_return_1d": float(daily_returns[0]) if len(daily_returns) > 0 else 0.0,
        "cumulative_return": float(cumulative_return),
        "predicted_price_final": float(combined[-1]),
        "daily_forecasts": daily_forecasts,
        "baseline_price": last_price,
        "theta": theta,
        "trend_slope": float(slope),
    }


def theta_backtest(
    series: pd.Series,
    horizon: int = 5,
    n_windows: int = 50,
    return_mode: str = "1d",
    use_log_return: bool = False,
    annualisation: Optional[float] = None,
    eval_dates: Optional[pd.DatetimeIndex] = None,
) -> dict:
    """
    Walk-forward backtest of the Theta model.

    Slides a window across the series (or evaluates at exact ``eval_dates``
    forecast origins when supplied) and evaluates directional accuracy
    and return metrics at each step.

    Args:
        series:         Historical closing prices sorted ascending.
        horizon:        Forecast horizon in trading days.
        n_windows:      Number of rolling evaluation steps.
        return_mode:    ``"1d"`` evaluates next-day return; ``"cumulative"`` evaluates
                        ``horizon``-day cumulative return to match TFT weekly evaluation.
        use_log_return: If True, evaluate in log-return space rather than simple return.
        annualisation:  Optional annualisation factor (defaults to 252 for 1d, 52 for cumulative).
        eval_dates:     Optional forecast-origin dates to align evaluation 1-to-1 with TFT.

    Returns:
        Dict with aggregate metrics for comparison against TFT-ASRO.
    """
    is_cumulative = str(return_mode).lower() == "cumulative"
    h_req = horizon if is_cumulative else 1

    window_slices: list[tuple[pd.Series, pd.Series]] = []
    if eval_dates is not None and len(eval_dates) > 0:
        series_norm = series.copy()
        idx_norm = pd.DatetimeIndex(pd.to_datetime(series_norm.index))
        if idx_norm.tz is not None:
            idx_norm = idx_norm.tz_localize(None)
        series_norm.index = idx_norm.normalize()
        series_norm = series_norm[~series_norm.index.duplicated(keep="last")].sort_index()

        dates_norm = pd.DatetimeIndex(pd.to_datetime(eval_dates))
        if dates_norm.tz is not None:
            dates_norm = dates_norm.tz_localize(None)
        dates_norm = dates_norm.normalize()

        for origin_dt in dates_norm:
            if pd.isna(origin_dt):
                continue
            pos = int(series_norm.index.searchsorted(origin_dt, side="right")) - 1
            if pos < 59 or pos + 1 + h_req > len(series_norm):
                continue
            train = series_norm.iloc[: pos + 1]
            actual_future = series_norm.iloc[pos + 1 : pos + 1 + horizon]
            window_slices.append((train, actual_future))
    else:
        if len(series) < 60 + n_windows:
            n_windows = max(10, len(series) - 60)

        for i in range(n_windows):
            end_idx = len(series) - n_windows + i
            if end_idx < 60:
                continue

            train = series.iloc[:end_idx]
            actual_future = series.iloc[end_idx:end_idx + horizon]
            if len(actual_future) < h_req:
                continue
            window_slices.append((train, actual_future))

    results = []
    for train, actual_future in window_slices:
        fc = theta_forecast(train, horizon=min(horizon, len(actual_future)))
        if "error" in fc:
            continue

        last_price = float(train.iloc[-1])
        if is_cumulative:
            target_price = float(actual_future.iloc[h_req - 1])
            forecasts = fc.get("daily_forecasts") or []
            pred_price = (
                float(forecasts[h_req - 1]["price"])
                if len(forecasts) >= h_req
                else float(fc["predicted_price_final"])
            )
            if use_log_return:
                actual_ret = float(np.log(max(target_price, 1e-8) / max(last_price, 1e-8)))
                pred_ret = float(np.log(max(pred_price, 1e-8) / max(last_price, 1e-8)))
            else:
                actual_ret = (target_price / max(last_price, 1e-8)) - 1.0
                pred_ret = (pred_price / max(last_price, 1e-8)) - 1.0
        else:
            target_price = float(actual_future.iloc[0])
            if use_log_return:
                actual_ret = float(np.log(max(target_price, 1e-8) / max(last_price, 1e-8)))
                pred_ret = float(np.log1p(max(float(fc["predicted_return_1d"]), -0.99)))
            else:
                pred_ret = float(fc["predicted_return_1d"])
                actual_ret = (target_price / max(last_price, 1e-8)) - 1.0

        results.append({
            "pred_return": pred_ret,
            "actual_return": actual_ret,
            "correct_direction": (pred_ret > 0) == (actual_ret > 0),
        })

    if not results:
        return {"error": "No valid backtest windows"}

    df = pd.DataFrame(results)
    da = float(df["correct_direction"].mean())
    strategy_returns = np.sign(df["pred_return"].values) * df["actual_return"].values
    ann = float(annualisation) if annualisation is not None else (52.0 if is_cumulative else 252.0)

    if is_cumulative:
        from deep_learning.training.metrics import sharpe_ratio_hac

        sharpe = float(sharpe_ratio_hac(strategy_returns, annualisation=ann))
    else:
        sr_std = strategy_returns.std()
        sharpe = float(np.sqrt(ann) * strategy_returns.mean() / sr_std) if sr_std > 1e-9 else 0.0

    mae = float(np.abs(df["pred_return"] - df["actual_return"]).mean())
    naive_zero_mae = float(np.abs(df["actual_return"]).mean())

    return {
        "model": "Theta",
        "n_windows": len(results),
        "return_mode": return_mode,
        "use_log_return": use_log_return,
        "directional_accuracy": da,
        "sharpe_ratio": sharpe,
        "mae": mae,
        "mae_vs_naive_zero": float(mae / naive_zero_mae) if naive_zero_mae > 1e-12 else 1.0,
        "mean_return": float(strategy_returns.mean()),
    }
