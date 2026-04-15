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
) -> dict:
    """
    Walk-forward backtest of the Theta model.

    Slides a window across the series and evaluates directional accuracy
    and return metrics at each step.

    Returns:
        Dict with aggregate metrics for comparison against TFT-ASRO.
    """
    if len(series) < 60 + n_windows:
        n_windows = max(10, len(series) - 60)

    results = []
    for i in range(n_windows):
        end_idx = len(series) - n_windows + i
        if end_idx < 60:
            continue

        train = series.iloc[:end_idx]
        actual_future = series.iloc[end_idx:end_idx + horizon]

        if len(actual_future) < 1:
            continue

        fc = theta_forecast(train, horizon=min(horizon, len(actual_future)))
        if "error" in fc:
            continue

        pred_ret = fc["predicted_return_1d"]
        actual_ret = (float(actual_future.iloc[0]) / float(train.iloc[-1])) - 1.0

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
    sr_std = strategy_returns.std()
    sharpe = float(np.sqrt(252) * strategy_returns.mean() / sr_std) if sr_std > 1e-9 else 0.0

    return {
        "model": "Theta",
        "n_windows": len(results),
        "directional_accuracy": da,
        "sharpe_ratio": sharpe,
        "mae": float(np.abs(df["pred_return"] - df["actual_return"]).mean()),
        "mean_return": float(strategy_returns.mean()),
    }
