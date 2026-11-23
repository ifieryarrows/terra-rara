"""
Comprehensive Walk-Forward Backtest for TFT-ASRO.

Evaluates model performance across multiple time windows with full
metric reporting, comparison against Theta baseline, and ensemble
analysis.

Usage:
    python -m deep_learning.validation.backtest --windows 50

Metrics computed per window:
    - Directional Accuracy (DA)
    - Sharpe Ratio
    - Sortino Ratio
    - Variance Ratio (VR)
    - MAE / RMSE
    - Tail Capture Rate
    - Prediction Interval Coverage

Results are saved to artifacts/backtest/ for CI comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_backtest(
    y_actual: np.ndarray,
    y_pred_median: np.ndarray,
    y_pred_q10: Optional[np.ndarray] = None,
    y_pred_q90: Optional[np.ndarray] = None,
    window_size: int = 50,
    step_size: int = 10,
    annualisation: float = 252.0,
    use_hac: bool = False,
    tail_threshold: Optional[float] = None,
) -> dict:
    """
    Walk-forward backtest across overlapping windows.

    Args:
        y_actual:       Full array of actual returns (test set).
        y_pred_median:  Full array of median predictions.
        y_pred_q10:     Optional Q10 predictions.
        y_pred_q90:     Optional Q90 predictions.
        window_size:    Evaluation window size.
        step_size:      Step between consecutive windows.
        annualisation:  Annualisation factor (252.0 for daily T+1, 52.0 for 5-day cumulative).
        use_hac:        Whether to apply Newey-West HAC variance adjustment to Sharpe.

    Returns:
        Dict with per-window metrics and aggregate summary.
    """
    from deep_learning.training.metrics import (
        directional_accuracy,
        sharpe_ratio,
        sharpe_ratio_hac,
        sortino_ratio,
        tail_capture_rate,
        prediction_interval_coverage,
    )

    y_actual = np.asarray(y_actual, dtype=float)
    y_pred_median = np.asarray(y_pred_median, dtype=float)
    n = len(y_actual)
    if n == 0:
        return {"error": "No samples for backtest"}
    if n < window_size:
        window_size = n
        step_size = max(1, n)

    windows = []
    start = 0
    while start + window_size <= n:
        end = start + window_size
        ya = y_actual[start:end]
        yp = y_pred_median[start:end]

        strategy_returns = np.sign(yp) * ya
        is_multi_day = bool(use_hac or annualisation < 200.0)
        if use_hac:
            win_sharpe = sharpe_ratio_hac(
                strategy_returns,
                annualisation=annualisation,
                max_lag=4 if annualisation < 200.0 else 0,
            )
        else:
            win_sharpe = sharpe_ratio(strategy_returns, annualisation=annualisation)

        if tail_threshold is not None:
            win_tail_threshold = float(tail_threshold)
        elif is_multi_day and len(ya) > 0:
            win_tail_threshold = float(np.nanpercentile(np.abs(ya), 75))
        else:
            win_tail_threshold = 0.015

        mae_val = float(np.abs(ya - yp).mean())
        naive_zero_mae = float(np.abs(ya).mean())
        mae_vs_naive = float(mae_val / naive_zero_mae) if naive_zero_mae > 1e-12 else 1.0

        w = {
            "start": start,
            "end": end,
            "da": directional_accuracy(ya, yp),
            "sharpe": win_sharpe,
            "sortino": sortino_ratio(strategy_returns, annualisation=annualisation),
            "tail_capture": tail_capture_rate(ya, yp, tail_threshold=win_tail_threshold),
            "mae": mae_val,
            "mae_vs_naive_zero": mae_vs_naive,
            "rmse": float(np.sqrt(((ya - yp) ** 2).mean())),
            "pred_std": float(yp.std()),
            "actual_std": float(ya.std()),
        }

        if ya.std() > 1e-9:
            w["variance_ratio"] = float(yp.std() / ya.std())
        else:
            w["variance_ratio"] = 0.0

        if y_pred_q10 is not None and y_pred_q90 is not None:
            q10 = np.asarray(y_pred_q10[start:end], dtype=float)
            q90 = np.asarray(y_pred_q90[start:end], dtype=float)
            w["pi80_coverage"] = prediction_interval_coverage(ya, q10, q90)

        windows.append(w)
        start += step_size

    if not windows:
        return {"error": "No valid windows"}

    df = pd.DataFrame(windows)
    summary = {
        "n_windows": len(windows),
        "window_size": window_size,
        "step_size": step_size,
        "total_samples": n,
        "mean_da": float(df["da"].mean()),
        "std_da": float(df["da"].std()) if len(df) > 1 else 0.0,
        "min_da": float(df["da"].min()),
        "max_da": float(df["da"].max()),
        "mean_sharpe": float(df["sharpe"].mean()),
        "std_sharpe": float(df["sharpe"].std()) if len(df) > 1 else 0.0,
        "mean_sortino": float(df["sortino"].mean()),
        "mean_vr": float(df["variance_ratio"].mean()),
        "mean_mae": float(df["mae"].mean()),
        "mean_mae_vs_naive_zero": float(df["mae_vs_naive_zero"].mean()),
        "mean_tail_capture": float(df["tail_capture"].mean()),
        "da_above_50pct": float((df["da"] > 0.50).mean()),
        "sharpe_positive_pct": float((df["sharpe"] > 0).mean()),
    }

    if "pi80_coverage" in df.columns:
        summary["mean_pi80_coverage"] = float(df["pi80_coverage"].mean())

    return {
        "summary": summary,
        "windows": windows,
    }


def compare_with_baseline(
    tft_metrics: dict,
    theta_metrics: dict,
) -> dict:
    """
    Compare TFT-ASRO backtest results against Theta baseline.

    Returns:
        Dict with comparison metrics and verdict.
    """
    tft_s = tft_metrics.get("summary", {})
    theta_s = theta_metrics

    comparison = {
        "tft_da": tft_s.get("mean_da", 0),
        "theta_da": theta_s.get("directional_accuracy", 0),
        "tft_sharpe": tft_s.get("mean_sharpe", 0),
        "theta_sharpe": theta_s.get("sharpe_ratio", 0),
        "tft_mae": tft_s.get("mean_mae", 999),
        "theta_mae": theta_s.get("mae", 999),
    }

    tft_wins = 0
    if comparison["tft_da"] > comparison["theta_da"]:
        tft_wins += 1
    if comparison["tft_sharpe"] > comparison["theta_sharpe"]:
        tft_wins += 1
    if comparison["tft_mae"] < comparison["theta_mae"]:
        tft_wins += 1

    comparison["tft_wins"] = tft_wins
    comparison["theta_wins"] = 3 - tft_wins
    comparison["verdict"] = (
        "TFT_SUPERIOR" if tft_wins >= 2
        else "THETA_SUPERIOR" if tft_wins == 0
        else "MIXED"
    )

    return comparison


def save_backtest_report(
    backtest_results: dict,
    comparison: Optional[dict] = None,
    output_dir: str = "artifacts/backtest",
) -> Path:
    """Save backtest results to a timestamped JSON file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "tft_backtest": backtest_results,
    }
    if comparison:
        report["baseline_comparison"] = comparison

    path = out / f"backtest_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Backtest report saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run TFT-ASRO walk-forward backtest")
    parser.add_argument("--windows", type=int, default=50, help="Backtest window size")
    parser.add_argument("--step", type=int, default=10, help="Step size between windows")
    args = parser.parse_args()

    print(f"Backtest configured: window={args.windows}, step={args.step}")
    print("Run after training to evaluate model with actual predictions.")
