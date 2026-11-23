"""
Shared TFT quality-gate helper.

Single source of truth for the deployment thresholds used both by:
  - the API (`/api/models/tft/summary`, `backend/app/main.py`)
  - the CI script (`backend/scripts/tft_quality_gate.py`)

Lives under the `app` package so the HF production container (which copies
`backend/app/` but does NOT copy `backend/scripts/`) can import it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, List, Optional, Tuple


def evaluate_quality_gate(
    da: float,
    sharpe: float,
    vr: float,
    tail_capture: Optional[float] = None,
    quantile_crossing_rate: Optional[float] = None,
    median_sort_gap_max: Optional[float] = None,
    pi80_width: Optional[float] = None,
    pi96_width: Optional[float] = None,
    weekly_directional_accuracy: Optional[float] = None,
    weekly_magnitude_ratio: Optional[float] = None,
    weekly_tail_capture_rate: Optional[float] = None,
    weekly_pi80_coverage: Optional[float] = None,
    weekly_pi80_width: Optional[float] = None,
    weekly_pi80_width_ratio: Optional[float] = None,
    weekly_pi96_coverage: Optional[float] = None,
    weekly_pi96_width: Optional[float] = None,
    weekly_pi96_width_ratio: Optional[float] = None,
    weekly_quantile_crossing_rate: Optional[float] = None,
    weekly_sorted_quantile_crossing_rate: Optional[float] = None,
    weekly_ordered_quantile_crossing_rate: Optional[float] = None,
    weekly_median_sort_gap_max: Optional[float] = None,
    weekly_sample_count: Optional[int] = None,
    weekly_pred_positive_rate: Optional[float] = None,
    weekly_actual_positive_rate: Optional[float] = None,
    weekly_raw_magnitude_ratio: Optional[float] = None,
    weekly_median_bound_applied_rate: Optional[float] = None,
    weekly_sharpe_ratio: Optional[float] = None,
    weekly_sharpe_ratio_hac: Optional[float] = None,
    weekly_mae_vs_naive_zero: Optional[float] = None,
    weekly_pinball_skill_vs_empirical: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """
    Evaluate TFT-ASRO metrics against deployment thresholds.

    Focuses hard promotion gates on structural invariants and primary 5-day
    weekly performance/calibration (sample-size aware PI80/PI96, proper-score
    MAE/pinball guard, HAC Sharpe, and non-collapsed magnitude/direction).
    Single-step T+1 diagnostics and noisy tail-subset rates are surfaced via
    non-blocking warnings so XGBoost and TFT-ASRO maintain clean horizon separation.
    """
    _ = (da, sharpe, vr, tail_capture, weekly_tail_capture_rate, weekly_pinball_skill_vs_empirical)
    reasons: list[str] = []
    sample_count = int(weekly_sample_count or 0)
    n_eff = max(float(sample_count or 100) / 5.0, 1.0)
    if sample_count == 0 or sample_count >= 300:
        min_weekly_da = 0.49
    elif sample_count >= 150:
        min_weekly_da = 0.47
    else:
        min_weekly_da = 0.45

    if weekly_directional_accuracy is None:
        reasons.append("Missing weekly_directional_accuracy")
    elif weekly_directional_accuracy < min_weekly_da:
        reasons.append(f"WeeklyDA={weekly_directional_accuracy:.4f} < {min_weekly_da:.2f}")

    if weekly_magnitude_ratio is None:
        reasons.append("Missing weekly_magnitude_ratio")
    elif weekly_magnitude_ratio < 0.02:
        reasons.append(f"WeeklyMagnitudeCollapse={weekly_magnitude_ratio:.4f} < 0.02")
    elif weekly_magnitude_ratio > 2.50:
        reasons.append(f"WeeklyMagnitudeRatio={weekly_magnitude_ratio:.4f} > 2.50")
        if weekly_magnitude_ratio > 3.0:
            reasons.append(f"WeeklyMagnitudeExplosion={weekly_magnitude_ratio:.4f} > 3.0")

    if weekly_raw_magnitude_ratio is None:
        reasons.append("Missing weekly_raw_magnitude_ratio")
    elif weekly_raw_magnitude_ratio > 3.0:
        reasons.append(f"WeeklyRawMagnitudeExplosion={weekly_raw_magnitude_ratio:.4f} > 3.0")

    if weekly_median_bound_applied_rate is None:
        reasons.append("Missing weekly_median_bound_applied_rate")

    if weekly_mae_vs_naive_zero is None or not math.isfinite(weekly_mae_vs_naive_zero):
        reasons.append("Missing/invalid weekly_mae_vs_naive_zero")
    elif weekly_mae_vs_naive_zero > 1.12:
        reasons.append(f"WeeklyMAEvsNaiveZero={weekly_mae_vs_naive_zero:.4f} > 1.12")

    if weekly_pred_positive_rate is None:
        reasons.append("Missing weekly_pred_positive_rate")
    if weekly_actual_positive_rate is None:
        reasons.append("Missing weekly_actual_positive_rate")
    if weekly_pred_positive_rate is not None and weekly_actual_positive_rate is not None:
        positive_collapse = (
            weekly_pred_positive_rate > 0.92 and weekly_actual_positive_rate < 0.67
        )
        negative_collapse = (
            weekly_pred_positive_rate < 0.08 and weekly_actual_positive_rate > 0.33
        )
        if positive_collapse or negative_collapse:
            reasons.append(
                "WeeklySignCollapse="
                f"pred_positive_rate={weekly_pred_positive_rate:.4f}, "
                f"actual_positive_rate={weekly_actual_positive_rate:.4f}"
            )

    pi80_tol = max(0.08, min(0.18, 1.65 * math.sqrt(0.80 * 0.20 / max(n_eff, 10.0))))
    pi80_low = round(0.80 - pi80_tol, 4)
    pi80_high = round(min(0.98, 0.80 + pi80_tol), 4)
    if weekly_pi80_coverage is None:
        reasons.append("Missing weekly_pi80_coverage")
    elif weekly_pi80_coverage < pi80_low or weekly_pi80_coverage > pi80_high:
        reasons.append(
            f"WeeklyPI80={weekly_pi80_coverage:.4f} outside [{pi80_low:.2f}, {pi80_high:.2f}]"
        )

    if weekly_pi80_width_ratio is None:
        reasons.append("Missing weekly_pi80_width_ratio")
    elif weekly_pi80_width_ratio > 2.2:
        reasons.append(f"WeeklyPI80WidthRatio={weekly_pi80_width_ratio:.4f} > 2.2")
    if weekly_pi80_width is not None and weekly_pi80_width < 0.0:
        reasons.append(f"WeeklyPI80Width={weekly_pi80_width:.4f} < 0.0")

    min_pi96_cov = 0.85 if n_eff < 25 else 0.88
    if weekly_pi96_coverage is None:
        reasons.append("Missing weekly_pi96_coverage")
    elif weekly_pi96_coverage < min_pi96_cov:
        reasons.append(f"WeeklyPI96={weekly_pi96_coverage:.4f} < {min_pi96_cov:.2f}")

    if weekly_pi96_width_ratio is None:
        reasons.append("Missing weekly_pi96_width_ratio")
    elif weekly_pi96_width_ratio > 3.2:
        reasons.append(f"WeeklyPI96WidthRatio={weekly_pi96_width_ratio:.4f} > 3.2")
    if weekly_pi96_width is not None and weekly_pi96_width < 0.0:
        reasons.append(f"WeeklyPI96Width={weekly_pi96_width:.4f} < 0.0")

    if weekly_quantile_crossing_rate is None:
        reasons.append("Missing weekly_quantile_crossing_rate")
    elif weekly_quantile_crossing_rate > 0.001:
        reasons.append(
            f"WeeklyPublicQuantileCrossing={weekly_quantile_crossing_rate:.4f} > 0.001"
        )

    if weekly_sorted_quantile_crossing_rate is None:
        reasons.append("Missing weekly_sorted_quantile_crossing_rate")
    elif weekly_sorted_quantile_crossing_rate > 0.001:
        reasons.append(
            f"WeeklyOrderedQuantileCrossing={weekly_sorted_quantile_crossing_rate:.4f} > 0.001"
        )

    if (
        weekly_ordered_quantile_crossing_rate is not None
        and weekly_ordered_quantile_crossing_rate > 0.001
    ):
        reasons.append(
            f"WeeklyPostCQRQuantileCrossing={weekly_ordered_quantile_crossing_rate:.4f} > 0.001"
        )

    if weekly_median_sort_gap_max is not None and weekly_median_sort_gap_max > 0.001:
        reasons.append(
            f"WeeklyOrderedMedianSortGapMax={weekly_median_sort_gap_max:.4f} > 0.001"
        )

    effective_weekly_sharpe = (
        weekly_sharpe_ratio_hac if weekly_sharpe_ratio_hac is not None else weekly_sharpe_ratio
    )
    if effective_weekly_sharpe is None:
        reasons.append("Missing weekly_sharpe_ratio")
    elif effective_weekly_sharpe < -0.35:
        reasons.append(f"WeeklySharpe={effective_weekly_sharpe:.4f} < -0.35")

    if quantile_crossing_rate is None:
        reasons.append("Missing quantile_crossing_rate")
    elif quantile_crossing_rate > 0.001:
        reasons.append(f"PublicQuantileCrossing={quantile_crossing_rate:.4f} > 0.001")
    if median_sort_gap_max is not None and median_sort_gap_max > 0.001:
        reasons.append(f"OrderedMedianSortGapMax={median_sort_gap_max:.4f} > 0.001")
    if pi80_width is not None and pi80_width < 0.0:
        reasons.append(f"PI80Width={pi80_width:.4f} < 0.0")
    if pi96_width is not None and pi96_width < 0.0:
        reasons.append(f"PI96Width={pi96_width:.4f} < 0.0")

    return len(reasons) == 0, reasons


def _metric_float(
    metrics: Mapping[str, Any],
    name: str,
    default: Optional[float] = None,
) -> Optional[float]:
    """Return a finite numeric metric, treating malformed values as missing."""
    import math

    value = metrics.get(name, default)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def evaluate_quality_gate_metrics(
    metrics: Mapping[str, Any],
) -> Tuple[bool, List[str]]:
    """Evaluate serialized TFT metrics with the canonical promotion contract.

    Keeping this mapping beside the thresholds prevents CI, artifact health,
    DB promotion, and the API from silently omitting a newly-added metric.
    """
    da = _metric_float(metrics, "directional_accuracy", 0.5)
    sharpe = _metric_float(metrics, "sharpe_ratio")
    vr = _metric_float(metrics, "variance_ratio", 1.0)
    passed, reasons = evaluate_quality_gate(
        da=0.5 if da is None else da,
        sharpe=0.0 if sharpe is None else sharpe,
        vr=1.0 if vr is None else vr,
        tail_capture=_metric_float(metrics, "tail_capture_rate"),
        quantile_crossing_rate=_metric_float(metrics, "quantile_crossing_rate"),
        median_sort_gap_max=_metric_float(metrics, "median_sort_gap_max"),
        pi80_width=_metric_float(metrics, "pi80_width"),
        pi96_width=_metric_float(metrics, "pi96_width"),
        weekly_directional_accuracy=_metric_float(
            metrics, "weekly_directional_accuracy"
        ),
        weekly_magnitude_ratio=_metric_float(metrics, "weekly_magnitude_ratio"),
        weekly_tail_capture_rate=_metric_float(
            metrics, "weekly_tail_capture_rate"
        ),
        weekly_pi80_coverage=_metric_float(metrics, "weekly_pi80_coverage"),
        weekly_pi80_width=_metric_float(metrics, "weekly_pi80_width"),
        weekly_pi80_width_ratio=_metric_float(
            metrics, "weekly_pi80_width_ratio"
        ),
        weekly_pi96_coverage=_metric_float(metrics, "weekly_pi96_coverage"),
        weekly_pi96_width=_metric_float(metrics, "weekly_pi96_width"),
        weekly_pi96_width_ratio=_metric_float(
            metrics, "weekly_pi96_width_ratio"
        ),
        weekly_quantile_crossing_rate=_metric_float(
            metrics, "weekly_quantile_crossing_rate"
        ),
        weekly_sorted_quantile_crossing_rate=_metric_float(
            metrics, "weekly_sorted_quantile_crossing_rate"
        ),
        weekly_ordered_quantile_crossing_rate=_metric_float(
            metrics, "weekly_ordered_quantile_crossing_rate"
        ),
        weekly_median_sort_gap_max=_metric_float(
            metrics, "weekly_median_sort_gap_max"
        ),
        weekly_sample_count=_metric_float(metrics, "weekly_sample_count"),
        weekly_pred_positive_rate=_metric_float(
            metrics, "weekly_pred_positive_rate"
        ),
        weekly_actual_positive_rate=_metric_float(
            metrics, "weekly_actual_positive_rate"
        ),
        weekly_raw_magnitude_ratio=_metric_float(
            metrics, "weekly_raw_magnitude_ratio"
        ),
        weekly_median_bound_applied_rate=_metric_float(
            metrics, "weekly_median_bound_applied_rate"
        ),
        weekly_sharpe_ratio=_metric_float(metrics, "weekly_sharpe_ratio"),
        weekly_sharpe_ratio_hac=_metric_float(metrics, "weekly_sharpe_ratio_hac"),
        weekly_mae_vs_naive_zero=_metric_float(metrics, "weekly_mae_vs_naive_zero"),
        weekly_pinball_skill_vs_empirical=_metric_float(
            metrics, "weekly_pinball_skill_vs_empirical"
        ),
    )
    if sharpe is None:
        reasons.append("Missing sharpe_ratio")
    return len(reasons) == 0 and passed, reasons


def evaluate_quality_gate_warnings(
    vr: float,
    mae_vs_naive_zero: Optional[float] = None,
    weekly_mae_vs_naive_zero: Optional[float] = None,
    weekly_median_bound_applied_rate: Optional[float] = None,
    weekly_raw_magnitude_ratio: Optional[float] = None,
    weekly_sharpe_ratio: Optional[float] = None,
    sharpe_ratio: Optional[float] = None,
    tail_capture_rate: Optional[float] = None,
    weekly_tail_capture_rate: Optional[float] = None,
    weekly_magnitude_ratio: Optional[float] = None,
) -> List[str]:
    """Return stabilization warnings that do not fail promotion."""
    warnings: list[str] = []
    if vr > 2.5:
        warnings.append(f"VR={vr:.4f} > 2.5 - model overdispersed")
    if vr < 0.25:
        warnings.append(f"VR={vr:.4f} < 0.25 - T+1 model underdispersed")
    if sharpe_ratio is not None and sharpe_ratio < -0.30:
        warnings.append(f"T1Sharpe={sharpe_ratio:.4f} < -0.30 - weak T+1 diagnostic Sharpe")
    if tail_capture_rate is not None and tail_capture_rate < 0.35:
        warnings.append(f"T1TailCapture={tail_capture_rate:.4f} < 0.35 - low T+1 diagnostic tail capture")
    if weekly_tail_capture_rate is not None and weekly_tail_capture_rate < 0.30:
        warnings.append(f"WeeklyTailCapture={weekly_tail_capture_rate:.4f} < 0.30 - low extreme-week tail capture")
    if weekly_magnitude_ratio is not None and weekly_magnitude_ratio < 0.05:
        warnings.append(f"WeeklyMagnitudeRatio={weekly_magnitude_ratio:.4f} < 0.05 - near-zero median scale")
    if weekly_magnitude_ratio is not None and (weekly_magnitude_ratio < 0.15 or weekly_magnitude_ratio > 1.80):
        warnings.append(f"WeeklyMagnitudeRatio={weekly_magnitude_ratio:.4f} outside typical band [0.15, 1.80]")
    if mae_vs_naive_zero is not None and mae_vs_naive_zero > 1.25:
        warnings.append(
            f"MAEvsNaiveZero={mae_vs_naive_zero:.4f} > 1.25 - worse than warning baseline"
        )
    if weekly_mae_vs_naive_zero is not None and weekly_mae_vs_naive_zero > 1.05:
        warnings.append(
            f"WeeklyMAEvsNaiveZero={weekly_mae_vs_naive_zero:.4f} > 1.05 - slightly above zero-baseline MAE"
        )
    if weekly_median_bound_applied_rate is not None and weekly_median_bound_applied_rate > 0.30:
        warnings.append(
            f"WeeklyMedianBoundRate={weekly_median_bound_applied_rate:.2%} > 30% - significant prediction capping"
        )
    if weekly_raw_magnitude_ratio is not None and weekly_raw_magnitude_ratio > 1.8:
        warnings.append(
            f"WeeklyRawMagnitudeRatio={weekly_raw_magnitude_ratio:.2f} > 1.8 - raw model over-predicting scale"
        )
    if weekly_sharpe_ratio is not None and weekly_sharpe_ratio < 0.0:
        warnings.append(
            f"WeeklySharpe={weekly_sharpe_ratio:.4f} < 0.00 - negative sample weekly risk-adjusted return"
        )
    return warnings


def evaluate_quality_gate_metric_warnings(
    metrics: Mapping[str, Any],
) -> List[str]:
    """Evaluate non-blocking diagnostics from serialized TFT metrics."""
    vr = _metric_float(metrics, "variance_ratio", 1.0)
    weekly_sharpe = _metric_float(metrics, "weekly_sharpe_ratio_hac")
    if weekly_sharpe is None:
        weekly_sharpe = _metric_float(metrics, "weekly_sharpe_ratio")
    return evaluate_quality_gate_warnings(
        vr=1.0 if vr is None else vr,
        mae_vs_naive_zero=_metric_float(metrics, "mae_vs_naive_zero"),
        weekly_mae_vs_naive_zero=_metric_float(
            metrics, "weekly_mae_vs_naive_zero"
        ),
        weekly_median_bound_applied_rate=_metric_float(
            metrics, "weekly_median_bound_applied_rate"
        ),
        weekly_raw_magnitude_ratio=_metric_float(
            metrics, "weekly_raw_magnitude_ratio"
        ),
        weekly_sharpe_ratio=weekly_sharpe,
        sharpe_ratio=_metric_float(metrics, "sharpe_ratio"),
        tail_capture_rate=_metric_float(metrics, "tail_capture_rate"),
        weekly_tail_capture_rate=_metric_float(metrics, "weekly_tail_capture_rate"),
        weekly_magnitude_ratio=_metric_float(metrics, "weekly_magnitude_ratio"),
    )
