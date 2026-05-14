"""
Shared TFT quality-gate helper.

Single source of truth for the deployment thresholds used both by:
  - the API (`/api/models/tft/summary`, `backend/app/main.py`)
  - the CI script (`backend/scripts/tft_quality_gate.py`)

Lives under the `app` package so the HF production container (which copies
`backend/app/` but does NOT copy `backend/scripts/`) can import it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


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
    weekly_median_sort_gap_max: Optional[float] = None,
    weekly_sample_count: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """
    Evaluate TFT-ASRO metrics against deployment thresholds.

    Returns:
        (passed, reasons) — passed is True when no threshold is violated;
        otherwise reasons contains a human-readable explanation for each
        breach. Thresholds align with the Sprint-1 quality gate defined in
        docs/reports/tft-asro-sprint1-kapsamli-iyilestirme-*.md.
    """
    reasons: list[str] = []
    sample_count = int(weekly_sample_count or 0)
    min_weekly_da = 0.51 if sample_count and sample_count < 80 else 0.53

    if weekly_directional_accuracy is None:
        reasons.append("Missing weekly_directional_accuracy")
    elif weekly_directional_accuracy < min_weekly_da:
        reasons.append(f"WeeklyDA={weekly_directional_accuracy:.4f} < {min_weekly_da:.2f}")

    if weekly_magnitude_ratio is None:
        reasons.append("Missing weekly_magnitude_ratio")
    elif weekly_magnitude_ratio < 0.65 or weekly_magnitude_ratio > 1.35:
        reasons.append(f"WeeklyMagnitudeRatio={weekly_magnitude_ratio:.4f} outside [0.65, 1.35]")
        if weekly_magnitude_ratio > 3.0:
            reasons.append(f"WeeklyMagnitudeExplosion={weekly_magnitude_ratio:.4f} > 3.0")

    if weekly_tail_capture_rate is None:
        reasons.append("Missing weekly_tail_capture_rate")
    elif weekly_tail_capture_rate < 0.45:
        reasons.append(f"WeeklyTailCapture={weekly_tail_capture_rate:.4f} < 0.45")

    if weekly_pi80_coverage is None:
        reasons.append("Missing weekly_pi80_coverage")
    elif weekly_pi80_coverage < 0.74 or weekly_pi80_coverage > 0.86:
        reasons.append(f"WeeklyPI80={weekly_pi80_coverage:.4f} outside [0.74, 0.86]")

    if weekly_pi80_width_ratio is None:
        reasons.append("Missing weekly_pi80_width_ratio")
    elif weekly_pi80_width_ratio > 2.0 and weekly_pi80_coverage is not None and weekly_pi80_coverage > 0.86:
        reasons.append(
            f"WeeklyPI80Overwide={weekly_pi80_width_ratio:.4f} with coverage={weekly_pi80_coverage:.4f}"
        )
    if weekly_pi80_width is not None and weekly_pi80_width < 0.0:
        reasons.append(f"WeeklyPI80Width={weekly_pi80_width:.4f} < 0.0")

    if weekly_pi96_coverage is None:
        reasons.append("Missing weekly_pi96_coverage")

    if weekly_pi96_width_ratio is None:
        reasons.append("Missing weekly_pi96_width_ratio")
    elif weekly_pi96_width_ratio > 3.0:
        reasons.append(f"WeeklyPI96WidthRatio={weekly_pi96_width_ratio:.4f} > 3.0")
    if weekly_pi96_width is not None and weekly_pi96_width < 0.0:
        reasons.append(f"WeeklyPI96Width={weekly_pi96_width:.4f} < 0.0")

    if weekly_quantile_crossing_rate is None:
        reasons.append("Missing weekly_quantile_crossing_rate")
    elif weekly_quantile_crossing_rate > 0.001:
        raise AssertionError(
            f"WeeklyPublicQuantileCrossing={weekly_quantile_crossing_rate:.4f} > 0.001"
        )

    if weekly_sorted_quantile_crossing_rate is None:
        reasons.append("Missing weekly_sorted_quantile_crossing_rate")
    elif weekly_sorted_quantile_crossing_rate > 0.001:
        raise AssertionError(
            f"WeeklyOrderedQuantileCrossing={weekly_sorted_quantile_crossing_rate:.4f} > 0.001"
        )

    if weekly_median_sort_gap_max is not None and weekly_median_sort_gap_max > 0.001:
        raise AssertionError(
            f"WeeklyOrderedMedianSortGapMax={weekly_median_sort_gap_max:.4f} > 0.001"
        )

    if sharpe < -0.30:
        reasons.append(f"Sharpe={sharpe:.4f} < -0.30")
    if tail_capture is not None and tail_capture < 0.35:
        reasons.append(f"TailCapture={tail_capture:.4f} < 0.35")
    if quantile_crossing_rate is None:
        reasons.append("Missing quantile_crossing_rate")
    elif quantile_crossing_rate > 0.001:
        raise AssertionError(f"PublicQuantileCrossing={quantile_crossing_rate:.4f} > 0.001")
    if median_sort_gap_max is not None and median_sort_gap_max > 0.001:
        raise AssertionError(f"OrderedMedianSortGapMax={median_sort_gap_max:.4f} > 0.001")
    if pi80_width is not None and pi80_width < 0.0:
        reasons.append(f"PI80Width={pi80_width:.4f} < 0.0")
    if pi96_width is not None and pi96_width < 0.0:
        reasons.append(f"PI96Width={pi96_width:.4f} < 0.0")

    return len(reasons) == 0, reasons


def evaluate_quality_gate_warnings(
    vr: float,
    mae_vs_naive_zero: Optional[float] = None,
    weekly_mae_vs_naive_zero: Optional[float] = None,
) -> List[str]:
    """Return stabilization warnings that do not fail promotion yet."""
    warnings: list[str] = []
    if vr > 2.5:
        warnings.append(f"VR={vr:.4f} > 2.5 - model overdispersed")
    if vr < 0.4:
        warnings.append(f"VR={vr:.4f} < 0.4 - model underdispersed")
    if mae_vs_naive_zero is not None and mae_vs_naive_zero > 1.25:
        warnings.append(
            f"MAEvsNaiveZero={mae_vs_naive_zero:.4f} > 1.25 - worse than warning baseline"
        )
    if weekly_mae_vs_naive_zero is not None and weekly_mae_vs_naive_zero > 1.25:
        warnings.append(
            f"WeeklyMAEvsNaiveZero={weekly_mae_vs_naive_zero:.4f} > 1.25 - worse than warning baseline"
        )
    return warnings
