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
    weekly_directional_accuracy: Optional[float] = None,
    weekly_magnitude_ratio: Optional[float] = None,
    weekly_tail_capture_rate: Optional[float] = None,
    weekly_pi80_coverage: Optional[float] = None,
    weekly_quantile_crossing_rate: Optional[float] = None,
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

    if weekly_tail_capture_rate is None:
        reasons.append("Missing weekly_tail_capture_rate")
    elif weekly_tail_capture_rate < 0.45:
        reasons.append(f"WeeklyTailCapture={weekly_tail_capture_rate:.4f} < 0.45")

    if weekly_pi80_coverage is None:
        reasons.append("Missing weekly_pi80_coverage")
    elif weekly_pi80_coverage < 0.74 or weekly_pi80_coverage > 0.86:
        reasons.append(f"WeeklyPI80={weekly_pi80_coverage:.4f} outside [0.74, 0.86]")

    if weekly_quantile_crossing_rate is not None and weekly_quantile_crossing_rate > 0.10:
        reasons.append(f"WeeklyQuantileCrossing={weekly_quantile_crossing_rate:.4f} > 0.10")

    if weekly_median_sort_gap_max is not None and weekly_median_sort_gap_max > 0.005:
        reasons.append(f"WeeklyMedianSortGapMax={weekly_median_sort_gap_max:.4f} > 0.005")

    if da < 0.49:
        reasons.append(f"DA={da:.4f} < 0.49")
    if sharpe < -0.30:
        reasons.append(f"Sharpe={sharpe:.4f} < -0.30")
    if vr < 0.2 or vr > 2.5:
        reasons.append(f"VR={vr:.4f} outside [0.2, 2.5]")
    if tail_capture is not None and tail_capture < 0.35:
        reasons.append(f"TailCapture={tail_capture:.4f} < 0.35")
    if quantile_crossing_rate is not None and quantile_crossing_rate > 0.20:
        reasons.append(f"QuantileCrossing={quantile_crossing_rate:.4f} > 0.20")
    if median_sort_gap_max is not None and median_sort_gap_max > 0.01:
        reasons.append(f"MedianSortGapMax={median_sort_gap_max:.4f} > 0.01")

    return len(reasons) == 0, reasons
