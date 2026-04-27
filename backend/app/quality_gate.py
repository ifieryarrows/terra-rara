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
