"""
CI quality gate for TFT-ASRO training.

Reads tft_metadata.json written by trainer.py and exits non-zero when
metrics fall below deployment thresholds.

Thin wrapper that delegates threshold logic to `app.quality_gate` so that
GitHub Actions CI and the FastAPI runtime always agree on the rules.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.quality_gate import evaluate_quality_gate

META_PATH = pathlib.Path(os.environ.get("TFT_METADATA_PATH", "/tmp/models/tft/tft_metadata.json"))


def main() -> int:
    if not META_PATH.exists():
        print("No metadata file found - quality gate cannot evaluate training output")
        return 1

    data = json.loads(META_PATH.read_text(encoding="utf-8-sig"))
    metrics = data.get("test_metrics", {})
    da = metrics.get("directional_accuracy", 0.5)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    vr = metrics.get("variance_ratio", 1.0)
    tail_capture = metrics.get("tail_capture_rate")
    quantile_crossing = metrics.get("quantile_crossing_rate")
    median_gap_max = metrics.get("median_sort_gap_max")
    weekly_da = metrics.get("weekly_directional_accuracy")
    weekly_mr = metrics.get("weekly_magnitude_ratio")
    weekly_tail = metrics.get("weekly_tail_capture_rate")
    weekly_pi80 = metrics.get("weekly_pi80_coverage")
    weekly_pi80_width_ratio = metrics.get("weekly_pi80_width_ratio")
    weekly_pi96 = metrics.get("weekly_pi96_coverage")
    weekly_pi96_width_ratio = metrics.get("weekly_pi96_width_ratio")
    weekly_qcross = metrics.get("weekly_quantile_crossing_rate")
    weekly_sorted_qcross = metrics.get("weekly_sorted_quantile_crossing_rate")
    weekly_gap = metrics.get("weekly_median_sort_gap_max")
    weekly_samples = metrics.get("weekly_sample_count")

    print(
        "Quality gate metrics: "
        f"DA={da:.4f} Sharpe={sharpe:.4f} VR={vr:.4f} "
        f"Tail={tail_capture if tail_capture is not None else 'n/a'} "
        f"QCross={quantile_crossing if quantile_crossing is not None else 'n/a'}"
    )
    print(
        "Weekly gate metrics: "
        f"WeeklyDA={weekly_da} WeeklyMR={weekly_mr} "
        f"WeeklyTail={weekly_tail} WeeklyPI80={weekly_pi80} "
        f"WeeklyPI96WidthRatio={weekly_pi96_width_ratio} "
        f"WeeklyQCross={weekly_qcross} WeeklySortedQCross={weekly_sorted_qcross} "
        f"WeeklyN={weekly_samples}"
    )

    passed, reasons = evaluate_quality_gate(
        da,
        sharpe,
        vr,
        tail_capture=tail_capture,
        quantile_crossing_rate=quantile_crossing,
        median_sort_gap_max=median_gap_max,
        weekly_directional_accuracy=weekly_da,
        weekly_magnitude_ratio=weekly_mr,
        weekly_tail_capture_rate=weekly_tail,
        weekly_pi80_coverage=weekly_pi80,
        weekly_pi80_width_ratio=weekly_pi80_width_ratio,
        weekly_pi96_coverage=weekly_pi96,
        weekly_pi96_width_ratio=weekly_pi96_width_ratio,
        weekly_quantile_crossing_rate=weekly_qcross,
        weekly_sorted_quantile_crossing_rate=weekly_sorted_qcross,
        weekly_median_sort_gap_max=weekly_gap,
        weekly_sample_count=weekly_samples,
    )

    if passed:
        print("QUALITY GATE: PASSED")
        return 0

    print(f"QUALITY GATE: FAILED — {reasons}")
    print("Model checkpoint will NOT be promoted. Previous checkpoint retained.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
