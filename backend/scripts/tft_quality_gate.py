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

    print(
        "Quality gate metrics: "
        f"DA={da:.4f} Sharpe={sharpe:.4f} VR={vr:.4f} "
        f"Tail={tail_capture if tail_capture is not None else 'n/a'} "
        f"QCross={quantile_crossing if quantile_crossing is not None else 'n/a'}"
    )

    passed, reasons = evaluate_quality_gate(
        da,
        sharpe,
        vr,
        tail_capture=tail_capture,
        quantile_crossing_rate=quantile_crossing,
        median_sort_gap_max=median_gap_max,
    )

    if passed:
        print("QUALITY GATE: PASSED")
        return 0

    print(f"QUALITY GATE: FAILED — {reasons}")
    print("Model checkpoint will NOT be promoted. Previous checkpoint retained.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
