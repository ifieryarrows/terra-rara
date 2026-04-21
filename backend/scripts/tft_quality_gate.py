"""
CI quality gate for TFT-ASRO training.

Reads tft_metadata.json written by trainer.py and exits non-zero when
metrics fall below deployment thresholds.

Thin wrapper that delegates threshold logic to `app.quality_gate` so that
GitHub Actions CI and the FastAPI runtime always agree on the rules.
"""

from __future__ import annotations

import json
import pathlib
import sys

from app.quality_gate import evaluate_quality_gate

META_PATH = pathlib.Path("/tmp/models/tft/tft_metadata.json")


def main() -> int:
    if not META_PATH.exists():
        print("No metadata file found — skipping quality gate")
        return 0

    data = json.loads(META_PATH.read_text(encoding="utf-8"))
    metrics = data.get("test_metrics", {})
    da = metrics.get("directional_accuracy", 0.5)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    vr = metrics.get("variance_ratio", 1.0)

    print(f"Quality gate metrics: DA={da:.4f} Sharpe={sharpe:.4f} VR={vr:.4f}")

    passed, reasons = evaluate_quality_gate(da, sharpe, vr)

    if passed:
        print("QUALITY GATE: PASSED")
        return 0

    print(f"QUALITY GATE: FAILED — {reasons}")
    print("Model checkpoint will NOT be promoted. Previous checkpoint retained.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
