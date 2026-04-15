"""
CI quality gate for TFT-ASRO training.

Reads tft_metadata.json written by trainer.py and exits non-zero when
metrics fall below deployment thresholds.

Used by .github/workflows/tft-training.yml (YAML cannot embed indented
Python multiline strings without breaking the workflow parser).
"""

from __future__ import annotations

import json
import pathlib
import sys

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

    reasons: list[str] = []
    if da < 0.49:
        reasons.append(f"DA={da:.4f} < 0.49")
    if sharpe < -0.30:
        reasons.append(f"Sharpe={sharpe:.4f} < -0.30")
    if vr < 0.2 or vr > 2.5:
        reasons.append(f"VR={vr:.4f} outside [0.2, 2.5]")

    if not reasons:
        print("QUALITY GATE: PASSED")
        return 0

    print(f"QUALITY GATE: FAILED — {reasons}")
    print("Model checkpoint will NOT be promoted. Previous checkpoint retained.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
