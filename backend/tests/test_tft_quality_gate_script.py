"""Regression tests for the TFT quality-gate CLI wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_tft_quality_gate_script_imports_app_from_scripts_dir(tmp_path: Path):
    metadata_path = tmp_path / "tft_metadata.json"
    payload = {
        "test_metrics": {
            "directional_accuracy": 0.52,
            "sharpe_ratio": 0.4,
            "variance_ratio": 0.9,
            "tail_capture_rate": 0.45,
            "quantile_crossing_rate": 0.0,
            "median_sort_gap_max": 0.0,
        }
    }
    metadata_path.write_text("\ufeff" + json.dumps(payload), encoding="utf-8")

    backend_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["TFT_METADATA_PATH"] = str(metadata_path)

    result = subprocess.run(
        [sys.executable, "scripts/tft_quality_gate.py"],
        cwd=backend_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "QUALITY GATE: PASSED" in result.stdout
