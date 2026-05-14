"""
HuggingFace Hub integration for TFT-ASRO model persistence.

Solves the ephemeral storage problem on HF Spaces: after training,
checkpoints are uploaded to a dedicated HF model repo; before inference,
they are downloaded if not present locally.
"""

from __future__ import annotations

import json
import logging
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HF_TOKEN_ENV = "HF_TOKEN"

_ARTIFACTS = [
    "best_tft_asro.ckpt",
    "tft_metadata.json",
    "conformal_calibration.json",
    "pca_finbert.joblib",
    "optuna_results.json",
    "artifact_manifest.json",
]

_REQUIRED_ARTIFACTS = [
    "best_tft_asro.ckpt",
    "tft_metadata.json",
    "artifact_manifest.json",
]


def _get_token() -> Optional[str]:
    return os.environ.get(_HF_TOKEN_ENV)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read JSON artifact %s: %s", path, exc)
        return {}


def build_artifact_health(local_dir: str | Path) -> dict:
    """Build promotion/inference health metadata for the TFT artifact set."""
    local_dir = Path(local_dir)
    metadata_path = local_dir / "tft_metadata.json"
    checkpoint_present = (local_dir / "best_tft_asro.ckpt").exists()
    metadata_present = metadata_path.exists()
    conformal_present = (local_dir / "conformal_calibration.json").exists()

    metadata = _load_json(metadata_path)
    config = metadata.get("config") or {}
    metrics = metadata.get("test_metrics") or {}
    optuna = _load_json(local_dir / "optuna_results.json")
    structural_report = optuna.get("structural_invalidity_report") or {}
    best_preflight = optuna.get("best_trial_preflight") or {}

    quality_gate_passed = False
    gate_error = None
    if metrics:
        try:
            from app.quality_gate import evaluate_quality_gate

            quality_gate_passed, reasons = evaluate_quality_gate(
                da=float(metrics.get("directional_accuracy", 0.5)),
                sharpe=float(metrics.get("sharpe_ratio", 0.0)),
                vr=float(metrics.get("variance_ratio", 1.0)),
                tail_capture=metrics.get("tail_capture_rate"),
                quantile_crossing_rate=metrics.get("quantile_crossing_rate"),
                median_sort_gap_max=metrics.get("median_sort_gap_max"),
                pi80_width=metrics.get("pi80_width"),
                pi96_width=metrics.get("pi96_width"),
                weekly_directional_accuracy=metrics.get("weekly_directional_accuracy"),
                weekly_magnitude_ratio=metrics.get("weekly_magnitude_ratio"),
                weekly_tail_capture_rate=metrics.get("weekly_tail_capture_rate"),
                weekly_pi80_coverage=metrics.get("weekly_pi80_coverage"),
                weekly_pi80_width=metrics.get("weekly_pi80_width"),
                weekly_pi80_width_ratio=metrics.get("weekly_pi80_width_ratio"),
                weekly_pi96_coverage=metrics.get("weekly_pi96_coverage"),
                weekly_pi96_width=metrics.get("weekly_pi96_width"),
                weekly_pi96_width_ratio=metrics.get("weekly_pi96_width_ratio"),
                weekly_quantile_crossing_rate=metrics.get("weekly_quantile_crossing_rate"),
                weekly_sorted_quantile_crossing_rate=metrics.get(
                    "weekly_sorted_quantile_crossing_rate"
                ),
                weekly_median_sort_gap_max=metrics.get("weekly_median_sort_gap_max"),
                weekly_sample_count=metrics.get("weekly_sample_count"),
            )
            if not quality_gate_passed:
                gate_error = "; ".join(reasons)
        except Exception as exc:
            gate_error = str(exc)
            quality_gate_passed = False
    else:
        gate_error = "missing test_metrics"

    safe = bool(quality_gate_passed and checkpoint_present and metadata_present)
    next_required_action = "No action required; artifact is promotable."
    if not safe:
        next_required_action = (
            gate_error
            or "Run deterministic validation and pass the weekly quality gate before upload."
        )

    return {
        "forecast_contract_version": (
            metadata.get("forecast_contract_version")
            or config.get("forecast_contract_version")
        ),
        "monotonic_quantile_transform": bool(
            config.get("monotonic_quantile_transform")
            or metadata.get("monotonic_quantile_transform")
        ),
        "checkpoint_present": checkpoint_present,
        "metadata_present": metadata_present,
        "conformal_present": conformal_present,
        "quality_gate_passed": quality_gate_passed,
        "best_trial_preflight_passed": bool(best_preflight.get("preflight_passed", False)),
        "structural_invalidity_verdict": structural_report.get("verdict", "UNKNOWN"),
        "safe_to_upload_to_hub": safe,
        "safe_for_inference": safe,
        "raw_quantile_crossing_rate": metrics.get("raw_quantile_crossing_rate"),
        "ordered_quantile_crossing_rate": metrics.get("ordered_quantile_crossing_rate"),
        "public_quantile_crossing_rate": metrics.get(
            "public_quantile_crossing_rate",
            metrics.get("quantile_crossing_rate"),
        ),
        "variance_ratio": metrics.get("variance_ratio"),
        "mae_vs_naive_zero": metrics.get("mae_vs_naive_zero"),
        "weekly_mae_vs_naive_zero": metrics.get("weekly_mae_vs_naive_zero"),
        "next_required_action": next_required_action,
    }


def build_artifact_manifest(local_dir: str | Path) -> dict:
    """Build a SHA256 manifest for every present TFT artifact except itself."""
    local_dir = Path(local_dir)
    artifacts = {}
    for name in _ARTIFACTS:
        if name == "artifact_manifest.json":
            continue
        path = local_dir / name
        if not path.exists():
            continue
        artifacts[name] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "required": name in {"best_tft_asro.ckpt", "tft_metadata.json"},
        }

    return {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "artifact_health": build_artifact_health(local_dir),
    }


def write_artifact_manifest(local_dir: str | Path) -> Path:
    """Write the artifact integrity manifest next to TFT artifacts."""
    local_dir = Path(local_dir)
    manifest_path = local_dir / "artifact_manifest.json"
    manifest = build_artifact_manifest(local_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def validate_artifact_manifest(local_dir: str | Path) -> bool:
    """Verify artifact_manifest.json hashes before loading checkpoint/joblib files."""
    local_dir = Path(local_dir)
    manifest_path = local_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        logger.warning("TFT artifact manifest missing in %s", local_dir)
        return False

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = data.get("artifacts") or {}
        for required in ("best_tft_asro.ckpt", "tft_metadata.json"):
            if required not in artifacts:
                logger.warning("TFT artifact manifest missing required entry: %s", required)
                return False

        for name, meta in artifacts.items():
            path = local_dir / name
            if not path.exists():
                logger.warning("TFT artifact listed in manifest is missing: %s", name)
                return False
            expected = str(meta.get("sha256", "")).lower()
            actual = _sha256_file(path).lower()
            if not expected or actual != expected:
                logger.warning("TFT artifact hash mismatch for %s", name)
                return False
            expected_size = meta.get("size_bytes")
            if expected_size is not None and int(expected_size) != path.stat().st_size:
                logger.warning("TFT artifact size mismatch for %s", name)
                return False
        return True
    except Exception as exc:
        logger.warning("TFT artifact manifest validation failed: %s", exc)
        return False


def _metadata_contract_valid(metadata_path: Path) -> bool:
    """Return True when metadata proves the current weekly TFT contract."""
    if not metadata_path.exists():
        return False

    try:
        from deep_learning.contract import (
            FORECAST_CONTRACT_VERSION,
            RETURN_SPACE,
            TARGET_RETURN_TYPE,
        )

        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = data.get("config") or {}
        version = data.get("forecast_contract_version") or config.get(
            "forecast_contract_version"
        )
        target_return_type = data.get("target_return_type") or config.get(
            "target_return_type"
        )
        primary_horizon = data.get("primary_horizon_days") or config.get(
            "primary_horizon_days"
        )
        return_space = data.get("return_space") or config.get("return_space")

        return (
            version == FORECAST_CONTRACT_VERSION
            and target_return_type == TARGET_RETURN_TYPE
            and int(primary_horizon or 0) == 5
            and return_space == RETURN_SPACE
        )
    except Exception as exc:
        logger.warning(
            "TFT metadata contract validation failed for %s: %s",
            metadata_path,
            exc,
        )
        return False


def validate_tft_artifact_set(local_dir: str | Path) -> bool:
    """Validate the minimum healthy TFT artifact set for weekly inference."""
    local_dir = Path(local_dir)
    missing = [
        name for name in _REQUIRED_ARTIFACTS if not (local_dir / name).exists()
    ]
    if missing:
        logger.warning(
            "TFT artifact set incomplete in %s: missing %s",
            local_dir,
            missing,
        )
        return False

    if not _metadata_contract_valid(local_dir / "tft_metadata.json"):
        logger.warning("TFT artifact set has incompatible metadata in %s", local_dir)
        return False

    if not validate_artifact_manifest(local_dir):
        return False

    return True


def _manifest_safe_to_upload(local_dir: str | Path) -> bool:
    manifest = _load_json(Path(local_dir) / "artifact_manifest.json")
    health = manifest.get("artifact_health") or {}
    return bool(health.get("safe_to_upload_to_hub"))


def upload_tft_artifacts(
    local_dir: str | Path,
    repo_id: str,
    commit_message: str = "Update TFT-ASRO checkpoint",
) -> bool:
    """
    Upload all TFT artifacts from *local_dir* to a HuggingFace model repo.

    Returns True on success, False if upload fails, token is missing, or the
    minimum weekly contract artifact set is incomplete.
    """
    token = _get_token()
    if not token:
        logger.warning("HF_TOKEN not set; skipping model upload to Hub")
        return False

    local_dir = Path(local_dir)
    if not _metadata_contract_valid(local_dir / "tft_metadata.json"):
        logger.warning(
            "TFT artifact set in %s is not contract-complete; upload skipped",
            local_dir,
        )
        return False

    write_artifact_manifest(local_dir)
    if not validate_tft_artifact_set(local_dir):
        logger.warning("TFT artifact manifest validation failed before upload")
        return False
    if not _manifest_safe_to_upload(local_dir):
        logger.warning("TFT artifact health is not safe for Hub upload; upload skipped")
        return False

    files_to_upload = [
        local_dir / name for name in _ARTIFACTS if (local_dir / name).exists()
    ]

    if not files_to_upload:
        logger.warning("No TFT artifacts found in %s", local_dir)
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)

        for fpath in files_to_upload:
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fpath.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            logger.info("Uploaded %s to %s/%s", fpath.name, repo_id, fpath.name)

        return True

    except Exception as exc:
        logger.error("HF Hub upload failed: %s", exc)
        return False


def download_tft_artifacts(
    local_dir: str | Path,
    repo_id: str,
) -> bool:
    """
    Download TFT artifacts from HuggingFace Hub to *local_dir*.

    Skips files that already exist locally unless required metadata is
    incompatible. Returns True only when the minimum weekly contract artifact
    set is present and valid.
    """
    token = _get_token()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if validate_tft_artifact_set(local_dir):
        logger.debug("TFT artifact set already present locally: %s", local_dir)
        return True

    force_download = set()
    metadata_path = local_dir / "tft_metadata.json"
    if metadata_path.exists() and not _metadata_contract_valid(metadata_path):
        force_download.add("tft_metadata.json")
    manifest_path = local_dir / "artifact_manifest.json"
    if manifest_path.exists() and not validate_artifact_manifest(local_dir):
        force_download.update(_ARTIFACTS)

    try:
        from huggingface_hub import hf_hub_download

        for name in _ARTIFACTS:
            dest = local_dir / name
            if dest.exists() and name not in force_download:
                continue
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=name,
                    local_dir=str(local_dir),
                    token=token,
                    force_download=name in force_download,
                )
                logger.info("Downloaded %s/%s to %s", repo_id, name, dest)
            except Exception:
                logger.debug(
                    "Artifact %s not found in %s (may not exist yet)",
                    name,
                    repo_id,
                )

        return validate_tft_artifact_set(local_dir)

    except ImportError:
        logger.warning("huggingface_hub not installed; cannot download model")
        return False
    except Exception as exc:
        logger.warning("HF Hub download failed: %s", exc)
        return False
