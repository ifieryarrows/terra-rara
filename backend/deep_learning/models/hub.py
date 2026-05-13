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
]

_REQUIRED_ARTIFACTS = [
    "best_tft_asro.ckpt",
    "tft_metadata.json",
]


def _get_token() -> Optional[str]:
    return os.environ.get(_HF_TOKEN_ENV)


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

    return True


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
    if not validate_tft_artifact_set(local_dir):
        logger.warning(
            "TFT artifact set in %s is not contract-complete; upload skipped",
            local_dir,
        )
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
