"""
HuggingFace Hub integration for TFT-ASRO model persistence.

Solves the ephemeral storage problem on HF Spaces: after training,
checkpoints are uploaded to a dedicated HF model repo; before inference,
they are downloaded if not present locally.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HF_TOKEN_ENV = "HF_TOKEN"

_ARTIFACTS = [
    "best_tft_asro.ckpt",
    "pca_finbert.joblib",
]


def _get_token() -> Optional[str]:
    return os.environ.get(_HF_TOKEN_ENV)


def upload_tft_artifacts(
    local_dir: str | Path,
    repo_id: str,
    commit_message: str = "Update TFT-ASRO checkpoint",
) -> bool:
    """
    Upload all TFT artifacts from *local_dir* to a HuggingFace model repo.

    Returns True on success, False if upload fails or token is missing.
    """
    token = _get_token()
    if not token:
        logger.warning("HF_TOKEN not set – skipping model upload to Hub")
        return False

    local_dir = Path(local_dir)
    files_to_upload = [
        local_dir / name
        for name in _ARTIFACTS
        if (local_dir / name).exists()
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
            logger.info("Uploaded %s → %s/%s", fpath.name, repo_id, fpath.name)

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

    Skips files that already exist locally.
    Returns True if at least the checkpoint was retrieved.
    """
    token = _get_token()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = local_dir / "best_tft_asro.ckpt"
    if ckpt_path.exists():
        logger.debug("TFT checkpoint already present locally: %s", ckpt_path)
        return True

    try:
        from huggingface_hub import hf_hub_download

        for name in _ARTIFACTS:
            dest = local_dir / name
            if dest.exists():
                continue
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=name,
                    local_dir=str(local_dir),
                    token=token,
                )
                logger.info("Downloaded %s/%s → %s", repo_id, name, dest)
            except Exception:
                logger.debug("Artifact %s not found in %s (may not exist yet)", name, repo_id)

        return ckpt_path.exists()

    except ImportError:
        logger.warning("huggingface_hub not installed – cannot download model")
        return False
    except Exception as exc:
        logger.warning("HF Hub download failed: %s", exc)
        return False
