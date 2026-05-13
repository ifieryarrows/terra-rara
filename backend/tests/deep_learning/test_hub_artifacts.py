import json
import sys
import types

from deep_learning.contract import (
    FORECAST_CONTRACT_VERSION,
    RETURN_SPACE,
    TARGET_RETURN_TYPE,
)
from deep_learning.models import hub


def _valid_metadata() -> dict:
    return {
        "forecast_contract_version": FORECAST_CONTRACT_VERSION,
        "target_return_type": TARGET_RETURN_TYPE,
        "primary_horizon_days": 5,
        "return_space": RETURN_SPACE,
    }


def test_download_refreshes_missing_companion_artifacts_when_checkpoint_exists(tmp_path, monkeypatch):
    local_dir = tmp_path / "tft"
    local_dir.mkdir()
    (local_dir / "best_tft_asro.ckpt").write_bytes(b"checkpoint")
    calls = []

    def fake_hf_hub_download(*, repo_id, filename, local_dir, token, force_download=False):
        del repo_id, token, force_download
        calls.append(filename)
        path = tmp_path / "tft" / filename
        if filename == "tft_metadata.json":
            path.write_text(json.dumps(_valid_metadata()), encoding="utf-8")
        else:
            path.write_text("artifact", encoding="utf-8")
        return str(path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    assert hub.download_tft_artifacts(local_dir, "org/model") is True
    assert "tft_metadata.json" in calls
    assert "conformal_calibration.json" in calls
    assert hub.validate_tft_artifact_set(local_dir) is True


def test_upload_includes_metadata_and_conformal_artifacts(tmp_path, monkeypatch):
    local_dir = tmp_path / "tft"
    local_dir.mkdir()
    for name in hub._ARTIFACTS:
        path = local_dir / name
        if name == "tft_metadata.json":
            path.write_text(json.dumps(_valid_metadata()), encoding="utf-8")
        else:
            path.write_text("artifact", encoding="utf-8")

    uploaded = []

    class FakeHfApi:
        def __init__(self, token):
            self.token = token

        def create_repo(self, *args, **kwargs):
            return None

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
            del path_or_fileobj, repo_id, repo_type, commit_message
            uploaded.append(path_in_repo)

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=FakeHfApi),
    )

    assert hub.upload_tft_artifacts(local_dir, "org/model") is True
    assert "tft_metadata.json" in uploaded
    assert "conformal_calibration.json" in uploaded
