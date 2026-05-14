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
        "config": {
            "forecast_contract_version": FORECAST_CONTRACT_VERSION,
            "target_return_type": TARGET_RETURN_TYPE,
            "primary_horizon_days": 5,
            "return_space": RETURN_SPACE,
            "monotonic_quantile_transform": True,
        },
        "test_metrics": {
            "directional_accuracy": 0.55,
            "sharpe_ratio": 0.1,
            "variance_ratio": 1.0,
            "tail_capture_rate": 0.5,
            "quantile_crossing_rate": 0.0,
            "median_sort_gap_max": 0.0,
            "weekly_directional_accuracy": 0.55,
            "weekly_magnitude_ratio": 1.0,
            "weekly_tail_capture_rate": 0.5,
            "weekly_pi80_coverage": 0.80,
            "weekly_pi80_width": 0.02,
            "weekly_pi80_width_ratio": 1.0,
            "weekly_pi96_coverage": 0.96,
            "weekly_pi96_width": 0.04,
            "weekly_pi96_width_ratio": 1.0,
            "weekly_quantile_crossing_rate": 0.0,
            "weekly_sorted_quantile_crossing_rate": 0.0,
            "weekly_median_sort_gap_max": 0.0,
            "weekly_sample_count": 120,
            "mae_vs_naive_zero": 0.9,
            "weekly_mae_vs_naive_zero": 0.9,
        },
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
        elif filename == "artifact_manifest.json":
            hub.write_artifact_manifest(path.parent)
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
    assert "artifact_manifest.json" in calls
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
    assert "artifact_manifest.json" in uploaded
    manifest = json.loads((local_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_health"]["quality_gate_passed"] is True
    assert manifest["artifact_health"]["safe_to_upload_to_hub"] is True


def test_validate_artifact_set_rejects_hash_mismatch(tmp_path):
    local_dir = tmp_path / "tft"
    local_dir.mkdir()
    (local_dir / "best_tft_asro.ckpt").write_bytes(b"checkpoint")
    (local_dir / "tft_metadata.json").write_text(json.dumps(_valid_metadata()), encoding="utf-8")
    hub.write_artifact_manifest(local_dir)

    (local_dir / "best_tft_asro.ckpt").write_bytes(b"tampered")

    assert hub.validate_tft_artifact_set(local_dir) is False


def test_upload_refuses_manifest_marked_unsafe_by_quality_gate(tmp_path, monkeypatch):
    local_dir = tmp_path / "tft"
    local_dir.mkdir()
    metadata = _valid_metadata()
    metadata["test_metrics"]["weekly_magnitude_ratio"] = 5.0
    for name in hub._ARTIFACTS:
        path = local_dir / name
        if name == "tft_metadata.json":
            path.write_text(json.dumps(metadata), encoding="utf-8")
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

    assert hub.upload_tft_artifacts(local_dir, "org/model") is False
    assert uploaded == []
    manifest = json.loads((local_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_health"]["quality_gate_passed"] is False
    assert manifest["artifact_health"]["safe_to_upload_to_hub"] is False
