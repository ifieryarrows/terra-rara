import pytest

from deep_learning.data.dataset import _identity_target_normalizer, _resolve_num_workers


def test_zero_workers_remains_zero_on_posix_for_reproducibility(monkeypatch):
    monkeypatch.setattr("deep_learning.data.dataset.os.name", "posix", raising=False)
    assert _resolve_num_workers(0) == 0


def test_tft_dataset_uses_identity_target_normalizer():
    pytest.importorskip("pytorch_forecasting")
    normalizer = _identity_target_normalizer()
    assert getattr(normalizer, "method", None) == "identity"
    assert getattr(normalizer, "center", None) is False
    assert getattr(normalizer, "transformation", None) is None
