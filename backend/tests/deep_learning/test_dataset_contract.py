import pytest

from deep_learning.data.dataset import _identity_target_normalizer


def test_tft_dataset_uses_identity_target_normalizer():
    pytest.importorskip("pytorch_forecasting")
    normalizer = _identity_target_normalizer()
    assert getattr(normalizer, "method", None) == "identity"
    assert getattr(normalizer, "center", None) is False
    assert getattr(normalizer, "transformation", None) is None
