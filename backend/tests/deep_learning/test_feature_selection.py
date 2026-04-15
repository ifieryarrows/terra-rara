"""Tests for MRMR feature selection and VSN pruning."""

import numpy as np
import pandas as pd
import pytest

from deep_learning.data.feature_selection import mrmr_select, vsn_prune, select_features


@pytest.fixture
def sample_df():
    """DataFrame with known signal and noise features."""
    np.random.seed(42)
    n = 200

    target = np.random.randn(n) * 0.02
    signal_1 = target * 0.5 + np.random.randn(n) * 0.005
    signal_2 = target * 0.3 + np.random.randn(n) * 0.01
    noise = np.random.randn(n) * 0.1

    data = {
        "signal_1": signal_1,
        "signal_2": signal_2,
        "redundant_1": signal_1 * 1.01 + 0.001,
        "noise_1": noise,
        "noise_2": np.random.randn(n),
        "noise_3": np.random.randn(n),
        "target": target,
        "time_idx": np.arange(n),
        "group_id": "copper",
    }

    for i in range(20):
        data[f"extra_noise_{i}"] = np.random.randn(n)

    return pd.DataFrame(data)


def test_mrmr_select_reduces_features(sample_df):
    selected = mrmr_select(sample_df, target_col="target", top_k=5)
    assert len(selected) == 5
    assert "target" not in selected
    assert "time_idx" not in selected
    assert "group_id" not in selected


def test_mrmr_skip_when_fewer_than_top_k(sample_df):
    small_df = sample_df[["signal_1", "signal_2", "target", "time_idx", "group_id"]]
    selected = mrmr_select(small_df, target_col="target", top_k=10)
    assert len(selected) == 2


def test_vsn_prune_respects_min_features():
    importance = {f"feat_{i}": 1.0 / (i + 1) for i in range(100)}
    features = list(importance.keys())
    pruned = vsn_prune(importance, features, min_features=40)
    assert len(pruned) >= 40


def test_vsn_prune_empty_importance():
    features = ["a", "b", "c"]
    pruned = vsn_prune({}, features)
    assert pruned == features


def test_select_features_preserves_known(sample_df):
    known = ["time_idx"]
    filtered, unknown, known_out = select_features(
        sample_df, target_col="target", mrmr_top_k=5, known_features=known,
    )
    assert "time_idx" in filtered.columns
    assert "target" in filtered.columns
    assert "group_id" in filtered.columns
    assert len(unknown) <= 5
