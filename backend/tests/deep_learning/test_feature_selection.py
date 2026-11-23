"""Tests for MRMR feature selection and VSN pruning."""

import numpy as np
import pandas as pd
import pytest

from deep_learning.data.feature_selection import (
    build_fold_scoped_features,
    mrmr_select,
    select_features,
    vsn_prune,
)


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


def test_mrmr_zero_or_negative_top_k(sample_df):
    assert mrmr_select(sample_df, target_col="target", top_k=0) == []
    assert mrmr_select(sample_df, target_col="target", top_k=-1) == []


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


def test_select_features_forces_sentiment_regime_and_drops_forbidden(sample_df):
    df = sample_df.copy()
    n = len(df)
    df["target_5d_log_return"] = sample_df["target"] * 5
    df["target_1d_log_return"] = sample_df["target"]
    df["realized_vol_20d"] = 0.01
    df["material_move_5d"] = 0.0
    df["HG_F_vol_10"] = 0.02
    df["cal_sin_day"] = np.sin(np.linspace(0, 6.0, n))
    # Provide healthy non-degenerate variance and coverage (>= 8%)
    df["sentiment_index"] = np.sin(np.linspace(0, 6.0, n)) * 0.4
    df["regime_usd_pressure"] = (np.arange(n) % 4 == 0).astype(float)
    filtered, unknown, _ = select_features(
        df,
        target_col="target_5d_log_return",
        mrmr_top_k=2,
        known_features=[],
        forced_unknown_features=["sentiment_index", "regime_usd_pressure"],
        forbidden_features=["HG_F_vol_10", "cal_sin_day"],
        preserve_features=["target", "target_1d_log_return", "realized_vol_20d", "material_move_5d"],
    )
    assert "sentiment_index" in unknown
    assert "regime_usd_pressure" in unknown
    assert "target_5d_log_return" in filtered.columns
    assert "target_1d_log_return" in filtered.columns
    assert "target_1d_log_return" not in unknown
    assert "material_move_5d" not in unknown
    assert "HG_F_vol_10" not in filtered.columns
    assert "HG_F_vol_10" not in unknown
    assert "cal_sin_day" not in filtered.columns
    assert "cal_sin_day" not in unknown


def test_select_features_drops_zero_variance_and_sparse_forced_columns(sample_df):
    df = sample_df.copy()
    df["sentiment_index"] = 0.0  # zero variance
    sparse = np.zeros(len(df), dtype=float)
    sparse[:3] = 1.0  # < 8% non-mode coverage on n=200
    df["regime_usd_pressure"] = sparse
    _filtered, unknown, _ = select_features(
        df,
        target_col="target",
        mrmr_top_k=3,
        known_features=[],
        forced_unknown_features=["sentiment_index", "regime_usd_pressure"],
    )
    assert "sentiment_index" not in unknown
    assert "regime_usd_pressure" not in unknown


def test_select_features_fits_mrmr_on_explicit_training_frame(monkeypatch, sample_df):
    observed = {}

    def fake_mrmr(frame, **_kwargs):
        observed["index"] = frame.index.tolist()
        return ["signal_1"]

    monkeypatch.setattr("deep_learning.data.feature_selection.mrmr_select", fake_mrmr)
    training_only = sample_df.iloc[:80].copy()

    filtered, unknown, _ = select_features(
        sample_df,
        target_col="target",
        mrmr_top_k=1,
        selection_df=training_only,
    )

    assert observed["index"] == training_only.index.tolist()
    assert unknown == ["signal_1"]
    assert len(filtered) == len(sample_df)


def test_mrmr_groups_target_hgf_and_curve_proxies():
    rng = np.random.default_rng(7)
    n = 160
    y = rng.normal(0.0, 0.02, size=n)
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n),
            "group_id": "copper",
            "target_5d_log_return": y,
            "HG=F_f1": y * 0.7 + rng.normal(0.0, 0.01, size=n),
            "HG=F_f2": y * 0.65 + rng.normal(0.0, 0.012, size=n),
            "HG=F_f3": y * 0.60 + rng.normal(0.0, 0.013, size=n),
            "HG=F_f4": y * 0.55 + rng.normal(0.0, 0.014, size=n),
            "backwardation_flag": (y > 0.0).astype(float),
            "contango_flag": (y <= 0.0).astype(float),
            "futures_curve_slope": y * 0.5 + rng.normal(0.0, 0.015, size=n),
            "BHP_ret10": y * 0.4 + rng.normal(0.0, 0.015, size=n),
            "FCX_ret10": y * 0.35 + rng.normal(0.0, 0.016, size=n),
        }
    )
    selected = mrmr_select(
        df,
        target_col="target_5d_log_return",
        top_k=5,
        max_per_group=2,
        group_prefixes=["HG=F", "BHP", "FCX"],
    )
    hg_selected = [c for c in selected if c.startswith("HG=F_")]
    curve_selected = [
        c for c in selected if c.startswith("futures_") or c in {"backwardation_flag", "contango_flag"}
    ]
    assert len(hg_selected) <= 2
    assert len(curve_selected) <= 2


def test_build_fold_scoped_features_prevents_validation_fold_leakage():
    rng = np.random.default_rng(99)
    n = 180
    y = rng.normal(0.0, 0.02, size=n)
    # leak_feat is pure noise on Fold 1 train ([:100]) but strongly predictive on Fold 1 val ([105:])
    leak_feat = np.concatenate([rng.normal(0.0, 0.02, 100), y[100:] * 0.9 + rng.normal(0.0, 0.002, 80)])
    # Boundary peek feature only matches in last 5 bars of train (95..99)
    real_feat = y * 0.45 + rng.normal(0.0, 0.012, size=n)
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n),
            "group_id": "copper",
            "target": rng.normal(0.0, 0.01, size=n),
            "target_5d_log_return": y,
            "day_of_week": (np.arange(n) % 5) / 4.0,
            "leak_feat": leak_feat,
            "real_feat": real_feat,
            "noise_a": rng.normal(0.0, 1.0, size=n),
            "noise_b": rng.normal(0.0, 1.0, size=n),
        }
    )
    filtered, unknown, known = build_fold_scoped_features(
        df,
        train_end_pos=100,
        target_col="target_5d_log_return",
        mrmr_top_k=1,
        horizon=5,
        known_features=["day_of_week"],
    )
    assert unknown == ["real_feat"]
    assert known == ["day_of_week"]
    assert "leak_feat" not in filtered.columns


def test_select_features_defers_severely_non_stationary_forced_prior():
    rng = np.random.default_rng(11)
    n = 220
    y = rng.normal(0.0, 0.02, size=n)
    drifting = np.concatenate([rng.normal(-2.0, 0.1, 110), rng.normal(2.0, 0.1, 110)])
    stationary = rng.normal(0.0, 0.02, size=n)
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n),
            "group_id": "copper",
            "target": y,
            "drifting_prior": drifting,
            "stationary_prior": stationary,
            "noise_a": rng.normal(0.0, 1.0, size=n),
            "noise_b": rng.normal(0.0, 1.0, size=n),
        }
    )
    _filtered, unknown, _ = select_features(
        df,
        target_col="target",
        mrmr_top_k=2,
        known_features=[],
        forced_unknown_features=["drifting_prior", "stationary_prior"],
    )
    assert "stationary_prior" in unknown
    assert "drifting_prior" not in unknown
