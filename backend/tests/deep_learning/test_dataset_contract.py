import numpy as np
import pandas as pd
import pytest

from deep_learning.config import get_tft_config
from deep_learning.data.dataset import (
    _identity_target_normalizer,
    _resolve_num_workers,
    build_cv_folds,
    build_datasets,
    compute_dataset_split_bounds,
    create_train_dataloader,
    make_dataloader_generator,
)


def test_zero_workers_remains_zero_on_posix_for_reproducibility(monkeypatch):
    monkeypatch.setattr("deep_learning.data.dataset.os.name", "posix", raising=False)
    assert _resolve_num_workers(0) == 0


def test_tft_dataset_uses_identity_target_normalizer():
    pytest.importorskip("pytorch_forecasting")
    normalizer = _identity_target_normalizer()
    assert getattr(normalizer, "method", None) == "identity"
    assert getattr(normalizer, "center", None) is False
    assert getattr(normalizer, "transformation", None) is None


def test_build_datasets_enforces_chronological_purge_gaps():
    pytest.importorskip("pytorch_forecasting")
    n = 160
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n, dtype=int),
            "group_id": "copper",
            "HG=F_ret1": np.sin(np.linspace(0, 10, n)).astype(np.float32) * 0.01,
            "day_of_week": (np.arange(n) % 5) / 4.0,
            "target": np.cos(np.linspace(0, 10, n)).astype(np.float32) * 0.01,
        }
    )
    cfg = get_tft_config()
    bounds = compute_dataset_split_bounds(df, cfg, purge_gap=5)
    tr_ds, va_ds, te_ds = build_datasets(df, ["HG=F_ret1"], ["day_of_week"], ["target"], cfg, purge_gap=5)

    tr_dl = tr_ds.to_dataloader(train=False, batch_size=len(tr_ds), num_workers=0)
    va_dl = va_ds.to_dataloader(train=False, batch_size=len(va_ds), num_workers=0)
    te_dl = te_ds.to_dataloader(train=False, batch_size=len(te_ds), num_workers=0)

    tr_x, _ = next(iter(tr_dl))
    va_x, _ = next(iter(va_dl))
    te_x, _ = next(iter(te_dl))

    tr_max_dec = int(tr_x["decoder_time_idx"].max())
    va_min_dec = int(va_x["decoder_time_idx"].min())
    va_max_dec = int(va_x["decoder_time_idx"].max())
    te_min_dec = int(te_x["decoder_time_idx"].min())

    assert tr_max_dec == bounds["train_cutoff"]
    assert va_min_dec - tr_max_dec == 6  # 5 unpredicted purge-gap bars in between
    assert va_max_dec <= bounds["val_cutoff"]
    assert te_min_dec - va_max_dec >= 6  # 5 unpredicted purge-gap bars in between


def test_per_seed_train_dataloader_is_isolated_from_other_seeds():
    pytest.importorskip("pytorch_forecasting")
    n = 140
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n, dtype=int),
            "group_id": "copper",
            "HG=F_ret1": np.linspace(-0.02, 0.02, n, dtype=np.float32),
            "day_of_week": (np.arange(n) % 5) / 4.0,
            "target": np.linspace(-0.01, 0.01, n, dtype=np.float32),
        }
    )
    cfg = get_tft_config()
    tr_ds, _, _ = build_datasets(df, ["HG=F_ret1"], ["day_of_week"], ["target"], cfg)

    dl_137_a = create_train_dataloader(tr_ds, cfg, seed=137)
    first_137_a = [b[0]["decoder_time_idx"].clone() for b in dl_137_a]

    # Exhaust multiple epochs on seed 42 in between
    dl_42 = create_train_dataloader(tr_ds, cfg, seed=42)
    for _ in range(3):
        list(dl_42)

    dl_137_b = create_train_dataloader(tr_ds, cfg, seed=137)
    first_137_b = [b[0]["decoder_time_idx"].clone() for b in dl_137_b]

    assert len(first_137_a) == len(first_137_b)
    for a, b in zip(first_137_a, first_137_b):
        assert (a == b).all()
    assert make_dataloader_generator(137).initial_seed() == 137


def test_build_cv_folds_runs_per_fold_mrmr_without_leakage():
    pytest.importorskip("pytorch_forecasting")
    rng = np.random.default_rng(101)
    n = 200
    y = rng.normal(0.0, 0.02, size=n)
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n, dtype=int),
            "group_id": "copper",
            "target": rng.normal(0.0, 0.01, size=n),
            "target_5d_log_return": y,
            "day_of_week": (np.arange(n) % 5) / 4.0,
            "f1": y * 0.5 + rng.normal(0.0, 0.01, size=n),
            "f2": -y * 0.4 + rng.normal(0.0, 0.01, size=n),
            "f3": rng.normal(0.0, 1.0, size=n),
            "f4": rng.normal(0.0, 1.0, size=n),
        }
    )
    cfg = get_tft_config()
    folds = build_cv_folds(
        df,
        ["f1", "f2", "f3", "f4"],
        ["day_of_week"],
        ["target"],
        cfg,
        n_folds=2,
        purge_gap=5,
        mrmr_top_k=2,
    )
    assert len(folds) == 2
    for tr_ds, va_ds in folds:
        assert len(tr_ds) > 0
        assert len(va_ds) > 0
        assert len(tr_ds.time_varying_unknown_reals) == 2
