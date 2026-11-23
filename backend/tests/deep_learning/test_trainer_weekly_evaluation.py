import numpy as np
import pytest
from dataclasses import replace

from deep_learning.config import get_tft_config
from deep_learning.training import metrics as metrics_module
from deep_learning.training.reproducibility import configure_tft_reproducibility
from deep_learning.training.trainer import (
    _compute_test_metrics_from_quantiles,
    _log_weekly_alignment_sample,
    _predict_quantiles_to_np,
    _require_promotable_metrics,
    _runtime_environment_metadata,
)


def _prediction_fixture(n: int = 12):
    actual = np.array(
        [[0.01, 0.005, -0.002, 0.003, 0.004] if i % 2 == 0 else [-0.01, -0.005, 0.002, -0.003, -0.004] for i in range(n)],
        dtype=float,
    )
    pred = np.zeros((n, 5, 7), dtype=float)
    offsets = np.array([-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02])
    for q_idx, offset in enumerate(offsets):
        pred[:, :, q_idx] = actual + offset
    return actual, pred


class _DummyModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.mode = None

    def predict(self, _dataloader, mode=None):
        self.mode = mode
        return self.prediction


def test_quantile_prediction_helper_requires_quantile_mode():
    cfg = get_tft_config()
    _actual, pred = _prediction_fixture()
    model = _DummyModel(pred)

    out = _predict_quantiles_to_np(model, dataloader=object(), cfg=cfg)

    assert model.mode == "quantiles"
    assert out.shape == pred.shape


def test_compute_test_metrics_from_quantiles_emits_t1_and_weekly_metrics():
    cfg = get_tft_config()
    actual, pred = _prediction_fixture()

    metrics = _compute_test_metrics_from_quantiles(actual, pred, cfg)

    assert metrics["directional_accuracy"] == 1.0
    assert metrics["quantile_crossing_rate"] == 0.0
    assert metrics["weekly_directional_accuracy"] == 1.0
    assert metrics["weekly_quantile_crossing_rate"] == 0.0
    assert metrics["weekly_sample_count"] == len(actual)


def test_compute_test_metrics_from_quantiles_uses_shared_evaluator(monkeypatch, caplog):
    caplog.set_level("INFO")
    cfg = get_tft_config()
    cfg = replace(cfg, weekly_loss=replace(cfg.weekly_loss, weekly_median_cap=0.08))
    actual, pred = _prediction_fixture()
    calls = []
    real_evaluator = metrics_module.evaluate_quantile_predictions

    def spy_evaluator(y_actual_path, pred_np, *, quantiles, horizon, weekly_median_cap=None):
        calls.append(
            (
                y_actual_path.shape,
                pred_np.shape,
                tuple(quantiles),
                horizon,
                weekly_median_cap,
            )
        )
        return real_evaluator(
            y_actual_path,
            pred_np,
            quantiles=quantiles,
            horizon=horizon,
            weekly_median_cap=weekly_median_cap,
        )

    monkeypatch.setattr(metrics_module, "evaluate_quantile_predictions", spy_evaluator)

    metrics = _compute_test_metrics_from_quantiles(actual, pred, cfg)

    assert calls == [
        (
            actual.shape,
            pred.shape,
            tuple(cfg.model.quantiles),
            cfg.forecast.primary_horizon_days,
            0.08,
        )
    ]
    assert "weekly_magnitude_ratio" in metrics
    assert "weekly_mae_vs_naive_zero" in metrics
    assert "WEEKLY ALIGNMENT SAMPLE:" in caplog.text


def test_log_weekly_alignment_sample_emits_first_rows(caplog):
    caplog.set_level("INFO")
    cfg = get_tft_config()
    actual, pred = _prediction_fixture(n=3)

    _log_weekly_alignment_sample(actual, pred, cfg, max_rows=2)

    assert "WEEKLY ALIGNMENT SAMPLE:" in caplog.text
    assert "sample=0" in caplog.text
    assert "actual_weekly=" in caplog.text
    assert "pred_weekly=" in caplog.text
    assert "actual_sign=" in caplog.text
    assert "pred_sign=" in caplog.text
    assert "sample=2" not in caplog.text


@pytest.mark.parametrize(
    ("bad_prediction", "message"),
    [
        (np.zeros((8, 7)), "Expected quantile prediction tensor"),
        (np.zeros((8, 4, 7)), "Prediction horizon too short"),
        (np.zeros((8, 5, 6)), "Quantile dim mismatch"),
    ],
)
def test_quantile_prediction_shape_guard_rejects_invalid_outputs(bad_prediction, message):
    cfg = get_tft_config()
    model = _DummyModel(bad_prediction)

    with pytest.raises(RuntimeError, match=message):
        _predict_quantiles_to_np(model, dataloader=object(), cfg=cfg)


def test_required_promotable_metrics_guard_blocks_incomplete_metadata():
    with pytest.raises(RuntimeError, match="Required TFT promotion metrics missing"):
        _require_promotable_metrics({"directional_accuracy": 0.55})


def test_runtime_environment_metadata_records_reproducibility_state():
    configure_tft_reproducibility()
    metadata = _runtime_environment_metadata()

    assert "packages" in metadata
    assert set(metadata["process_controls"]) == {
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    }
    assert metadata["torch_runtime"] is not None
    assert metadata["torch_runtime"]["num_threads"] >= 1
    assert metadata["torch_runtime"]["deterministic_algorithms"] is True


def test_filter_ensemble_members_avoids_overpruning_and_preserves_top2_fallback():
    from deep_learning.training.trainer import _filter_ensemble_members

    # On ~19 validation weeks, realistic mae=1.02 and sample val_mean=0.004 should pass together
    healthy = [
        {"seed": 42, "path": "a.ckpt", "val_loss": 1.00, "val_mae_vs_naive": 1.02, "val_pred_mean": 0.004, "train_pred_mean": 0.002},
        {"seed": 137, "path": "b.ckpt", "val_loss": 1.04, "val_mae_vs_naive": 1.03, "val_pred_mean": -0.005, "train_pred_mean": -0.003},
        {"seed": 256, "path": "c.ckpt", "val_loss": 1.35, "val_mae_vs_naive": 1.15, "val_pred_mean": 0.025, "train_pred_mean": 0.015},
    ]
    accepted, _ = _filter_ensemble_members(healthy)
    assert [m["seed"] for m in accepted] == [42, 137]

    # When all 3 exceed strict bounds modestly, fallback preserves top-2 finite seeds instead of 1 noisy seed
    borderline = [
        {"seed": 42, "path": "a.ckpt", "val_loss": 1.00, "val_mae_vs_naive": 1.08, "val_pred_mean": 0.008, "train_pred_mean": 0.004},
        {"seed": 137, "path": "b.ckpt", "val_loss": 1.02, "val_mae_vs_naive": 1.09, "val_pred_mean": 0.009, "train_pred_mean": 0.004},
        {"seed": 256, "path": "c.ckpt", "val_loss": 1.50, "val_mae_vs_naive": 1.30, "val_pred_mean": 0.050, "train_pred_mean": 0.020},
    ]
    fb_acc, _ = _filter_ensemble_members(borderline)
    assert [m["seed"] for m in fb_acc] == [42, 137]


def test_hybrid_blend_macro_zscore_and_theta_rolling_invariance():
    import pandas as pd
    from deep_learning.calibration.hybrid import (
        apply_theta_hybrid_shift_to_path,
        compute_causal_macro_fair_value_zscore,
        compute_causal_theta_5d_series,
        fit_theta_hybrid_blend,
    )

    rng = np.random.default_rng(11)
    y = rng.normal(0.0, 0.03, size=80)
    # TFT has slight real edge; Theta correlates modestly with opposite timing -> blending worsens score
    p_tft = y * 0.40 + rng.normal(0.0, 0.005, size=80)
    p_theta = -y * 0.10 + rng.normal(0.0, 0.010, size=80)
    blend = fit_theta_hybrid_blend(y, p_tft, p_theta)
    assert blend["enabled"] is False
    assert blend["w_theta"] == 0.0

    # apply_theta_hybrid_shift_to_path(..., already_ordered=True) shifts all channels coherently
    ord_path = np.tile(np.array([[-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]]), (80, 5, 1))
    shifted = apply_theta_hybrid_shift_to_path(
        ord_path,
        np.full(80, 0.02),
        hybrid_blend={"enabled": True, "w_theta": 0.5},
        already_ordered=True,
    )
    assert np.all(np.diff(shifted, axis=-1) > 0)
    assert np.allclose(shifted[:, :, 5] - shifted[:, :, 1], 0.04)

    # Low-vol macro proxy (std ~ 0.002) should contribute comparably to high-vol target in Ridge zscore
    idx = pd.bdate_range("2023-01-02", periods=100)
    dxy = rng.normal(0.0, 0.002, size=100)
    hg = -3.0 * dxy + rng.normal(0.0, 0.001, size=100)
    df = pd.DataFrame({"HG=F_ret20": hg, "DX_Y_NYB_ret20": dxy}, index=idx)
    z = compute_causal_macro_fair_value_zscore(df, window=60, min_periods=40, ridge_alpha=1.0)
    # Without standardization, ridge_alpha=1 on variance 4e-6 would leave residual == hg (std ~0.006);
    # with standardized X, strong linear signal removes explained variance cleanly and standardizes z
    assert np.isfinite(z.to_numpy()).all()
    assert abs(float(z.iloc[-1])) <= 4.0

    # Rolling theta_5d_prior depends only on trailing lookback_window bars, invariant to earlier years
    px_base = pd.Series(4.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=200))), index=pd.bdate_range("2022-01-03", periods=200))
    px_shifted_early = px_base.copy()
    px_shifted_early.iloc[:50] = px_shifted_early.iloc[:50] * 0.5
    s1 = compute_causal_theta_5d_series(px_base, lookback_window=126, max_eval_tail=10)
    s2 = compute_causal_theta_5d_series(px_shifted_early, lookback_window=126, max_eval_tail=10)
    assert np.allclose(s1.iloc[-10:].to_numpy(), s2.iloc[-10:].to_numpy(), atol=1e-9)

    # Heavily single-sign skewed Theta prior (>70% positive on balanced validation truth) must be rejected
    y_bal = rng.normal(0.0, 0.03, size=92)
    p_tft_ok = 0.25 * y_bal + rng.normal(0.0, 0.005, size=92)
    p_theta_skewed = np.abs(rng.normal(0.004, 0.002, size=92)) + 0.001  # ~100% positive
    skew_blend = fit_theta_hybrid_blend(y_bal, p_tft_ok, p_theta_skewed)
    assert skew_blend["enabled"] is False
    assert skew_blend["w_theta"] == 0.0
    assert skew_blend["reason"] == "tft_standalone_preferred"


def test_ordered_space_ensemble_averaging_avoids_jensen_pi80_shrinkage():
    from types import SimpleNamespace
    from deep_learning.models.tft_copper import ensure_ordered_quantiles_from_model_output

    # Raw crossed logits vs pre-ordered WeeklyASROPFLoss output handling
    ordered_loss_mdl = SimpleNamespace(loss=SimpleNamespace(emits_ordered_quantiles=True))
    cfg = get_tft_config()
    seed1_ord = np.tile(np.array([[-0.03, -0.02, -0.01, -0.01, 0.00, 0.01, 0.02]]), (6, 5, 1))
    seed2_ord = np.tile(np.array([[-0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05]]), (6, 5, 1))

    p1 = _predict_quantiles_to_np(_DummyModel(seed1_ord), dataloader=object(), cfg=cfg, ordered=True)
    # Bypass re-softplussing when emits_ordered_quantiles=True
    p1_checked = ensure_ordered_quantiles_from_model_output(p1, model=ordered_loss_mdl)
    p2 = _predict_quantiles_to_np(_DummyModel(seed2_ord), dataloader=object(), cfg=cfg, ordered=True)
    avg_ord = np.mean(np.stack([p1_checked, p2], axis=0), axis=0)

    # Mean of two ordered spreads (0.03 and 0.04) keeps 0.035 exact without Jensen collapse
    assert np.allclose(avg_ord[:, :, 5] - avg_ord[:, :, 1], 0.035)
    assert np.all(np.diff(avg_ord, axis=-1) > 0)

    # Crossed raw logits on legacy mock without emits_ordered_quantiles are ordered once
    crossed = np.tile(np.array([[-2.5, -1.8, -1.5, 0.001, -1.5, -1.8, -2.5]]), (4, 5, 1))
    converted = ensure_ordered_quantiles_from_model_output(crossed, model=_DummyModel(crossed))
    assert np.all(np.diff(converted, axis=-1) > 0)
