import numpy as np
import pandas as pd
import torch
from dataclasses import replace
import json

from deep_learning.calibration.conformal import (
    apply_conformal_interval,
    interval_coverage,
    rolling_conformal_adjustment,
    select_bucket_adjustment,
)
from deep_learning.config import get_tft_config
from deep_learning.training.metrics import fit_weekly_interval_scale
from deep_learning.training.trainer import _write_conformal_calibration_artifact


def test_undercovered_intervals_produce_positive_adjustment():
    actual = np.linspace(-0.05, 0.05, 60)
    lower = actual - 0.001
    upper = actual + 0.001
    upper[:30] = actual[:30] - 0.01
    adj = rolling_conformal_adjustment(actual, lower, upper, alpha=0.20)
    assert adj > 0


def test_adjustment_widens_interval():
    lower, upper = apply_conformal_interval(np.array([0.0]), np.array([1.0]), 0.2)
    assert lower[0] == -0.2
    assert upper[0] == 1.2


def test_fewer_than_30_samples_returns_zero_adjustment():
    actual = np.ones(20)
    assert rolling_conformal_adjustment(actual, actual - 0.1, actual + 0.1) == 0.0


def test_bucket_fallback_uses_global_adjustment():
    assert select_bucket_adjustment({"global_adjustment": 0.12, "bucket_adjustments": {}}, "high_vol") == 0.12


class _DummyModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, _dl, mode=None):
        assert mode == "quantiles"
        return self.prediction


def _cfg_for_tmp_model(tmp_path):
    cfg = get_tft_config()
    training = replace(
        cfg.training,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        best_model_path=str(tmp_path / "best.ckpt"),
    )
    return replace(cfg, training=training)


def test_conformal_artifact_skips_overcovered_validation_intervals(tmp_path):
    cfg = _cfg_for_tmp_model(tmp_path)
    actual = torch.zeros((40, 5), dtype=torch.float32)
    pred = torch.zeros((40, 5, 7), dtype=torch.float32)
    pred[..., 0] = -2.0
    pred[..., 1] = -1.0
    pred[..., 2] = -0.5
    pred[..., 3] = 0.0
    pred[..., 4] = 0.5
    pred[..., 5] = 1.0
    pred[..., 6] = 2.0

    path = _write_conformal_calibration_artifact(
        cfg=cfg,
        model=_DummyModel(pred),
        val_dl=[(None, (actual, None))],
        feature_frame=None,
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["global_adjustment"] == 0.0
    assert data["calibration_status"] == "skipped_interval_already_overcovered"
    assert data["validation_pi80_coverage"] == 1.0


def test_conformal_artifact_fits_positive_adjustment_for_undercoverage(tmp_path):
    cfg = _cfg_for_tmp_model(tmp_path)
    weekly_actual = torch.linspace(-0.05, 0.05, 40, dtype=torch.float32)
    actual = weekly_actual.reshape(-1, 1).repeat(1, 5) / 5.0
    pred = torch.zeros((40, 5, 7), dtype=torch.float32)
    pred[..., 0] = -0.002
    pred[..., 1] = -0.001
    pred[..., 2] = -0.0005
    pred[..., 3] = 0.0
    pred[..., 4] = 0.0005
    pred[..., 5] = 0.001
    pred[..., 6] = 0.002

    path = _write_conformal_calibration_artifact(
        cfg=cfg,
        model=_DummyModel(pred),
        val_dl=[(None, (actual, None))],
        feature_frame=None,
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["global_adjustment"] > 0.0
    assert data["calibration_status"] == "fit"
    assert data["validation_pi80_coverage"] < 0.90


def test_conformal_artifact_applies_width_floor_and_reports_calibrated_coverage(tmp_path):
    cfg = _cfg_for_tmp_model(tmp_path)
    weekly_actual = torch.linspace(-0.05, 0.05, 40, dtype=torch.float32)
    actual = weekly_actual.reshape(-1, 1).repeat(1, 5) / 5.0
    pred = torch.zeros((40, 5, 7), dtype=torch.float32)
    pred[..., 0] = -0.006
    pred[..., 1] = -0.004
    pred[..., 2] = -0.002
    pred[..., 3] = 0.0
    pred[..., 4] = 0.002
    pred[..., 5] = 0.004
    pred[..., 6] = 0.006

    path = _write_conformal_calibration_artifact(
        cfg=cfg,
        model=_DummyModel(pred),
        val_dl=[(None, (actual, None))],
        feature_frame=None,
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["width_floor_adjustment"] > 0.0
    assert data["global_adjustment"] >= data["width_floor_adjustment"]
    assert data["calibrated_validation_pi80_coverage"] >= data["validation_pi80_coverage"]
    assert data["validation_pi80_width_ratio"] < data["target_min_pi80_width_ratio"]


def test_conformal_artifact_reuses_validation_origin_interval_conditioning(tmp_path):
    cfg = _cfg_for_tmp_model(tmp_path)
    weekly_actual = torch.linspace(-0.05, 0.05, 40, dtype=torch.float32)
    actual = weekly_actual.reshape(-1, 1).repeat(1, 5) / 5.0
    pred = torch.zeros((40, 5, 7), dtype=torch.float32)
    pred[..., 1] = -0.004
    pred[..., 3] = 0.0
    pred[..., 5] = 0.004
    conditioning = np.linspace(0.5, 1.5, len(actual))
    interval_calibration = fit_weekly_interval_scale(
        actual.numpy(),
        pred.numpy(),
        horizon=5,
        conditioning_values=conditioning,
        conditioning_feature="realized_vol_20d",
    )
    feature_frame = pd.DataFrame(
        {
            "time_idx": np.arange(41),
            "realized_vol_20d": np.concatenate(([0.5], conditioning)),
        }
    )

    path = _write_conformal_calibration_artifact(
        cfg=cfg,
        model=_DummyModel(pred),
        val_dl=[(None, (actual, None))],
        feature_frame=feature_frame,
        validation_start_time=0,
        validation_end_time=40,
        weekly_interval_calibration=interval_calibration,
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["weekly_interval_conditioning_enabled"] is True
    assert data["weekly_interval_conditioning_feature"] == "realized_vol_20d"
    assert data["weekly_interval_conditioning_reference"] == interval_calibration[
        "weekly_interval_conditioning_reference"
    ]


def test_interval_coverage_improves_after_symmetric_widening_without_median_change():
    actual = np.array([-0.04, -0.02, 0.02, 0.04])
    median = np.array([-0.01, -0.01, 0.01, 0.01])
    lower = median - 0.005
    upper = median + 0.005
    widened_lower, widened_upper = apply_conformal_interval(lower, upper, 0.04)

    assert np.allclose((widened_lower + widened_upper) / 2.0, median)
    assert interval_coverage(actual, widened_lower, widened_upper) > interval_coverage(
        actual,
        lower,
        upper,
    )


def test_predict_quantiles_to_np_single_pass_avoids_double_iteration():
    """Verify single-pass prediction and target extraction avoids iterating dataloader twice."""
    from collections import namedtuple
    from deep_learning.training.trainer import _predict_quantiles_to_np

    Prediction = namedtuple("Prediction", ["output", "y"])

    pred_tensor = torch.zeros((10, 5, 7))
    y_tensor = torch.ones((10, 5))

    class _SinglePassModel:
        def predict(self, _dl, mode=None, return_y=False):
            assert mode == "quantiles"
            if return_y:
                return Prediction(output=pred_tensor, y=[y_tensor])
            return pred_tensor

    class _SingleIterationDataLoader:
        """Dataloader that can only be iterated once."""
        def __init__(self):
            self.iter_count = 0

        def __iter__(self):
            self.iter_count += 1
            if self.iter_count > 1:
                raise RuntimeError("DataLoader was iterated more than once!")
            yield (None, (y_tensor, None))

    cfg = get_tft_config()
    model = _SinglePassModel()
    dl = _SingleIterationDataLoader()

    # Should succeed without raising RuntimeError from double iteration
    pred_np, y_np = _predict_quantiles_to_np(model, dl, cfg, return_y=True)

    assert pred_np.shape == (10, 5, 7)
    assert y_np.shape == (10, 5)
    # dl was never iterated because return_y=True handled target extraction in model.predict
    assert dl.iter_count == 0


def test_fit_weekly_cqr_records_n_eff_and_caps_outlier_level():
    from deep_learning.calibration.conformal import fit_weekly_cqr

    rng = np.random.default_rng(7)
    n = 100
    y = rng.normal(0.0, 0.025, size=n)
    y[0] = 0.45  # extreme single crash outlier should not blow q_hat_96 to max outlier at level=1.0
    med = np.zeros(n)
    offsets = np.array([-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04])
    q_arr = med[:, None] + offsets[None, :]
    vol = np.linspace(0.010, 0.020, n)

    cqr = fit_weekly_cqr(y, q_arr, realized_vol=vol, overlap_horizon=5)

    assert cqr["cqr_enabled"] is True
    assert cqr["overlap_horizon"] == 5
    assert cqr["n_eff"] == 20.0
    assert cqr["base_vol_power"] == 1.0
    # Extreme single outlier (0.45 - 0.04 = 0.41) must not dominate capped conformal level
    assert float(cqr["q_hat_96"]) < 0.20


def test_apply_weekly_cqr_scales_inner_quartiles_proportionally():
    from deep_learning.calibration.conformal import apply_weekly_cqr

    raw = np.array([[-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04]])
    # Identity calibration leaves q25/q75 exact
    same = apply_weekly_cqr(
        raw,
        cqr_params={"cqr_enabled": True, "q_hat_80": 0.0, "q_hat_96": 0.0, "base_vol_power": 1.0, "median_val_vol": 0.02},
        realized_vol=0.02,
    )
    assert np.allclose(same, raw)

    # Doubling vol + positive q_hat_80 scales inner quartiles proportionally inside [q10, q90]
    widened = apply_weekly_cqr(
        raw,
        cqr_params={"cqr_enabled": True, "q_hat_80": 0.01, "q_hat_96": 0.01, "base_vol_power": 1.0, "median_val_vol": 0.01},
        realized_vol=0.02,
    )
    assert np.all(np.diff(widened[0]) > 0)
    frac_lo_in = (raw[0, 3] - raw[0, 2]) / (raw[0, 3] - raw[0, 1])
    frac_lo_out = (widened[0, 3] - widened[0, 2]) / (widened[0, 3] - widened[0, 1])
    assert abs(frac_lo_out - frac_lo_in) < 1e-6


def test_write_conformal_calibration_artifact_persists_base_vol_power(tmp_path):
    from deep_learning.calibration.conformal import apply_weekly_cqr, fit_weekly_cqr

    cfg = _cfg_for_tmp_model(tmp_path)
    actual = torch.linspace(-0.03, 0.03, 40).reshape(-1, 1).repeat(1, 5) / 5.0
    pred = torch.zeros((40, 5, 7), dtype=torch.float32)
    offsets = [-0.012, -0.006, -0.003, 0.0, 0.003, 0.006, 0.012]
    for i, off in enumerate(offsets):
        pred[..., i] = off
    vol = np.linspace(0.01, 0.03, 40)
    feature_frame = pd.DataFrame(
        {"time_idx": np.arange(41), "realized_vol_20d": np.concatenate(([0.01], vol))}
    )
    from deep_learning.models.monotonic_quantiles import aggregate_weekly_quantiles_np
    weekly_q = aggregate_weekly_quantiles_np(pred.numpy(), median_idx=3)
    weekly_y = actual.numpy().sum(axis=1)
    in_mem_cqr = fit_weekly_cqr(weekly_y, weekly_q, realized_vol=vol)

    path = _write_conformal_calibration_artifact(
        cfg=cfg,
        model=_DummyModel(pred),
        val_dl=[(None, (actual, None))],
        feature_frame=feature_frame,
        validation_start_time=0,
        validation_end_time=40,
        weekly_cqr_params=in_mem_cqr,
        already_ordered=True,
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["base_vol_power"] == 1.0
    assert loaded["overlap_horizon"] == 5
    assert loaded["n_eff"] == in_mem_cqr["n_eff"]

    probe_q = weekly_q[:1]
    assert np.allclose(
        apply_weekly_cqr(probe_q, cqr_params=in_mem_cqr, realized_vol=0.025),
        apply_weekly_cqr(probe_q, cqr_params=loaded, realized_vol=0.025),
    )
