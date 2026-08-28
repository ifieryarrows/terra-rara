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
