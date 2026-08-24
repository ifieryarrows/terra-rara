import numpy as np

from deep_learning.training.metrics import (
    apply_weekly_sign_correction_np,
    apply_weekly_sign_threshold_np,
    fit_direction_sign_calibration,
)


def _quantile_path(median_path: np.ndarray) -> np.ndarray:
    offsets = np.array([-0.012, -0.008, -0.004, 0.0, 0.004, 0.008, 0.012])
    return median_path[..., None] + offsets


def test_validation_only_direction_calibration_flips_stable_inverse_signal():
    signs = np.array([1.0, -1.0] * 20)
    magnitudes = np.linspace(0.016, 0.032, len(signs))
    actual = np.repeat((signs * magnitudes)[:, None], 5, axis=1)
    pred = _quantile_path(-actual)

    calibration = fit_direction_sign_calibration(actual, pred)

    assert calibration["fit_split"] == "validation"
    assert calibration["direction_sign_multiplier"] == -1
    assert calibration["daily_directional_accuracy"] == 0.0
    assert calibration["daily_directional_accuracy_flipped"] == 1.0
    assert calibration["weekly_directional_accuracy"] == 0.0
    assert calibration["weekly_directional_accuracy_flipped"] == 1.0


def test_direction_calibration_keeps_aligned_signal_unchanged():
    signs = np.array([1.0, -1.0] * 20)
    magnitudes = np.linspace(0.016, 0.032, len(signs))
    actual = np.repeat((signs * magnitudes)[:, None], 5, axis=1)
    pred = _quantile_path(actual)

    calibration = fit_direction_sign_calibration(actual, pred)

    assert calibration["direction_sign_multiplier"] == 1


def test_weekly_sign_threshold_shifts_cumulative_median_only():
    pred = np.zeros((2, 5, 7), dtype=float)
    pred[..., 3] = 0.02
    shifted = apply_weekly_sign_threshold_np(pred, 0.05, horizon=5)

    assert np.allclose(shifted[:, :, 3], 0.01)
    assert np.allclose(shifted[:, :, 0], -0.01)
    assert np.allclose(shifted[:, :, 6], -0.01)


def test_weekly_sign_correction_preserves_t1_scale_and_quantile_order():
    pred = np.zeros((2, 5, 7), dtype=float)
    pred[0, :, 3] = 0.01
    pred[0, :, 2] = 0.008
    pred[0, :, 4] = 0.012
    pred[1, :, 3] = -0.01
    pred[1, :, 2] = -0.012
    pred[1, :, 4] = -0.008

    corrected = apply_weekly_sign_correction_np(pred, 0.08, horizon=5)

    assert np.allclose(corrected[0, :4], pred[0, :4])
    assert np.allclose(corrected[0, -1, 3], -0.09)
    assert np.allclose(corrected[0, -1, 2], -0.092)
    assert np.allclose(corrected[0, -1, 4], -0.088)
    assert np.allclose(corrected[1], pred[1])
    assert np.isclose(np.abs(corrected[0, :, 3].sum()), np.abs(pred[0, :, 3].sum()))
