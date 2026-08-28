import numpy as np

from deep_learning.training.metrics import (
    apply_daily_sign_correction_np,
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
    assert calibration["daily_sign_multiplier"] == 1
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


def test_weekly_sign_correction_flips_negative_collapse_toward_positive():
    pred = np.zeros((2, 5, 7), dtype=float)
    pred[0, :, 3] = -0.01
    pred[0, :, 2] = -0.012
    pred[0, :, 4] = -0.008
    pred[1, :, 3] = 0.01
    pred[1, :, 2] = 0.008
    pred[1, :, 4] = 0.012

    corrected = apply_weekly_sign_correction_np(pred, -0.08, horizon=5)

    assert np.allclose(corrected[0, :4], pred[0, :4])
    assert np.allclose(corrected[0, -1, 3], 0.09)
    assert np.allclose(corrected[0, -1, 2], 0.088)
    assert np.allclose(corrected[0, -1, 4], 0.092)
    assert np.allclose(corrected[1], pred[1])
    assert corrected[0, :, 3].sum() > 0.0
    assert np.isclose(
        np.abs(corrected[0, :, 3].sum()),
        np.abs(pred[0, :, 3].sum()),
    )


def test_negative_collapse_calibration_and_correction_are_end_to_end_symmetric():
    actual_sign = np.array([-1.0] * 20 + [1.0] * 20)
    actual = np.repeat((actual_sign * 0.01)[:, None], 5, axis=1)
    weekly_pred = np.concatenate(
        [np.linspace(-0.10, -0.06, 20), np.linspace(-0.04, -0.01, 20)]
    )
    pred = _quantile_path(np.repeat((weekly_pred / 5.0)[:, None], 5, axis=1))

    calibration = fit_direction_sign_calibration(actual, pred)
    threshold = float(calibration["weekly_sign_threshold"])
    corrected = apply_weekly_sign_correction_np(pred, threshold, horizon=5)
    corrected_weekly = corrected[:, :5, 3].sum(axis=1)
    corrected_da = np.mean((actual.sum(axis=1) > 0.0) == (corrected_weekly > 0.0))

    assert np.all(weekly_pred < 0.0)
    assert calibration["weekly_sign_threshold_applied"] is True
    assert threshold < 0.0
    assert np.any(corrected_weekly > 0.0)
    assert corrected_da == calibration["weekly_sign_threshold_validation_da"]


def test_daily_sign_correction_only_changes_first_prediction_step():
    pred = np.ones((2, 5, 7), dtype=float)
    corrected = apply_daily_sign_correction_np(pred, -1)

    assert np.allclose(corrected[:, 0], -1.0)
    assert np.allclose(corrected[:, 1:4], 1.0)
    assert np.allclose(corrected[:, 4], 3.0)
    assert np.allclose(corrected[:, :, 3].sum(axis=1), pred[:, :, 3].sum(axis=1))
