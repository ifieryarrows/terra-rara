import numpy as np

from deep_learning.models.monotonic_quantiles import enforce_monotonic_quantiles
from deep_learning.training.metrics import (
    apply_daily_sign_correction_np,
    apply_direction_sign_multiplier_np,
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


def test_apply_direction_sign_multiplier_preserves_softplus_logit_intervals():
    import torch

    # Raw TFT softplus logits: positive step logits around median m=0.01
    raw = np.tile(np.array([1.30, 0.90, 1.05, 0.01, 1.10, 0.85, 1.15], dtype=float), (2, 5, 1))
    ordered_orig = enforce_monotonic_quantiles(torch.tensor(raw)).numpy()

    flipped_raw = apply_direction_sign_multiplier_np(raw, -1)
    ordered_flipped = enforce_monotonic_quantiles(torch.tensor(flipped_raw)).numpy()

    np.testing.assert_allclose(ordered_flipped[..., 3], -ordered_orig[..., 3], atol=1e-12)
    np.testing.assert_allclose(ordered_flipped, -ordered_orig[..., ::-1], atol=1e-12)


def test_fit_direction_sign_calibration_recenters_collapsed_forecast_without_global_flip():
    # Validation has 65% positive actuals while raw forecast is 96% negative-collapsed
    # with random ordering among predictions (flipped DA = 65% >= 55% would previously flip!).
    rng = np.random.default_rng(11)
    n = 60
    actual_signs = np.array([1.0] * 39 + [-1.0] * 21)
    actual = np.repeat((actual_signs * 0.005)[:, None], 5, axis=1)
    # Mix a few slightly positive so pred_pos_rate = 0.05 (< 0.15)
    raw_med = np.concatenate([np.full(57, -0.02), np.full(3, 0.005)]) + rng.normal(0.0, 0.002, n)
    pred = _quantile_path(np.repeat((raw_med / 5.0)[:, None], 5, axis=1))

    cal = fit_direction_sign_calibration(actual, pred, horizon=5, min_samples=30)

    assert cal["direction_sign_multiplier"] == 1
    assert cal["weekly_sign_threshold_applied"] is True
    assert 0.25 <= cal["weekly_sign_threshold_validation_pred_positive_rate"] <= 0.75
