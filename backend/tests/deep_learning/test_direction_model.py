import numpy as np
import pandas as pd

from deep_learning.training.direction_model import (
    apply_weekly_direction_model,
    fit_weekly_direction_model,
    predict_weekly_direction,
)


def test_weekly_direction_model_fits_train_origins_and_scores_later_origins():
    time_idx = np.arange(220)
    signal = np.where(time_idx % 2 == 0, 1.0, -1.0)
    frame = pd.DataFrame(
        {
            "time_idx": time_idx,
            "signal": signal,
            "noise": np.sin(time_idx / 7.0),
            "target_5d_log_return": signal * 0.01,
        }
    )

    calibrator = fit_weekly_direction_model(
        frame,
        ["signal", "noise"],
        train_cutoff=150,
        horizon=5,
        max_encoder_length=20,
    )
    probabilities = predict_weekly_direction(
        calibrator,
        frame,
        start_exclusive=150,
        end_inclusive=214,
    )

    assert calibrator["fit_split"] == "train"
    assert calibrator["train_origin_count"] > 60
    assert len(probabilities) == 64
    assert probabilities[0] != probabilities[1]


def test_weekly_direction_model_prefers_fixed_causal_feature_family():
    time_idx = np.arange(220)
    signal = np.where(time_idx % 2 == 0, 1.0, -1.0)
    frame = pd.DataFrame(
        {
            "time_idx": time_idx,
            "news_count": signal,
            "noise": np.sin(time_idx / 7.0),
            "target_5d_log_return": signal * 0.01,
        }
    )

    calibrator = fit_weekly_direction_model(
        frame,
        ["noise", "news_count"],
        train_cutoff=150,
        horizon=5,
        max_encoder_length=20,
    )

    assert calibrator["feature_names"] == ["news_count"]


def test_weekly_direction_model_preserves_quantile_ordering():
    pred = np.zeros((2, 5, 7), dtype=float)
    pred[..., 3] = 0.02
    pred[..., 2] = 0.01
    pred[..., 4] = 0.03
    shifted = apply_weekly_direction_model(pred, np.array([0.0, 1.0]))

    assert np.allclose(shifted[0, :, 3], -0.02)
    assert np.all(shifted[1, :, 2] < shifted[1, :, 3])
    assert np.all(shifted[1, :, 3] < shifted[1, :, 4])
