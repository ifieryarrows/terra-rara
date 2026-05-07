import numpy as np
import pandas as pd
import pytest

from deep_learning.data.validation import validate_weekly_target_contract


def test_log_return_formula():
    close = pd.Series([100, 101, 103, 102, 104, 110], dtype=float)
    log_close = np.log(close)
    target_5d = log_close.shift(-5) - log_close
    assert np.isclose(target_5d.iloc[0], np.log(110 / 100))


def test_validation_train_requires_real_weekly_labels():
    df = pd.DataFrame(
        {
            "target": [0.01, 0.02],
            "target_1d_log_return": [0.01, 0.02],
            "target_5d_log_return": [0.05, np.nan],
            "realized_vol_20d": [0.01, 0.01],
            "material_move_5d": [0.0, 1.0],
            "time_idx": [0, 1],
            "group_id": ["copper", "copper"],
        }
    )
    with pytest.raises(ValueError, match="target_5d_log_return contains NaN"):
        validate_weekly_target_contract(df, mode="train")


def test_validation_inference_allows_dummy_decoder_labels():
    df = pd.DataFrame(
        {
            "target": [0.01, 0.0],
            "target_1d_log_return": [0.01, 0.0],
            "target_5d_log_return": [0.05, 0.0],
            "realized_vol_20d": [0.01, 0.0],
            "material_move_5d": [0.0, 0.0],
            "time_idx": [0, 1],
            "group_id": ["copper", "copper"],
        }
    )
    validate_weekly_target_contract(df, mode="inference")
