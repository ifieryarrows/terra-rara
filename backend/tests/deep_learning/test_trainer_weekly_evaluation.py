import numpy as np
import pytest

from deep_learning.config import get_tft_config
from deep_learning.training import metrics as metrics_module
from deep_learning.training.trainer import (
    _compute_test_metrics_from_quantiles,
    _log_weekly_alignment_sample,
    _predict_quantiles_to_np,
    _require_promotable_metrics,
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
    actual, pred = _prediction_fixture()
    calls = []
    real_evaluator = metrics_module.evaluate_quantile_predictions

    def spy_evaluator(y_actual_path, pred_np, *, quantiles, horizon):
        calls.append((y_actual_path.shape, pred_np.shape, tuple(quantiles), horizon))
        return real_evaluator(
            y_actual_path,
            pred_np,
            quantiles=quantiles,
            horizon=horizon,
        )

    monkeypatch.setattr(metrics_module, "evaluate_quantile_predictions", spy_evaluator)

    metrics = _compute_test_metrics_from_quantiles(actual, pred, cfg)

    assert calls == [
        (
            actual.shape,
            pred.shape,
            tuple(cfg.model.quantiles),
            cfg.forecast.primary_horizon_days,
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
