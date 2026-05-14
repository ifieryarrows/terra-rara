import pytest
import torch

from deep_learning.models import tft_copper


pytestmark = pytest.mark.skipif(
    getattr(tft_copper, "WeeklyASROPFLoss", None) is None,
    reason="pytorch_forecasting is not installed",
)


QUANTILES = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]


def _path_from_median(median: torch.Tensor, spread: float = 0.004) -> torch.Tensor:
    offsets = torch.tensor([-3, -2, -1, 0, 1, 2, 3], dtype=median.dtype) * spread
    return median.unsqueeze(-1) + offsets.view(1, 1, -1)


def test_correct_weekly_direction_has_lower_loss_than_anti_direction():
    actual = torch.tensor([[0.01, 0.01, 0.0, 0.0, 0.0], [-0.01, -0.01, 0.0, 0.0, 0.0]])
    pred_good = _path_from_median(actual)
    pred_bad = _path_from_median(-actual)
    loss = tft_copper.WeeklyASROPFLoss(QUANTILES)
    assert loss.loss(pred_good, actual) < loss.loss(pred_bad, actual)


def test_dispersion_loss_prefers_matching_weekly_median_scale():
    actual = torch.tensor(
        [
            [0.010, 0.005, 0.000, 0.000, 0.000],
            [-0.010, -0.005, 0.000, 0.000, 0.000],
            [0.020, 0.005, 0.000, 0.000, 0.000],
            [-0.020, -0.005, 0.000, 0.000, 0.000],
        ]
    )
    pred_matched = _path_from_median(actual)
    pred_overdispersed = _path_from_median(actual * 5.0)
    loss = tft_copper.WeeklyASROPFLoss(
        QUANTILES,
        lambda_weekly_quantile=0.0,
        lambda_t1_quantile=0.0,
        lambda_dispersion=1.0,
        lambda_directional=0.0,
    )

    assert loss.loss(pred_matched, actual) < loss.loss(pred_overdispersed, actual)


def test_weekly_loss_rejects_removed_soft_guard_parameters():
    with pytest.raises(TypeError):
        tft_copper.WeeklyASROPFLoss(QUANTILES, lambda_crossing=10.0)
    with pytest.raises(TypeError):
        tft_copper.WeeklyASROPFLoss(QUANTILES, lambda_width=10.0)
    with pytest.raises(TypeError):
        tft_copper.WeeklyASROPFLoss(QUANTILES, lambda_tail_width=10.0)
    with pytest.raises(TypeError):
        tft_copper.WeeklyASROPFLoss(QUANTILES, lambda_sanity=10.0)


def test_weekly_loss_tracks_component_means():
    actual = torch.tensor([[0.01, 0.01, 0.0, 0.0, 0.0], [-0.01, -0.01, 0.0, 0.0, 0.0]])
    pred = _path_from_median(actual)
    loss = tft_copper.WeeklyASROPFLoss(QUANTILES)

    loss.reset_component_accumulators()
    total = loss.loss(pred, actual)
    means = loss.component_means()

    assert means["n_batches"] == 1
    assert means["weekly_q_loss_mean"] >= 0.0
    assert means["t1_q_loss_mean"] >= 0.0
    assert means["dispersion_loss_mean"] >= 0.0
    assert means["directional_loss_mean"] >= 0.0
    assert means["total_loss_mean"] == pytest.approx(float(total.detach()))
    assert means["dominant_component"] in {"weekly_q", "t1_q", "dispersion", "directional"}
