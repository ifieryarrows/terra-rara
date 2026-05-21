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
        lambda_magnitude=0.0,
        lambda_naive=0.0,
        lambda_directional=0.0,
    )

    assert loss.loss(pred_matched, actual) < loss.loss(pred_overdispersed, actual)


def test_naive_relative_loss_penalizes_worse_than_zero_weekly_median():
    actual = torch.tensor(
        [
            [0.010, 0.005, 0.000, 0.000, 0.000],
            [-0.010, -0.005, 0.000, 0.000, 0.000],
            [0.020, 0.005, 0.000, 0.000, 0.000],
            [-0.020, -0.005, 0.000, 0.000, 0.000],
        ]
    )
    pred_matched = _path_from_median(actual)
    pred_bad = _path_from_median(actual * -5.0)
    loss = tft_copper.WeeklyASROPFLoss(
        QUANTILES,
        lambda_weekly_quantile=0.0,
        lambda_t1_quantile=0.0,
        lambda_dispersion=0.0,
        lambda_magnitude=0.0,
        lambda_naive=1.0,
        lambda_directional=0.0,
    )

    assert loss.loss(pred_matched, actual) < loss.loss(pred_bad, actual)


def test_bias_loss_prefers_matching_weekly_mean():
    actual = torch.tensor(
        [
            [0.010, 0.005, 0.000, 0.000, 0.000],
            [-0.010, -0.005, 0.000, 0.000, 0.000],
            [0.020, 0.000, 0.000, 0.000, 0.000],
            [-0.020, 0.000, 0.000, 0.000, 0.000],
        ]
    )
    pred_matched = _path_from_median(actual)
    pred_biased = _path_from_median(actual + 0.02)
    loss = tft_copper.WeeklyASROPFLoss(
        QUANTILES,
        lambda_weekly_quantile=0.0,
        lambda_t1_quantile=0.0,
        lambda_dispersion=0.0,
        lambda_magnitude=0.0,
        lambda_naive=0.0,
        lambda_bias=1.0,
        lambda_directional=0.0,
    )

    assert loss.loss(pred_matched, actual) < loss.loss(pred_biased, actual)


def test_positive_rate_penalty_prefers_balanced_weekly_signs_for_mixed_actuals():
    actual = torch.tensor(
        [
            [-0.006, -0.004, 0.000, 0.000, 0.000],
            [0.005, 0.004, 0.000, 0.000, 0.000],
            [0.006, 0.006, 0.000, 0.000, 0.000],
            [0.007, 0.006, 0.000, 0.000, 0.000],
        ]
    )
    pred_balanced = _path_from_median(actual, spread=0.002)
    pred_all_positive = _path_from_median(actual.abs() + 0.004, spread=0.002)
    pred_all_negative = _path_from_median(-(actual.abs() + 0.004), spread=0.002)
    loss = tft_copper.WeeklyASROPFLoss(
        QUANTILES,
        lambda_weekly_quantile=0.0,
        lambda_t1_quantile=0.0,
        lambda_dispersion=0.0,
        lambda_magnitude=0.0,
        lambda_naive=0.0,
        lambda_bias=0.0,
        lambda_directional=0.0,
        lambda_saturation=0.0,
        lambda_positive_rate=1.0,
        lambda_interval=0.0,
    )

    assert loss.loss(pred_balanced, actual).item() == pytest.approx(0.0, abs=1e-6)
    assert loss.loss(pred_balanced, actual) < loss.loss(pred_all_positive, actual)
    assert loss.loss(pred_balanced, actual) < loss.loss(pred_all_negative, actual)


def test_interval_loss_prefers_wider_weekly_pi80_without_changing_q50():
    actual = torch.tensor(
        [
            [-0.008, -0.008, -0.008, -0.008, -0.008],
            [-0.004, -0.004, -0.004, -0.004, -0.004],
            [0.004, 0.004, 0.004, 0.004, 0.004],
            [0.008, 0.008, 0.008, 0.008, 0.008],
        ]
    )
    median = actual * 0.25
    pred_narrow = _path_from_median(median, spread=0.0002)
    pred_wide = _path_from_median(median, spread=0.0040)
    loss = tft_copper.WeeklyASROPFLoss(
        QUANTILES,
        lambda_weekly_quantile=0.0,
        lambda_t1_quantile=0.0,
        lambda_dispersion=0.0,
        lambda_magnitude=0.0,
        lambda_naive=0.0,
        lambda_bias=0.0,
        lambda_directional=0.0,
        lambda_saturation=0.0,
        lambda_positive_rate=0.0,
        lambda_interval=1.0,
    )

    assert torch.equal(pred_narrow[..., 3], pred_wide[..., 3])
    assert torch.sign(pred_narrow[..., 3].sum(dim=1)).tolist() == torch.sign(
        pred_wide[..., 3].sum(dim=1)
    ).tolist()
    assert loss.loss(pred_wide, actual) < loss.loss(pred_narrow, actual)


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
    assert means["magnitude_loss_mean"] >= 0.0
    assert means["naive_loss_mean"] >= 0.0
    assert means["bias_loss_mean"] >= 0.0
    assert means["saturation_loss_mean"] >= 0.0
    assert means["positive_rate_loss_mean"] >= 0.0
    assert means["interval_loss_mean"] >= 0.0
    assert means["directional_loss_mean"] >= 0.0
    assert means["total_loss_mean"] == pytest.approx(float(total.detach()))
    assert means["dominant_component"] in {
        "weekly_q",
        "t1_q",
        "dispersion",
        "magnitude",
        "naive",
        "bias",
        "saturation",
        "positive_rate",
        "interval",
        "directional",
    }


def test_weekly_loss_applies_median_cap_before_monotonic_transform():
    actual = torch.tensor([[0.010, 0.000, 0.000, 0.000, 0.000]])
    median = torch.full_like(actual, 0.20)
    raw_pred = _path_from_median(median, spread=0.002).requires_grad_(True)
    loss = tft_copper.WeeklyASROPFLoss(
        QUANTILES,
        weekly_median_cap=0.08,
        debug_mode=True,
    )

    bounded = tft_copper._bound_weekly_median_path(
        raw_pred,
        median_idx=3,
        weekly_median_cap=0.08,
        horizon=5,
    )
    ordered = tft_copper.enforce_monotonic_quantiles(
        bounded,
        median_idx=3,
        min_gap=1e-5,
        gap_scale=tft_copper.DEFAULT_MONOTONIC_GAP_SCALE,
        init_bias=-3.0,
    )
    diagnostics = tft_copper.validate_monotonicity(ordered)

    assert diagnostics["crossing_rate"] == 0.0
    assert torch.abs(ordered[:, :5, 3].sum(dim=1)).max().item() <= 0.080001

    total = loss.loss(raw_pred, actual)
    total.backward()
    assert torch.isfinite(raw_pred.grad).all()


def test_weekly_loss_penalizes_raw_q50_saturation_before_bounding():
    actual = torch.tensor(
        [
            [0.004, 0.004, 0.004, 0.004, 0.004],
            [-0.004, -0.004, -0.004, -0.004, -0.004],
        ]
    )
    pred_in_range = _path_from_median(actual * 2.0, spread=0.001)
    pred_saturated = _path_from_median(actual * 12.5, spread=0.001)
    loss = tft_copper.WeeklyASROPFLoss(
        QUANTILES,
        lambda_weekly_quantile=0.0,
        lambda_t1_quantile=0.0,
        lambda_dispersion=0.0,
        lambda_magnitude=0.0,
        lambda_naive=0.0,
        lambda_bias=0.0,
        lambda_directional=0.0,
        lambda_saturation=1.0,
        weekly_median_cap=0.05,
    )

    assert loss.loss(pred_saturated, actual) > loss.loss(pred_in_range, actual)
