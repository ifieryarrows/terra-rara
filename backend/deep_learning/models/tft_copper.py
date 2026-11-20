"""
TFT-ASRO Model for Copper Futures Prediction.

Wraps pytorch_forecasting's TemporalFusionTransformer with:
- ASRO (Adaptive Sharpe Ratio Optimization) loss
- 7-quantile probabilistic output
- Variable Selection Network for dynamic feature weighting
- Interpretable attention for temporal pattern analysis
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from deep_learning.contract import RETURN_SPACE, log_to_simple_return
from deep_learning.config import TFTASROConfig, get_tft_config
from deep_learning.models.monotonic_quantiles import (
    DEFAULT_MONOTONIC_GAP_SCALE,
    aggregate_cumulative_quantiles_np,
    aggregate_weekly_quantiles_np,
    aggregate_weekly_quantiles_torch,
    enforce_monotonic_quantiles,
    validate_monotonicity,
)
from deep_learning.models.losses import (
    AdaptiveSharpeRatioLoss,
    CombinedQuantileLoss,
    quantile_crossing_penalty,
)

logger = logging.getLogger(__name__)


def _bound_weekly_median_path(
    y_pred: torch.Tensor,
    *,
    median_idx: int,
    weekly_median_cap: Optional[float],
    horizon: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Scale each sample's q50 path so its cumulative weekly median stays bounded.

    The cap is resolved from training targets outside the loss. This helper does
    not inspect validation/test targets and leaves non-median raw channels as the
    monotonic-distance controls for the structural quantile transform.
    """
    if weekly_median_cap is None or float(weekly_median_cap) <= 0.0:
        return y_pred
    if y_pred.ndim != 3:
        raise ValueError(f"Expected prediction tensor [batch,horizon,q], got {tuple(y_pred.shape)}")

    cap = torch.as_tensor(float(weekly_median_cap), device=y_pred.device, dtype=y_pred.dtype)
    median_path = y_pred[..., median_idx]
    effective_horizon = min(int(horizon), int(median_path.shape[1]))
    weekly_median = median_path[:, :effective_horizon].sum(dim=1)
    scale = torch.clamp(cap / weekly_median.abs().clamp_min(eps), max=1.0)
    bounded_median = median_path * scale.unsqueeze(-1)
    return torch.cat(
        [
            y_pred[..., :median_idx],
            bounded_median.unsqueeze(-1),
            y_pred[..., median_idx + 1:],
        ],
        dim=-1,
    )


def _weekly_saturation_loss(
    raw_pred_weekly_median: torch.Tensor,
    *,
    weekly_median_cap: Optional[float],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize raw weekly q50 paths that rely on clipping at the median cap."""
    if weekly_median_cap is None or float(weekly_median_cap) <= 0.0:
        return raw_pred_weekly_median.sum() * 0.0

    cap = torch.as_tensor(
        float(weekly_median_cap),
        device=raw_pred_weekly_median.device,
        dtype=raw_pred_weekly_median.dtype,
    )
    raw_abs = raw_pred_weekly_median.abs()
    near_threshold = 0.90 * cap
    near_band = torch.clamp(cap - near_threshold, min=eps)
    near_cap_excess = torch.relu(raw_abs - near_threshold) / near_band
    above_cap_excess = torch.relu(raw_abs - cap) / cap.clamp_min(eps)

    # The raw model can start far outside the train-derived cap.  A quadratic
    # barrier (and even a Huber tail) lets a single early batch dominate the
    # objective: the fixed replay reached saturation losses above 170, after
    # which tiny CPU floating-point differences selected different training
    # trajectories.  Keep a strong penalty near the cap, but use a smooth
    # logarithmic tail so the penalty stays informative without growing
    # linearly with an implausibly large raw forecast.
    def _robust_squared(excess: torch.Tensor) -> torch.Tensor:
        return torch.log1p(excess).pow(2)

    # The logarithmic tail keeps pathological first-epoch forecasts stable, but
    # it became too permissive once a candidate was only moderately above the
    # cap: the fresh Linux gate finished at raw MR=3.0397 with 72.6% of weekly
    # medians still relying on the cap.  Add a bounded barrier whose gradient is
    # strongest for ordinary 1-3x cap violations and whose value cannot exceed
    # one per sample.  This creates deployment margin without letting extreme
    # initialization outliers dominate the objective again.
    moderate_violation_barrier = torch.tanh(above_cap_excess).pow(2)
    # Keep a sublinear sqrt restoring penalty on moderate excess above the cap
    # so gradients remain strong at 1.2x-5x violations while staying sublinear
    # on extreme initialization outliers.
    sqrt_excess_penalty = torch.sqrt(1.0 + above_cap_excess) - 1.0

    return (
        0.10 * _robust_squared(near_cap_excess).mean()
        + 2.0 * _robust_squared(above_cap_excess).mean()
        + moderate_violation_barrier.mean()
        + 1.5 * sqrt_excess_penalty.mean()
    )


def _weekly_scale_losses(
    pred_weekly_median: torch.Tensor,
    actual_weekly: torch.Tensor,
    eps: float = 1e-8,
    raw_pred_weekly_median: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """
    Scale diagnostics used by WeeklyASROPFLoss.

    Hyperopt promotes on weekly_magnitude_ratio, so the loss must penalize the
    same failure mode directly instead of relying only on slowly growing
    log-ratio pressure. When raw_pred_weekly_median is provided, also apply
    scale penalties before cap clipping so gradients never zero out at the cap.
    """
    actual_ref = actual_weekly.clamp(min=-0.25, max=0.25)
    pred_abs_median = pred_weekly_median.abs().median() + eps
    actual_abs_median = actual_ref.abs().median() + eps
    magnitude_ratio = pred_abs_median / actual_abs_median

    pred_abs_mean = pred_weekly_median.abs().mean() + eps
    actual_abs_mean = actual_ref.abs().mean() + eps
    mean_magnitude_ratio = pred_abs_mean / actual_abs_mean

    def _bounded_scale_loss(ratio: torch.Tensor) -> torch.Tensor:
        quality_band_excess = torch.relu(ratio - 1.35)
        quality_band_deficit = torch.relu(0.65 - ratio)
        structural_explosion = torch.relu(ratio - 3.0)
        return (
            F.smooth_l1_loss(
                torch.log(ratio.clamp_min(1e-4)),
                torch.zeros_like(ratio),
                beta=0.25,
            )
            + 2.0 * F.smooth_l1_loss(
                quality_band_excess,
                torch.zeros_like(quality_band_excess),
                beta=0.50,
            )
            + 4.0 * quality_band_deficit.pow(2)
            + 2.5 * F.smooth_l1_loss(
                structural_explosion,
                torch.zeros_like(structural_explosion),
                beta=0.50,
            )
        )

    bounded_mag_loss = _bounded_scale_loss(mean_magnitude_ratio)
    if raw_pred_weekly_median is not None:
        raw_abs_mean = raw_pred_weekly_median.abs().mean() + eps
        raw_ratio = raw_abs_mean / actual_abs_mean
        raw_excess = torch.relu(raw_ratio - 1.20)
        raw_mag_loss = (
            _bounded_scale_loss(raw_ratio)
            + F.smooth_l1_loss(raw_excess, torch.zeros_like(raw_excess), beta=0.50)
            + torch.relu(0.75 - raw_ratio).pow(2)
        )
        magnitude_loss = 0.5 * bounded_mag_loss + 0.5 * raw_mag_loss
        effective_pred_weekly_median = 0.5 * pred_weekly_median + 0.5 * raw_pred_weekly_median
    else:
        magnitude_loss = bounded_mag_loss
        effective_pred_weekly_median = pred_weekly_median

    pred_std = effective_pred_weekly_median.std(unbiased=False) + eps
    actual_std = actual_ref.std(unbiased=False) + eps
    dispersion_log_error = torch.log(
        (pred_std / actual_std).clamp(min=0.25, max=4.0)
    )
    dispersion_loss = F.smooth_l1_loss(
        dispersion_log_error,
        torch.zeros_like(dispersion_log_error),
        beta=0.5,
    )

    model_mae = torch.mean(torch.abs(effective_pred_weekly_median - actual_ref))
    zero_mae = actual_abs_mean
    naive_relative_loss = torch.relu((model_mae / zero_mae) - 1.0)
    mean_gap = (effective_pred_weekly_median.mean() - actual_ref.mean()) / zero_mae
    median_gap = (effective_pred_weekly_median.median() - actual_ref.median()) / actual_abs_median
    bias_loss = 1.50 * torch.abs(mean_gap) + 0.50 * torch.abs(median_gap)

    return {
        "dispersion_loss": dispersion_loss,
        "magnitude_loss": magnitude_loss,
        "magnitude_ratio": magnitude_ratio,
        "mean_magnitude_ratio": mean_magnitude_ratio,
        "naive_relative_loss": naive_relative_loss,
        "bias_loss": bias_loss,
    }


def _weekly_positive_rate_loss(
    pred_weekly_median: torch.Tensor,
    actual_weekly: torch.Tensor,
    temperature: float = 0.01,
    lower_bound: float = 0.20,
    upper_bound: float = 0.75,
    tolerance: float = 0.20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Keep predicted weekly signs aligned with the observed batch signs.

    The detached sign targets provide a strong gradient against all-positive
    collapse. The additional bounds are a tolerant band around the observed
    rate rather than an exact-match objective, so the term remains robust to
    small batch-composition noise.
    """
    temp = max(float(temperature), eps)
    scaled_logits = pred_weekly_median / temp
    pred_positive_probability = torch.sigmoid(scaled_logits)
    pred_positive_rate = pred_positive_probability.mean()
    bounded_prob = torch.sigmoid(scaled_logits.clamp(min=-2.5, max=2.5))
    actual_positive_rate = (actual_weekly.detach() > 0).to(
        device=pred_weekly_median.device,
        dtype=pred_weekly_median.dtype,
    ).mean()
    actual_sign = (actual_weekly.detach() > 0).to(
        device=pred_weekly_median.device,
        dtype=pred_weekly_median.dtype,
    )
    positive_fraction = actual_positive_rate.clamp(min=eps, max=1.0 - eps)
    class_weights = torch.where(
        actual_sign > 0.5,
        0.5 / positive_fraction,
        0.5 / (1.0 - positive_fraction),
    )
    sign_loss = F.binary_cross_entropy(
        bounded_prob,
        actual_sign,
        weight=class_weights,
    )
    band = max(float(tolerance), eps)
    lower = torch.clamp(
        actual_positive_rate - band,
        min=float(lower_bound),
        max=float(upper_bound),
    )
    upper = torch.clamp(
        actual_positive_rate + band,
        min=float(lower_bound),
        max=float(upper_bound),
    )
    upper = torch.maximum(upper, lower)
    rate_calibration_loss = 0.5 * (pred_positive_rate - positive_fraction).pow(2)
    rate_band_loss = torch.relu(lower - pred_positive_rate).pow(2) + torch.relu(
        pred_positive_rate - upper
    ).pow(2)
    return sign_loss + rate_calibration_loss + rate_band_loss


def _directional_sign_loss(
    prediction: torch.Tensor,
    actual: torch.Tensor,
    *,
    tanh_scale: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale-matched directional sign and cosine alignment loss for a forecast horizon."""
    soft_prediction_sign = torch.tanh(prediction * float(tanh_scale))
    actual_detached = actual.detach()
    soft_actual_sign = torch.tanh(actual_detached * float(tanh_scale))
    actual_sign = torch.sign(actual_detached).float()
    weights = actual_detached.abs() / actual_detached.abs().mean().clamp_min(eps)
    squared_error = (soft_prediction_sign - soft_actual_sign).pow(2)
    wrong_sign_hinge = torch.relu(-soft_prediction_sign * actual_sign)
    pred_norm = prediction.pow(2).sum().add(eps).sqrt()
    actual_norm = actual_detached.pow(2).sum().add(eps).sqrt()
    cosine_sim = (prediction * actual_detached).sum() / (pred_norm * actual_norm)
    cosine_loss = (1.0 - cosine_sim).clamp(min=0.0, max=2.0)
    return (weights * squared_error).mean() + 0.25 * wrong_sign_hinge.mean() + 0.35 * cosine_loss


def _weekly_interval_undercoverage_loss(
    pred_weekly_quantiles: torch.Tensor,
    actual_weekly: torch.Tensor,
    quantiles: Sequence[float],
    *,
    target_width_ratio: float = 0.85,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize weekly PI80 misses and deviation from the target interval scale."""
    q = list(quantiles)
    q10_idx = q.index(0.10) if 0.10 in q else 1
    q90_idx = q.index(0.90) if 0.90 in q else len(q) - 2
    lower = pred_weekly_quantiles[:, q10_idx]
    upper = pred_weekly_quantiles[:, q90_idx]
    width = torch.clamp(upper - lower, min=0.0)

    actual_ref = actual_weekly.clamp(min=-0.25, max=0.25)
    actual_scale = actual_ref.abs().mean().clamp_min(eps)
    miss_loss = (
        torch.relu(lower - actual_ref)
        + torch.relu(actual_ref - upper)
    ).mean() / actual_scale

    actual_std = actual_ref.std(unbiased=False).clamp_min(eps)
    width_ratio = width.mean() / (2.56 * actual_std + eps)
    target_width = torch.as_tensor(
        float(target_width_ratio),
        device=pred_weekly_quantiles.device,
        dtype=pred_weekly_quantiles.dtype,
    )
    width_scale_loss = F.smooth_l1_loss(
        width_ratio.clamp(max=4.0),
        target_width.clamp(max=4.0),
        beta=0.25,
    )
    return miss_loss + width_scale_loss


# ---------------------------------------------------------------------------
# Module-level ASRO loss class (must be at module level for pickle / checkpoint)
# ---------------------------------------------------------------------------

try:
    from pytorch_forecasting.metrics import QuantileLoss as _PFQuantileLoss

    class ASROPFLoss(_PFQuantileLoss):
        """
        pytorch_forecasting >= 1.0 compatible ASRO loss.

        Inherits from ``QuantileLoss`` (a proper torchmetrics ``Metric``) so
        that ``TemporalFusionTransformer.from_dataset()`` accepts it.
        Defined at module level so Lightning checkpoints can pickle it.
        """

        def __init__(
            self,
            quantiles: list,
            lambda_vol: float = 0.3,
            lambda_quantile: float = 0.2,
            lambda_madl: float = 0.25,
            lambda_crossing: float = 1.0,
            risk_free_rate: float = 0.0,
            sharpe_eps: float = 1e-6,
        ):
            super().__init__(quantiles=quantiles)
            self.lambda_vol = lambda_vol
            self.lambda_quantile = lambda_quantile
            self.lambda_madl = lambda_madl
            self.lambda_crossing = lambda_crossing
            self.rf = risk_free_rate
            self.sharpe_eps = sharpe_eps
            self.median_idx = len(quantiles) // 2
            q = list(quantiles)
            self._q10_idx = q.index(0.10) if 0.10 in q else 1
            self._q90_idx = q.index(0.90) if 0.90 in q else len(q) - 2

        def loss(self, y_pred: torch.Tensor, target) -> torch.Tensor:  # type: ignore[override]
            if isinstance(target, (list, tuple)):
                y_actual = target[0]
            else:
                y_actual = target

            y_actual = y_actual.float()
            median_pred = y_pred[..., self.median_idx]

            # Mirrors losses.AdaptiveSharpeRatioLoss exactly.
            # Sample-level directional reward: each sample gets a clear gradient
            # for its direction, breaking the "batch-average safe mode" trap.
            _TANH_SCALE = 20.0
            signal = torch.tanh(median_pred * _TANH_SCALE)
            strategy_returns = signal * y_actual.float() - self.rf
            directional_reward = (signal * y_actual.float()).mean()
            risk_norm = strategy_returns.std() + self.sharpe_eps
            sharpe_loss = -directional_reward / risk_norm

            # Magnitude-weighted directional bonus (replaces BCE which created
            # noisy labels for small returns, causing anti-correlation)
            abs_actual = y_actual.float().abs()
            magnitude_weight = abs_actual / (abs_actual.mean() + self.sharpe_eps)
            weighted_directional = (signal * y_actual.float() * magnitude_weight).mean()
            sharpe_loss = sharpe_loss - 0.3 * weighted_directional

            # Volatility calibration: match Q90-Q10 spread to 2× actual σ
            pred_spread = (
                y_pred[..., self._q90_idx] - y_pred[..., self._q10_idx]
            ).mean()
            actual_std = y_actual.std() + self.sharpe_eps
            vol_loss = torch.abs(pred_spread - 2.0 * actual_std)

            # Median amplitude: penalise if median pred variance < actual variance
            median_std = median_pred.std() + self.sharpe_eps
            vr = median_std / actual_std
            under_severe   = 2.0 * torch.relu(0.5 - vr)   # fires hard when VR < 0.5
            under_moderate = torch.relu(1.0 - vr)          # fires when VR < 1.0
            over_variance  = 1.0 * torch.relu(vr - 1.5)
            amplitude_loss = under_severe + under_moderate + over_variance

            # Quantile (pinball) loss via parent — covers all 7 quantile bands
            q_loss = super().loss(y_pred, target)
            crossing_loss = quantile_crossing_penalty(y_pred)

            # MADL: direct directional accuracy via magnitude-weighted sign match
            soft_sign_madl = torch.tanh(median_pred * 20.0)
            direction_match = soft_sign_madl * y_actual.float()
            madl_loss = (-direction_match * y_actual.float().abs()).mean()

            w_directional = 1.0 - self.lambda_quantile
            calibration = (
                q_loss
                + self.lambda_vol * (vol_loss + amplitude_loss)
                + self.lambda_crossing * crossing_loss
            )
            directional = sharpe_loss + self.lambda_madl * madl_loss
            return self.lambda_quantile * calibration + w_directional * directional


    class WeeklyASROPFLoss(_PFQuantileLoss):
        """Scale-normalized weekly ASRO + quantile loss over a 5-step log-return path."""

        def __init__(
            self,
            quantiles: list,
            lambda_weekly_quantile: float = 1.00,
            lambda_t1_quantile: float = 0.25,
            lambda_t1_directional: float = 0.0,
            lambda_dispersion: float = 0.0,
            lambda_directional: float = 0.14,
            lambda_magnitude: float = 0.0,
            lambda_naive: float = 0.0,
            lambda_bias: float = 0.02,
            lambda_drift_prior: float = 0.0,
            lambda_saturation: float = 0.0,
            lambda_positive_rate: float = 0.0,
            lambda_interval: float = 0.0,
            weekly_median_cap: Optional[float] = None,
            weekly_target_scale: Optional[float] = None,
            sharpe_eps: float = 1e-8,
            debug_mode: bool = False,
        ):
            super().__init__(quantiles=quantiles)
            self.lambda_weekly_quantile = lambda_weekly_quantile
            self.lambda_t1_quantile = lambda_t1_quantile
            self.lambda_t1_directional = lambda_t1_directional
            self.lambda_dispersion = lambda_dispersion
            self.lambda_directional = lambda_directional
            self.lambda_magnitude = lambda_magnitude
            self.lambda_naive = lambda_naive
            self.lambda_bias = lambda_bias
            self.lambda_drift_prior = lambda_drift_prior
            self.lambda_saturation = lambda_saturation
            self.lambda_positive_rate = lambda_positive_rate
            self.lambda_interval = lambda_interval
            self.weekly_median_cap = weekly_median_cap
            self.weekly_target_scale = (
                float(weekly_target_scale)
                if weekly_target_scale is not None and float(weekly_target_scale) > 0.0
                else None
            )
            self.sharpe_eps = sharpe_eps
            self.debug_mode = debug_mode
            self.median_idx = len(quantiles) // 2
            self.register_buffer(
                "_pinball_q_buf",
                torch.tensor(quantiles, dtype=torch.float32).view(1, -1),
                persistent=False,
            )
            self.reset_component_accumulators()

        def update(self, y_pred: torch.Tensor, target) -> None:  # type: ignore[override]
            """Accumulate scalar weekly loss without MultiHorizonMetric's 1/(B*H) dilution."""
            if isinstance(target, (list, tuple)):
                y_actual = target[0]
            else:
                y_actual = target

            if isinstance(y_actual, torch.nn.utils.rnn.PackedSequence):
                from pytorch_forecasting.utils import unpack_sequence

                y_unpacked, lengths = unpack_sequence(y_actual)
                target_arg = y_unpacked
            else:
                target_arg = y_actual
                lengths = torch.full(
                    (y_actual.size(0),),
                    fill_value=y_actual.size(1),
                    dtype=torch.long,
                    device=y_actual.device,
                )

            loss_val = self.loss(y_pred, target_arg)
            if loss_val.ndim > 0:
                loss_val = loss_val.mean()
            sample_weight = lengths.sum().to(dtype=loss_val.dtype, device=loss_val.device).clamp_min(1.0)
            if not torch.isfinite(loss_val):
                loss_val = torch.tensor(1e9, dtype=loss_val.dtype, device=loss_val.device)
            self.losses = self.losses + loss_val * sample_weight
            self.lengths = self.lengths + sample_weight

        def reset_component_accumulators(self) -> None:
            self._component_sums = {
                "weekly_q": 0.0,
                "t1_q": 0.0,
                "t1_directional": 0.0,
                "dispersion": 0.0,
                "magnitude": 0.0,
                "naive": 0.0,
                "bias": 0.0,
                "drift_prior": 0.0,
                "saturation": 0.0,
                "positive_rate": 0.0,
                "interval": 0.0,
                "directional": 0.0,
                "total": 0.0,
            }
            self._component_batches = 0

        def _record_components(
            self,
            weekly_q_loss: torch.Tensor,
            t1_q_loss: torch.Tensor,
            t1_directional_loss: torch.Tensor,
            dispersion_loss: torch.Tensor,
            magnitude_loss: torch.Tensor,
            naive_relative_loss: torch.Tensor,
            bias_loss: torch.Tensor,
            drift_prior_loss: torch.Tensor,
            saturation_loss: torch.Tensor,
            positive_rate_loss: torch.Tensor,
            interval_loss: torch.Tensor,
            directional_loss: torch.Tensor,
            total_loss: torch.Tensor,
        ) -> None:
            self._component_sums["weekly_q"] += float(weekly_q_loss.detach().mean().cpu())
            self._component_sums["t1_q"] += float(t1_q_loss.detach().mean().cpu())
            self._component_sums["t1_directional"] += float(t1_directional_loss.detach().mean().cpu())
            self._component_sums["dispersion"] += float(dispersion_loss.detach().mean().cpu())
            self._component_sums["magnitude"] += float(magnitude_loss.detach().mean().cpu())
            self._component_sums["naive"] += float(naive_relative_loss.detach().mean().cpu())
            self._component_sums["bias"] += float(bias_loss.detach().mean().cpu())
            self._component_sums["drift_prior"] += float(drift_prior_loss.detach().mean().cpu())
            self._component_sums["saturation"] += float(saturation_loss.detach().mean().cpu())
            self._component_sums["positive_rate"] += float(positive_rate_loss.detach().mean().cpu())
            self._component_sums["interval"] += float(interval_loss.detach().mean().cpu())
            self._component_sums["directional"] += float(directional_loss.detach().mean().cpu())
            self._component_sums["total"] += float(total_loss.detach().mean().cpu())
            self._component_batches += 1

        def component_means(self) -> dict:
            n_batches = self._component_batches
            if n_batches <= 0:
                return {
                    "n_batches": 0,
                    "weekly_q_loss_mean": 0.0,
                    "t1_q_loss_mean": 0.0,
                    "t1_directional_loss_mean": 0.0,
                    "dispersion_loss_mean": 0.0,
                    "magnitude_loss_mean": 0.0,
                    "naive_loss_mean": 0.0,
                    "bias_loss_mean": 0.0,
                    "drift_prior_loss_mean": 0.0,
                    "saturation_loss_mean": 0.0,
                    "positive_rate_loss_mean": 0.0,
                    "interval_loss_mean": 0.0,
                    "directional_loss_mean": 0.0,
                    "total_loss_mean": 0.0,
                    "dominant_component": None,
                }

            components = {
                "weekly_q": abs(self.lambda_weekly_quantile) * self._component_sums["weekly_q"],
                "t1_q": abs(self.lambda_t1_quantile) * self._component_sums["t1_q"],
                "t1_directional": abs(self.lambda_t1_directional) * self._component_sums["t1_directional"],
                "dispersion": abs(self.lambda_dispersion) * self._component_sums["dispersion"],
                "magnitude": abs(self.lambda_magnitude) * self._component_sums["magnitude"],
                "naive": abs(self.lambda_naive) * self._component_sums["naive"],
                "bias": abs(self.lambda_bias) * self._component_sums["bias"],
                "drift_prior": abs(self.lambda_drift_prior) * self._component_sums["drift_prior"],
                "saturation": abs(self.lambda_saturation) * self._component_sums["saturation"],
                "positive_rate": abs(self.lambda_positive_rate) * self._component_sums["positive_rate"],
                "interval": abs(self.lambda_interval) * self._component_sums["interval"],
                "directional": abs(self.lambda_directional) * self._component_sums["directional"],
            }
            return {
                "n_batches": n_batches,
                "weekly_q_loss_mean": self._component_sums["weekly_q"] / n_batches,
                "t1_q_loss_mean": self._component_sums["t1_q"] / n_batches,
                "t1_directional_loss_mean": self._component_sums["t1_directional"] / n_batches,
                "dispersion_loss_mean": self._component_sums["dispersion"] / n_batches,
                "magnitude_loss_mean": self._component_sums["magnitude"] / n_batches,
                "naive_loss_mean": self._component_sums["naive"] / n_batches,
                "bias_loss_mean": self._component_sums["bias"] / n_batches,
                "drift_prior_loss_mean": self._component_sums["drift_prior"] / n_batches,
                "saturation_loss_mean": self._component_sums["saturation"] / n_batches,
                "positive_rate_loss_mean": self._component_sums["positive_rate"] / n_batches,
                "interval_loss_mean": self._component_sums["interval"] / n_batches,
                "directional_loss_mean": self._component_sums["directional"] / n_batches,
                "total_loss_mean": self._component_sums["total"] / n_batches,
                "dominant_component": max(components, key=components.get),
            }

        def _pinball(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
            q = self._pinball_q_buf.to(device=pred.device, dtype=pred.dtype)
            err = actual.unsqueeze(-1) - pred
            return torch.maximum(q * err, (q - 1.0) * err).mean()

        def loss(self, y_pred: torch.Tensor, target) -> torch.Tensor:  # type: ignore[override]
            if isinstance(target, (list, tuple)):
                y_actual = target[0]
            else:
                y_actual = target

            y_actual = y_actual.float()
            y_pred = y_pred.float()
            zero = y_pred.sum() * 0.0

            raw_median_path = y_pred[..., self.median_idx]
            raw_horizon = min(int(y_actual.shape[1]), int(raw_median_path.shape[1]))
            raw_pred_weekly_median = raw_median_path[:, :raw_horizon].sum(dim=1)
            saturation_loss = (
                _weekly_saturation_loss(
                    raw_pred_weekly_median,
                    weekly_median_cap=self.weekly_median_cap,
                    eps=self.sharpe_eps,
                )
                if self.lambda_saturation != 0.0
                else zero
            )

            bounded_pred = _bound_weekly_median_path(
                y_pred,
                median_idx=self.median_idx,
                weekly_median_cap=self.weekly_median_cap,
                horizon=y_actual.shape[1],
                eps=self.sharpe_eps,
            )
            ordered_pred = enforce_monotonic_quantiles(
                bounded_pred,
                median_idx=self.median_idx,
                min_gap=1e-5,
                gap_scale=DEFAULT_MONOTONIC_GAP_SCALE,
                init_bias=-3.0,
            )
            if self.debug_mode:
                ordered_diagnostics = validate_monotonicity(ordered_pred)
                assert ordered_diagnostics["is_valid"], (
                    f"Monotonic transform produced crossings: "
                    f"rate={ordered_diagnostics['crossing_rate']}, "
                    f"max_violation={ordered_diagnostics['max_violation']}"
                )
                assert torch.allclose(
                    ordered_pred[..., self.median_idx],
                    bounded_pred[..., self.median_idx],
                    rtol=1e-6,
                    atol=1e-7,
                ), "Monotonic transform must preserve the median quantile exactly"

            median_path = ordered_pred[..., self.median_idx]
            pred_weekly_quantiles = aggregate_weekly_quantiles_torch(
                ordered_pred,
                median_idx=self.median_idx,
            )
            actual_weekly = y_actual.sum(dim=1)

            horizon_days = max(int(y_actual.shape[1]), 1)
            if self.weekly_target_scale is not None:
                scale_w = torch.as_tensor(
                    self.weekly_target_scale,
                    device=y_pred.device,
                    dtype=y_pred.dtype,
                ).clamp_min(0.005)
            else:
                scale_w = (
                    actual_weekly.detach().pow(2).mean().sqrt().clamp_min(0.005)
                )
            scale_1d = scale_w / math.sqrt(float(horizon_days))

            weekly_q_loss_bounded = self._pinball(pred_weekly_quantiles, actual_weekly) / scale_w
            if self.weekly_median_cap is not None and float(self.weekly_median_cap) > 0.0:
                median_shift_sum = (raw_median_path - median_path).sum(dim=1, keepdim=True)
                weekly_q_loss_raw = self._pinball(
                    pred_weekly_quantiles + median_shift_sum,
                    actual_weekly,
                ) / scale_w
                weekly_q_loss = 0.85 * weekly_q_loss_bounded + 0.15 * weekly_q_loss_raw
            else:
                weekly_q_loss = weekly_q_loss_bounded

            t1_q_loss = (
                self._pinball(ordered_pred[:, 0, :], y_actual[:, 0]) / scale_1d
                if self.lambda_t1_quantile != 0.0
                else zero
            )
            effective_t1_median = 0.80 * median_path[:, 0] + 0.20 * raw_median_path[:, 0]
            t1_directional_loss = (
                _directional_sign_loss(
                    effective_t1_median,
                    y_actual[:, 0],
                    tanh_scale=100.0,
                    eps=self.sharpe_eps,
                )
                if self.lambda_t1_directional != 0.0
                else zero
            )

            pred_weekly_median = median_path.sum(dim=1)
            effective_weekly_median = 0.80 * pred_weekly_median + 0.20 * raw_pred_weekly_median
            eps = self.sharpe_eps

            # Demeaned, scale-matched ASRO directional utility: adding any constant
            # bullish/bearish offset gives strictly zero reward on mixed batches.
            if self.lambda_directional != 0.0:
                actual_det = actual_weekly.detach()
                if actual_det.numel() > 1 and float(actual_det.std(unbiased=False)) > 1e-6:
                    pred_for_dir = effective_weekly_median - effective_weekly_median.mean()
                    norm_actual = (actual_det - actual_det.mean()) / scale_w
                else:
                    pred_for_dir = effective_weekly_median
                    norm_actual = actual_det / scale_w
                pos = torch.tanh(pred_for_dir / scale_w)
                directional_loss = (norm_actual.abs() - pos * norm_actual).mean()
            else:
                directional_loss = zero

            # Scale-consistent bias regularizer (O(0.1) in normalized units)
            if self.lambda_bias != 0.0:
                bias_loss = (
                    (effective_weekly_median.mean() - actual_weekly.detach().mean()).abs()
                    / scale_w
                )
            else:
                bias_loss = zero

            # Zero-drift prior (L1 + L2) penalizes both linear pinball skew pull and large offsets
            if self.lambda_drift_prior != 0.0:
                norm_mean = effective_weekly_median.mean() / scale_w
                drift_prior_loss = norm_mean.abs() + norm_mean.pow(2)
            else:
                drift_prior_loss = zero

            # Optional legacy heuristic components computed only when explicitly enabled
            if (
                self.lambda_dispersion != 0.0
                or self.lambda_magnitude != 0.0
                or self.lambda_naive != 0.0
            ):
                scale_losses = _weekly_scale_losses(
                    pred_weekly_median,
                    actual_weekly,
                    eps=eps,
                    raw_pred_weekly_median=raw_pred_weekly_median,
                )
                dispersion_loss = scale_losses["dispersion_loss"]
                magnitude_loss = scale_losses["magnitude_loss"]
                naive_relative_loss = scale_losses["naive_relative_loss"]
            else:
                dispersion_loss = zero
                magnitude_loss = zero
                naive_relative_loss = zero

            positive_rate_loss = (
                _weekly_positive_rate_loss(
                    effective_weekly_median,
                    actual_weekly,
                    eps=eps,
                )
                if self.lambda_positive_rate != 0.0
                else zero
            )
            interval_loss = (
                _weekly_interval_undercoverage_loss(
                    pred_weekly_quantiles,
                    actual_weekly,
                    self.quantiles,
                    eps=eps,
                )
                if self.lambda_interval != 0.0
                else zero
            )

            def _to_scalar(x: torch.Tensor) -> torch.Tensor:
                return x.mean() if x.ndim > 0 else x

            weekly_q_loss = _to_scalar(weekly_q_loss)
            t1_q_loss = _to_scalar(t1_q_loss)
            t1_directional_loss = _to_scalar(t1_directional_loss)
            dispersion_loss = _to_scalar(dispersion_loss)
            magnitude_loss = _to_scalar(magnitude_loss)
            naive_relative_loss = _to_scalar(naive_relative_loss)
            bias_loss = _to_scalar(bias_loss)
            drift_prior_loss = _to_scalar(drift_prior_loss)
            saturation_loss = _to_scalar(saturation_loss)
            positive_rate_loss = _to_scalar(positive_rate_loss)
            interval_loss = _to_scalar(interval_loss)
            directional_loss = _to_scalar(directional_loss)

            total_loss = (
                self.lambda_weekly_quantile * weekly_q_loss
                + self.lambda_t1_quantile * t1_q_loss
                + self.lambda_t1_directional * t1_directional_loss
                + self.lambda_dispersion * dispersion_loss
                + self.lambda_magnitude * magnitude_loss
                + self.lambda_naive * naive_relative_loss
                + self.lambda_bias * bias_loss
                + self.lambda_drift_prior * drift_prior_loss
                + self.lambda_saturation * saturation_loss
                + self.lambda_positive_rate * positive_rate_loss
                + self.lambda_interval * interval_loss
                + self.lambda_directional * directional_loss
            )

            self._record_components(
                weekly_q_loss,
                t1_q_loss,
                t1_directional_loss,
                dispersion_loss,
                magnitude_loss,
                naive_relative_loss,
                bias_loss,
                drift_prior_loss,
                saturation_loss,
                positive_rate_loss,
                interval_loss,
                directional_loss,
                total_loss,
            )
            return total_loss

except ImportError:
    ASROPFLoss = None  # type: ignore[assignment,misc]
    WeeklyASROPFLoss = None  # type: ignore[assignment,misc]


def _initialize_tft_output_layer(model, quantiles: Sequence[float]) -> None:
    """Initialize output layer weights and quantile step biases to empirical return scale."""
    output_layer = getattr(model, "output_layer", None)
    if output_layer is None:
        return
    layers = (
        list(output_layer)
        if isinstance(output_layer, torch.nn.ModuleList)
        else [output_layer]
    )
    q_dim = len(quantiles)
    median_idx = q_dim // 2
    # RSS-calibrated initial step biases corresponding to empirical weekly spread
    # (~+/-2.0% PI50, ~+/-3.8% PI80, ~+/-6.5% PI96 across sqrt(5) trading days).
    default_biases = (
        (2.29, 1.83, 1.94, 0.0, 1.94, 1.83, 2.29)
        if q_dim == 7
        else tuple(0.0 if idx == median_idx else 1.85 for idx in range(q_dim))
    )
    with torch.no_grad():
        for layer in layers:
            weight = getattr(layer, "weight", None)
            bias = getattr(layer, "bias", None)
            if weight is None or weight.ndim < 2 or weight.shape[0] != q_dim:
                continue
            for idx in range(q_dim):
                weight[idx].mul_(0.02 if idx == median_idx else 0.10)
            if bias is not None and bias.shape[0] == q_dim:
                for idx, val in enumerate(default_biases):
                    bias[idx].fill_(val)


def create_tft_model(
    training_dataset,
    cfg: Optional[TFTASROConfig] = None,
    use_asro: bool = True,
    weekly_target_scale: Optional[float] = None,
):
    """
    Instantiate a TFT model from a training dataset and config.

    Args:
        training_dataset: pytorch_forecasting.TimeSeriesDataSet
        cfg: TFT-ASRO configuration
        use_asro: if True, use ASRO loss; otherwise standard QuantileLoss.

    Returns:
        TemporalFusionTransformer instance
    """
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    if cfg is None:
        cfg = get_tft_config()

    quantiles = list(cfg.model.quantiles)

    if use_asro and cfg.forecast.primary_horizon_days == 5 and WeeklyASROPFLoss is not None:
        loss = WeeklyASROPFLoss(
            quantiles=quantiles,
            lambda_weekly_quantile=cfg.weekly_loss.lambda_weekly_quantile,
            lambda_t1_quantile=cfg.weekly_loss.lambda_t1_quantile,
            lambda_t1_directional=cfg.weekly_loss.lambda_t1_directional,
            lambda_dispersion=cfg.weekly_loss.lambda_dispersion,
            lambda_magnitude=cfg.weekly_loss.lambda_magnitude,
            lambda_naive=cfg.weekly_loss.lambda_naive,
            lambda_bias=cfg.weekly_loss.lambda_bias,
            lambda_drift_prior=cfg.weekly_loss.lambda_drift_prior,
            lambda_directional=cfg.weekly_loss.lambda_directional,
            lambda_saturation=cfg.weekly_loss.lambda_saturation,
            lambda_positive_rate=cfg.weekly_loss.lambda_positive_rate,
            lambda_interval=cfg.weekly_loss.lambda_interval,
            weekly_median_cap=cfg.weekly_loss.weekly_median_cap,
            weekly_target_scale=weekly_target_scale,
        )
        logger.info(
            "Using weekly ASRO loss | weekly_q=%.2f t1_q=%.2f t1_dir=%.2f dispersion=%.2f "
            "magnitude=%.2f naive=%.2f bias=%.2f dir=%.2f saturation=%.2f "
            "positive_rate=%.2f interval=%.2f "
            "median_cap=%s "
            "monotonic_transform=true gap_scale=%.3f",
            cfg.weekly_loss.lambda_weekly_quantile,
            cfg.weekly_loss.lambda_t1_quantile,
            cfg.weekly_loss.lambda_t1_directional,
            cfg.weekly_loss.lambda_dispersion,
            cfg.weekly_loss.lambda_magnitude,
            cfg.weekly_loss.lambda_naive,
            cfg.weekly_loss.lambda_bias,
            cfg.weekly_loss.lambda_directional,
            cfg.weekly_loss.lambda_saturation,
            cfg.weekly_loss.lambda_positive_rate,
            cfg.weekly_loss.lambda_interval,
            (
                f"{cfg.weekly_loss.weekly_median_cap:.6f}"
                if cfg.weekly_loss.weekly_median_cap is not None
                else "none"
            ),
            DEFAULT_MONOTONIC_GAP_SCALE,
        )
    elif use_asro and ASROPFLoss is not None:
        loss = ASROPFLoss(
            quantiles=quantiles,
            lambda_vol=cfg.asro.lambda_vol,
            lambda_quantile=cfg.asro.lambda_quantile,
            lambda_madl=cfg.asro.lambda_madl,
            lambda_crossing=cfg.asro.lambda_crossing,
            risk_free_rate=cfg.asro.risk_free_rate,
        )
        logger.info(
            "Using ASRO loss | w_quantile=%.2f w_sharpe=%.2f lambda_vol=%.2f lambda_crossing=%.2f",
            cfg.asro.lambda_quantile,
            1.0 - cfg.asro.lambda_quantile,
            cfg.asro.lambda_vol,
            cfg.asro.lambda_crossing,
        )
    else:
        loss = QuantileLoss(quantiles=quantiles)
        logger.info("Using standard QuantileLoss with %d quantiles", len(quantiles))

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=cfg.model.learning_rate,
        hidden_size=cfg.model.hidden_size,
        attention_head_size=cfg.model.attention_head_size,
        dropout=cfg.model.dropout,
        hidden_continuous_size=cfg.model.hidden_continuous_size,
        output_size=len(quantiles),
        loss=loss,
        optimizer="adamw",
        reduce_on_plateau_patience=cfg.model.reduce_on_plateau_patience,
        log_interval=10,
        log_val_interval=1,
    )
    _initialize_tft_output_layer(model, quantiles)

    # Configure AdamW with decoupled output-head learning rates so feature weights
    # can learn conditional return sensitivity while output biases stay conservative
    # (preventing unconditional positive/negative drift collapse).
    _base_lr = float(cfg.model.learning_rate)
    _head_weight_lr = _base_lr * float(cfg.model.output_head_lr_mult)
    _head_bias_lr = _base_lr * min(0.05, 0.25 * float(cfg.model.output_head_lr_mult))
    _weight_decay = float(cfg.model.weight_decay)
    _lr_patience = int(cfg.model.reduce_on_plateau_patience)

    def _differential_configure_optimizers(m=model):
        head_weight_params = []
        head_bias_params = []
        body_params = []
        for name, param in m.named_parameters():
            if not param.requires_grad:
                continue
            if "output_layer" in name:
                if name.endswith("bias"):
                    head_bias_params.append(param)
                else:
                    head_weight_params.append(param)
            else:
                body_params.append(param)
        param_groups = [{"params": body_params, "lr": _base_lr, "weight_decay": _weight_decay}]
        if head_weight_params:
            param_groups.append(
                {"params": head_weight_params, "lr": _head_weight_lr, "weight_decay": 0.0}
            )
        if head_bias_params:
            param_groups.append(
                {"params": head_bias_params, "lr": _head_bias_lr, "weight_decay": 0.0}
            )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=_lr_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    model.configure_optimizers = _differential_configure_optimizers
    logger.info(
        "Configured AdamW | body_lr=%.1e head_weight_lr=%.1e head_bias_lr=%.1e weight_decay=%.1e",
        _base_lr,
        _head_weight_lr,
        _head_bias_lr,
        _weight_decay,
    )

    model.save_hyperparameters(ignore=['loss', 'logging_metrics'])

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("TFT model created: %d total params, %d trainable", n_params, n_trainable)

    return model


def load_tft_model(
    checkpoint_path: str,
    map_location: str = "cpu",
):
    """Load a trained TFT model from a Lightning checkpoint."""
    from pytorch_forecasting import TemporalFusionTransformer

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = TemporalFusionTransformer.load_from_checkpoint(str(path), map_location=map_location)
    model.eval()
    logger.info("Loaded TFT model from %s", path)
    return model


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

def _named_weights(names, weights) -> list[Dict[str, Any]]:
    """Zip variable names with a per-sample weight vector, validating the layout."""
    if len(names) != len(weights):
        raise ValueError(
            f"interpretation names/weights mismatch: got {len(names)} names "
            f"for {len(weights)} weights"
        )
    return [
        {"feature": name, "weight": float(weight)}
        for name, weight in zip(names, weights)
    ]


def _run_interpretation(model, x: dict, attention_prediction_horizon: int):
    """Evaluate ``model`` on one batch and return ``(out, per-sample interpretation)``.

    Placement and mode safety: tensors are moved to the model's device (an
    inference model may legitimately live on GPU while the dataloader yields
    CPU tensors — the same registration pytorch-forecasting's ``predict``
    performs internally), eval mode is forced so dropout can never add noise
    to the weights, and the original training mode is restored afterwards.
    """
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = None
    moved = {
        key: value.to(device) if (device is not None and torch.is_tensor(value)) else value
        for key, value in x.items()
    }
    was_training = bool(getattr(model, "training", False))
    if was_training and hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            out = model(moved)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    interpretation = model.interpret_output(
        out,
        reduction="none",
        attention_prediction_horizon=attention_prediction_horizon,
    )
    return out, interpretation


def interpret_batch(model, x: dict, attention_prediction_horizon: int = 0) -> dict:
    """
    Interpret a single TFT batch: VSN selection weights plus encoder attention.

    Runs the raw forward pass under ``torch.no_grad()`` (batch moved to the
    model's device, eval mode enforced so dropout adds no noise) and reduces
    nothing (``reduction="none"``), so every score stays per-sample; attention
    is renormalized per sample and clipped to the encoder window
    (``[: model.hparams.max_encoder_length]``). Group names come from
    ``model.encoder_variables`` / ``decoder_variables`` / ``static_variables``;
    a names/weights length mismatch raises ``ValueError``.
    """
    out, interpretation = _run_interpretation(model, x, attention_prediction_horizon)

    max_encoder_length = int(model.hparams.max_encoder_length)
    attention = (
        interpretation["attention"][0].detach().cpu().numpy()[:max_encoder_length].tolist()
    )

    return {
        "encoder": _named_weights(
            model.encoder_variables,
            interpretation["encoder_variables"][0].detach().cpu().numpy(),
        ),
        "decoder": _named_weights(
            model.decoder_variables,
            interpretation["decoder_variables"][0].detach().cpu().numpy(),
        ),
        "static": _named_weights(
            model.static_variables,
            interpretation["static_variables"][0].detach().cpu().numpy(),
        ),
        "attention": attention,
        "encoder_length": int(out["encoder_lengths"][0]),
    }


def get_variable_importance(model, val_dataloader=None) -> Dict[str, float]:
    """
    Mean encoder VSN variable importance over every validation sample.

    ``interpret_output(reduction="none")`` returns per-sample selection weights
    (shape ``(batch, num_variables)``); those are pooled over *all samples of
    every batch*, per feature. Non-finite weights are dropped per sample so a
    single NaN/Inf batch cannot poison the whole map (an exploding checkpoint's
    softmax weights may be NaN). The result is normalised to sum 1 and sorted in
    descending order; an absent or empty loader yields ``{}``, and any failure
    while extracting/averaging importance also yields ``{}``. This is
    best-effort promotion metadata and must never raise.
    ``val_dataloader`` must be passed explicitly (``model.val_dataloader()``
    only works inside a Lightning Trainer context and raises an error otherwise).
    """
    if val_dataloader is None:
        return {}

    try:
        weight_sums: Dict[str, float] = {}
        sample_counts: Dict[str, int] = {}
        n_samples = 0
        for batch in val_dataloader:
            sample = batch[0] if isinstance(batch, (tuple, list)) else batch
            _, interpretation = _run_interpretation(model, sample, 0)
            per_sample = interpretation["encoder_variables"].detach().cpu().numpy()
            names = model.encoder_variables
            if per_sample.ndim != 2 or len(names) != per_sample.shape[1]:
                raise ValueError(
                    f"interpretation names/weights mismatch: got {len(names)} names "
                    f"for encoder variables of shape {per_sample.shape!r}"
                )
            for index, name in enumerate(names):
                column = per_sample[:, index]
                finite = column[np.isfinite(column)]
                if finite.size == 0:
                    continue
                weight_sums[name] = weight_sums.get(name, 0.0) + float(finite.sum())
                sample_counts[name] = sample_counts.get(name, 0) + int(finite.size)
            n_samples += int(per_sample.shape[0])

        if n_samples == 0 or not weight_sums:
            return {}

        averaged: Dict[str, float] = {}
        for name, total in weight_sums.items():
            count = sample_counts.get(name, 0)
            if count <= 0:
                continue
            average = total / count
            if math.isfinite(average):
                averaged[name] = average
        if not averaged:
            return {}

        total_weight = sum(averaged.values())
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            return {}

        normalized = {name: weight / total_weight for name, weight in averaged.items()}
        return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))
    except Exception:
        logger.warning(
            "Variable importance extraction failed; returning empty importance",
            exc_info=True,
        )
        return {}


# ---------------------------------------------------------------------------
# Prediction formatting
# ---------------------------------------------------------------------------

def _format_prediction_legacy_simple_return(
    raw_prediction: torch.Tensor,
    quantiles: Sequence[float] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    baseline_price: float = 1.0,
    reference_price_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert raw TFT quantile output to a structured prediction dict.

    The model emits next-day *simple returns* in return-space (target was
    ``close.pct_change().shift(-1)``). We therefore treat every quantile as a
    daily return and compound to price using ``baseline_price`` as the
    reference. The returned dict is the single source of truth: both the
    headline percentage (``predicted_return_median``) and the T+1 price
    (``daily_forecasts[0].price_median``) are derived from the same value.

    Hard return clamps used to be set to 3% which was the root cause of the
    "stuck at 3%" display bug: any time the raw median exceeded 3% it was
    silently snapped to exactly 3% and the UI rendered the clamp value. We
    now use a calibrated sanity-check anomaly bound (``ANOMALY_DAILY_RET``)
    which only triggers on genuinely implausible moves (> ~5× copper's daily
    σ) and logs loudly when it does. Within a reasonable range the raw
    model output is passed through untouched.
    """
    import math as _math

    pred = raw_prediction.detach().cpu().numpy() if isinstance(raw_prediction, torch.Tensor) else raw_prediction
    pred = np.array(pred, dtype=np.float64, copy=True)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    n_days = pred.shape[0]
    median_idx = len(quantiles) // 2
    raw_pred = pred.copy()

    quantile_diffs = np.diff(raw_pred, axis=-1) if raw_pred.shape[-1] > 1 else np.array([])
    crossing_mask = quantile_diffs < -1e-12 if quantile_diffs.size else np.array([], dtype=bool)
    quantile_crossing_detected = bool(crossing_mask.any())
    raw_quantile_crossing_rate = float(crossing_mask.mean()) if crossing_mask.size else 0.0
    ordered_tensor = enforce_monotonic_quantiles(
        torch.as_tensor(raw_pred, dtype=torch.float64),
        median_idx=median_idx,
        min_gap=1e-5,
        gap_scale=DEFAULT_MONOTONIC_GAP_SCALE,
        init_bias=-3.0,
    )
    pred = ordered_tensor.detach().cpu().numpy()
    quantile_crossing_rate = 0.0
    sorted_pred = pred
    median_sort_gap = float(
        np.max(np.abs(raw_pred[..., median_idx] - sorted_pred[..., median_idx]))
    )
    if quantile_crossing_detected:
        logger.info(
            "format_prediction: non-monotonic quantiles detected "
            "(raw_crossing_rate=%.3f, max_median_sort_gap=%.4f); public output "
            "uses the structural monotonic transform and exposes raw_quantiles for audit.",
            raw_quantile_crossing_rate,
            median_sort_gap,
        )

    if _math.isnan(baseline_price) or _math.isinf(baseline_price) or baseline_price <= 0:
        logger.warning(
            "format_prediction: invalid baseline_price=%s — price fields will be null",
            baseline_price,
        )

    # ------------------------------------------------------------------
    # Calibrated anomaly bound (NOT a "display cap").
    # Copper daily σ ≈ 0.024 (2.4%). A 5σ daily move (~12%) is a genuine
    # regime-break event; if the model outputs that, it is almost
    # certainly a bug in preprocessing / scaling, not a real forecast.
    # Below this level we trust the model output as-is.
    # ------------------------------------------------------------------
    ANOMALY_DAILY_RET = 0.12
    raw_median_0 = float(raw_pred[0, median_idx])
    corrected_median_0 = float(pred[0, median_idx])
    anomaly_detected = (
        abs(raw_median_0) > ANOMALY_DAILY_RET
        or abs(corrected_median_0) > ANOMALY_DAILY_RET
        or quantile_crossing_detected
    )
    if abs(raw_median_0) > ANOMALY_DAILY_RET or abs(corrected_median_0) > ANOMALY_DAILY_RET:
        logger.info(
            "format_prediction: anomalous return detected at T+1: raw=%.4f corrected=%.4f "
            "(|r| > %.3f). Likely a scaling / target-space bug; the value "
            "will be bounded at +/-%.2f and flagged in the response.",
            raw_median_0, corrected_median_0, ANOMALY_DAILY_RET, ANOMALY_DAILY_RET,
        )

    def _bound(x: float) -> float:
        """Only clip if outside the anomaly bound; otherwise pass through."""
        if abs(x) > ANOMALY_DAILY_RET:
            return float(np.sign(x) * ANOMALY_DAILY_RET)
        return float(x)

    # Quantile spreads (distance of each quantile from the median, in
    # return-space). We do NOT clip *spreads* — models with a healthy
    # variance ratio produce spreads of ~2σ and that is fine.
    raw_med_0 = corrected_median_0
    spread_q10 = float(pred[0, 1]) - raw_med_0 if len(quantiles) > 2 else 0.0
    spread_q90 = float(pred[0, -2]) - raw_med_0 if len(quantiles) > 2 else 0.0
    spread_q02 = float(pred[0, 0]) - raw_med_0
    spread_q98 = float(pred[0, -1]) - raw_med_0

    daily_forecasts = []
    cum_price_med = baseline_price

    for d in range(n_days):
        raw_med = float(raw_pred[d, median_idx])
        corrected_med = float(pred[d, median_idx])
        med = _bound(corrected_med)
        cum_price_med *= (1 + med)
        cum_return = (cum_price_med / baseline_price) - 1.0

        # √t spread expansion keeps multi-day uncertainty realistic instead
        # of exponentially compounding tail quantiles.
        scale = (d + 1) ** 0.5

        daily_forecasts.append({
            "day": d + 1,
            "daily_return": med,
            "raw_daily_return": raw_med,
            "corrected_daily_return": corrected_med,
            "cumulative_return": cum_return,
            "price_median": cum_price_med,
            "price_q10": cum_price_med * (1 + spread_q10 * scale),
            "price_q90": cum_price_med * (1 + spread_q90 * scale),
            "price_q02": cum_price_med * (1 + spread_q02 * scale),
            "price_q98": cum_price_med * (1 + spread_q98 * scale),
        })

    first = daily_forecasts[0]
    last = daily_forecasts[-1]
    vol_estimate = (first["price_q90"] - first["price_q10"]) / (2.0 * baseline_price)

    return {
        # Single source of truth for the UI — headline percentage and T+1
        # price are now derived from the *same* value.
        "predicted_return_median": first["daily_return"],
        "predicted_return_q10": float(pred[0, 1]) if len(quantiles) > 2 else first["daily_return"],
        "predicted_return_q90": float(pred[0, -2]) if len(quantiles) > 2 else first["daily_return"],
        "predicted_price_median": first["price_median"],
        "predicted_price_q10": first["price_q10"],
        "predicted_price_q90": first["price_q90"],
        "confidence_band_96": (first["price_q02"], first["price_q98"]),
        "volatility_estimate": vol_estimate,
        "quantiles": {f"q{q:.2f}": float(pred[0, i]) for i, q in enumerate(quantiles)},
        "raw_quantiles": {f"q{q:.2f}": float(raw_pred[0, i]) for i, q in enumerate(quantiles)},
        "quantile_crossing_detected": quantile_crossing_detected,
        "quantile_crossing_rate": quantile_crossing_rate,
        "median_sort_gap": median_sort_gap,
        "weekly_return": last["cumulative_return"],
        "weekly_price": last["price_median"],
        "prediction_horizon_days": n_days,
        "daily_forecasts": daily_forecasts,
        # Explicit contract for the frontend — no more guessing which price
        # the percentages are relative to.
        "reference_price": float(baseline_price),
        "reference_price_date": reference_price_date,
        "return_basis": "simple_next_day_return",
        "raw_predicted_return_median": raw_median_0,
        "anomaly_detected": bool(anomaly_detected),
    }


def format_prediction(
    raw_prediction: torch.Tensor,
    quantiles: Sequence[float] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    baseline_price: float = 1.0,
    reference_price_date: Optional[str] = None,
    conformal_adjustment: float = 0.0,
    weekly_cqr_params: Optional[Dict[str, Any]] = None,
    realized_vol: Optional[float] = None,
    is_monotonic_logits: bool = False,
) -> Dict[str, Any]:
    """
    Convert daily log-return TFT quantiles into public simple-return fields.

    Internal model space is a 5-step daily log-return path. Public return
    fields are simple returns, and prices are derived with
    ``baseline_price * exp(cumulative_log_return)``. When
    ``is_monotonic_logits`` is True, ``raw_prediction`` holds unconstrained
    delta-step logits parameterized for ``enforce_monotonic_quantiles`` rather
    than directly ordered quantiles.
    """
    import math as _math

    pred = raw_prediction.detach().cpu().numpy() if isinstance(raw_prediction, torch.Tensor) else raw_prediction
    pred = np.array(pred, dtype=np.float64, copy=True)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)

    n_days = pred.shape[0]
    q_list = list(quantiles)
    median_idx = len(q_list) // 2
    q10_idx = q_list.index(0.10) if 0.10 in q_list else 1
    q90_idx = q_list.index(0.90) if 0.90 in q_list else len(q_list) - 2
    q02_idx = q_list.index(0.02) if 0.02 in q_list else 0
    q98_idx = q_list.index(0.98) if 0.98 in q_list else len(q_list) - 1
    raw_pred = pred.copy()

    ordered_tensor = enforce_monotonic_quantiles(
        torch.as_tensor(raw_pred, dtype=torch.float64),
        median_idx=median_idx,
        min_gap=1e-5,
        gap_scale=DEFAULT_MONOTONIC_GAP_SCALE,
        init_bias=-3.0,
    )
    pred = ordered_tensor.detach().cpu().numpy()
    ordered_diffs = np.diff(pred, axis=-1) if pred.shape[-1] > 1 else np.array([])
    ordered_crossing_mask = (
        ordered_diffs < -1e-12 if ordered_diffs.size else np.array([], dtype=bool)
    )
    ordered_quantile_crossing_rate = (
        float(ordered_crossing_mask.mean()) if ordered_crossing_mask.size else 0.0
    )
    if ordered_quantile_crossing_rate > 0.0:
        raise AssertionError(
            "Monotonic quantile transform produced public crossings: "
            f"{ordered_quantile_crossing_rate:.6f}"
        )

    if is_monotonic_logits:
        quantile_crossing_detected = bool(ordered_crossing_mask.any())
        raw_quantile_crossing_rate = 0.0
        raw_median_sort_gap = 0.0
    else:
        quantile_diffs = np.diff(raw_pred, axis=-1) if raw_pred.shape[-1] > 1 else np.array([])
        crossing_mask = quantile_diffs < -1e-12 if quantile_diffs.size else np.array([], dtype=bool)
        quantile_crossing_detected = bool(crossing_mask.any())
        raw_quantile_crossing_rate = float(crossing_mask.mean()) if crossing_mask.size else 0.0
        sorted_raw = np.sort(raw_pred, axis=-1)
        raw_median_sort_gap = float(
            np.max(np.abs(raw_pred[..., median_idx] - sorted_raw[..., median_idx]))
        )

    ordered_median_sort_gap = float(
        np.max(np.abs(pred[..., median_idx] - np.sort(pred, axis=-1)[..., median_idx]))
    )
    if quantile_crossing_detected:
        logger.info(
            "format_prediction: non-monotonic quantiles detected "
            "(raw_crossing_rate=%.3f, raw_max_median_sort_gap=%.4f); public output "
            "uses the structural monotonic transform and exposes raw_quantiles for audit.",
            raw_quantile_crossing_rate,
            raw_median_sort_gap,
        )

    if _math.isnan(baseline_price) or _math.isinf(baseline_price) or baseline_price <= 0:
        logger.warning(
            "format_prediction: invalid baseline_price=%s; price fields will be null",
            baseline_price,
        )

    anomaly_bound = 0.12
    raw_median_0 = float(raw_pred[0, median_idx])
    corrected_median_0 = float(pred[0, median_idx])
    anomaly_detected = (
        abs(raw_median_0) > anomaly_bound
        or abs(corrected_median_0) > anomaly_bound
        or quantile_crossing_detected
    )
    if abs(raw_median_0) > anomaly_bound or abs(corrected_median_0) > anomaly_bound:
        logger.info(
            "format_prediction: anomalous log return detected at T+1: raw=%.4f corrected=%.4f",
            raw_median_0,
            corrected_median_0,
        )

    bounded_pred = pred.copy()
    bounded_pred[..., median_idx] = np.clip(
        bounded_pred[..., median_idx],
        -anomaly_bound,
        anomaly_bound,
    )

    def _valid_price_base() -> bool:
        return bool(baseline_price > 0 and np.isfinite(baseline_price))

    def _price(cum_log_return: float) -> Optional[float]:
        if not _valid_price_base():
            return None
        return float(baseline_price * np.exp(cum_log_return))

    cum_quantiles_rss = aggregate_cumulative_quantiles_np(bounded_pred, median_idx=median_idx)
    daily_forecasts = []
    cum_log_median = 0.0
    forecast_dates: list[Optional[str]] = []
    if reference_price_date:
        try:
            base_date = pd.Timestamp(reference_price_date)
            forecast_dates = [
                value.date().isoformat()
                for value in pd.bdate_range(base_date + pd.offsets.BDay(1), periods=n_days)
            ]
        except Exception:
            forecast_dates = []
    for d in range(n_days):
        raw_med = float(raw_pred[d, median_idx])
        daily_log = float(bounded_pred[d, median_idx])
        cum_log_median += daily_log
        cum_q = cum_quantiles_rss[d]

        daily_forecasts.append(
            {
                "day": d + 1,
                "forecast_date": forecast_dates[d] if d < len(forecast_dates) else None,
                "daily_return": log_to_simple_return(daily_log),
                "daily_log_return": daily_log,
                "raw_daily_log_return": raw_med,
                "corrected_daily_log_return": daily_log,
                "cumulative_return": log_to_simple_return(cum_log_median),
                "cumulative_log_return": float(cum_log_median),
                "price_median": _price(cum_log_median),
                "price_q10": _price(float(cum_q[q10_idx])),
                "price_q90": _price(float(cum_q[q90_idx])),
                "price_q02": _price(float(cum_q[q02_idx])),
                "price_q98": _price(float(cum_q[q98_idx])),
            }
        )

    first = daily_forecasts[0]
    weekly_quantile_logs = aggregate_weekly_quantiles_np(bounded_pred, median_idx=median_idx)
    weekly_log_return = float(weekly_quantile_logs[median_idx])
    weekly_q10_log_raw = float(weekly_quantile_logs[q10_idx])
    weekly_q90_log_raw = float(weekly_quantile_logs[q90_idx])
    if weekly_cqr_params and bool(weekly_cqr_params.get("cqr_enabled", False)):
        from deep_learning.calibration.conformal import apply_weekly_cqr

        weekly_calibrated_logs = apply_weekly_cqr(
            weekly_quantile_logs,
            quantiles=tuple(q_list),
            cqr_params=weekly_cqr_params,
            realized_vol=realized_vol,
        )
        weekly_q10_log_cal = float(weekly_calibrated_logs[q10_idx])
        weekly_q90_log_cal = float(weekly_calibrated_logs[q90_idx])
        weekly_q02_log_cal = float(weekly_calibrated_logs[q02_idx])
        weekly_q98_log_cal = float(weekly_calibrated_logs[q98_idx])
        conformal_adjustment = float(
            max(
                0.0,
                max(weekly_q10_log_raw - weekly_q10_log_cal, weekly_q90_log_cal - weekly_q90_log_raw),
            )
        )
        interval_calibrated = True
        if daily_forecasts:
            daily_forecasts[-1]["price_q10"] = _price(weekly_q10_log_cal)
            daily_forecasts[-1]["price_q90"] = _price(weekly_q90_log_cal)
            daily_forecasts[-1]["price_q02"] = _price(weekly_q02_log_cal)
            daily_forecasts[-1]["price_q98"] = _price(weekly_q98_log_cal)
    else:
        conformal_adjustment = max(float(conformal_adjustment or 0.0), 0.0)
        weekly_q10_log_cal = weekly_q10_log_raw - conformal_adjustment
        weekly_q90_log_cal = weekly_q90_log_raw + conformal_adjustment
        interval_calibrated = conformal_adjustment > 0

    q10_price = first.get("price_q10")
    q90_price = first.get("price_q90")
    vol_estimate = (
        float((q90_price - q10_price) / (2.0 * baseline_price))
        if q10_price is not None and q90_price is not None and _valid_price_base()
        else 0.0
    )

    return {
        "predicted_return_median": first["daily_return"],
        "predicted_return_q10": log_to_simple_return(float(bounded_pred[0, q10_idx])),
        "predicted_return_q90": log_to_simple_return(float(bounded_pred[0, q90_idx])),
        "predicted_log_return_median": first["daily_log_return"],
        "predicted_log_return_q10": float(bounded_pred[0, q10_idx]),
        "predicted_log_return_q90": float(bounded_pred[0, q90_idx]),
        "predicted_price_median": first["price_median"],
        "predicted_price_q10": first["price_q10"],
        "predicted_price_q90": first["price_q90"],
        "confidence_band_96": (first["price_q02"], first["price_q98"]),
        "volatility_estimate": vol_estimate,
        "quantiles": {f"q{q:.2f}": log_to_simple_return(float(bounded_pred[0, i])) for i, q in enumerate(q_list)},
        "quantiles_log": {f"q{q:.2f}": float(bounded_pred[0, i]) for i, q in enumerate(q_list)},
        "raw_quantiles": {f"q{q:.2f}": float(raw_pred[0, i]) for i, q in enumerate(q_list)},
        "quantile_crossing_detected": quantile_crossing_detected,
        "quantile_crossing_rate": ordered_quantile_crossing_rate,
        "raw_quantile_crossing_rate": raw_quantile_crossing_rate,
        "ordered_quantile_crossing_rate": ordered_quantile_crossing_rate,
        "public_quantile_crossing_rate": ordered_quantile_crossing_rate,
        "median_sort_gap": ordered_median_sort_gap,
        "raw_median_sort_gap": raw_median_sort_gap,
        "ordered_median_sort_gap": ordered_median_sort_gap,
        "weekly_return": log_to_simple_return(weekly_log_return),
        "weekly_log_return": weekly_log_return,
        "weekly_price": _price(weekly_log_return),
        "weekly_return_q10_raw": log_to_simple_return(weekly_q10_log_raw),
        "weekly_return_q90_raw": log_to_simple_return(weekly_q90_log_raw),
        "weekly_log_return_q10_raw": weekly_q10_log_raw,
        "weekly_log_return_q90_raw": weekly_q90_log_raw,
        "weekly_return_q10_calibrated": log_to_simple_return(weekly_q10_log_cal),
        "weekly_return_q90_calibrated": log_to_simple_return(weekly_q90_log_cal),
        "weekly_log_return_q10_calibrated": weekly_q10_log_cal,
        "weekly_log_return_q90_calibrated": weekly_q90_log_cal,
        "weekly_interval_calibration_adjustment": conformal_adjustment,
        "weekly_interval_calibrated": bool(interval_calibrated),
        "prediction_horizon_days": n_days,
        "daily_forecasts": daily_forecasts,
        "reference_price": float(baseline_price) if _valid_price_base() else None,
        "reference_price_date": reference_price_date,
        "return_basis": "daily_log_return_path",
        "return_space": RETURN_SPACE,
        "raw_predicted_return_median": log_to_simple_return(raw_median_0),
        "raw_predicted_log_return_median": raw_median_0,
        "anomaly_detected": bool(anomaly_detected),
    }
