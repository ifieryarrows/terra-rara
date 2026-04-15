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
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import numpy as np

from deep_learning.config import TFTASROConfig, get_tft_config
from deep_learning.models.losses import AdaptiveSharpeRatioLoss, CombinedQuantileLoss

logger = logging.getLogger(__name__)

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
            risk_free_rate: float = 0.0,
            sharpe_eps: float = 1e-6,
        ):
            super().__init__(quantiles=quantiles)
            self.lambda_vol = lambda_vol
            self.lambda_quantile = lambda_quantile
            self.lambda_madl = lambda_madl
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
            amplitude_loss = (
                torch.relu(1.0 - vr)              # under-variance: VR < 1 → strong penalty
                + 1.0 * torch.relu(vr - 1.5)      # over-variance:  VR > 1.5 → symmetric penalty
            )

            # Quantile (pinball) loss via parent — covers all 7 quantile bands
            q_loss = super().loss(y_pred, target)

            # MADL: direct directional accuracy via magnitude-weighted sign match
            soft_sign_madl = torch.tanh(median_pred * 20.0)
            direction_match = soft_sign_madl * y_actual.float()
            madl_loss = (-direction_match * y_actual.float().abs()).mean()

            w_directional = 1.0 - self.lambda_quantile
            calibration = q_loss + self.lambda_vol * (vol_loss + amplitude_loss)
            directional = sharpe_loss + self.lambda_madl * madl_loss
            return self.lambda_quantile * calibration + w_directional * directional

except ImportError:
    ASROPFLoss = None  # type: ignore[assignment,misc]


def create_tft_model(
    training_dataset,
    cfg: Optional[TFTASROConfig] = None,
    use_asro: bool = True,
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

    if use_asro and ASROPFLoss is not None:
        loss = ASROPFLoss(
            quantiles=quantiles,
            lambda_vol=cfg.asro.lambda_vol,
            lambda_quantile=cfg.asro.lambda_quantile,
            lambda_madl=cfg.asro.lambda_madl,
            risk_free_rate=cfg.asro.risk_free_rate,
        )
        logger.info(
            "Using ASRO loss | w_quantile=%.2f w_sharpe=%.2f lambda_vol=%.2f",
            cfg.asro.lambda_quantile,
            1.0 - cfg.asro.lambda_quantile,
            cfg.asro.lambda_vol,
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
        reduce_on_plateau_patience=cfg.model.reduce_on_plateau_patience,
        optimizer_kwargs={"weight_decay": cfg.model.weight_decay},
        log_interval=10,
        log_val_interval=1,
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

def get_variable_importance(model, val_dataloader=None) -> Dict[str, float]:
    """
    Extract learned variable importance from the TFT's Variable Selection Networks.

    Returns a dict mapping feature name -> normalised importance score.
    val_dataloader must be passed explicitly (model.val_dataloader() only works
    inside a Lightning Trainer context and raises an error otherwise).
    """
    if val_dataloader is None:
        return {}
    try:
        interpretation = model.interpret_output(
            model.predict(val_dataloader, return_x=True),
            reduction="sum",
        )
        importance = interpretation.get("encoder_variables", {})
        if not importance:
            return {}

        total = sum(importance.values())
        if total == 0:
            return importance

        return {k: v / total for k, v in sorted(importance.items(), key=lambda x: -x[1])}
    except Exception as exc:
        logger.warning("Could not extract variable importance: %s", exc)
        return {}


def get_attention_weights(model, dataloader) -> Optional[np.ndarray]:
    """
    Extract temporal self-attention weights for interpretability.

    Returns array of shape (n_samples, n_heads, encoder_length, encoder_length)
    or None if extraction fails.
    """
    try:
        out = model.predict(dataloader, return_x=True, mode="raw")
        attn = out.get("attention")
        if attn is not None:
            return attn.cpu().numpy()
    except Exception as exc:
        logger.warning("Could not extract attention weights: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Prediction formatting
# ---------------------------------------------------------------------------

def format_prediction(
    raw_prediction: torch.Tensor,
    quantiles: Sequence[float] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    baseline_price: float = 1.0,
) -> Dict[str, Any]:
    """
    Convert raw TFT quantile output to a structured prediction dict.

    Args:
        raw_prediction: tensor of shape (prediction_length, n_quantiles)
        quantiles: quantile levels
        baseline_price: current price for return-to-price conversion

    Returns:
        Dict with per-day forecasts, confidence bands, and volatility estimate.
        Top-level fields use the *final* day (end of horizon) for backward compat.
    """
    import math as _math

    pred = raw_prediction.cpu().numpy() if isinstance(raw_prediction, torch.Tensor) else raw_prediction
    n_days = pred.shape[0]
    median_idx = len(quantiles) // 2

    # Guard: log if baseline_price is invalid (NaN prices will be sanitised
    # to null by the API layer's _sanitize_floats, keeping the chart clean).
    if _math.isnan(baseline_price) or _math.isinf(baseline_price) or baseline_price <= 0:
        logger.warning(
            "format_prediction: invalid baseline_price=%s — price fields will be null",
            baseline_price,
        )

    # Hard clamp: prevents overconfident models (VR >> 1) from producing
    # absurd compound prices.  Copper's actual daily σ ≈ 0.024; capping at
    # ~1.25σ keeps the 5-day compound under ≈16 %.  The clamp is inactive
    # once the model is retrained with a healthy VR (0.5–1.5).
    _MAX_DAILY_RET = 0.03

    # T+1 quantile spreads (return-space distance from median).
    # Used as the base width for confidence bands; scaled by sqrt(d) for
    # later days so uncertainty grows realistically instead of compounding
    # tail quantiles exponentially (which would produce absurd bands).
    med_0 = float(np.clip(pred[0, median_idx], -_MAX_DAILY_RET, _MAX_DAILY_RET))
    _raw_med_0 = float(pred[0, median_idx])
    spread_q10 = np.clip(float(pred[0, 1]) - _raw_med_0, -_MAX_DAILY_RET, 0) if len(quantiles) > 2 else 0.0
    spread_q90 = np.clip(float(pred[0, -2]) - _raw_med_0, 0, _MAX_DAILY_RET) if len(quantiles) > 2 else 0.0
    spread_q02 = np.clip(float(pred[0, 0]) - _raw_med_0, -_MAX_DAILY_RET * 1.5, 0)
    spread_q98 = np.clip(float(pred[0, -1]) - _raw_med_0, 0, _MAX_DAILY_RET * 1.5)

    daily_forecasts = []
    cum_price_med = baseline_price

    for d in range(n_days):
        med = float(np.clip(pred[d, median_idx], -_MAX_DAILY_RET, _MAX_DAILY_RET))
        cum_price_med *= (1 + med)
        cum_return = (cum_price_med / baseline_price) - 1.0

        scale = (d + 1) ** 0.5

        daily_forecasts.append({
            "day": d + 1,
            "daily_return": med,
            "cumulative_return": cum_return,
            "price_median": cum_price_med,
            "price_q10": cum_price_med * (1 + spread_q10 * scale),
            "price_q90": cum_price_med * (1 + spread_q90 * scale),
            "price_q02": cum_price_med * (1 + spread_q02 * scale),
            "price_q98": cum_price_med * (1 + spread_q98 * scale),
        })

    # T+1 is the primary signal (most reliable, highest signal-to-noise).
    first = daily_forecasts[0]
    last = daily_forecasts[-1]
    vol_estimate = (first["price_q90"] - first["price_q10"]) / (2.0 * baseline_price)

    return {
        "predicted_return_median": first["daily_return"],
        "predicted_return_q10": float(np.clip(pred[0, 1], -_MAX_DAILY_RET * 2, _MAX_DAILY_RET * 2)) if len(quantiles) > 2 else first["daily_return"],
        "predicted_return_q90": float(np.clip(pred[0, -2], -_MAX_DAILY_RET * 2, _MAX_DAILY_RET * 2)) if len(quantiles) > 2 else first["daily_return"],
        "predicted_price_median": first["price_median"],
        "predicted_price_q10": first["price_q10"],
        "predicted_price_q90": first["price_q90"],
        "confidence_band_96": (first["price_q02"], first["price_q98"]),
        "volatility_estimate": vol_estimate,
        "quantiles": {f"q{q:.2f}": float(pred[0, i]) for i, q in enumerate(quantiles)},
        "weekly_return": last["cumulative_return"],
        "weekly_price": last["price_median"],
        "prediction_horizon_days": n_days,
        "daily_forecasts": daily_forecasts,
    }
