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
            under_severe   = 2.0 * torch.relu(0.5 - vr)   # fires hard when VR < 0.5
            under_moderate = torch.relu(1.0 - vr)          # fires when VR < 1.0
            over_variance  = 1.0 * torch.relu(vr - 1.5)
            amplitude_loss = under_severe + under_moderate + over_variance

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
        log_interval=10,
        log_val_interval=1,
    )

    # Apply weight decay post-construction by patching each param group.
    # pytorch_forecasting's TFT does not expose optimizer_kwargs in from_dataset(),
    # so we reach into the already-configured optimizer after the first
    # configure_optimizers call, which Lightning triggers during fit().
    _weight_decay = cfg.model.weight_decay
    if _weight_decay > 0:
        _orig_configure_optimizers = model.configure_optimizers

        def _wd_configure_optimizers():
            result = _orig_configure_optimizers()
            # result may be a single optimizer or a Lightning dict/list
            opts = result if isinstance(result, (list, tuple)) else [result]
            for item in opts:
                opt = item.get("optimizer", item) if isinstance(item, dict) else item
                if hasattr(opt, "param_groups"):
                    for pg in opt.param_groups:
                        if pg.get("weight_decay", 0.0) == 0.0:
                            pg["weight_decay"] = _weight_decay
            return result

        model.configure_optimizers = _wd_configure_optimizers
        logger.info("Weight decay %.1e applied to optimizer param groups", _weight_decay)

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

    pred = raw_prediction.cpu().numpy() if isinstance(raw_prediction, torch.Tensor) else raw_prediction
    n_days = pred.shape[0]
    median_idx = len(quantiles) // 2

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
    raw_median_0 = float(pred[0, median_idx])
    anomaly_detected = abs(raw_median_0) > ANOMALY_DAILY_RET
    if anomaly_detected:
        logger.error(
            "format_prediction: anomalous raw return detected at T+1: %.4f "
            "(|r| > %.3f). Likely a scaling / target-space bug — the value "
            "will be bounded at ±%.2f and flagged in the response.",
            raw_median_0, ANOMALY_DAILY_RET, ANOMALY_DAILY_RET,
        )

    def _bound(x: float) -> float:
        """Only clip if outside the anomaly bound; otherwise pass through."""
        if abs(x) > ANOMALY_DAILY_RET:
            return float(np.sign(x) * ANOMALY_DAILY_RET)
        return float(x)

    # Quantile spreads (distance of each quantile from the median, in
    # return-space). We do NOT clip *spreads* — models with a healthy
    # variance ratio produce spreads of ~2σ and that is fine.
    raw_med_0 = raw_median_0
    spread_q10 = float(pred[0, 1]) - raw_med_0 if len(quantiles) > 2 else 0.0
    spread_q90 = float(pred[0, -2]) - raw_med_0 if len(quantiles) > 2 else 0.0
    spread_q02 = float(pred[0, 0]) - raw_med_0
    spread_q98 = float(pred[0, -1]) - raw_med_0

    daily_forecasts = []
    cum_price_med = baseline_price

    for d in range(n_days):
        raw_med = float(pred[d, median_idx])
        med = _bound(raw_med)
        cum_price_med *= (1 + med)
        cum_return = (cum_price_med / baseline_price) - 1.0

        # √t spread expansion keeps multi-day uncertainty realistic instead
        # of exponentially compounding tail quantiles.
        scale = (d + 1) ** 0.5

        daily_forecasts.append({
            "day": d + 1,
            "daily_return": med,
            "raw_daily_return": raw_med,
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
