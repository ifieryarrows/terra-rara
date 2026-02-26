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


def _build_asro_pf_loss(asro_cfg, quantiles: list):
    """
    Build an ASRO loss that satisfies pytorch_forecasting >= 1.0 requirements.

    pytorch_forecasting requires the loss to be a torchmetrics ``Metric``
    subclass.  We subclass ``QuantileLoss`` (which already satisfies this)
    and override its ``loss()`` method with our ASRO logic.
    """
    from pytorch_forecasting.metrics import QuantileLoss

    # Capture ASRO nn.Module so it runs inside the Metric wrapper
    _asro_module = AdaptiveSharpeRatioLoss.from_config(asro_cfg, quantiles)

    class _ASROPFLoss(QuantileLoss):
        """pytorch_forecasting-compatible ASRO loss wrapper."""

        def loss(self, y_pred: torch.Tensor, target) -> torch.Tensor:  # type: ignore[override]
            # pytorch_forecasting passes target as a tuple (values, weights)
            if isinstance(target, (list, tuple)):
                y_actual = target[0]
            else:
                y_actual = target
            return _asro_module(y_pred, y_actual)

    return _ASROPFLoss(quantiles=quantiles)


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

    if use_asro:
        loss = _build_asro_pf_loss(cfg.asro, quantiles)
        logger.info("Using ASRO loss (lambda_vol=%.2f, lambda_quantile=%.2f)", cfg.asro.lambda_vol, cfg.asro.lambda_quantile)
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

def get_variable_importance(model) -> Dict[str, float]:
    """
    Extract learned variable importance from the TFT's Variable Selection Networks.

    Returns a dict mapping feature name -> normalised importance score.
    """
    try:
        interpretation = model.interpret_output(
            model.predict(model.val_dataloader(), return_x=True),
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
        Dict with median forecast, confidence bands, and volatility estimate.
    """
    pred = raw_prediction.cpu().numpy() if isinstance(raw_prediction, torch.Tensor) else raw_prediction

    median_idx = len(quantiles) // 2
    q_dict = {f"q{q:.2f}": float(pred[0, i]) for i, q in enumerate(quantiles)}

    median_return = float(pred[0, median_idx])
    q10_return = float(pred[0, 1]) if len(quantiles) > 2 else median_return
    q90_return = float(pred[0, -2]) if len(quantiles) > 2 else median_return
    q02_return = float(pred[0, 0])
    q98_return = float(pred[0, -1])

    vol_estimate = (q90_return - q10_return) / 2.0

    return {
        "predicted_return_median": median_return,
        "predicted_return_q10": q10_return,
        "predicted_return_q90": q90_return,
        "predicted_price_median": baseline_price * (1 + median_return),
        "predicted_price_q10": baseline_price * (1 + q10_return),
        "predicted_price_q90": baseline_price * (1 + q90_return),
        "confidence_band_96": (
            baseline_price * (1 + q02_return),
            baseline_price * (1 + q98_return),
        ),
        "volatility_estimate": vol_estimate,
        "quantiles": q_dict,
        "prediction_horizon_days": pred.shape[0],
    }
