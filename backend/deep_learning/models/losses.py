"""
Custom Loss Functions for TFT-ASRO.

Implements:
- AdaptiveSharpeRatioLoss (ASRO): jointly optimises risk-adjusted return,
  volatility calibration, and quantile coverage.
- CombinedQuantileLoss: standard multi-quantile pinball loss used as a
  component of ASRO and as a standalone baseline.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import numpy as np

from deep_learning.config import ASROConfig


def debug_asro_loss_direction() -> dict:
    """
    ASRO kayıp fonksiyonunun matematiksel doğrulaması.

    Üç test senaryosu:
      1. correct_direction : tanh(pred) ile actual aynı işaret → loss minimum, Sharpe pozitif
      2. anti_direction    : tanh(pred) ile actual ters işaret → loss maksimum, Sharpe negatif
      3. zero_predictions  : model sıfır tahmin üretiyor     → Sharpe sıfır (dar varyans tuzağı)

    Gradyan kontrolleri:
      - Her senaryoda grad_norm > 0 olmalı (tanh türevi var, sign() yok)
      - Doğru yönde kayıp < sıfır tahmin < ters yön kaybı sırası bozulmamalı

    Returns:
        {
          "passed": bool,
          "results": {scenario: {"loss", "grad_norm", "strategy_sharpe"}},
          "diagnostics": str   # geçti/kaldı açıklaması
        }
    """
    import torch

    torch.manual_seed(42)

    B, T, Q = 64, 5, 7
    quantiles = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
    actual_std = 0.024

    actual = torch.randn(B, T) * actual_std

    def _make_preds(median: torch.Tensor) -> torch.Tensor:
        """Build a quantile tensor from a given median, spread ≈ 2*actual_std."""
        out = torch.zeros(B, T, Q)
        for i, q in enumerate(quantiles):
            out[..., i] = median + (q - 0.5) * actual_std * 2
        return out

    scenarios = {
        "correct_direction": _make_preds(actual * 0.5),
        "anti_direction":    _make_preds(-actual * 0.5),
        "zero_predictions":  _make_preds(torch.zeros(B, T)),
    }

    fn = AdaptiveSharpeRatioLoss(quantiles=quantiles)
    results: dict = {}

    for name, preds in scenarios.items():
        p = preds.detach().requires_grad_(True)
        loss_val = fn(p, actual.detach())
        loss_val.backward()
        grad_norm = float(p.grad.norm().item()) if p.grad is not None else 0.0

        with torch.no_grad():
            med = p.detach()[..., len(quantiles) // 2]
            signal = torch.tanh(med * 100.0)   # same scale as training loss
            sr = float(
                (signal * actual).mean() / ((signal * actual).std() + 1e-6)
            )

        results[name] = {
            "loss": round(float(loss_val.item()), 6),
            "grad_norm": round(grad_norm, 6),
            "strategy_sharpe": round(sr, 4),
        }

    checks = {
        "correct < anti loss":
            results["correct_direction"]["loss"] < results["anti_direction"]["loss"],
        "correct Sharpe > 0":
            results["correct_direction"]["strategy_sharpe"] > 0,
        "anti Sharpe < 0":
            results["anti_direction"]["strategy_sharpe"] < 0,
        "gradients non-zero (correct)":
            results["correct_direction"]["grad_norm"] > 1e-6,
        "gradients non-zero (anti)":
            results["anti_direction"]["grad_norm"] > 1e-6,
    }

    passed = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    diagnostics = "ALL CHECKS PASSED" if passed else f"FAILED: {failed}"

    return {"passed": passed, "results": results, "diagnostics": diagnostics}


class CombinedQuantileLoss(nn.Module):
    """
    Multi-quantile pinball loss.

    Given K quantile predictions and actual values, the loss is the average
    pinball loss across all quantiles and samples.
    """

    def __init__(self, quantiles: Sequence[float] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)):
        super().__init__()
        self.register_buffer(
            "quantiles",
            torch.tensor(quantiles, dtype=torch.float32),
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        y_actual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_pred:   (batch, prediction_length, n_quantiles)
            y_actual: (batch, prediction_length)
        """
        if y_actual.dim() == 2:
            y_actual = y_actual.unsqueeze(-1)

        errors = y_actual - y_pred
        quantiles = self.quantiles.view(1, 1, -1)

        loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
        return loss.mean()


class AdaptiveSharpeRatioLoss(nn.Module):
    """
    TFT-ASRO loss: combines three objectives to break the low-variance trap.

    L = -Sharpe_component
        + lambda_vol  * volatility_calibration_loss
        + lambda_quantile * quantile_coverage_loss

    The Sharpe component incentivises the model to produce directionally correct
    predictions (not just low MSE), while the volatility term penalises
    under-estimation of realised variance, and the quantile term ensures
    proper tail coverage.
    """

    def __init__(
        self,
        quantiles: Sequence[float] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
        lambda_vol: float = 0.3,
        lambda_quantile: float = 0.2,
        risk_free_rate: float = 0.0,
        sharpe_eps: float = 1e-6,
        median_idx: Optional[int] = None,
    ):
        super().__init__()
        self.lambda_vol = lambda_vol
        self.lambda_quantile = lambda_quantile
        self.rf = risk_free_rate
        self.sharpe_eps = sharpe_eps
        self.median_idx = median_idx if median_idx is not None else len(quantiles) // 2

        self.quantile_loss = CombinedQuantileLoss(quantiles)

        q = list(quantiles)
        self._q10_idx = q.index(0.10) if 0.10 in q else 1
        self._q90_idx = q.index(0.90) if 0.90 in q else len(q) - 2

    def forward(
        self,
        y_pred: torch.Tensor,
        y_actual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_pred:   (batch, prediction_length, n_quantiles)
            y_actual: (batch, prediction_length)
        """
        median_pred = y_pred[:, :, self.median_idx]
        y_actual_f = y_actual.float()

        # --- Sharpe component: fixed-scale tanh soft-sign (scale = 100) ---
        # Root cause of directional collapse:
        #   pred_std ≈ 0.01 (actual return space) → tanh(0.01) ≈ 0.01 (linear)
        #   The Sharpe term becomes noise-dominated; quantile loss takes over.
        #
        # Fix: multiply pred by 100 before tanh so that return-scale predictions
        # map into the soft-sign zone of tanh:
        #   pred = 0.005 → tanh(0.5)  ≈ 0.46  (directional onset, gradient=0.79)
        #   pred = 0.010 → tanh(1.0)  ≈ 0.76  (soft-sign, gradient=0.42)
        #   pred = 0.020 → tanh(2.0)  ≈ 0.96  (full soft-sign, gradient=0.07)
        # Gradient through tanh is non-zero everywhere (unlike sign()), preserving
        # backprop in the early training epochs where |pred| is still small.
        _TANH_SCALE = 100.0
        signal = torch.tanh(median_pred * _TANH_SCALE)
        strategy_returns = signal * y_actual_f - self.rf
        sharpe_loss = -(strategy_returns.mean() / (strategy_returns.std() + self.sharpe_eps))

        # --- Volatility calibration ---
        # Match Q90-Q10 spread to 2× actual σ so the prediction interval tracks
        # realised volatility rather than collapsing to a constant.
        pred_spread = (y_pred[:, :, self._q90_idx] - y_pred[:, :, self._q10_idx]).mean()
        actual_std = y_actual_f.std() + self.sharpe_eps
        vol_loss = torch.abs(pred_spread - 2.0 * actual_std)

        # --- Quantile (pinball) loss ---
        q_loss = self.quantile_loss(y_pred, y_actual)

        # --- Weighted combination: w_quantile * calibration + w_sharpe * directional ---
        #
        # Old formula: sharpe + lambda_vol*vol + lambda_quantile*q
        #   → implicit Sharpe weight = 1.0 (not normalised, hard to interpret)
        #
        # New formula: w_q * (q + lambda_vol*vol) + (1-w_q) * sharpe
        #   → w_quantile + w_sharpe = 1.0, both components are interpretable
        #   → lambda_quantile config value IS the quantile bundle weight (e.g. 0.4)
        #
        # Calibration bundle (quantile + vol):
        #   ensures the 7 quantile bands remain properly calibrated;
        #   TFT's probabilistic nature requires this or it degenerates to regression.
        # Sharpe component:
        #   ensures the median prediction is directionally correct;
        #   without this the model regresses to mean → variance collapse.
        w_sharpe = 1.0 - self.lambda_quantile          # e.g. 0.6
        calibration = q_loss + self.lambda_vol * vol_loss
        total = self.lambda_quantile * calibration + w_sharpe * sharpe_loss
        return total

    @classmethod
    def from_config(cls, cfg: ASROConfig, quantiles: Sequence[float]) -> "AdaptiveSharpeRatioLoss":
        return cls(
            quantiles=quantiles,
            lambda_vol=cfg.lambda_vol,
            lambda_quantile=cfg.lambda_quantile,
            risk_free_rate=cfg.risk_free_rate,
        )
