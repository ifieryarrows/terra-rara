"""Fail-open Chronos-2 foundation time-series prior for 5-day copper quantiles.

Integrates zero-shot multi-quantile price forecasts from ``amazon/chronos-2``
strictly causally over trailing closes ``closes[:pos + 1]``, converts them to
7-quantile 5-day cumulative log returns, validates on chronological validation
without tuning extra degrees of freedom, and blends in ordered quantile space
with the TFT-ASRO path while preserving strict quantile monotonicity.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

CHRONOS2_MODEL_ID = "amazon/chronos-2"
CHRONOS2_QUANTILE_LEVELS: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)
_TAIL_RATIO_98_90 = 2.0537489 / 1.2815516
_PIPELINE_CACHE: dict[tuple[str, str], Any] = {}


def is_foundation_prior_runtime_allowed(cfg_enabled: bool = True) -> bool:
    """Return True only when runtime Chronos-2 inference is allowed.

    Disables model loading/Hub downloads during pytest runs unless
    ``COPPER_FORCE_CHRONOS2_IN_TESTS=1`` is explicitly set, or when
    ``COPPER_DISABLE_CHRONOS2`` is truthy.
    """
    if not cfg_enabled:
        return False
    disable_flag = os.environ.get("COPPER_DISABLE_CHRONOS2", "").strip().lower()
    if disable_flag in {"1", "true", "yes", "on"}:
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        force_flag = os.environ.get("COPPER_FORCE_CHRONOS2_IN_TESTS", "").strip().lower()
        return force_flag in {"1", "true", "yes", "on"}
    return True


def _map_9_to_7_log_quantiles(
    q9_prices: np.ndarray,
    anchor_closes: np.ndarray,
) -> np.ndarray:
    """Map ``(N, 9)`` price quantiles ``[0.1..0.9]`` to 7 log-return quantiles.

    Output order matches ``(0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)`` and
    guarantees strict non-crossing monotonicity.
    """
    q9 = np.asarray(q9_prices, dtype=np.float64)
    anchors = np.asarray(anchor_closes, dtype=np.float64)
    if q9.ndim == 1:
        q9 = q9[None, :]
    anchors = anchors.reshape(-1, 1)
    if q9.shape[0] != anchors.shape[0] or q9.shape[1] != 9:
        raise ValueError(
            f"Expected q9_prices shape ({anchors.shape[0]}, 9), got {q9.shape}"
        )

    q9_log = np.log(np.maximum(q9, 1e-8) / np.maximum(anchors, 1e-8))
    q10 = q9_log[:, 0]
    q25 = 0.5 * (q9_log[:, 1] + q9_log[:, 2])
    q50 = q9_log[:, 4]
    q75 = 0.5 * (q9_log[:, 6] + q9_log[:, 7])
    q90 = q9_log[:, 8]
    q02 = q50 - _TAIL_RATIO_98_90 * np.maximum(q50 - q10, 1e-5)
    q98 = q50 + _TAIL_RATIO_98_90 * np.maximum(q90 - q50, 1e-5)

    stacked = np.stack([q02, q10, q25, q50, q75, q90, q98], axis=-1)
    return np.maximum.accumulate(stacked, axis=-1)


def load_chronos2_pipeline(
    model_id: str = CHRONOS2_MODEL_ID,
    device: str = "auto",
) -> Optional[Any]:
    """Lazily load and cache ``Chronos2Pipeline``, failing open on error."""
    try:
        import torch
        from chronos import Chronos2Pipeline
    except Exception as exc:
        logger.warning("Chronos-2 unavailable (import failed, continuing TFT-only): %s", exc)
        return None

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = str(device)

    cache_key = (str(model_id), resolved_device)
    cached = _PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        local_source: str | Path = str(model_id)
        if not Path(local_source).exists():
            # Respect HF_HOME offline caches cleanly when present; Chronos2Pipeline handles HF Hub lookup.
            local_source = str(model_id)
        pipe = Chronos2Pipeline.from_pretrained(local_source, device_map=resolved_device)
        _PIPELINE_CACHE[cache_key] = pipe
        logger.info("Loaded Chronos-2 pipeline (%s on %s)", model_id, resolved_device)
        return pipe
    except Exception as exc:
        logger.warning(
            "Chronos-2 pipeline load failed for %s (%s); continuing fail-open TFT-only",
            model_id,
            exc,
        )
        return None


def predict_chronos2_weekly_log_quantiles(
    closes: np.ndarray | Sequence[float],
    origin_positions: Sequence[int] | np.ndarray,
    *,
    horizon: int = 5,
    context_len: int = 252,
    batch_size: int = 32,
    model_id: str = CHRONOS2_MODEL_ID,
    pipeline: Optional[Any] = None,
    allow_runtime: Optional[bool] = None,
) -> Optional[np.ndarray]:
    """Predict strictly causal 5-day cumulative log-return quantiles at ``origin_positions``.

    Returns ``(N, 7)`` float64 array or ``None`` if disabled/unavailable/failed.
    Rows with insufficient or non-positive trailing closes yield ``np.nan`` so
    the caller leaves those rows at the pure TFT forecast.
    """
    runtime_ok = (
        is_foundation_prior_runtime_allowed(True)
        if allow_runtime is None
        else bool(allow_runtime)
    )
    if pipeline is None and not runtime_ok:
        return None

    close_arr = np.asarray(closes, dtype=np.float64).reshape(-1)
    pos_arr = np.asarray(origin_positions, dtype=int).reshape(-1)
    n_req = len(pos_arr)
    if n_req == 0 or len(close_arr) < 20:
        return None

    pipe = pipeline if pipeline is not None else load_chronos2_pipeline(model_id=model_id)
    if pipe is None:
        return None

    ctx_len = max(int(context_len), 20)
    bs = max(int(batch_size), 1)
    h = max(int(horizon), 1)
    q_levels = list(CHRONOS2_QUANTILE_LEVELS)

    contexts: list[Any] = []
    anchors: list[float] = []
    valid_indices: list[int] = []

    try:
        import torch
    except Exception as exc:
        logger.warning("Torch unavailable for Chronos-2 prediction: %s", exc)
        return None

    for req_idx, pos in enumerate(pos_arr.tolist()):
        if pos < 0 or pos >= len(close_arr):
            continue
        anchor = float(close_arr[pos])
        if not np.isfinite(anchor) or anchor <= 0.0:
            continue
        start = max(0, pos + 1 - ctx_len)
        window = close_arr[start : pos + 1]
        window = window[np.isfinite(window) & (window > 0.0)]
        if len(window) < 20:
            continue
        contexts.append(torch.tensor(window, dtype=torch.float32))
        anchors.append(anchor)
        valid_indices.append(req_idx)

    if not contexts:
        return None

    out = np.full((n_req, 7), np.nan, dtype=np.float64)
    try:
        q9_batches: list[np.ndarray] = []
        for b in range(0, len(contexts), bs):
            batch_ctx = contexts[b : b + bs]
            q_out, _ = pipe.predict_quantiles(
                inputs=batch_ctx,
                prediction_length=h,
                quantile_levels=q_levels,
            )
            if isinstance(q_out, list):
                arr = np.stack(
                    [
                        (
                            item.squeeze(0)[-1]
                            if getattr(item, "ndim", 0) == 3
                            else item[-1]
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        for item in q_out
                    ],
                    axis=0,
                )
            elif hasattr(q_out, "detach"):
                arr_all = q_out.detach().cpu().numpy()
                arr = arr_all[:, -1, :] if arr_all.ndim == 3 else arr_all
            else:
                arr_all = np.asarray(q_out, dtype=np.float64)
                arr = arr_all[:, -1, :] if arr_all.ndim == 3 else arr_all
            q9_batches.append(np.asarray(arr, dtype=np.float64))

        q9_all = np.concatenate(q9_batches, axis=0)
        q7_valid = _map_9_to_7_log_quantiles(q9_all, np.asarray(anchors, dtype=np.float64))
        out[np.asarray(valid_indices, dtype=int)] = q7_valid
        return out
    except Exception as exc:
        logger.warning(
            "Chronos-2 prediction failed (%s); continuing fail-open TFT-only",
            exc,
        )
        return None


def evaluate_foundation_prior_on_validation(
    val_actual_weekly: np.ndarray,
    val_tft_weekly_q: np.ndarray,
    val_foundation_weekly_q: Optional[np.ndarray],
    *,
    weight: float = 0.20,
    model_id: str = CHRONOS2_MODEL_ID,
    context_len: int = 252,
    max_val_mae_vs_naive: float = 1.25,
    max_blend_mae_degradation: float = 0.02,
    quantiles: Sequence[float] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
) -> dict[str, Any]:
    """Run zero-DoF validation sanity check for the fixed-weight foundation prior."""
    from deep_learning.training.metrics import directional_accuracy

    base_meta: dict[str, Any] = {
        "enabled": False,
        "provider": "chronos2",
        "model_id": str(model_id),
        "weight": 0.0,
        "configured_weight": float(weight),
        "context_len": int(context_len),
        "fit_split": "validation",
        "sample_count": 0,
        "reason": "foundation_unavailable",
    }
    if val_foundation_weekly_q is None:
        return base_meta

    q_list = list(quantiles)
    med_idx = q_list.index(0.50) if 0.50 in q_list else len(q_list) // 2

    y = np.asarray(val_actual_weekly, dtype=np.float64).reshape(-1)
    q_tft = np.asarray(val_tft_weekly_q, dtype=np.float64)
    q_fnd = np.asarray(val_foundation_weekly_q, dtype=np.float64)
    n = min(len(y), len(q_tft), len(q_fnd))
    if n < 20:
        base_meta["sample_count"] = int(n)
        base_meta["reason"] = "insufficient_validation_samples"
        return base_meta

    y = y[:n]
    q_tft = q_tft[:n]
    q_fnd = q_fnd[:n]
    valid = np.isfinite(y) & np.isfinite(q_tft[:, med_idx]) & np.isfinite(q_fnd[:, med_idx])
    y = y[valid]
    p_tft = q_tft[valid, med_idx]
    p_fnd = q_fnd[valid, med_idx]
    if len(y) < 20:
        base_meta["sample_count"] = int(len(y))
        base_meta["reason"] = "insufficient_finite_validation_samples"
        return base_meta

    naive_mae = max(float(np.mean(np.abs(y))), 1e-8)
    tft_mae_ratio = float(np.mean(np.abs(y - p_tft)) / naive_mae)
    fnd_mae_ratio = float(np.mean(np.abs(y - p_fnd)) / naive_mae)

    w = float(np.clip(weight, 0.0, 0.50))
    p_blend = (1.0 - w) * p_tft + w * p_fnd
    blend_mae_ratio = float(np.mean(np.abs(y - p_blend)) / naive_mae)

    actual_pos = float((y > 0.0).mean())
    blend_pos = float((p_blend > 0.0).mean())
    tft_da = float(directional_accuracy(y, p_tft))
    blend_da = float(directional_accuracy(y, p_blend))

    balanced_actual = bool(0.30 <= actual_pos <= 0.70)
    sign_collapsed = bool(balanced_actual and (blend_pos > 0.85 or blend_pos < 0.15))

    enabled = True
    reason = "validation_sanity_passed"
    if w <= 0.0:
        enabled = False
        reason = "zero_configured_weight"
    elif not np.isfinite(fnd_mae_ratio) or fnd_mae_ratio > float(max_val_mae_vs_naive):
        enabled = False
        reason = f"foundation_mae_exceeded:{fnd_mae_ratio:.4f}>{max_val_mae_vs_naive:.2f}"
    elif blend_mae_ratio > tft_mae_ratio + float(max_blend_mae_degradation):
        enabled = False
        reason = (
            f"blend_mae_degraded:{blend_mae_ratio:.4f}>{tft_mae_ratio:.4f}"
            f"+{max_blend_mae_degradation:.2f}"
        )
    elif sign_collapsed:
        enabled = False
        reason = f"blend_sign_collapsed:pos={blend_pos:.3f}"

    return {
        "enabled": enabled,
        "provider": "chronos2",
        "model_id": str(model_id),
        "weight": w if enabled else 0.0,
        "configured_weight": w,
        "context_len": int(context_len),
        "fit_split": "validation",
        "sample_count": int(len(y)),
        "val_tft_mae_vs_naive": round(tft_mae_ratio, 4),
        "val_foundation_mae_vs_naive": round(fnd_mae_ratio, 4),
        "val_blended_mae_vs_naive": round(blend_mae_ratio, 4),
        "val_tft_da": round(tft_da, 4),
        "val_blended_da": round(blend_da, 4),
        "val_blended_pos_rate": round(blend_pos, 4),
        "reason": reason,
    }


def apply_foundation_prior_to_ordered_path(
    pred_path_np: np.ndarray,
    foundation_weekly_q: Optional[np.ndarray],
    *,
    foundation_prior: Optional[dict[str, Any]],
    quantiles: Sequence[float] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    horizon: int = 5,
) -> np.ndarray:
    """Blend an ordered daily quantile path with weekly Chronos-2 quantiles.

    Translates the median and scales lower/upper daily quantile half-widths so
    RSS-aggregated 5-day cumulative quantiles equal ``(1 - w) * W_tft + w * F_q``
    with strict per-day and weekly monotonicity preserved.
    """
    from deep_learning.models.monotonic_quantiles import aggregate_weekly_quantiles_np

    out = np.asarray(pred_path_np, dtype=np.float64).copy()
    if foundation_weekly_q is None or not foundation_prior or not foundation_prior.get("enabled"):
        return out

    w = float(foundation_prior.get("weight", 0.0) or 0.0)
    if not np.isfinite(w) or w <= 0.0:
        return out
    w = float(np.clip(w, 0.0, 1.0))

    squeezed = out.ndim == 2
    if squeezed:
        out = out[None, ...]
    if out.ndim != 3 or out.shape[1] < 1:
        return out[0] if squeezed else out

    f_arr = np.asarray(foundation_weekly_q, dtype=np.float64)
    if f_arr.ndim == 1:
        f_arr = f_arr[None, :]
    if f_arr.shape[0] != out.shape[0] or f_arr.shape[1] != out.shape[2]:
        raise ValueError(
            f"foundation_weekly_q shape {f_arr.shape} incompatible with pred_path {out.shape}"
        )

    h = min(int(horizon), out.shape[1])
    q_list = list(quantiles)
    median_idx = q_list.index(0.50) if 0.50 in q_list else len(q_list) // 2

    valid_rows = np.isfinite(f_arr).all(axis=1)
    if not valid_rows.any():
        return out[0] if squeezed else out

    ordered_slice = np.maximum.accumulate(out[:, :h, :], axis=-1)
    w_tft = aggregate_weekly_quantiles_np(ordered_slice, median_idx=median_idx)
    w_target = (1.0 - w) * w_tft + w * np.maximum.accumulate(f_arr, axis=-1)

    idx_arr = np.where(valid_rows)[0]
    med_tft = ordered_slice[idx_arr, :, median_idx]
    delta_med = (w_target[idx_arr, median_idx] - w_tft[idx_arr, median_idx]) / float(max(h, 1))
    new_med = med_tft + delta_med[:, None]

    new_slice = ordered_slice[idx_arr].copy()
    new_slice[:, :, median_idx] = new_med

    sqrt_h = float(np.sqrt(max(h, 1)))
    for q_idx in range(out.shape[2]):
        if q_idx == median_idx:
            continue
        is_lower = q_idx < median_idx
        if is_lower:
            tft_half = np.maximum(w_tft[idx_arr, median_idx] - w_tft[idx_arr, q_idx], 0.0)
            target_half = np.maximum(
                w_target[idx_arr, median_idx] - w_target[idx_arr, q_idx], 0.0
            )
            daily_half_tft = np.maximum(med_tft - ordered_slice[idx_arr, :, q_idx], 0.0)
            scale = np.where(tft_half > 1e-8, target_half / np.maximum(tft_half, 1e-8), 1.0)
            daily_half_new = np.where(
                tft_half[:, None] > 1e-8,
                daily_half_tft * scale[:, None],
                target_half[:, None] / sqrt_h,
            )
            new_slice[:, :, q_idx] = new_med - daily_half_new
        else:
            tft_half = np.maximum(w_tft[idx_arr, q_idx] - w_tft[idx_arr, median_idx], 0.0)
            target_half = np.maximum(
                w_target[idx_arr, q_idx] - w_target[idx_arr, median_idx], 0.0
            )
            daily_half_tft = np.maximum(ordered_slice[idx_arr, :, q_idx] - med_tft, 0.0)
            scale = np.where(tft_half > 1e-8, target_half / np.maximum(tft_half, 1e-8), 1.0)
            daily_half_new = np.where(
                tft_half[:, None] > 1e-8,
                daily_half_tft * scale[:, None],
                target_half[:, None] / sqrt_h,
            )
            new_slice[:, :, q_idx] = new_med + daily_half_new

    new_slice = np.maximum.accumulate(new_slice, axis=-1)
    out[idx_arr, :h, :] = new_slice
    return out[0] if squeezed else out
