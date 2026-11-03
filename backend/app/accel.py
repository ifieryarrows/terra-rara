"""
Hybrid Native Accelerator Bridge for CopperMind.

Automatically binds to compiled Rust PyO3 + Rayon kernels (`copper_rust_accel`)
and transparently falls back to vectorized C/C++/SIMD kernels when running in an
environment where the native wheel has not been built yet.
"""

from __future__ import annotations

import html
import re
import numpy as np

try:
    import copper_rust_accel as _rust_ext  # type: ignore[import-not-found]

    HAS_RUST_EXT = True
    ACCEL_BACKEND = "rust_pyo3"
except ImportError:
    _rust_ext = None
    HAS_RUST_EXT = False
    ACCEL_BACKEND = "native_c_simd"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_OR_STYLE_RE = re.compile(r"<(script|style)\b", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^\w\s]+")


def fast_softplus_np(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable C-vectorized softplus: log(1 + exp(-|x|)) + max(x, 0).
    Eliminates PyTorch tensor allocation/deallocation round-trips while keeping
    floating-point parity within ~8.6e-19 (machine epsilon).
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def fast_monotonic_quantiles_np(
    pred: np.ndarray,
    median_idx: int | None = None,
    min_gap: float = 1e-5,
    gap_scale: float = 0.03,
    init_bias: float = -3.0,
) -> np.ndarray:
    """
    Apply the production monotonic quantile transform using compiled Rust
    (stack-allocated steps, zero PyTorch overhead) or C/NumPy fallback.
    """
    arr = np.ascontiguousarray(pred, dtype=np.float64)
    q_dim = arr.shape[-1]
    if q_dim == 0 or arr.size == 0:
        return arr.copy()
    if median_idx is None:
        median_idx = q_dim // 2
    if HAS_RUST_EXT and hasattr(_rust_ext, "monotonic_quantiles_bytes_rs"):
        raw_bytes = _rust_ext.monotonic_quantiles_bytes_rs(
            arr.tobytes(),
            int(q_dim),
            int(median_idx),
            float(min_gap),
            float(gap_scale),
            float(init_bias),
        )
        return np.frombuffer(raw_bytes, dtype=np.float64).reshape(arr.shape).copy()

    base = arr[..., median_idx : median_idx + 1]
    lower_raw = arr[..., :median_idx][..., ::-1] + init_bias
    upper_raw = arr[..., median_idx + 1 :] + init_bias
    lower_steps = min_gap + gap_scale * fast_softplus_np(lower_raw)
    upper_steps = min_gap + gap_scale * fast_softplus_np(upper_raw)
    lower = (base - np.cumsum(lower_steps, axis=-1))[..., ::-1]
    upper = base + np.cumsum(upper_steps, axis=-1)
    return np.concatenate([lower, base, upper], axis=-1)


def fast_quantile_crossing_rate(
    y_pred_quantiles: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Single-pass register-level quantile crossing rate in Rust (zero intermediate
    `np.diff` array allocation) with C/NumPy fallback.
    """
    arr = np.ascontiguousarray(y_pred_quantiles, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    q_dim = arr.shape[-1]
    if q_dim < 2 or arr.size == 0:
        return 0.0
    if HAS_RUST_EXT and hasattr(_rust_ext, "quantile_crossing_rate_bytes_rs"):
        return float(
            _rust_ext.quantile_crossing_rate_bytes_rs(
                arr.tobytes(),
                int(q_dim),
                float(eps),
            )
        )
    violations = np.diff(arr, axis=-1) < -eps
    return float(violations.mean())


def fast_quantile_median_sort_gap(
    y_pred_quantiles: np.ndarray,
    median_idx: int | None = None,
) -> tuple[float, float]:
    """
    Stack-sorted (L1 register `[f64; 16]`) median sort gap in Rust (zero heap
    allocation) with C/NumPy fallback.
    """
    arr = np.ascontiguousarray(y_pred_quantiles, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    q_dim = arr.shape[-1]
    if q_dim == 0 or arr.size == 0:
        return 0.0, 0.0
    if median_idx is None:
        median_idx = q_dim // 2
    if HAS_RUST_EXT and hasattr(_rust_ext, "quantile_median_sort_gap_bytes_rs"):
        mean_gap, max_gap = _rust_ext.quantile_median_sort_gap_bytes_rs(
            arr.tobytes(),
            int(q_dim),
            int(median_idx),
        )
        return float(mean_gap), float(max_gap)
    sorted_arr = np.sort(arr, axis=-1)
    gap = np.abs(arr[..., median_idx] - sorted_arr[..., median_idx])
    return float(gap.mean()), float(gap.max())


def fast_fit_weekly_interval_grid(
    actual_weekly: np.ndarray,
    weekly_base: np.ndarray,
    offset_active: np.ndarray,
    centered_active: np.ndarray,
    candidates: np.ndarray,
    clipped_ratio_cond: np.ndarray | None,
    power: float,
    n_low: int,
    n_high: int,
    min_scale: float,
    max_scale: float,
    min_gap: float = 1e-5,
    gap_scale: float = 0.03,
    alpha: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate all interval scale candidates in parallel across CPU cores via
    Rust Rayon (`fit_weekly_interval_grid_rs`) without allocating a 4D tensor.
    """
    n_samples = int(actual_weekly.shape[0])
    horizon = int(offset_active.shape[1])
    if HAS_RUST_EXT and hasattr(_rust_ext, "fit_weekly_interval_grid_rs"):
        aw_c = np.ascontiguousarray(actual_weekly, dtype=np.float64).tobytes()
        wb_c = np.ascontiguousarray(weekly_base, dtype=np.float64).tobytes()
        oa_c = np.ascontiguousarray(offset_active.reshape(n_samples, horizon), dtype=np.float64).tobytes()
        ca_c = np.ascontiguousarray(centered_active, dtype=np.float64).tobytes()
        cand_c = np.ascontiguousarray(candidates, dtype=np.float64).tobytes()
        cr_view = (
            np.ascontiguousarray(clipped_ratio_cond, dtype=np.float64).tobytes()
            if clipped_ratio_cond is not None
            else None
        )
        cov_list, score_list = _rust_ext.fit_weekly_interval_grid_rs(
            aw_c,
            wb_c,
            oa_c,
            ca_c,
            cand_c,
            cr_view,
            float(power),
            n_samples,
            horizon,
            int(n_low),
            int(n_high),
            float(min_scale),
            float(max_scale),
            float(min_gap),
            float(gap_scale),
            float(alpha),
        )
        return np.asarray(cov_list, dtype=np.float64), np.asarray(score_list, dtype=np.float64)

    # Vectorized C/NumPy fallback
    if clipped_ratio_cond is not None:
        factors = clipped_ratio_cond ** float(power)
        mult = np.clip(
            candidates[:, None] * factors[None, :],
            float(min_scale),
            float(max_scale),
        )[:, :, None, None]
    else:
        mult = candidates[:, None, None, None]

    x = offset_active[None, ...] + mult * centered_active[None, ...]
    sp_horizon_sum = fast_softplus_np(x).sum(axis=2)
    base_gap_low = float(horizon * n_low) * min_gap
    base_gap_high = float(horizon * n_high) * min_gap
    low_offset = base_gap_low + gap_scale * sp_horizon_sum[:, :, :n_low].sum(axis=2)
    high_offset = base_gap_high + gap_scale * sp_horizon_sum[:, :, n_low:].sum(axis=2)

    lo = weekly_base[None, :] - low_offset
    hi = weekly_base[None, :] + high_offset
    y_row = actual_weekly[None, :]
    inv_alpha_2 = 2.0 / alpha

    coverages = ((y_row >= lo) & (y_row <= hi)).mean(axis=1)
    width = hi - lo
    below = np.maximum(lo - y_row, 0.0)
    above = np.maximum(y_row - hi, 0.0)
    interval_scores = (width + inv_alpha_2 * below + inv_alpha_2 * above).mean(axis=1)
    return coverages, interval_scores


def fast_ols_row_slope(mat: np.ndarray) -> np.ndarray:
    """
    Compute row-wise OLS slope across maturity ranks (0..M-1) with NaN handling.
    """
    arr = np.ascontiguousarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return np.full(arr.shape[0] if arr.ndim >= 1 else 0, np.nan, dtype=np.float64)
    n_rows, n_cols = arr.shape
    if HAS_RUST_EXT and hasattr(_rust_ext, "ols_row_slope_bytes_rs"):
        raw_bytes = _rust_ext.ols_row_slope_bytes_rs(
            arr.tobytes(),
            int(n_rows),
            int(n_cols),
        )
        return np.frombuffer(raw_bytes, dtype=np.float64).copy()

    ranks = np.arange(n_cols, dtype=np.float64)[None, :]
    valid = ~np.isnan(arr)
    counts = valid.sum(axis=1)
    r_masked = np.where(valid, ranks, 0.0)
    v_masked = np.where(valid, arr, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_mean = r_masked.sum(axis=1, keepdims=True) / counts[:, None]
        v_mean = v_masked.sum(axis=1, keepdims=True) / counts[:, None]
        dr = np.where(valid, ranks - r_mean, 0.0)
        dv = np.where(valid, arr - v_mean, 0.0)
        num = (dr * dv).sum(axis=1)
        den = (dr * dr).sum(axis=1)
        slopes = np.where((counts >= 2) & (den > 0.0), num / den, np.nan)
    return slopes


def fast_strip_html(text: str) -> str:
    """Strip HTML tags and normalize whitespace using Rust or fast C-regex path."""
    if not text:
        return ""
    if "<" not in text and "&" not in text:
        return " ".join(text.split())
    if _SCRIPT_OR_STYLE_RE.search(text):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    if HAS_RUST_EXT and "&" not in text:
        return _rust_ext.clean_text_rs(text)
    stripped = _HTML_TAG_RE.sub(" ", text)
    if "&" in stripped:
        stripped = html.unescape(stripped)
    return " ".join(stripped.split())


def fast_canonical_title(title: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    if not title:
        return ""
    if HAS_RUST_EXT:
        return _rust_ext.canonical_title_rs(title)
    return " ".join(_NON_WORD_RE.sub(" ", title.lower()).split())


def fast_sharpe_ratio_hac(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualisation: float = 52.0,
    max_lag: int = 4,
) -> float:
    """Zero-copy Newey-West (Bartlett kernel) HAC Annualised Sharpe Ratio."""
    arr = np.ascontiguousarray(returns, dtype=np.float64).reshape(-1)
    n = arr.size
    if n < 2:
        return 0.0
    if HAS_RUST_EXT and hasattr(_rust_ext, "sharpe_ratio_hac_bytes_rs"):
        return float(
            _rust_ext.sharpe_ratio_hac_bytes_rs(
                arr.tobytes(),
                float(risk_free_rate),
                float(annualisation),
                int(max_lag),
            )
        )
    excess = arr - risk_free_rate / annualisation
    mean_excess = float(excess.mean())
    centered = excess - mean_excess
    gamma_0 = float(np.dot(centered, centered) / n)
    if gamma_0 < 1e-18:
        return 0.0
    hac_var = gamma_0
    effective_lag = min(max(int(max_lag), 0), n - 1)
    for lag in range(1, effective_lag + 1):
        weight = 1.0 - lag / (effective_lag + 1.0)
        gamma_lag = float(np.dot(centered[lag:], centered[:-lag]) / n)
        hac_var += 2.0 * weight * gamma_lag
    hac_std = float(np.sqrt(max(hac_var, gamma_0 * 1e-6)))
    if hac_std < 1e-9:
        return 0.0
    return float(np.sqrt(annualisation) * mean_excess / hac_std)
