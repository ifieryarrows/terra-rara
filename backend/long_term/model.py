"""Safeguarded 4-family structural committee for 20D / 60D / 120D copper outlook.

Components (all causal, low-variance, fit strictly on purged historical windows):

1. **CatBoost (F2 Theory-of-Storage + COT)** Huber regressor & 3-class classifier
   (with HistGradientBoosting fail-open fallback).
2. **XGBoost (F2 Theory-of-Storage + COT)** shallow regularized regressor & classifier.
3. **PyTorch MLP-PLR (3-seed)** Periodic Linear Embeddings neural net trained on joint
   Huber return + cross-entropy regime loss.
4. **Damped-trend Theta** econometric baseline + Gaussian implied probabilities.
5. **Ridge macro-trend anchor** retained for interpretable signed ``key_drivers``
   factor attribution and diagnostic telemetry.
6. **Causal OOF Shrinkage + Purged Split-Conformal Intervals (PI50 / PI80)** and
   5-tier Institutional Rating (``STRONG_OVERWEIGHT`` .. ``STRONG_UNDERWEIGHT``)
   with capped 50% total tree weight so trees cannot overfit alone.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from long_term.features import FEATURE_CATALOG, compute_macro_scorecard
from long_term.schemas import (
    InstitutionalRatingType,
    LongTermDataWindow,
    LongTermDriverItem,
    LongTermEnsembleComponents,
    LongTermHorizonForecast,
    LongTermOutlookResponse,
    LongTermValidationSummary,
    StanceType,
)

if TYPE_CHECKING:  # pragma: no cover
    from long_term.backtest import WalkForwardHorizonResult
    from long_term.features import LongTermPanelBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HorizonSpec:
    key: str
    label: str
    days: int
    weeks: int
    annualisation: float
    target_col: str


HORIZONS: dict[str, HorizonSpec] = {
    "20D": HorizonSpec("20D", "1 Month (20D)", 20, 4, 12.0, "target_log_ret_4w"),
    "60D": HorizonSpec("60D", "1 Quarter (60D)", 60, 12, 4.0, "target_log_ret_12w"),
    "120D": HorizonSpec("120D", "6 Months (120D)", 120, 26, 2.0, "target_log_ret_26w"),
}

# Full F2 Theory-of-Storage + non-linear COT + causal macro feature basis.
# ``select_model_features`` automatically retains columns available in the panel.
MODEL_FEATURES: tuple[str, ...] = (
    "hg_sma10w_ratio",
    "hg_sma40w_ratio",
    "hg_dd52",
    "hg_vol26",
    "dxy_ret26",
    "real_yield_10y",
    "real_yield_chg26",
    "breakeven_10y_z",
    "fxa_ret26",
    "cl_ret12",
    "usdcnh_ret12",
    "usdcnh_ret26",
    "miners_vs_hg_12w",
    "copx_vs_hg_12w",
    "hg_gc_ret12",
    "hg_gc_z104",
    "lme_scarcity",
    "lme_stock_pctl_3y",
    "lme_roll_yield_ann",
    "lme_tightness_convex",
    "lme_stock_accel",
    "lme_stock_chg12w",
    "cot_mm_net_pct_oi",
    "cot_mm_flow_4w",
    "cot_mm_flow_12w",
    "cot_mm_crowding_z156",
    "cot_mm_crowding_convex",
    "cot_spec_vs_prod_div",
    "t10y2y",
    "indpro_yoy",
    "vix_z52",
)

QUANTILE_LEVELS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass(frozen=True)
class LongTermModelParams:
    ridge_alpha: float = 3000.0
    ridge_window: int = 260  # ~5 years of weekly origins
    z_clip: float = 3.5
    x_clip: float = 3.5
    theta_lookback: int = 64
    theta_phi: float = 0.82
    ridge_weight: float = 0.55
    logit_C: float = 0.05
    logit_weight: float = 0.35
    regime_k: float = 0.5  # class threshold in units of sigma_H
    min_z_scale: float = 0.85
    min_feature_coverage: float = 0.70
    min_train_rows: int = 80
    stance_edge: float = 0.12
    # Safeguarded 4-family committee weights (trees capped at 50% total)
    enable_committee: bool = True
    catboost_weight: float = 0.25
    xgboost_weight: float = 0.25
    mlp_weight: float = 0.25
    theta_weight: float = 0.25
    tree_iterations: int = 220
    mlp_epochs: int = 35
    mlp_seeds: tuple[int, ...] = (42, 137, 256)
    default_shrinkage_alpha: float = 0.70
    min_shrinkage_alpha: float = 0.35

    def effective_ridge_alpha(self, horizon_weeks: int) -> float:
        """Scale L2 penalty for reduced effective independent N of overlapping horizons."""
        return float(self.ridge_alpha * max(1.0, horizon_weeks / 4.0) ** 0.8)

    def effective_ridge_weight(self, horizon_weeks: int) -> float:
        """Balance macro-trend Ridge with Damped Theta (used in Ridge diagnostic / drivers)."""
        if horizon_weeks >= 12:
            return min(self.ridge_weight, 0.50)
        return self.ridge_weight

    def normalized_committee_weights(self) -> tuple[float, float, float, float]:
        """Return normalized (catboost, xgboost, mlp, theta) weights summing to 1."""
        raw = np.array(
            [
                max(0.0, self.catboost_weight),
                max(0.0, self.xgboost_weight),
                max(0.0, self.mlp_weight),
                max(0.0, self.theta_weight),
            ],
            dtype=np.float64,
        )
        total = float(raw.sum())
        if total <= 1e-12:
            return (0.25, 0.25, 0.25, 0.25)
        raw /= total
        return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


DEFAULT_PARAMS = LongTermModelParams()


# ---------------------------------------------------------------------------
# Feature selection & horizon frames
# ---------------------------------------------------------------------------


def select_model_features(
    panel: pd.DataFrame,
    params: LongTermModelParams = DEFAULT_PARAMS,
    warmup_rows: int = 52,
) -> list[str]:
    """Return model features present in the panel with adequate post-warmup coverage."""
    body = panel.iloc[warmup_rows:] if len(panel) > warmup_rows + 20 else panel
    selected: list[str] = []
    for col in MODEL_FEATURES:
        if col not in panel.columns:
            continue
        coverage = float(pd.to_numeric(body[col], errors="coerce").notna().mean()) if len(body) else 0.0
        if coverage >= params.min_feature_coverage:
            selected.append(col)
    return selected


@dataclass
class HorizonFrame:
    """Aligned numpy views of one horizon's supervised problem."""

    spec: HorizonSpec
    features: list[str]
    X: np.ndarray  # (n, k) raw feature matrix (NaN allowed; imputed per training fold)
    y: np.ndarray  # forward log return (NaN where unknown)
    sigma_h: np.ndarray  # causal horizon volatility
    feature_ok: np.ndarray  # bool: sigma finite and >= 50% features finite
    trainable: np.ndarray  # bool: feature_ok & target known
    index: pd.DatetimeIndex


def build_horizon_frame(panel: pd.DataFrame, spec: HorizonSpec, features: list[str]) -> HorizonFrame:
    X = panel[features].to_numpy(dtype=np.float64) if features else np.zeros((len(panel), 0))
    y = pd.to_numeric(panel[spec.target_col], errors="coerce").to_numpy(dtype=np.float64)
    sigma_w = pd.to_numeric(panel["sigma_w"], errors="coerce").to_numpy(dtype=np.float64)
    sigma_h = sigma_w * math.sqrt(spec.weeks)
    feature_ok = np.isfinite(sigma_h) & (sigma_h > 0.0)
    if X.shape[1]:
        finite_ratio = np.mean(np.isfinite(X), axis=1)
        feature_ok &= finite_ratio >= 0.50
    trainable = feature_ok & np.isfinite(y)
    return HorizonFrame(spec, features, X, y, sigma_h, feature_ok, trainable, panel.index)


# ---------------------------------------------------------------------------
# Damped-trend Theta
# ---------------------------------------------------------------------------

_SES_ALPHAS = np.linspace(0.05, 1.0, 20)


def ses_final_level(values: np.ndarray, alphas: np.ndarray = _SES_ALPHAS) -> float:
    """Simple exponential smoothing level (one-step forecast) with SSE-optimal alpha grid."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    level = np.full(alphas.shape, vals[0], dtype=np.float64)
    sse = np.zeros(alphas.shape, dtype=np.float64)
    for obs in vals[1:]:
        err = obs - level
        sse += err * err
        level = level + alphas * err
    return float(level[int(np.argmin(sse))])


@dataclass
class ThetaComponents:
    ses_level: np.ndarray
    slope: np.ndarray
    trend_last: np.ndarray
    y_last: np.ndarray

    def forecast_log_return(self, positions: np.ndarray, horizon_weeks: int, phi: float) -> np.ndarray:
        pos = np.asarray(positions, dtype=int)
        damped_steps = float(sum(phi**k for k in range(1, horizon_weeks + 1)))
        fc = 0.5 * (self.ses_level[pos] + self.trend_last[pos] + self.slope[pos] * damped_steps)
        out = fc - self.y_last[pos]
        return np.where(np.isfinite(out), np.clip(out, -0.45, 0.45), 0.0)


def compute_theta_components(log_prices: np.ndarray, lookback: int = 64, min_obs: int = 24) -> ThetaComponents:
    """Pre-compute causal Theta state for every origin (uses only prices up to the origin)."""
    lp = np.asarray(log_prices, dtype=np.float64)
    n = lp.size
    ses = np.full(n, np.nan)
    slope_arr = np.full(n, np.nan)
    trend_last = np.full(n, np.nan)
    y_last = np.full(n, np.nan)
    for i in range(n):
        window = lp[max(0, i - lookback + 1) : i + 1]
        window = window[np.isfinite(window)]
        if window.size < min_obs:
            continue
        x = np.arange(window.size, dtype=np.float64)
        slope, intercept = np.polyfit(x, window, 1)
        trend = intercept + slope * x
        theta2 = 2.0 * window - trend
        ses[i] = ses_final_level(theta2)
        slope_arr[i] = slope
        trend_last[i] = trend[-1]
        y_last[i] = window[-1]
    return ThetaComponents(ses, slope_arr, trend_last, y_last)


# ---------------------------------------------------------------------------
# Standardization & Ridge macro-trend model
# ---------------------------------------------------------------------------


@dataclass
class CausalPreprocessor:
    medians: np.ndarray
    scaler: Optional[StandardScaler]
    x_clip: float

    def transform(self, X: np.ndarray) -> np.ndarray:
        X2 = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X2.shape[1] == 0 or self.scaler is None:
            return np.zeros((X2.shape[0], 0), dtype=np.float64)
        imp = np.where(np.isfinite(X2), X2, self.medians[None, :])
        Xs = self.scaler.transform(imp)
        Xs = np.where(np.isfinite(Xs), Xs, 0.0)
        return np.clip(Xs, -self.x_clip, self.x_clip)


def fit_causal_preprocessor(X: np.ndarray, x_clip: float) -> CausalPreprocessor:
    X2 = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X2.shape[1] == 0:
        return CausalPreprocessor(np.zeros(0, dtype=np.float64), None, x_clip)
    med = np.nanmedian(X2, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    imp = np.where(np.isfinite(X2), X2, med[None, :])
    scaler = StandardScaler().fit(imp)
    scale = scaler.scale_.copy()
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    scaler.scale_ = scale
    return CausalPreprocessor(med, scaler, x_clip)


@dataclass
class FittedRidge:
    prep: CausalPreprocessor
    model: Ridge
    features: list[str]

    def standardize(self, X: np.ndarray) -> np.ndarray:
        return self.prep.transform(X)

    def predict_z(self, X: np.ndarray) -> np.ndarray:
        Xs = self.standardize(X)
        if Xs.shape[1] == 0:
            return np.zeros(Xs.shape[0], dtype=np.float64)
        return self.model.predict(Xs)

    def contributions_z(self, x_row: np.ndarray) -> np.ndarray:
        Xs = self.standardize(x_row)
        if Xs.shape[1] == 0:
            return np.zeros(0, dtype=np.float64)
        return Xs[0] * self.model.coef_


def fit_ridge(
    X: np.ndarray,
    z: np.ndarray,
    features: list[str],
    params: LongTermModelParams,
    horizon_weeks: int = 4,
) -> FittedRidge:
    prep = fit_causal_preprocessor(X, params.x_clip)
    Xs = prep.transform(X)
    model = Ridge(alpha=params.effective_ridge_alpha(horizon_weeks), fit_intercept=False)
    if Xs.shape[1] > 0:
        model.fit(Xs, np.clip(z, -params.z_clip, params.z_clip))
    else:
        model.coef_ = np.zeros(0, dtype=np.float64)
    return FittedRidge(prep, model, list(features))


# ---------------------------------------------------------------------------
# Regime classifier & institutional rating
# ---------------------------------------------------------------------------


def regime_classes(y: np.ndarray, sigma_h: np.ndarray, k: float) -> np.ndarray:
    """0 = BEARISH (r < -k sigma), 1 = NEUTRAL, 2 = BULLISH (r > k sigma)."""
    thr = k * sigma_h
    return np.where(y > thr, 2, np.where(y < -thr, 0, 1)).astype(int)


@dataclass
class FittedRegime:
    prep: CausalPreprocessor
    clf: Optional[LogisticRegression]
    prior: np.ndarray

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X2 = np.atleast_2d(X)
        if self.clf is None:
            return np.repeat(self.prior[None, :], X2.shape[0], axis=0)
        Xs = self.prep.transform(X2)
        if Xs.shape[1] == 0:
            return np.repeat(self.prior[None, :], X2.shape[0], axis=0)
        raw = self.clf.predict_proba(Xs)
        out = np.full((X2.shape[0], 3), 1e-6)
        for col, cls in enumerate(self.clf.classes_):
            out[:, int(cls)] = raw[:, col]
        return out / out.sum(axis=1, keepdims=True)


def fit_regime_classifier(
    X: np.ndarray, classes: np.ndarray, params: LongTermModelParams
) -> FittedRegime:
    counts = np.bincount(classes, minlength=3).astype(np.float64) + 1.0  # Laplace prior
    prior = counts / counts.sum()
    prep = fit_causal_preprocessor(X, params.x_clip)
    Xs = prep.transform(X)
    if len(np.unique(classes)) < 2 or Xs.shape[1] == 0:
        return FittedRegime(prep, None, prior)
    clf = LogisticRegression(C=params.logit_C, max_iter=1000)
    clf.fit(Xs, classes)
    return FittedRegime(prep, clf, prior)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return ndtr(x)


def gaussian_regime_proba(z_pred: np.ndarray, z_scale: float, k: float) -> np.ndarray:
    """Class probabilities implied by a Gaussian predictive law around the ensemble median."""
    z = np.atleast_1d(np.asarray(z_pred, dtype=np.float64))
    scale = max(float(z_scale), 1e-6)
    p_bear = _norm_cdf((-k - z) / scale)
    p_bull = 1.0 - _norm_cdf((k - z) / scale)
    p_neut = np.clip(1.0 - p_bear - p_bull, 0.0, 1.0)
    out = np.column_stack([p_bear, p_neut, p_bull])
    return out / out.sum(axis=1, keepdims=True)


def blend_regime_proba(p_logit: np.ndarray, p_gauss: np.ndarray, logit_weight: float) -> np.ndarray:
    w = float(np.clip(logit_weight, 0.0, 1.0))
    out = w * np.atleast_2d(p_logit) + (1.0 - w) * np.atleast_2d(p_gauss)
    out = np.clip(out, 1e-6, None)
    return out / out.sum(axis=1, keepdims=True)


def institutional_rating_from_forecast(
    pred_log_ret: float,
    sigma_h: float,
    p_bear: float,
    p_bull: float,
) -> tuple[InstitutionalRatingType, StanceType, float, str]:
    """Compute 5-tier institutional rating, 3-class stance, and conviction."""
    z_edge = float(pred_log_ret) / max(float(sigma_h), 1e-4)
    p_net = float(p_bull - p_bear)
    score = 0.65 * z_edge + 0.35 * p_net
    if score >= 0.30:
        rating: InstitutionalRatingType = "STRONG_OVERWEIGHT"
        stance: StanceType = "BULLISH"
    elif score >= 0.12:
        rating = "OVERWEIGHT"
        stance = "BULLISH"
    elif score <= -0.30:
        rating = "STRONG_UNDERWEIGHT"
        stance = "BEARISH"
    elif score <= -0.12:
        rating = "UNDERWEIGHT"
        stance = "BEARISH"
    else:
        rating = "NEUTRAL"
        stance = "NEUTRAL"
    conviction = float(np.clip(abs(score) / 0.45, 0.0, 1.0))
    label = "HIGH" if conviction >= 0.5 else ("MEDIUM" if conviction >= 0.2 else "LOW")
    return rating, stance, conviction, label


def stance_from_proba(p_bear: float, p_neut: float, p_bull: float, edge: float) -> tuple[StanceType, float, str]:
    tilt = float(p_bull - p_bear)
    stance: StanceType = "BULLISH" if tilt >= edge else ("BEARISH" if tilt <= -edge else "NEUTRAL")
    conviction = float(np.clip(abs(tilt) * 2.0, 0.0, 1.0))
    label = "HIGH" if conviction >= 0.5 else ("MEDIUM" if conviction >= 0.2 else "LOW")
    return stance, conviction, label


# ---------------------------------------------------------------------------
# Committee Sub-Models (CatBoost, XGBoost, PyTorch MLP-PLR)
# ---------------------------------------------------------------------------


def _scatter_3class_proba(raw_proba: np.ndarray, fitted_classes: np.ndarray, prior: np.ndarray) -> np.ndarray:
    n = raw_proba.shape[0]
    out = np.full((n, 3), 1e-4, dtype=np.float64)
    for col_idx, orig_cls in enumerate(fitted_classes):
        c = int(orig_cls)
        if 0 <= c < 3:
            out[:, c] = np.clip(raw_proba[:, col_idx], 1e-4, 1.0)
    out = 0.90 * out + 0.10 * prior[None, :]
    return out / out.sum(axis=1, keepdims=True)


@dataclass
class FittedClassifierWrapper:
    clf: Any
    classes_map: Optional[np.ndarray]
    prior: np.ndarray

    def predict_proba(self, Xs: np.ndarray) -> np.ndarray:
        if self.clf is None or Xs.shape[1] == 0:
            return np.repeat(self.prior[None, :], Xs.shape[0], axis=0)
        raw = np.asarray(self.clf.predict_proba(Xs), dtype=np.float64)
        cls = self.classes_map if self.classes_map is not None else getattr(self.clf, "classes_", np.arange(raw.shape[1]))
        return _scatter_3class_proba(raw, np.asarray(cls), self.prior)


def _fit_catboost_pair(
    Xs: np.ndarray, z_tr: np.ndarray, cls_tr: np.ndarray, params: LongTermModelParams
) -> tuple[Any, FittedClassifierWrapper]:
    counts = np.bincount(cls_tr, minlength=3).astype(np.float64) + 1.0
    prior = counts / counts.sum()
    if Xs.shape[1] == 0:
        return None, FittedClassifierWrapper(None, None, prior)

    uniq = np.unique(cls_tr)
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        reg = CatBoostRegressor(
            iterations=params.tree_iterations,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=18.0,
            loss_function="Huber:delta=1.0",
            random_seed=42,
            thread_count=2,
            allow_writing_files=False,
            verbose=0,
        )
        reg.fit(Xs, z_tr)
        if uniq.size >= 2:
            clf = CatBoostClassifier(
                iterations=max(100, params.tree_iterations - 20),
                depth=4,
                learning_rate=0.03,
                l2_leaf_reg=18.0,
                random_seed=42,
                thread_count=2,
                allow_writing_files=False,
                verbose=0,
            )
            clf.fit(Xs, cls_tr)
            clf_wrap = FittedClassifierWrapper(clf, np.asarray(clf.classes_), prior)
        else:
            clf_wrap = FittedClassifierWrapper(None, None, prior)
        return reg, clf_wrap
    except Exception as exc:
        logger.debug("CatBoost unavailable, falling back to HistGradientBoosting: %s", exc)
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        reg = HistGradientBoostingRegressor(
            max_iter=min(120, params.tree_iterations),
            max_depth=3,
            l2_regularization=15.0,
            learning_rate=0.04,
            random_state=42,
        )
        reg.fit(Xs, z_tr)
        if uniq.size >= 2:
            clf = HistGradientBoostingClassifier(
                max_iter=min(120, params.tree_iterations),
                max_depth=3,
                l2_regularization=15.0,
                learning_rate=0.04,
                random_state=42,
            )
            clf.fit(Xs, cls_tr)
            clf_wrap = FittedClassifierWrapper(clf, np.asarray(clf.classes_), prior)
        else:
            clf_wrap = FittedClassifierWrapper(None, None, prior)
        return reg, clf_wrap


def _fit_xgboost_pair(
    Xs: np.ndarray, z_tr: np.ndarray, cls_tr: np.ndarray, params: LongTermModelParams
) -> tuple[Any, FittedClassifierWrapper]:
    import xgboost as xgb

    counts = np.bincount(cls_tr, minlength=3).astype(np.float64) + 1.0
    prior = counts / counts.sum()
    if Xs.shape[1] == 0:
        return None, FittedClassifierWrapper(None, None, prior)

    reg = xgb.XGBRegressor(
        n_estimators=params.tree_iterations,
        max_depth=3,
        learning_rate=0.03,
        reg_lambda=15.0,
        subsample=0.8,
        colsample_bytree=0.65,
        random_state=42,
        n_jobs=2,
        verbosity=0,
    )
    reg.fit(Xs, z_tr)

    uniq = np.unique(cls_tr)
    if uniq.size >= 2:
        y_mapped = np.searchsorted(uniq, cls_tr)
        clf = xgb.XGBClassifier(
            n_estimators=max(100, params.tree_iterations - 20),
            max_depth=3,
            learning_rate=0.03,
            reg_lambda=15.0,
            subsample=0.8,
            colsample_bytree=0.65,
            random_state=42,
            n_jobs=2,
            verbosity=0,
        )
        clf.fit(Xs, y_mapped)
        clf_wrap = FittedClassifierWrapper(clf, uniq, prior)
    else:
        clf_wrap = FittedClassifierWrapper(None, None, prior)
    return reg, clf_wrap


class _PLRBlock:
    """Lazy-imported torch wrapper kept inside module cleanly."""

    @staticmethod
    def build_model(d_in: int, hidden: int = 48, dropout: float = 0.35):
        import torch
        import torch.nn as nn

        class _PLRLayer(nn.Module):
            def __init__(self, features_in: int, n_bins: int = 8, d_emb: int = 4, sigma: float = 0.25):
                super().__init__()
                self.freqs = nn.Parameter(torch.randn(features_in, n_bins) * sigma)
                self.proj = nn.Linear(features_in * n_bins * 2, features_in * d_emb)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                v = 2.0 * math.pi * x.unsqueeze(-1) * self.freqs.unsqueeze(0)
                pe = torch.cat([torch.sin(v), torch.cos(v)], dim=-1).flatten(1)
                return torch.relu(self.proj(pe))

        class _TabularMLPPLR(nn.Module):
            def __init__(self, features_in: int):
                super().__init__()
                self.plr = _PLRLayer(features_in, n_bins=8, d_emb=4)
                self.net = nn.Sequential(
                    nn.Linear(features_in * 4, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden // 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
                self.head_reg = nn.Linear(hidden // 2, 1)
                self.head_cls = nn.Linear(hidden // 2, 3)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                h = self.net(self.plr(x))
                return self.head_reg(h).squeeze(-1), self.head_cls(h)

        return _TabularMLPPLR(d_in)


@dataclass
class FittedMLPPLR:
    models: list[Any]
    prior: np.ndarray

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = Xs.shape[0]
        if not self.models or Xs.shape[1] == 0:
            return np.zeros(n, dtype=np.float64), np.repeat(self.prior[None, :], n, axis=0)
        import torch

        X_t = torch.tensor(Xs, dtype=torch.float32)
        z_list = []
        p_list = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                pr, pc = m(X_t)
                z_list.append(pr.numpy().astype(np.float64))
                p_list.append(torch.softmax(pc, dim=-1).numpy().astype(np.float64))
        proba = np.mean(p_list, axis=0)
        proba = np.clip(proba, 1e-4, 1.0)
        proba /= proba.sum(axis=1, keepdims=True)
        return np.mean(z_list, axis=0), proba


def _fit_mlp_plr(
    Xs: np.ndarray, z_tr: np.ndarray, cls_tr: np.ndarray, params: LongTermModelParams
) -> FittedMLPPLR:
    counts = np.bincount(cls_tr, minlength=3).astype(np.float64) + 1.0
    prior = counts / counts.sum()
    if Xs.shape[1] == 0 or Xs.shape[0] < 20:
        return FittedMLPPLR([], prior)

    import torch
    import torch.nn as nn

    Xt = torch.tensor(Xs, dtype=torch.float32)
    zt = torch.tensor(z_tr, dtype=torch.float32)
    ct = torch.tensor(cls_tr, dtype=torch.long)
    huber = nn.HuberLoss(delta=1.0)
    ce = nn.CrossEntropyLoss()
    bs = min(256, len(Xs))
    models: list[Any] = []
    prev_threads = torch.get_num_threads()
    try:
        # Single-thread CPU execution avoids OpenMP synchronization overhead on small tabular batches.
        if prev_threads != 1:
            torch.set_num_threads(1)
        for seed in params.mlp_seeds:
            torch.manual_seed(int(seed))
            m = _PLRBlock.build_model(Xs.shape[1])
            opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=2e-2)
            m.train()
            rng = np.random.default_rng(int(seed))
            for _epoch in range(params.mlp_epochs):
                perm = rng.permutation(len(Xs))
                for i in range(0, len(Xs), bs):
                    idx = perm[i : i + bs]
                    opt.zero_grad()
                    pr, pc = m(Xt[idx])
                    loss = huber(pr, zt[idx]) + 0.4 * ce(pc, ct[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
            m.eval()
            models.append(m)
    finally:
        if prev_threads != 1:
            torch.set_num_threads(prev_threads)
    return FittedMLPPLR(models, prior)


# ---------------------------------------------------------------------------
# Conformal intervals
# ---------------------------------------------------------------------------


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """Finite-sample split-conformal quantile ``ceil((n+1) * coverage) / n``."""
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("nan")
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(s, level, method="higher"))


def conformal_halfwidths(abs_norm_residuals: np.ndarray, fallback_scale: float = 1.0) -> tuple[float, float]:
    """Return ``(hw50, hw80)`` half-widths in sigma_H units, strictly ordered."""
    res = np.asarray(abs_norm_residuals, dtype=np.float64)
    res = res[np.isfinite(res)]
    if res.size >= 20:
        hw50 = conformal_quantile(res, 0.50)
        hw80 = conformal_quantile(res, 0.80)
    else:
        hw50 = 0.6745 * fallback_scale
        hw80 = 1.2816 * fallback_scale
    hw50 = max(float(hw50), 1e-3)
    hw80 = max(float(hw80), hw50 + 1e-3)
    return hw50, hw80


def build_return_quantiles(median: np.ndarray, sigma_h: np.ndarray, hw50: float, hw80: float) -> np.ndarray:
    """Non-crossing log-return quantiles ``[q10, q25, q50, q75, q90]`` (shape ``(n, 5)``)."""
    m = np.atleast_1d(np.asarray(median, dtype=np.float64))
    s = np.atleast_1d(np.asarray(sigma_h, dtype=np.float64))
    return np.column_stack([m - hw80 * s, m - hw50 * s, m, m + hw50 * s, m + hw80 * s])


# ---------------------------------------------------------------------------
# Ensemble fit on a training slice (shared by backtest & live)
# ---------------------------------------------------------------------------


@dataclass
class EnsembleFit:
    ridge: FittedRidge
    regime: FittedRegime
    z_scale: float
    hw50: float
    hw80: float
    n_train: int
    prep: Optional[CausalPreprocessor] = None
    catboost_reg: Any = None
    catboost_clf: Optional[FittedClassifierWrapper] = None
    xgboost_reg: Any = None
    xgboost_clf: Optional[FittedClassifierWrapper] = None
    mlp_plr: Optional[FittedMLPPLR] = None
    shrinkage_alpha: float = 0.70
    extras: dict[str, Any] = field(default_factory=dict)


def inner_purged_conformal_residuals(
    frame: HorizonFrame,
    train_pos: np.ndarray,
    theta: ThetaComponents,
    params: LongTermModelParams,
) -> tuple[np.ndarray, float]:
    """Abs normalized residuals & shrinkage alpha on a purged inner tail of train_pos."""
    H = frame.spec.weeks
    if train_pos.size < params.min_train_rows + 40:
        return np.array([]), params.default_shrinkage_alpha
    cal_n = int(np.clip(0.30 * train_pos.size, 40, 156))
    cal = train_pos[-cal_n:]
    fit_sub = train_pos[train_pos <= cal[0] - H]
    if fit_sub.size < 60:
        return np.array([]), params.default_shrinkage_alpha

    z_fit = np.clip(frame.y[fit_sub] / frame.sigma_h[fit_sub], -params.z_clip, params.z_clip)
    cls_fit = regime_classes(frame.y[fit_sub], frame.sigma_h[fit_sub], params.regime_k)
    prep = fit_causal_preprocessor(frame.X[fit_sub], params.x_clip)
    Xs_fit = prep.transform(frame.X[fit_sub])
    Xs_cal = prep.transform(frame.X[cal])

    if params.enable_committee and Xs_fit.shape[1] > 0:
        cb_r, _ = _fit_catboost_pair(Xs_fit, z_fit, cls_fit, params)
        xg_r, _ = _fit_xgboost_pair(Xs_fit, z_fit, cls_fit, params)
        mlp = _fit_mlp_plr(Xs_fit, z_fit, cls_fit, params)
        w_cb, w_xg, w_ml, w_th = params.normalized_committee_weights()
        s_cal = frame.sigma_h[cal]
        p_cb = np.clip(cb_r.predict(Xs_cal), -params.z_clip, params.z_clip) * s_cal
        p_xg = np.clip(xg_r.predict(Xs_cal), -params.z_clip, params.z_clip) * s_cal
        p_ml_z, _ = mlp.predict(Xs_cal)
        p_ml = np.clip(p_ml_z, -params.z_clip, params.z_clip) * s_cal
        p_th = theta.forecast_log_return(cal, H, params.theta_phi)
        p_raw = w_cb * p_cb + w_xg * p_xg + w_ml * p_ml + w_th * p_th
    else:
        ridge = fit_ridge(frame.X[fit_sub[-params.ridge_window :]], z_fit[-params.ridge_window :], frame.features, params, horizon_weeks=H)
        p_r = ridge.predict_z(frame.X[cal]) * frame.sigma_h[cal]
        p_t = theta.forecast_log_return(cal, H, params.theta_phi)
        w_r = params.effective_ridge_weight(H)
        p_raw = w_r * p_r + (1.0 - w_r) * p_t

    y_cal = frame.y[cal]
    var_y = max(float(np.var(y_cal)), 1e-6)
    alpha = float(
        np.clip(
            float(np.dot(y_cal, p_raw)) / max(float(np.dot(p_raw, p_raw)) + 25.0 * len(y_cal) * var_y, 1e-9),
            params.min_shrinkage_alpha,
            1.0,
        )
    )
    p_cal = alpha * p_raw
    return np.abs(y_cal - p_cal) / frame.sigma_h[cal], alpha


def fit_ensemble(
    frame: HorizonFrame,
    train_pos: np.ndarray,
    theta: ThetaComponents,
    params: LongTermModelParams,
    conformal_residuals: Optional[np.ndarray] = None,
    shrinkage_alpha: Optional[float] = None,
    compute_inner_conformal: bool = True,
) -> EnsembleFit:
    """Fit all committee + attribution components strictly on purged ``train_pos``."""
    if train_pos.size < params.min_train_rows:
        raise ValueError(f"Insufficient training rows ({train_pos.size}) for {frame.spec.key}")

    z_all = np.clip(frame.y[train_pos] / frame.sigma_h[train_pos], -params.z_clip, params.z_clip)
    classes = regime_classes(frame.y[train_pos], frame.sigma_h[train_pos], params.regime_k)

    roll = train_pos[-params.ridge_window :]
    ridge = fit_ridge(
        frame.X[roll],
        frame.y[roll] / frame.sigma_h[roll],
        frame.features,
        params,
        horizon_weeks=frame.spec.weeks,
    )
    regime = fit_regime_classifier(frame.X[train_pos], classes, params)
    z_scale = max(params.min_z_scale, float(np.nanstd(z_all)))

    prep = fit_causal_preprocessor(frame.X[train_pos], params.x_clip)
    Xs_tr = prep.transform(frame.X[train_pos])

    cb_reg = cb_clf = xg_reg = xg_clf = mlp_plr = None
    if params.enable_committee:
        cb_reg, cb_clf = _fit_catboost_pair(Xs_tr, z_all, classes, params)
        xg_reg, xg_clf = _fit_xgboost_pair(Xs_tr, z_all, classes, params)
        mlp_plr = _fit_mlp_plr(Xs_tr, z_all, classes, params)

    if conformal_residuals is not None and np.isfinite(conformal_residuals).sum() >= 20:
        residuals = conformal_residuals
        alpha_val = (
            float(shrinkage_alpha)
            if shrinkage_alpha is not None and np.isfinite(shrinkage_alpha)
            else params.default_shrinkage_alpha
        )
    elif compute_inner_conformal:
        residuals, inner_alpha = inner_purged_conformal_residuals(frame, train_pos, theta, params)
        alpha_val = (
            float(shrinkage_alpha)
            if shrinkage_alpha is not None and np.isfinite(shrinkage_alpha)
            else inner_alpha
        )
    else:
        residuals = np.array([])
        alpha_val = (
            float(shrinkage_alpha)
            if shrinkage_alpha is not None and np.isfinite(shrinkage_alpha)
            else params.default_shrinkage_alpha
        )

    hw50, hw80 = conformal_halfwidths(residuals, fallback_scale=z_scale)
    return EnsembleFit(
        ridge=ridge,
        regime=regime,
        z_scale=z_scale,
        hw50=hw50,
        hw80=hw80,
        n_train=int(train_pos.size),
        prep=prep,
        catboost_reg=cb_reg,
        catboost_clf=cb_clf,
        xgboost_reg=xg_reg,
        xgboost_clf=xg_clf,
        mlp_plr=mlp_plr,
        shrinkage_alpha=float(np.clip(alpha_val, params.min_shrinkage_alpha, 1.0)),
    )


@dataclass
class EnsemblePrediction:
    median: np.ndarray
    raw_median: np.ndarray
    ridge: np.ndarray
    theta: np.ndarray
    quantiles: np.ndarray  # (n, 5) log returns
    proba: np.ndarray  # (n, 3) [bear, neutral, bull]
    sigma_h: np.ndarray
    catboost: Optional[np.ndarray] = None
    xgboost: Optional[np.ndarray] = None
    mlp_plr: Optional[np.ndarray] = None


def predict_ensemble(
    fit: EnsembleFit,
    frame: HorizonFrame,
    positions: np.ndarray,
    theta: ThetaComponents,
    params: LongTermModelParams,
    *,
    apply_shrinkage: bool = True,
) -> EnsemblePrediction:
    pos = np.asarray(positions, dtype=int)
    s = frame.sigma_h[pos]
    X_sub = frame.X[pos]
    p_r = fit.ridge.predict_z(X_sub) * s
    p_t = theta.forecast_log_return(pos, frame.spec.weeks, params.theta_phi)

    # Damped Theta Gaussian 3-class probabilities
    p_gauss_th = gaussian_regime_proba(p_t / np.maximum(s, 1e-4), fit.z_scale, params.regime_k)

    p_cb = p_xg = p_ml = None
    if params.enable_committee and fit.prep is not None and fit.catboost_reg is not None and fit.xgboost_reg is not None and fit.mlp_plr is not None:
        Xs = fit.prep.transform(X_sub)
        p_cb = np.clip(np.asarray(fit.catboost_reg.predict(Xs), dtype=np.float64), -params.z_clip, params.z_clip) * s
        p_xg = np.clip(np.asarray(fit.xgboost_reg.predict(Xs), dtype=np.float64), -params.z_clip, params.z_clip) * s
        p_ml_z, prob_ml = fit.mlp_plr.predict(Xs)
        p_ml = np.clip(p_ml_z, -params.z_clip, params.z_clip) * s

        w_cb, w_xg, w_ml, w_th = params.normalized_committee_weights()
        p_raw = w_cb * p_cb + w_xg * p_xg + w_ml * p_ml + w_th * p_t

        prob_cb = fit.catboost_clf.predict_proba(Xs) if fit.catboost_clf is not None else p_gauss_th
        prob_xg = fit.xgboost_clf.predict_proba(Xs) if fit.xgboost_clf is not None else p_gauss_th
        proba = w_cb * prob_cb + w_xg * prob_xg + w_ml * prob_ml + w_th * p_gauss_th
        proba = np.clip(proba, 1e-4, 1.0)
        proba /= proba.sum(axis=1, keepdims=True)
    else:
        w_r = params.effective_ridge_weight(frame.spec.weeks)
        p_raw = w_r * p_r + (1.0 - w_r) * p_t
        p_gauss = gaussian_regime_proba(p_raw / np.maximum(s, 1e-4), fit.z_scale, params.regime_k)
        p_logit = fit.regime.predict_proba(X_sub)
        proba = blend_regime_proba(p_logit, p_gauss, params.logit_weight)

    alpha = fit.shrinkage_alpha if apply_shrinkage else 1.0
    p_cal = alpha * p_raw
    q = build_return_quantiles(p_cal, s, fit.hw50, fit.hw80)
    return EnsemblePrediction(
        median=p_cal,
        raw_median=p_raw,
        ridge=p_r,
        theta=p_t,
        quantiles=q,
        proba=proba,
        sigma_h=s,
        catboost=p_cb,
        xgboost=p_xg,
        mlp_plr=p_ml,
    )


# ---------------------------------------------------------------------------
# Live outlook assembly
# ---------------------------------------------------------------------------


def _last_finite(values: np.ndarray) -> Optional[float]:
    finite = np.where(np.isfinite(values))[0]
    return float(values[finite[-1]]) if finite.size else None


def _round(x: float, nd: int = 6) -> float:
    return float(round(float(x), nd))


def _driver_items(
    fit: EnsembleFit,
    x_row: np.ndarray,
    sigma_h: float,
    params: LongTermModelParams,
    horizon_weeks: int,
    top_k: int = 5,
) -> list[LongTermDriverItem]:
    contrib_z = fit.ridge.contributions_z(x_row)
    if contrib_z.size == 0:
        return []
    z_scores = fit.ridge.standardize(x_row)[0]
    contrib_ret = contrib_z * sigma_h * params.effective_ridge_weight(horizon_weeks)
    order = np.argsort(-np.abs(contrib_ret))[:top_k]
    items: list[LongTermDriverItem] = []
    for j in order:
        feat = fit.ridge.features[j]
        meta = FEATURE_CATALOG.get(feat, {"label": feat, "pillar": "trend_structure"})
        c = float(contrib_ret[j])
        items.append(
            LongTermDriverItem(
                feature=feat,
                label=meta["label"],
                pillar=meta["pillar"],
                z_score=_round(z_scores[j], 4),
                contribution_return=_round(c, 6),
                direction="BULLISH" if c > 1e-6 else ("BEARISH" if c < -1e-6 else "NEUTRAL"),
            )
        )
    return items


def _target_date(as_of: str, horizon_days: int) -> Optional[str]:
    try:
        base = pd.Timestamp(datetime.strptime(as_of[:10], "%Y-%m-%d"))
        return (base + pd.offsets.BDay(horizon_days)).strftime("%Y-%m-%d")
    except Exception:
        return None


def forecast_horizon(
    panel: pd.DataFrame,
    spec: HorizonSpec,
    features: list[str],
    theta: ThetaComponents,
    as_of_date: str,
    params: LongTermModelParams = DEFAULT_PARAMS,
    oof_residuals: Optional[np.ndarray] = None,
    oof_shrinkage_alpha: Optional[float] = None,
) -> Optional[LongTermHorizonForecast]:
    """Fit on all realized targets and forecast from the latest causal row."""
    frame = build_horizon_frame(panel, spec, features)
    train_pos = np.where(frame.trainable)[0]
    if train_pos.size < params.min_train_rows:
        return None
    fit = fit_ensemble(
        frame,
        train_pos,
        theta,
        params,
        conformal_residuals=oof_residuals,
        shrinkage_alpha=oof_shrinkage_alpha,
    )

    last = len(panel) - 1
    sigma_w_last = _last_finite(pd.to_numeric(panel["sigma_w"], errors="coerce").to_numpy(dtype=np.float64))
    ref_price = _last_finite(pd.to_numeric(panel["target_price"], errors="coerce").to_numpy(dtype=np.float64))
    if sigma_w_last is None or ref_price is None or ref_price <= 0.0:
        return None
    sigma_h = sigma_w_last * math.sqrt(spec.weeks)
    frame.sigma_h[last] = sigma_h  # guard against NaN vol on a partial final week

    pred = predict_ensemble(fit, frame, np.array([last]), theta, params)
    q = pred.quantiles[0]
    p_bear, p_neut, p_bull = (float(v) for v in pred.proba[0])

    if params.enable_committee:
        rating, stance, conviction, conviction_label = institutional_rating_from_forecast(
            float(q[2]), sigma_h, p_bear, p_bull
        )
    else:
        stance, conviction, conviction_label = stance_from_proba(p_bear, p_neut, p_bull, params.stance_edge)
        rating, _, _, _ = institutional_rating_from_forecast(float(q[2]), sigma_h, p_bear, p_bull)

    prob_bull = _round(p_bull, 4)
    prob_bear = _round(p_bear, 4)
    prob_neut = _round(max(0.0, 1.0 - prob_bull - prob_bear), 4)

    prices = [ref_price * math.exp(v) for v in q]
    simple = [math.exp(v) - 1.0 for v in q]
    median_log = float(q[2])
    hw_lo80 = float(q[2] - q[0])
    hw_hi80 = float(q[4] - q[2])
    asymmetry_ratio = _round((prices[4] - ref_price) / max(ref_price - prices[0], 1e-4), 4)

    w_cb, w_xg, w_ml, w_th = params.normalized_committee_weights()
    has_comm = pred.catboost is not None and pred.xgboost is not None and pred.mlp_plr is not None

    return LongTermHorizonForecast(
        horizon_key=spec.key,  # type: ignore[arg-type]
        horizon_label=spec.label,
        horizon_days=spec.days,
        horizon_weeks=spec.weeks,
        target_date=_target_date(as_of_date, spec.days),
        reference_price=_round(ref_price, 4),
        target_price_q50=_round(prices[2], 4),
        expected_return=_round(simple[2], 6),
        expected_return_pct=_round(simple[2] * 100.0, 4),
        expected_log_return=_round(median_log, 6),
        price_q10=_round(prices[0], 4),
        price_q25=_round(prices[1], 4),
        price_q75=_round(prices[3], 4),
        price_q90=_round(prices[4], 4),
        return_q10=_round(simple[0], 6),
        return_q25=_round(simple[1], 6),
        return_q75=_round(simple[3], 6),
        return_q90=_round(simple[4], 6),
        stance=stance,
        institutional_rating=rating,
        asymmetry_ratio=asymmetry_ratio,
        prob_bullish=prob_bull,
        prob_neutral=prob_neut,
        prob_bearish=prob_bear,
        conviction=_round(conviction, 4),
        conviction_label=conviction_label,  # type: ignore[arg-type]
        regime_threshold_pct=_round((math.exp(params.regime_k * sigma_h) - 1.0) * 100.0, 4),
        components=LongTermEnsembleComponents(
            macro_ridge_return=_round(math.exp(float(pred.ridge[0])) - 1.0, 6),
            theta_damped_return=_round(math.exp(float(pred.theta[0])) - 1.0, 6),
            macro_ridge_weight=params.effective_ridge_weight(spec.weeks),
            theta_weight=_round(w_th if has_comm else (1.0 - params.effective_ridge_weight(spec.weeks)), 4),
            realized_vol_annualized=_round(sigma_w_last * math.sqrt(52.0), 6),
            horizon_vol=_round(sigma_h, 6),
            cqr_q10_width=_round(hw_lo80, 6),
            cqr_q90_width=_round(hw_hi80, 6),
            catboost_return=_round(math.exp(float(pred.catboost[0])) - 1.0, 6) if pred.catboost is not None else None,
            xgboost_return=_round(math.exp(float(pred.xgboost[0])) - 1.0, 6) if pred.xgboost is not None else None,
            mlp_plr_return=_round(math.exp(float(pred.mlp_plr[0])) - 1.0, 6) if pred.mlp_plr is not None else None,
            tree_weight=_round(w_cb + w_xg, 4) if has_comm else None,
            mlp_weight=_round(w_ml, 4) if has_comm else None,
            shrinkage_alpha=_round(fit.shrinkage_alpha, 4),
        ),
        key_drivers=_driver_items(fit, frame.X[last], sigma_h, params, spec.weeks),
    )


def fit_and_predict_long_term_outlook(
    panel_bundle: "LongTermPanelBundle",
    validation: Optional[Mapping[str, "WalkForwardHorizonResult"]] = None,
    model_version: str = "lt_committee_v2",
    symbol: str = "HG=F",
    params: LongTermModelParams = DEFAULT_PARAMS,
    generated_at: Optional[str] = None,
) -> LongTermOutlookResponse:
    """Assemble the full 20D/60D/120D outlook payload from a weekly panel + walk-forward results."""
    panel = panel_bundle.panel
    if panel is None or panel.empty:
        raise ValueError("Empty long-term panel")
    features = select_model_features(panel, params)
    theta = compute_theta_components(
        panel["target_log_price"].to_numpy(dtype=np.float64), lookback=params.theta_lookback
    )

    horizons: dict[str, LongTermHorizonForecast] = {}
    for key, spec in HORIZONS.items():
        oof = None
        oof_alpha = None
        if validation and key in validation:
            oof = getattr(validation[key], "oof_abs_norm_residuals", None)
            oof_alpha = getattr(validation[key], "oof_shrinkage_alpha", None)
        fc = forecast_horizon(
            panel,
            spec,
            features,
            theta,
            panel_bundle.as_of_date,
            params,
            oof_residuals=oof,
            oof_shrinkage_alpha=oof_alpha,
        )
        if fc is not None:
            horizons[key] = fc
    if not horizons:
        raise ValueError("Insufficient weekly history to fit any long-term horizon")

    validation_summaries: dict[str, LongTermValidationSummary] = {}
    for key, res in (validation or {}).items():
        summary = getattr(res, "summary", None)
        if isinstance(summary, LongTermValidationSummary):
            validation_summaries[key] = summary

    current_price = _last_finite(pd.to_numeric(panel["target_price"], errors="coerce").to_numpy(dtype=np.float64))
    default_h = "60D" if "60D" in horizons else ("20D" if "20D" in horizons else next(iter(horizons)))
    return LongTermOutlookResponse(
        status="ok",
        symbol=symbol,
        as_of_date=panel_bundle.as_of_date,
        generated_at=generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        model_version=model_version,
        source="snapshot",
        current_price=_round(current_price, 4) if current_price is not None else None,
        available_horizons=list(horizons.keys()),
        default_horizon=default_h,
        data_window=LongTermDataWindow(
            start_date=panel_bundle.start_date,
            end_date=panel_bundle.end_date,
            weekly_bars=panel_bundle.weekly_bars,
            years_covered=panel_bundle.years_covered,
            fred_macro_enabled=panel_bundle.fred_macro_enabled,
            physical_data_enabled=getattr(panel_bundle, "physical_data_enabled", False),
            cot_positioning_enabled=getattr(panel_bundle, "cot_positioning_enabled", False),
            symbols_used=list(panel_bundle.symbols_used),
        ),
        horizons=horizons,
        macro_scorecard=compute_macro_scorecard(panel),
        validation=validation_summaries,
        message=None,
    )
