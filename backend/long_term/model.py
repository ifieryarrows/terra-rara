"""Parsimonious structural ensemble for the 20D / 60D long-term copper outlook.

Components (all low-variance, fit strictly on purged historical windows):

1. **Standardized Ridge macro-trend model** on volatility-normalized forward
   log returns ``z_H = r_{t->t+H} / (sigma_w,t * sqrt(H))`` (rolling window).
2. **Damped-trend Theta model** (Theta-0 / Theta-2 decomposition with a
   damped linear drift) on weekly log prices.
3. **Regime probabilities**: a heavily L2-regularized multinomial logistic
   classifier over standardized factors, blended with the Gaussian-implied
   class probabilities of the ensemble forecast. Classes are defined by
   economically meaningful moves (``|r| > 0.5 * sigma_H``), not noise.
4. **Purged split-conformal intervals** on volatility-normalized absolute
   residuals, scaled back by the current realized weekly volatility.

No component reads future information: features/volatility are trailing,
training targets are purged by the horizon before every evaluation origin.
"""

from __future__ import annotations

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
}

# Stationary, scale-free factor basis (one or two per economic pillar).
MODEL_FEATURES: tuple[str, ...] = (
    "hg_sma10w_ratio",
    "hg_sma40w_ratio",
    "hg_dd52",
    "dxy_ret26",
    "real_yield_chg26",
    "breakeven_10y_z",
    "fxa_ret26",
    "cl_ret12",
    "miners_vs_hg_12w",
    "copx_vs_hg_12w",
)

QUANTILE_LEVELS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass(frozen=True)
class LongTermModelParams:
    ridge_alpha: float = 3000.0
    ridge_window: int = 260  # ~5 years of weekly origins
    z_clip: float = 3.0
    x_clip: float = 3.0
    theta_lookback: int = 64
    theta_phi: float = 0.82
    ridge_weight: float = 0.55
    logit_C: float = 0.002
    logit_weight: float = 0.25
    regime_k: float = 0.5  # class threshold in units of sigma_H
    min_z_scale: float = 0.85
    min_feature_coverage: float = 0.70
    min_train_rows: int = 80
    stance_edge: float = 0.10


    def effective_ridge_alpha(self, horizon_weeks: int) -> float:
        """Scale L2 penalty for reduced effective independent N of overlapping horizons."""
        return float(self.ridge_alpha * max(1.0, horizon_weeks / 4.0) ** 0.8)

    def effective_ridge_weight(self, horizon_weeks: int) -> float:
        """Balance macro-trend Ridge with Damped Theta (Theta excels at longer horizons)."""
        if horizon_weeks >= 12:
            return min(self.ridge_weight, 0.50)
        return self.ridge_weight


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
    X: np.ndarray  # (n, k) raw feature matrix (NaN allowed)
    y: np.ndarray  # forward log return (NaN where unknown)
    sigma_h: np.ndarray  # causal horizon volatility
    feature_ok: np.ndarray  # bool: all features finite and sigma finite
    trainable: np.ndarray  # bool: feature_ok & target known
    index: pd.DatetimeIndex


def build_horizon_frame(panel: pd.DataFrame, spec: HorizonSpec, features: list[str]) -> HorizonFrame:
    X = panel[features].to_numpy(dtype=np.float64) if features else np.zeros((len(panel), 0))
    y = pd.to_numeric(panel[spec.target_col], errors="coerce").to_numpy(dtype=np.float64)
    sigma_w = pd.to_numeric(panel["sigma_w"], errors="coerce").to_numpy(dtype=np.float64)
    sigma_h = sigma_w * math.sqrt(spec.weeks)
    feature_ok = np.isfinite(sigma_h) & (sigma_h > 0.0)
    if X.shape[1]:
        feature_ok &= np.all(np.isfinite(X), axis=1)
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
        return np.where(np.isfinite(out), out, 0.0)


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
# Ridge macro-trend model
# ---------------------------------------------------------------------------


@dataclass
class FittedRidge:
    scaler: StandardScaler
    model: Ridge
    features: list[str]
    x_clip: float

    def standardize(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(np.atleast_2d(X))
        Xs = np.where(np.isfinite(Xs), Xs, 0.0)  # missing live factor -> neutral (train mean)
        return np.clip(Xs, -self.x_clip, self.x_clip)

    def predict_z(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.standardize(X))

    def contributions_z(self, x_row: np.ndarray) -> np.ndarray:
        return self.standardize(x_row)[0] * self.model.coef_


def fit_ridge(
    X: np.ndarray,
    z: np.ndarray,
    features: list[str],
    params: LongTermModelParams,
    horizon_weeks: int = 4,
) -> FittedRidge:
    scaler = StandardScaler().fit(X)
    scale = scaler.scale_.copy()
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    scaler.scale_ = scale
    Xs = np.clip(scaler.transform(X), -params.x_clip, params.x_clip)
    model = Ridge(alpha=params.effective_ridge_alpha(horizon_weeks), fit_intercept=False)
    model.fit(Xs, np.clip(z, -params.z_clip, params.z_clip))
    return FittedRidge(scaler, model, list(features), params.x_clip)


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------


def regime_classes(y: np.ndarray, sigma_h: np.ndarray, k: float) -> np.ndarray:
    """0 = BEARISH (r < -k sigma), 1 = NEUTRAL, 2 = BULLISH (r > k sigma)."""
    thr = k * sigma_h
    return np.where(y > thr, 2, np.where(y < -thr, 0, 1)).astype(int)


@dataclass
class FittedRegime:
    scaler: Optional[StandardScaler]
    clf: Optional[LogisticRegression]
    prior: np.ndarray
    x_clip: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X2 = np.atleast_2d(X)
        if self.clf is None or self.scaler is None:
            return np.repeat(self.prior[None, :], X2.shape[0], axis=0)
        Xs = self.scaler.transform(X2)
        Xs = np.clip(np.where(np.isfinite(Xs), Xs, 0.0), -self.x_clip, self.x_clip)
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
    if len(np.unique(classes)) < 2 or X.shape[1] == 0:
        return FittedRegime(None, None, prior, params.x_clip)
    scaler = StandardScaler().fit(X)
    scale = scaler.scale_.copy()
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    scaler.scale_ = scale
    Xs = np.clip(scaler.transform(X), -params.x_clip, params.x_clip)
    clf = LogisticRegression(C=params.logit_C, max_iter=1000)
    clf.fit(Xs, classes)
    return FittedRegime(scaler, clf, prior, params.x_clip)


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


def stance_from_proba(p_bear: float, p_neut: float, p_bull: float, edge: float) -> tuple[StanceType, float, str]:
    tilt = float(p_bull - p_bear)
    stance: StanceType = "BULLISH" if tilt >= edge else ("BEARISH" if tilt <= -edge else "NEUTRAL")
    conviction = float(np.clip(abs(tilt) * 2.0, 0.0, 1.0))
    label = "HIGH" if conviction >= 0.5 else ("MEDIUM" if conviction >= 0.2 else "LOW")
    return stance, conviction, label


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
    extras: dict[str, Any] = field(default_factory=dict)


def inner_purged_conformal_residuals(
    frame: HorizonFrame,
    train_pos: np.ndarray,
    theta: ThetaComponents,
    params: LongTermModelParams,
) -> np.ndarray:
    """Abs normalized residuals on a purged calibration tail of the training slice."""
    H = frame.spec.weeks
    if train_pos.size < params.min_train_rows + 40:
        return np.array([])
    cal_n = int(np.clip(0.30 * train_pos.size, 40, 156))
    cal = train_pos[-cal_n:]
    fit_sub = train_pos[train_pos <= cal[0] - H][-params.ridge_window :]
    if fit_sub.size < 60:
        return np.array([])
    z_fit = frame.y[fit_sub] / frame.sigma_h[fit_sub]
    ridge = fit_ridge(frame.X[fit_sub], z_fit, frame.features, params, horizon_weeks=H)
    p_r = ridge.predict_z(frame.X[cal]) * frame.sigma_h[cal]
    p_t = theta.forecast_log_return(cal, H, params.theta_phi)
    w_r = params.effective_ridge_weight(H)
    p = w_r * p_r + (1.0 - w_r) * p_t
    return np.abs(frame.y[cal] - p) / frame.sigma_h[cal]


def fit_ensemble(
    frame: HorizonFrame,
    train_pos: np.ndarray,
    theta: ThetaComponents,
    params: LongTermModelParams,
    conformal_residuals: Optional[np.ndarray] = None,
) -> EnsembleFit:
    """Fit all components on purged training origins ``train_pos`` (targets fully realized)."""
    if train_pos.size < params.min_train_rows:
        raise ValueError(f"Insufficient training rows ({train_pos.size}) for {frame.spec.key}")
    z_all = frame.y[train_pos] / frame.sigma_h[train_pos]
    roll = train_pos[-params.ridge_window :]
    ridge = fit_ridge(
        frame.X[roll],
        frame.y[roll] / frame.sigma_h[roll],
        frame.features,
        params,
        horizon_weeks=frame.spec.weeks,
    )
    classes = regime_classes(frame.y[train_pos], frame.sigma_h[train_pos], params.regime_k)
    regime = fit_regime_classifier(frame.X[train_pos], classes, params)
    z_scale = max(params.min_z_scale, float(np.nanstd(z_all)))
    residuals = (
        conformal_residuals
        if conformal_residuals is not None and np.isfinite(conformal_residuals).sum() >= 20
        else inner_purged_conformal_residuals(frame, train_pos, theta, params)
    )
    hw50, hw80 = conformal_halfwidths(residuals, fallback_scale=z_scale)
    return EnsembleFit(ridge, regime, z_scale, hw50, hw80, int(train_pos.size))


@dataclass
class EnsemblePrediction:
    median: np.ndarray
    ridge: np.ndarray
    theta: np.ndarray
    quantiles: np.ndarray  # (n, 5) log returns
    proba: np.ndarray  # (n, 3) [bear, neutral, bull]
    sigma_h: np.ndarray


def predict_ensemble(
    fit: EnsembleFit,
    frame: HorizonFrame,
    positions: np.ndarray,
    theta: ThetaComponents,
    params: LongTermModelParams,
) -> EnsemblePrediction:
    pos = np.asarray(positions, dtype=int)
    s = frame.sigma_h[pos]
    p_r = fit.ridge.predict_z(frame.X[pos]) * s
    p_t = theta.forecast_log_return(pos, frame.spec.weeks, params.theta_phi)
    w_r = params.effective_ridge_weight(frame.spec.weeks)
    p = w_r * p_r + (1.0 - w_r) * p_t
    q = build_return_quantiles(p, s, fit.hw50, fit.hw80)
    p_gauss = gaussian_regime_proba(p / s, fit.z_scale, params.regime_k)
    p_logit = fit.regime.predict_proba(frame.X[pos])
    proba = blend_regime_proba(p_logit, p_gauss, params.logit_weight)
    return EnsemblePrediction(p, p_r, p_t, q, proba, s)


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
) -> Optional[LongTermHorizonForecast]:
    """Fit on all realized targets and forecast from the latest causal row."""
    frame = build_horizon_frame(panel, spec, features)
    train_pos = np.where(frame.trainable)[0]
    if train_pos.size < params.min_train_rows:
        return None
    fit = fit_ensemble(frame, train_pos, theta, params, conformal_residuals=oof_residuals)

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
    stance, conviction, conviction_label = stance_from_proba(p_bear, p_neut, p_bull, params.stance_edge)

    prob_bull = _round(p_bull, 4)
    prob_bear = _round(p_bear, 4)
    prob_neut = _round(1.0 - prob_bull - prob_bear, 4)

    prices = [ref_price * math.exp(v) for v in q]
    simple = [math.exp(v) - 1.0 for v in q]
    median_log = float(q[2])
    hw_lo80 = float(q[2] - q[0])
    hw_hi80 = float(q[4] - q[2])

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
            theta_weight=_round(1.0 - params.effective_ridge_weight(spec.weeks), 4),
            realized_vol_annualized=_round(sigma_w_last * math.sqrt(52.0), 6),
            horizon_vol=_round(sigma_h, 6),
            cqr_q10_width=_round(hw_lo80, 6),
            cqr_q90_width=_round(hw_hi80, 6),
        ),
        key_drivers=_driver_items(fit, frame.X[last], sigma_h, params, spec.weeks),
    )


def fit_and_predict_long_term_outlook(
    panel_bundle: "LongTermPanelBundle",
    validation: Optional[Mapping[str, "WalkForwardHorizonResult"]] = None,
    model_version: str = "lt_structural_v1",
    symbol: str = "HG=F",
    params: LongTermModelParams = DEFAULT_PARAMS,
    generated_at: Optional[str] = None,
) -> LongTermOutlookResponse:
    """Assemble the full 20D/60D outlook payload from a weekly panel + walk-forward results."""
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
        if validation and key in validation:
            oof = getattr(validation[key], "oof_abs_norm_residuals", None)
        fc = forecast_horizon(panel, spec, features, theta, panel_bundle.as_of_date, params, oof)
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
    return LongTermOutlookResponse(
        status="ok",
        symbol=symbol,
        as_of_date=panel_bundle.as_of_date,
        generated_at=generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        model_version=model_version,
        source="snapshot",
        current_price=_round(current_price, 4) if current_price is not None else None,
        available_horizons=list(horizons.keys()),
        default_horizon="20D" if "20D" in horizons else next(iter(horizons)),
        data_window=LongTermDataWindow(
            start_date=panel_bundle.start_date,
            end_date=panel_bundle.end_date,
            weekly_bars=panel_bundle.weekly_bars,
            years_covered=panel_bundle.years_covered,
            fred_macro_enabled=panel_bundle.fred_macro_enabled,
            symbols_used=list(panel_bundle.symbols_used),
        ),
        horizons=horizons,
        macro_scorecard=compute_macro_scorecard(panel),
        validation=validation_summaries,
        message=None,
    )
