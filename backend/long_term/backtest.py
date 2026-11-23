"""Purged, horizon-matched expanding-origin walk-forward validation (20D / 60D)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from long_term.model import (
    DEFAULT_PARAMS,
    HORIZONS,
    QUANTILE_LEVELS,
    HorizonFrame,
    HorizonSpec,
    LongTermModelParams,
    ThetaComponents,
    build_horizon_frame,
    compute_theta_components,
    fit_ensemble,
    predict_ensemble,
    regime_classes,
    select_model_features,
)
from long_term.schemas import LongTermFoldMetrics, LongTermValidationSummary


@dataclass
class WalkForwardHorizonResult:
    summary: LongTermValidationSummary
    oof_abs_norm_residuals: np.ndarray
    oof_frame: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Splits & metrics
# ---------------------------------------------------------------------------


def purged_walk_forward_splits(
    valid_positions: np.ndarray,
    horizon_weeks: int,
    *,
    n_folds: int = 5,
    min_train: int = 156,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-origin folds with a purge of ``horizon_weeks``.

    Every training origin ``t`` satisfies ``t + H <= first_test_origin`` so its
    forward target window is fully realized before the first test forecast is made.
    """
    valid = np.sort(np.asarray(valid_positions, dtype=int))
    if valid.size <= min_train + n_folds:
        return []
    test_pool = valid[min_train:]
    chunks = [c for c in np.array_split(test_pool, n_folds) if c.size]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for te in chunks:
        tr = valid[valid <= te[0] - horizon_weeks]
        if tr.size:
            splits.append((tr, te))
    return splits


def sharpe_hac(returns: np.ndarray, annualisation: float, max_lag: int) -> float:
    """Newey-West (Bartlett) autocorrelation-adjusted annualized Sharpe ratio."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return 0.0
    mu = r.mean()
    d = r - mu
    lrv = float(d @ d) / n
    for lag in range(1, min(max_lag, n - 1) + 1):
        lrv += 2.0 * (1.0 - lag / (max_lag + 1.0)) * float(d[lag:] @ d[:-lag]) / n
    if lrv <= 1e-18:
        return 0.0
    return float(math.sqrt(annualisation) * mu / math.sqrt(lrv))


def simple_sharpe(returns: np.ndarray, annualisation: float) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    sd = r.std()
    return float(math.sqrt(annualisation) * r.mean() / sd) if sd > 1e-12 else 0.0


def directional_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.sign(a) == np.sign(p)))


def pinball_loss(actual: np.ndarray, quantiles: np.ndarray, levels: Iterable[float] = QUANTILE_LEVELS) -> float:
    a = np.asarray(actual, dtype=np.float64)[:, None]
    q = np.asarray(quantiles, dtype=np.float64)
    tau = np.asarray(list(levels), dtype=np.float64)[None, :]
    diff = a - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def quantile_crossing_rate(quantiles: np.ndarray, eps: float = 1e-12) -> float:
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 2 or q.shape[0] == 0:
        return 0.0
    return float(np.mean(np.diff(q, axis=1) < -eps))


def brier_multiclass(proba: np.ndarray, classes: np.ndarray) -> float:
    p = np.asarray(proba, dtype=np.float64)
    if p.size == 0:
        return 0.0
    onehot = np.eye(p.shape[1])[np.asarray(classes, dtype=int)]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def _coverage(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    return float(np.mean((actual >= lo) & (actual <= hi)))


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 1e-12 else 1.0


def _non_overlapping(n: int, horizon_weeks: int) -> np.ndarray:
    return np.arange(0, n, max(1, horizon_weeks))


def _fmt_date(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------


def run_horizon_walk_forward(
    frame: HorizonFrame,
    theta: ThetaComponents,
    *,
    params: LongTermModelParams = DEFAULT_PARAMS,
    n_folds: int = 5,
    min_train_weeks: Optional[int] = None,
) -> Optional[WalkForwardHorizonResult]:
    spec: HorizonSpec = frame.spec
    H = spec.weeks
    valid = np.where(frame.trainable)[0]
    min_train = (
        int(min_train_weeks)
        if min_train_weeks is not None
        else int(max(params.min_train_rows + H, min(208, valid.size // 2)))
    )
    splits = purged_walk_forward_splits(valid, H, n_folds=n_folds, min_train=min_train)
    if not splits:
        return None

    records: list[pd.DataFrame] = []
    folds: list[LongTermFoldMetrics] = []
    for fold_idx, (tr, te) in enumerate(splits):
        if tr.size < params.min_train_rows:
            continue
        fit = fit_ensemble(frame, tr, theta, params)
        pred = predict_ensemble(fit, frame, te, theta, params)
        y = frame.y[te]
        s = frame.sigma_h[te]
        cls = regime_classes(y, s, params.regime_k)
        emp_window = frame.y[tr[-params.ridge_window :]]
        emp_q = np.quantile(emp_window, QUANTILE_LEVELS)
        prior = np.bincount(regime_classes(frame.y[tr], frame.sigma_h[tr], params.regime_k), minlength=3)
        prior = (prior + 1.0) / (prior.sum() + 3.0)

        df = pd.DataFrame(
            {
                "pos": te,
                "date": frame.index[te],
                "fold": fold_idx,
                "y": y,
                "sigma_h": s,
                "pred": pred.median,
                "pred_ridge": pred.ridge,
                "pred_theta": pred.theta,
                "q10": pred.quantiles[:, 0],
                "q25": pred.quantiles[:, 1],
                "q50": pred.quantiles[:, 2],
                "q75": pred.quantiles[:, 3],
                "q90": pred.quantiles[:, 4],
                "p_bear": pred.proba[:, 0],
                "p_neut": pred.proba[:, 1],
                "p_bull": pred.proba[:, 2],
                "cls": cls,
                "prior_bear": prior[0],
                "prior_neut": prior[1],
                "prior_bull": prior[2],
            }
        )
        for j, lvl in enumerate(QUANTILE_LEVELS):
            df[f"emp_q{int(lvl * 100)}"] = emp_q[j]
        records.append(df)

        no = _non_overlapping(len(df), H)
        strat = np.sign(df["pred"].to_numpy()) * y
        q_arr = df[["q10", "q25", "q50", "q75", "q90"]].to_numpy()
        folds.append(
            LongTermFoldMetrics(
                fold_index=fold_idx,
                train_start=_fmt_date(frame.index[tr[0]]),
                train_end=_fmt_date(frame.index[tr[-1]]),
                test_start=_fmt_date(frame.index[te[0]]),
                test_end=_fmt_date(frame.index[te[-1]]),
                n_train=int(tr.size),
                n_test=int(te.size),
                n_non_overlapping=int(no.size),
                directional_accuracy=round(directional_accuracy(y, df["pred"].to_numpy()), 4),
                non_overlapping_da=round(directional_accuracy(y[no], df["pred"].to_numpy()[no]), 4),
                mae_vs_naive_zero=round(_safe_ratio(np.mean(np.abs(y - df["pred"])), np.mean(np.abs(y))), 4),
                mae_vs_theta=round(
                    _safe_ratio(np.mean(np.abs(y - df["pred"])), np.mean(np.abs(y - df["pred_theta"]))), 4
                ),
                sharpe_ratio=round(sharpe_hac(strat, spec.annualisation, H - 1), 4),
                pi80_coverage=round(_coverage(y, q_arr[:, 0], q_arr[:, 4]), 4),
                brier_score=round(
                    brier_multiclass(df[["p_bear", "p_neut", "p_bull"]].to_numpy(), cls), 4
                ),
            )
        )

    if not records:
        return None
    oof = pd.concat(records, ignore_index=True).sort_values("pos").reset_index(drop=True)
    y = oof["y"].to_numpy()
    p = oof["pred"].to_numpy()
    p_theta = oof["pred_theta"].to_numpy()
    q_arr = oof[["q10", "q25", "q50", "q75", "q90"]].to_numpy()
    emp_arr = oof[[f"emp_q{int(l * 100)}" for l in QUANTILE_LEVELS]].to_numpy()
    proba = oof[["p_bear", "p_neut", "p_bull"]].to_numpy()
    prior = oof[["prior_bear", "prior_neut", "prior_bull"]].to_numpy()
    cls = oof["cls"].to_numpy(dtype=int)
    no = _non_overlapping(len(oof), H)
    strat = np.sign(p) * y

    mae = float(np.mean(np.abs(y - p)))
    pinball_model = pinball_loss(y, q_arr)
    pinball_emp = pinball_loss(y, emp_arr)

    summary = LongTermValidationSummary(
        horizon_key=spec.key,  # type: ignore[arg-type]
        horizon_weeks=H,
        purge_gap_weeks=H,
        annualisation=spec.annualisation,
        n_folds=len(folds),
        n_purged_oos_samples=int(len(oof)),
        n_non_overlapping_samples=int(no.size),
        directional_accuracy=round(directional_accuracy(y, p), 4),
        non_overlapping_da=round(directional_accuracy(y[no], p[no]), 4),
        mae_vs_naive_zero=round(_safe_ratio(mae, float(np.mean(np.abs(y)))), 4),
        mae_vs_theta=round(_safe_ratio(mae, float(np.mean(np.abs(y - p_theta)))), 4),
        theta_directional_accuracy=round(directional_accuracy(y, p_theta), 4),
        sharpe_ratio=round(sharpe_hac(strat, spec.annualisation, H - 1), 4),
        non_overlapping_sharpe=round(simple_sharpe(strat[no], spec.annualisation), 4),
        pi80_coverage=round(_coverage(y, q_arr[:, 0], q_arr[:, 4]), 4),
        pi50_coverage=round(_coverage(y, q_arr[:, 1], q_arr[:, 3]), 4),
        pinball_loss_mean=round(pinball_model, 6),
        pinball_skill_vs_empirical=round(1.0 - _safe_ratio(pinball_model, pinball_emp), 4),
        brier_score=round(brier_multiclass(proba, cls), 4),
        brier_score_prior_baseline=round(brier_multiclass(prior, cls), 4),
        regime_accuracy=round(float(np.mean(np.argmax(proba, axis=1) == cls)), 4),
        quantile_crossing_rate=round(quantile_crossing_rate(q_arr), 6),
        folds=folds,
    )
    abs_norm_res = np.abs(y - p) / oof["sigma_h"].to_numpy()
    return WalkForwardHorizonResult(summary=summary, oof_abs_norm_residuals=abs_norm_res, oof_frame=oof)


def run_purged_walk_forward_suite(
    panel: pd.DataFrame,
    *,
    horizons: Iterable[str] = ("20D", "60D"),
    n_folds: int = 5,
    min_train_weeks: Optional[int] = None,
    params: LongTermModelParams = DEFAULT_PARAMS,
) -> dict[str, WalkForwardHorizonResult]:
    """Run purged expanding-origin walk-forward validation for each requested horizon."""
    if panel is None or panel.empty:
        return {}
    features = select_model_features(panel, params)
    theta = compute_theta_components(
        panel["target_log_price"].to_numpy(dtype=np.float64), lookback=params.theta_lookback
    )
    results: dict[str, WalkForwardHorizonResult] = {}
    for key in horizons:
        spec = HORIZONS.get(key)
        if spec is None:
            continue
        frame = build_horizon_frame(panel, spec, features)
        res = run_horizon_walk_forward(
            frame, theta, params=params, n_folds=n_folds, min_train_weeks=min_train_weeks
        )
        if res is not None:
            results[key] = res
    return results
