"""Unit tests for the causal HAR-GJR forward volatility and market stress engine."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from deep_learning.calibration.conformal import apply_weekly_cqr, fit_weekly_cqr
from deep_learning.calibration.volatility_stress import (
    build_har_gjr_features,
    fit_har_gjr_volatility_stress,
    is_valid_har_gjr_params,
    predict_har_gjr_volatility_stress,
    predict_latest_volatility_stress,
    resolve_cqr_vol_scale,
)


def _make_synthetic_master_df(n: int = 320, *, inject_tail_shock: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    idx = pd.bdate_range("2025-01-01", periods=n)
    rets = rng.normal(0.0, 0.008, size=n)
    # Create realistic persistent volatility / downside shock clusters in training
    for start in (40, 80, 130, 175):
        if n > start + 15:
            rets[start : start + 12] = np.array(
                [-0.032, -0.025, 0.015, -0.029, -0.021, 0.018, -0.034, -0.022, 0.014, -0.028, -0.019, 0.012]
            )
    if inject_tail_shock and n >= 30:
        # Calm 20d window followed by an abrupt 2-day downside break (HAR-GJR spikes faster than RV20)
        rets[-25:-2] = rng.normal(0.0, 0.004, size=23)
        rets[-2:] = np.array([-0.045, -0.039])
    close = 4.20 * np.exp(np.cumsum(rets))
    log_c = np.log(close)
    target = np.concatenate([[0.0], np.diff(log_c)])
    rv20 = (
        pd.Series(target)
        .rolling(20, min_periods=5)
        .std()
        .ffill()
        .fillna(0.011)
        .to_numpy(dtype=np.float64)
    )
    target_5d = (pd.Series(log_c).shift(-5) - pd.Series(log_c)).to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "target_close": close,
            "target": target,
            "observed_return_1d": target,
            "realized_vol_20d": rv20,
            "HG=F_ret5": pd.Series(log_c).diff(5).fillna(0.0).to_numpy(dtype=np.float64),
            "HG=F_dd_60": (
                close / pd.Series(close).rolling(60, min_periods=1).max().to_numpy(dtype=np.float64)
            ) - 1.0,
            "HG=F_rsi_14_c": rng.normal(0.0, 0.25, size=n),
            "DX-Y.NYB_ret5": rng.normal(0.0, 0.01, size=n),
            "CL=F_ret5": rng.normal(0.0, 0.02, size=n),
            "target_5d_log_return": target_5d,
        },
        index=idx,
    )


def test_build_har_gjr_features_is_strictly_causal():
    df_base = _make_synthetic_master_df(180)
    df_mutated = df_base.copy()
    cutoff = 120
    # Mutate all future rows after cutoff violently
    df_mutated.iloc[cutoff + 1 :, df_mutated.columns.get_loc("observed_return_1d")] = -0.15
    df_mutated.iloc[cutoff + 1 :, df_mutated.columns.get_loc("realized_vol_20d")] = 0.09
    df_mutated.iloc[cutoff + 1 :, df_mutated.columns.get_loc("target_close")] = 1.50
    df_mutated.iloc[cutoff + 1 :, df_mutated.columns.get_loc("HG=F_ret5")] = -0.30

    mat_a, _, rv_a = build_har_gjr_features(df_base)
    mat_b, _, rv_b = build_har_gjr_features(df_mutated)

    np.testing.assert_allclose(mat_a[: cutoff + 1], mat_b[: cutoff + 1], atol=1e-12)
    np.testing.assert_allclose(rv_a[: cutoff + 1], rv_b[: cutoff + 1], atol=1e-12)


def test_fit_json_roundtrip_and_predict_identity():
    df = _make_synthetic_master_df(320)
    val_pos = np.arange(230, 270)
    params = fit_har_gjr_volatility_stress(
        df,
        train_end_pos=220,
        val_origin_pos=val_pos,
        horizon=5,
    )
    assert params["enabled"] is True
    assert is_valid_har_gjr_params(params)

    roundtrip = json.loads(json.dumps(params))
    assert is_valid_har_gjr_params(roundtrip)

    out_orig = predict_har_gjr_volatility_stress(df, params, positions=np.arange(275, 315))
    out_rt = predict_har_gjr_volatility_stress(df, roundtrip, positions=np.arange(275, 315))

    np.testing.assert_allclose(out_orig["cqr_vol_scale"], out_rt["cqr_vol_scale"], atol=1e-12)
    np.testing.assert_allclose(out_orig["hat_vol_5d"], out_rt["hat_vol_5d"], atol=1e-12)
    np.testing.assert_allclose(out_orig["p_shock"], out_rt["p_shock"], atol=1e-12)
    assert set(out_rt["regime"]).issubset({"CALM", "ELEVATED", "HIGH_STRESS"})


def test_fail_open_on_missing_or_disabled_har_gjr():
    df = _make_synthetic_master_df(120)
    fallback = df["realized_vol_20d"].iloc[-10:].to_numpy(dtype=np.float64)
    pos = np.arange(len(df) - 10, len(df))

    for bad_params in (None, {}, {"enabled": False}, {"enabled": True, "feature_names": ["x"]}):
        out = resolve_cqr_vol_scale(df, pos, har_gjr_params=bad_params, fallback_vol=fallback)
        np.testing.assert_allclose(out, fallback)
        vol, stress_payload = predict_latest_volatility_stress(df, bad_params)
        assert vol is None
        assert stress_payload is None


def test_stress_signal_elevated_and_cqr_preserves_monotonicity():
    df = _make_synthetic_master_df(320, inject_tail_shock=True)
    params = fit_har_gjr_volatility_stress(
        df,
        train_end_pos=220,
        val_origin_pos=np.arange(230, 270),
        horizon=5,
    )
    latest_vol, stress = predict_latest_volatility_stress(df, params)
    assert latest_vol is not None and latest_vol > 0.0
    assert stress is not None
    assert stress["enabled"] is True
    assert stress["vol_scale_source"] == "har_gjr"
    assert stress["regime"] in {"ELEVATED", "HIGH_STRESS"}

    # Fit CQR on validation using HAR-GJR scale and verify widening + strict ordering on live tail
    rng = np.random.default_rng(7)
    val_q = np.tile(np.array([-0.04, -0.025, -0.012, 0.0, 0.012, 0.025, 0.04]), (40, 1))
    val_y = rng.normal(0.0, 0.02, size=40)
    val_scales = predict_har_gjr_volatility_stress(df, params, positions=np.arange(230, 270))["cqr_vol_scale"]
    cqr = fit_weekly_cqr(val_y, val_q, realized_vol=val_scales)

    calm_scale = float(np.min(val_scales) * 0.7)
    raw_single = np.array([-0.04, -0.025, -0.012, 0.004, 0.016, 0.029, 0.044])
    cal_calm = apply_weekly_cqr(raw_single, cqr_params=cqr, realized_vol=calm_scale)
    cal_stress = apply_weekly_cqr(raw_single, cqr_params=cqr, realized_vol=latest_vol)

    assert np.all(np.diff(cal_stress) > 0.0)
    assert np.isclose(cal_stress[3], raw_single[3], atol=1e-12)
    assert (cal_stress[5] - cal_stress[1]) >= (cal_calm[5] - cal_calm[1])
