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


def test_v2_ohlc_rough_features_strictly_causal_and_v1_backward_compat():
    from deep_learning.calibration.volatility_stress import build_v2_vol_stress_features

    df = _make_synthetic_master_df(200)
    close = df["target_close"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(99)
    df["target_open"] = close * np.exp(rng.normal(0.0, 0.003, size=len(df)))
    df["target_high"] = np.maximum(close, df["target_open"]) * (1.0 + rng.uniform(0.002, 0.012, size=len(df)))
    df["target_low"] = np.minimum(close, df["target_open"]) * (1.0 - rng.uniform(0.002, 0.012, size=len(df)))
    df["target_volume"] = rng.uniform(1000.0, 5000.0, size=len(df))

    cutoff = 130
    df_mut = df.copy()
    for col in ("target_open", "target_high", "target_low", "target_close", "observed_return_1d", "target_volume"):
        df_mut.iloc[cutoff + 1 :, df_mut.columns.get_loc(col)] *= 1.8

    mat_a, _, rv_a, anc_a = build_v2_vol_stress_features(df)
    mat_b, _, rv_b, anc_b = build_v2_vol_stress_features(df_mut)
    np.testing.assert_allclose(mat_a[: cutoff + 1], mat_b[: cutoff + 1], atol=1e-12)
    np.testing.assert_allclose(rv_a[: cutoff + 1], rv_b[: cutoff + 1], atol=1e-12)
    np.testing.assert_allclose(anc_a["ewma_094_var5"][: cutoff + 1], anc_b["ewma_094_var5"][: cutoff + 1], atol=1e-12)

    # Fit V2 engine and verify JSON roundtrip + V2 telemetry fields
    params_v2 = fit_har_gjr_volatility_stress(
        df,
        train_end_pos=140,
        val_origin_pos=np.arange(145, 185),
        horizon=5,
    )
    assert params_v2["version"] == 2
    assert params_v2["enabled"] is True
    rt_v2 = json.loads(json.dumps(params_v2))
    assert is_valid_har_gjr_params(rt_v2)
    _, payload_v2 = predict_latest_volatility_stress(df, rt_v2)
    assert payload_v2 is not None
    assert payload_v2["engine_version"] == "v2_ewma_har_rough"
    assert payload_v2["vol_scale_source"] == "har_gjr"
    assert 0.5 <= float(payload_v2["risk_budget_mult"]) <= 1.0

    # Verify legacy V1 params still run cleanly without error
    from deep_learning.calibration.volatility_stress import FEATURE_NAMES

    d_v1 = len(FEATURE_NAMES)
    v1_params = {
        "version": 1,
        "enabled": True,
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": [0.0] * d_v1,
        "scaler_scale": [1.0] * d_v1,
        "vol_coef": [0.05] * d_v1,
        "vol_intercept": -3.5,
        "vol_scale_ratio": 1.0,
        "shock_enabled": False,
    }
    assert is_valid_har_gjr_params(v1_params)
    v1_vol, v1_payload = predict_latest_volatility_stress(df, v1_params)
    assert v1_vol is not None and v1_vol > 0.0
    assert v1_payload is not None and v1_payload["engine_version"] == "v1_har_gjr"

    # Legacy V1 CQR params still preserve exact v1 clamping & lambda=1.0 behavior
    legacy_cqr = {
        "cqr_enabled": True,
        "q_hat_80": 0.005,
        "q_hat_96": 0.01,
        "median_val_vol": 0.015,
        "base_vol_power": 1.0,
    }
    raw_q = np.array([-0.04, -0.025, -0.012, 0.003, 0.015, 0.028, 0.043])
    cal_v1 = apply_weekly_cqr(raw_q, cqr_params=legacy_cqr, realized_vol=0.025)
    assert np.all(np.diff(cal_v1) > 0.0)
    assert np.isclose(cal_v1[3], raw_q[3], atol=1e-12)


def test_sustained_high_vol_crisis_does_not_classify_calm():
    """When elevated volatility persists >20 sessions so RV20 catches up, regime must not drop to CALM."""
    df = _make_synthetic_master_df(320)
    params = fit_har_gjr_volatility_stress(
        df,
        train_end_pos=220,
        val_origin_pos=np.arange(225, 265),
        horizon=5,
    )
    assert float(params.get("vol5_p75", 0.0)) > 0.0
    assert float(params.get("vol5_p90", 0.0)) >= float(params["vol5_p75"])
    assert len(params.get("vol5_ref_quantiles", [])) == 101

    # Inject a sustained 28-session crisis at the end so 20D RV rises alongside 5D forecast vol
    rng = np.random.default_rng(20200319)
    crisis_rets = rng.choice([-0.038, 0.034, -0.042, 0.029], size=28)
    df_crisis = df.copy()
    close = df_crisis["target_close"].to_numpy(dtype=np.float64)
    rets = df_crisis["observed_return_1d"].to_numpy(dtype=np.float64)
    rets[-28:] = crisis_rets
    base_price = float(close[-29])
    close[-28:] = base_price * np.exp(np.cumsum(crisis_rets))
    log_c = np.log(close)
    df_crisis["target_close"] = close
    df_crisis["target"] = rets
    df_crisis["observed_return_1d"] = rets
    df_crisis["realized_vol_20d"] = (
        pd.Series(rets).rolling(20, min_periods=5).std().ffill().fillna(0.011).to_numpy(dtype=np.float64)
    )
    df_crisis["HG=F_ret5"] = pd.Series(log_c).diff(5).fillna(0.0).to_numpy(dtype=np.float64)

    _, stress = predict_latest_volatility_stress(df_crisis, params)
    assert stress is not None
    assert stress["regime"] in {"ELEVATED", "HIGH_STRESS"}
    assert stress["regime"] != "CALM"
    assert float(stress["risk_budget_mult"]) <= 0.85
    assert stress["vol_percentile"] is not None
    assert float(stress["vol_percentile"]) >= 75.0
