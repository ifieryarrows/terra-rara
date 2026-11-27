"""Offline unit and integration tests for the Long-Term Copper Outlook Engine (20D / 60D)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import long_term_routes
from app.long_term_models import LongTermOutlookSnapshot
from app.models import PriceBar
from long_term.backtest import (
    brier_multiclass,
    purged_walk_forward_splits,
    quantile_crossing_rate,
    run_purged_walk_forward_suite,
    sharpe_hac,
)
from long_term.features import (
    LONG_TERM_SYMBOLS,
    build_long_term_feature_panel,
    clean_price_rows_to_series,
)
from long_term.model import fit_and_predict_long_term_outlook
from long_term.schemas import LongTermOutlookResponse
from long_term.service import run_long_term_outlook_stage


def _make_synthetic_daily_panel(
    n_bdays: int = 1350,
    *,
    seed: int = 42,
    start: str = "2021-01-04",
) -> pd.DataFrame:
    """Create a correlated synthetic weekday price panel for all long-term symbols."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_bdays)
    dxy_ret = rng.normal(0.0001, 0.0045, size=n_bdays)
    china_ret = rng.normal(0.0001, 0.011, size=n_bdays)
    # Copper negatively correlated with USD and positively with China/industrial risk
    hg_ret = -0.35 * dxy_ret + 0.30 * china_ret + rng.normal(0.0002, 0.012, size=n_bdays)

    panel: dict[str, np.ndarray] = {}
    base_prices = {
        "HG=F": 4.2,
        "DX-Y.NYB": 100.0,
        "CL=F": 72.0,
        "GXC": 78.0,
        "COPX": 38.0,
        "FCX": 42.0,
        "SCCO": 75.0,
        "RIO": 66.0,
        "BHP": 58.0,
        "FXA": 66.0,
        "XME": 54.0,
        "USDCNH=X": 7.1,
        "CNY=X": 7.1,
        "GC=F": 2400.0,
        "^VIX": 18.0,
    }
    for sym in LONG_TERM_SYMBOLS:
        if sym == "HG=F":
            r = hg_ret
        elif sym == "DX-Y.NYB":
            r = dxy_ret
        elif sym in ("USDCNH=X", "CNY=X"):
            r = -0.35 * china_ret + 0.30 * dxy_ret + rng.normal(0.0, 0.003, size=n_bdays)
        elif sym == "^VIX":
            r = -1.5 * china_ret + rng.normal(0.0, 0.04, size=n_bdays)
        elif sym == "GC=F":
            r = -0.25 * dxy_ret + rng.normal(0.0001, 0.008, size=n_bdays)
        elif sym in ("GXC", "FXA"):
            r = china_ret + rng.normal(0.0, 0.005, size=n_bdays)
        elif sym == "CL=F":
            r = 0.25 * china_ret + rng.normal(0.0001, 0.016, size=n_bdays)
        else:
            r = 1.2 * hg_ret + rng.normal(0.0001, 0.011, size=n_bdays)
        panel[sym] = base_prices[sym] * np.exp(np.cumsum(r))
    if "^VIX" in panel:
        panel["^VIX"] = np.clip(panel["^VIX"], 10.0, 75.0)
    return pd.DataFrame(panel, index=dates)


def _make_synthetic_fred(dates: pd.DatetimeIndex, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = len(dates)
    dgs10 = 3.5 + np.cumsum(rng.normal(0.0, 0.03, size=n))
    t10yie = 2.2 + np.cumsum(rng.normal(0.0, 0.015, size=n))
    dtwex = 115.0 * np.exp(np.cumsum(rng.normal(0.0, 0.003, size=n)))
    return {
        "DGS10": pd.DataFrame({"date": dates, "value": np.clip(dgs10, 0.5, 8.0)}),
        "T10YIE": pd.DataFrame({"date": dates, "value": np.clip(t10yie, 0.5, 5.0)}),
        "DTWEXBGS": pd.DataFrame({"date": dates, "value": dtwex}),
    }


def test_clean_price_rows_dedup_adj_close_and_weekend_exclusion():
    rows = [
        # Legacy 04:00 UTC row should lose to canonical 00:00 UTC row on the same calendar date
        ("2026-11-16T04:00:00+00:00", 9.9, 10.2, 9.8, 10.0, 5000.0, None),
        ("2026-11-16T00:00:00+00:00", 9.9, 10.2, 9.8, 10.0, 5000.0, 9.5),
        # Weekend Sunday ghost bar should be dropped
        ("2026-11-15T00:00:00+00:00", 9.8, 9.9, 9.7, 9.85, 100.0, 9.4),
        # Valid row missing adj_close should fall back to close
        ("2026-11-17T00:00:00+00:00", 10.0, 10.3, 9.9, 10.25, 4200.0, None),
        # Non-positive / non-finite closes should be dropped
        ("2026-11-18T00:00:00+00:00", 10.0, 10.1, 9.9, -1.0, 100.0, 9.0),
        ("2026-11-19T00:00:00+00:00", 10.0, 10.1, 9.9, float("nan"), 100.0, 9.0),
        # Synthetic test fixture row leaked alongside real prices should be dropped
        ("2026-11-20T00:00:00+00:00", 100.0, 105.0, 95.0, 102.0, 1000.0, 102.0),
    ]
    ser = clean_price_rows_to_series(rows, "RIO")
    assert list(ser.index.strftime("%Y-%m-%d")) == ["2026-11-16", "2026-11-17"]
    assert ser.loc["2026-11-16"] == pytest.approx(9.5)
    assert ser.loc["2026-11-17"] == pytest.approx(10.25)


def test_weekly_causal_feature_alignment_and_zero_future_leakage():
    daily = _make_synthetic_daily_panel(n_bdays=900, seed=11)
    fred = _make_synthetic_fred(daily.index, seed=11)

    bundle_orig = build_long_term_feature_panel(daily, target_symbol="HG=F", fred_frames=fred)
    cutoff_pos = 120
    cutoff_friday = bundle_orig.panel.index[cutoff_pos]

    # Mutate all daily prices and FRED observations strictly after cutoff_friday
    mutated_daily = daily.copy()
    mutated_daily.loc[mutated_daily.index > cutoff_friday, :] *= 4.5

    mutated_fred = {}
    for k, df in fred.items():
        copy_df = df.copy()
        # For DTWEXBGS (7-day lag), even a mutation 2 days BEFORE cutoff_friday must not leak into cutoff_friday!
        if k == "DTWEXBGS":
            copy_df.loc[copy_df["date"] >= cutoff_friday - pd.Timedelta(days=3), "value"] *= 3.0
        else:
            copy_df.loc[copy_df["date"] > cutoff_friday, "value"] += 15.0
        mutated_fred[k] = copy_df

    bundle_mut = build_long_term_feature_panel(
        mutated_daily, target_symbol="HG=F", fred_frames=mutated_fred
    )

    fcols = bundle_orig.feature_columns + ["sigma_w"]
    orig_slice = bundle_orig.panel.iloc[: cutoff_pos + 1][fcols]
    mut_slice = bundle_mut.panel.iloc[: cutoff_pos + 1][fcols]

    pd.testing.assert_frame_equal(orig_slice, mut_slice, check_exact=False, rtol=1e-12, atol=1e-12)


def test_purged_walk_forward_splits_enforce_horizon_purge_gap():
    valid = np.arange(52, 420)
    for horizon_weeks in (4, 12):
        splits = purged_walk_forward_splits(valid, horizon_weeks, n_folds=5, min_train=120)
        assert len(splits) == 5
        for tr, te in splits:
            assert int(tr.max()) + horizon_weeks <= int(te.min())
            assert np.intersect1d(tr, te).size == 0


def test_ensemble_outputs_non_crossing_quantiles_and_calibrated_probs():
    daily = _make_synthetic_daily_panel(n_bdays=1400, seed=21)
    fred = _make_synthetic_fred(daily.index, seed=21)

    from long_term.model import LongTermModelParams

    for use_fred in (True, False):
        bundle = build_long_term_feature_panel(
            daily,
            target_symbol="HG=F",
            fred_frames=fred if use_fred else None,
        )
        test_params = (
            LongTermModelParams()
            if use_fred
            else LongTermModelParams(tree_iterations=100, mlp_epochs=15, mlp_seeds=(42,))
        )
        validation = run_purged_walk_forward_suite(bundle.panel, n_folds=5, params=test_params)
        resp = fit_and_predict_long_term_outlook(
            bundle, validation=validation, symbol="HG=F", params=test_params
        )

        assert resp.status == "ok"
        assert resp.fred_macro_enabled if hasattr(resp, "fred_macro_enabled") else True
        assert resp.data_window is not None
        assert resp.data_window.fred_macro_enabled is use_fred
        assert set(resp.horizons.keys()) == {"20D", "60D", "120D"}
        assert resp.default_horizon == "60D"
        assert resp.model_version == "lt_committee_v2"

        for h_key, exp_weeks, exp_ann in (("20D", 4, 12.0), ("60D", 12, 4.0), ("120D", 26, 2.0)):
            fc = resp.horizons[h_key]
            assert fc.horizon_weeks == exp_weeks
            assert (
                fc.price_q10
                <= fc.price_q25
                <= fc.target_price_q50
                <= fc.price_q75
                <= fc.price_q90
            )
            assert (
                fc.return_q10
                <= fc.return_q25
                <= fc.expected_return
                <= fc.return_q75
                <= fc.return_q90
            )
            assert fc.expected_return_pct == pytest.approx(fc.expected_return * 100.0, abs=1e-2)
            assert 0.0 <= fc.prob_bullish <= 1.0
            assert 0.0 <= fc.prob_neutral <= 1.0
            assert 0.0 <= fc.prob_bearish <= 1.0
            assert (fc.prob_bullish + fc.prob_neutral + fc.prob_bearish) == pytest.approx(
                1.0, abs=1e-3
            )
            assert 0.0 <= fc.conviction <= 1.0
            assert fc.stance in {"BULLISH", "NEUTRAL", "BEARISH"}
            assert fc.institutional_rating in {
                "STRONG_OVERWEIGHT",
                "OVERWEIGHT",
                "NEUTRAL",
                "UNDERWEIGHT",
                "STRONG_UNDERWEIGHT",
            }
            assert fc.asymmetry_ratio is not None and fc.asymmetry_ratio > 0.0
            assert fc.components.catboost_return is not None
            assert fc.components.xgboost_return is not None
            assert fc.components.mlp_plr_return is not None
            assert fc.components.tree_weight is not None and fc.components.tree_weight <= 0.5 + 1e-4
            assert fc.components.mlp_weight is not None and fc.components.mlp_weight >= 0.25 - 1e-4
            assert (
                fc.components.tree_weight + fc.components.mlp_weight + fc.components.theta_weight
            ) == pytest.approx(1.0, abs=1e-3)
            assert fc.components.shrinkage_alpha is not None and 0.35 <= fc.components.shrinkage_alpha <= 1.0
            assert len(fc.key_drivers) > 0

            val = resp.validation[h_key]
            assert val.purge_gap_weeks == exp_weeks
            assert val.annualisation == exp_ann
            assert val.quantile_crossing_rate == 0.0
            assert val.n_folds >= 4
            assert val.n_non_overlapping_samples > 0
            assert 0.0 <= val.pi80_coverage <= 1.0
            assert val.brier_score >= 0.0

        assert resp.macro_scorecard is not None
        assert set(resp.macro_scorecard.pillars.keys()) == {
            "trend_structure",
            "usd_macro",
            "china_industrial_demand",
            "cross_asset_confirmation",
        }
        for pillar in resp.macro_scorecard.pillars.values():
            assert -1.0 <= pillar.score <= 1.0
            assert 0 <= pillar.score_pct <= 100
            assert len(pillar.metrics) > 0


def test_worker_stage_db_snapshot_roundtrip_offline_and_fail_open(monkeypatch):
    import requests
    import yfinance as yf

    def _no_network(*_args, **_kwargs):
        raise AssertionError("Offline test must never invoke network calls")

    monkeypatch.setattr(yf, "download", _no_network)
    monkeypatch.setattr(requests, "get", _no_network)

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    PriceBar.__table__.create(bind=engine)
    LongTermOutlookSnapshot.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    try:
        # Empty DB fails open cleanly without raising
        with Session() as session:
            fail_res = run_long_term_outlook_stage(
                session, symbol="HG=F", run_id="run-empty", allow_network=False
            )
            assert fail_res["ok"] is False
            assert fail_res["generated"] is False

        # Seed 5+ years of weekday DB PriceBar rows (no network)
        daily = _make_synthetic_daily_panel(n_bdays=1320, seed=99)
        with Session() as session:
            bars = []
            for sym in LONG_TERM_SYMBOLS:
                ser = daily[sym]
                for dt, val in ser.items():
                    ts = datetime(dt.year, dt.month, dt.day, 0, 0, tzinfo=timezone.utc)
                    bars.append(
                        PriceBar(
                            symbol=sym,
                            date=ts,
                            open=float(val),
                            high=float(val * 1.01),
                            low=float(val * 0.99),
                            close=float(val),
                            volume=10000.0,
                            adj_close=float(val),
                        )
                    )
            session.bulk_save_objects(bars)
            session.commit()

        with Session() as session:
            ok_res = run_long_term_outlook_stage(
                session, symbol="HG=F", run_id="run-lt-99", allow_network=False
            )
            assert ok_res["ok"] is True
            assert ok_res["generated"] is True
            assert set(ok_res["horizons"].keys()) == {"20D", "60D", "120D"}

            row = session.query(LongTermOutlookSnapshot).one()
            assert row.symbol == "HG=F"
            assert row.run_id == "run-lt-99"
            parsed = LongTermOutlookResponse.model_validate(row.payload_json)
            assert parsed.status == "ok"
            assert parsed.data_window is not None
            assert parsed.data_window.fred_macro_enabled is False
    finally:
        engine.dispose()


def test_api_router_snapshot_only_and_unavailable_state(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    LongTermOutlookSnapshot.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(long_term_routes, "SessionLocal", Session)

    test_app = FastAPI()
    test_app.include_router(long_term_routes.long_term_router)
    client = TestClient(test_app)

    try:
        # 1. Empty table -> HTTP 200 clean unavailable contract
        resp_empty = client.get("/api/outlook/long-term/HG=F")
        assert resp_empty.status_code == 200
        data_empty = resp_empty.json()
        assert data_empty["status"] == "unavailable"
        assert data_empty["source"] == "unavailable"
        assert data_empty["horizons"] == {}

        from long_term.model import LongTermModelParams

        # 2. Seed persisted snapshot and verify HTTP handler serves it without calling generator
        daily = _make_synthetic_daily_panel(n_bdays=1250, seed=55)
        bundle = build_long_term_feature_panel(daily, target_symbol="HG=F", fred_frames=None)
        fast_params = LongTermModelParams(tree_iterations=60, mlp_epochs=10, mlp_seeds=(42,))
        validation = run_purged_walk_forward_suite(bundle.panel, n_folds=4, params=fast_params)
        payload = fit_and_predict_long_term_outlook(
            bundle, validation=validation, symbol="HG=F", params=fast_params
        )

        with Session() as session:
            session.add(
                LongTermOutlookSnapshot(
                    symbol="HG=F",
                    as_of_date=payload.as_of_date or "2026-11-20",
                    payload_json=payload.model_dump(mode="json"),
                    validation_json={
                        k: v.model_dump(mode="json") for k, v in payload.validation.items()
                    },
                    model_version="lt_structural_v1",
                    generated_at=datetime.now(timezone.utc),
                    run_id="api-test-run",
                )
            )
            session.commit()

        import long_term.service as lt_service

        def _forbid_inline(*_args, **_kwargs):
            raise AssertionError("HTTP router must never run inline outlook generation")

        monkeypatch.setattr(lt_service, "generate_long_term_outlook_report", _forbid_inline)

        # Also verifies canonicalization from COMEX:HG1! -> HG=F
        resp_ok = client.get("/api/outlook/long-term/COMEX:HG1!")
        assert resp_ok.status_code == 200
        body = resp_ok.json()
        assert body["status"] == "ok"
        assert body["source"] == "snapshot"
        assert body["symbol"] == "HG=F"
        assert set(body["horizons"].keys()) == {"20D", "60D", "120D"}
        assert set(body["validation"].keys()) == {"20D", "60D", "120D"}
        assert body["macro_scorecard"]["composite_stance"] in {"BULLISH", "NEUTRAL", "BEARISH"}
    finally:
        engine.dispose()


def test_metric_helpers_sanity():
    assert sharpe_hac(np.array([0.01, 0.01]), 12.0, 3) == 0.0
    assert quantile_crossing_rate(np.array([[0.0, 0.1, 0.2], [0.0, 0.3, 0.2]])) == pytest.approx(0.25)
    b = brier_multiclass(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([0, 1]))
    assert b == pytest.approx(0.0)


def test_dockerfile_copies_long_term_package():
    """Guard against HF Space / Docker startup ModuleNotFoundError for long_term."""
    from pathlib import Path

    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY ./long_term /code/long_term" in dockerfile


def test_physical_and_cot_strict_causality_and_fifth_pillar():
    from long_term.features import compute_macro_scorecard

    daily = _make_synthetic_daily_panel(n_bdays=900, seed=33)
    rng = np.random.default_rng(33)
    lme = pd.DataFrame(
        {
            "lme_stock_tonnes": 220000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, size=len(daily)))),
            "lme_cash_3m_spread_pct": rng.normal(-0.004, 0.006, size=len(daily)),
        },
        index=daily.index,
    )
    tuesdays = pd.date_range(daily.index.min(), daily.index.max(), freq="W-TUE")
    cot = pd.DataFrame(
        {
            "cot_mm_net_pct_oi": np.clip(np.cumsum(rng.normal(0.0, 0.02, size=len(tuesdays))), -0.35, 0.35),
            "cot_prod_net_pct_oi": np.clip(-0.1 - np.cumsum(rng.normal(0.0, 0.015, size=len(tuesdays))), -0.45, 0.1),
        },
        index=tuesdays,
    )

    # Verify _align_dated_frame_to_weekly works cleanly when date/report_date is a column (not index)
    lme_with_date_col = lme.reset_index().rename(columns={"index": "date"})
    cot_with_report_col = cot.reset_index().rename(columns={"index": "report_date"})
    bundle_col_check = build_long_term_feature_panel(
        daily, target_symbol="HG=F", lme_frame=lme_with_date_col, cot_frame=cot_with_report_col
    )
    assert bundle_col_check.physical_data_enabled is True
    assert bundle_col_check.cot_positioning_enabled is True

    bundle_orig = build_long_term_feature_panel(
        daily, target_symbol="HG=F", lme_frame=lme, cot_frame=cot
    )
    assert bundle_orig.physical_data_enabled is True
    assert bundle_orig.cot_positioning_enabled is True
    assert "physical_positioning" in compute_macro_scorecard(bundle_orig.panel).pillars

    cutoff_pos = 120
    cutoff_friday = bundle_orig.panel.index[cutoff_pos]

    lme_mut = lme.copy()
    lme_mut.loc[lme_mut.index >= cutoff_friday, "lme_stock_tonnes"] *= 4.0
    lme_mut.loc[lme_mut.index >= cutoff_friday, "lme_cash_3m_spread_pct"] += 0.25

    cot_mut = cot.copy()
    # Tuesday COT within 4-day publication lag before cutoff_friday must NOT leak into cutoff_friday
    cot_mut.loc[cot_mut.index >= cutoff_friday - pd.Timedelta(days=3), "cot_mm_net_pct_oi"] = -0.95
    cot_mut.loc[cot_mut.index >= cutoff_friday - pd.Timedelta(days=3), "cot_prod_net_pct_oi"] = 0.45

    bundle_mut = build_long_term_feature_panel(
        daily, target_symbol="HG=F", lme_frame=lme_mut, cot_frame=cot_mut
    )
    phys_cols = [
        c
        for c in (
            "lme_stock_chg4w",
            "lme_stock_chg12w",
            "lme_stock_z52",
            "lme_scarcity",
            "lme_stock_pctl_3y",
            "lme_roll_yield_ann",
            "lme_tightness_convex",
            "lme_stock_accel",
            "lme_cash_3m_spread_pct",
            "cot_mm_net_pct_oi",
            "cot_mm_net_chg4w",
            "cot_mm_flow_12w",
            "cot_mm_crowding_z156",
            "cot_mm_crowding_convex",
            "cot_spec_vs_prod_div",
        )
        if c in bundle_orig.panel.columns
    ]
    assert len(phys_cols) >= 13
    pd.testing.assert_frame_equal(
        bundle_orig.panel.iloc[: cutoff_pos + 1][phys_cols],
        bundle_mut.panel.iloc[: cutoff_pos + 1][phys_cols],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_cny_fallback_when_usdcnh_sparse_and_legacy_v1_schema_compat():
    daily = _make_synthetic_daily_panel(n_bdays=400, seed=77)
    # Simulate yfinance USDCNH=X returning only 1 valid row while CNY=X has full history
    daily["CNY=X"] = daily["USDCNH=X"].copy()
    daily["USDCNH=X"] = np.nan
    daily.iloc[-1, daily.columns.get_loc("USDCNH=X")] = 7.25

    bundle = build_long_term_feature_panel(daily, target_symbol="HG=F")
    assert "usdcnh_ret12" in bundle.panel.columns
    assert bundle.panel["usdcnh_ret12"].notna().sum() >= 50

    # Legacy lt_structural_v1 snapshot without institutional_rating/v2 committee fields validates cleanly
    legacy_payload = {
        "status": "ok",
        "symbol": "HG=F",
        "as_of_date": "2026-11-20",
        "model_version": "lt_structural_v1",
        "source": "snapshot",
        "available_horizons": ["20D", "60D"],
        "default_horizon": "20D",
        "horizons": {
            "20D": {
                "horizon_key": "20D",
                "horizon_label": "1 Month (20D)",
                "horizon_days": 20,
                "horizon_weeks": 4,
                "reference_price": 4.9,
                "target_price_q50": 5.0,
                "expected_return": 0.02,
                "expected_return_pct": 2.0,
                "expected_log_return": 0.0198,
                "price_q10": 4.6,
                "price_q25": 4.8,
                "price_q75": 5.2,
                "price_q90": 5.4,
                "return_q10": -0.06,
                "return_q25": -0.02,
                "return_q75": 0.06,
                "return_q90": 0.10,
                "stance": "BULLISH",
                "prob_bullish": 0.52,
                "prob_neutral": 0.30,
                "prob_bearish": 0.18,
                "conviction": 0.55,
                "conviction_label": "HIGH",
                "regime_threshold_pct": 2.5,
                "components": {
                    "macro_ridge_return": 0.02,
                    "theta_damped_return": 0.015,
                    "macro_ridge_weight": 0.55,
                    "theta_weight": 0.45,
                    "realized_vol_annualized": 0.21,
                    "horizon_vol": 0.06,
                    "cqr_q10_width": 0.08,
                    "cqr_q90_width": 0.08,
                },
                "key_drivers": [],
            }
        },
        "validation": {},
    }
    parsed = LongTermOutlookResponse.model_validate(legacy_payload)
    assert parsed.horizons["20D"].institutional_rating is None
    assert parsed.horizons["20D"].components.catboost_return is None


def test_causal_shrinkage_alpha_uses_effective_sample_size_not_n_lock():
    from long_term.model import compute_causal_shrinkage_alpha

    rng = np.random.default_rng(123)
    y = rng.normal(0.0, 0.08, size=200)
    # Oracle predictor: must NOT collapse to min_alpha=0.35 (was broken by 25 * n * var_y)
    alpha_oracle = compute_causal_shrinkage_alpha(y, y, horizon_weeks=12, prior_blocks=6.0)
    assert alpha_oracle > 0.70

    # Pure orthogonal / anti-correlated noise: shrinks down to floor min_alpha=0.35
    alpha_anti = compute_causal_shrinkage_alpha(y, -y, horizon_weeks=12, prior_blocks=6.0)
    assert alpha_anti == pytest.approx(0.35, abs=1e-9)


def test_select_model_features_prefers_structural_core_when_available_and_fails_open():
    from long_term.model import (
        MODEL_FEATURES,
        STRUCTURAL_CORE_FEATURES,
        TACTICAL_STRUCTURAL_FEATURES,
        select_model_features,
    )

    daily = _make_synthetic_daily_panel(n_bdays=900, seed=31)
    fred = _make_synthetic_fred(daily.index, seed=31)
    bundle = build_long_term_feature_panel(daily, target_symbol="HG=F", fred_frames=fred)

    # Synthetic FRED omits INDPRO, LME/COT missing -> should still resolve cleanly (or fall back to MODEL_FEATURES)
    f60 = select_model_features(bundle.panel, horizon_weeks=12)
    assert len(f60) >= 6

    # When full structural core (incl. LME/COT/INDPRO) exists, 60D/120D strictly uses compact core
    enriched = bundle.panel.copy()
    for col in STRUCTURAL_CORE_FEATURES:
        if col not in enriched.columns:
            enriched[col] = np.linspace(-1.0, 1.0, len(enriched))
    f120_enriched = select_model_features(enriched, horizon_weeks=26)
    assert f120_enriched == list(STRUCTURAL_CORE_FEATURES)
    f20_enriched = select_model_features(enriched, horizon_weeks=4)
    assert f20_enriched == list(TACTICAL_STRUCTURAL_FEATURES)
    # Unspecified horizon preserves full MODEL_FEATURES selection
    f_all = select_model_features(enriched)
    assert len(f_all) > len(f120_enriched)
    assert set(f_all).issubset(set(MODEL_FEATURES))
