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
    }
    for sym in LONG_TERM_SYMBOLS:
        if sym == "HG=F":
            r = hg_ret
        elif sym == "DX-Y.NYB":
            r = dxy_ret
        elif sym in ("GXC", "FXA"):
            r = china_ret + rng.normal(0.0, 0.005, size=n_bdays)
        elif sym == "CL=F":
            r = 0.25 * china_ret + rng.normal(0.0001, 0.016, size=n_bdays)
        else:
            r = 1.2 * hg_ret + rng.normal(0.0001, 0.011, size=n_bdays)
        panel[sym] = base_prices[sym] * np.exp(np.cumsum(r))
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

    for use_fred in (True, False):
        bundle = build_long_term_feature_panel(
            daily,
            target_symbol="HG=F",
            fred_frames=fred if use_fred else None,
        )
        validation = run_purged_walk_forward_suite(bundle.panel, n_folds=5)
        resp = fit_and_predict_long_term_outlook(bundle, validation=validation, symbol="HG=F")

        assert resp.status == "ok"
        assert resp.fred_macro_enabled if hasattr(resp, "fred_macro_enabled") else True
        assert resp.data_window is not None
        assert resp.data_window.fred_macro_enabled is use_fred
        assert set(resp.horizons.keys()) == {"20D", "60D"}

        for h_key, exp_weeks, exp_ann in (("20D", 4, 12.0), ("60D", 12, 4.0)):
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
            assert set(ok_res["horizons"].keys()) == {"20D", "60D"}

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

        # 2. Seed persisted snapshot and verify HTTP handler serves it without calling generator
        daily = _make_synthetic_daily_panel(n_bdays=1250, seed=55)
        bundle = build_long_term_feature_panel(daily, target_symbol="HG=F", fred_frames=None)
        validation = run_purged_walk_forward_suite(bundle.panel, n_folds=4)
        payload = fit_and_predict_long_term_outlook(bundle, validation=validation, symbol="HG=F")

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
        assert set(body["horizons"].keys()) == {"20D", "60D"}
        assert set(body["validation"].keys()) == {"20D", "60D"}
        assert body["macro_scorecard"]["composite_stance"] in {"BULLISH", "NEUTRAL", "BEARISH"}
    finally:
        engine.dispose()


def test_metric_helpers_sanity():
    assert sharpe_hac(np.array([0.01, 0.01]), 12.0, 3) == 0.0
    assert quantile_crossing_rate(np.array([[0.0, 0.1, 0.2], [0.0, 0.3, 0.2]])) == pytest.approx(0.25)
    b = brier_multiclass(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([0, 1]))
    assert b == pytest.approx(0.0)
