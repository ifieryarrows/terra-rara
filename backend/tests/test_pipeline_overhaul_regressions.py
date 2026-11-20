"""Regression tests verifying first-principles overhaul across TFT, Screener, Backtest, and Sync."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from app.features import generate_symbol_features
from app.price_utils import earliest_finite_price_dates
from backtest.runner import (
    BacktestConfig,
    BacktestRunner,
    SymbolSet,
    TFTBacktestRunner,
)
from deep_learning.calibration.conformal import apply_weekly_cqr, fit_weekly_cqr
from deep_learning.config import get_tft_config
from deep_learning.data.regime_features import build_regime_event_features
from deep_learning.models import hub as hub_module
from deep_learning.models.monotonic_quantiles import (
    aggregate_weekly_quantiles_np,
    aggregate_weekly_quantiles_torch,
)
from deep_learning.models.tft_copper import WeeklyASROPFLoss
from screener.contracts.universe import (
    FilterParameters,
    LibVersions,
    UniverseMeta,
    UniverseOutput,
    UniverseSummary,
    UniverseSymbol,
)
from screener.core.config import AnalysisConfig, ScreenerConfig
from screener.feature_screener.orchestrator import FeatureScreener


QUANTILES = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)


def test_weekly_asro_loss_update_eliminates_160x_dilution():
    torch.manual_seed(11)
    y_pred = torch.randn(32, 5, 7) * 0.02
    y_actual = torch.randn(32, 5) * 0.015
    loss_fn = WeeklyASROPFLoss(quantiles=list(QUANTILES))

    raw_loss = loss_fn.loss(y_pred, y_actual)
    metric_out = loss_fn(y_pred, y_actual)

    assert torch.isclose(metric_out, raw_loss, rtol=1e-5, atol=1e-6)


def test_rss_weekly_quantile_aggregation_scales_with_sqrt_horizon_and_is_monotonic():
    daily = torch.zeros(4, 5, 7)
    offsets = torch.tensor([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03])
    daily[..., :] = offsets
    daily[..., :] += 0.002  # constant +0.002/day median drift -> +0.010 weekly median

    weekly_t = aggregate_weekly_quantiles_torch(daily, median_idx=3)
    weekly_np = aggregate_weekly_quantiles_np(daily.numpy(), median_idx=3)

    assert torch.allclose(weekly_t, torch.tensor(weekly_np, dtype=torch.float32), atol=1e-6)
    # Median is additive (+0.010); PI80 half-width scales by sqrt(5) * 0.02, NOT 5 * 0.02
    expected_q50 = 0.010
    expected_half80 = np.sqrt(5.0) * 0.02
    assert weekly_np[0, 3] == pytest.approx(expected_q50, abs=1e-6)
    assert (weekly_np[0, 5] - weekly_np[0, 3]) == pytest.approx(expected_half80, abs=1e-5)
    assert np.all(np.diff(weekly_np[0]) > 0.0)


def test_split_cqr_calibrates_pi80_and_preserves_median_and_monotonicity():
    rng = np.random.default_rng(123)
    n = 300
    vols = rng.uniform(0.01, 0.04, size=n)
    y = rng.normal(0.0, vols * 2.0)

    # Undercovering uncalibrated quantiles (too narrow relative to realized vol)
    q50 = rng.normal(0.0, 0.003, size=n)
    pred = np.stack(
        [
            q50 - 0.8 * vols,
            q50 - 0.4 * vols,
            q50 - 0.2 * vols,
            q50,
            q50 + 0.2 * vols,
            q50 + 0.4 * vols,
            q50 + 0.8 * vols,
        ],
        axis=1,
    )

    cqr = fit_weekly_cqr(y[:180], pred[:180], realized_vol=vols[:180])
    assert cqr["cqr_enabled"] is True

    cal_test = apply_weekly_cqr(pred[180:], cqr_params=cqr, realized_vol=vols[180:])
    assert np.allclose(cal_test[:, 3], q50[180:])
    assert np.all(np.diff(cal_test, axis=1) >= -1e-12)
    cov80 = float(np.mean((y[180:] >= cal_test[:, 1]) & (y[180:] <= cal_test[:, 5])))
    assert 0.68 <= cov80 <= 0.92


def test_regime_features_produce_nonzero_variance_with_dxy_gxc_and_proxy_lme():
    n = 120
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "DX-Y.NYB_ret1": rng.normal(0.001, 0.008, size=n),
            "GXC_ret1": rng.normal(0.002, 0.015, size=n),
            "observed_return_1d": rng.normal(0.0, 0.012, size=n),
            "futures_curve_slope": rng.normal(0.0, 0.01, size=n),
            "proxy_vol_spike": (rng.random(size=n) > 0.85).astype(float),
        },
        index=idx,
    )

    out = build_regime_event_features(df)
    for col in ("regime_usd_pressure", "regime_risk_on_demand", "regime_risk_off_macro", "regime_inventory_tightness"):
        assert out[col].std() > 0.05


def test_generate_symbol_features_stationary_mode_omits_raw_levels():
    n = 100
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.linspace(4.0, 8.0, n)
    px = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )
    feats = generate_symbol_features(px, "FCX", include_raw_ma=False)
    assert "FCX_SMA_5" not in feats.columns
    assert "FCX_EMA_20" not in feats.columns
    assert "FCX_close_sma_20_ratio" in feats.columns
    assert "FCX_ret5" in feats.columns


def test_backtest_runner_short_circuits_identical_hash_and_blocks_target_leak():
    cfg = BacktestConfig(oos_start="2025-01-06", oos_end="2025-02-07", train_window_years=1)
    runner = BacktestRunner(cfg)

    dates = pd.bdate_range("2023-01-02", "2025-02-07")
    rng = np.random.default_rng(9)
    hg = 4.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates))))
    dxy = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, len(dates))))
    prices = pd.concat(
        [
            pd.DataFrame({"date": dates, "symbol": "HG=F", "close": hg}),
            pd.DataFrame({"date": dates, "symbol": "DX-Y.NYB", "close": dxy}),
        ],
        ignore_index=True,
    )
    feats = runner.prepare_features(prices, "HG=F")
    assert "DX-Y.NYB_ratio" not in feats.columns
    assert "DX-Y.NYB_sma20_ratio" in feats.columns
    assert "target_date" in feats.columns

    # Verify target_date <= train_end excludes Friday->Monday boundary leak
    train_end = pd.Timestamp("2025-01-05")
    mask = (feats.index <= train_end) & (pd.to_datetime(feats["target_date"]) <= train_end)
    assert pd.Timestamp("2025-01-03") not in feats.index[mask]

    runner._price_cache[("HG=F", "2024-01-06", "2025-02-07")] = prices[prices["symbol"] == "HG=F"].drop(columns=[]).copy()
    runner.fetch_prices = lambda syms, s, e: prices[prices["symbol"].isin(syms)].copy()
    s1 = SymbolSet("champ", ["DX-Y.NYB"], "1.0", "a.json", "sha256:same")
    s2 = SymbolSet("chall", ["DX-Y.NYB"], "2.0", "b.json", "sha256:same")
    result, _ = runner.compare(s1, s2)
    assert result.decision == "KEEP_CHAMPION"


def test_tft_backtest_runner_raises_when_artifacts_missing(tmp_path, monkeypatch):
    cfg = BacktestConfig(oos_start="2025-01-06", oos_end="2025-02-07")
    runner = TFTBacktestRunner(cfg)
    base_cfg = get_tft_config()
    patched_cfg = replace(
        base_cfg,
        training=replace(base_cfg.training, best_model_path=str(tmp_path / "best_tft_asro.ckpt")),
    )
    monkeypatch.setattr("deep_learning.config.get_tft_config", lambda: patched_cfg)
    monkeypatch.setattr("deep_learning.models.hub.sync_tft_artifacts_if_stale", lambda **_kw: False)

    with pytest.raises(RuntimeError, match="unavailable or incomplete"):
        runner.run_backtest(["DX-Y.NYB"])


def test_screener_rolling_windows_and_pairwise_redundancy():
    analysis = AnalysisConfig(window_mode="rolling", is_years=4, oos_years=1)
    is_s, is_e, oos_s, oos_e = analysis.resolve_windows(date(2026, 6, 15))
    assert oos_e == date(2026, 6, 15)
    assert oos_s == date(2025, 6, 15)
    assert is_e == date(2025, 6, 14)

    universe = UniverseOutput(
        meta=UniverseMeta(
            run_id="univ-20260615-abc1",
            generated_at="2026-06-15T00:00:00Z",
            config_hash="sha256:0123456789abcdef",
            lib_versions=LibVersions(python="3.12"),
        ),
        sources=[],
        filter_parameters=FilterParameters(min_history_days=365, min_coverage_pct=80.0, frequency="W-FRI"),
        summary=UniverseSummary(total_candidates=3, included=3, excluded=0),
        universe=[
            UniverseSymbol(ticker="HG=F", canonical_ticker="HG=F", category="target", status="included"),
            UniverseSymbol(ticker="FCX", canonical_ticker="FCX", category="miner_major", status="included"),
            UniverseSymbol(ticker="SCCO", canonical_ticker="SCCO", category="miner_major", status="included"),
        ],
        content_fingerprint="sha256:0123456789abcdef",
        output_fingerprint="sha256:0123456789abcdef",
    )
    scr = FeatureScreener(ScreenerConfig(analysis=analysis), universe)
    idx = pd.date_range("2025-01-03", periods=40, freq="W-FRI")
    base = pd.Series(np.sin(np.linspace(0, 8, 40)) * 0.03, index=idx)
    scr.prices = {
        "FCX": pd.DataFrame({"returns": base}),
        "SCCO": pd.DataFrame({"returns": base * 1.01 + 1e-4}),
        "CL=F": pd.DataFrame({"returns": pd.Series(np.cos(np.linspace(0, 8, 40)) * 0.03, index=idx)}),
    }
    assert scr._is_pairwise_redundant("SCCO", ["FCX"], 0.85) is True
    assert scr._is_pairwise_redundant("CL=F", ["FCX"], 0.85) is False


def test_hub_revision_sync_refreshes_on_sha_change_and_skips_when_current(tmp_path, monkeypatch):
    for name in ("best_tft_asro.ckpt", "tft_metadata.json", "artifact_manifest.json"):
        (tmp_path / name).write_text("{}")

    monkeypatch.setattr(hub_module, "validate_tft_artifact_set", lambda _d, quiet=False: True)
    monkeypatch.setattr(hub_module, "_remote_repo_sha", lambda _r, _t: "sha_v2")

    calls = {"download": 0, "reset": 0}
    monkeypatch.setattr(hub_module, "_notify_predictor_reload", lambda: calls.__setitem__("reset", calls["reset"] + 1))

    def fake_download(*, local_dir, repo_id, force_all=False):
        calls["download"] += 1
        for name in ("best_tft_asro.ckpt", "tft_metadata.json", "artifact_manifest.json"):
            (Path(local_dir) / name).write_text("updated")
        return True

    monkeypatch.setattr(hub_module, "download_tft_artifacts", fake_download)

    (tmp_path / hub_module._REVISION_FILE).write_text("sha_v1")
    assert hub_module.sync_tft_artifacts_if_stale(tmp_path, "org/repo") is True
    assert calls["download"] == 1
    assert calls["reset"] == 1
    assert (tmp_path / hub_module._REVISION_FILE).read_text() == "sha_v2"

    # Second call with matching SHA skips download completely
    assert hub_module.sync_tft_artifacts_if_stale(tmp_path, "org/repo") is True
    assert calls["download"] == 1


def test_embeddings_insert_count_and_price_earliest_boundary_sqlite():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models import Base, NewsEmbedding, NewsProcessed, NewsRaw, PriceBar
    from deep_learning.data.embeddings import _insert_embedding_batch

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        # 1. Price boundary check (earliest & latest finite bars)
        base_dt = datetime(2025, 1, 10, tzinfo=timezone.utc)
        session.add_all(
            [
                PriceBar(symbol="HG=F", date=base_dt, open=4.1, high=4.2, low=4.0, close=4.15, volume=100),
                PriceBar(symbol="HG=F", date=base_dt + timedelta(days=30), open=4.2, high=4.3, low=4.1, close=4.25, volume=110),
            ]
        )
        # 2. News embedding insert with conflict deduplication via RETURNING
        raw = NewsRaw(
            url="https://example.com/1",
            title="Copper news",
            source="reuters",
            published_at=base_dt,
        )
        session.add(raw)
        session.flush()
        proc = NewsProcessed(
            raw_id=raw.id,
            canonical_title="copper news",
            canonical_title_hash="h1",
            dedup_key="dk1",
        )
        session.add(proc)
        session.flush()

        payloads = [
            {
                "news_processed_id": proc.id,
                "embedding_full": b"\x01" * 16,
                "embedding_pca": b"\x02" * 32,
                "pca_version": "v1",
            }
        ]
        ins1, skip1 = _insert_embedding_batch(session, payloads)
        assert ins1 == 1 and skip1 == 0
        # Re-inserting same news_processed_id reports accurate conflict skip
        ins2, skip2 = _insert_embedding_batch(session, payloads)
        assert ins2 == 0 and skip2 == 1

        earliest = earliest_finite_price_dates(session, ["HG=F"])
        assert "HG=F" in earliest
        assert earliest["HG=F"].date() == base_dt.date()


def test_backtest_json_safe_and_db_persistence_handles_numpy_scalars(monkeypatch):
    from dataclasses import asdict

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models import BacktestReport, Base
    from backtest.runner import (
        BacktestMetrics,
        BacktestResult,
        _persist_report_to_db,
        _persist_tft_report_to_db,
        _to_json_safe,
    )

    sample = {
        "f32": np.float32(5.5263),
        "nan_val": float("nan"),
        "flag": np.bool_(True),
        "cnt": np.int64(42),
        "ts": pd.Timestamp("2026-11-19"),
        "arr": np.array([np.float32(1.5), np.float64("inf")]),
    }
    safe = _to_json_safe(sample)
    assert type(safe["f32"]) is float and safe["f32"] == pytest.approx(5.5263, rel=1e-5)
    assert safe["nan_val"] is None
    assert safe["flag"] is True and type(safe["flag"]) is bool
    assert safe["cnt"] == 42 and type(safe["cnt"]) is int
    assert safe["ts"].startswith("2026-11-19")
    assert safe["arr"] == [pytest.approx(1.5), None]
    json.dumps(safe)

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    import app.db as app_db

    monkeypatch.setattr(app_db, "SessionLocal", Session)

    metrics = BacktestMetrics(
        mae=np.float32(0.012),
        rmse=np.float32(0.020),
        directional_accuracy=np.float32(0.55),
        n_predictions=int(np.int64(10)),
        mean_actual=np.float32(0.001),
        mean_predicted=np.float32(0.0005),
        precision_up=0.5,
        recall_up=0.5,
        mcc=np.float32(0.1),
        confusion_matrix={"tp": 5, "tn": 5, "fp": 0, "fn": 0},
        baseline_always_up=0.5,
        baseline_always_down=0.5,
        baseline_repeat=0.5,
    )
    res = BacktestResult(
        run_id="bt-test-np",
        generated_at="2026-11-19T00:00:00+00:00",
        config=BacktestConfig(oos_start="2025-01-01", oos_end="2025-12-31"),
        champion={"symbol_set": {"symbols": ["DX-Y.NYB"]}, "metrics": asdict(metrics)},
        challenger={"symbol_set": {"symbols": ["DX-Y.NYB"]}, "metrics": asdict(metrics)},
        delta_mae_pct=np.float32(0.0),
        delta_rmse_pct=np.float32(0.0),
        delta_dir_acc_pct=np.float32(0.0),
        improvement_mae_pct=np.float32(0.0),
        decision="KEEP_CHAMPION",
        decision_reason="test",
    )
    _persist_report_to_db(res, kind="xgb_challenger")

    _persist_tft_report_to_db(
        {
            "run_id": "tft-bt-test-np",
            "xgboost": asdict(metrics),
            "tft_asro": asdict(metrics),
            "tft_walkforward_summary": {"mean_da": np.float32(0.54)},
            "windows": [{"da": np.float32(0.54), "sharpe": np.float32(0.2)}],
            "theta_comparison": {"verdict": "TFT_SUPERIOR"},
            "delta_mae_pct": np.float32(-10.5),
            "delta_dir_acc_pct": np.float32(2.5),
            "tft_better_mae": np.bool_(True),
            "tft_better_dir": np.bool_(True),
            "verdict": "TFT_SUPERIOR",
        }
    )

    with Session() as session:
        rows = session.query(BacktestReport).order_by(BacktestReport.id.asc()).all()
        assert len(rows) == 2
        assert isinstance(rows[0].summary_json["champion"]["mean_predicted"], float)
        assert isinstance(rows[1].summary_json["tft_better_mae"], bool)
    engine.dispose()


def test_backfill_embeddings_fits_pca_from_historical_corpus_and_reprojects(monkeypatch, tmp_path):
    import uuid
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import db
    from app.models import Base, NewsEmbedding, NewsProcessed, NewsRaw
    from deep_learning import config
    from deep_learning.data import embeddings
    from deep_learning.models import hub

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    run_id = uuid.uuid4()
    published = datetime.now(timezone.utc) - timedelta(days=2)
    rng = np.random.default_rng(77)

    with Session() as session:
        # Insert 8 historical articles with full 768-d embeddings and legacy pca32_v1
        for idx in range(8):
            raw = NewsRaw(
                url=f"https://example.test/hist-{idx}",
                url_hash=f"{idx:064x}",
                title=f"Historical copper article {idx}",
                source="test",
                published_at=published - timedelta(days=idx + 1),
                run_id=run_id,
            )
            session.add(raw)
            session.flush()
            proc = NewsProcessed(
                raw_id=raw.id,
                canonical_title=f"hist copper {idx}",
                canonical_title_hash=f"{idx + 100:064x}",
                cleaned_text=f"Historical copper text body number {idx}",
                dedup_key=f"{idx + 200:064x}",
                run_id=run_id,
            )
            session.add(proc)
            session.flush()
            vec768 = rng.normal(0.0, 1.0, size=768).astype(np.float32)
            vec32 = rng.normal(0.0, 1.0, size=32).astype(np.float32)
            session.add(
                NewsEmbedding(
                    news_processed_id=proc.id,
                    embedding_full=vec768.tobytes(),
                    embedding_pca=vec32.tobytes(),
                    pca_version="pca32_v1",
                )
            )
        # Insert 3 new articles (<= pca_dim=4) without embeddings
        for idx in range(3):
            raw = NewsRaw(
                url=f"https://example.test/new-{idx}",
                url_hash=f"{idx + 50:064x}",
                title=f"Fresh copper article {idx}",
                source="test",
                published_at=published,
                run_id=run_id,
            )
            session.add(raw)
            session.flush()
            proc = NewsProcessed(
                raw_id=raw.id,
                canonical_title=f"fresh copper {idx}",
                canonical_title_hash=f"{idx + 300:064x}",
                cleaned_text=f"Fresh copper market text {idx}",
                dedup_key=f"{idx + 400:064x}",
                run_id=run_id,
            )
            session.add(proc)
        session.commit()

    pca_path = tmp_path / "pca_finbert.joblib"
    monkeypatch.setattr(db, "SessionLocal", Session)
    monkeypatch.setattr(
        config,
        "get_tft_config",
        lambda: SimpleNamespace(
            embedding=SimpleNamespace(pca_model_path=str(pca_path)),
            training=SimpleNamespace(hf_model_repo="org/repo"),
        ),
    )
    monkeypatch.setattr(hub, "download_pca_artifact", lambda *_a, **_k: False)
    monkeypatch.setattr(
        embeddings,
        "extract_embeddings_batch",
        lambda texts, **_kw: rng.normal(0.0, 1.0, size=(len(texts), 768)).astype(np.float32),
    )

    stats = embeddings.backfill_embeddings(days=30, pca_dim=4, batch_size=8)
    assert stats["pca_fitted"] is True
    assert stats["embedded"] == 3
    assert stats["reprojected"] == 8
    assert pca_path.exists()

    with Session() as session:
        all_rows = session.query(NewsEmbedding).all()
        assert len(all_rows) == 11
        assert all(r.pca_version == "pca4_v1" for r in all_rows)
        assert all(len(np.frombuffer(r.embedding_pca, dtype=np.float32)) == 4 for r in all_rows)
    engine.dispose()


def test_universe_prober_batch_uses_chunked_download_with_fallback(monkeypatch):
    from screener.universe_builder import prober

    idx = pd.date_range("2022-01-03", periods=800, freq="B")
    rng = np.random.default_rng(19)
    close_fcx = pd.Series(40.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(idx)))), index=idx)
    close_scco = pd.Series(80.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(idx)))), index=idx)

    cols = pd.MultiIndex.from_product([["FCX", "SCCO"], ["Open", "High", "Low", "Close", "Volume"]])
    batch_df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for sym, s in [("FCX", close_fcx), ("SCCO", close_scco)]:
        batch_df[(sym, "Open")] = s
        batch_df[(sym, "High")] = s * 1.01
        batch_df[(sym, "Low")] = s * 0.99
        batch_df[(sym, "Close")] = s
        batch_df[(sym, "Volume")] = 1000.0

    download_calls = []

    def fake_download(symbols, **_kwargs):
        download_calls.append(list(symbols))
        return batch_df

    fallback_df = pd.DataFrame(
        {"Open": close_fcx, "High": close_fcx, "Low": close_fcx, "Close": close_fcx, "Volume": 1000.0},
        index=idx,
    )

    class FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        def history(self, **_kwargs):
            if self.sym == "MISS":
                return fallback_df
            return pd.DataFrame()

    monkeypatch.setattr(prober, "_rate_limiter", prober.RateLimiter(calls_per_hour=1_000_000))
    monkeypatch.setattr(prober.yf, "download", fake_download)
    monkeypatch.setattr(prober.yf, "Ticker", FakeTicker)

    results = prober.probe_batch(
        [
            {"ticker": "FCX", "canonical_ticker": "FCX"},
            {"ticker": "SCCO", "canonical_ticker": "SCCO"},
            {"ticker": "MISS", "canonical_ticker": "MISS"},
        ],
        frequency="W-FRI",
        min_history_days=300,
    )
    assert len(download_calls) == 1
    assert len(results) == 3
    assert all(r.valid for r in results)
    assert [r.canonical_ticker for r in results] == ["FCX", "SCCO", "MISS"]


def test_backtest_module_cli_help_has_no_runpy_warning():
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "backend")
    proc = subprocess.run(
        [sys.executable, "-m", "backend.backtest.runner", "--help"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "RuntimeWarning" not in proc.stderr


def test_tft_causal_target_eliminates_one_day_double_shift(monkeypatch):
    from deep_learning.data import feature_store
    from deep_learning.data.dataset import build_datasets

    n = 90
    dates = pd.date_range("2025-01-02", periods=n, freq="B")
    rng = np.random.default_rng(77)
    close = pd.Series(4.5 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n))), index=dates)
    price_df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 10000.0),
        },
        index=dates,
    )
    feats = generate_symbol_features(price_df, "HG=F", include_raw_ma=False)
    assert "HG=F_ret10" in feats.columns

    monkeypatch.setattr(feature_store, "load_training_symbols", lambda: ["HG=F"])
    monkeypatch.setattr(feature_store, "_build_price_features", lambda *_a, **_kw: (price_df, feats))
    monkeypatch.setattr(
        feature_store,
        "_build_daily_embedding_features",
        lambda _s, index, pca_dim=0: pd.DataFrame(index=index),
    )
    monkeypatch.setattr("app.features.load_sentiment_data", lambda *_a, **_kw: pd.DataFrame())
    monkeypatch.setattr("deep_learning.data.lme_warehouse.fetch_lme_data", lambda _c, **_kw: pd.DataFrame())
    monkeypatch.setattr(
        "deep_learning.data.lme_warehouse.compute_proxy_lme_features",
        lambda df: pd.DataFrame({"lme_proxy": 0.0}, index=df.index),
    )
    monkeypatch.setattr(
        "deep_learning.data.futures_curve.build_futures_features_from_yfinance",
        lambda *_a, **_kw: pd.DataFrame(),
    )

    cfg = replace(
        get_tft_config(),
        training=replace(get_tft_config().training, lookback_days=180),
        feature_store=replace(get_tft_config().feature_store, mrmr_top_k=12),
    )
    master, tv_unknown, tv_known, target_cols, _ = feature_store.build_tft_dataframe(object(), cfg)

    assert tv_known == ["day_of_week"]
    assert "cal_sin_day" not in tv_known
    assert "cal_sin_day" not in tv_unknown
    assert "HG=F_ret5" in tv_unknown
    assert "HG=F_ret10" in tv_unknown

    tr_ds, _, _ = build_datasets(master, tv_unknown, tv_known, target_cols, cfg)
    dl = tr_ds.to_dataloader(train=False, batch_size=len(tr_ds), num_workers=0)
    _, y = next(iter(dl))
    dec_target = y[0].numpy()
    origin_pos = tr_ds.index["index_end"].to_numpy(dtype=int) - cfg.forecast.primary_horizon_days
    expected_weekly = master["target_5d_log_return"].iloc[origin_pos].to_numpy(dtype=np.float32)

    np.testing.assert_allclose(dec_target.sum(axis=1), expected_weekly, atol=1e-6)


def test_hybrid_mrmr_preserves_directional_copper_features_and_respects_budget():
    from deep_learning.data.feature_selection import select_features

    rng = np.random.default_rng(31)
    n = 240
    z = rng.normal(0.0, 1.0, n)
    regime_vol = rng.uniform(0.5, 2.5, n)
    y = z * regime_vol * 0.02

    df = pd.DataFrame(
        {
            "time_idx": np.arange(n),
            "group_id": "copper",
            "day_of_week": np.arange(n) % 5 / 4.0,
            "target": rng.normal(0.0, 0.01, n),
            "target_5d_log_return": y,
            "HG=F_ret5": -0.4 * y + rng.normal(0.0, 0.015, n),
            "HG=F_ret10": -0.3 * y + rng.normal(0.0, 0.018, n),
            "HG=F_close_sma_20_ratio": -0.25 * y + rng.normal(0.0, 0.012, n),
            "pure_vol_noise": np.abs(y) + rng.normal(0.0, 0.002, n),
            "directional_cross": 0.6 * y + rng.normal(0.0, 0.015, n),
            "noise_a": rng.normal(0.0, 1.0, n),
            "noise_b": rng.normal(0.0, 1.0, n),
        }
    )

    _, selected_unknown, selected_known = select_features(
        df,
        target_col="target_5d_log_return",
        mrmr_top_k=4,
        known_features=["day_of_week"],
        forced_unknown_features=["HG=F_ret5", "HG=F_ret10"],
        forbidden_features=["target"],
    )

    assert selected_known == ["day_of_week"]
    assert len(selected_unknown) == 4
    assert "HG=F_ret5" in selected_unknown
    assert "HG=F_ret10" in selected_unknown
    assert "directional_cross" in selected_unknown


def test_weekly_loss_drift_prior_and_adamw_differential_head_lr():
    from deep_learning.data.dataset import build_datasets
    from deep_learning.models.tft_copper import create_tft_model

    actual = torch.tensor([[0.01, 0.0, -0.01, 0.0, 0.0], [-0.01, 0.0, 0.01, 0.0, 0.0]])
    pred_zero = torch.stack([actual * 0.0] * 7, dim=-1) + torch.tensor([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03])
    pred_drifted = pred_zero + 0.015

    loss_fn = WeeklyASROPFLoss(quantiles=list(QUANTILES), lambda_weekly_quantile=0.0, lambda_t1_quantile=0.0, lambda_bias=0.0, lambda_drift_prior=1.0, lambda_directional=0.0)
    assert loss_fn.loss(pred_zero, actual) < loss_fn.loss(pred_drifted, actual)

    n = 40
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n),
            "group_id": "copper",
            "day_of_week": np.arange(n) % 5 / 4.0,
            "HG=F_ret1": np.linspace(-0.02, 0.02, n, dtype=np.float32),
            "target": np.linspace(-0.01, 0.01, n, dtype=np.float32),
        }
    )
    cfg = get_tft_config()
    tr_ds, _, _ = build_datasets(df, ["HG=F_ret1"], ["day_of_week"], ["target"], cfg)
    model = create_tft_model(tr_ds, cfg, use_asro=True, weekly_target_scale=0.04)
    opt_cfg = model.configure_optimizers()
    opt = opt_cfg["optimizer"]
    assert isinstance(opt, torch.optim.AdamW)
    lrs = sorted(pg["lr"] for pg in opt.param_groups)
    assert lrs[0] == pytest.approx(cfg.model.learning_rate * cfg.model.output_head_lr_mult)
    assert lrs[1] == pytest.approx(cfg.model.learning_rate)


def test_split_cqr_uses_linear_volatility_power():
    rng = np.random.default_rng(17)
    n = 80
    vols = rng.uniform(0.01, 0.03, size=n)
    y = rng.normal(0.0, vols * 2.0)
    pred = np.stack([-2 * vols, -vols, -0.5 * vols, np.zeros(n), 0.5 * vols, vols, 2 * vols], axis=1)
    cqr = fit_weekly_cqr(y, pred, realized_vol=vols)
    assert cqr["cqr_enabled"] is True
    assert cqr["base_vol_power"] == pytest.approx(1.0)


def test_dataset_origin_conditioning_values_maps_global_time_idx_for_val_and_test():
    from deep_learning.data.dataset import build_datasets
    from deep_learning.training.trainer import _dataset_origin_conditioning_values

    n = 120
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n),
            "group_id": "copper",
            "day_of_week": np.arange(n) % 5 / 4.0,
            "HG=F_ret1": np.linspace(-0.02, 0.02, n, dtype=np.float32),
            "target": np.linspace(-0.01, 0.01, n, dtype=np.float32),
            "realized_vol_20d": 0.01 + 0.0001 * np.arange(n, dtype=np.float64),
        }
    )
    cfg = get_tft_config()
    _, va_ds, te_ds = build_datasets(df, ["HG=F_ret1"], ["day_of_week"], ["target"], cfg)
    lookup = df.set_index("time_idx")["realized_vol_20d"]

    for ds in (va_ds, te_ds):
        dl = ds.to_dataloader(train=False, batch_size=len(ds), num_workers=0)
        x, _ = next(iter(dl))
        expected_origins = x["decoder_time_idx"][:, 0].numpy().astype(int) - 1
        expected_vals = lookup.loc[expected_origins].to_numpy(dtype=np.float64)
        got_vals = _dataset_origin_conditioning_values(
            df,
            ds,
            feature_name="realized_vol_20d",
            horizon=cfg.forecast.primary_horizon_days,
            start_exclusive=0,
            end_inclusive=0,
            expected_count=len(expected_vals),
        )
        np.testing.assert_allclose(got_vals, expected_vals, atol=1e-12)


def test_screener_retains_mandatory_clf_even_when_is_corr_below_threshold():
    from datetime import date
    from screener.contracts.universe import (
        FilterParameters,
        LibVersions,
        UniverseMeta,
        UniverseOutput,
        UniverseSummary,
        UniverseSymbol,
    )
    from screener.core.config import AnalysisConfig, ScreenerConfig
    from screener.feature_screener.evaluator import CandidateEvaluation, PeriodMetrics
    from screener.feature_screener.orchestrator import FeatureScreener

    universe = UniverseOutput(
        meta=UniverseMeta(
            run_id="univ-20261120-test1",
            generated_at="2026-11-20T00:00:00Z",
            config_hash="sha256:0123456789abcdef",
            lib_versions=LibVersions(python="3.12"),
        ),
        sources=[],
        filter_parameters=FilterParameters(min_history_days=365, min_coverage_pct=80.0, frequency="W-FRI"),
        summary=UniverseSummary(total_candidates=2, included=2, excluded=0),
        universe=[
            UniverseSymbol(ticker="HG=F", canonical_ticker="HG=F", category="target", status="included"),
            UniverseSymbol(ticker="CL=F", canonical_ticker="CL=F", category="commodity_energy", status="included"),
        ],
        content_fingerprint="sha256:0123456789abcdef",
        output_fingerprint="sha256:0123456789abcdef",
    )
    scr = FeatureScreener(ScreenerConfig(analysis=AnalysisConfig(window_mode="rolling")), universe)
    idx = pd.date_range("2022-01-07", periods=60, freq="W-FRI")
    scr.prices = {
        "HG=F": pd.DataFrame({"returns": pd.Series(np.zeros(len(idx)), index=idx)}),
        "CL=F": pd.DataFrame({"returns": pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)}),
    }

    cl_eval = CandidateEvaluation("CL=F")
    cl_eval.category = "commodity_energy"
    cl_eval.pairwise_obs = 260
    cl_eval.is_metrics = PeriodMetrics()
    cl_eval.is_metrics.pearson = 0.08
    cl_eval.is_metrics.n_obs = 208
    cl_eval.is_metrics.passed_retention = False
    cl_eval.oos_metrics = PeriodMetrics()
    cl_eval.oos_metrics.pearson = 0.22
    cl_eval.oos_metrics.n_obs = 52
    scr.evaluations = [cl_eval]

    scr.build_output()
    selected = scr._generate_selected_symbols()
    selected_tickers = [item["ticker"] for item in selected["selected"]]
    assert "CL=F" in selected_tickers
    assert "CL=F" in selected["mandatory_added"]


def test_latest_backtest_prefers_tft_report_and_flattens_top_level_metrics(monkeypatch):
    import asyncio
    from dataclasses import asdict
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import main
    from app.models import Base
    from backtest.runner import (
        BacktestConfig,
        BacktestMetrics,
        BacktestResult,
        _persist_report_to_db,
        _persist_tft_report_to_db,
    )

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    import app.db as app_db

    monkeypatch.setattr(app_db, "SessionLocal", Session)
    monkeypatch.setattr(main, "SessionLocal", Session)

    metrics_tft = BacktestMetrics(
        mae=0.021,
        rmse=0.029,
        n_predictions=180,
        mean_actual=0.002,
        mean_predicted=0.001,
        directional_accuracy=0.535,
        sharpe_ratio=0.45,
        mae_vs_naive_zero=0.98,
    )
    metrics_xgb = BacktestMetrics(
        mae=0.013,
        rmse=0.017,
        n_predictions=250,
        mean_actual=0.001,
        mean_predicted=0.001,
        directional_accuracy=0.49,
        sharpe_ratio=0.80,
        mae_vs_naive_zero=1.04,
    )

    _persist_tft_report_to_db(
        {
            "run_id": "tft-bt-pref",
            "xgboost": asdict(metrics_xgb),
            "tft_asro": asdict(metrics_tft),
            "tft_walkforward_summary": {"mean_da": 0.54, "mean_vr": 0.32, "n_windows": 12, "total_samples": 180},
            "windows": [{"window_id": 1, "da": 0.54, "sharpe": 0.45, "mae": 0.021, "rmse": 0.029, "variance_ratio": 0.32}],
            "theta_comparison": {"tft_da": 0.54, "theta_da": 0.48, "tft_sharpe": 0.45, "theta_sharpe": -0.1, "tft_mae": 0.021, "theta_mae": 0.026},
            "verdict": "TFT_SUPERIOR",
        }
    )

    # Persist an xgb_challenger report *after* the TFT report to prove report_kind filtering
    xgb_res = BacktestResult(
        run_id="xgb-newer",
        generated_at=(datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat(),
        config=BacktestConfig(oos_start="2025-01-01", oos_end="2025-12-31"),
        champion={"symbol_set": {"symbols": ["DX-Y.NYB"]}, "metrics": asdict(metrics_xgb)},
        challenger={"symbol_set": {"symbols": ["DX-Y.NYB"]}, "metrics": asdict(metrics_xgb)},
        delta_mae_pct=0.0,
        delta_rmse_pct=0.0,
        delta_dir_acc_pct=0.0,
        improvement_mae_pct=0.0,
        decision="MANUAL_REVIEW",
        decision_reason="test",
    )
    _persist_report_to_db(xgb_res, kind="xgb_challenger")

    payload = asyncio.run(main.get_latest_backtest("HG=F"))
    assert payload["available"] is True
    assert payload["verdict"] == "TFT_SUPERIOR"
    assert payload["summary_metrics"]["directional_accuracy"] == pytest.approx(0.535)
    assert payload["summary_metrics"]["sharpe_ratio"] == pytest.approx(0.45)
    assert payload["summary_metrics"]["mae"] == pytest.approx(0.021)
    assert payload["summary_metrics"]["rmse"] == pytest.approx(0.029)
    assert payload["summary_metrics"]["variance_ratio"] == pytest.approx(0.32)
    assert len(payload["window_metrics"]) == 1
    assert payload["theta_comparison"]["theta_da"] == pytest.approx(0.48)
    engine.dispose()


def test_tft_backtest_resolves_close_series_from_target_fallback_for_theta(monkeypatch):
    from backtest.runner import BacktestConfig, BacktestRunner, TFTBacktestRunner
    from deep_learning.baselines.theta_model import theta_backtest
    from deep_learning.config import get_tft_config

    dates = pd.bdate_range("2025-01-01", periods=140)
    rng = np.random.default_rng(12)
    log_rets = rng.normal(0.0005, 0.015, size=len(dates))
    df = pd.DataFrame({"target": log_rets}, index=dates)

    cfg = get_tft_config()
    runner = TFTBacktestRunner(BacktestConfig(oos_start="2025-05-01", oos_end="2025-07-31"))

    import app.db as app_db

    monkeypatch.setattr(app_db, "SessionLocal", lambda: (_ for _ in ()).throw(RuntimeError("no db")))
    monkeypatch.setattr(BacktestRunner, "fetch_prices", lambda *_args, **_kwargs: pd.DataFrame())

    close_series = runner._resolve_target_close_series(cfg, df, last_known_price=5.625)
    assert len(close_series) == len(df)
    assert close_series.iloc[-1] == pytest.approx(5.625)
    assert (close_series > 0).all()

    theta_res = theta_backtest(
        close_series,
        horizon=5,
        n_windows=30,
        return_mode="cumulative",
        use_log_return=True,
        annualisation=52.0,
    )
    assert "error" not in theta_res
    assert theta_res["n_windows"] == 26


def test_tft_summary_schema_preserves_variable_importance_labels():
    from app.schemas import TFTModelSummaryResponse

    model = TFTModelSummaryResponse(
        symbol="HG=F",
        checkpoint_path="/tmp/models/tft/best_tft_asro.ckpt",
        variable_importance=[
            {
                "feature": "CL=F_ema_fast_slow_spread",
                "importance": 0.2916,
                "label": "WTI Crude Oil · 5-day vs. 20-day EMA spread",
                "description": "WTI Crude Oil — 5-day vs. 20-day EMA spread",
                "category": "Trend",
                "time_horizon": "",
            }
        ],
        variable_importance_method="vsn_encoder",
    )
    dumped = model.model_dump()
    assert dumped["variable_importance"][0]["label"] == "WTI Crude Oil · 5-day vs. 20-day EMA spread"
    assert dumped["variable_importance"][0]["category"] == "Trend"


def test_format_prediction_monotonic_logits_not_flagged_as_crossing_or_anomaly():
    from deep_learning.models.tft_copper import format_prediction

    # Raw output from monotonic-transform model: delta logits [u0..u2, q50, u4..u6]
    raw_logits = np.tile(
        np.array([[2.3, 1.8, 1.9, 0.0005, 1.9, 1.8, 2.3]], dtype=np.float64),
        (5, 1),
    )
    result = format_prediction(raw_logits, baseline_price=5.625, is_monotonic_logits=True)
    assert result["quantile_crossing_detected"] is False
    assert result["raw_quantile_crossing_rate"] == 0.0
    assert result["anomaly_detected"] is False


def test_commentary_prompt_marks_xgboost_importance_as_model_weight_not_return(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from app import commentary as commentary_module

    captured: dict[str, str] = {}
    settings = SimpleNamespace(
        ifm_api_key="",
        openrouter_api_key="test-key",
        resolved_commentary_model="test/model",
        openrouter_max_retries=1,
        openrouter_rpm=18,
        openrouter_fallback_models_list=[],
    )
    monkeypatch.setattr(commentary_module, "get_settings", lambda: settings)

    async def fake_openrouter(**kwargs):
        captured["prompt"] = kwargs["messages"][1]["content"]
        return {"choices": [{"message": {"content": '{"stance":"NEUTRAL","commentary":"Summary: balanced."}'}}]}

    monkeypatch.setattr(commentary_module, "create_chat_completion", fake_openrouter)
    asyncio.run(
        commentary_module._generate_commentary_and_stance(
            current_price=5.60,
            predicted_price=5.61,
            predicted_return=0.001,
            sentiment_index=0.0,
            sentiment_label="Neutral",
            top_influencers=[{"feature": "FCX_ret1", "importance": 0.0984}],
            news_count=10,
        )
    )
    prompt = captured["prompt"]
    assert "XGBoost model importance share: 9.8%" in prompt
    assert "NOT an observed price change or volatility level" in prompt




