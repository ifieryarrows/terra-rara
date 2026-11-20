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

    n = 80
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
    assert lrs[0] == pytest.approx(
        cfg.model.learning_rate * min(0.05, 0.25 * cfg.model.output_head_lr_mult)
    )
    assert lrs[1] == pytest.approx(cfg.model.learning_rate * cfg.model.output_head_lr_mult)
    assert lrs[2] == pytest.approx(cfg.model.learning_rate)


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


def test_backtest_runner_normalizes_mixed_timezones_and_excludes_non_target_holidays():
    cfg = BacktestConfig(oos_start="2025-01-06", oos_end="2025-05-30", train_window_years=1)
    runner = BacktestRunner(cfg)

    dates = pd.bdate_range("2025-01-02", "2025-05-30")
    rng = np.random.default_rng(101)
    hg_close = 4.5 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, len(dates))))
    bhp_close = 45.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.016, len(dates))))
    anto_close = 1800.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, len(dates))))

    # Simulate mixed yfinance timezones:
    # - HG=F tz-naive midnight
    # - BHP America/New_York tz-aware (-05:00/-04:00)
    # - ANTO.L Europe/London tz-aware (+00:00/+01:00) + an extra non-US Saturday/holiday date
    ny_dates = pd.DatetimeIndex(dates).tz_localize("America/New_York")
    london_dates = pd.DatetimeIndex(dates).tz_localize("Europe/London")
    holiday_dt = pd.Timestamp("2025-04-19", tz="Europe/London")  # not in HG=F dates

    prices = pd.concat(
        [
            pd.DataFrame({"date": dates, "symbol": "HG=F", "close": hg_close}),
            pd.DataFrame({"date": ny_dates, "symbol": "BHP", "close": bhp_close}),
            pd.DataFrame(
                {
                    "date": list(london_dates) + [holiday_dt],
                    "symbol": "ANTO.L",
                    "close": list(anto_close) + [1815.0],
                }
            ),
        ],
        ignore_index=True,
    )

    feats = runner.prepare_features(prices, "HG=F")
    assert feats.index.is_unique
    assert pd.Timestamp("2025-04-19") not in feats.index
    assert len(feats) == len(dates) - 1
    assert feats["ANTO.L_ret_1d"].abs().gt(1e-9).mean() > 0.90
    zero_ret_fraction = float((feats["y_target"].abs() < 1e-12).mean())
    pos_ret_fraction = float((feats["y_target"] > 0).mean())
    assert zero_ret_fraction < 0.05
    assert 0.25 < pos_ret_fraction < 0.75


def test_theta_backtest_eval_dates_matches_exact_tft_origins():
    from deep_learning.baselines.theta_model import theta_backtest

    dates = pd.bdate_range("2025-01-01", periods=140)
    rng = np.random.default_rng(202)
    close_series = pd.Series(
        5.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=len(dates)))),
        index=dates,
        name="close",
    )
    eval_dates = dates[65:95]
    assert len(eval_dates) == 30

    res = theta_backtest(
        close_series,
        horizon=5,
        return_mode="cumulative",
        use_log_return=True,
        annualisation=52.0,
        eval_dates=eval_dates,
    )
    assert "error" not in res
    assert res["n_windows"] == 30


def test_tft_compare_with_xgboost_uses_horizon_normalized_delta_mae(monkeypatch):
    from backtest.runner import BacktestMetrics

    cfg = BacktestConfig(oos_start="2025-01-06", oos_end="2025-05-30")
    runner = TFTBacktestRunner(cfg)

    tft_metrics = BacktestMetrics(
        mae=0.025,
        rmse=0.031,
        n_predictions=200,
        mean_actual=0.003,
        mean_predicted=0.001,
        directional_accuracy=0.55,
        mae_vs_naive_zero=0.98,
        sharpe_ratio=0.82,
        metric_space="cumulative_log_return_5d",
    )
    xgb_metrics = BacktestMetrics(
        mae=0.011,
        rmse=0.015,
        n_predictions=250,
        mean_actual=0.0005,
        mean_predicted=0.0002,
        directional_accuracy=0.50,
        mae_vs_naive_zero=1.02,
        sharpe_ratio=0.40,
        metric_space="simple_return_1d",
    )

    monkeypatch.setattr(runner, "run_backtest", lambda _syms: (tft_metrics, pd.DataFrame()))
    res = runner.compare_with_xgboost(
        SymbolSet("active", ["DX-Y.NYB"], "2.0", "active.json", "sha256:abc"),
        xgb_metrics=xgb_metrics,
    )
    assert res["tft_better_mae"] is True
    assert res["delta_mae_basis"] == "mae_vs_naive_zero"
    assert res["delta_mae_pct"] == pytest.approx(-3.92, abs=0.05)


def test_ingest_prices_normalizes_tz_aware_dates_to_utc_midnight_without_duplicates(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import data_manager
    from app.models import Base, PriceBar

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    dates = pd.bdate_range("2026-11-02", periods=6)
    ny_df = pd.DataFrame(
        {"Open": 5.5, "High": 5.6, "Low": 5.4, "Close": np.linspace(5.5, 5.6, len(dates)), "Volume": 100.0},
        index=dates.tz_localize("America/New_York"),
    )
    london_df = pd.DataFrame(
        {"Open": 20.0, "High": 20.5, "Low": 19.5, "Close": np.linspace(20.0, 20.5, len(dates)), "Volume": 200.0},
        index=dates.tz_localize("Europe/London"),
    )
    naive_hg_df = ny_df.copy()
    naive_hg_df.index = dates
    naive_anto_df = london_df.copy()
    naive_anto_df.index = dates

    fake_settings = SimpleNamespace(
        symbols_list=["HG=F", "ANTO.L"],
        training_symbols=["HG=F", "ANTO.L"],
        lookback_days=30,
    )
    monkeypatch.setattr(data_manager, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(data_manager, "get_db_type", lambda: "sqlite")
    monkeypatch.setattr(data_manager, "_ensure_yf_tz_cache", lambda: None)

    phase = {"step": 1}

    def fake_fetch(symbol, *_args, **_kwargs):
        if phase["step"] == 1:
            return ny_df if symbol == "HG=F" else london_df
        return naive_hg_df if symbol == "HG=F" else naive_anto_df

    monkeypatch.setattr(data_manager, "fetch_symbol_with_retry", fake_fetch)

    try:
        with Session() as session:
            data_manager.ingest_prices(session)
            phase["step"] = 2
            data_manager.ingest_prices(session)

            bars = session.query(PriceBar).order_by(PriceBar.symbol, PriceBar.date).all()
            assert len(bars) == 12
            hg_dates = [b.date.date() for b in bars if b.symbol == "HG=F"]
            anto_dates = [b.date.date() for b in bars if b.symbol == "ANTO.L"]
            expected_dates = [d.date() for d in dates]
            assert hg_dates == expected_dates
            assert anto_dates == expected_dates
            assert all(b.date.hour == 0 and b.date.minute == 0 and b.date.second == 0 for b in bars)
    finally:
        engine.dispose()


def test_get_history_and_inference_deduplicate_same_calendar_day_price_bars(monkeypatch):
    import asyncio
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import db as app_db, inference, main
    from app.models import Base, PriceBar

    engine_dup = create_engine("sqlite:///:memory:")
    engine_clean = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine_dup)
    Base.metadata.create_all(engine_clean)
    SessionDup = sessionmaker(bind=engine_dup)
    SessionClean = sessionmaker(bind=engine_clean)

    now_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    bar_days = [now_day - timedelta(days=offset) for offset in range(25, 0, -1) if (now_day - timedelta(days=offset)).weekday() < 5][:20]
    rng = np.random.default_rng(303)
    closes = 5.4 * np.exp(np.cumsum(rng.normal(0.0, 0.015, size=len(bar_days))))

    try:
        with SessionDup() as s_dup, SessionClean() as s_clean:
            for day_dt, c_val in zip(bar_days, closes):
                # Older bar at 00:00Z, newer bar on same day at 04:00Z with slightly updated close
                updated_close = float(c_val * 1.001)
                s_dup.add(PriceBar(symbol="HG=F", date=day_dt, open=float(c_val), high=float(c_val), low=float(c_val), close=float(c_val), volume=100.0))
                s_dup.add(PriceBar(symbol="HG=F", date=day_dt + timedelta(hours=4), open=updated_close, high=updated_close, low=updated_close, close=updated_close, volume=100.0))
                s_clean.add(PriceBar(symbol="HG=F", date=day_dt + timedelta(hours=4), open=updated_close, high=updated_close, low=updated_close, close=updated_close, volume=100.0))
            s_dup.commit()
            s_clean.commit()

            fake_settings = SimpleNamespace(scoring_source="news_articles", sentiment_horizon_days=5)
            monkeypatch.setattr(main, "SessionLocal", SessionDup)
            monkeypatch.setattr(main, "get_settings", lambda: fake_settings)
            monkeypatch.setattr(app_db, "SessionLocal", SessionDup)
            monkeypatch.setattr(inference, "get_settings", lambda: fake_settings)

            hist = asyncio.run(main.get_history(symbol="HG=F", days=60))
            assert len(hist.data) == len(bar_days)
            assert [pt.date for pt in hist.data] == [d.strftime("%Y-%m-%d") for d in bar_days]
            assert hist.data[-1].price == pytest.approx(round(float(closes[-1] * 1.001), 4))

            dq_dup = inference.get_data_quality_stats(s_dup, symbol="HG=F", days=30)
            dq_clean = inference.get_data_quality_stats(s_clean, symbol="HG=F", days=30)
            assert dq_dup["missing_days"] == dq_clean["missing_days"]
            assert dq_dup["coverage_pct"] == dq_clean["coverage_pct"]

            band_dup = inference.calculate_confidence_band(s_dup, symbol="HG=F", predicted_price=5.6, lookback_days=60)
            band_clean = inference.calculate_confidence_band(s_clean, symbol="HG=F", predicted_price=5.6, lookback_days=60)
            assert band_dup[0] == pytest.approx(band_clean[0], rel=1e-9)
            assert band_dup[1] == pytest.approx(band_clean[1], rel=1e-9)
    finally:
        engine_dup.dispose()
        engine_clean.dispose()


def test_commentary_prompt_includes_latest_level_alongside_recent_delta():
    from app.commentary import _build_tft_xai_prompt_block
    from app.tft_explain import summarize_explain_for_commentary

    payload = {
        "horizon": {"label": "5-day cumulative log return (primary)", "days": 5},
        "reference_price_date": "2026-11-19",
        "variable_selection": {
            "encoder": [
                {
                    "feature": "CL=F_ret10",
                    "label": "WTI Crude Oil · 10-day price change",
                    "category": "Momentum",
                    "weight": 0.18,
                    "recent": {
                        "basis": "5 sessions",
                        "kind": "abs",
                        "last_value": -0.0842,
                        "change": 0.0777,
                        "change_pct": None,
                        "as_of": "2026-11-19",
                    },
                }
            ]
        },
        "attention": {"top_days": [], "window": {}},
    }

    digest = summarize_explain_for_commentary(payload)
    assert digest is not None
    sig = digest["top_encoder_signals"][0]
    assert sig["recent_summary"] == "+0.078 over 5 sessions"
    assert sig["recent_last_value"] == pytest.approx(-0.0842)

    prompt_block = _build_tft_xai_prompt_block(None, digest)
    assert "recent: +0.078 over 5 sessions" in prompt_block
    assert "latest level: -0.084" in prompt_block


def test_differential_optimizer_decouples_output_head_weights_from_biases():
    from pytorch_forecasting import TimeSeriesDataSet
    from deep_learning.data.dataset import _identity_target_normalizer
    from deep_learning.models.tft_copper import create_tft_model

    n = 45
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n, dtype=int),
            "group_id": "copper",
            "target": np.linspace(-0.01, 0.01, n),
            "HG_F_ret1": np.sin(np.linspace(0, 4, n)) * 0.01,
            "day_of_week": (np.arange(n) % 5).astype(float),
        }
    )
    cfg = get_tft_config()
    cfg = replace(
        cfg,
        model=replace(
            cfg.model,
            max_encoder_length=10,
            hidden_size=8,
            hidden_continuous_size=4,
            learning_rate=8e-4,
            output_head_lr_mult=0.25,
        ),
    )
    ds = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=cfg.model.max_encoder_length,
        max_prediction_length=cfg.model.max_prediction_length,
        time_varying_unknown_reals=["HG_F_ret1"],
        time_varying_known_reals=["day_of_week"],
        static_categoricals=["group_id"],
        target_normalizer=_identity_target_normalizer(),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
    )
    model = create_tft_model(ds, cfg, use_asro=True, weekly_target_scale=0.025)
    opt_cfg = model.configure_optimizers()
    groups = opt_cfg["optimizer"].param_groups
    lrs = sorted(group["lr"] for group in groups)

    expected_bias_lr = 8e-4 * 0.05
    expected_weight_lr = 8e-4 * 0.25
    expected_body_lr = 8e-4
    assert len(groups) == 3
    assert lrs[0] == pytest.approx(expected_bias_lr, rel=1e-6)
    assert lrs[1] == pytest.approx(expected_weight_lr, rel=1e-6)
    assert lrs[2] == pytest.approx(expected_body_lr, rel=1e-6)
    assert expected_weight_lr > expected_bias_lr


def test_weekly_asro_loss_t1_and_weekly_pinball_share_same_scale_convention():
    from deep_learning.models.monotonic_quantiles import enforce_monotonic_quantiles

    torch.manual_seed(17)
    y_pred = torch.randn(16, 5, 7) * 0.015
    y_actual = torch.randn(16, 5) * 0.012
    ordered = enforce_monotonic_quantiles(y_pred, median_idx=3)

    actual_weekly = y_actual.sum(dim=1)
    scale_w = float((actual_weekly ** 2).mean().sqrt().clamp_min(0.005))
    scale_1d = scale_w / np.sqrt(5.0)

    loss_fn = WeeklyASROPFLoss(
        quantiles=list(QUANTILES),
        lambda_weekly_quantile=0.0,
        lambda_t1_quantile=1.0,
        lambda_t1_directional=0.0,
        lambda_dispersion=0.0,
        lambda_magnitude=0.0,
        lambda_naive=0.0,
        lambda_bias=0.0,
        lambda_drift_prior=0.0,
        lambda_directional=0.0,
        lambda_saturation=0.0,
        lambda_positive_rate=0.0,
        lambda_interval=0.0,
    )
    actual_t1_loss = float(loss_fn.loss(y_pred, y_actual))
    expected_t1_loss = float(loss_fn._pinball(ordered[:, 0, :], y_actual[:, 0]) / scale_1d)
    assert actual_t1_loss == pytest.approx(expected_t1_loss, rel=1e-5, abs=1e-6)


def test_mrmr_select_enforces_per_group_diversity_cap():
    from deep_learning.data.feature_selection import select_features

    rng = np.random.default_rng(42)
    n = 160
    target = rng.normal(0.0, 0.025, size=n)
    data = {
        "time_idx": np.arange(n),
        "group_id": "copper",
        "target_5d_log_return": target,
        "HG_F_ret1": rng.normal(0.0, 0.01, size=n),
        "HG_F_ret5": rng.normal(0.0, 0.02, size=n),
        "HG_F_ret10": rng.normal(0.0, 0.025, size=n),
        "HG_F_ret20": rng.normal(0.0, 0.03, size=n),
        "HG_F_close_sma_20_ratio": rng.normal(0.0, 0.02, size=n),
        "HG_F_vol_10": np.abs(rng.normal(0.01, 0.005, size=n)),
    }
    # 5 independent strong CL_F features that would otherwise consume all 6 free slots
    for idx in range(5):
        data[f"CL_F_feat_{idx}"] = target * (0.45 - 0.03 * idx) + rng.normal(0.0, 0.015, size=n)
    data["FCX_ret10"] = target * 0.35 + rng.normal(0.0, 0.02, size=n)
    data["DX_Y_NYB_ret10"] = -target * 0.30 + rng.normal(0.0, 0.02, size=n)
    data["emb_pca_0"] = target * 0.28 + rng.normal(0.0, 0.02, size=n)
    data["sentiment_index"] = target * 0.25 + rng.normal(0.0, 0.02, size=n)
    data["regime_usd_pressure"] = -target * 0.22 + rng.normal(0.0, 0.02, size=n)

    df = pd.DataFrame(data)
    forced = [
        "HG_F_ret1",
        "HG_F_ret5",
        "HG_F_ret10",
        "HG_F_ret20",
        "HG_F_close_sma_20_ratio",
        "HG_F_vol_10",
    ]
    _filtered, selected_unknown, _ = select_features(
        df,
        target_col="target_5d_log_return",
        mrmr_top_k=12,
        known_features=[],
        forced_unknown_features=forced,
        max_per_group=2,
        group_prefixes=["HG_F", "CL_F", "FCX", "DX_Y_NYB"],
    )
    cl_selected = [c for c in selected_unknown if c.startswith("CL_F_")]
    assert len(cl_selected) <= 2
    assert any(c.startswith("FCX_") for c in selected_unknown)
    assert any(c.startswith("DX_Y_NYB_") for c in selected_unknown)
    assert any(c.startswith("emb_pca_") or c == "sentiment_index" or c.startswith("regime_") for c in selected_unknown)


def test_optuna_weekly_objective_prefers_signal_model_over_flat_hedger_and_overdispersed():
    from deep_learning.training.hyperopt import _compute_weekly_fold_objective

    _, signal_score = _compute_weekly_fold_objective(
        weekly_pinball=0.0066,
        weekly_actual_std=0.032,
        weekly_interval_score_80=0.115,
        weekly_mae_vs_naive_zero=0.975,
        weekly_mr=0.22,
        weekly_pred_positive_rate=0.58,
        weekly_actual_positive_rate=0.53,
        weekly_sharpe=0.95,
        weekly_da=0.58,
        weekly_sign_corr=0.18,
    )
    _, flat_hedger_score = _compute_weekly_fold_objective(
        weekly_pinball=0.0070,
        weekly_actual_std=0.032,
        weekly_interval_score_80=0.117,
        weekly_mae_vs_naive_zero=0.993,
        weekly_mr=0.06,
        weekly_pred_positive_rate=0.85,
        weekly_actual_positive_rate=0.53,
        weekly_sharpe=0.28,
        weekly_da=0.51,
        weekly_sign_corr=0.04,
    )
    _, overdispersed_score = _compute_weekly_fold_objective(
        weekly_pinball=0.0075,
        weekly_actual_std=0.032,
        weekly_interval_score_80=0.125,
        weekly_mae_vs_naive_zero=1.20,
        weekly_mr=1.10,
        weekly_pred_positive_rate=0.52,
        weekly_actual_positive_rate=0.53,
        weekly_sharpe=0.30,
        weekly_da=0.52,
        weekly_sign_corr=0.05,
    )

    assert signal_score < flat_hedger_score
    assert signal_score < overdispersed_score


def test_tft_datasets_disable_constant_static_bias_channels():
    from deep_learning.data.dataset import build_cv_folds, build_datasets
    from deep_learning.models.tft_copper import create_tft_model

    n = 160
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n, dtype=int),
            "group_id": "copper",
            "target": np.sin(np.linspace(0, 12, n)).astype(np.float32) * 0.015,
            "HG_F_ret1": np.cos(np.linspace(0, 12, n)).astype(np.float32) * 0.012,
            "day_of_week": (np.arange(n) % 5).astype(np.float32),
        }
    )
    cfg = get_tft_config()
    cfg = replace(cfg, model=replace(cfg.model, max_encoder_length=10))
    tr_ds, val_ds, _test_ds = build_datasets(df, ["HG_F_ret1"], ["day_of_week"], ["target"], cfg)
    folds = build_cv_folds(df, ["HG_F_ret1"], ["day_of_week"], ["target"], cfg, n_folds=2)

    for ds_item in (tr_ds, val_ds, folds[0][0], folds[0][1]):
        assert "relative_time_idx" not in ds_item.reals
        assert "target_center" not in ds_item.reals
        assert "target_scale" not in ds_item.reals
        assert "encoder_length" not in ds_item.reals

    model = create_tft_model(tr_ds, cfg, use_asro=True, weekly_target_scale=0.03)
    assert list(model.static_variables) == ["group_id"]
    assert "relative_time_idx" not in model.encoder_variables
    assert "relative_time_idx" not in model.decoder_variables


def test_signed_risk_adjusted_features_macro_zscore_and_unsigned_vol_exclusion():
    from deep_learning.calibration.hybrid import (
        compute_causal_macro_fair_value_zscore,
        compute_causal_theta_5d_series,
    )
    from deep_learning.data.feature_store import _is_unsigned_vol_feature

    n = 90
    idx = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(7)
    hg_ret = rng.normal(0.0, 0.012, n)
    hg_px = pd.DataFrame({"close": 4.5 * np.exp(np.cumsum(hg_ret))}, index=idx)
    macro_px = pd.DataFrame({"close": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, n)))}, index=idx)

    hg_feat = generate_symbol_features(hg_px, "HG=F", include_raw_ma=False)
    macro_feat = generate_symbol_features(macro_px, "DX-Y.NYB", include_raw_ma=False)
    combined = pd.concat([hg_feat, macro_feat], axis=1)

    assert "HG=F_risk_adj_ret5" in hg_feat.columns
    assert "HG=F_risk_adj_ret10" in hg_feat.columns
    assert "HG=F_risk_adj_ret20" in hg_feat.columns
    assert "HG=F_signed_vol_ratio" in hg_feat.columns
    assert hg_feat["HG=F_risk_adj_ret10"].dropna().between(-5.0, 5.0).all()

    fv = compute_causal_macro_fair_value_zscore(combined, target_prefix="HG=F", window=50, min_periods=30)
    assert len(fv) == n
    assert float(fv.iloc[-30:].std()) > 0.1
    assert fv.between(-4.0, 4.0).all()

    th = compute_causal_theta_5d_series(hg_px["close"], horizon=5, min_window=40, max_eval_tail=20)
    assert len(th) == n
    assert (th.iloc[:-20] == 0.0).all()
    assert th.iloc[-20:].abs().max() > 0.0

    assert _is_unsigned_vol_feature("CEW_vol_10") is True
    assert _is_unsigned_vol_feature("HG_F_vol_5_20_ratio") is True
    assert _is_unsigned_vol_feature("HG_F_signed_vol_ratio") is False
    assert _is_unsigned_vol_feature("HG_F_risk_adj_ret10") is False


def test_demeaned_asro_directional_loss_is_invariant_to_constant_bullish_drift():
    loss_fn = WeeklyASROPFLoss(
        quantiles=list(QUANTILES),
        lambda_weekly_quantile=0.0,
        lambda_t1_quantile=0.0,
        lambda_t1_directional=0.0,
        lambda_dispersion=0.0,
        lambda_magnitude=0.0,
        lambda_naive=0.0,
        lambda_bias=0.0,
        lambda_drift_prior=0.0,
        lambda_directional=1.0,
    )
    # Positive drift in actuals (+0.008 mean) with zero conditional spread across two constant forecasts
    actual = torch.tensor(
        [
            [0.004, 0.004, 0.004, 0.004, 0.004],
            [-0.002, -0.002, -0.002, -0.002, -0.002],
            [0.005, 0.005, 0.005, 0.005, 0.005],
            [-0.001, -0.001, -0.001, -0.001, -0.001],
        ],
        dtype=torch.float32,
    )
    const_zero = torch.zeros(4, 5, 7, dtype=torch.float32)
    const_bull = torch.zeros(4, 5, 7, dtype=torch.float32)
    const_bull[..., 3] = 0.005  # Constant always-positive prediction

    # Constant predictions yield identical demeaned loss (no free reward for constant bullish bias)
    assert torch.isclose(loss_fn.loss(const_bull, actual), loss_fn.loss(const_zero, actual), atol=1e-6)

    # Conditional covariance with demeaned returns reduces loss strictly
    actual_weekly = actual.sum(dim=1)
    centered_sig = (actual_weekly - actual_weekly.mean()) / 5.0
    good_cond = torch.zeros(4, 5, 7, dtype=torch.float32)
    good_cond[..., 3] = centered_sig.unsqueeze(1)
    assert loss_fn.loss(good_cond, actual) < loss_fn.loss(const_bull, actual)


def test_theta_hybrid_blend_preserves_monotonicity_and_enforces_tft_floor():
    from deep_learning.calibration.hybrid import (
        apply_theta_hybrid_shift_to_path,
        fit_theta_hybrid_blend,
    )
    from deep_learning.models.monotonic_quantiles import enforce_monotonic_quantiles

    rng = np.random.default_rng(19)
    n = 90
    theta_sig = rng.normal(0.0, 0.025, n)
    actual = 0.7 * theta_sig + rng.normal(0.0, 0.015, n)
    tft_noisy = rng.normal(0.0, 0.005, n)

    blend = fit_theta_hybrid_blend(actual, tft_noisy, theta_sig, min_tft_weight=0.35, max_theta_weight=0.65)
    assert blend["enabled"] is True
    assert blend["w_tft"] >= 0.35
    assert blend["w_theta"] <= 0.65
    assert blend["val_hybrid_da"] > blend["val_tft_da"]

    raw_path = np.zeros((10, 5, 7), dtype=np.float64)
    raw_path[..., :] = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    raw_path[..., 3] = tft_noisy[:10, None] / 5.0

    shifted = apply_theta_hybrid_shift_to_path(
        raw_path,
        theta_sig[:10],
        hybrid_blend=blend,
        quantiles=QUANTILES,
        horizon=5,
    )
    ordered = enforce_monotonic_quantiles(torch.tensor(shifted, dtype=torch.float32), median_idx=3)
    weekly = aggregate_weekly_quantiles_np(ordered.numpy(), median_idx=3)
    assert np.all(np.diff(weekly, axis=1) > 0.0)
    expected_med = blend["w_tft"] * tft_noisy[:10] + blend["w_theta"] * theta_sig[:10]
    assert np.allclose(weekly[:, 3], expected_med, atol=1e-5)


def test_tft_backtest_runner_separates_strict_oos_from_validation_overlap(tmp_path, monkeypatch):
    dates = pd.bdate_range("2025-11-03", periods=70)
    rng = np.random.default_rng(99)
    closes = 5.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(dates))))
    eval_df = pd.DataFrame(
        {
            "target": np.concatenate([[0.0], np.diff(np.log(closes))]),
            "HG_F_ret1": rng.normal(0.0, 0.01, len(dates)),
            "HG_F_theta_5d_prior": np.full(len(dates), 0.015),
            "day_of_week": dates.dayofweek.astype(float),
            "realized_vol_20d": np.full(len(dates), 0.02),
            "time_idx": np.arange(len(dates), dtype=int),
            "group_id": "copper",
        },
        index=dates,
    )

    val_end_date = dates[29].date().isoformat()
    (tmp_path / "best_tft_asro.ckpt").write_text("stub")
    (tmp_path / "tft_metadata.json").write_text(
        json.dumps(
            {
                "split_dates": {"val_end_date": val_end_date},
                "hybrid_blend": {
                    "enabled": True,
                    "prior_feature": "HG=F_theta_5d_prior",
                    "w_tft": 0.5,
                    "w_theta": 0.5,
                },
                "ensemble_checkpoints": ["best_tft_asro.ckpt"],
            }
        )
    )

    class FakeModel:
        class loss:
            quantiles = list(QUANTILES)

        def predict(self, dl, mode="quantiles", return_y=False):
            n_items = len(dl.dataset)
            preds = torch.zeros((n_items, 5, 7), dtype=torch.float32)
            preds[..., :] = torch.tensor([-2.5, -1.8, -1.5, 0.001, -1.5, -1.8, -2.5])
            if return_y:
                ys = torch.cat([b[1][0] for b in dl])
                return SimpleNamespace(output=preds, y=(ys,))
            return preds

    runner = TFTBacktestRunner(
        BacktestConfig(
            oos_start=dates[15].date().isoformat(),
            oos_end=dates[-1].date().isoformat(),
        )
    )
    monkeypatch.setattr(runner, "_ensure_tft_artifacts", lambda _cfg: tmp_path / "best_tft_asro.ckpt")
    monkeypatch.setattr(runner, "_build_eval_dataframe", lambda _cfg, _feats: eval_df.copy())
    monkeypatch.setattr(
        runner,
        "_resolve_target_close_series",
        lambda _cfg, _df, last_known_price=float("nan"): pd.Series(closes, index=dates),
    )
    monkeypatch.setattr("deep_learning.models.tft_copper.load_tft_model", lambda _p: FakeModel())
    monkeypatch.setattr(
        "deep_learning.inference.predictor._checkpoint_dataset_parameters",
        lambda _m: {
            "time_idx": "time_idx",
            "target": "target",
            "group_ids": ["group_id"],
            "max_encoder_length": 10,
            "max_prediction_length": 5,
            "time_varying_unknown_reals": ["HG_F_ret1"],
            "time_varying_known_reals": ["day_of_week"],
            "static_categoricals": ["group_id"],
            "allow_missing_timesteps": True,
        },
    )
    monkeypatch.setattr(
        "deep_learning.inference.predictor._checkpoint_feature_contract",
        lambda _p: (["HG_F_ret1"], ["day_of_week"]),
    )

    metrics, pred_df = runner.run_backtest(["DX-Y.NYB"])
    assert len(pred_df) > runner.last_strict_oos_report["n_predictions"] >= 15
    assert runner.last_strict_oos_report["val_end_date"] == val_end_date
    assert runner.last_strict_oos_report["overlap_excluded_windows"] > 0
    # Blended median = 0.5*(5*0.001) + 0.5*0.015 = 0.010
    assert pred_df["pred_return"].iloc[0] == pytest.approx(0.010, abs=1e-4)
    assert metrics.mae >= 0.0


def test_dedup_price_bars_dry_run_and_apply_on_sqlite():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import Base
    from app.models import PriceBar
    from scripts.dedup_price_bars import dedup_price_bars

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    try:
        with Session() as session:
            session.add_all(
                [
                    PriceBar(symbol="HG=F", date=datetime(2025, 11, 18, 5, 0), close=4.50),
                    PriceBar(symbol="HG=F", date=datetime(2025, 11, 18, 17, 0), close=4.55),
                    PriceBar(symbol="HG=F", date=datetime(2025, 11, 19, 0, 0), close=4.60),
                ]
            )
            session.commit()

            dry = dedup_price_bars(session, dry_run=True)
            assert dry["mode"] == "dry_run"
            assert dry["duplicate_groups"] == 1
            assert dry["rows_to_delete"] == 1
            assert session.query(PriceBar).count() == 3

            applied = dedup_price_bars(session, dry_run=False)
            assert applied["mode"] == "apply"
            assert applied["rows_deleted"] == 1
            remaining = session.query(PriceBar).order_by(PriceBar.date.asc()).all()
            assert len(remaining) == 2
            assert remaining[0].close == pytest.approx(4.55)
            assert remaining[0].date == datetime(2025, 11, 18, 0, 0)
    finally:
        engine.dispose()







