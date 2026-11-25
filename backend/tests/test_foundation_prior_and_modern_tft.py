"""Unit & parity tests for Modern TFT-ASRO defaults and fail-open Chronos-2 prior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from deep_learning.calibration.foundation_prior import (
    CHRONOS2_MODEL_ID,
    _map_9_to_7_log_quantiles,
    apply_foundation_prior_to_ordered_path,
    evaluate_foundation_prior_on_validation,
    is_foundation_prior_runtime_allowed,
    predict_chronos2_weekly_log_quantiles,
)
from deep_learning.config import get_tft_config
from deep_learning.models.monotonic_quantiles import aggregate_weekly_quantiles_np
from deep_learning.training.hyperopt import KNOWN_GOOD_TRIAL_PARAMS, create_trial_config

QUANTILES = get_tft_config().model.quantiles


def test_modern_tft_defaults_and_hyperopt_search_space_guardrails():
    cfg = get_tft_config()
    assert cfg.model.output_head_lr_mult == pytest.approx(0.35)
    assert cfg.model.weight_decay == pytest.approx(1.5e-2)
    assert cfg.weekly_loss.lambda_drift_prior == pytest.approx(0.75)
    assert cfg.weekly_loss.lambda_directional == pytest.approx(0.12)
    assert cfg.feature_store.mrmr_top_k == 5
    assert cfg.foundation_prior.enabled is True
    assert cfg.foundation_prior.model_id == CHRONOS2_MODEL_ID
    assert cfg.foundation_prior.weight == pytest.approx(0.20)

    assert KNOWN_GOOD_TRIAL_PARAMS["lambda_drift_prior"] == pytest.approx(0.75)
    assert KNOWN_GOOD_TRIAL_PARAMS["output_head_lr_mult"] == pytest.approx(0.35)
    assert KNOWN_GOOD_TRIAL_PARAMS["mrmr_top_k"] == 5

    recorded: dict[str, list] = {}

    class _FakeTrial:
        number = 7

        def suggest_categorical(self, name, choices):
            recorded[name] = list(choices)
            return choices[0]

        def suggest_float(self, name, low, high, log=False):
            recorded[name] = [low, high, log]
            return 5e-4

    trial_cfg = create_trial_config(_FakeTrial(), cfg)
    assert recorded["mrmr_top_k"] == [5]
    assert min(recorded["lambda_drift_prior"]) >= 0.75
    assert 0.35 in recorded["output_head_lr_mult"]
    assert trial_cfg.feature_store.mrmr_top_k == 5
    assert trial_cfg.foundation_prior.enabled is True


def test_map_9_to_7_log_quantiles_monotonicity_and_tail_mapping():
    anchors = np.array([4.0, 5.0])
    # 9 quantile levels 0.1..0.9 centered around anchor*exp(+0.02) and anchor*exp(-0.01)
    q9_row0 = 4.0 * np.exp(np.array([-0.04, -0.025, -0.015, 0.0, 0.02, 0.03, 0.04, 0.05, 0.07]))
    q9_row1 = 5.0 * np.exp(np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]))
    q7 = _map_9_to_7_log_quantiles(np.vstack([q9_row0, q9_row1]), anchors)

    assert q7.shape == (2, 7)
    assert q7[0, 1] == pytest.approx(-0.04, abs=1e-6)
    assert q7[0, 3] == pytest.approx(0.02, abs=1e-6)
    assert q7[0, 5] == pytest.approx(0.07, abs=1e-6)
    assert q7[1, 3] == pytest.approx(-0.01, abs=1e-6)
    assert np.all(np.diff(q7, axis=-1) >= -1e-12)


def test_foundation_prior_runtime_guard_and_fail_open(monkeypatch):
    # Under pytest, runtime Chronos-2 downloads must stay disabled unless explicitly forced.
    assert is_foundation_prior_runtime_allowed(True) is False
    monkeypatch.setenv("COPPER_FORCE_CHRONOS2_IN_TESTS", "1")
    assert is_foundation_prior_runtime_allowed(True) is True
    monkeypatch.setenv("COPPER_DISABLE_CHRONOS2", "1")
    assert is_foundation_prior_runtime_allowed(True) is False
    monkeypatch.delenv("COPPER_DISABLE_CHRONOS2", raising=False)
    monkeypatch.delenv("COPPER_FORCE_CHRONOS2_IN_TESTS", raising=False)

    # Strict fail-open when runtime disabled or pipeline raises
    closes = np.linspace(4.0, 5.0, 60)
    assert predict_chronos2_weekly_log_quantiles(closes, [40, 50], allow_runtime=False) is None

    class _BrokenPipe:
        def predict_quantiles(self, **_kwargs):
            raise RuntimeError("simulated upstream failure")

    assert (
        predict_chronos2_weekly_log_quantiles(
            closes, [40, 50], pipeline=_BrokenPipe(), allow_runtime=True
        )
        is None
    )


def test_predict_chronos2_causal_slicing_and_list_output_handling():
    # Verify slice passed to pipeline stops strictly at closes[:pos + 1] (zero lookahead).
    closes = 4.0 + 0.01 * np.arange(100, dtype=np.float64)
    # Inject extreme future jump after pos=50 — causal slice at pos=50 must not see it.
    closes[51:] = 999.0
    seen_last_values: list[float] = []

    class _RecordingPipe:
        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels):
            assert prediction_length == 5
            assert len(quantile_levels) == 9
            out_list = []
            for series in inputs:
                arr = series.detach().cpu().numpy()
                seen_last_values.append(float(arr[-1]))
                offsets = np.linspace(0.96, 1.04, 9)
                step_block = np.tile(arr[-1] * offsets, (prediction_length, 1))
                out_list.append(torch.tensor(step_block, dtype=torch.float32).unsqueeze(0))
            return out_list, None

    q7 = predict_chronos2_weekly_log_quantiles(
        closes,
        [25, 50],
        horizon=5,
        context_len=30,
        pipeline=_RecordingPipe(),
        allow_runtime=True,
    )
    assert q7 is not None
    assert q7.shape == (2, 7)
    assert seen_last_values == [pytest.approx(closes[25]), pytest.approx(closes[50])]
    assert np.all(np.diff(q7, axis=-1) >= -1e-12)


def test_apply_foundation_prior_preserves_rss_weekly_quantiles_and_monotonicity():
    rng = np.random.default_rng(123)
    n, h, q_dim = 6, 5, 7
    med_idx = 3
    # Construct strictly ordered daily quantile path (N, 5, 7)
    daily_med = rng.normal(0.001, 0.004, size=(n, h, 1))
    offsets = np.array([-0.018, -0.012, -0.006, 0.0, 0.006, 0.012, 0.018])[None, None, :]
    pred_path = daily_med + offsets

    fnd_weekly = np.tile(
        np.array([-0.06, -0.04, -0.02, 0.025, 0.04, 0.07, 0.10]), (n, 1)
    )
    # Row 2 has NaN -> must remain untouched at TFT forecast
    fnd_weekly[2, :] = np.nan

    fnd_cfg = {"enabled": True, "weight": 0.20}
    w_tft_before = aggregate_weekly_quantiles_np(pred_path, median_idx=med_idx)
    blended_path = apply_foundation_prior_to_ordered_path(
        pred_path,
        fnd_weekly,
        foundation_prior=fnd_cfg,
        quantiles=QUANTILES,
        horizon=h,
    )
    w_after = aggregate_weekly_quantiles_np(blended_path, median_idx=med_idx)

    # Zero crossings at daily AND weekly level
    assert np.all(np.diff(blended_path, axis=-1) >= -1e-12)
    assert np.all(np.diff(w_after, axis=-1) >= -1e-12)
    # Valid rows match exact convex target (0.8 * W_tft + 0.2 * W_fnd)
    expected_valid = 0.80 * w_tft_before[0] + 0.20 * fnd_weekly[0]
    np.testing.assert_allclose(w_after[0], expected_valid, atol=1e-6)
    # NaN row left untouched
    np.testing.assert_allclose(w_after[2], w_tft_before[2], atol=1e-12)

    # Also test 2D (H, Q) path for live predictor parity
    blended_2d = apply_foundation_prior_to_ordered_path(
        pred_path[0],
        fnd_weekly[0],
        foundation_prior=fnd_cfg,
        quantiles=QUANTILES,
        horizon=h,
    )
    assert blended_2d.shape == (h, q_dim)
    np.testing.assert_allclose(blended_2d, blended_path[0], atol=1e-12)


def test_evaluate_foundation_prior_sanity_guards():
    rng = np.random.default_rng(9)
    n = 60
    y = rng.normal(0.0, 0.025, size=n)
    q_tft = np.stack(
        [y * 0.2 - 0.03, y * 0.2 - 0.02, y * 0.2 - 0.01, y * 0.2, y * 0.2 + 0.01, y * 0.2 + 0.02, y * 0.2 + 0.03],
        axis=-1,
    )
    q_good_fnd = np.stack(
        [y * 0.5 - 0.04, y * 0.5 - 0.025, y * 0.5 - 0.01, y * 0.5, y * 0.5 + 0.01, y * 0.5 + 0.025, y * 0.5 + 0.04],
        axis=-1,
    )
    ok_res = evaluate_foundation_prior_on_validation(y, q_tft, q_good_fnd, weight=0.20)
    assert ok_res["enabled"] is True
    assert ok_res["weight"] == pytest.approx(0.20)

    # Exploding foundation error must trip guard
    q_bad_fnd = q_good_fnd + 0.20
    bad_res = evaluate_foundation_prior_on_validation(y, q_tft, q_bad_fnd, weight=0.20)
    assert bad_res["enabled"] is False
    assert bad_res["weight"] == 0.0


def test_predictor_and_backtest_apply_foundation_prior_identically(tmp_path, monkeypatch):
    from backtest.runner import BacktestConfig, TFTBacktestRunner
    from deep_learning.contract import (
        FORECAST_CONTRACT_VERSION,
        RETURN_SPACE,
        TARGET_RETURN_TYPE,
    )
    from deep_learning.inference.predictor import TFTPredictor

    dates = pd.bdate_range("2025-11-03", periods=45)
    closes = 4.5 * np.exp(np.linspace(0.0, 0.12, len(dates)))
    eval_df = pd.DataFrame(
        {
            "target": np.concatenate([[0.0], np.diff(np.log(closes))]),
            "HG_F_ret1": np.full(len(dates), 0.001),
            "realized_vol_20d": np.full(len(dates), 0.02),
            "target_close": closes,
            "day_of_week": (dates.dayofweek.astype(float) / 4.0),
            "time_idx": np.arange(len(dates), dtype=int),
            "group_id": "copper",
        },
        index=dates,
    )

    ckpt_1 = tmp_path / "best_tft_asro.ckpt"
    ckpt_1.write_text("dummy")
    meta = {
        "forecast_contract_version": FORECAST_CONTRACT_VERSION,
        "target_return_type": TARGET_RETURN_TYPE,
        "primary_horizon_days": 5,
        "return_space": RETURN_SPACE,
        "config": {"weekly_median_cap": 0.20, "monotonic_quantile_transform": True},
        "direction_calibration": {"direction_sign_multiplier": 1, "weekly_sign_threshold": 0.0, "weekly_magnitude_scale": 1.0},
        "magnitude_calibration": {"scale": 1.0},
        "interval_calibration": {"weekly_interval_scale": 1.0},
        "hybrid_blend": {"enabled": False, "w_theta": 0.0},
        "foundation_prior": {
            "enabled": True,
            "provider": "chronos2",
            "model_id": CHRONOS2_MODEL_ID,
            "weight": 0.20,
            "context_len": 252,
        },
        "ensemble_checkpoints": [ckpt_1.name],
        "test_metrics": {"ensemble_size": 1},
    }
    (tmp_path / "tft_metadata.json").write_text(json.dumps(meta))
    (tmp_path / "conformal_calibration.json").write_text(
        json.dumps({"cqr_enabled": False, "global_adjustment": 0.0})
    )

    # Ordered daily logits whose raw weekly median is 5 * 0.002 = +0.010
    ordered_row = np.array([-0.020, -0.012, -0.005, 0.002, 0.007, 0.014, 0.022], dtype=np.float32)

    class _FakeModel:
        class loss:
            quantiles = list(QUANTILES)
            emits_ordered_quantiles = True

        def predict(self, dl, mode="quantiles", return_y=False, **_kw):
            n_items = len(dl.dataset)
            preds = torch.tensor(ordered_row).view(1, 1, -1).repeat(n_items, 5, 1)
            if return_y:
                ys = torch.cat([b[1][0] for b in dl])
                return SimpleNamespace(output=preds, y=(ys,))
            return preds

    fake_model = _FakeModel()
    ds_params = {
        "time_idx": "time_idx",
        "target": "target",
        "group_ids": ["group_id"],
        "max_encoder_length": 10,
        "max_prediction_length": 5,
        "time_varying_unknown_reals": ["HG_F_ret1"],
        "time_varying_known_reals": ["day_of_week"],
        "static_categoricals": ["group_id"],
        "allow_missing_timesteps": True,
    }

    # Stub Chronos-2 prediction with fixed 5D cumulative log-return quantiles (median = +0.060)
    fixed_fnd_row = np.array([-0.03, 0.00, 0.03, 0.06, 0.09, 0.12, 0.15], dtype=np.float64)
    monkeypatch.setattr(
        "deep_learning.calibration.foundation_prior.predict_chronos2_weekly_log_quantiles",
        lambda _closes, pos, **_kw: np.tile(fixed_fnd_row, (len(pos), 1)),
    )
    monkeypatch.setattr("deep_learning.models.tft_copper.load_tft_model", lambda _p: fake_model)
    monkeypatch.setattr("deep_learning.inference.predictor._checkpoint_dataset_parameters", lambda _m: ds_params)
    monkeypatch.setattr("deep_learning.inference.predictor._checkpoint_feature_contract", lambda _p: (["HG_F_ret1"], ["day_of_week"]))

    # 1. Backtest runner check: 0.80 * 0.010 + 0.20 * 0.060 = +0.020
    runner = TFTBacktestRunner(
        BacktestConfig(oos_start=dates[15].date().isoformat(), oos_end=dates[-1].date().isoformat())
    )
    monkeypatch.setattr(runner, "_ensure_tft_artifacts", lambda _cfg: ckpt_1)
    monkeypatch.setattr(runner, "_build_eval_dataframe", lambda _cfg, _feats: eval_df.copy())
    monkeypatch.setattr(runner, "_resolve_target_close_series", lambda _cfg, _df, last_known_price=float("nan"): pd.Series(closes, index=dates))
    _, pred_df = runner.run_backtest(["DX-Y.NYB"])
    assert pred_df["pred_return"].iloc[0] == pytest.approx(0.020, abs=1e-6)

    # 2. Live predictor check: exact same blended weekly log return +0.020
    monkeypatch.setattr("deep_learning.models.hub.validate_artifact_manifest", lambda _d: True)
    monkeypatch.setattr("deep_learning.models.hub.validate_tft_artifact_set", lambda _d, **_kw: True)
    monkeypatch.setattr(
        "deep_learning.data.feature_store.build_tft_dataframe",
        lambda _s, _cfg, **_kw: (eval_df.copy(), ["HG_F_ret1"], ["day_of_week"], ["target"], float(closes[-1])),
    )
    monkeypatch.setattr(
        "deep_learning.data.future_frame.build_future_decoder_rows",
        lambda hist, pred_len, _cfg: hist.tail(pred_len).assign(day_of_week=0.25).set_index(
            pd.bdate_range(hist.index[-1] + pd.offsets.BDay(1), periods=pred_len)
        ),
    )

    predictor = TFTPredictor(checkpoint_path=str(ckpt_1))
    monkeypatch.setattr(predictor, "_check_price_freshness", lambda _s, _sym: (0, False))
    monkeypatch.setattr(predictor, "_latest_price_baseline", lambda _s, _sym, fallback_price=None, fallback_date=None: (float(closes[-1]), dates[-1].date().isoformat()))
    res = predictor.predict(session=None, symbol="HG=F", include_explain=False)
    assert res["weekly_log_return"] == pytest.approx(0.020, abs=1e-6)
    assert res["model_info"]["foundation_prior_applied"] is True
    assert res["model_info"]["foundation_prior_weight"] == pytest.approx(0.20)


def test_predictor_real_multi_seed_ensemble_does_not_leak_predict_callback(tmp_path, monkeypatch):
    import lightning.pytorch as pl
    from deep_learning.contract import (
        FORECAST_CONTRACT_VERSION,
        RETURN_SPACE,
        TARGET_RETURN_TYPE,
    )
    from deep_learning.data.dataset import build_datasets
    from deep_learning.data.future_frame import build_future_decoder_rows
    from deep_learning.inference.predictor import TFTPredictor
    from deep_learning.models.hub import write_artifact_manifest
    from deep_learning.models.tft_copper import create_tft_model

    n = 160
    dates = pd.bdate_range("2025-01-02", periods=n)
    rng = np.random.default_rng(42)
    closes = 4.5 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    df = pd.DataFrame(
        {
            "target": np.concatenate([[0.0], np.diff(np.log(closes))]),
            "target_close": closes,
            "HG=F_ret1": np.concatenate([[0.0], np.diff(np.log(closes))]),
            "HG=F_ret5": pd.Series(np.log(closes)).diff(5).fillna(0.0).to_numpy(),
            "realized_vol_20d": np.full(n, 0.015),
            "day_of_week": (dates.dayofweek.astype(float) / 4.0),
            "time_idx": np.arange(n, dtype=int),
            "group_id": "copper",
        },
        index=dates,
    )

    cfg = get_tft_config()
    tr_ds, val_ds, _ = build_datasets(df, ["HG=F_ret1", "HG=F_ret5"], ["day_of_week"], ["target"], cfg)
    model = create_tft_model(tr_ds, cfg, use_asro=True, weekly_target_scale=0.03)
    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(
        model,
        tr_ds.to_dataloader(train=True, batch_size=32, num_workers=0),
        val_ds.to_dataloader(train=False, batch_size=32, num_workers=0),
    )

    ckpt_1 = tmp_path / "best_tft_asro.ckpt"
    ckpt_2 = tmp_path / "best_tft_asro_seed1.ckpt"
    trainer.save_checkpoint(str(ckpt_1))
    trainer.save_checkpoint(str(ckpt_2))
    (tmp_path / "tft_metadata.json").write_text(
        json.dumps(
            {
                "forecast_contract_version": FORECAST_CONTRACT_VERSION,
                "target_return_type": TARGET_RETURN_TYPE,
                "primary_horizon_days": 5,
                "return_space": RETURN_SPACE,
                "config": {"weekly_median_cap": 0.20, "monotonic_quantile_transform": True},
                "direction_calibration": {
                    "direction_sign_multiplier": 1,
                    "weekly_sign_threshold": 0.0,
                    "weekly_magnitude_scale": 1.0,
                },
                "magnitude_calibration": {"scale": 1.0},
                "interval_calibration": {"weekly_interval_scale": 1.0},
                "hybrid_blend": {"enabled": False, "w_theta": 0.0},
                "foundation_prior": {
                    "enabled": True,
                    "provider": "chronos2",
                    "model_id": CHRONOS2_MODEL_ID,
                    "weight": 0.20,
                    "context_len": 252,
                },
                "ensemble_checkpoints": [ckpt_1.name, ckpt_2.name],
                "test_metrics": {"ensemble_size": 2},
            }
        )
    )
    (tmp_path / "conformal_calibration.json").write_text(
        json.dumps({"cqr_enabled": False, "global_adjustment": 0.0})
    )
    write_artifact_manifest(tmp_path)

    fixed_fnd_row = np.array([-0.03, -0.015, 0.0, 0.01, 0.02, 0.035, 0.05], dtype=np.float64)
    monkeypatch.setattr(
        "deep_learning.calibration.foundation_prior.predict_chronos2_weekly_log_quantiles",
        lambda _closes, pos, **_kw: np.tile(fixed_fnd_row, (len(pos), 1)),
    )
    monkeypatch.setattr(
        "deep_learning.data.feature_store.build_tft_dataframe",
        lambda _s, _cfg, **_kw: (df.copy(), ["HG=F_ret1", "HG=F_ret5"], ["day_of_week"], ["target"], float(closes[-1])),
    )
    monkeypatch.setattr(
        "deep_learning.data.future_frame.build_future_decoder_rows",
        lambda hist, pred_len, _cfg: build_future_decoder_rows(hist, pred_len, _cfg),
    )

    predictor = TFTPredictor(checkpoint_path=str(ckpt_1))
    monkeypatch.setattr(predictor, "_check_price_freshness", lambda _s, _sym: (0, False))
    monkeypatch.setattr(
        predictor,
        "_latest_price_baseline",
        lambda _s, _sym, fallback_price=None, fallback_date=None: (float(closes[-1]), dates[-1].date().isoformat()),
    )

    res = predictor.predict(session=None, symbol="HG=F", include_explain=False)
    assert "error" not in res, res
    assert np.isfinite(res["weekly_log_return"])
    assert np.isfinite(res["weekly_return"])
    assert res["model_info"]["ensemble_size"] == 2
    assert res["model_info"]["foundation_prior_applied"] is True
