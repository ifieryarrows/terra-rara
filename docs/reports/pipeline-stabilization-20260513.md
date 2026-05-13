# Pipeline Stabilization - 2026-05-13

## Summary

Implemented the 2026-05-13 daily pipeline stabilization patch. The work keeps the weekly TFT contract strict, prevents a lone checkpoint from being treated as a healthy artifact set, separates degraded TFT snapshots from healthy forecasts, and hardens sentiment aggregation against overflow from stale/backfilled article timestamps.

## Changes

- HF TFT artifacts now include `tft_metadata.json` and `conformal_calibration.json` alongside the checkpoint, PCA model, and Optuna results.
- TFT artifact validation now requires at least `best_tft_asro.ckpt` plus `tft_metadata.json`, and the metadata must match `weekly_log_v1`.
- Hub download now attempts missing companion files even when `best_tft_asro.ckpt` already exists locally.
- `retrain_required` TFT payloads now expose `is_forecast_healthy=false` and null primary forecast fields.
- Worker Stage 5.5 now persists degraded TFT payloads for visibility without marking `tft_snapshot_generated=True`.
- Commentary generation excludes degraded TFT inputs and can include an explicit model-status note.
- V2 sentiment aggregation now uses bounded negative age weighting instead of overflow-prone positive exponent weighting.
- Sentiment aggregation logs age and weight audit counters before writing `DailySentimentV2`.
- Sentiment QC now skips correlation with explicit warnings when inputs are non-finite or have insufficient variance.
- V2 LLM scoring now retries missing ids through the reliable model and reports final unresolved ids instead of inflating parse failures for recovered articles.
- Frontend TFT card now renders an explicit degraded forecast state instead of showing a generic untrained-model empty state.

## Validation

```text
$env:PYTHONPATH='backend'; py -m py_compile backend/app/ai_engine.py backend/app/commentary.py backend/deep_learning/data/feature_store.py backend/deep_learning/inference/predictor.py backend/deep_learning/models/hub.py backend/worker/tasks.py
PASS
```

```text
$env:PYTHONPATH='backend'; py -m pytest backend/tests/deep_learning/test_hub_artifacts.py backend/tests/deep_learning/test_sentiment_qc.py backend/tests/deep_learning/test_tft_predictor_baseline.py backend/tests/test_ai_engine.py::TestV2ScoringBundleShape backend/tests/test_ai_engine.py::TestSentimentV2Aggregation backend/tests/test_commentary.py -q
16 passed, 2 warnings
```

```text
$env:PYTHONPATH='backend'; py -m pytest backend/tests -q -m "not online"
406 passed, 8 skipped, 8 warnings
```

```text
cd frontend
npm.cmd run build
PASS
```

The frontend build still reports the existing Vite large chunk warning for `assets/index-*.js`; the build completed successfully.

## Operational Follow-up

The remote weekly `tft-training.yml` workflow was not triggered by this local patch. Production recovery still requires running the weekly TFT training workflow, passing the quality gate, verifying the HF repo contains `best_tft_asro.ckpt`, `tft_metadata.json`, `conformal_calibration.json`, `pca_finbert.joblib`, and `optuna_results.json`, then running the daily pipeline with `train_model=false` to produce a healthy Stage 5.5 snapshot.
