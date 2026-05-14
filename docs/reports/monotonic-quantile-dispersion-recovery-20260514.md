# Monotonic Quantile Dispersion Recovery - 2026-05-14

## Context

The failed weekly TFT-ASRO run showed a structural output problem, not a marginal threshold problem:

- All completed Optuna trials had high prediction dispersion, with the best observed `avg_variance_ratio` still above `4.71`.
- Raw quantile crossing stayed high across completed trials.
- Weekly magnitude ratios stayed far above useful scale.
- Prior soft crossing, width, tail-width, and sanity penalties did not prevent the failure mode.

This patch moves quantile ordering from soft loss pressure into a structural monotonic transform and changes weekly loss calibration to directly target the observed variance/magnitude failure.

## Changes Applied

1. Structural monotonic quantile transform

- Added `backend/deep_learning/models/monotonic_quantiles.py`.
- Added `enforce_monotonic_quantiles()` with `gap_scale=0.01` and `init_bias=-3.0` for log-return scale compatibility.
- Added `validate_monotonicity()` for debug/evaluation checks.
- Added unit coverage proving random `[64, 5, 7]` raw outputs become monotonic and preserve q50 exactly.

2. Weekly loss redesign

- Replaced weekly soft guard terms with four terms:
  - `lambda_weekly_quantile=0.55`
  - `lambda_t1_quantile=0.15`
  - `lambda_dispersion=0.20`
  - `lambda_directional=0.10`
- Removed weekly `lambda_crossing`, `lambda_width`, `lambda_tail_width`, `lambda_sanity`, `lambda_magnitude`, and `lambda_vol`.
- `WeeklyASROPFLoss` now applies the monotonic transform before all loss computation.
- Added batch-level dispersion/magnitude calibration and smooth tanh directional loss.
- Added loss component accumulators and validation-epoch component logging.

3. Evaluation and public-output contract

- Public/unqualified crossing metrics now mean ordered/public outputs.
- Raw crossing remains available only through explicit `raw_*` diagnostics.
- Trainer evaluation, hyperopt fold metrics, conformal calibration, and `format_prediction()` now use ordered quantiles for public metrics and outputs.
- Weekly metrics still aggregate weekly log return by summing the daily log-return path over the configured horizon.

4. Quality gate and artifact health

- Quality gate no longer hard-fails on raw crossing.
- Public daily/weekly crossing above `0.001` now raises an assertion because it indicates a transform bug.
- Negative PI80/PI96 widths fail as impossible structural states.
- Variance ratio and naive-zero baseline checks are warning-only during stabilization.
- `artifact_manifest.json` now includes `artifact_health`.
- HF upload refuses artifacts when `artifact_health.safe_to_upload_to_hub` is false.

5. Hyperopt diagnostics and deterministic mode

- Added `scripts/hyperopt_diagnostics.py`.
- `optuna_results.json` now includes:
  - `structural_invalidity_report`
  - `trial_distribution_summary`
  - `best_trial_preflight`
- Hyperopt raises after writing the artifact if the structural report is `STRUCTURAL_FAILURE`.
- Added trainer CLI flag `--deterministic-weekly-validation` to bypass Optuna overlays and apply the fixed monotonic weekly config.

## Validation Evidence

First hard gate:

```text
py -m pytest backend/tests/deep_learning/test_monotonic_quantiles.py -q
1 passed
```

Focused validation:

```text
py -m pytest backend/tests/deep_learning/test_monotonic_quantiles.py backend/tests/deep_learning/test_metrics.py backend/tests/deep_learning/test_weekly_metrics.py backend/tests/deep_learning/test_weekly_asro_loss.py backend/tests/deep_learning/test_trainer_weekly_evaluation.py backend/tests/deep_learning/test_hyperopt.py backend/tests/deep_learning/test_hyperopt_diagnostics.py backend/tests/deep_learning/test_hub_artifacts.py backend/tests/deep_learning/test_tft_format_prediction.py backend/tests/deep_learning/test_config.py backend/tests/deep_learning/test_forecast_contract_config.py backend/tests/test_quality_gate.py backend/tests/test_quality_gate_weekly.py backend/tests/test_tft_quality_gate_script.py -q
71 passed, 4 skipped
```

Compile validation:

```text
py -m compileall backend/app backend/deep_learning backend/scripts scripts
passed
```

Full backend validation:

```text
py -m pytest backend/tests -q -m "not online"
419 passed, 6 skipped
```

## Deterministic Training Status

The deterministic training run was not executed in this local environment because required training dependencies are missing:

```text
py -c "import pytorch_forecasting; print(pytorch_forecasting.__version__)"
ModuleNotFoundError: No module named 'pytorch_forecasting'

py -c "import lightning; print(lightning.__version__)"
ModuleNotFoundError: No module named 'lightning'

py -c "import torch; print(torch.__version__)"
2.7.1+cpu
```

The missing packages are declared in `backend/requirements.txt`:

```text
torch>=2.1.0
pytorch-forecasting>=1.0.0
lightning>=2.0.0
```

No held-out model metrics were produced locally. The following deterministic-run metrics remain pending until the training environment has `pytorch_forecasting` and `lightning` installed:

- `ordered_quantile_crossing_rate`
- `public_quantile_crossing_rate`
- `pi80_width`
- `pi96_width`
- `variance_ratio`
- `mae_vs_naive_zero`
- `weekly_mae_vs_naive_zero`
- `weekly_magnitude_ratio`
- `weekly_tail_capture_rate`
- `directional_accuracy`
- `quality_gate_passed`

## Next Required Action

Run deterministic weekly validation in the training environment:

```text
cd backend
py -m deep_learning.training.trainer --deterministic-weekly-validation
```

Only proceed to expanded hyperopt if public crossing is zero, interval widths are positive, variance ratio falls below the observed `4.71` floor, and naive-zero / weekly-tail metrics improve.
