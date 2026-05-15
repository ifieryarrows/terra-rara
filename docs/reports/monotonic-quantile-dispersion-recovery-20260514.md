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
- Added `enforce_monotonic_quantiles()` with `gap_scale=0.02` and `init_bias=-3.0` for log-return scale compatibility after deterministic validation showed positive but too-narrow public intervals.
- Added `validate_monotonicity()` for debug/evaluation checks.
- Added unit coverage proving random `[64, 5, 7]` raw outputs become monotonic and preserve q50 exactly.

2. Weekly loss redesign

- Replaced weekly soft guard terms with six recovery terms:
  - `lambda_weekly_quantile=0.70`
  - `lambda_t1_quantile=0.20`
  - `lambda_dispersion=0.35`
  - `lambda_magnitude=0.50`
  - `lambda_naive=0.35`
  - `lambda_directional=0.05`
- Removed weekly `lambda_crossing`, `lambda_width`, `lambda_tail_width`, `lambda_sanity`, and `lambda_vol`.
- `WeeklyASROPFLoss` now applies the monotonic transform before all loss computation.
- Added batch-level dispersion calibration, median magnitude calibration, naive-zero relative loss, and smooth tanh directional loss logging.
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

Second deterministic recovery patch validation:

```text
py -m pytest backend/tests/deep_learning/test_weekly_asro_loss.py backend/tests/deep_learning/test_config.py backend/tests/deep_learning/test_metrics.py backend/tests/deep_learning/test_tft_format_prediction.py -q
26 passed, 5 skipped

py -m pytest backend/tests/deep_learning/test_hyperopt.py backend/tests/deep_learning/test_forecast_contract_config.py backend/tests/deep_learning/test_hub_artifacts.py backend/tests/deep_learning/test_trainer_weekly_evaluation.py -q
20 passed

py -m pytest backend/tests -q -m "not online"
419 passed, 7 skipped

py -m compileall backend/app backend/deep_learning backend/scripts scripts
passed
```

## Deterministic Training Status

The first deterministic training run in the training environment proved the structural monotonic fix works, but did not produce a promotable model:

```text
ordered_quantile_crossing_rate: 0.0
public_quantile_crossing_rate: 0.0
weekly_ordered_quantile_crossing_rate: 0.0
weekly_public_quantile_crossing_rate: 0.0
pi80_width: 0.0030409098069930068
pi96_width: 0.004415266469298083
weekly_pi80_width: 0.015095555772430545
weekly_pi96_width: 0.021998214256075826
variance_ratio: 3.4537061248499867
mae_vs_naive_zero: 3.367233609415377
weekly_mae_vs_naive_zero: 4.065073084058982
weekly_magnitude_ratio: 4.16903637631012
weekly_pi80_coverage: 0.07407407407407407
weekly_tail_capture_rate: 0.5
quality_gate_passed: false
```

Decision: do not proceed to expanded hyperopt. The second deterministic patch kept monotonic ordering structural, raised public gap scale consistently to `0.02`, initially disabled directional pressure, and added explicit median-scale plus naive-zero relative losses.

The second deterministic training run showed the median-scale and naive-zero recovery terms are working, while directional and tail quality still need controlled recovery:

```text
ordered_quantile_crossing_rate: 0.0
public_quantile_crossing_rate: 0.0
variance_ratio: 1.1288
weekly_variance_ratio: 0.9118
weekly_magnitude_ratio: 1.6282
weekly_pi80_width_ratio: 0.7517
weekly_pi80_coverage: 0.3889
weekly_directional_accuracy: 0.4259
weekly_tail_capture_rate: 0.2857
weekly_mae_vs_naive_zero: 1.8857
quality_gate_passed: false
```

Decision: do not proceed to expanded hyperopt. The directional reintroduction patch keeps magnitude and dispersion controls intact, reduces `lambda_naive` from `0.50` to `0.35`, and re-enables `lambda_directional` at `0.05` to test whether weekly direction and tail capture recover without reintroducing magnitude explosion.

The directional reintroduction run kept structural calibration usable but direction and tail metrics degraded, which suggests a sign-alignment or objective-alignment problem rather than a crossing or interval architecture problem:

```text
ordered_quantile_crossing_rate: 0.0
public_quantile_crossing_rate: 0.0
variance_ratio: 1.7660
weekly_variance_ratio: 1.6156
weekly_magnitude_ratio: 1.5268
weekly_pi80_coverage: 0.4444
weekly_directional_accuracy: 0.3519
weekly_tail_capture_rate: 0.2143
weekly_sharpe_ratio: -5.2951
weekly_mae_vs_naive_zero: 2.1431
quality_gate_passed: false
```

Decision: do not proceed to expanded hyperopt. Add flipped-direction diagnostics to weekly metrics and align `WeeklyASROPFLoss` directional pressure with the weekly cumulative target instead of per-day signs. Keep the current weights unchanged while testing whether weekly direction recovers.

The sign-alignment diagnostic run confirmed the weekly prediction is systematically inverted relative to the weekly actual target:

```text
ordered_quantile_crossing_rate: 0.0
public_quantile_crossing_rate: 0.0
variance_ratio: 1.6087
weekly_variance_ratio: 1.2461
weekly_magnitude_ratio: 1.4417
weekly_pi80_coverage: 0.3148
weekly_directional_accuracy: 0.3148
weekly_directional_accuracy_flipped: 0.6852
weekly_sharpe_ratio: -8.1774
weekly_sharpe_ratio_flipped: 8.1774
weekly_tail_capture_rate: 0.0714
weekly_tail_capture_rate_flipped: 0.9286
weekly_sign_correlation: -0.5616
weekly_mae_vs_naive_zero: 2.0840
quality_gate_passed: false
```

Decision: do not proceed to expanded hyperopt. Add weekly sign-bias diagnostics, an explicit weekly directional loss sign unit test, and a trainer alignment sample log so the next deterministic run can distinguish systematic sign inversion from target/prediction alignment drift.

The deterministic training run was not executed in this local development environment because required training dependencies are missing:

```text
py -c "import pytorch_forecasting; print(pytorch_forecasting.__version__)"
ModuleNotFoundError: No module named 'pytorch_forecasting'

py -c "import lightning; print(lightning.__version__)"
ModuleNotFoundError: No module named 'lightning'

py -m deep_learning.training.trainer --deterministic-weekly-validation
ModuleNotFoundError: No module named 'lightning'
ModuleNotFoundError: No module named 'pytorch_lightning'

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

For the directional reintroduction run, the specific improvement targets are:

```text
ordered/public crossing = 0.0
variance_ratio < 1.8
weekly_variance_ratio < 1.5
weekly_magnitude_ratio <= 1.5, or at least not worse than 1.6282
weekly_directional_accuracy > 0.50
weekly_tail_capture_rate > 0.40
weekly_mae_vs_naive_zero < 1.5, or clear improvement from 1.8857
weekly_pi80_coverage > 0.50
```

For the sign-alignment diagnostic run, inspect these additional metrics:

```text
weekly_directional_accuracy_flipped
weekly_sharpe_ratio_flipped
weekly_tail_capture_rate_flipped
weekly_sign_correlation
weekly_pred_positive_rate
weekly_actual_positive_rate
weekly_pred_mean
weekly_actual_mean
weekly_pred_median
weekly_actual_median
```

If flipped weekly metrics are materially better than the public weekly metrics, inspect target/prediction sign convention and weekly actual alignment before any hyperopt expansion.

The training log also emits `WEEKLY ALIGNMENT SAMPLE` rows for the first ten test samples:

```text
sample=<idx> actual_weekly=<sum_actual_path> pred_weekly=<sum_ordered_median_path> actual_sign=<sign> pred_sign=<sign>
```

## 2026-05-15 Direction Recovery Follow-Up

The latest deterministic training run showed that the weekly sign inversion is no longer the dominant failure mode:

```text
weekly_directional_accuracy: 0.5556
weekly_directional_accuracy_flipped: 0.4444
weekly_sharpe_ratio: 1.6267
weekly_sharpe_ratio_flipped: -1.6267
weekly_tail_capture_rate: 0.5000
weekly_sign_correlation: 0.0507
ordered_quantile_crossing_rate: 0.0
public_quantile_crossing_rate: 0.0
```

Decision: do not flip predictions and do not start expanded hyperopt yet. The remaining deterministic failure is now interval/magnitude calibration:

```text
weekly_magnitude_ratio: 1.3999
weekly_pi80_coverage: 0.2963
weekly_pi80_width_ratio: 0.4756
weekly_mae_vs_naive_zero: 1.4626
quality_gate_passed: false
next_required_action: WeeklyMagnitudeRatio=1.3999 outside [0.65, 1.35]; WeeklyPI80=0.2963 outside [0.74, 0.86]
```

The follow-up deterministic patch keeps weekly direction pressure unchanged while tightening magnitude/naive pressure and widening the structural quantile gaps:

```text
lambda_weekly_quantile = 0.70
lambda_t1_quantile = 0.20
lambda_dispersion = 0.35
lambda_magnitude = 0.55
lambda_naive = 0.40
lambda_directional = 0.05
DEFAULT_MONOTONIC_GAP_SCALE = 0.03
```

Expected next deterministic run targets:

```text
weekly_directional_accuracy >= 0.53
weekly_tail_capture_rate >= 0.45
weekly_magnitude_ratio <= 1.35
weekly_pi80_coverage improves toward 0.45-0.55
weekly_mae_vs_naive_zero improves toward 1.25-1.35
ordered/public crossing = 0.0
```

Local validation for the deterministic patch:

```text
py -m pytest backend/tests/deep_learning/test_config.py backend/tests/deep_learning/test_forecast_contract_config.py backend/tests/deep_learning/test_hyperopt.py backend/tests/deep_learning/test_monotonic_quantiles.py -q
23 passed

py -m pytest backend/tests/deep_learning/test_monotonic_quantiles.py backend/tests/deep_learning/test_metrics.py backend/tests/deep_learning/test_weekly_metrics.py backend/tests/deep_learning/test_weekly_asro_loss.py backend/tests/deep_learning/test_weekly_direction_alignment.py backend/tests/deep_learning/test_trainer_weekly_evaluation.py backend/tests/deep_learning/test_hyperopt.py backend/tests/deep_learning/test_hub_artifacts.py backend/tests/deep_learning/test_tft_format_prediction.py backend/tests/deep_learning/test_config.py backend/tests/deep_learning/test_forecast_contract_config.py backend/tests/test_quality_gate.py backend/tests/test_quality_gate_weekly.py backend/tests/test_tft_quality_gate_script.py -q
72 passed, 6 skipped

py -m pytest backend/tests -q -m "not online"
423 passed, 8 skipped

py -m compileall backend/app backend/deep_learning backend/scripts scripts
passed
```

The deterministic training command still cannot run in this local development environment because trainer runtime dependencies are missing:

```text
cd backend
py -m deep_learning.training.trainer --deterministic-weekly-validation
ModuleNotFoundError: No module named 'lightning'
ModuleNotFoundError: No module named 'pytorch_lightning'

py -c "import pytorch_forecasting; print(pytorch_forecasting.__version__)"
ModuleNotFoundError: No module named 'pytorch_forecasting'
```
