# TFT-ASRO Quality Gate Recovery — 2026-08-17

## Scope and current architecture

Copper Mind is a copper-futures forecasting system. The online pipeline ingests price bars, news and physical-market inputs; derives technical, sentiment, embedding, LME, futures-curve, regime and calendar features; and serves XGBoost and TFT-ASRO forecasts through the FastAPI backend. The TFT path models a five-step daily log-return decoder, exposes a five-day cumulative forecast as the primary product contract, and persists the checkpoint plus metadata, interval-calibration and integrity artifacts to Hugging Face only after promotion succeeds. The React frontend consumes the backend forecast APIs; GitHub Actions runs offline tests, security audits, the daily pipeline and scheduled TFT training.

The TFT training protocol is chronological `train -> validation -> test`. The final test slice is held out from hyperparameter CV and calibration. Optuna uses purged expanding-window folds on the pre-test region; deterministic validation bypasses Optuna and uses seed `42`.

## Quality gate: source of truth

`backend/app/quality_gate.py` is shared by the API, artifact manifest and `backend/scripts/tft_quality_gate.py`. No threshold was relaxed in this recovery.

| Metric | Required condition |
| --- | --- |
| Weekly directional accuracy | `>= 0.51` when weekly `n < 80`; otherwise `>= 0.53` |
| Weekly magnitude ratio | `[0.65, 1.35]` |
| Weekly tail capture | `>= 0.45` |
| Weekly PI80 coverage | `[0.74, 0.86]` |
| Weekly PI96 width ratio | `<= 3.0` |
| Public daily and weekly quantile crossing | `<= 0.001` (an assertion failure) |
| Sorted weekly crossing / median sort gap | `<= 0.001` (an assertion failure) |
| T+1 strategy Sharpe | `>= -0.30` |
| T+1 tail capture | `>= 0.35` |

Weekly PI80/PI96 widths must be non-negative and all required weekly metrics must be present. T+1 directional accuracy and variance ratio remain diagnostics; the latter also produces warnings outside its stabilization band. MAE relative to zero is warning-only, not a promotion condition.

## Baseline supplied for this run

The attached `determistic_metrics.md` is an observed baseline, not an implementation instruction. Its test output used 61 weekly samples and failed the gate on three conditions:

| Gate metric | Baseline | Result |
| --- | ---: | --- |
| Weekly directional accuracy | 0.3934 | fail (`< 0.51`) |
| T+1 Sharpe | -3.8766 | fail (`< -0.30`) |
| T+1 tail capture | 0.3158 | fail (`< 0.35`) |
| Weekly magnitude ratio | 0.9946 | pass |
| Weekly tail capture | 0.4667 | pass |
| Weekly PI80 coverage | 0.7869 | pass |
| Weekly PI96 width ratio | 0.9717 | pass |
| Public daily / weekly crossings | 0.0000 / 0.0000 | pass |

The same artifact reports `weekly_directional_accuracy_flipped=0.6066`, `weekly_sharpe_ratio_flipped=3.1148`, `weekly_sign_correlation=-0.4356`, and a weekly predicted-positive rate of `0.0328` versus actual `0.5738`. This is evidence of a sign-orientation failure, not a magnitude, interval-width or quantile-ordering failure.

## Implemented recovery

1. A magnitude-aware T+1 directional term is now part of `WeeklyASROPFLoss`. It directly optimizes the T+1 sign-sensitive Sharpe and tail metrics that the gate enforces, while retaining the existing weekly directional objective.
2. A validation-only global sign calibration accepts `-1` only when both T+1 and weekly validation paths are strongly inverse and the flipped validation path clears the relevant directional, Sharpe and tail safeguards. The selected multiplier is stored in metadata and applied identically to held-out evaluation, conformal interval fitting and live inference. Test labels are never used to fit it.
3. Regime volatility now derives from the observed close-to-close return rather than the forward target label. This removes a forward-label feature path.
4. MRMR feature selection is fit on the chronological training partition only; validation and test targets cannot influence the selected feature set.
5. Training records seed and split protocol in model metadata and requests deterministic execution. The deterministic workflow now runs the shared quality-gate CLI as a required CI step.

## Reproduction and validation

```powershell
cd backend
py -m pytest tests -q -m "not online"
py -m compileall app deep_learning scripts ..\scripts

# Requires the repository's GitHub Actions secrets and runs seed 42.
gh workflow run tft-deterministic-validation.yml --ref <branch>
gh run watch <run-id> --exit-status
```

The workflow uploads `/tmp/models/tft/` even when the gate fails, including `tft_metadata.json` and `artifact_manifest.json`, so the before/after comparison remains auditable. The final run ID and resulting gate metrics are appended below after CI completes.

## Results

Pending deterministic CI execution.
