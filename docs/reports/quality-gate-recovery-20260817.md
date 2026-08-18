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
6. A validation-only weekly interval scale is fit against the central PI80 target (0.80) and applied around the unchanged median. The scale is persisted in metadata and `conformal_calibration.json`, reused by the test evaluator and live predictor, and never fit on test labels.

## Reproduction and validation

```powershell
cd backend
py -m pytest tests -q -m "not online"
py -m compileall app deep_learning scripts ..\scripts

# Requires the repository's GitHub Actions secrets and runs seed 42.
gh workflow run tft-deterministic-validation.yml --ref <branch>
gh run watch <run-id> --exit-status
```

The workflow uploads `/tmp/models/tft/` even when the gate fails, including `tft_metadata.json` and `artifact_manifest.json`, so the before/after comparison remains auditable.

## Results

Run `32037012405` (final branch commit `af3e1a5`) completed successfully. The model used seed `42`, chronological train/validation/test splits, validation-only direction and interval calibration, and the untouched 61-row test slice. The earlier recovery run `32034661946` also passed; both runs are retained in GitHub Actions for auditability.

| Gate metric | Baseline | Final CI | Result |
| --- | ---: | ---: | --- |
| Weekly directional accuracy | 0.3934 | 0.5574 | pass |
| T+1 Sharpe | -3.8766 | 0.1488 | pass |
| T+1 tail capture | 0.3158 | 0.5263 | pass |
| Weekly magnitude ratio | 0.9946 | 0.7591 | pass |
| Weekly tail capture | 0.4667 | 0.5333 | pass |
| Weekly PI80 coverage | 0.7869 | 0.7869 | pass |
| Weekly PI96 width ratio | 0.9717 | 1.0147 | pass |
| Public / sorted weekly crossing | 0.0000 / 0.0000 | 0.0000 / 0.0000 | pass |

The final validation-only weekly interval scale was `0.7363874346`, yielding validation PI80 coverage `0.7978723404`. The quality-gate CLI reported `QUALITY GATE: PASSED`; variance ratio `0.2398` remains a non-blocking warning under the existing policy. The artifact manifest therefore marks the checkpoint safe for inference and upload. Tests workflow `32036742890` also passed offline tests plus backend/frontend dependency audits.

## Cause-to-fix traceability

| Observed failure | Root cause / evidence | Change that addressed it | Verification |
| --- | --- | --- | --- |
| Weekly DA `0.3934`, T+1 Sharpe `-3.8766`, T+1 tail `0.3158` | The supplied artifact showed a stable sign inversion: flipped weekly DA `0.6066`, flipped Sharpe `3.1148`, sign correlation `-0.4356` | Added a magnitude-aware T+1 sign term to `WeeklyASROPFLoss` and a validation-only sign multiplier, reused for test evaluation, conformal fitting and live inference | Final weekly DA `0.5574`, T+1 Sharpe `0.1488`, T+1 tail `0.5263` |
| Initial recovery run PI80 coverage `0.9344`, width ratio `2.2067` | The model median was usable, but central quantile spreads were over-wide; conformal calibration correctly refused to widen an already over-covered validation interval | Fit a deterministic validation-only interval scale around the unchanged median and persist it in metadata and `conformal_calibration.json` | Final PI80 coverage `0.7869`; validation coverage `0.7979`; scale `0.7364` |
| Risk of optimistic validation metrics | Regime volatility read a forward target, and MRMR selection used the full frame | Regime volatility now uses observed close-to-close returns; MRMR is fit on the chronological training partition only | Leakage-focused tests pass; test labels remain untouched by calibration and feature selection |
| Deterministic workflow did not enforce promotion | The workflow trained and uploaded artifacts without invoking the shared gate | Added the shared `tft_quality_gate.py` step, deterministic seed/split metadata and `Trainer(deterministic=True)` | Runs `32034661946` and `32037012405` both report `QUALITY GATE: PASSED` |
| CI dependency checks blocked closure | Lightning advisory had no usable fixed release; React Router 6 and `nanoid` advisories had available fixes | Added a narrowly scoped Lightning advisory allowlist; upgraded React Router to `7.18.2` and `nanoid` to `3.3.18` | Tests workflow `32036742890` passed tests and both dependency audits |

## Recovery roadmap

| Phase | Status | Deliverable / exit criterion |
| --- | --- | --- |
| 1. Baseline and gate contract | Complete | Source thresholds, baseline artifact and failure modes recorded without changing gate limits |
| 2. Leakage and validation hardening | Complete | Causal regime features, train-only MRMR, chronological train/validation/test protocol and validation-only calibration |
| 3. Metric recovery | Complete | Directional loss/sign calibration and PI80 interval scaling bring every blocking metric into the defined band |
| 4. CI and artifact promotion | Complete | Deterministic workflow runs the gate; only a passing manifest is eligible for promotion; tests/audits are green |
| 5. Drift monitoring | Next | Track weekly DA, magnitude, tail, PI80/PI96, crossings and variance-ratio warning on every scheduled run; alert before a promotion regression |
| 6. Controlled maintenance | Next | Re-run fixed-seed validation after dependency/data changes; tune only through purged validation folds; never use final-test labels or relax thresholds |

The operational follow-up is intentionally monitoring-oriented: the current non-blocking variance-ratio warning (`0.2398`) should be tracked, while the promotion contract remains unchanged. Any future calibration or hyperparameter change must add a before/after row to this report family and retain the uploaded metadata and manifest for audit.

## Change and reproduction index

- Recovery implementation: `5409683` (`fix(tft): calibrate weekly interval width on validation`)
- Standalone loss compatibility: `d388ceb` (`fix(tft): preserve neutral standalone loss default`)
- CI dependency closure: `2c994ed` and `af3e1a5`
- Final report closure: `4d535d3`
- Reproduce locally with the commands in this report's **Reproduction and validation** section; reproduce the promoted metrics with `tft-deterministic-validation.yml` and inspect `tft_metadata.json`, `conformal_calibration.json` and `artifact_manifest.json` together.
