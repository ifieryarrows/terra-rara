# TFT-ASRO Quality Gate Recovery Change Summary — 2026-08-17

## Summary

The TFT-ASRO recovery work addressed a promotion failure caused primarily by forecast sign inversion, while also closing validation-boundary, interval-width, reproducibility and CI dependency gaps. The recovery series from `main` through `67f03f7` changed 24 files and preserved the existing quality-gate thresholds. The detailed evidence remains in `quality-gate-recovery-20260817.md`; the companion roadmap is in `quality-gate-recovery-roadmap-20260817.md`.

## Problem diagnosed

The supplied 61-sample weekly test baseline failed three blocking conditions:

| Gate metric | Baseline | Required condition |
| --- | ---: | --- |
| Weekly directional accuracy | `0.3934` | `>= 0.51` for weekly `n < 80` |
| T+1 strategy Sharpe | `-3.8766` | `>= -0.30` |
| T+1 tail capture | `0.3158` | `>= 0.35` |

The baseline also showed weekly sign correlation `-0.4356`. Reversing the forecast sign raised weekly directional accuracy to `0.6066` and Sharpe to `3.1148`. This identified signal orientation as the primary failure mode, rather than quantile ordering, interval coverage or forecast magnitude.

## Changes applied

| Area | Change | Outcome and guardrail |
| --- | --- | --- |
| T+1 objective | Added a magnitude-aware T+1 directional loss to `WeeklyASROPFLoss` and enabled it explicitly in production and hyperopt configuration with weight `0.20`. | The loss now trains the T+1 sign-sensitive metrics enforced by the gate. The standalone loss API keeps a neutral default for isolated component tests. |
| Direction calibration | Added validation-only global sign calibration and persisted the selected `-1`/`1` multiplier in model metadata. | A flip is accepted only when daily and weekly validation evidence plus Sharpe/tail safeguards agree. Test labels are not used. |
| Inference consistency | Applied the stored direction multiplier and weekly interval scale in `TFTPredictor`; invalid calibration metadata makes the checkpoint incompatible. | Evaluation, conformal fitting and live inference use the same calibrated artifact contract. |
| Weekly intervals | Added deterministic validation-only PI80 interval scaling around the unchanged median and persisted it in `tft_metadata.json` and `conformal_calibration.json`. | Quantile spreads can be narrowed or expanded without changing the median path or directional/magnitude metrics. |
| Leakage prevention | Replaced target-derived regime volatility with observed close-to-close returns. | Regime features are causal at the forecast origin; forward labels are not used as features. |
| Feature selection | Added a training-only `selection_df` path for MRMR and fit selection on the chronological training partition. | Validation and final-test target information cannot influence the selected feature set. |
| Reproducibility | Recorded seed/split metadata, enabled worker seeding and deterministic trainer execution. | The protocol is explicit: chronological `train -> validation -> test`, validation-only calibration and untouched final test. |
| Promotion | Added the shared `tft_quality_gate.py` invocation to the deterministic validation workflow. | A model cannot be promoted from that workflow without passing the same gate used by the application and artifact manifest. |
| CI dependency closure | Upgraded `react-router-dom` to `7.18.2` and `nanoid` to `3.3.18`; narrowly allowlisted Lightning advisory `PYSEC-2026-3624` because its reported fix is unusable for the required stack. | Fixable frontend advisories were removed while the backend audit remains strict for new findings. |

## Recovery roadmap status

| Phase | Status | Exit criterion |
| --- | --- | --- |
| 1. Establish the baseline and gate contract | Complete | Failure metrics, source thresholds and sign-inversion evidence recorded |
| 2. Harden data and validation boundaries | Complete | Causal regime inputs, train-only MRMR and untouched final-test split |
| 3. Recover the blocking metrics | Complete | Direction, Sharpe, tail, interval and quantile-coherence conditions pass together |
| 4. Enforce deterministic promotion | Complete | CI runs the shared gate and publishes auditable metadata and manifest artifacts |
| 5. Monitor drift | Next | Track weekly DA, magnitude, tail, PI80/PI96, crossings and variance-ratio warnings |
| 6. Controlled retuning | Next | Compare fixed-seed changes through purged validation folds without relaxing thresholds |

## Verified outcome

Final deterministic validation run `32037012405` passed the shared quality gate:

| Gate metric | Final CI result | Status |
| --- | ---: | --- |
| Weekly directional accuracy | `0.5574` | pass |
| Weekly magnitude ratio | `0.7591` | pass |
| Weekly tail capture | `0.5333` | pass |
| Weekly PI80 coverage | `0.7869` | pass |
| Weekly PI96 width ratio | `1.0147` | pass |
| T+1 strategy Sharpe | `0.1488` | pass |
| T+1 tail capture | `0.5263` | pass |
| Public and sorted weekly crossings | `0.0000` / `0.0000` | pass |

The validation-only weekly interval scale was `0.7363874346`, with validation PI80 coverage `0.7978723404`. Variance ratio `0.2398` remains a non-blocking warning and is monitored without changing the promotion contract. Tests workflow `32036742890` passed offline tests and both dependency audits.

## Operating follow-up

1. Run deterministic validation after model, feature, dependency or data-contract changes.
2. Inspect `tft_metadata.json`, `conformal_calibration.json` and `artifact_manifest.json` as one artifact set.
3. Compare the complete gate metric table using the same chronological split protocol.
4. Treat variance-ratio warnings as investigation triggers, not threshold-tuning requests.
5. Append a before/after result row to the recovery report family before promoting a changed checkpoint.

## Commit index

- `b5a30bc` — recover the deterministic quality gate and harden validation boundaries
- `5409683` — calibrate weekly interval width on validation
- `d388ceb` — preserve the neutral standalone loss default
- `2c994ed` — allowlist the Lightning advisory without a usable fix
- `af3e1a5` — upgrade audited frontend dependencies
- `4d535d3`, `7e79032`, `67f03f7` — record final validation, traceability and roadmap documentation
