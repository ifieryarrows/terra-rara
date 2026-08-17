# TFT-ASRO Quality Gate Recovery Roadmap — 2026-08-17

## Purpose

This document records how the TFT-ASRO quality-gate failure was diagnosed, which changes resolved each blocking condition, and how the recovery is maintained without weakening the promotion contract. The source of truth for threshold definitions remains `backend/app/quality_gate.py`; no quality-gate limit was relaxed.

The detailed baseline, implementation and final metrics are retained in `quality-gate-recovery-20260817.md`. This companion report is the concise change-to-outcome roadmap for future maintenance work.

## Failure summary

The supplied baseline used 61 weekly test samples and failed three promotion conditions:

| Blocking metric | Baseline | Required condition |
| --- | ---: | --- |
| Weekly directional accuracy | `0.3934` | `>= 0.51` for weekly `n < 80` |
| T+1 strategy Sharpe | `-3.8766` | `>= -0.30` |
| T+1 tail capture | `0.3158` | `>= 0.35` |

The same artifact provided a clear diagnostic: flipping the forecast sign produced weekly directional accuracy `0.6066` and Sharpe `3.1148`, while weekly sign correlation was `-0.4356`. The primary failure was therefore signal orientation, rather than weekly magnitude, quantile ordering or interval coverage.

## Changes that resolved the failure

| Workstream | Change | Why it was needed | Guardrail |
| --- | --- | --- | --- |
| T+1 directional learning | Added a magnitude-aware T+1 directional loss term to `WeeklyASROPFLoss` | The gate evaluates T+1 Sharpe and tail capture, so the loss needed a direct T+1 sign-sensitive objective | Production configuration passes the explicit configured weight; standalone loss defaults remain neutral for isolated component tests |
| Sign orientation | Added validation-only global sign calibration | A stable inverse validation signal can be corrected without inspecting test labels | The multiplier may be `-1` only when daily and weekly validation evidence, plus Sharpe and tail safeguards, agree; it is persisted and reused by evaluation and inference |
| Weekly intervals | Added deterministic validation-only PI80 interval scaling around the unchanged median | The first recovery run passed directional metrics but produced over-wide weekly intervals | Calibration targets PI80 `0.80`, changes only quantile spreads, and is stored in metadata and `conformal_calibration.json` |
| Leakage prevention | Replaced forward target-derived regime volatility with observed close-to-close returns | Forward labels must not become model features | Regime features are causal at the prediction timestamp |
| Feature selection | Fit MRMR on the chronological training partition only | Full-frame feature selection can leak validation/test target information | Validation and final-test targets cannot influence selected features |
| Reproducibility and promotion | Fixed seed `42`, deterministic trainer execution, split metadata and required gate invocation in deterministic CI | Training output must be comparable and unsafe artifacts must not be promoted | Chronological `train -> validation -> test`; validation-only calibration; shared gate CLI is required |
| CI dependency closure | Upgraded React Router to `7.18.2` and `nanoid` to `3.3.18`; added a narrow Lightning advisory exception with no usable fixed release | Tests workflow must remain green without concealing fixable frontend advisories | Backend and frontend audits run in the Tests workflow |

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

The validation-only weekly interval scale was `0.7363874346`, with validation PI80 coverage `0.7978723404`. Variance ratio `0.2398` remains a non-blocking warning and is explicitly treated as a monitoring signal, not a reason to weaken the gate.

The companion Tests workflow `32036742890` passed offline tests and both dependency audits.

## Roadmap

| Phase | Status | Exit criterion |
| --- | --- | --- |
| 1. Establish baseline and gate contract | Complete | Baseline artifact, source thresholds and failure reasons are recorded |
| 2. Harden data and validation boundaries | Complete | Causal regime features, train-only MRMR and an untouched final-test split |
| 3. Recover blocking metrics | Complete | Directional, Sharpe, tail, interval and quantile-coherence conditions pass together |
| 4. Enforce promotion in CI | Complete | Deterministic workflow invokes the shared gate and produces auditable artifacts |
| 5. Monitor drift | Next | Record weekly DA, magnitude, tail, PI80/PI96, crossings and variance-ratio warning on every scheduled run |
| 6. Controlled retuning | Next | Use only purged validation folds and fixed-seed comparison; never fit on final-test labels or relax promotion thresholds |

## Operating procedure

1. Run the deterministic workflow after model, feature, dependency or data-contract changes.
2. Inspect `tft_metadata.json`, `conformal_calibration.json` and `artifact_manifest.json` as one artifact set.
3. Compare all gate metrics with the table above using the same chronological split protocol.
4. Treat a variance-ratio warning as an investigation trigger; do not change thresholds to suppress it.
5. Append a before/after result row to the recovery report family before promoting a changed checkpoint.

## Change index

- `5409683` — validation-only weekly interval-width calibration
- `d388ceb` — neutral standalone loss default while retaining explicit production loss weight
- `2c994ed` — narrowly scoped Lightning audit exception
- `af3e1a5` — React Router and `nanoid` security updates
- `7e79032` — recovery traceability and roadmap added to the detailed report

