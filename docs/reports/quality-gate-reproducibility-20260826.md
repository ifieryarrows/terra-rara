# TFT-ASRO Quality-Gate Reproducibility and OOS Recovery

Date: 2026-08-26
Scope: `codex/tft-quality-gate-recovery-20260824`, `1f2a9df`, plus the
working-tree calibration changes

## Current outcome

The remaining OOS failure was narrowed to two separate post-training effects:
the promoted checkpoint can differ between runs even with the same immutable
snapshot and pinned runtime, and one scalar validation interval scale does not
transfer between the validation and test volatility regimes. The quality-gate
thresholds remain unchanged.

The working tree now contains a validation-only weekly direction calibrator
using a fixed causal regime/news feature family and a validation-referenced
`realized_vol_20d` interval-width calibrator. Both are applied identically in
gate evaluation, conformal-artifact generation, and live inference. No test
labels are used by either calibrator.

A new remote pipeline run has intentionally not been started after this change:
the user paused the four-to-five-hour training job. The controlled artifact
replays below are therefore the current evidence, not a completed remote-run
acceptance.

## Remote evidence

| Run | Commit / frame | Result | Key evidence |
| --- | --- | --- | --- |
| [32885956455](https://github.com/ifieryarrows/terra-rara/actions/runs/32885956455) | `eb19436`, snapshot `2c2bfb15…` | Pass | Prior `.15` interval objective; WeeklyDA `0.5323`, MR `1.2858`, PI80 `0.8548` |
| [32906424221](https://github.com/ifieryarrows/terra-rara/actions/runs/32906424221) | `1f2a9df`, snapshot `2c2bfb15…` | Fail | Current `.40` objective; WeeklyDA `0.4516`, Tail `0.4375`, PI80 `0.9516` |
| [32999841194](https://github.com/ifieryarrows/terra-rara/actions/runs/32999841194) | `1f2a9df`, snapshot `2c2bfb15…` | Fail | Same inputs/runtime family; WeeklyDA `0.6452`, PI80 `0.9516`; only PI80 failed |

The two failing `.40` runs used the same snapshot SHA, cutoff, Optuna
artifact, and pinned package family, but selected different best checkpoints
(`epoch=36` versus `epoch=20`). This confirms that the residual variation is in
the training trajectory/checkpoint outcome, not a mutable data frame. The
direction model and interval calibration operate after that checkpoint is
selected and do not alter the underlying model weights.

## Controlled OOS replay

The downloaded validation/test prediction arrays from both failing runs were
replayed locally with the new production calibration path. The direction
model was fitted on chronological training origins only. The interval base
scale and volatility reference were fitted on validation only; test labels were
used only for this read-only report of the resulting OOS metrics.

| Source artifact | WeeklyDA | Weekly MR | Weekly tail | PI80 | PI80 width ratio | PI96 | PI96 width ratio | Daily Sharpe | Daily tail | Sign collapse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `32999841194` | 0.5968 | 1.2858 | 0.6875 | 0.8548 | 1.4724 | 0.9194 | 1.0470 | 2.3575 | 0.7000 | none |
| `32906424221` | 0.5968 | 1.2858 | 0.6875 | 0.8548 | 1.3798 | 0.9194 | 1.0071 | 0.0078 | 0.5500 | none |

Both rows satisfy the unchanged gate contract: WeeklyDA exceeds `0.51`, MR is
inside `[0.65, 1.35]`, tail capture exceeds `0.45`, PI80 is inside
`[0.74, 0.86]`, PI96 and width ratios are valid, daily Sharpe is above `-0.30`,
daily tail is above `0.35`, and public crossing rates are zero. The weekly
predicted-positive rate is `0.5000` versus actual `0.5484`, so the structural
sign-collapse guard is not triggered.

## Changes made

- The auxiliary weekly direction model now selects a fixed 18-feature causal
  regime/news family when those features are present, uses `C=0.03`, and still
  enables only after the existing validation improvement and balanced-rate
  checks. Custom feature lists remain supported for unit tests.
- Weekly interval calibration can now fit a deterministic per-origin width
  multiplier from `realized_vol_20d`. The reference is the positive validation
  median; the exponent is selected only by validation proper score from the
  fixed grid `{0.25, 0.35, 0.50, 0.75, 1.00}`. Both replay artefacts select
  `0.25`; the relative factor is bounded to `[0.5, 2.0]` and the final scale is
  bounded to `[0.20, 2.50]`.
- The exact forecast-origin rows are used for validation and test (`decoder
  time_idx - 1`), eliminating the previous one-row mismatch in the conformal
  artifact path.
- Final training and the separate Optuna job now share the same CPU/thread and
  deterministic-algorithm contract before Lightning is imported, removing a
  reproducibility gap between hyperopt and final training.
- The promoted metadata now records the process controls and effective Torch
  thread/deterministic-algorithm state needed to compare a future remote run.
- The same persisted conditioning metadata is consumed by test evaluation,
  conformal calibration, and live `TFTPredictor` inference. The median path is
  unchanged; only interval spread changes.
- Added regression coverage for conditioned interval replay, conformal-origin
  reuse, and median preservation. `backend/app/quality_gate.py` was not
  changed.

## Verification

- Backend suite: **502 passed, 8 warnings**.
- Focused TFT calibration, hyperopt, direction, conformal, trainer, and
  predictor suite: **63 passed, 2 warnings**.
- Python compileall and `git diff --check` passed.
- The unrelated untracked `frontend/rr_versions.json` remains untouched.

## Remaining acceptance step

The next run must execute the TFT-ASRO pipeline on the intended snapshot and
produce a fresh artifact. When that long pipeline is started, monitoring will
stop immediately after dispatch; the user will be asked to report when it has
finished. Only then will the new manifest, logs, quality-gate result, and
promotion safety be inspected and the goal closed.
