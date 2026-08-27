# TFT-ASRO Quality-Gate Reproducibility and OOS Recovery

Date: 2026-08-26
Scope: `codex/tft-quality-gate-recovery-20260824`, through `fa3e7c4`, plus the
promotion-isolation review performed on 2026-08-27

## Current outcome

Remote run
[33020232075](https://github.com/ifieryarrows/terra-rara/actions/runs/33020232075)
completed successfully on commit `20d144c`. It used the known-good fallback
configuration and a fresh 663-row snapshot
(`4e2e5caee7f084ce1e79e56ee198bfce29ce4993aa9168faa30b55320cbf0700`).
The model passed the shared gate with WeeklyDA `0.6452`, weekly magnitude ratio
`1.3252`, weekly tail capture `0.6250`, PI80 `0.8387`, zero public crossing,
daily Sharpe `5.7683`, and weekly Sharpe `1.9603` after the corrected `sqrt(52)`
annualisation. The latter also satisfies the `weekly_sharpe_ratio >= -0.20`
guard added in the following `fa3e7c4` commit.

The pass has two explicit non-blocking stability warnings: raw weekly
magnitude ratio `1.8433` and median-cap application rate `0.6290`. They do not
invalidate the bounded OOS metrics, but they identify raw-scale stabilization
as the next model-quality target.

The 2026-08-27 review found that candidate training persisted DB metadata
before CI gate and HF Hub promotion. Because `tft_model_metadata.symbol` is
unique, a rejected candidate could overwrite the only active row; querying for
an older passed row therefore could not recover it. Migration 004 also existed
only as a SQL file and was not connected to the application migration runner.
The reviewed working tree now keeps candidate metrics in CI artifacts, writes
the active DB row only after both the shared gate and Hub upload succeed,
strictly serves passed metadata, and applies the promotion-column migration
idempotently at application startup and CI promotion time.

## Earlier diagnosis

The remaining OOS failure was narrowed to two separate post-training effects:
the promoted checkpoint can differ between runs even with the same immutable
snapshot and pinned runtime, and one scalar validation interval scale does not
transfer between the validation and test volatility regimes. The existing
thresholds were not weakened; later commits added raw-magnitude audit fields
and a primary-horizon weekly-Sharpe lower bound.

The working tree now contains a validation-only weekly direction calibrator
using a fixed causal regime/news feature family and a validation-referenced
`realized_vol_20d` interval-width calibrator. Both are applied identically in
gate evaluation, conformal-artifact generation, and live inference. No test
labels are used by either calibrator.

At the time these calibration changes were prepared, no new remote run had
been started because the user paused the four-to-five-hour training job. The
controlled artifact replays below were the acceptance evidence at that stage;
the later successful remote run is recorded above.

## Remote evidence

| Run | Commit / frame | Result | Key evidence |
| --- | --- | --- | --- |
| [32885956455](https://github.com/ifieryarrows/terra-rara/actions/runs/32885956455) | `eb19436`, snapshot `2c2bfb15…` | Pass | Prior `.15` interval objective; WeeklyDA `0.5323`, MR `1.2858`, PI80 `0.8548` |
| [32906424221](https://github.com/ifieryarrows/terra-rara/actions/runs/32906424221) | `1f2a9df`, snapshot `2c2bfb15…` | Fail | Current `.40` objective; WeeklyDA `0.4516`, Tail `0.4375`, PI80 `0.9516` |
| [32999841194](https://github.com/ifieryarrows/terra-rara/actions/runs/32999841194) | `1f2a9df`, snapshot `2c2bfb15…` | Fail | Same inputs/runtime family; WeeklyDA `0.6452`, PI80 `0.9516`; only PI80 failed |
| [33020232075](https://github.com/ifieryarrows/terra-rara/actions/runs/33020232075) | `20d144c`, snapshot `4e2e5cae…` | Pass | WeeklyDA `0.6452`, MR `1.3252`, tail `0.6250`, PI80 `0.8387`, weekly Sharpe `1.9603`; cap-rate warning `0.6290` |

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
  reuse, and median preservation. The calibration patch itself did not alter
  `backend/app/quality_gate.py`; later commits added the magnitude and weekly
  Sharpe audits described above.

## Verification

- Backend suite after the promotion-isolation review: **515 passed, 8 warnings**.
- Focused TFT calibration, hyperopt, direction, conformal, trainer, and
  predictor suite: **63 passed, 2 warnings**.
- Python compileall and `git diff --check` passed.
- Frontend TypeScript and Vite production build passed. Vite retained its
  existing large-main-chunk warning (`852.08 kB`, `253.53 kB` gzip).

## Remaining acceptance step

Run `33020232075` proves the model/calibration path on `20d144c` and also passes
the subsequently-added weekly-Sharpe rule when evaluated retrospectively. A
fresh run from the final promotion-isolation commit is still required to prove
the workflow ordering end to end: failed candidates must leave the active DB
row untouched, while a passing candidate must upload to Hub before updating
the DB. When that long pipeline is started, monitoring will stop immediately
after dispatch and the user will be asked to report when it finishes.
