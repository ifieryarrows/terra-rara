# TFT-ASRO Quality-Gate Reproducibility and OOS Recovery

Date: 2026-08-26
Scope: `codex/tft-quality-gate-recovery-20260824`, through `c93fc45`, plus the
post-acceptance raw-scale review performed on 2026-08-28

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

Remote run
[33117940637](https://github.com/ifieryarrows/terra-rara/actions/runs/33117940637)
then completed successfully on the final promotion-isolation commit `c93fc45`.
The shared gate passed before Hub upload, the Hub/DB promotion step succeeded,
and the manifest reports both `safe_to_upload_to_hub=true` and
`safe_for_inference=true`. WeeklyDA improved to `0.6774`, weekly Sharpe to
`2.3982`, tail capture to `0.6875`, and PI80 remained `0.8387`.

The same run exposed the remaining reproducibility margin: raw weekly magnitude
rose from `1.8433` to `2.9573` while train/validation target scale changed by
only roughly 2-3%. Cap application rose from `0.6290` to `0.7097`; bounded
magnitude still passed at `1.3367`, only `0.0133` below its upper gate. Loss
logs and an autograd replay identified a structural gradient break: after the
hard weekly median cap, the summed q50 is exactly `+/-cap`, so weekly direction
and positive-rate objectives have zero raw-q50 gradient on clipped samples.
Run
[33120071397](https://github.com/ifieryarrows/terra-rara/actions/runs/33120071397)
tested a training-only smooth `cap * tanh(raw / cap)` sign surrogate on the
same `200ce6d4…` snapshot. Raw magnitude improved from `2.9573` to `2.1567`,
weekly Sharpe remained `2.4120`, WeeklyDA remained strong at `0.6613`, and tail
capture stayed `0.6875`. The hypothesis was nevertheless rejected: daily
Sharpe fell to `-1.0652`, variance ratio rose to `2.8432`, and PI80 overcovered
at `0.9032`. The gate correctly rejected the candidate before Hub/DB promotion,
so the smooth sign-surrogate implementation was removed while retaining its
diagnostic evidence. No threshold was changed.

## Validation-ranked checkpoint stabilization

The failed smooth-sign run retained three validation-ranked checkpoints, which
made it possible to test checkpoint instability without retraining. Its
selected minimum-loss checkpoint (`epoch=19`) failed the gate, while the second
ranked checkpoint (`epoch=13`) passed in retrospective replay. The validation
daily Sharpe ranked those two checkpoints in the opposite order, so adding a
new directional selection threshold would have overfit the wrong signal.

The adopted rule instead reduces single-epoch dependence: the two lowest
`val_weekly_loss` checkpoints are uniformly averaged in weight space and saved
as one normal Lightning checkpoint. Source ranking occurs before any test
prediction, the same averaged artifact is used for validation calibration,
untouched-test gate evaluation, conformal metadata, Hub upload, and live
inference, and metadata records the source checkpoint names and scores. No
ensemble-only serving path or gate change is introduced.

Pinned local replay on both downloaded `200ce6d4…` artefact sets produced the
following diagnostic comparison. Small absolute differences from Linux CI are
possible on Windows; the `c93fc45` single checkpoint crossed the raw-magnitude
`3.0` boundary locally (`3.1011` versus recorded CI `2.9573`), which itself
demonstrates why additional margin is required.

| Source | Candidate | Daily Sharpe | WeeklyDA | Weekly Sharpe | Raw weekly MR | Cap rate | PI80 | Current gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `33120071397` | selected single checkpoint | -1.0652 | 0.6613 | 2.4120 | 2.1762 | 0.7419 | 0.9032 | Fail |
| `33120071397` | validation top-2 weight soup | 1.0242 | 0.6613 | 2.4120 | 1.3151 | 0.4839 | 0.8387 | Pass |
| `33117940637` | selected single checkpoint | 1.5404 | 0.6935 | 2.5823 | 3.1011 | 0.7097 | 0.8387 | Fail locally on raw MR |
| `33117940637` | validation top-2 weight soup | 2.5848 | 0.6613 | 2.4068 | 2.5388 | 0.6935 | 0.8065 | Pass |

These old test windows were consulted to decide whether to adopt the fixed
top-2 rule, so they are retrospective evidence rather than a new unbiased
acceptance set. This led to the fresh-snapshot run described next.

## Fresh-snapshot cap-margin follow-up

Run
[33188247918](https://github.com/ifieryarrows/terra-rara/actions/runs/33188247918)
tested the frozen top-2 rule on commit `95882a3` and a fresh 665-row snapshot
(`3d302527…`). The intended artifact was produced from validation epochs 17 and
18 before test evaluation. Direction and interval quality were strong:
WeeklyDA `0.6774`, weekly Sharpe `2.4988`, tail `0.6875`, PI80 `0.8548`, and
daily Sharpe `2.5572`. The run missed only bounded weekly magnitude at `1.3866`
versus the unchanged `1.35` upper bound. Hub and DB promotion were skipped, so
the active passing `c93fc45` model remained unchanged.

Replaying all three saved checkpoints plus the top-2 and top-3 soups produced
the same bounded magnitude `1.3866`. The common value equals the cap-to-test
median ratio: the train-only `1.25x` cap was `0.02951`, slightly too high for
the fresh test regime regardless of checkpoint choice. The next single-variable
change therefore reduces only the train-derived absolute-median multiplier to
`1.20`; loss weights, checkpoint ranking, direction/interval calibration, and
gate thresholds remain unchanged.

The recorded top-2 artifacts were replayed with the cap multiplied by `0.96`
(`1.20 / 1.25`) before changing the production default:

| Source | WeeklyMR | Raw weekly MR | Cap rate | PI80 | Daily Sharpe | WeeklyDA | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `33120071397` | 1.2832 | 1.3151 | 0.5161 | 0.8387 | 1.0242 | 0.6613 | Pass |
| `33117940637` | 1.2832 | 2.5388 | 0.6935 | 0.7903 | 2.5848 | 0.6613 | Pass |
| `33188247918` | 1.3311 | 2.7163 | 0.7419 | 0.8548 | 2.0378 | 0.6774 | Pass |

These replayed test windows now inform the cap choice and are not an unbiased
final acceptance set. One new fresh immutable snapshot is still required after
the `1.20` default is frozen.

## Fresh-snapshot raw-scale follow-up

Run
[33190434601](https://github.com/ifieryarrows/terra-rara/actions/runs/33190434601)
tested the frozen `1.20` cap on commit `db24df0` and the 665-row `3d302527…`
snapshot. The bounded magnitude moved inside the unchanged gate as intended:
WeeklyMR fell from `1.3866` to `1.3311`. WeeklyDA remained `0.6774`, weekly
Sharpe `2.4988`, tail `0.6875`, PI80 `0.8226`, and daily Sharpe `1.4692`.
The only failing reason was `WeeklyRawMagnitudeExplosion=3.0397 > 3.0`, a
`1.32%` miss. Hub upload and DB promotion were skipped, so the active passing
`c93fc45` model remained unchanged.

The exact promoted checkpoint and pinned package versions replayed locally at
raw MR `2.8397` on Windows, while Linux CI recorded `3.0397`. The bounded gate
metrics remained close. This is not evidence that the CI result is wrong; it
shows that a candidate near `3.0` lacks enough cross-platform numerical margin.
The remote Linux gate remains authoritative.

All retained checkpoints and soup candidates were compared before changing the
loss. The best single checkpoint replayed at raw MR `2.9608`; the other two
single checkpoints were `8.8842` and `10.5684`, while top-3 soup was `3.1288`.
Changing the two-checkpoint weight was also rejected. A `0.40/0.60` blend was
the only tested fixed ratio that avoided regressions across the retained old
windows, but its current raw MR (`2.8721`) had less local margin than the
existing uniform soup. Ratios with more current-run margin caused old PI80 or
daily-Sharpe regressions. Production therefore keeps the validation-ranked
uniform top-2 rule.

The remaining issue is direct reliance on the train-derived q50 cap: `72.58%`
of test weekly medians were bounded. The existing logarithmic saturation tail
is intentionally sublinear because earlier quadratic/Huber tails made extreme
first-epoch predictions dominate the objective. The next change keeps that
tail and adds a bounded `tanh(excess)^2` barrier for ordinary `1-3x` cap
violations. Its value is at most one per sample, so it strengthens pressure in
the observed moderate-violation regime without recreating the unbounded early
training loss. Gate thresholds, cap resolution, soup ranking, direction
calibration, and interval calibration remain unchanged.

## Raw-scale margin pass and live promotion

Run
[33192762447](https://github.com/ifieryarrows/terra-rara/actions/runs/33192762447)
tested the bounded moderate-violation barrier on commit `e867857` and the same
665-row `3d302527…` snapshot. The unchanged quality gate passed. The top-2
validation soup used epochs 20 and 29 (`val_weekly_loss=2.3906` and `2.4340`)
without test-label selection.

| Metric | Previous run `33190434601` | Passing run `33192762447` | Gate status |
| --- | ---: | ---: | --- |
| WeeklyDA | 0.6774 | 0.6774 | Pass |
| Weekly Sharpe | 2.4988 | 2.4988 | Pass |
| Weekly tail capture | 0.6875 | 0.6875 | Pass |
| Weekly magnitude ratio | 1.3311 | 1.3311 | Pass |
| Weekly raw magnitude ratio | 3.0397 | 1.6506 | Pass with margin |
| Weekly cap-applied rate | 0.7258 | 0.6452 | Warning improved |
| Weekly PI80 | 0.8226 | 0.8548 | Pass |
| Daily DA | 0.5000 | 0.5484 | Pass |
| Daily Sharpe | 1.4692 | 3.2388 | Pass |
| Daily tail capture | 0.5500 | 0.6500 | Pass |

The artifact manifest marks the candidate `quality_gate_passed=true`,
`safe_for_inference=true`, and `safe_to_upload_to_hub=true`; the promoted
checkpoint SHA-256 is `7a34aa07…`. Hub upload, promoted DB metadata persistence,
and the chained backtest all completed successfully. A live API read after the
workflow returned `trained_at=2026-08-28T17:10:10Z`, `quality_gate.passed=true`,
and no rejection reasons, confirming that the production Models page is backed
by this passed model rather than a rejected candidate.

The chained daily walk-forward comparison is supportive on error but not a
replacement for the weekly gate. TFT MAE was `0.077388` versus XGBoost
`0.124224` (`37.7%` lower), while daily directional accuracy was `0.4933`
versus `0.4980` (`0.94%` relatively lower). The weekly promotion result remains
valid; the small daily-direction deficit is retained as a separate development
signal and must not be conflated with the 5-day WeeklyDA contract.

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
| [33117940637](https://github.com/ifieryarrows/terra-rara/actions/runs/33117940637) | `c93fc45`, snapshot `200ce6d4…` | Pass | Promotion order proven; WeeklyDA `0.6774`, MR `1.3367`, tail `0.6875`, PI80 `0.8387`, weekly Sharpe `2.3982`; raw MR `2.9573`, cap rate `0.7097` |
| [33120071397](https://github.com/ifieryarrows/terra-rara/actions/runs/33120071397) | `9e4006f`, snapshot `200ce6d4…` | Fail | Smooth sign surrogate lowered raw MR to `2.1567`, but PI80 rose to `0.9032` and daily Sharpe fell to `-1.0652`; Hub/DB promotion blocked |
| [33188247918](https://github.com/ifieryarrows/terra-rara/actions/runs/33188247918) | `95882a3`, snapshot `3d302527…` | Fail | Top-2 soup preserved WeeklyDA `0.6774`, weekly Sharpe `2.4988`, PI80 `0.8548`, and daily Sharpe `2.5572`; only bounded MR `1.3866` failed; promotion blocked |
| [33190434601](https://github.com/ifieryarrows/terra-rara/actions/runs/33190434601) | `db24df0`, snapshot `3d302527…` | Fail | Cap margin fixed bounded MR at `1.3311`; WeeklyDA `0.6774`, weekly Sharpe `2.4988`, tail `0.6875`, and PI80 `0.8226` held; only raw MR `3.0397` failed; promotion blocked |
| [33192762447](https://github.com/ifieryarrows/terra-rara/actions/runs/33192762447) | `e867857`, snapshot `3d302527…` | Pass | WeeklyDA `0.6774`, weekly Sharpe `2.4988`, tail `0.6875`, bounded MR `1.3311`, raw MR `1.6506`, PI80 `0.8548`, and daily Sharpe `3.2388`; Hub/DB promotion and backtest succeeded |

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

- Backend suite after adding the bounded moderate-violation barrier:
  **519 passed, 7 warnings**.
- Backend suite after adding validation-ranked top-2 checkpoint stabilization:
  **518 passed, 7 warnings**.
- Focused TFT calibration, hyperopt, direction, conformal, trainer, and
  predictor suite: **63 passed, 2 warnings**.
- Python compileall and `git diff --check` passed.
- Frontend TypeScript and Vite production build passed. Vite retained its
  existing large-main-chunk warning (`852.08 kB`, `253.53 kB` gzip).

## Remaining acceptance step

Runs `33120071397`, `33188247918`, and `33190434601` prove that rejected
candidates leave Hub and DB untouched. Run `33192762447` then passed and
replaced the active metadata only after Hub upload succeeded. The top-2 soup,
`1.20` train-derived cap, and quality thresholds remain fixed.

The model-quality hypothesis is now supported by a passing fresh-snapshot run
with substantial raw-scale margin. The remaining reproducibility acceptance
step is a deterministic validation followed by an immutable-snapshot replay at
the same commit and dependency contract. Those runs must preserve all required
weekly/daily metrics without further tuning between them. The separate daily
backtest direction gap remains a later roadmap item, not a reason to alter the
now-passing weekly gate.
