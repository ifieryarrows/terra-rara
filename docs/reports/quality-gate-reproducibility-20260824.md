# TFT-ASRO Quality-Gate Reproducibility and OOS Recovery

Date: 2026-08-24
Scope: `9360011dde1177e1118ddfe6b8de4abfe94d8cf0` and the working-tree changes in this task

## Outcome

The failure chain was traced to three independent sources: an unbounded rolling
data cutoff, a weekly positive-rate loss that ignored its actual target, and
reproducibility gaps in worker allocation, Optuna sampling, and dependency
resolution. The working tree now contains fixes for each source without
changing quality-gate thresholds, using test labels for calibration, or
weakening leakage controls.

The modified code has not yet been executed by a new GitHub training run, so a
new remote quality-gate pass and an out-of-sample improvement are not claimed
by this report.

At inspection time, `HEAD` was `9360011` on `main`, there were no staged
files, and the only pre-existing unrelated working-tree item was the
untracked `frontend/rr_versions.json`, which remains untouched. The existing
`docs/implementation/implementation_plan.md` is an older April roadmap; no
new staged implementation-plan file was present.

## Remote evidence

| Run | Date / head | Result | Key evidence |
| --- | --- | --- | --- |
| [32037012405](https://github.com/ifieryarrows/terra-rara/actions/runs/32037012405) | Aug 17 / `af3e1a5` | Pass | Weekly DA 0.5574, MR 0.7591, PI80 0.7869, Sharpe 0.1488 |
| [32136653304](https://github.com/ifieryarrows/terra-rara/actions/runs/32136653304) | Aug 18 / `67f03f7` | Pass | Weekly DA 0.5738, MR 1.0815, PI80 0.7869, Sharpe 0.8730 |
| [32670262933](https://github.com/ifieryarrows/terra-rara/actions/runs/32670262933) | Aug 23 / `bdd5b9f` | Fail | MR 1.6317 and PI80 0.5161 failed the unchanged gate |
| [32759937911](https://github.com/ifieryarrows/terra-rara/actions/runs/32759937911) | Aug 24 / `9360011` | Fail | MR 1.8192, PI80 0.7097, and Sharpe -0.9852 failed the unchanged gate |

Aug 17 and Aug 18 used the same seed, old weekly loss configuration, and
reported 439 training samples / 61 weekly test samples, yet their test metrics
were materially different. That is direct evidence of nondeterminism beyond
the declared seed. Aug 23 and Aug 24 moved to 442 training samples / 62 weekly
test samples after the source grew from 661 to 666 price bars, so those runs
were not a like-for-like OOS comparison.

## Root causes

1. `feature_store.build_tft_dataframe()` used `datetime.now()` independently
   on every run. Historical price, sentiment, LME, and futures inputs therefore
   had no replayable cutoff.
2. `_weekly_positive_rate_loss()` discarded `actual_weekly` and penalized only
   a fixed `[0.20, 0.75]` predicted-positive band. The Aug 24 artifact still
   produced a 1.0000 weekly predicted-positive rate against 0.5484 actual,
   despite the increased fixed-band loss weight in the previous commit.
3. POSIX data loaders converted configured `num_workers=0` to `2`. Hyperopt
   also used the default Optuna sampler and a persistent fixed study name, so
   worker scheduling and prior study state could change the selected model.
4. The training workflows resolved broad `>=` requirements and moved from
   CPython 3.11.15 on Aug 18 to 3.11.16 later, with additional ML-adjacent
   package patch drift.

## Controlled loss experiment

Using a 62-sample synthetic batch with 34 positive and 28 negative weekly
targets (the Aug 23/24 test-window rate of 0.5484), the old fixed-band term
gave an all-positive forecast a loss of approximately 0.0536. The new
detached-sign BCE plus rate-band term gives the same all-positive forecast a
loss of 1.8792, while a correctly signed forecast gives 0.0181. The gradient
on the all-positive forecast is non-zero. This is a component-level control,
not a post-hoc metric adjustment or a test-set calibration.

## Changes made

- The weekly positive-rate loss now uses detached observed batch signs with a
  BCE term plus a tolerant ±0.20 rate band, bounded by the existing safety
  limits. It remains a training loss and does not read validation or test
  labels.
- POSIX data loaders now honor `num_workers=0`; hyperopt seeds Python/worker
  state, uses a seed-specific `TPESampler`, seeds each trial/fold, and enables
  Lightning deterministic mode.
- Hyperopt study names include the data snapshot, GitHub run ID, and therefore
  do not reuse stale trials from another dataset or prior workflow run.
- `TFT_DATA_AS_OF` is an optional ISO-8601 workflow input and is propagated to
  price, sentiment, LME, and futures queries. Training metadata now records a
  SHA-256 frame fingerprint, row/index bounds, cutoff, Python/platform, commit,
  and core package versions.
- `backend/constraints-tft.txt` pins the core stack to the Aug 18 successful
  run’s versions, and the training workflows pin Python 3.11.15, deterministic
  thread settings, and an uploaded `pip freeze` record.
- Weekly training logs now include bias, saturation, positive-rate, and
  interval component means so the next run can verify which loss terms are
  actually dominating instead of inferring it from final metrics.

The deterministic validation workflow requires `data_as_of`; the scheduled
training workflow may still use the latest available cutoff when no replay
input is supplied.

The quality-gate contract remains unchanged: weekly DA, magnitude ratio, tail
capture, PI80/PI96 coverage and width, crossing rate, Sharpe, and T+1 tail
thresholds are still enforced by the existing gate.

## Verification

Executed locally from `backend` with the available Python 3.12 environment:

- `py -3.12 -m compileall -q deep_learning tests/deep_learning`
- `py -3.12 -m pytest tests/ -q -m "not online"`
- Result: **464 passed, 14 skipped, 8 warnings**

The next required evidence is a clean deterministic workflow run with a fixed
`data_as_of` value, followed by the normal training workflow on the same
snapshot. Only those runs can establish repeatability and real untouched-test
performance for this change set.
