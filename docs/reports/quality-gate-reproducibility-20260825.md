# TFT-ASRO Quality-Gate Reproducibility and OOS Recovery

Date: 2026-08-25  
Scope: `codex/tft-quality-gate-recovery-20260824`, current head `fb6b96f`

## Current outcome

The reproducibility failure is resolved at the data and artifact level, but a
repeatable quality-gate pass has not yet been established on the current
immutable OOS frame. The current fixed configuration is deterministic and
passes every mandatory metric except Weekly PI80 coverage; the unchanged gate
rejects it at `0.8871` because the allowed interval is `[0.74, 0.86]`.

No quality-gate threshold was changed, no test label was used for calibration,
and no metric was rewritten. A 30-trial normal-workflow Optuna search was
started on the same immutable frame as run `32861632419`; its final result was
pending when this report was written.

## Evidence chain

| Run | Commit / frame | Result | Evidence |
| --- | --- | --- | --- |
| [32849638659](https://github.com/ifieryarrows/terra-rara/actions/runs/32849638659) | `b9ef925`, earlier live frame | Pass | WeeklyDA `0.5806`, MR `1.0850`, PI80 `0.8065`; not repeatable after the DB frame changed |
| [32852940617](https://github.com/ifieryarrows/terra-rara/actions/runs/32852940617) | `b9ef925`, snapshot `2c2bfb15…` | Fail | WeeklyDA `0.5645`, MR `1.2858`, Tail `0.5625`, PI80 `0.8871`; only PI80 failed |
| [32853910347](https://github.com/ifieryarrows/terra-rara/actions/runs/32853910347) | `efe3bdd`, changed live frame | Fail | WeeklyDA `0.5161`, PI80 `0.3226`, Sharpe `-1.2132`; chronological `shuffle=False` also degraded OOS direction |
| [32855673086](https://github.com/ifieryarrows/terra-rara/actions/runs/32855673086) | `578b2e9`, captured snapshot `2c2bfb15…` | Fail / capture | 662 rows, last index `2026-08-17 04:00:00`; WeeklyDA `0.5161`, Tail `0.4375`, PI80 `0.3226`, Sharpe `-1.2132` |
| [32856572025](https://github.com/ifieryarrows/terra-rara/actions/runs/32856572025) | `578b2e9`, same snapshot | Diagnostic | Snapshot SHA matched, but prior checkpoints downloaded into the active directory and the training result diverged |
| [32857587031](https://github.com/ifieryarrows/terra-rara/actions/runs/32857587031) | `462a348`, same snapshot, isolated artifact | Fail / deterministic replay | Exact match to `32855673086`: same promoted checkpoint `epoch=29`, same gate metrics |
| [32858780393](https://github.com/ifieryarrows/terra-rara/actions/runs/32858780393) | `d449b03`, seeded shuffle restored | Fail | WeeklyDA `0.5645`, MR `1.2858`, Tail `0.5625`, PI80 `0.8871`; only PI80 failed |
| [32860786879](https://github.com/ifieryarrows/terra-rara/actions/runs/32860786879) | `34da1df`, symmetric interval-width loss | Fail | Same directional/magnitude metrics; PI96 width improved `1.1166 → 1.0148`, but PI80 remained `0.8871` |
| [32862903964](https://github.com/ifieryarrows/terra-rara/actions/runs/32862903964) | `fb6b96f`, current snapshot, 30-trial normal workflow | Pending | Replay-aware workflow skipped live embedding backfill and entered Optuna; the earlier `32861632419` attempt was cancelled while backfill remained active |

The historical baseline remains relevant: [32037012405](https://github.com/ifieryarrows/terra-rara/actions/runs/32037012405)
passed on Aug 17, while [32670262933](https://github.com/ifieryarrows/terra-rara/actions/runs/32670262933)
and [32759937911](https://github.com/ifieryarrows/terra-rara/actions/runs/32759937911)
failed on later data. The earlier [32136653304](https://github.com/ifieryarrows/terra-rara/actions/runs/32136653304)
pass was majority-sign dominated (`predicted-positive-rate=1.0`) and is not
accepted as a robust OOS result.

## Root causes

1. `TFT_DATA_AS_OF` was only a time cutoff. The underlying database and
   external feature sources were mutable, so the same cutoff produced changing
   rows, last-index bounds, target scales, and frame hashes. A cutoff is not an
   immutable data version.
2. The replay workflow downloaded the entire prior validation artifact into
   the active model directory. Prior checkpoints were therefore present in the
   new `ModelCheckpoint` directory and contaminated the replay protocol.
3. `shuffle=False` removed a useful stochastic training protocol. With the
   explicit seeded generator removed from the global RNG path, the correct
   repeatable protocol is seeded shuffling, not chronological batch order.
4. The weekly interval objective penalized under-width but not over-width. The
   symmetric width regularizer in `34da1df` improves PI96 width, but the
   current OOS frame still overcovers PI80 after validation calibration.

## Implemented fixes

- `578b2e9` adds an immutable post-selection feature-frame snapshot with
  metadata, SHA-256 integrity verification, cutoff verification, and an
  optional expected SHA guard. The trainer and feature store share the same
  digest implementation.
- `462a348` makes both TFT workflows download only the two snapshot files into
  a separate temporary directory, preventing prior checkpoints and metadata
  from entering a replay run.
- `d449b03` restores seeded shuffled training windows while retaining the
  explicit CPU generator seed and single-thread deterministic settings.
- `34da1df` changes the weekly interval-width regularizer from one-sided
  under-width pressure to symmetric deviation from the train-objective target.
- `493e176` validates an expected frame SHA before persisting a live-built
  snapshot.
- `fb6b96f` skips the live FinBERT backfill when a supplied immutable snapshot
  is being replayed, preventing an unnecessary DB mutation/blocking step.

The quality gate contract in `backend/app/quality_gate.py` is unchanged. The
weekly DA, magnitude, tail, PI80/PI96, crossing, and daily risk checks remain
mandatory; the existing sign-collapse guard remains active.

## Verification

Local verification completed before the pending normal workflow:

- Full backend suite on the final code: `471 passed, 14 skipped, 8 warnings`.
- Focused interval/model tests: `27 passed, 11 skipped`.
- Feature-store replay tests: `10 passed`.
- Compileall and workflow YAML parsing passed; `git diff --check` passed.
- Unrelated `frontend/rr_versions.json` remains untracked and untouched.

The remaining acceptance evidence is a passing, repeatable deterministic run on
the immutable snapshot, followed by a successful normal training workflow
using that same snapshot. Until both exist, the current result must be
reported as reproducible failure with one remaining PI80 defect, not as a
completed quality-gate recovery.
