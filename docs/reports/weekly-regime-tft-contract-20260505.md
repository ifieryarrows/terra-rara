# Weekly Regime/Event-Conditioned TFT Contract - 2026-05-05

## Summary

Implemented the weekly-first TFT contract for Terra Rara/CopperMind. The TFT pipeline now treats the model output as a five-step daily log-return path, optimizes weekly cumulative log return through a dedicated weekly ASRO loss, and exposes public forecast fields as simple returns.

T+1 remains available as an auxiliary impulse signal, but the backend API and frontend headline now use the 5D weekly forecast as the primary contract.

## Return-Space Contract

| Layer | Return Space |
| --- | --- |
| Training target | daily log return |
| Weekly loss | cumulative 5D log return |
| Metrics internal | log return |
| Conformal calibration | weekly log return |
| API public fields | simple return |
| Frontend display | percentage simple return |
| Price conversion | `price * exp(cumulative_log_return)` |

## Backend Changes

- Added `deep_learning.contract` as the single source for `weekly_log_v1`, public/internal return-space constants, and `log_to_simple_return`.
- Added weekly forecast and weekly loss configs while preserving `max_prediction_length=5`.
- Rebuilt feature-store target construction:
  - `target` is now next-day daily log return.
  - Added `target_1d_log_return`, `target_5d_log_return`, `realized_vol_20d`, and `material_move_5d`.
  - Target/helper columns are hard-forbidden as TFT covariates.
  - MRMR relevance now targets `target_5d_log_return`.
  - Core sentiment/regime features are forced through feature selection.
- Added train/inference-aware weekly target validation.
- Added identity/no-op target normalization for TFT datasets.
- Added leakage-safe market-date sentiment/event aggregation, market-date utilities, `Date` ORM columns, and SQL migrations.
- Added a dry-run capable market-date backfill script with count and sample mapping audit output.
- Added regime/event conditioning features and no-lookahead future decoder rows.
- Added `WeeklyASROPFLoss` and skipped the old ASRO curriculum mutation for weekly loss.
- Rewrote `format_prediction` so log-return outputs convert to public simple returns and weekly prices through `exp(sum(log_returns))`.
- Added checkpoint contract metadata and predictor compatibility guard. Old/missing metadata now returns a retrain-required degraded payload instead of silently interpreting simple returns as log returns.
- Added weekly metrics, weekly-first hyperopt objective wiring, validation-only conformal calibration artifact writing, and weekly-first quality gate thresholds.
- Updated TFT snapshot persistence to preserve `primary_*`, `t1_*`, and `return_space` fields.

## Frontend Changes

- Added weekly forecast response types.
- Updated the Overview page headline card to `Deep Learning Weekly Forecast`.
- The headline now reads `primary_forecast_return`, `primary_forecast_q10`, and `primary_forecast_q90`.
- T+1 is displayed only as a secondary impulse.
- The forecast chart now displays the 5-step path instead of only the first day.

## Validation

Backend:

```bash
cd backend
py -m compileall backend/deep_learning backend/app backend/pipelines backend/scripts backend/worker
py -m pytest -q
```

Result:

```text
378 passed, 5 skipped
```

Frontend:

```bash
cd frontend
npm.cmd run build
```

Result:

```text
tsc && vite build passed
```

Blocked validation:

```text
npm.cmd run typecheck
```

The frontend package does not define a `typecheck` script. TypeScript validation was covered through `npm.cmd run build`, which runs `tsc`.

```text
npm.cmd run lint
```

The lint script exists, but ESLint cannot run because the frontend has no ESLint configuration file in the project tree.

## Operational Notes

- Historical training should not be trusted until `scripts/backfill_market_dates.py` has been run against the production database.
- Existing TFT checkpoints are intentionally incompatible with `weekly_log_v1`; retraining is required before live weekly forecasts can be served as healthy.
- Conformal adjustment is fit from validation/calibration data only. Final test data remains reserved for reporting and quality-gate promotion.
