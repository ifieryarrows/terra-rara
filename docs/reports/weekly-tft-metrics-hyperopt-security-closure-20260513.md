# Weekly TFT Metrics, Hyperopt, and Security Closure - 2026-05-13

## Evidence Pass

Execution began with a deterministic full-document pass over `docs`.

- `docs/**/*.md`: 32 files, 536,094 extracted text characters.
- `docs/**/*.mmd`: 15 files, 276,686 extracted text characters.
- `docs/**/*.pdf`: 16 files, 305,966 extracted text characters via PyMuPDF.
- PDF extraction failures: 0.

The active project contract remains `weekly_log_v1`: daily log-return path targets, `max_prediction_length=5`, weekly `5D` primary objective, internal log-return evaluation, and public simple-return formatting only at inference/API boundaries.

## Failed Run Interpretation

The pasted run is a valid failed promotion. The workflow correctly retained the previous HF checkpoint.

Primary failure signals:

- Weekly magnitude exploded: `weekly_magnitude_ratio=5.1970`, with predicted absolute weekly move `0.1490` against actual `0.0280`.
- Weekly PI80 coverage collapsed: `weekly_pi80_coverage=0.4717`, far below the calibrated range `[0.74, 0.86]`.
- PI96 tails were unusably wide: `weekly_pi96_width_ratio=10.6438`.
- Weekly quantile incoherence was severe: `weekly_quantile_crossing_rate=0.2421`, `weekly_median_sort_gap_max=0.1532`.
- Small-sample weekly DA did not pass the relaxed gate: `weekly_directional_accuracy=0.5094 < 0.51` with `weekly_sample_count=53`.

Sharpe and Sortino were not treated as sufficient evidence because the current strategy metric is a sign-only synthetic return series without transaction costs, slippage, position sizing, or deployment execution constraints.

## Remediations Applied

Metrics and promotion:

- Added Wilson confidence intervals and sample counts for directional accuracy.
- Added naive-zero baseline-relative MAE/RMSE diagnostics.
- Added PI80 and PI96 interval scores.
- Preserved raw quantile crossing diagnostics and added sorted quantile diagnostics.
- Weekly promoted interval metrics now use monotonic sorted cumulative quantiles, while raw approximate cumulative quantile crossing remains auditable.
- Tightened the quality gate to require PI80 width ratio, PI96 coverage, PI96 width ratio, raw weekly crossing `<= 0.05`, and sorted weekly crossing `== 0.0`.
- Added explicit gate rejection for PI96 width explosion and overcovered-but-overwide PI80 intervals.

Hyperparameter and loss recovery:

- Raised weekly fallback/default loss controls to magnitude/width/tail-aware values:
  - `lambda_weekly_quantile=0.60`
  - `lambda_directional=0.10`
  - `lambda_magnitude=0.55`
  - `weekly_lambda_vol=0.35`
  - `lambda_width=0.50`
  - `lambda_tail_width=0.30`
  - `lambda_sanity=0.20`
  - `lambda_crossing=7.0`
- Moved Optuna to the approved 30-trial default and raised finite completed trial protection to 10.
- Changed the search space to the weekly-safe bounds from the remediation plan.
- Made `lambda_width`, `lambda_tail_width`, `lambda_sanity`, and `lambda_crossing` searched parameters.
- Added PI96 width, raw weekly crossing, sorted weekly crossing, PI96 interval score, and stronger magnitude penalties to the Optuna objective and pruning diagnostics.
- Clamped stale/unsafe Optuna overlays before final training, including encoder length below 40, learning rate above `6e-4`, weight decay above `5e-4`, weak magnitude/width/tail/sanity controls, and weak crossing controls.

Data integrity:

- Sentiment and event features now use the later of publication market date and availability/fetch market date.
- Backfilled articles no longer alter historical feature rows unless their availability time was valid for that market date.
- Embedding features now filter by availability/fetch cutoff and assign delayed embeddings to the later availability market date.

Security:

- Removed wildcard CORS defaults and added environment-driven allowed origins.
- Production startup now rejects `CORS_ALLOWED_ORIGINS=*`.
- Pipeline trigger auth keeps constant-time comparison, adds invalid-attempt throttling, and constrains `trigger_source` with length and pattern validation.
- HF sync workflow no longer prints token length.
- TFT artifacts now require `artifact_manifest.json` with SHA256 hashes before upload, download validation, or checkpoint load.
- Docker base image is pinned by digest: `python:3.11-slim@sha256:9a7765b36773a37061455b332f18e265e7f58f6fea9c419a550d2a8b0e9db834`.
- CI now runs backend `pip-audit` and frontend `npm audit --audit-level=high`.
- Frontend dependency advisories were closed by updating Vite, React plugin, ESLint, and TypeScript ESLint packages; `npm audit` now reports zero vulnerabilities.

## Validation

Local validation completed:

- `py -m compileall backend/app backend/deep_learning backend/scripts` - passed.
- `py -m pytest backend/tests -q -m "not online"` - `413 passed, 8 skipped`.
- `npm.cmd run build` in `frontend` - passed with the existing large chunk warning.
- `py -m pip_audit -r requirements.txt` in `backend` - no known vulnerabilities found.
- `npm.cmd audit --audit-level=high` in `frontend` - found 0 vulnerabilities.

Focused regression validation completed before the full run:

- Quality gate tests: `13 passed`.
- Weekly metrics/loss/hyperopt/config/hub/sentiment/settings tests: `38 passed, 6 skipped`.

## Remaining Operational Step

No model checkpoint was trained or uploaded during this code change. The next promotion attempt must run the updated GitHub TFT workflow. HF upload remains gated and will only occur after the revised quality gate passes against the new training metadata and artifact manifest.
