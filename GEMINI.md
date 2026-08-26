# CopperMind Modeling & Quality Gate Invariants

## 1. Financial Metric Invariants & Lookahead/Autocorrelation Safeguards
- **Annualization Consistency:** Always match the annualization factor in `sharpe_ratio` and `sortino_ratio` to the target horizon. Daily returns must use $\sqrt{252}$, while 5-day / weekly cumulative returns must use $\sqrt{52}$ (or $\sqrt{252 / \text{horizon}}$). Never annualize weekly series with $\sqrt{252}$.
- **Overlapping Horizon Transparency:** Sliding-window cumulative returns ($R_{t \to t+h}$) exhibit MA($h-1$) autocorrelation. When reporting Sharpe or Sortino on overlapping test sets, note the autocorrelation inflation or report non-overlapping / HAC (Newey-West) adjusted statistics.
- **Causal Feature Boundaries:** Never compute rolling statistics on forward-looking target variables (e.g. `target_1d`). Use historical observed returns (`observed_return_1d`) for regime and volatility features. All feature selection (MRMR) and scaler fittings must occur strictly on the training partition.

## 2. Magnitude Transparency & Clipping Audits
- **Raw vs. Bounded Magnitude:** Bounding mechanisms (`weekly_median_cap`) prevent inference explosion but can mask an unstable neural network.
- **Quality Gate Invariant:** Quality gate evaluation must audit `weekly_raw_magnitude_ratio` alongside `weekly_bounded_magnitude_ratio`. If `weekly_median_bound_applied_rate > 0.30` (more than 30% of predictions clipped by the cap), flag the model for magnitude instability regardless of bounded performance.

## 3. Artifact Discovery & Deployment Isolation
- **Dual Storage Verification:** When searching for previous models or release candidates, inspect both local database/disk paths AND remote CI/CD storage (GitHub Actions run artifacts via `gh run download`).
- **Strict Quality Gate Filtering:** Endpoints serving the latest model (`/api/models/tft/summary`, etc.) and snapshot invalidation logic must explicitly filter for `quality_gate_passed.is_(True)`. Rejected checkpoints must never serve predictions or invalidate valid snapshots.

## 4. Horizon Consistency & Presentation Hierarchy
- **Primary vs. Diagnostic Separation:** Multi-horizon forecasting models must strictly separate primary horizon metrics (e.g. 5D cumulative weekly returns) from single-step diagnostics (e.g. T+1 daily path).
- **UI Horizon Alignment:** Dashboards and summary views must never place daily risk-adjusted ratios (Daily Sharpe/Sortino) directly alongside weekly accuracy (Weekly DA) without explicit horizon labels and grouping.
- **Promotion Gate Horizon Matching:** A model whose primary operational deliverable is a multi-day forecast must enforce risk-adjusted performance on that same primary horizon (weekly_sharpe_ratio), retaining single-step metrics purely as lower-bound sanity checks.
