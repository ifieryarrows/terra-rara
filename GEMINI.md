# CopperMind Modeling, Quality Gate & Database Invariants

These rules define the **low-freedom (hard) domain invariants** for CopperMind. Outside these boundaries, exercise full autonomy and flexibility in implementation design, refactoring, and testing.

## 1. Financial Metric Invariants & Lookahead Safeguards
- **Annualization Consistency:** Match the annualization factor in `sharpe_ratio` and `sortino_ratio` to the target horizon. Daily ($T+1$) returns use $\sqrt{252}$; 5-day / weekly cumulative returns use $\sqrt{52}$ ($\sqrt{252 / \text{horizon}}$). Never annualize weekly series with $\sqrt{252}$.
- **Overlapping Horizon Transparency:** Sliding-window cumulative returns ($R_{t \to t+h}$) exhibit $\text{MA}(h-1)$ autocorrelation. When reporting Sharpe or Sortino on overlapping test sets, note autocorrelation inflation or report non-overlapping / HAC (Newey-West) adjusted statistics.
- **Causal Feature Boundaries:** Never compute rolling statistics on forward-looking targets (e.g., `target_1d`). Use historical observed returns (`observed_return_1d`) for regime and volatility features. Fit MRMR feature selection and scalers strictly on the training partition.

## 2. Magnitude Transparency & Clipping Audits
- **Raw vs. Bounded Magnitude:** Inference bounding (`weekly_median_cap`) prevents extreme outliers from reaching snapshots but can mask network instability if unaudited.
- **Quality Gate Thresholds:** Audit `weekly_raw_magnitude_ratio` alongside `weekly_bounded_magnitude_ratio` in `backend/app/quality_gate.py`:
  - **Hard Block:** `weekly_raw_magnitude_ratio > 3.0` blocks promotion.
  - **Non-Blocking Stability Warning:** `weekly_raw_magnitude_ratio > 1.8` or `weekly_median_bound_applied_rate > 0.30`.

## 3. Artifact Discovery & Deployment Isolation
- **Strict Quality Gate Filtering:** Active model serving endpoints (`/api/models/tft/summary`, etc.) and snapshot invalidation logic must explicitly filter `quality_gate_passed.is_(True)`. Candidate training must never overwrite active serving rows; active metadata is committed only after both the shared quality gate and Hugging Face Hub upload succeed.
- **Dual Storage Verification (Release Recovery Only):** When explicitly asked to audit or recover historical model checkpoints/release candidates, inspect both local DB/disk paths and GitHub Actions run artifacts (`gh run download`). Skip remote CI artifact checks during routine coding tasks.

## 4. Horizon Consistency & Presentation Hierarchy
- **Primary vs. Diagnostic Separation:** Strictly separate primary 5-day cumulative weekly forecasts (TFT-ASRO) from single-step $T+1$ daily diagnostics (XGBoost) across backend schemas and UI views.
- **UI Horizon Alignment:** Never place daily risk-adjusted ratios (Daily Sharpe/Sortino) alongside weekly directional accuracy (Weekly DA) without explicit horizon labels and visual grouping.
- **Promotion Gate Horizon Matching:** Enforce risk-adjusted promotion criteria on the primary 5-day operational horizon (`weekly_sharpe_ratio`), retaining single-step metrics only as lower-bound sanity checks.
- **Explainability Framing & Fail-Open Contract:** TFT-ASRO explanations (`explain_json`, served by `/api/explain/{symbol}`) are optional, fail-open worker artifacts; their absence must never fail a pipeline run or invalidate a snapshot. Always frame them as raw, pre-calibration 5-day model focus (VSN weights + encoder-window attention), never as causal attribution.

## 5. Database & Connection Pooler Invariants
- **Dialect Explicit Binding & Fallback:** Dynamically normalize unspecified `postgresql://` URIs in `backend/app/db.py` to the installed driver (`postgresql+psycopg://` or `postgresql+psycopg2://`) to remain resilient across SQLAlchemy 2.1+ dialect shifts.
- **PgBouncer / Supabase Prepared Statement Safety:** When connecting to transaction-mode poolers (`:6543`, `pooler.supabase.com`, `supabase.co`, `pgbouncer`) with `psycopg` v3, disable prepared statements via `connect_args={"prepare_threshold": None}` to prevent `DuplicatePreparedStatement` / `InvalidSqlStatementName` errors.
