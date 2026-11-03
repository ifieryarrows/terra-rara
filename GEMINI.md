# CopperMind Modeling, Quality Gate & Database Contracts

These guidelines capture the core mathematical, operational, and database contracts for CopperMind while preserving full engineering autonomy over implementation and optimization.

## 1. Financial Metric Contracts & Lookahead Safeguards
- **Annualization & Horizon Matching:** Match annualization factors in `sharpe_ratio` and `sortino_ratio` to the target horizon:
  - **Daily ($T+1$) single-step returns:** Scale by $\sqrt{252}$.
  - **Weekly (5-day) cumulative returns:** Scale by the codebase convention $\sqrt{52}$ (approximating $\sqrt{252 / 5} \approx 7.10$ non-overlapping 5-day trading blocks per year). Never annualize 5-day cumulative returns with $\sqrt{252}$.
- **Overlapping Horizon Autocorrelation:** Daily-sampled 5-day cumulative returns ($R_{t \to t+5}$) overlap by 4 trading days and exhibit $\text{MA}(4)$ serial correlation. Unadjusted sample Sharpe/Sortino on daily sliding windows reflects overlapping observations; when evaluating statistical significance or reporting standalone risk-adjusted performance, account for overlap via non-overlapping subsampling or Newey-West HAC variance adjustment.
- **Causal Feature Boundaries:** Compute rolling regime and volatility features exclusively from historical observed returns (`observed_return_1d`), never forward-looking targets (`target_1d`). Fit MRMR feature selection and scalers strictly on the training partition.

## 2. Magnitude Transparency & Clipping Audits
- **Raw vs. Bounded Magnitude:** Inference bounding (`weekly_median_cap`) prevents extreme outliers from reaching snapshots.
- **Quality Gate Thresholds (`backend/app/quality_gate.py`):**
  - **Hard Block:** `weekly_raw_magnitude_ratio > 3.0` blocks promotion.
  - **Non-Blocking Stability Warning:** `weekly_raw_magnitude_ratio > 1.8` or `weekly_median_bound_applied_rate > 0.30`.

## 3. Artifact Discovery & Promotion Isolation
- **Strict Quality Gate Filtering:** Active model serving endpoints (`/api/models/tft/summary`, etc.) and snapshot invalidation logic must filter `quality_gate_passed.is_(True)`. Candidate training must never overwrite active serving rows; active metadata is committed only after both the quality gate and Hugging Face Hub upload succeed.

## 4. Horizon Consistency & Explainability
- **Primary vs. Diagnostic Separation:** Keep primary 5-day cumulative weekly forecasts (TFT-ASRO) distinct from single-step $T+1$ daily diagnostics (XGBoost) in API schemas and UI views.
- **Explainability Contract:** TFT-ASRO explanations (`explain_json`, served by `/api/explain/{symbol}`) are optional, fail-open worker artifacts; their absence must never fail a pipeline run or invalidate a snapshot. Frame them as raw, pre-calibration 5-day model focus (VSN weights + encoder-window attention), never as causal attribution.

## 5. Database & Connection Pooler Contracts
- **Dialect Explicit Binding & Fallback:** Dynamically normalize unspecified `postgresql://` URIs in `backend/app/db.py` to the installed driver (`postgresql+psycopg://` or `postgresql+psycopg2://`).
- **Transaction Pooler vs. Direct Connection Prepared Statements:**
  - When connecting through **transaction-mode connection poolers** (`:6543`, `pooler.supabase.com`, `pgbouncer`) with `psycopg` v3, disable prepared statements via `connect_args={"prepare_threshold": None}` to prevent `DuplicatePreparedStatement` / `InvalidSqlStatementName` errors.
  - On **direct PostgreSQL / session-mode connections** (e.g., port `:5432` on `db.<project>.supabase.co` without pooler host/port markers), leave `psycopg` prepared statements enabled so repeated queries benefit from server-side statement caching.
