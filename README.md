<div align="center">

<a href="https://terra-rara.vercel.app">
  <img src="frontend/public/brand/terra-rara-social.png" alt="Terra Rara — CopperMind COMEX Copper Forecasting Platform" width="100%" />
</a>

<h1>Terra Rara · CopperMind</h1>

<p>
  <strong>Multi-horizon COMEX copper futures (<code>HG=F</code>) forecasting &amp; market intelligence platform</strong><br />
  Modern 3-Seed TFT-ASRO + Fail-Open Chronos-2 Foundation Prior · Split CQR · XGBoost · Structural Macro · LLM Sentiment
</p>

<p>
  <a href="https://terra-rara.vercel.app"><img src="https://img.shields.io/badge/Live_Demo-terra--rara.vercel.app-0969da?style=for-the-badge&logo=vercel&logoColor=white" alt="Live Demo" /></a>
  <a href="https://ifieryarrows-copper-mind.hf.space/api/docs"><img src="https://img.shields.io/badge/API_Docs-FastAPI_Swagger-10b981?style=for-the-badge&logo=fastapi&logoColor=white" alt="API Docs" /></a>
  <a href="https://github.com/ifieryarrows/terra-rara/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/ifieryarrows/terra-rara/tests.yml?branch=main&style=for-the-badge&label=CI_Tests" alt="CI Tests" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-e29a5f?style=for-the-badge" alt="License: MIT" /></a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/PyTorch_Lightning-TFT--ASRO-792EE5?style=flat-square&logo=lightning&logoColor=white" alt="PyTorch Lightning" />
  <img src="https://img.shields.io/badge/Amazon-Chronos--2-FF9900?style=flat-square" alt="Chronos-2" />
  <img src="https://img.shields.io/badge/XGBoost-T%2B1_Diagnostic-189FDD?style=flat-square" alt="XGBoost" />
  <img src="https://img.shields.io/badge/React_18-Vite_7_+_Tailwind_v4-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React 18" />
  <img src="https://img.shields.io/badge/PostgreSQL-Supabase-3ECF8E?style=flat-square&logo=supabase&logoColor=white" alt="Supabase" />
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#architecture-evolution--benchmark-rationale">29-Model Benchmark</a> ·
  <a href="#cuprum-companion">Cuprum Companion</a> ·
  <a href="#getting-started">Quickstart</a> ·
  <a href="#api-reference">API Reference</a>
</p>

</div>

---

> [!IMPORTANT]
> **Informational Research Only:** Terra Rara does not provide trading signals or financial advice. The HTTP API is strictly **snapshot-only** (`backend/app/main.py`): all data ingestion, LLM scoring, feature generation, inference, and training execute asynchronously in background workers (`backend/worker/runner.py`) or GitHub Actions CI pipelines.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Cuprum Companion](#cuprum-companion)
- [Architecture](#architecture)
- [Symbol Sets](#symbol-sets)
- [Model Details](#model-details)
  - [XGBoost](#xgboost-parameters)
  - [TFT-ASRO Deep Learning](#tft-asro-deep-learning)
  - [Architecture Evolution & Benchmark Rationale](#architecture-evolution--benchmark-rationale)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Security](#security)

## Overview

Terra Rara predicts COMEX copper futures (`HG=F`) **returns** (never raw price levels) across three strictly separated horizons, combining causal cross-asset signals, deep probabilistic time-series modeling, foundation priors, and institutional LLM market reasoning:

| Horizon Layer | Engine & Target | Annualization & Calibration | Production Role |
| --- | --- | --- | --- |
| **Primary · 5-Day Cumulative** | **65% `Chronos-2` Foundation Prior + 35% 3-Seed Modern TFT-ASRO** (`target_5d_log_return`) | `sqrt(52)` Newey-West HAC · RSS quantiles · **HAR-GJR** Stress + Split CQR (`q02..q98`) | Headline weekly probabilistic trajectory, market stress regime & risk bands (`/api/analysis/tft/HG=F`) |
| **Structural · 20D / 60D Outlook** | **Causal Weekly Macro Scorecard** (`long_term/`) | FRED 10Y real yields, breakevens, USD, & miner cross-asset ratios | Slow fundamental macro regime layer (`/models` & `/dashboard`) |
| **Legacy Read-Only ($T+1$)** | **XGBoost Regressor** *(retired from daily cron & Overview UI)* | `sqrt(252)` · Archived `AnalysisSnapshot` read-only compatibility | Preserved read-only at `/api/analysis`; `screener/` & `backtest/` remain intact |

In parallel, daily scheduled workers ingest commodity news via Google News RSS, score article sentiment using resilient multi-tier LLM routing (`IFM K2-Horizon-375B` / NVIDIA Nemotron 3.5 Super VL / Liquid LFM 2.5 / Gemma 4 with local FinBERT fallback), and drive institutional commentary and interactive **Cuprum** companion dialogue — while intentionally keeping news/FinBERT embeddings decoupled (`include_news_features=False`) from the 5-day neural core.

## Features

- Forecast 5-day cumulative log-returns using **65% univariate zero-shot `Chronos-2` (`amazon/chronos-2`) + 35% 3-seed compact Modern TFT-ASRO** (providing macro-regime stabilization & VSN/Attention explainability) with RSS quantile paths and causal **HAR-GJR** dynamic volatility + Split CQR risk cones (`Calm` / `Elevated` / `High Stress`) — no artificial post-hoc amplitude scaling
- Produce weekly causal 20D/60D structural macro outlooks (`long_term/`) using FRED real yields, inflation breakevens, broad USD, and miner cross-asset ratios
- Clean unified 5D workspace without conflicting T+1 XGBoost tree-driver cards in daily commentary or Overview UI (`XGBoost` routine triggers/snapshots retired)
- Score news sentiment using multi-tier LLM routing (NVIDIA Nemotron 3.5 Super VL / Liquid LFM 2.5) with FinBERT fallback (post-model commentary & risk layer)
- Extract FinBERT CLS embeddings (768-dim → 8-dim PCA) for deep semantic encoding optimized for sample efficiency
- Track 17 correlated assets via configurable symbol sets (active, champion, challenger)
- Aggregate daily sentiment using time-weighted exponential decay
- Compute advanced sentiment features: momentum, surprise (Z-score), volume-weighted scores
- Ingest LME warehouse stock data and futures curve signals for physical-market context
- Generate AI-powered market commentary and dynamic in-character Cuprum companion dialogue powered primarily by IFM K2-Horizon (375B reasoning model) with fail-open OpenRouter fallback
- Interactive Cuprum Companion: living reactive market spirit with dynamic moods (Molten Bull, Patina Bear, Tempered Neutral) and RAG-grounded daily dialogue nodes ('What is pulling at your wires?', 'Will it hold?', 'Tell me something I don't know') that explain model drivers in plain terms
- Interactive S&P/Finviz-style Market Heatmap treemap with live quotes, sector grouping, logos and sparklines
- News Intelligence panel with drawer inspection, commodity relevance filtering and LLM reasoning
- Multi-page workspace (`/` Landing, `/dashboard` Overview, `/models` Models, `/validation` Backtesting, `/system` Status)
- Enforce strict promotion Quality Gate (Weekly DA, Magnitude Ratio, Sharpe, Tail Capture, monotonic quantile guards)
- Pipeline execution tracking via `pipeline_run_metrics` table
- Optimize TFT hyperparameters via Optuna TPE search with variance ratio penalty and deterministic replay
- Dual storage model persistence with HuggingFace Hub checkpoints and atomic database artifacts

## Cuprum Companion

**Cuprum** is CopperMind's reactive market spirit surfaced in `/dashboard`, shifting dynamically with model consensus, volatility, and RAG-grounded top drivers in plain human language (`greeting`, `drivers_reply`, `outlook_reply`):

<div align="center">

| <img src="frontend/public/assets/cuprum/cuprum-molten-bull.png" width="96" alt="Molten Bull" /> | <img src="frontend/public/assets/cuprum/cuprum-tempered-neutral.png" width="96" alt="Tempered Neutral" /> | <img src="frontend/public/assets/cuprum/cuprum-patina-bear.png" width="96" alt="Patina Bear" /> |
| :---: | :---: | :---: |
| **Molten Bull** | **Tempered Neutral** | **Patina Bear** |
| Consensus upside alignment with strong macro & miner confirmation | Mixed or range-bound signals; highlights opposing wires calmly | Downside / USD headwind regime with defensive risk-band context |

</div>

## Architecture

```mermaid
flowchart TB
    classDef fe fill:#141A26,stroke:#61DAFB,color:#F0F6FC,stroke-width:1.5px
    classDef api fill:#141A26,stroke:#E29A5F,color:#F0F6FC,stroke-width:1.5px
    classDef worker fill:#141A26,stroke:#3FB950,color:#F0F6FC,stroke-width:1.5px
    classDef store fill:#0D111A,stroke:#7AA2F7,color:#F0F6FC,stroke-width:1.5px

    subgraph Frontend["Vercel SPA + Prerendered Edge"]
        UI["React 18 · Vite 7 · Tailwind v4<br/>/ · /dashboard · /models · /validation · /system"]:::fe
    end

    subgraph Backend["Hugging Face Space · FastAPI + Arq Worker"]
        API["Snapshot-Only REST API (main.py)<br/>/api/analysis · /api/analysis/tft · /api/explain<br/>/api/market-heatmap · /api/commentary · /api/models/tft/summary"]:::api
        CRON["GitHub Actions Schedulers<br/>Daily 00:10 UTC · TFT Sun 21:00 UTC · Screener Sun 18:00 UTC"]:::worker
        WRK["Async Arq Worker (worker/runner.py)<br/>yfinance · Google RSS · LLM Router + FinBERT"]:::worker
        MOD["Horizon-Separated Forecast Stack<br/>1) 65% Chronos-2 + 35% TFT-ASRO + HAR-GJR Split CQR (5D)<br/>2) XGBoost Regressor (Legacy T+1 Read-Only)<br/>3) long_term/ Causal Macro Scorecard (20D/60D)"]:::worker
    end

    subgraph Data["Persistence & Registries"]
        PG[("Supabase PostgreSQL<br/>price_bars · snapshots · tft_model_metadata<br/>backtest_reports · pipeline_run_metrics")]:::store
        RED[("Redis Queue")]:::store
        HUB[("HuggingFace Hub<br/>3-Seed Checkpoints · CQR · SHA Manifest")]:::store
    end

    UI <-->|"HTTPS / JSON"| API
    API <-->|"Cached Snapshots Only"| PG
    API -->|"Enqueue Job"| RED
    CRON -->|"POST /api/pipeline/trigger"| API
    RED --> WRK
    WRK --> MOD
    MOD -->|"Quality-Gated Write"| PG
    MOD <-->|"Revision-Synced Artifacts"| HUB
```

<details>
<summary><strong>Full Endpoint & Table Breakdown (Expand)</strong></summary>

- **REST API (`backend/app/main.py`):** `/api/analysis`, `/api/analysis/tft/{symbol}`, `/api/analysis/consensus`, `/api/explain/{symbol}`, `/api/history`, `/api/market-heatmap`, `/api/market-prices`, `/api/commentary`, `/api/news`, `/api/models/tft/summary`, `/api/health`, `/api/pipeline/trigger`, `/api/pipeline/runs/{id}`.
- **Supabase Tables:** `price_bars`, `news_raw`, `news_processed`, `news_sentiments_v2`, `news_embeddings`, `daily_sentiments_v2`, `lme_warehouse_data`, `analysis_snapshots`, `tft_prediction_snapshots`, `ai_commentaries`, `model_metadata`, `model_artifacts`, `tft_model_metadata`, `heatmap_cache`, `backtest_reports`, `pipeline_run_metrics`.

</details>

## Symbol Sets

The system separates symbol discovery, testing and production into distinct phases.

### How Symbols Are Found

The **Screener module** identifies symbols correlated with COMEX Copper (HG=F):

1. **Universe Builder**: Loads ~3000 candidate tickers from seed files (ETFs, miners, macro indices)
2. **Data Probe**: Fetches 2 years of weekly returns for each candidate via yfinance
3. **Correlation Screening**: Computes Pearson and partial correlations with HG=F
4. **Lead-Lag Analysis**: Tests if symbol leads or lags copper by 0-4 weeks
5. **Stability Check**: Validates correlation holds in both in-sample and out-of-sample periods

Output: A ranked list of symbols with stable, significant correlations.

### Champion / Challenger Testing

Symbol sets are managed via JSON files in `backend/config/symbol_sets/`:

| File                | Purpose                                                                       |
| ------------------- | ----------------------------------------------------------------------------- |
| `active.json`     | **Production set** - Currently used for training. This is the "winner". |
| `champion.json`   | **Previous best** - Backup of last known good set before changes.       |
| `challenger.json` | **Candidate set** - New symbols being tested. Not used in production.   |

**Flow**:

1. Run screener → produces candidate symbols
2. Add candidates to `challenger.json`
3. Train model with `SYMBOL_SET=challenger`, compare MAE/RMSE to active
4. If challenger outperforms → promote to `active.json`, demote old active to `champion.json`
5. If not → discard or iterate

This allows risk-free experimentation without breaking production.

### Current Active Set (17 symbols)

Used for XGBoost feature engineering. Loaded from `backend/config/symbol_sets/active.json` (v2.0.0, promoted after screener correlation and walk-forward backtest validation with 8.73% MAE improvement).

| Category | Symbols | Rationale |
| -------- | ------- | --------- |
| Target | HG=F | COMEX Copper Futures - prediction target |
| Macro & FX | DX-Y.NYB, CL=F, FXA, CEW | USD Index, crude oil proxy, Australian Dollar trust, emerging market currency ETF |
| Industrial Metals | PL=F, ALI=F | Platinum futures, Aluminum futures (correlated industrial metal demand) |
| Mining & Sector ETFs | COPX, PICK, XME, GXC | Copper Miners ETF, Global Metals & Mining, SPDR Metals & Mining, SPDR China ETF |
| Producers & Miners | BHP, FCX, RIO, SCCO, VALE, LUN.TO, HBM.TO | Diversified majors and pure-play copper miners (Freeport, Southern Copper, Lundin, Hudbay) |

### Dashboard Symbols (14) - Separate List

Used for real-time price display only. Configured via `YFINANCE_SYMBOLS` env var. Does not affect model training.

| Category | Symbols                                            |
| -------- | -------------------------------------------------- |
| Target   | HG=F                                               |
| Macro    | DX-Y.NYB, CL=F                                     |
| ETFs     | FXI, COPX, COPJ                                    |
| Miners   | BHP, FCX, SCCO, RIO, TECK, IVN.TO, LUN.TO, 2899.HK |

### Switching Symbol Sets

```bash
# Default: uses active.json
SYMBOL_SET=active

# Test challenger set without changing production
SYMBOL_SET=challenger python -m app.ai_engine --train-only

# Compare metrics, then promote if better
cp config/symbol_sets/active.json config/symbol_sets/champion.json
cp config/symbol_sets/challenger.json config/symbol_sets/active.json
```

### Audit Trail

Each model training run records:

- `symbol_set_name`: Which set was used (active/champion/challenger)
- `training_symbols`: Full list of symbols
- `training_symbols_hash`: SHA256 hash for reproducibility

## Model Details

### Target Definition

The model predicts **next-day simple return**, not price:

```
target = (close[t+1] / close[t]) - 1
```

This is recorded in model metadata as:

- `target_type`: `"simple_return"`
- `target_shift_days`: `1`
- `target_definition`: `"simple_return(close,1).shift(-1)"`
- `baseline_price_source`: `"yfinance_close"`

### Price Calculation

```
predicted_price = baseline_price × (1 + predicted_return)
```

Where `baseline_price` is the latest HG=F close from the database.

### XGBoost Parameters

| Parameter            | Value | Purpose                              |
| -------------------- | ----- | ------------------------------------ |
| `max_depth`        | 4     | Shallow trees, prevents overfitting  |
| `learning_rate`    | 0.05  | Slow learning, better generalization |
| `subsample`        | 0.8   | 80% of data per tree                 |
| `colsample_bytree` | 0.6   | 60% of features per tree             |
| `reg_alpha` (L1)   | 0.5   | Sparsity regularization              |
| `reg_lambda` (L2)  | 2.0   | Weight decay                         |
| `min_child_weight` | 5     | Minimum samples per leaf             |

These settings are **conservative** - the model avoids large predictions.

### Feature Count

Approximately 250 features are generated for XGBoost:

- Technical indicators per symbol (RSI, SMA, volatility, returns)
- Lag features (1-5 day lags)
- Cross-asset correlations
- Sentiment features (index, news count)

### TFT-ASRO Deep Learning

The primary 5-day probabilistic engine promotes **`Chronos-2` (`amazon/chronos-2`, 65% primary price-dynamics forecaster, fail-open)** paired with a **3-seed compact Modern TFT-ASRO ensemble (35% multivariate macro-regime stabilizer + VSN/Attention XAI engine)** (`pytorch-forecasting` + PyTorch Lightning), RSS quantile horizon aggregation, and linear-volatility Split CQR calibration. Daily XGBoost $T+1$ snapshots and conflicting UI drivers are retired, leaving `/api/analysis` strictly read-only while `long_term/` serves the 20D/60D structural horizon.

#### Model Architecture

| Parameter                  | Value            | Notes                                                             |
| -------------------------- | ---------------- | ----------------------------------------------------------------- |
| `hidden_size`              | `12`             | Compact capacity limits overfitting on ~175 independent weekly blocks |
| `hidden_continuous_size`   | `6`              | Paired compact continuous projection                              |
| `attention_head_size`      | `1`              | Single temporal head prevents attention fragmentation on small series |
| `dropout`                  | `0.30`           | Regularization without directional signal collapse                |
| `learning_rate`            | `5e-4`           | AdamW backbone LR (`1e-4`-band differential head & bias decay)    |
| `output_head_lr_mult`      | `0.35`           | Weight LR multiplier (biases capped at `0.05x`); lifts over-compression |
| `weight_decay`             | `1.5e-2`         | L2 regularization via decoupled `AdamW`                           |
| `max_encoder_length`       | `15`             | 15 trading days (3 weeks) of stationary context                   |
| `max_prediction_length`    | `5`              | 5-day cumulative log-return horizon                               |
| `gradient_clip_val`        | `0.5`            | Gradient norm stabilization                                       |
| `batch_size`               | `32`             | Chronological batches with `val_ratio=0.08`, `test_ratio=0.10`     |
| `lookback_days`            | `1825`           | 5 years of weekday-only deduplicated futures bars                 |
| `ensemble_seeds`           | `(42, 137, 256)` | Filtered on validation MAE/0 & drift bounds (no `val_mean` shift) |
| `foundation_prior.weight`  | `0.65`           | Primary 65% `amazon/chronos-2` + 35% TFT-ASRO (validation-guarded, fail-open) |

#### Quantile Output, RSS Aggregation & Split CQR

The model outputs 7 strictly ordered quantiles over realized 1-day causal log returns (`target = log(close[t]/close[t-1])`):

- `q0.02`, `q0.10`, `q0.25`, **`q0.50` (median)**, `q0.75`, `q0.90`, `q0.98`

Multi-day uncertainty is aggregated using **root-sum-of-squares (RSS)** L2 quantile half-widths (`aggregate_weekly_quantiles_np`) alongside additive medians, preserving strict monotonicity (`quantile_crossing_rate == 0`) without assuming perfect daily comonotonicity. At calibration time (validation split only), **linear-volatility Split CQR** (`base_vol_power=1.0`, normalized by `realized_vol_20d`) tunes empirical 80% and 96% prediction intervals before test evaluation or live inference.

#### WeeklyASROPFLoss Formulation

`WeeklyASROPFLoss` overrides `Metric.update()` so batch backpropagation and `val_loss` share exact raw batch scale (eliminating `MultiHorizonMetric` `B * H` dilution):

| Component                      | Weight / Default | Purpose                                                                 |
| ------------------------------ | ---------------- | ----------------------------------------------------------------------- |
| Weekly quantile pinball        | `1.00`           | Proper-scoring calibration across all 7 weekly quantiles                |
| Auxiliary $T+1$ pinball        | `0.20`           | Short-step anchor scaled identically to weekly pinball loss             |
| Sub-horizon pinball            | `0.15`           | Smooths intermediate path steps `[t+2..t+4]`                            |
| Zero-drift L1+L2 prior         | `0.25`           | Prevents single-sign collapse without artificially damping trend amplitude |
| Demeaned directional utility   | `0.12`           | Batch-demeaned ASRO Sharpe alignment (`WeeklyTargetScale`-normalized)   |
| Legacy batch heuristic terms   | `0.00`           | `lambda_bias`, `dispersion`, `magnitude`, `naive`, `positive_rate` off  |

#### Hybrid Inference Flow (65% Chronos-2 Primary + 35% Modern TFT-ASRO)

Training (`trainer.py`), rolling walk-forward validation (`backtest/runner.py`), and live inference (`predictor.py`) execute identical stages in strict order:

```mermaid
flowchart LR
    X["core5 Stationary Features<br/>(15d x 5 + day_of_week)"] --> TFT["3-Seed Modern TFT-ASRO<br/>(35% Stabilizer + VSN XAI)"]
    C["Causal HG=F Closes<br/>(252d Univariate Zero-Shot)"] --> C2["amazon/chronos-2<br/>(65% Primary 9 -> 7 Log Quantiles)"]
    C2 -->|"Val Sanity Gate<br/>(Fail-Open w=0.65)"| BL["Ordered-Space Blend<br/>0.65 Chronos-2 + 0.35 TFT"]
    TFT --> BL
    BL --> CQR["Linear-Vol Split CQR<br/>(realized_vol_20d)"]
    CQR --> PUB["Public Simple Returns<br/>& Price Bands (q02..q98)"]
```

1. **Modern TFT-ASRO Ensemble (35%, 3 seeds):** Predicts ordered 5-day quantile paths from stationary `core5` inputs and drives VSN/Attention explainability (`/api/explain/{symbol}`).
2. **Causal `Chronos-2` Univariate Zero-Shot (`amazon/chronos-2`, 65% Primary):** Takes only trailing historical `HG=F` closes up to origin `t` (`context_len=252` trading days, zero covariates, zero future leakage) and maps 9 step-5 price quantiles into 7 cumulative log-return quantiles. Because price-patch transformers are free of L1/MSE return-regression shrinkage toward `0`, `Chronos-2` preserves authentic directional amplitude during sharp selloffs without post-hoc scaling.
3. **Zero-DoF Validation Sanity Gate:** Enables the `0.65` weight (clip ceiling `0.85`) only if `val_foundation_mae_vs_naive <= 1.25`, blend MAE degradation `<= 0.05`, and blended positive rate avoids single-sign collapse (`[0.15, 0.85]`).
4. **Monotonic Ordered-Space Blend (`0.35 * W_TFT + 0.65 * W_C2`):** Shifts daily medians and scales quantile half-widths consistently with RSS aggregation, failing open to `100%` TFT-ASRO if `Chronos-2` is unavailable at runtime.
5. **Split CQR & Contract Formatting:** Applies validation-fitted volatility-scaled CQR intervals and converts internal log returns to public simple returns and price targets.

#### TFT Feature Categories

- **Time-varying unknown (`core5`, `mrmr_top_k=5` locked):** `HG=F_ret5`, `HG=F_close_sma_20_ratio`, `HG=F_signed_vol_ratio`, `HG=F_theta_5d_prior`, `macro_fair_value_zscore`. Raw unnormalized price levels (`include_raw_ma=False`) and unsigned `*_vol_*` features are excluded from encoder inputs (`realized_vol_20d` is retained strictly for Split CQR normalization).
- **Time-varying known:** `day_of_week` normalized over Mon–Fri (`clip(dayofweek, 0, 4) / 4.0`). Weekend ghost futures bars (`dayofweek >= 5`) and static bias channels (`add_target_scales`, `add_encoder_length`, `add_relative_time_idx`) are stripped.
- **Post-model news & sentiment:** Live news/FinBERT DB lookups are decoupled from TFT training (`include_news_features=False`) and operate in the institutional commentary, risk, and Cuprum companion layer.

#### Architecture Evolution & Benchmark Rationale

Rather than choosing model families or hyperparameters from single runs, we evaluated **29 models and variants** across **8 contiguous purged expanding-window folds × 250 test days = 2,000 OOS trading days (~400 non-overlapping 5D weeks, `2018-12-10` to `2026-11-18`)** plus a **750-day Modern Regime slice (`2023-11-27` to `2026-11-18`, Folds 5–7)** on deduplicated, weekend-free futures bars, scored with paired circular block-bootstrap tests and unrelaxed `app/quality_gate.py` checks.

<p align="center">
  <img src="docs/assets/benchmark-frontier.png" alt="29-Model Multi-Regime Benchmark Frontier" width="100%" />
</p>

<p align="center">
  <img src="docs/assets/amplitude-polarity-audit.png" alt="Modern Regime Polarity Balance & Amplitude Recovery Audit" width="100%" />
</p>

<details>
<summary><strong>Full 29-Model Benchmark Leaderboard &amp; What Failed vs. What Worked (Expand)</strong></summary>

| Model / Variant | Fold Gate (8F) | Pooled DA (NoOv) | Pooled HAC Sharpe (NoOv) | Modern HAC Sharpe | Pooled `MAE/0` | Modern `MAE/0` | Med \|Pred\| | Verdict |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| **`80% Modern TFT + 20% Chronos-2` (Selected)** | **6/8** | **54.2%** *(—)* | **+0.351** | **+0.347** | **0.9944** (#1) | **0.9877** (#1) | **0.15–0.20%** | **Selected: best overall accuracy/MAE/regime balance** |
| `dual_tft_5y_15y_core5` (Unweighted 5y+15y) | 3/8 | 56.0% (57.8%) | **+0.509** (+0.941) | **+0.625** | 0.9961 | 0.9942 | 0.09% | Highest Sharpe, but 74% modern bull skew & 3/8 gate |
| `tft_modern_5y_amp` (Standalone Modern TFT) | 4/8 | 52.6% (55.0%) | +0.280 (+0.693) | +0.375 | 0.9955 | 0.9921 | 0.12% | #1 standalone MAE/0, 49.9% neutral polarity in modern slice |
| `tft_modern_15y_expdecay` (3y half-life) | 4/8 | 52.1% (56.0%) | +0.176 (+0.658) | +0.544 | 1.0062 | 1.0064 | 0.14% | Strong modern Sharpe & 50.1% balance; slight MAE cost |
| `chronos_2_univariate_zs` (Zero-shot 252d) | **7/8** | 52.7% (54.7%) | +0.199 (+0.582) | +0.127 | 1.0315 | 1.0267 | **0.39%** | Best gate stability & Fold 7 (`DA 58.4%`, `SR +0.70`), weak long-run MAE |
| `chronos_2_cov_core5_zs` (Multivariate ZS) | 5/8 | 50.7% (51.2%) | +0.039 (+0.168) | +0.045 | 1.0355 | 1.0231 | 0.48% | Raw zero-shot covariates hurt vs univariate price series |
| `chronos_bolt_small_zs` | 7/8 | 52.3% (57.3%) | +0.120 (+0.548) | -0.095 | 1.0272 | 1.0335 | 0.55% | Negative modern-regime Sharpe; inferior to Chronos-2 |
| `timesfm_2p5_200m_zs` (Google TimesFM 2.5) | 6/8 | 50.9% (51.7%) | +0.024 (+0.355) | -0.175 | 1.0035 | 1.0104 | 0.24% | Negative modern-regime Sharpe (-0.175); outperformed by Chronos-2 |
| `ridge_volnorm_expanded` | 2/8 | 49.7% (51.5%) | +0.112 (+0.420) | +0.299 | 1.0136 | 1.0167 | 0.17% | Weak Pinball skill (-0.1405) & high MAE/0 |
| `catboost_quantile_expanded` | 2/8 | 51.2% (52.5%) | +0.068 (+0.184) | +0.486 | 1.0035 | 0.9994 | 0.09% | 2/8 gate passes & weak pooled Sharpe |
| `lgbm_quantile_expanded` | 2/8 | 50.0% (52.0%) | -0.023 (+0.151) | -0.026 | 1.0045 | 1.0011 | 0.12% | Negative pooled & modern Sharpe |
| `dlinear_quantile_expanded` | 4/8 | 48.6% (49.2%) | -0.028 (-0.081) | +0.194 | 1.0796 | 1.0606 | 0.71% | Severe MAE degradation under noisy 5D returns |
| `patchtst_micro_expanded` | 4/8 | 48.2% (49.5%) | -0.072 (-0.161) | +0.097 | 1.0451 | 1.0358 | 0.53% | Negative pooled Sharpe & high MAE/0 |
| `theta_126d` (Statistical baseline) | 4/8 | 50.1% (52.5%) | -0.109 (+0.031) | -0.163 | 1.0036 | 0.9974 | 0.22% | Beaten by TFT-ASRO ($p = 0.001$ on Sharpe & Pinball) |
| `panel_tft_15y` (Multi-asset shared TFT) | 2/8 | 48.2% (48.2%) | -0.241 (-0.438) | -0.088 | 1.0008 | 0.9992 | 0.17% | Cross-asset weight sharing hurt copper specificity |
| `tft_asro_15y_macro_cot` (30 FRED/COT feats) | 1/8 | 45.6% (46.8%) | -0.252 (-0.484) | -0.358 | 1.0116 | 1.0237 | 0.11% | Worst model: 30 macro/COT columns caused severe 5D overfitting |

**Key Architectural Takeaways (What Failed vs. What Worked):**

1. **Why we kept TFT-ASRO over Tabular / Lightweight Neural Models:** On paired block-bootstrap tests over 2,000 OOS days, TFT-ASRO outperformed `naive_zero`, `theta_126d` ($\Delta \text{Sharpe} = +0.64 \text{ to } +1.06$, $p \le 0.044$), `Ridge`, `LightGBM`, `CatBoost`, `DLinear`, `PatchTST`, and multi-asset `Panel-TFT` across Sharpe, MAE/0, and Pinball skill.
2. **Why we locked `core5` (`mrmr_top_k=5`) on 5D and kept FRED Macro in `long_term/`:** Expanding 5D inputs to 30 features (`FRED` rates/breakevens/dollar + `CFTC COT` positioning + extra assets) degraded **every** model family (`tft_asro_15y_macro_cot` dropped to `DA=45.6%`, `HAC SR=-0.252`). Slow macro and positioning signals belong in the 20D/60D `long_term/` layer, not a 5-day horizon.
3. **Why we fixed forecast flatness without artificial upscaling:** Actual 5D copper moves have median magnitude `1.93%`, while old TFT medians averaged `0.09%`. Multiplying predictions up artificially (`__amp_scaled` or `DLinear`'s `0.71%–0.90%`) wrecked `MAE/0` (`1.025–1.131`, bootstrap $p \le 0.001$) because 5D predictable $R^2$ is ~1–3%. Relaxing head learning rate (`output_head_lr_mult=0.35`), tuning drift regularization (`lambda_drift_prior=0.75`), and blending **20% `Chronos-2`** lifted amplitude naturally to `0.15–0.20%` while achieving **the #1 lowest `MAE/0`** (`0.9944` pooled / `0.9877` modern).
4. **Why we paired Modern TFT-ASRO with 20% `Chronos-2` (instead of running either alone or fitting weights on validation):** `Chronos-2` univariate zero-shot excelled at stability (`7/8` gate passes) and 2025–26 structural shifts (`Fold 7 DA=58.4%, SR=+0.70`), whereas `Modern TFT-ASRO` excelled at cross-asset copper accuracy (`MAE/0=0.9955`) and modern polarity balance (`49.9%` vs. the old `74%` bull bias). Grid-searching the blend weight on a ~20-week validation split (`ValFit`) overfit and dropped OOS Sharpe (`+0.351 -> +0.172`), so we adopted a **fixed 80/20 structural blend with a validation sanity guard**.
5. **Why we tightened Optuna bounds & cleaned `price_bars`:** Removing `lambda_drift_prior < 0.75` and locking `mrmr_top_k=5` eliminated the Nov 23 Optuna `0.15` overfit (`94.8%` positive bull collapse), and deduplicating `16,532` duplicate/weekend futures bars in Supabase ensured clean training and evaluation boundaries.

</details>

#### Benchmark V2 Summary (34+ Models + Live Nov 2026 `-15%` Crash Audit)

Building on our initial 29-model 8-fold study (2,000 OOS trading days), **Benchmark V2** investigated why 5D point forecasts remained compressed (`±0.05%–0.15%`) during real market moves — including the November 2026 `-15%` COMEX copper selloff — evaluating 12 additional Direct-5D neural/tabular (`Student-T VSN-Attn`, `TCN`, `Two-Stage Bayesian CatBoost/Logistic`, `Ridge Z`) and LoRA/Full fine-tuned `Chronos-2` variants:

- **Why Direct-5D Return Regressors Failed:** Solving amplitude with volatility-normalized direct 5D neural/tabular heads (`med|p| = 0.40%–1.28%`) learned counter-trend mean-reversion (`Crash DA 30%–37%`), emitting wrong-way rally forecasts (`+0.38%..+1.03%`) throughout the live Nov 2026 selloff.
- **Why Promoting `Chronos-2` Solves Amplitude Without Artificial Scaling:** Zero-shot `Chronos-2` operates on price patches rather than L1-shrunk noisy return targets (`blend_60c2zs_40tft` hits `HAC Sharpe +0.285`, `Fold 7 DA 61.2%`, `SR +0.76`, `MAE/0 0.971`, and emitted `-0.80%..-1.19%` bearish signals during the Nov 2026 crash), while fine-tuned `chronos2_ft_lora_panel` passed all `8/8` quality gates (`DA 53.4%`, `HAC SR +0.230`). Promoting zero-shot `Chronos-2` to `65%` (with `35%` `TFT-ASRO`, `lambda_drift_prior=0.25`) naturally increases median forecast responsiveness ~2.5×–3× without ad-hoc multipliers or GPU/LoRA overhead in weekly CI.
- **Why Routine XGBoost Drivers Were Retired:** Keeping conflicting T+1 XGBoost tree weights (`FCX_ret1`, `SCCO_RSI_14`) beside 5D TFT-ASRO XAI misled users and blocked commentary; routine daily XGBoost snapshots/retrains and Overview UI cards are retired cleanly while `/api/analysis`, `screener/`, and `backtest/` remain intact.

#### Training and Optimization

TFT training runs weekly (Sundays at 21:00 UTC via `.github/workflows/tft-training.yml`) or on-demand:

```bash
cd backend

# Train Modern TFT-ASRO + Chronos-2 hybrid with Optuna results (or KNOWN_GOOD_CONFIG)
python -m deep_learning.training.trainer --symbol HG=F

# Deterministic validation replay (bypasses Optuna overlays)
python -m deep_learning.training.trainer --symbol HG=F --deterministic-weekly-validation

# Run Optuna hyperparameter search (mrmr_top_k locked to 5, lambda_drift_prior >= 0.25)
python -m deep_learning.training.hyperopt --n-trials 30
```

#### Financial Metrics & Promotion Quality Gate

Candidate checkpoints never overwrite active serving metadata or HF Hub state until `backend/app/quality_gate.py` passes **all** primary weekly blocking checks and HuggingFace Hub upload succeeds:

| Gate Metric | Threshold | Invariant Rationale |
| ----------- | --------- | ------------------- |
| **Weekly Directional Accuracy** | $\ge 0.49$ ($\ge 0.47$ if $150 \le N < 300$, $\ge 0.45$ if $N < 150$) | Sample-size-aware directional edge floor |
| **Weekly Magnitude Ratio** | $[0.02, 2.50]$ (`raw <= 3.0`) | Prevents trivial zero collapse or neural output explosion |
| **Weekly MAE vs Naive Zero** | $\le 1.12$ | Guards point accuracy under noisy commodity returns |
| **Weekly Sign Collapse Guard** | Rejects `pred_pos > 0.92` (`actual < 0.67`) or `< 0.08` (`actual > 0.33`) | Blocks degenerate single-direction models |
| **Weekly PI80 Coverage** | Binomial tolerance around `0.80` (`width_ratio <= 2.2`) | Sample-size-aware 80% Split CQR interval calibration |
| **Weekly PI96 Coverage** | $\ge 0.88$ ($\ge 0.85$ if $n_{\text{eff}} < 25$, `width_ratio <= 3.2`) | Tail risk band validity |
| **Weekly HAC Sharpe Ratio** | $\ge -0.35$ (`sqrt(52)` Newey-West HAC) | Accounts for overlapping 5-day return autocorrelation |
| **Quantile Crossing Rate** | $\le 0.001$ (raw, ordered & post-CQR) | Zero tolerance for quantile inversions or median sort gaps |

#### Model Persistence

TFT artifacts are stored locally in `$MODEL_DIR` and synchronized with HuggingFace Hub (`ifieryarrows/copper-mind-tft`) with revision hash tracking (`sync_tft_artifacts_if_stale`):

| Artifact | Description |
| -------- | ----------- |
| `best_tft_asro.ckpt`, `_seed1.ckpt`, `_seed2.ckpt` | Accepted 3-seed PyTorch Lightning ensemble checkpoints |
| `tft_metadata.json` | Metrics, quality gate audit, hybrid/foundation prior params & feature contract |
| `conformal_calibration.json` | Validation-fitted linear-volatility Split CQR parameters |
| `optuna_results.json` | Validation-selected Optuna hyperparameters & structural diagnostics |
| `artifact_manifest.json` | SHA-256 hashes and contract health metadata |
| `pca_finbert.joblib` | Optional FinBERT PCA artifact (post-model / legacy news mode) |

## Directory Structure

<details>
<summary><strong>Full Repository Tree (<code>backend/</code>, <code>frontend/</code>, <code>.github/workflows/</code>)</strong></summary>

```
terra-rara/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app & REST endpoints
│   │   ├── ai_engine.py      # XGBoost training, LLM sentiment scoring
│   │   ├── inference.py      # XGBoost live prediction & consensus
│   │   ├── heatmap.py        # S&P/Finviz-style market heatmap engine
│   │   ├── quality_gate.py   # Shared TFT promotion quality gate
│   │   ├── features.py       # Technical indicator computation
│   │   ├── data_manager.py   # News ingestion & market price fetcher
│   │   ├── ifm_client.py     # IFM K2-Horizon async client & error handling
│   │   ├── commentary.py     # AI commentary generation (IFM K2 & OpenRouter)
│   │   ├── companion.py      # Cuprum companion state builder & dialogue extraction
│   │   ├── models.py         # SQLAlchemy ORM models
│   │   ├── db.py             # Database engine & driver normalization
│   │   └── settings.py       # Pydantic settings & env management
│   ├── adapters/             # Infrastructure adapters (Redis, Queue, Locks)
│   ├── worker/               # Background task worker
│   │   ├── runner.py         # Worker process entrypoint
│   │   └── tasks.py          # Daily pipeline orchestration tasks
│   ├── deep_learning/        # TFT-ASRO deep learning module
│   │   ├── config.py         # All TFT hyperparameters & loss weights
│   │   ├── contract.py       # Primary & auxiliary forecast contracts
│   │   ├── data/
│   │   │   ├── embeddings.py         # FinBERT CLS extraction + 8-dim PCA
│   │   │   ├── sentiment_features.py # Momentum, surprise, volume-weighted
│   │   │   ├── lme_warehouse.py      # LME warehouse stock data + features
│   │   │   ├── futures_curve.py      # Contango/backwardation signals
│   │   │   ├── feature_store.py      # Centralized data fusion for TFT
│   │   │   └── dataset.py            # TimeSeriesDataSet builder
│   │   ├── calibration/
│   │   │   ├── foundation_prior.py   # Fail-open Chronos-2 prior & monotonic blend
│   │   │   ├── hybrid.py             # Causal Theta prior & macro fair-value z-score
│   │   │   └── conformal.py          # Linear-volatility Split CQR calibration
│   │   ├── models/
│   │   │   ├── tft_copper.py         # TFT model + WeeklyASROPFLoss
│   │   │   ├── losses.py             # ASRO + WeeklyLoss + MADL
│   │   │   ├── monotonic_quantiles.py# RSS horizon & non-crossing projection
│   │   │   └── hub.py                # HuggingFace Hub upload/download & sync
│   │   ├── training/
│   │   │   ├── trainer.py            # 3-seed PyTorch Lightning training loop
│   │   │   ├── hyperopt.py           # Optuna TPE hyperparameter search
│   │   │   └── metrics.py            # HAC Sharpe, Sortino, DA, tail capture
│   │   └── inference/
│   │       └── predictor.py          # TFT + Chronos-2 live prediction pipeline
│   ├── long_term/                    # Causal weekly 20D/60D FRED macro outlook
│   ├── config/
│   │   └── symbol_sets/      # Training symbol configurations
│   │       ├── active.json   # Current training symbols (v2.0.0)
│   │       ├── champion.json # Best performing backup set
│   │       └── challenger.json
│   ├── pipelines/            # Reproducible data processing
│   │   ├── ingestion/        # RSS/API → news_raw
│   │   ├── processing/       # news_raw → news_processed
│   │   └── cutoff.py         # Market cut-off calculation
│   ├── screener/             # Universe Builder + Feature Screener
│   │   ├── core/             # Config, fingerprint, cache
│   │   ├── contracts/        # Pydantic output models
│   │   ├── universe_builder/ # Seed loading, probing, categorization
│   │   └── feature_screener/ # Correlation analysis
│   ├── backtest/             # Walk-forward backtesting
│   ├── migrations/           # SQL schema migrations (001-004)
│   ├── tests/                # pytest test suite
│   │   ├── deep_learning/    # TFT-ASRO unit tests (36 test files)
│   │   ├── screener/         # Screener unit tests (9 test files)
│   │   └── ...               # Core API & pipeline tests (23 test files)
│   ├── Dockerfile            # Container configuration
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Router & workspace lifecycle
│   │   ├── api.ts            # API client (XGBoost + TFT + Heatmap)
│   │   ├── types.ts          # TypeScript domain interfaces
│   │   ├── pages/            # Workspace views (Overview, Models, Validation, System)
│   │   ├── features/         # Modular feature slices (Heatmap, News, Forecast, Landing)
│   │   ├── layouts/          # AppShell & workspace navigation
│   │   └── components/       # Design system, charts & skeleton loaders
│   ├── scripts/              # Static prerendering & budget benchmarks
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
├── data/
│   └── models/               # Local model artifacts (gitignored)
├── .github/
│   └── workflows/
│       ├── daily-pipeline.yml               # Daily pipeline @ 23:00 UTC (XGBoost retrain on Sun)
│       ├── tft-training.yml                 # Weekly TFT retrain, Optuna hyperopt & HF upload
│       ├── tft-deterministic-validation.yml # Deterministic validation replay
│       ├── backtest.yml                     # Champion vs Challenger backtesting
│       ├── screener-schedule.yml            # Automated correlation screening
│       ├── hf-sync.yml                      # GitHub → HuggingFace Space sync
│       └── tests.yml                        # Automated CI test suite
├── docker-compose.yml
├── env.example
└── README.md
```

</details>

## Screener Module

The screener module identifies symbols with stable correlations to COMEX Copper (HG=F) for feature engineering. It provides audit-first, reproducible analysis with full data lineage.

### CLI Usage

```bash
cd backend

# Build universe from seed sources
python -m screener universe_builder --config config/screener_config.yaml

# Run correlation screening
python -m screener feature_screener --universe artifacts/universes/latest/universe.json
```

### Dual Fingerprint System

The screener uses two fingerprints to ensure both reproducibility and auditability:

- **`content_fingerprint`**: Deterministic hash of analysis content only. Same inputs + same config = same hash. Excludes timestamps and run IDs.
- **`output_fingerprint`**: Hash of full output envelope including metadata. Changes with each run.

### Key Features

| Feature                  | Description                                               |
| ------------------------ | --------------------------------------------------------- |
| Weekly via 1d + resample | Downloads daily data, resamples to W-FRI for consistency  |
| Pairwise dropna          | Correlation uses per-pair intersection, not global dropna |
| Frozen lead-lag          | Best lag discovered in IS, frozen for OOS evaluation      |
| Partial correlation      | Residual correlation with ^GSPC/UUP controls              |
| Multi-source provenance  | Each ticker tracks all sources it appeared in             |
| Collision-proof cache    | FetchParams fingerprint prevents cache key collisions     |

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL database (or Supabase account)
- IFM API key (recommended primary for high-speed institutional commentary & Cuprum dialogue, available at [ifm.ai](https://ifm.ai/))
- OpenRouter API key (secondary/fallback LLM sentiment and commentary, available at [openrouter.ai](https://openrouter.ai))

### Quickstart

```bash
# Clone repository
git clone https://github.com/ifieryarrows/terra-rara.git
cd terra-rara

# Start backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../env.example .env
# Edit .env with your credentials
uvicorn app.main:app --reload --port 8000

# In another terminal: start frontend
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`, backend at `http://localhost:8000`.

## Installation

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

### Docker (Alternative)

```bash
docker-compose up --build
```

## Configuration

Copy `env.example` to `backend/.env` and configure core credentials (`DATABASE_URL`, `PIPELINE_TRIGGER_SECRET`, `HF_TOKEN`, `IFM_API_KEY` / `OPENROUTER_API_KEY`):

<details>
<summary><strong>Full Environment Variables Reference (Expand)</strong></summary>

| Variable                              | Required | Default                                 | Description                                                    |
| ------------------------------------- | -------- | --------------------------------------- | -------------------------------------------------------------- |
| `DATABASE_URL`                      | Yes      | -                                       | PostgreSQL connection string                                   |
| `IFM_API_KEY`                       | No       | -                                       | Primary IFM API key for K2-Horizon-375B commentary & dialogue   |
| `IFM_MODEL`                         | No       | `IFM/K2-Horizon-375B-A23B`              | Primary reasoning LLM model identifier                         |
| `IFM_BASE_URL`                      | No       | `https://api.ifm.ai/v1`                 | IFM API base endpoint                                          |
| `IFM_REASONING_EFFORT`              | No       | `low`                                   | Fast reasoning budget for structured output                    |
| `OPENROUTER_API_KEY`                | Yes      | -                                       | OpenRouter API key for LLM sentiment and commentary fallback   |
| `OPENROUTER_API_KEY_BACKUP`         | No       | -                                       | Optional second key; tried after primary HTTP 429/402            |
| `PIPELINE_TRIGGER_SECRET`           | Yes      | -                                       | Secret token for POST /api/pipeline/trigger (32+ random chars) |
| `REDIS_URL`                          | No       | `redis://localhost:6379/0`              | Redis connection string for pipeline task queue                 |
| `SYMBOL_SET`                        | No       | `active`                              | Which symbol set to use (`active`, `champion`, `challenger`)   |
| `OPENROUTER_MODEL_SCORING_FAST`     | No       | `nvidia/nemotron-3.5-super-vl:free`      | Primary fast sentiment scoring role                            |
| `OPENROUTER_MODEL_SCORING_RELIABLE` | No       | `google/gemma-4-31b-it:free`             | Reliable scoring & escalation repair role                      |
| `OPENROUTER_MODEL_COMMENTARY`       | No       | `google/gemma-4-31b-it:free`             | Commentary generation role                                     |
| `OPENROUTER_RPM`                    | No       | `12`                                    | Soft throttle target for OpenRouter calls                      |
| `OPENROUTER_MAX_RETRIES`            | No       | `1`                                     | One bounded retry for 408/429/5xx/network errors               |
| `OPENROUTER_CHAIN_DEADLINE_SECONDS` | No       | `120`                                   | Total sentiment batch deadline                                 |
| `MAX_LLM_ARTICLES_PER_RUN`          | No       | `100`                                   | Per-run LLM scoring budget                                     |
| `OPENROUTER_FALLBACK_MODELS`        | No       | `google/gemma-4-31b-it:free,xingchen-agi/xing4.0-29b-a4b:free,google/gemma-4-26b-a4b-it:free` | Client-controlled fallback model chain |
| `XGB_ARTIFACT_SOURCE`               | No       | `auto`                                | Atomic DB artifact first, verified paired files during rollout |
| `VITE_LOGO_DEV_PUBLISHABLE_KEY`     | No       | -                                       | Optional publishable key for Logo.dev company icons in heatmap |
| `TWELVEDATA_API_KEY`                | No       | -                                       | Backup live price source                                       |
| `SCHEDULER_ENABLED`                 | No       | `false`                               | Deprecated in API (pipeline triggered via GitHub Actions cron) |
| `SCHEDULE_TIME`                     | No       | `02:00`                               | Deprecated in API (scheduled via CI runner)                    |
| `TZ`                                | No       | `Europe/Istanbul`                     | Scheduler timezone                                             |
| `YFINANCE_SYMBOLS`                  | No       | (14 symbols)                            | Dashboard symbols (comma-separated)                            |
| `NEWS_LOOKBACK_DAYS`                | No       | `30`                                  | Days of news to fetch                                          |
| `SENTIMENT_DECAY_HALF_LIFE`         | No       | `7.0`                                 | Sentiment decay half-life (days)                               |
| `HF_TOKEN`                          | No       | -                                       | HuggingFace token for TFT model upload/download                |
| `NASDAQ_DATA_LINK_API_KEY`          | No       | -                                       | Nasdaq Data Link API key for LME warehouse data                |
| `MODEL_DIR`                         | No       | `/data/models`                        | Base directory for model artifacts (XGBoost + TFT)             |
| `HF_HUB_DISABLE_PROGRESS_BARS`      | No       | `1`                                   | Disable Hugging Face progress bars                             |
| `TRANSFORMERS_VERBOSITY`            | No       | `error`                               | Reduce Transformers log noise                                  |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | No       | `1`                                   | Disable advisory warnings from Transformers                    |

</details>

### AI Commentary & Companion Routing Hierarchy

Market commentary and daily Cuprum companion dialogue follow a resilient, fail-open routing hierarchy:
1. **IFM K2-Horizon-375B-A23B (Primary):** State-of-the-art 375B parameter reasoning model providing deep institutional commodity commentary and in-character Cuprum dialogue slots (`greeting`, `drivers_reply`, `outlook_reply`).
2. **OpenRouter (Secondary Fallback):** Rotates through configured OpenRouter models if `IFM_API_KEY` is omitted or encounters rate limits.
3. **Deterministic Fallback (Tertiary):** Guaranteed rule-based template generation if all upstream LLMs are unavailable, ensuring pipeline continuity without failing.

OpenRouter's free-model daily allowance applies across free models on an account, not separately per model. Fallback and failed requests can also consume that allowance.

For local runs, set `OPENROUTER_API_KEY_BACKUP` beside `OPENROUTER_API_KEY` in `backend/.env`. For the Hugging Face Space, add it as a Space Secret with that exact name; the primary key remains `OPENROUTER_API_KEY`. The worker tries the backup key after a primary HTTP 429 or 402, and skips a key with an explicitly exhausted free-tier daily quota for the rest of that request. It does not log either key value.

The `env.example` file includes `PIPELINE_TRIGGER_SECRET=` with no value. Generate a random secret before deploying.

## Usage

### Workspace & Dashboard

Access the live dashboard at [terra-rara.vercel.app](https://terra-rara.vercel.app):

1. **Cinematic Landing Page (`/`)**: Real-time copper market signal ticker, hero statistics and interactive platform narrative
2. **Market Overview (`/dashboard`)**:
   - **Consensus & Forecast Header**: Blended prediction, latest price quote, AI market stance and living Cuprum Companion avatar
   - **Cuprum Companion & Dialogue Tree**: Interactive living pet reflecting market mood (Molten Bull, Patina Bear, Tempered Neutral), speaking daily RAG-grounded dialogue explaining top drivers in plain human language without technical lag jargon
   - **Price Forecast Chart**: 180 days of historical prices with T+1 prediction dot and Q10-Q90 quantile confidence bands
   - **Market Heatmap Treemap**: S&P/Finviz-style interactive treemap across miners, metals ETFs and commodities with company logos, sparklines and news context
   - **Market Grid**: Real-time quotes for all 14 dashboard symbols
3. **News Intelligence Drawer**: Slide-over drawer providing commodity relevance scores, sentiment polarity and LLM reasoning for ingested articles
4. **Model Intelligence (`/models`)**: XGBoost next-day point estimates, top feature importances and TFT-ASRO multi-horizon quantile forecast paths
5. **Walk-Forward Validation (`/validation`)**: Backtest metrics, empirical coverage tests and deployment quality gate audits
6. **System Status (`/system`)**: Pipeline run history, database table statistics and API latency monitoring

### API Endpoints

Interactive API documentation: [ifieryarrows-copper-mind.hf.space/api/docs](https://ifieryarrows-copper-mind.hf.space/api/docs)

```bash
# Get current XGBoost prediction
curl https://ifieryarrows-copper-mind.hf.space/api/analysis

# Get TFT-ASRO deep learning forecast
curl https://ifieryarrows-copper-mind.hf.space/api/analysis/tft/HG=F

# Get stored TFT-ASRO model-focus explanation
curl https://ifieryarrows-copper-mind.hf.space/api/explain/HG=F

# Get AI commentary
curl https://ifieryarrows-copper-mind.hf.space/api/commentary

# Get historical data
curl https://ifieryarrows-copper-mind.hf.space/api/history

# Get market heatmap
curl https://ifieryarrows-copper-mind.hf.space/api/market-heatmap

# Get news intelligence
curl https://ifieryarrows-copper-mind.hf.space/api/news?limit=20

# Trigger pipeline manually (requires authentication)
curl -X POST "https://ifieryarrows-copper-mind.hf.space/api/pipeline/trigger?train_model=false&trigger_source=manual" \
  -H "Authorization: Bearer YOUR_PIPELINE_TRIGGER_SECRET"
```

## API Reference

### GET /api/analysis

Returns current XGBoost prediction with model metrics and feature importance.

```json
{
  "symbol": "HG=F",
  "current_price": 4.2500,
  "baseline_price": 4.2350,
  "baseline_price_date": "2026-01-25",
  "predicted_return": 0.001500,
  "predicted_return_pct": 0.15,
  "predicted_price": 4.2414,
  "target_type": "simple_return",
  "price_source": "yfinance_db_close",
  "confidence_lower": 4.1800,
  "confidence_upper": 4.3000,
  "sentiment_index": 0.227,
  "sentiment_label": "Bullish",
  "top_influencers": [
    {"feature": "HG=F_vol_10", "importance": 0.1808, "description": "10-day volatility"},
    {"feature": "FXI_lag_ret1_2", "importance": 0.1019, "description": "China ETF 2-day lagged return"}
  ],
  "data_quality": {
    "coverage_pct": 98.5,
    "missing_features": []
  },
  "training_symbols_hash": "sha256:7b7dd017b79da296",
  "generated_at": "2026-01-17T09:00:00Z"
}
```

### GET /api/analysis/consensus

Returns the blended model consensus forecast combining XGBoost and TFT-ASRO outputs.

```json
{
  "consensus_direction": "BULLISH",
  "confidence": "HIGH",
  "position_scale": 1.0,
  "blended_return": 0.0028,
  "xgb_return_raw": 0.0015,
  "xgb_return_adjusted": 0.0005,
  "tft_return": 0.0035,
  "xgb_direction": 1,
  "tft_direction": 1
}
```

### GET /api/market-heatmap

Returns the S&P/Finviz-style hierarchical market map with performance, sparklines and market cap weightings. Cached for 15 minutes.

### GET /api/news

Returns processed commodity market news items with relevance scores, sentiment polarity, event tags and LLM reasoning.

### GET /api/models/tft/summary

Returns active TFT-ASRO model parameters, deployment Quality Gate audit status (`quality_gate_passed: true`) and detailed backtest evaluation metrics. Only the quality-gate-passed metadata row is ever read — a rejected candidate can never surface here. `variable_importance` and `variable_importance_method` report where the entry list came from: `vsn_encoder` (TFT attention-derived weights stored by the trainer) or `mrmr` (MRMR scores recomputed from the training-analysis artifacts), or `null` with an empty list when neither source is available.

### GET /api/commentary

Returns AI-generated market analysis with stance classification, section breakdown, and dynamic Cuprum companion dialogue slots.

```json
{
  "symbol": "HG=F",
  "commentary": "Copper futures show bullish momentum supported by tightening inventory...\nThis is NOT financial advice.",
  "ai_stance": "BULLISH",
  "generation_mode": "ifm_k2",
  "model_name": "IFM/K2-Horizon-375B-A23B",
  "generated_at": "2026-10-29T23:00:00Z",
  "sections": {
    "synthesis": "Copper futures show bullish momentum...",
    "risks": [
      "Dollar index strengthening above resistance",
      "Smelter treatment charges at historical lows"
    ],
    "opportunities": [
      "Physical stock drawdowns in LME warehouses"
    ],
    "bias_warning": null,
    "cuprum_daily": {
      "mood": "molten_bull",
      "greeting": "The forge is roaring with fresh dollar and china currents.",
      "drivers_reply": "The Dollar and China Large-Cap are pulling the hardest on my wires today.",
      "outlook_reply": "The trend has momentum, but keep an eye on warehouse inventories."
    }
  },
  "companion_state": {
    "character": "cuprum",
    "mood": "molten_bull",
    "stance": "BULLISH",
    "energy": "charged",
    "confidence_band": "high",
    "headline_quote": "The forge is roaring with fresh dollar and china currents.",
    "xai_highlights": {
      "horizon_label": "5-day cumulative log return (primary)",
      "top_signals": [
        {
          "feature": "DX-Y.NYB_ret1",
          "weight": 0.28,
          "label": "US Dollar Index",
          "category": "Macro"
        }
      ]
    },
    "cuprum_daily": {
      "mood": "molten_bull",
      "greeting": "The forge is roaring with fresh dollar and china currents.",
      "drivers_reply": "The Dollar and China Large-Cap are pulling the hardest on my wires today.",
      "outlook_reply": "The trend has momentum, but keep an eye on warehouse inventories."
    }
  }
}
```

### GET /api/market-prices

Returns real-time quotes for all dashboard symbols.

```json
{
  "symbols": {
    "HG=F": {"price": 4.25, "change": 1.23},
    "BHP": {"price": 45.67, "change": -0.45}
  },
  "updated_at": "2026-01-17T15:30:00Z"
}
```

### GET /api/history

Returns 180 days of historical price and sentiment data.

### GET /api/health

Returns health status including database connectivity.

### GET /api/analysis/tft/{symbol}

Returns TFT-ASRO deep learning forecast with 7-quantile prediction paths and confidence intervals. Results are cached for 5 minutes.

```json
{
  "symbol": "HG=F",
  "direction": "BULLISH",
  "risk_level": "MEDIUM",
  "weekly_trend": "BULLISH",
  "prediction": {
    "predicted_return_median": 0.0035,
    "predicted_return_q10": -0.0080,
    "predicted_return_q90": 0.0150,
    "predicted_price_median": 4.2650,
    "volatility_estimate": 0.0115,
    "daily_forecasts": [
      {"day": 1, "return_median": 0.0035, "return_q10": -0.0080, "return_q90": 0.0150}
    ]
  },
  "model_metadata": {
    "trained_at": "2026-03-10T02:30:00Z",
    "metrics": {
      "directional_accuracy": 0.58,
      "sharpe_ratio": 1.12,
      "variance_ratio": 0.85,
      "tail_capture_rate": 0.62
    }
  }
}
```

### GET /api/explain/{symbol}

Returns the per-forecast TFT-ASRO model-focus explanation stored with the latest TFT snapshot (`explain_json`, `schema_version: 1`). It is computed once by the pipeline worker at Stage 5.5. This endpoint is snapshot-only: unlike `/api/analysis/tft/{symbol}`, it never runs live inference.

The payload holds the raw VSN variable-selection weights split into encoder, decoder and static groups (encoder entries sorted by weight and capped at the top 25, each with a label and category; the top five encoder features may also carry a `recent` block with the last value and its change over the observed span — the `basis` label counts the sessions actually spanned, which is fewer than 5 when missing values shorten the window), the TFT attention distribution over the encoder window at the forecast day recorded in `attention.step.horizon_index` (pipeline default: t+1), the reference price date, the model checkpoint identity (`checkpoint`, `trained_at`), encoder/decoder/static feature-layout hashes and a `notes` disclaimer.

Returns `404` when:

- No snapshot exists for the symbol yet.
- The snapshot is stale under the same freshness rules as `/api/analysis/tft/{symbol}` (a quality-gate-passed model trained after the snapshot, or a missing/mismatched reference price date relative to the latest PriceBar). Where `/api/analysis/tft/{symbol}` falls back to live inference, this endpoint returns `404`.
- The snapshot carries no explanation yet; retry after the next pipeline refresh.

These are raw, pre-calibration model-focus signals for the 5-day cumulative log-return (primary) horizon, **not causality**. Explanations are optional and fail-open: if the payload cannot be built, the forecast and its snapshot are still persisted without it. The Overview dashboard surfaces this payload in the "Model focus (TFT-ASRO)" card.

### POST /api/pipeline/trigger (Privileged)

Enqueues an asynchronous ML pipeline job to the Redis queue.

**Authentication requirement**: This endpoint requires a valid `Authorization: Bearer <PIPELINE_TRIGGER_SECRET>` header. Requests without a valid token receive 401 Unauthorized.

**Expected responses**:

| Status | Condition |
| ------ | --------- |
| 200 | Pipeline job enqueued successfully (`{"run_id": "...", "enqueued": true}`) |
| 401 Unauthorized | Missing or invalid `Authorization` header |
| 409 Conflict | Pipeline already running (advisory lock engaged) |
| 503 Service Unavailable | Redis queue unavailable |

**Parameters**:

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `train_model` | boolean | `false` | Train/retrain XGBoost model (`true` for Sunday retrain, `false` for daily sentiment refresh) |
| `trigger_source` | string | `api` | Source of trigger (`api`, `cron`, `manual`, `github-actions`) |

**Job Tracking**:

Clients can poll the asynchronous pipeline execution status using:
`GET /api/pipeline/runs/{run_id}`

## Development

### Running Tests

```bash
cd backend

# All tests
pytest tests/ -v

# Deep learning tests only (36 test suites)
pytest tests/deep_learning/ -v

# Screener tests only (9 test suites)
pytest tests/screener/ -v
```

### Linting

```bash
# Frontend
cd frontend
npm run lint

# Backend (if ruff is installed)
cd backend
ruff check .
```

### Daily Pipeline

The pipeline is triggered daily by GitHub Actions cron (`.github/workflows/daily-pipeline.yml`) at 00:10 UTC (03:10 Istanbul), right after OpenRouter's 00:00 UTC daily quota reset and after Sunday's 21:00 UTC weekly TFT-ASRO training workflow completes. Local scheduler mode (`SCHEDULER_ENABLED=true`) is deprecated and reserved for standalone offline development:

1. Fetch news from 8 strategic Google News RSS queries
2. Process news through reproducible pipeline (`news_raw` → `news_processed`)
3. Fetch market price bars for 17 training symbols via yfinance
4. Score sentiment via multi-tier LLM routing in chunks of 12 articles with fallback to FinBERT
5. Compute FinBERT CLS embeddings and store 8-dimensional PCA-reduced vectors
6. Aggregate daily sentiment with time-weighted decay (half-life: 7 days)
7. Generate XGBoost prediction using active model (or retrain if `train_model=true` on Sundays)
8. Generate 5-day TFT-ASRO + Chronos-2 inference from HF Hub checkpoint (quantile forecasts)
9. Generate AI commentary and daily Cuprum companion dialogue via IFM K2-Horizon (with OpenRouter fallback)
10. Cache prediction snapshots in database
11. Record run duration, fallback counts and `quality_state` to `pipeline_run_metrics` table

TFT deep learning model retraining runs on a separate weekly workflow (Sundays at 21:00 UTC) with hardened Optuna hyperparameter optimization (`mrmr_top_k=5`, `lambda_drift_prior >= 0.75`) and deployment Quality Gate validation.

### Pipeline Monitoring

Each pipeline run records detailed execution telemetry to the database:

```sql
SELECT run_id, run_started_at, duration_seconds, 
       symbols_failed, status, quality_state,
       news_raw_inserted, articles_scored_v2,
       llm_success_count, fallback_count
FROM pipeline_run_metrics 
ORDER BY run_started_at DESC 
LIMIT 10;
```

Tracked metrics:

- `duration_seconds`: Total pipeline execution time
- `trigger_source` / `train_model_requested`: Job initiation context
- `symbols_requested` / `symbols_fetched_ok` / `symbols_failed`: Market data fetch telemetry
- `news_raw_inserted` / `news_processed_inserted` / `articles_scored_v2`: News pipeline throughput
- `llm_success_count` / `operational_fallback_count` / `policy_fallback_count`: LLM router health
- `tft_embeddings_computed`: FinBERT embeddings generated
- `quality_state`: Health designation (`ok`, `stale`, `degraded`, `failed`)
- `status`: Execution status (`queued`, `running`, `success`, `failed`)
- `error_message`: Error diagnostics if failed

## Troubleshooting

### Database connection refused / Driver mismatch

**Symptom**: `psycopg2.OperationalError: could not connect to server`, `ModuleNotFoundError: No module named 'psycopg2'` or `DuplicatePreparedStatement` on Supabase / PgBouncer.

**Cause**:
1. PostgreSQL is not running or `DATABASE_URL` is incorrect.
2. SQLAlchemy 2.1+ default dialect shifts or missing driver packages.
3. Transaction-mode connection poolers (e.g. Supabase port 6543, PgBouncer) colliding with prepared statements.

**Fix**:
1. Verify PostgreSQL is running. Check `DATABASE_URL` format: `postgresql://user:password@host:port/database`.
2. The system dynamically normalizes unspecified `postgresql://` URIs to the installed driver (`postgresql+psycopg://` for psycopg v3 or `postgresql+psycopg2://` for psycopg2). Ensure `psycopg[binary]` or `psycopg2-binary` is installed.
3. For Supabase / PgBouncer transaction-mode poolers (`:6543`, `pooler.supabase.com`), prepared statements are automatically disabled via `connect_args={"prepare_threshold": None}` to prevent statement collisions. If configuring custom engines, ensure `prepare_threshold: None` is set.

### LLM sentiment scoring fails, falling back to FinBERT

**Symptom**: Logs show `LLM sentiment failed, using FinBERT fallback`

**Cause**: OpenRouter API key is missing, invalid or rate-limited.

**Fix**: Verify `OPENROUTER_API_KEY` is set correctly in `.env`. Check OpenRouter dashboard for rate limit status. The system will continue using FinBERT as a fallback.

### Frontend cannot connect to backend

**Symptom**: Network errors in browser console, data not loading.

**Cause**: Backend is not running or CORS is blocking requests.

**Fix**: Ensure backend is running at the expected URL. For local development, verify `http://localhost:8000` is accessible. Check that the frontend API base URL matches the backend.

### Pipeline runs but predictions are stale

**Symptom**: `generated_at` timestamp in `/api/analysis` does not update.

**Cause**: Pipeline failed silently or scheduler is disabled.

**Fix**: Check backend logs for pipeline errors and queue health. Manually trigger pipeline via `POST /api/pipeline/trigger` to test execution end-to-end.

### Invalid or missing target_type error

**Symptom**: `/api/analysis` returns null, logs show `Invalid or missing target_type in model metadata`

**Cause**: Model was trained before `target_type` field was added.

**Fix**: Retrain the model via `POST /api/pipeline/trigger?train_model=true`. New models include `target_type: "simple_return"` in metadata.

### Feature mismatch error during prediction

**Symptom**: `ValueError: feature_names mismatch` in logs

**Cause**: Inference is using different symbols than training.

**Fix**: Ensure both training and inference use the same symbol set. The system automatically aligns features via `reindex(columns=expected, fill_value=0)`.

### TFT model unavailable or returns null

**Symptom**: `/api/analysis/tft/HG=F` returns null or `{"status": "unavailable"}`

**Cause**: No trained TFT checkpoint exists locally or on HuggingFace Hub.

**Fix**: Train the TFT model via `python -m deep_learning.training.trainer --symbol HG=F`. Ensure `HF_TOKEN` is set if using HF Hub persistence. The predictor will automatically download the checkpoint from Hub on first inference if available.

### TFT predictions are flat or near-zero

**Symptom**: All quantile medians cluster near zero (`|pred| < 0.05%`) or collapse to a single direction.

**Cause**: Over-aggressive drift regularization (`lambda_drift_prior` collapsed or overly tight head LR) or uncalibrated Optuna parameters fitting noise. Note that 5-day copper $R^2$ is naturally ~1–3%, so Bayes-optimal medians sit around `0.15%–0.40%` while uncertainty is captured by Q10–Q90 bands.

**Fix**: Use the modern `KNOWN_GOOD_CONFIG` (`output_head_lr_mult=0.35`, `lambda_drift_prior=0.75`, `lambda_directional=0.12`, `mrmr_top_k=5`) with the fail-open 20% `Chronos-2` prior. Avoid artificial multiplicative post-hoc scaling, which worsens `MAE/0` and Pinball skill.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests if applicable
4. Run linting and tests locally before committing
5. Commit with descriptive messages: `git commit -m "Add feature X"`
6. Push to your fork: `git push origin feature/your-feature`
7. Open a Pull Request with a clear description of changes

Issues and feature requests are welcome via GitHub Issues.

## Security

### Credential Masking

The system masks sensitive credentials in logs:

- Database passwords are replaced with `***:***` in connection URLs
- API keys in httpx request logs are suppressed
- Error messages containing credentials are sanitized

### Privileged Endpoint: POST /api/pipeline/trigger

This endpoint triggers the full ML pipeline, which fetches news, calls the LLM API for sentiment scoring, computes FinBERT embeddings, retrains the XGBoost model, runs TFT-ASRO inference and generates AI commentary. Unauthenticated access to this endpoint creates the following risks:

- **Request flooding**: Repeated triggers can degrade service availability.
- **Quota and cost burn**: Each pipeline run consumes OpenRouter API quota. Uncontrolled access can exhaust free-tier limits or incur costs.
- **Resource exhaustion**: Model training and batch LLM calls are CPU and memory intensive.

**Authentication**: This endpoint requires a valid `Authorization: Bearer <PIPELINE_TRIGGER_SECRET>` header. Requests without a valid token receive 401 Unauthorized.

### Configuration

1. **Set `PIPELINE_TRIGGER_SECRET`** in your `.env` file. Use a random string of 32 or more characters. Store this as a secret in your deployment platform.
2. **Rotate the secret immediately** if it is ever exposed in logs, commits or third-party systems.

### Abuse Prevention (Recommended Controls)

Even with authentication, apply additional safeguards:

- **Rate limiting**: Limit to 5-10 requests per minute per IP or per token. Use a reverse proxy (nginx, Caddy) or middleware.
- **Idempotency lock**: The endpoint already checks if the pipeline is running and returns 409 Conflict. Ensure this lock file mechanism is reliable across restarts.
- **Network restriction**: If the endpoint is only used by internal automation (CI/CD, cron jobs), restrict access via IP allowlist, VPN or deploy the backend on an internal network.
- **Monitoring and alerting**: Log all calls to this endpoint. Alert on spikes in 401 responses, sudden trigger volume or upstream API quota warnings.

### Secret Management

- Do not commit secrets to version control.
- Use your deployment platform's secret management (HuggingFace Spaces secrets, Vercel environment variables, etc.).
- The `env.example` file should include `PIPELINE_TRIGGER_SECRET=` with no value as a placeholder.

## License

MIT License - see [LICENSE](LICENSE) for details.
