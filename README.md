# Terra Rara

AI-powered copper futures price prediction platform combining XGBoost ML, TFT-ASRO deep learning, LLM-based sentiment analysis and cross-asset market data.

[![Live Demo](https://img.shields.io/badge/demo-terra--rara.vercel.app-0969da)](https://terra-rara.vercel.app)
[![API Docs](https://img.shields.io/badge/api-docs-10b981)](https://ifieryarrows-copper-mind.hf.space/api/docs)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Symbol Sets](#symbol-sets)
- [Model Details](#model-details)
  - [XGBoost](#xgboost-parameters)
  - [TFT-ASRO Deep Learning](#tft-asro-deep-learning)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

## Overview

Terra Rara predicts COMEX copper futures (HG=F) **returns** using two parallel models: an XGBoost regression model for next-day point estimates and a Temporal Fusion Transformer with Adaptive Sharpe Ratio Optimization (TFT-ASRO) for multi-day probabilistic forecasts. The system ingests daily news via Google News RSS, scores article sentiment using a multi-tier resilient LLM routing system (NVIDIA Nemotron 3.5 Super VL as primary, Liquid LFM 2.5 as reliable scoring/repair, with Gemma 4 fallbacks and local FinBERT fallback), extracts 768-dimensional FinBERT embeddings reduced to 8 dimensions via PCA and combines these with 250+ technical features computed from 17 correlated assets. LME warehouse stock data and futures curve signals provide physical-market context. A scheduled pipeline runs daily to refresh sentiment, generate AI-driven institutional market commentary and produce dynamic Cuprum companion dialogue powered primarily by IFM K2-Horizon (375B parameter reasoning model) with fail-open fallback to OpenRouter.

**XGBoost target**: Next-day simple return: `(close[t+1] / close[t]) - 1`

**TFT-ASRO target**: 5-day cumulative log-return (`target_5d_log_return`) primary forecast with 7-quantile probabilistic output (q0.02 through q0.98), accompanied by 1-day log-return auxiliary diagnostic.

**Target users**: Traders, analysts and developers building commodity forecasting tools.

**Non-goals**: This system does not provide trading signals or financial advice. Predictions are for informational purposes only.

## Features

- Predict next-day copper futures returns using XGBoost regression trained on 250+ features
- Forecast 5-day returns with probabilistic confidence bands using TFT-ASRO deep learning
- Score news sentiment using multi-tier LLM routing (NVIDIA Nemotron 3.5 Super VL / Liquid LFM 2.5) with FinBERT fallback
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

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (Vercel)                                                      │
│  React 18 + Vite 7 + TailwindCSS v4 + React Router 7 + Framer Motion    │
│  ├── Multi-page workspace: Landing (/), Overview (/dashboard),          │
│  │   Models (/models), Validation (/validation), System (/system)       │
│  ├── Bespoke Price Forecast & quantile bands chart (Recharts)           │
│  ├── S&P / Finviz-style interactive treemap Heatmap (D3-hierarchy)     │
│  ├── News Intelligence drawer with sentiment & commodity relevance      │
│  ├── Blended Model Consensus & XGBoost feature importance               │
│  └── Model Reliability & Walk-Forward Validation reporting              │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BACKEND (HuggingFace Spaces - Docker)                                  │
│  FastAPI + Python 3.11 + Uvicorn + Redis                                │
│                                                                         │
│  REST API                                                               │
│  ├── GET /api/analysis          → XGBoost next-day return prediction    │
│  ├── GET /api/analysis/tft/{s}  → TFT-ASRO 5-day quantile forecast      │
│  ├── GET /api/analysis/consensus→ Blended prediction (XGBoost + TFT)    │
│  ├── GET /api/explain/{s}       → Stored TFT model-focus explanation    │
│  ├── GET /api/history           → 180d historical price & sentiment     │
│  ├── GET /api/market-heatmap    → Sector/category asset treemap (15m)   │
│  ├── GET /api/market-prices     → Real-time quotes (14 symbols)         │
│  ├── GET /api/commentary        → AI commentary, stance & Cuprum dialog │
│  ├── GET /api/news              → Processed news intelligence feed      │
│  ├── GET /api/models/tft/summary→ Quality gate status & TFT metrics     │
│  ├── GET /api/health            → System & database health              │
│  ├── POST /api/pipeline/trigger → Enqueue background pipeline job       │
│  └── GET /api/pipeline/runs/{id}→ Pipeline run status tracking          │
│                                                                         │
│  ML PIPELINE (Daily @ 23:00 UTC / 02:00 Istanbul)                       │
│  ├── 8 strategic queries → Google News RSS ingestion                    │
│  ├── Resilient multi-tier LLM sentiment scoring (batch size 12)         │
│  ├── FinBERT CLS embedding extraction + PCA (768→8 dim)                 │
│  ├── 250+ feature engineering across 17 training symbols                │
│  ├── XGBoost training (scheduled weekly on Sundays)                     │
│  ├── TFT-ASRO inference (quantile forecasts from HF Hub checkpoint)    │
│  ├── AI commentary & Cuprum dialogue via IFM K2 (OpenRouter fallback)  │
│  └── Pipeline metrics & quality state recorded in database              │
│                                                                         │
│  DEEP LEARNING (Scheduled Retraining & Validation)                      │
│  ├── Feature store: price + sentiment + embeddings + LME + calendar    │
│  ├── TFT-ASRO model (pytorch-forecasting + Lightning)                  │
│  ├── ASRO & Weekly Loss: Sharpe + quantile + MADL + crossing penalties │
│  ├── Optuna hyperparameter search with variance ratio penalty          │
│  ├── Strict Quality Gate: Weekly DA, magnitude, Sharpe & coverage audit│
│  └── Checkpoint persistence via HuggingFace Hub + atomic DB artifacts  │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                             │
│  ├── Supabase PostgreSQL                                                │
│  │   ├── news_raw / news_processed  (reproducible news pipeline)        │
│  │   ├── news_sentiments_v2         (commodity-aware LLM scores)        │
│  │   ├── news_embeddings            (FinBERT CLS + 8-dim PCA vectors)   │
│  │   ├── daily_sentiments_v2        (aggregated index)                  │
│  │   ├── price_bars                 (OHLCV market bars)                 │
│  │   ├── lme_warehouse_data         (LME copper stock levels)           │
│  │   ├── analysis_snapshots         (cached XGBoost predictions)        │
│  │   ├── tft_prediction_snapshots   (cached TFT quantile forecasts)     │
│  │   ├── ai_commentaries            (market commentaries)               │
│  │   ├── model_metadata             (XGBoost artifacts & params)        │
│  │   ├── model_artifacts            (serialized model weights)          │
│  │   ├── tft_model_metadata         (TFT-ASRO metrics & quality gate)   │
│  │   ├── heatmap_cache              (market heatmap snapshots)          │
│  │   ├── backtest_reports           (walk-forward validation runs)      │
│  │   └── pipeline_run_metrics       (pipeline executions & monitoring)  │
│  ├── Redis (pipeline job queue)                                         │
│  ├── HuggingFace Hub (TFT checkpoint, calibration & PCA persistence)   │
│  ├── yfinance (OHLCV price data & live quotes)                          │
│  ├── Nasdaq Data Link (LME warehouse stocks, optional)                  │
│  ├── TwelveData (backup live copper price)                              │
│  └── Google News RSS (news source)                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

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

The TFT-ASRO (Temporal Fusion Transformer with Adaptive Sharpe Ratio Optimization) operates as a parallel model producing 5-day probabilistic forecasts. It runs independently from the XGBoost pipeline and uses a richer feature set.

#### Model Architecture

| Parameter                  | Value | Notes                                                      |
| -------------------------- | ----- | ---------------------------------------------------------- |
| `hidden_size`            | 48    | Optimal size post-MRMR pruning; prevents VSN overfitting   |
| `attention_head_size`    | 2     | Fewer heads for small, single-series dataset               |
| `dropout`                | 0.30  | Regularization for small sample regime without collapse    |
| `hidden_continuous_size` | 16    | Paired reduction with hidden_size                          |
| `learning_rate`          | 2e-4  | Conservative AdamW learning rate                           |
| `weight_decay`           | 5e-5  | L2 regularization preventing weight explosion              |
| `max_encoder_length`     | 60    | 60 trading days of lookback                                |
| `max_prediction_length`  | 5     | 5-day forecast horizon                                     |
| `gradient_clip_val`      | 1.0   | Relaxed from 0.5; tanh-based Sharpe gradients are bounded  |
| `batch_size`             | 16    | ~19 batches/epoch for stable gradient estimates            |
| `lookback_days`          | 1095  | 3 years of market data (~500 samples vs ~313 previously)   |

#### Quantile Output & Monotonic Ordering

The model produces 7 quantiles instead of a single point estimate:

- q0.02, q0.10, q0.25, **q0.50 (median)**, q0.75, q0.90, q0.98

Quantile crossings are strictly penalized during training (`lambda_crossing = 1.0`) and non-crossing order is mathematically guaranteed at inference via monotonic projection (`enforce_monotonic_quantiles`). This directly models 96% and 80% confidence intervals and captures tail risk.

#### ASRO & Weekly Loss Formulation

The custom loss aligns optimization directly with financial risk-adjusted performance and scale control:

```
L_total = λ_quantile × L_calibration + (1 - λ_quantile) × (-tanh_sharpe) + L_auxiliary
```

| Component | Weight | Purpose |
| --------- | ------ | ------- |
| Quantile calibration | `λ_quantile = 0.25` | Keeps TFT probabilistic |
| Sharpe component | `w_sharpe = 0.75` | Drives directional risk-adjusted learning |
| Volatility calibration | `λ_vol = 0.30` | Q90-Q10 spread tracks actual return volatility |
| Mean Absolute Directional Loss (MADL) | `λ_madl = 0.40` | Rewards correct direction scaled by magnitude (Kisiel & Gorse) |
| Quantile crossing guard | `λ_crossing = 1.0` | Structural penalty enforcing monotonic quantile order |
| Weekly quantile loss | `λ_weekly_quantile = 0.70` | Aligns loss with 5-day cumulative horizon |
| Weekly directional weight | `λ_directional = 0.25` | Prioritizes multi-day direction over noise |
| Weekly magnitude penalty | `λ_magnitude = 0.58` | Penalizes scale mismatches against historical targets |
| Positive rate sign loss | `λ_positive_rate = 0.75` | Guards against single-direction prediction collapse |
| Weekly median cap multiple | `1.20×` train median | Structural median boundary derived from training split |

#### TFT Feature Categories

The TFT ingests ~340 features across three categories:

**Time-varying unknown (observed in past, unknown in future)**:

- Copper prices, log returns, technical indicators (from 17 training assets)
- FinBERT PCA embedding vectors (8 dimensions, reduced from 768 for sample efficiency)
- Sentiment momentum (5/10/30-day SMA/EMA), surprise Z-score, volume-weighted sentiment
- 11 commodity event-type intensities (supply disruption, inventory draw, macro USD, etc.)
- LME warehouse stock changes (1/5/10/20-day), depletion rate, cancelled warrant ratio
- Futures curve spread, contango/backwardation flags
- Regime indicators (volatility regimes, rolling return percentiles)

**Time-varying known (known in future)**:

- Day of week (one-hot), month (sinusoidal encoding)
- US/China holiday calendars and market schedule boundaries

**Static**:

- Group identifier (single series: "copper")

#### Training and Optimization

TFT training is scheduled weekly (Sundays at 22:00 UTC via GitHub Actions) or on-demand:

```bash
cd backend

# Train TFT-ASRO model with default parameters
python -m deep_learning.training.trainer --symbol HG=F

# Run deterministic weekly validation replay (bypass Optuna overlays)
python -m deep_learning.training.trainer --symbol HG=F --deterministic-weekly-validation

# Run Optuna hyperparameter search
python -m deep_learning.training.hyperopt --n-trials 50

# Backfill FinBERT embeddings for historical news
python -m deep_learning.data.embeddings --backfill --days 180 --pca-dim 8
```

#### Financial Metrics & Promotion Quality Gate

To prevent overfitting or prediction explosion, models are promoted to production **only** if they pass the strict deployment Quality Gate (`app/quality_gate.py`):

| Gate Metric | Threshold | Invariant Rationale |
| ----------- | --------- | ------------------- |
| **Weekly Directional Accuracy** | $\ge 0.53$ ($\ge 0.51$ if $N < 80$) | Statistically meaningful directional edge |
| **Weekly Magnitude Ratio** | $[0.65, 1.35]$ | Prevents muted or inflated multi-day return forecasts |
| **Weekly Raw Magnitude Ratio** | $\le 3.0$ (hard block) | Raw unconstrained neural net output must not explode; warning if $> 1.8$ |
| **Weekly Median Bound Applied Rate** | Warning if $> 0.30$ | Non-blocking alert if clipping applies to $> 30\%$ of predictions |
| **Weekly Tail Capture Rate** | $\ge 0.45$ | Captures outsized commodity moves ($>75^{\text{th}}$ percentile) |
| **Weekly PI80 Coverage** | $[0.74, 0.86]$ | Calibrated 80% empirical prediction interval |
| **Weekly Sharpe Ratio** | $\ge -0.20$ | Annualized with $\sqrt{52}$; warning if $< 0.20$ |
| **Daily Sharpe Ratio** | $\ge -0.30$ | Single-step diagnostic baseline, annualized with $\sqrt{252}$ |
| **Quantile Crossing Rate** | $\le 0.001$ | Zero tolerance for quantile order inversions |

#### Model Persistence

TFT artifacts are saved both locally in `$MODEL_DIR` and synchronized with HuggingFace Hub:

| Artifact | Description |
| -------- | ----------- |
| `best_tft_asro.ckpt` | PyTorch Lightning model checkpoint |
| `tft_metadata.json` | Training metrics, quality gate results, hyperparameters |
| `conformal_calibration.json` | Conformal prediction interval calibration scalars |
| `pca_finbert.joblib` | Fitted IncrementalPCA dimensionality reduction model |
| `optuna_results.json` | Best hyperparameter trial configurations |
| `artifact_manifest.json` | SHA-256 hashes and metadata verifying artifact integrity |

## Directory Structure

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
│   │   ├── models/
│   │   │   ├── tft_copper.py         # TFT model + ASROPFLoss
│   │   │   ├── losses.py             # ASRO + WeeklyLoss + MADL
│   │   │   ├── monotonic_quantiles.py# Non-crossing quantile projection
│   │   │   └── hub.py                # HuggingFace Hub upload/download
│   │   ├── training/
│   │   │   ├── trainer.py            # PyTorch Lightning training loop
│   │   │   ├── hyperopt.py           # Optuna TPE hyperparameter search
│   │   │   └── metrics.py            # Sharpe, Sortino, DA, tail capture
│   │   └── inference/
│   │       └── predictor.py          # TFT live prediction pipeline
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

Copy `env.example` to `backend/.env` and configure:

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

The pipeline is triggered daily by GitHub Actions cron (`.github/workflows/daily-pipeline.yml`) at 00:10 UTC (03:10 Istanbul), right after OpenRouter's 00:00 UTC daily quota reset and after Sunday's 22:00 UTC weekly TFT-ASRO training workflow completes. Local scheduler mode (`SCHEDULER_ENABLED=true`) is deprecated and reserved for standalone offline development:

1. Fetch news from 8 strategic Google News RSS queries
2. Process news through reproducible pipeline (`news_raw` → `news_processed`)
3. Fetch market price bars for 17 training symbols via yfinance
4. Score sentiment via multi-tier LLM routing in chunks of 12 articles with fallback to FinBERT
5. Compute FinBERT CLS embeddings and store 8-dimensional PCA-reduced vectors
6. Aggregate daily sentiment with time-weighted decay (half-life: 7 days)
7. Generate XGBoost prediction using active model (or retrain if `train_model=true` on Sundays)
8. Generate TFT-ASRO inference from HF Hub checkpoint (quantile forecasts)
9. Generate AI commentary and daily Cuprum companion dialogue via IFM K2-Horizon (with OpenRouter fallback)
10. Cache prediction snapshots in database
11. Record run duration, fallback counts and `quality_state` to `pipeline_run_metrics` table

TFT deep learning model retraining runs on a separate weekly workflow (Sundays at 22:00 UTC) with Optuna hyperparameter optimization and deployment Quality Gate validation.

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

**Symptom**: All quantile predictions cluster around zero with minimal spread.

**Cause**: Variance ratio collapse (VR < 0.5). The model learned to minimize loss by predicting near-constant values.

**Fix**: Run Optuna hyperparameter search with `python -m deep_learning.training.hyperopt --n-trials 50`. The search space includes variance ratio penalty to prevent this failure mode. Check that `lambda_vol >= 0.25` and `learning_rate <= 1e-3`.

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
