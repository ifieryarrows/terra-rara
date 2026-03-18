# Terra Rara

AI-powered copper futures price prediction platform combining XGBoost ML, TFT-ASRO deep learning, LLM-based sentiment analysis, and cross-asset market data.

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

Terra Rara predicts COMEX copper futures (HG=F) **returns** using two parallel models: an XGBoost regression model for next-day point estimates, and a Temporal Fusion Transformer with Adaptive Sharpe Ratio Optimization (TFT-ASRO) for multi-day probabilistic forecasts. The system ingests daily news via Google News RSS, scores article sentiment using an LLM (Arcee Trinity Large Preview via OpenRouter), extracts 768-dimensional FinBERT embeddings reduced to 32 dimensions via PCA, and combines these with 250+ technical features computed from 17 correlated assets. LME warehouse stock data and futures curve signals provide physical-market context. A scheduled pipeline runs daily to refresh sentiment and generate AI-driven market commentary.

**XGBoost target**: Next-day simple return: `(close[t+1] / close[t]) - 1`

**TFT-ASRO target**: 5-day daily return forecasts with 7-quantile probabilistic output (q0.02 through q0.98)

**Target users**: Traders, analysts, and developers building commodity forecasting tools.

**Non-goals**: This system does not provide trading signals or financial advice. Predictions are for informational purposes only.

## Features

- Predict next-day copper futures returns using XGBoost regression trained on 250+ features
- Forecast 5-day returns with probabilistic confidence bands using TFT-ASRO deep learning
- Score news sentiment using LLM (Arcee Trinity Large Preview) with FinBERT fallback when API is unavailable
- Extract FinBERT CLS embeddings (768-dim → 32-dim PCA) for deep semantic encoding
- Track 17 correlated assets via configurable symbol sets (active, champion, challenger)
- Aggregate daily sentiment using time-weighted exponential decay
- Compute advanced sentiment features: momentum, surprise (Z-score), volume-weighted scores
- Ingest LME warehouse stock data and futures curve signals for physical-market context
- Generate AI-powered market commentary with stance classification (BULLISH/NEUTRAL/BEARISH)
- Display real-time prices for dashboard symbols via yfinance
- Visualize historical price, sentiment, and TFT forecast bands over 180 days
- Optimize TFT hyperparameters via Optuna TPE search with variance ratio penalty
- Trigger manual pipeline execution via authenticated API endpoint
- Monitor pipeline health via `pipeline_run_metrics` table

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (Vercel)                                                      │
│  React 18 + TypeScript + Vite + TailwindCSS                             │
│  ├── TradingView widget (lazy loaded)                                   │
│  ├── Price & Sentiment chart with TFT forecast bands (Recharts)         │
│  ├── Deep Learning Forecast card (T+1 return, risk, weekly trend)       │
│  ├── Deep Learning Metrics card (DA, Sharpe, variance ratio)            │
│  ├── XGBoost Forecast card with AI Commentary                           │
│  ├── XGBoost feature importance display                                 │
│  └── Market grid showing 14 dashboard symbols                           │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BACKEND (HuggingFace Spaces - Docker)                                  │
│  FastAPI + Python 3.11 + Uvicorn + APScheduler                          │
│                                                                         │
│  REST API                                                               │
│  ├── GET /api/analysis          → XGBoost prediction + metrics          │
│  ├── GET /api/analysis/tft/{s}  → TFT-ASRO quantile forecast (5m TTL)  │
│  ├── GET /api/history           → Historical price & sentiment (180d)   │
│  ├── GET /api/market-prices     → Real-time quotes (14 symbols)         │
│  ├── GET /api/commentary        → AI market analysis                    │
│  ├── GET /api/health            → Health check                          │
│  └── POST /api/pipeline/trigger → Manual pipeline execution             │
│                                                                         │
│  ML PIPELINE (Daily @ 02:00 Istanbul)                                   │
│  ├── 16 strategic queries → Google News RSS                             │
│  ├── LLM sentiment scoring with FinBERT fallback                        │
│  ├── FinBERT CLS embedding extraction + PCA (768→32 dim)               │
│  ├── 250+ feature engineering across 17 training symbols                │
│  ├── XGBoost training with early stopping (when train_model=True)       │
│  ├── TFT-ASRO inference (quantile forecasts from HF Hub checkpoint)    │
│  ├── AI commentary generation                                           │
│  └── Pipeline metrics saved to database                                 │
│                                                                         │
│  DEEP LEARNING (Separate Training)                                      │
│  ├── Feature store: price + sentiment + embeddings + LME + calendar    │
│  ├── TFT-ASRO model (pytorch-forecasting + Lightning)                  │
│  ├── ASRO loss: Sharpe + quantile calibration + volatility matching    │
│  ├── Optuna hyperparameter search with variance ratio penalty          │
│  └── Checkpoint persistence via HuggingFace Hub                        │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                             │
│  ├── Supabase PostgreSQL                                                │
│  │   ├── news_raw / news_processed  (reproducible news pipeline)        │
│  │   ├── news_sentiments_v2         (commodity-aware LLM scores)        │
│  │   ├── news_embeddings            (FinBERT CLS + PCA vectors)        │
│  │   ├── daily_sentiments_v2        (aggregated index)                  │
│  │   ├── price_bars                 (OHLCV data)                        │
│  │   ├── lme_warehouse_data         (LME copper stock levels)           │
│  │   ├── analysis_snapshots         (cached predictions)                │
│  │   ├── ai_commentaries            (market commentary)                 │
│  │   ├── model_metadata             (XGBoost artifacts)                 │
│  │   ├── tft_model_metadata         (TFT-ASRO artifacts)               │
│  │   └── pipeline_run_metrics       (monitoring)                        │
│  ├── HuggingFace Hub (TFT checkpoint + PCA model persistence)           │
│  ├── yfinance (OHLCV price data)                                        │
│  ├── Nasdaq Data Link (LME warehouse stocks, optional)                  │
│  ├── TwelveData (backup live copper price)                              │
│  └── Google News RSS (news source)                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Symbol Sets

The system separates symbol discovery, testing, and production into distinct phases.

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

Used for XGBoost feature engineering. Loaded from `backend/config/symbol_sets/active.json`.

| Category | Symbols                        | Rationale                                                        |
| -------- | ------------------------------ | ---------------------------------------------------------------- |
| Target   | HG=F                           | COMEX Copper Futures - prediction target                         |
| Macro    | DX-Y.NYB, CL=F                 | USD strength inversely correlated; oil = industrial demand proxy |
| Precious | GC=F, SI=F, PL=F               | Safe-haven flows; silver/platinum share industrial use           |
| ETFs     | FXI, COPX                      | China demand (FXI); copper miner basket (COPX)                   |
| Majors   | BHP, FCX, SCCO, RIO, TECK      | Large-cap producers with copper exposure                         |
| Regional | IVN.TO, LUN.TO, FM.TO, 2899.HK | Mid-cap miners; Zijin (2899) = China stockpiling signal          |

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
| `hidden_size`            | 32    | Reduced from 64 to prevent VSN overfitting on ~313 samples |
| `attention_head_size`    | 2     | Fewer heads for single-series dataset                      |
| `dropout`                | 0.3   | Heavy regularization for small sample size                 |
| `hidden_continuous_size` | 16    | Paired reduction with hidden_size                          |
| `learning_rate`          | 3e-4  | Conservative to avoid overshooting narrow loss landscape   |
| `max_encoder_length`     | 60    | 60 trading days of lookback                                |
| `max_prediction_length`  | 5     | 5-day forecast horizon                                     |
| `gradient_clip_val`      | 1.0   | Relaxed from 0.5; tanh-based Sharpe gradients are bounded  |
| `batch_size`             | 16    | ~19 batches/epoch for stable gradient estimates            |

#### Quantile Output

The model produces 7 quantiles instead of a single point estimate:

- q0.02, q0.10, q0.25, **q0.50 (median)**, q0.75, q0.90, q0.98

This directly models 96% and 80% confidence intervals and captures tail risk.

#### ASRO Loss Function

The custom loss combines three objectives:

```
L = w_quantile × (quantile_loss + λ_vol × vol_calibration)
  + w_sharpe × (-tanh_sharpe)
  + amplitude_loss
```

| Component              | Weight            | Purpose                             |
| ---------------------- | ----------------- | ----------------------------------- |
| Quantile calibration   | λ_quantile = 0.4 | Keeps TFT probabilistic             |
| Sharpe component       | w_sharpe = 0.6    | Drives directional learning         |
| Volatility calibration | λ_vol = 0.35     | Q90-Q10 spread tracks 2× actual σ |

The normalized sum-to-1 formulation prevents either component from silently dominating.

#### TFT Feature Categories

The TFT ingests ~340 features across three categories:

**Time-varying unknown (observed in past, unknown in future)**:

- Copper prices, returns, technical indicators (from all 17 training symbols)
- FinBERT PCA embedding vectors (32 dimensions)
- Sentiment momentum (5/10/30-day SMA/EMA), surprise Z-score, volume-weighted sentiment
- Event-type intensity (supply_disruption, inventory_draw, etc.)
- LME warehouse stock changes, depletion rate, cancelled warrant ratio
- Futures curve spread, contango/backwardation flags

**Time-varying known (known in future)**:

- Day of week (one-hot), month (sinusoidal encoding)
- US/China holiday calendars

**Static**:

- Group identifier (single series: "copper")

#### Training and Optimization

TFT training is separate from the daily pipeline. It runs on-demand or via scheduled retraining:

```bash
cd backend

# Train TFT-ASRO model
python -m deep_learning.training.trainer --symbol HG=F

# Run Optuna hyperparameter search (50 trials)
python -m deep_learning.training.hyperopt --n-trials 50

# Backfill FinBERT embeddings for existing articles
python -m deep_learning.data.embeddings --backfill --days 180
```

Optuna search uses a composite objective with variance ratio penalty: trials producing flat predictions (VR < 0.5) or overconfident predictions (VR > 1.5) are penalized.

#### Financial Metrics

| Metric               | Description                                            |
| -------------------- | ------------------------------------------------------ |
| Sharpe Ratio         | Annualized risk-adjusted return of long/short strategy |
| Sortino Ratio        | Sharpe variant penalizing only downside volatility     |
| Directional Accuracy | Fraction of days with correct sign prediction          |
| Tail Capture Rate    | DA on days where\|return\| > 1.5%                      |
| Variance Ratio       | pred_std / actual_std (healthy range: 0.5–1.5)        |
| PI80/PI96 Coverage   | Empirical coverage of 80% and 96% prediction intervals |

#### Model Persistence

TFT artifacts are stored on HuggingFace Hub to survive ephemeral container restarts:

| Artifact                | Description                      |
| ----------------------- | -------------------------------- |
| `best_tft_asro.ckpt`  | Lightning checkpoint             |
| `pca_finbert.joblib`  | Fitted IncrementalPCA model      |
| `optuna_results.json` | Best hyperparameters from search |

## Directory Structure

```
terra-rara/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app, endpoints, scheduler
│   │   ├── ai_engine.py      # XGBoost training, LLM sentiment
│   │   ├── inference.py      # XGBoost live prediction
│   │   ├── features.py       # Technical indicator computation
│   │   ├── data_manager.py   # News ingestion, price fetching
│   │   ├── commentary.py     # AI commentary generation
│   │   ├── scheduler.py      # APScheduler daily pipeline
│   │   ├── models.py         # SQLAlchemy ORM models
│   │   ├── db.py             # Database connection
│   │   └── settings.py       # Pydantic settings
│   ├── deep_learning/        # TFT-ASRO deep learning module
│   │   ├── config.py         # All TFT hyperparameters and training config
│   │   ├── data/
│   │   │   ├── embeddings.py         # FinBERT CLS extraction + PCA
│   │   │   ├── sentiment_features.py # Momentum, surprise, volume-weighted
│   │   │   ├── lme_warehouse.py      # LME warehouse stock data + features
│   │   │   ├── futures_curve.py      # Contango/backwardation signals
│   │   │   ├── feature_store.py      # Centralized data fusion for TFT
│   │   │   └── dataset.py            # TimeSeriesDataSet builder
│   │   ├── models/
│   │   │   ├── tft_copper.py         # TFT model + ASROPFLoss
│   │   │   ├── losses.py             # ASRO + CombinedQuantileLoss
│   │   │   └── hub.py                # HuggingFace Hub upload/download
│   │   ├── training/
│   │   │   ├── trainer.py            # PyTorch Lightning training loop
│   │   │   ├── hyperopt.py           # Optuna TPE hyperparameter search
│   │   │   └── metrics.py            # Sharpe, Sortino, DA, tail capture
│   │   └── inference/
│   │       └── predictor.py          # TFT live prediction pipeline
│   ├── config/
│   │   └── symbol_sets/      # Training symbol configurations
│   │       ├── active.json   # Current training symbols
│   │       ├── champion.json # Best performing set
│   │       └── challenger.json
│   ├── pipelines/            # Faz 2 reproducible data processing
│   │   ├── ingestion/        # RSS/API → news_raw
│   │   ├── processing/       # news_raw → news_processed
│   │   └── cutoff.py         # Market cut-off calculation
│   ├── screener/             # Universe Builder + Feature Screener
│   │   ├── core/             # Config, fingerprint, cache
│   │   ├── contracts/        # Pydantic output models
│   │   ├── universe_builder/ # Seed loading, probing, categorization
│   │   └── feature_screener/ # Correlation analysis
│   ├── worker/               # arq + Redis async task worker
│   │   └── tasks.py          # Pipeline orchestration tasks
│   ├── backtest/             # Walk-forward backtesting
│   │   └── runner.py         # XGBoost + TFT backtest runner
│   ├── tests/                # pytest tests
│   │   ├── deep_learning/    # TFT-ASRO unit tests (7 files)
│   │   └── screener/         # Screener unit tests
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main dashboard
│   │   ├── api.ts            # API client (XGBoost + TFT)
│   │   ├── types.ts          # TypeScript interfaces
│   │   └── components/       # React components
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
├── data/
│   └── models/               # Local model artifacts (gitignored)
├── .github/
│   └── workflows/
│       ├── hf-sync.yml       # GitHub → HuggingFace sync
│       └── screener-schedule.yml
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
- OpenRouter API key (free tier available at [openrouter.ai](https://openrouter.ai))

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
| `OPENROUTER_API_KEY`                | Yes      | -                                       | OpenRouter API key for LLM                                     |
| `PIPELINE_TRIGGER_SECRET`           | Yes      | -                                       | Secret token for POST /api/pipeline/trigger (32+ random chars) |
| `SYMBOL_SET`                        | No       | `active`                              | Which symbol set to use (active/champion/challenger)           |
| `OPENROUTER_MODEL_SCORING`          | No       | `arcee-ai/trinity-large-preview:free` | Primary model for sentiment scoring                            |
| `OPENROUTER_MODEL_COMMENTARY`       | No       | `arcee-ai/trinity-large-preview:free` | Primary model for commentary generation                        |
| `OPENROUTER_RPM`                    | No       | `18`                                  | Soft throttle target for OpenRouter calls                      |
| `OPENROUTER_MAX_RETRIES`            | No       | `3`                                   | Max retry attempts for 429/5xx OpenRouter errors               |
| `MAX_LLM_ARTICLES_PER_RUN`          | No       | `200`                                 | Per-run LLM scoring budget before FinBERT overflow             |
| `OPENROUTER_FALLBACK_MODELS`        | No       | empty                                   | Optional comma-separated fallback model list                   |
| `OPENROUTER_MODEL`                  | No       | -                                       | Deprecated fallback model env (backward compatibility)         |
| `LLM_SENTIMENT_MODEL`               | No       | -                                       | Deprecated fallback scoring model env                          |
| `TWELVEDATA_API_KEY`                | No       | -                                       | Backup live price source                                       |
| `SCHEDULER_ENABLED`                 | No       | `false`                               | Local-only scheduler flag (production uses external trigger)   |
| `SCHEDULE_TIME`                     | No       | `02:00`                               | Daily pipeline time (HH:MM)                                    |
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

The `env.example` file includes `PIPELINE_TRIGGER_SECRET=` with no value. Generate a random secret before deploying.

## Usage

### Dashboard

Access the live dashboard at [terra-rara.vercel.app](https://terra-rara.vercel.app):

1. **Deep Learning Forecast** displays the TFT-ASRO T+1 predicted return, direction (BULLISH/BEARISH/NEUTRAL), risk level, and 5-day weekly trend
2. **Deep Learning Metrics** shows directional accuracy, Sharpe ratio, variance ratio, and tail capture rate
3. **XGBoost Forecast Card** displays the predicted next-day copper price, predicted return percentage, and AI stance
4. **Price Forecast Chart** shows 180 days of historical copper prices with T+1 TFT forecast dot and Q10-Q90 confidence band
5. **Market Drivers** lists the top XGBoost feature importances
6. **Market Grid** shows real-time prices for all 14 dashboard symbols

### API Endpoints

Interactive API documentation: [ifieryarrows-copper-mind.hf.space/api/docs](https://ifieryarrows-copper-mind.hf.space/api/docs)

```bash
# Get current XGBoost prediction
curl https://ifieryarrows-copper-mind.hf.space/api/analysis

# Get TFT-ASRO deep learning forecast
curl https://ifieryarrows-copper-mind.hf.space/api/analysis/tft/HG=F

# Get AI commentary
curl https://ifieryarrows-copper-mind.hf.space/api/commentary

# Get historical data
curl https://ifieryarrows-copper-mind.hf.space/api/history

# Trigger pipeline manually (requires authentication)
curl -X POST "https://ifieryarrows-copper-mind.hf.space/api/pipeline/trigger?fetch_data=true&train_model=true" \
  -H "Authorization: Bearer YOUR_PIPELINE_TRIGGER_SECRET"
```

## API Reference

### GET /api/analysis

Returns current prediction with model metrics.

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

### GET /api/commentary

Returns AI-generated market analysis.

```json
{
  "symbol": "HG=F",
  "commentary": "Copper futures show bullish momentum...",
  "ai_stance": "BULLISH",
  "generated_at": "2026-01-17T09:15:00Z"
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

### GET /api/analysis/tft/

Returns TFT-ASRO deep learning forecast with quantile predictions. Results are cached for 5 minutes.

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

### POST /api/pipeline/trigger (Privileged)

Manually triggers the ML pipeline. This is a privileged endpoint that consumes significant resources (LLM API calls, database writes, model training).

**Authentication requirement**: This endpoint requires a valid `Authorization: Bearer <PIPELINE_TRIGGER_SECRET>` header. Requests without a valid token receive 401 Unauthorized.

**Expected responses**:

| Status           | Condition                                   |
| ---------------- | ------------------------------------------- |
| 200              | Pipeline triggered successfully             |
| 401 Unauthorized | Missing or invalid `Authorization` header |
| 409 Conflict     | Pipeline already running                    |

**Parameters**:

| Parameter       | Type    | Default  | Description               |
| --------------- | ------- | -------- | ------------------------- |
| `fetch_data`  | boolean | `true` | Fetch new news and prices |
| `train_model` | boolean | `true` | Retrain XGBoost model     |

**Daily vs Manual Pipeline**:

- Daily scheduler runs with `train_model=False` (uses existing model, refreshes sentiment only)
- Manual trigger defaults to `train_model=True` (retrains model)

## Development

### Running Tests

```bash
cd backend

# All tests
pytest tests/ -v

# Deep learning tests only
pytest tests/deep_learning/ -v

# Screener tests only
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

The pipeline is triggered daily by external scheduler automation (GitHub Actions cron). Local scheduler mode (`SCHEDULER_ENABLED=true`) is for development only:

1. Fetch news from 16 strategic Google News RSS queries
2. Process news through reproducible pipeline (news_raw → news_processed)
3. Fetch price data for 17 training symbols via yfinance
4. Score sentiment via LLM in batches of 20 articles
5. Compute FinBERT CLS embeddings and store PCA-reduced vectors
6. Aggregate daily sentiment with time-weighted decay (half-life: 7 days)
7. Generate XGBoost prediction using existing model (no retraining by default)
8. Generate TFT-ASRO inference from HF Hub checkpoint (quantile forecasts)
9. Generate AI commentary via OpenRouter
10. Cache prediction snapshots
11. Save pipeline metrics to `pipeline_run_metrics` table

TFT model training is separate from the daily pipeline and runs on-demand.

### Pipeline Monitoring

Each pipeline run records metrics to the database:

```sql
SELECT run_id, run_started_at, duration_seconds, 
       symbols_failed, status, symbol_set_name,
       tft_trained, tft_sharpe, tft_directional_accuracy
FROM pipeline_run_metrics 
ORDER BY run_started_at DESC 
LIMIT 10;
```

Tracked metrics:

- `duration_seconds`: Total pipeline runtime
- `symbols_requested` / `symbols_fetched_ok` / `symbols_failed`: Data fetch stats
- `news_raw_inserted` / `news_raw_duplicates`: News ingestion stats
- `news_processed_inserted` / `articles_scored_v2`: Processing stats
- `tft_embeddings_computed`: FinBERT embeddings generated
- `tft_trained` / `tft_val_loss` / `tft_sharpe` / `tft_directional_accuracy`: TFT metrics
- `snapshot_generated` / `commentary_generated` / `tft_snapshot_generated`: Output flags
- `status`: success/failed
- `error_message`: Error details if failed

## Troubleshooting

### Database connection refused

**Symptom**: `psycopg2.OperationalError: could not connect to server`

**Cause**: PostgreSQL is not running or `DATABASE_URL` is incorrect.

**Fix**: Verify PostgreSQL is running. Check `DATABASE_URL` format: `postgresql://user:password@host:port/database`. For Supabase, use the connection pooler URL.

### LLM sentiment scoring fails, falling back to FinBERT

**Symptom**: Logs show `LLM sentiment failed, using FinBERT fallback`

**Cause**: OpenRouter API key is missing, invalid, or rate-limited.

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

This endpoint triggers the full ML pipeline, which fetches news, calls the LLM API for sentiment scoring, computes FinBERT embeddings, retrains the XGBoost model, runs TFT-ASRO inference, and generates AI commentary. Unauthenticated access to this endpoint creates the following risks:

- **Request flooding**: Repeated triggers can degrade service availability.
- **Quota and cost burn**: Each pipeline run consumes OpenRouter API quota. Uncontrolled access can exhaust free-tier limits or incur costs.
- **Resource exhaustion**: Model training and batch LLM calls are CPU and memory intensive.

**Authentication**: This endpoint requires a valid `Authorization: Bearer <PIPELINE_TRIGGER_SECRET>` header. Requests without a valid token receive 401 Unauthorized.

### Configuration

1. **Set `PIPELINE_TRIGGER_SECRET`** in your `.env` file. Use a random string of 32 or more characters. Store this as a secret in your deployment platform.
2. **Rotate the secret immediately** if it is ever exposed in logs, commits, or third-party systems.

### Abuse Prevention (Recommended Controls)

Even with authentication, apply additional safeguards:

- **Rate limiting**: Limit to 5-10 requests per minute per IP or per token. Use a reverse proxy (nginx, Caddy) or middleware.
- **Idempotency lock**: The endpoint already checks if the pipeline is running and returns 409 Conflict. Ensure this lock file mechanism is reliable across restarts.
- **Network restriction**: If the endpoint is only used by internal automation (CI/CD, cron jobs), restrict access via IP allowlist, VPN, or deploy the backend on an internal network.
- **Monitoring and alerting**: Log all calls to this endpoint. Alert on spikes in 401 responses, sudden trigger volume, or upstream API quota warnings.

### Secret Management

- Do not commit secrets to version control.
- Use your deployment platform's secret management (HuggingFace Spaces secrets, Vercel environment variables, etc.).
- The `env.example` file should include `PIPELINE_TRIGGER_SECRET=` with no value as a placeholder.

## License

MIT License - see [LICENSE](LICENSE) for details.
