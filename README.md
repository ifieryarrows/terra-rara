# Terra Rara

AI-powered copper futures price prediction platform combining XGBoost ML, LLM-based sentiment analysis, and cross-asset market data.

[![Live Demo](https://img.shields.io/badge/demo-terra--rara.vercel.app-0969da)](https://terra-rara.vercel.app)
[![API Docs](https://img.shields.io/badge/api-docs-10b981)](https://ifieryarrows-copper-mind.hf.space/api/docs)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
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
- [License](#license)

## Overview

Terra Rara predicts next-day COMEX copper futures (HG=F) closing prices using an XGBoost regression model. The system ingests daily news via Google News RSS, scores article sentiment using an LLM (MiMo Flash via OpenRouter), and combines this with 200+ technical features computed from 14 correlated assets. A scheduled pipeline runs daily to retrain the model and generate AI-driven market commentary.

**Target users**: Traders, analysts, and developers building commodity forecasting tools.

**Non-goals**: This system does not provide trading signals or financial advice. Predictions are for informational purposes only.

## Features

- Predict next-day copper futures prices using XGBoost regression trained on 200+ features
- Score news sentiment using LLM (MiMo Flash) with FinBERT fallback when API is unavailable
- Track 14 correlated assets including copper miners, ETFs, and macro indicators
- Aggregate daily sentiment using time-weighted exponential decay
- Generate AI-powered market commentary with stance classification (BULLISH/NEUTRAL/BEARISH)
- Display real-time prices for all tracked symbols via dashboard
- Visualize historical price and sentiment data over 180 days
- Trigger manual pipeline execution via API endpoint

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (Vercel)                                                      │
│  React 18 + TypeScript + Vite + TailwindCSS                             │
│  ├── TradingView widget (lazy loaded)                                   │
│  ├── Price & Sentiment chart (Recharts)                                 │
│  ├── Model Forecast Card with AI Commentary                             │
│  ├── XGBoost feature importance display                                 │
│  └── Market grid showing 14 symbols                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BACKEND (HuggingFace Spaces - Docker)                                  │
│  FastAPI + Python 3.11 + Uvicorn + APScheduler                          │
│                                                                         │
│  REST API                                                               │
│  ├── GET /api/analysis      → XGBoost prediction + metrics              │
│  ├── GET /api/history       → Historical price & sentiment (180d)       │
│  ├── GET /api/market-prices → Real-time quotes (14 symbols)             │
│  ├── GET /api/commentary    → AI market analysis                        │
│  ├── GET /api/health        → Health check                              │
│  └── POST /api/pipeline/trigger → Manual pipeline execution             │
│                                                                         │
│  ML PIPELINE (Daily @ 02:00 Istanbul)                                   │
│  ├── 16 strategic queries → Google News RSS                             │
│  ├── LLM sentiment scoring with FinBERT fallback                        │
│  ├── 200+ feature engineering across 14 symbols                         │
│  ├── XGBoost training with early stopping                               │
│  └── AI commentary generation                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                             │
│  ├── Supabase PostgreSQL                                                │
│  │   ├── news_articles       (raw news)                                 │
│  │   ├── news_sentiments     (LLM scores)                               │
│  │   ├── daily_sentiments    (aggregated index)                         │
│  │   ├── analysis_snapshots  (cached predictions)                       │
│  │   ├── ai_commentaries     (market commentary)                        │
│  │   └── model_metadata      (XGBoost artifacts)                        │
│  ├── yfinance (OHLCV price data)                                        │
│  ├── TwelveData (backup live copper price)                              │
│  └── Google News RSS (news source)                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tracked Symbols (14)

| Category | Symbols | Description |
|----------|---------|-------------|
| Target | HG=F | COMEX Copper Futures |
| Macro | DX-Y.NYB, CL=F | USD Index, Crude Oil |
| ETFs | FXI, COPX, COPJ | China Large-Cap, Global Miners, Junior Miners |
| Miners | BHP, FCX, SCCO, RIO | Major copper producers |
| Regional | TECK, IVN.TO, LUN.TO, 2899.HK | Regional mining companies |

## Directory Structure

```
terra-rara/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app, endpoints, scheduler
│   │   ├── ai_engine.py      # XGBoost training, LLM sentiment
│   │   ├── inference.py      # Live prediction
│   │   ├── features.py       # Technical indicator computation
│   │   ├── data_manager.py   # News ingestion, price fetching
│   │   ├── commentary.py     # AI commentary generation
│   │   ├── models.py         # SQLAlchemy ORM models
│   │   ├── db.py             # Database connection
│   │   └── settings.py       # Pydantic settings
│   ├── tests/                # pytest tests
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main dashboard
│   │   ├── api.ts            # API client
│   │   ├── types.ts          # TypeScript interfaces
│   │   └── components/       # React components
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
├── data/
│   └── models/               # Local model artifacts (gitignored)
├── .github/
│   └── workflows/
│       └── hf-sync.yml       # GitHub → HuggingFace sync
├── docker-compose.yml
├── env.example
└── README.md
```

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

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `DATABASE_URL` | Yes | - | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` |
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key for LLM | `sk-or-v1-xxxx` |
| `PIPELINE_TRIGGER_SECRET` | Yes | - | Secret token for POST /api/pipeline/trigger (32+ random chars) | `a1b2c3...` (32+ chars) |
| `OPENROUTER_MODEL` | No | `xiaomi/mimo-v2-flash:free` | Model for AI commentary | - |
| `LLM_SENTIMENT_MODEL` | No | `xiaomi/mimo-v2-flash:free` | Model for sentiment scoring | - |
| `TWELVEDATA_API_KEY` | No | - | Backup live price source | `xxxx` |
| `SCHEDULER_ENABLED` | No | `true` | Enable daily pipeline | `true` or `false` |
| `SCHEDULE_TIME` | No | `02:00` | Daily pipeline time (HH:MM) | `09:00` |
| `TZ` | No | `Europe/Istanbul` | Scheduler timezone | `UTC` |
| `YFINANCE_SYMBOLS` | No | (14 symbols) | Comma-separated symbols | `HG=F,BHP,FCX` |
| `NEWS_LOOKBACK_DAYS` | No | `30` | Days of news to fetch | `30` |
| `SENTIMENT_DECAY_HALF_LIFE` | No | `7.0` | Sentiment decay half-life (days) | `7.0` |

The `env.example` file includes `PIPELINE_TRIGGER_SECRET=` with no value. Generate a random secret before deploying.

## Usage

### Dashboard

Access the live dashboard at [terra-rara.vercel.app](https://terra-rara.vercel.app):

1. **Forecast Card** displays the predicted next-day copper price, predicted return percentage, and AI stance
2. **Price & Sentiment Chart** shows 180 days of historical copper prices overlaid with the daily sentiment index
3. **Market Drivers** lists the top XGBoost feature importances
4. **Market Grid** shows real-time prices for all 14 tracked symbols

### API Endpoints

Interactive API documentation: [ifieryarrows-copper-mind.hf.space/api/docs](https://ifieryarrows-copper-mind.hf.space/api/docs)

```bash
# Get current prediction
curl https://ifieryarrows-copper-mind.hf.space/api/analysis

# Get AI commentary
curl https://ifieryarrows-copper-mind.hf.space/api/commentary

# Get historical data
curl https://ifieryarrows-copper-mind.hf.space/api/history

# Trigger pipeline manually
curl -X POST https://ifieryarrows-copper-mind.hf.space/api/pipeline/trigger
```

## API Reference

### GET /api/analysis

Returns current prediction with model metrics.

```json
{
  "symbol": "HG=F",
  "current_price": 4.2500,
  "predicted_price": 4.3137,
  "predicted_return": 0.0150,
  "sentiment_index": 0.227,
  "sentiment_label": "Bullish",
  "top_influencers": [
    {"feature": "HG=F_vol_10", "importance": 0.1808},
    {"feature": "FXI_lag_ret1_2", "importance": 0.1019}
  ],
  "data_quality": {
    "coverage_pct": 98.5,
    "missing_features": []
  },
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

Returns real-time quotes for all tracked symbols.

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

### POST /api/pipeline/trigger (Privileged)

Manually triggers the ML pipeline. This is a privileged endpoint that consumes significant resources (LLM API calls, database writes, model training).

**Authentication requirement**: This endpoint must be protected before exposing it publicly. Requests must include a valid secret token in the header:

- **Header**: `Authorization: Bearer <PIPELINE_TRIGGER_SECRET>`

**Expected responses**:

| Status | Condition |
|--------|-----------|
| 200 | Pipeline triggered successfully |
| 401 Unauthorized | Missing or invalid `Authorization` header |
| 409 Conflict | Pipeline already running |

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fetch_data` | boolean | `true` | Fetch new news and prices |
| `train_model` | boolean | `true` | Retrain XGBoost model |

**Warning**: The current implementation does not enforce authentication. The endpoint must be protected before public deployment. See the [Security](#security) section for details.

## Development

### Running Tests

```bash
cd backend
pytest tests/ -v
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

The pipeline runs automatically at 02:00 Istanbul time when `SCHEDULER_ENABLED=true`:

1. Fetch news from 16 strategic Google News RSS queries (~1200 articles)
2. Score sentiment via LLM in batches of 20 articles
3. Fetch 2 years of price data for 14 symbols via yfinance
4. Aggregate daily sentiment with time-weighted decay (half-life: 7 days)
5. Train XGBoost model with 80/20 train/validation split
6. Generate AI commentary via OpenRouter
7. Cache prediction snapshot

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

**Fix**: Check backend logs for pipeline errors. Verify `SCHEDULER_ENABLED=true`. Manually trigger pipeline via `POST /api/pipeline/trigger` to test.

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

### Privileged Endpoint: POST /api/pipeline/trigger

This endpoint triggers the full ML pipeline, which fetches news, calls the LLM API for sentiment scoring, retrains the XGBoost model, and generates AI commentary. Unauthenticated access to this endpoint creates the following risks:

- **Request flooding**: Repeated triggers can degrade service availability.
- **Quota and cost burn**: Each pipeline run consumes OpenRouter and TwelveData API quota. Uncontrolled access can exhaust free-tier limits or incur costs.
- **Resource exhaustion**: Model training and batch LLM calls are CPU and memory intensive.

**Current status**: The endpoint does not currently implement authentication. Before exposing this API publicly, you must implement protection.

### Required Protection

1. **Set `PIPELINE_TRIGGER_SECRET`** in your `.env` file. Use a random string of 32 or more characters. Store this as a secret in your deployment platform.

2. **Implement header validation** in the backend. Requests to POST /api/pipeline/trigger must include:
   - `Authorization: Bearer <PIPELINE_TRIGGER_SECRET>`
   - The backend must return 401 Unauthorized if the header is missing or the token does not match.

3. **Rotate the secret immediately** if it is ever exposed in logs, commits, or third-party systems.

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
