# Terra Rara (CopperMind)

AI-powered copper futures price prediction platform combining XGBoost ML, LLM-based sentiment analysis, and real-time market intelligence.

[![Live Demo](https://img.shields.io/badge/demo-terra--rara.vercel.app-0969da)](https://terra-rara.vercel.app)
[![API Docs](https://img.shields.io/badge/api-docs-10b981)](https://ifieryarrows-copper-mind.hf.space/api/docs)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## Overview

Terra Rara predicts next-day copper futures (HG=F) closing prices using an XGBoost regression model trained on:
- **200+ technical features** from copper and 13 correlated assets
- **LLM-scored news sentiment** (MiMo Flash via OpenRouter) with FinBERT fallback
- **Cross-asset dynamics** including mining stocks, ETFs, and macro indicators

## Live Demo

| Service | URL |
|---------|-----|
| Dashboard | [terra-rara.vercel.app](https://terra-rara.vercel.app) |
| API Docs | [ifieryarrows-copper-mind.hf.space/api/docs](https://ifieryarrows-copper-mind.hf.space/api/docs) |

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  FRONTEND (Vercel)                                                    │
│  React 18 + TypeScript + Vite + TailwindCSS                           │
│  ├── Real-time TradingView widget (lazy loaded)                       │
│  ├── Price & Sentiment Chart (Recharts)                               │
│  ├── Model Forecast Card with AI Commentary                           │
│  ├── Market Drivers (XGBoost feature importance)                      │
│  └── Global Intelligence Map (14 symbols, live)                       │
│  Performance: Lazy loading, skeleton loaders, preconnect hints        │
└───────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  BACKEND (HuggingFace Spaces - Docker)                                │
│  FastAPI + Python 3.11 + Uvicorn + APScheduler                        │
│                                                                       │
│  REST API                                                             │
│  ├── GET /api/analysis      → Live XGBoost prediction + metrics       │
│  ├── GET /api/history       → Historical price & sentiment (180d)     │
│  ├── GET /api/market-prices → Real-time quotes (14 symbols)           │
│  ├── GET /api/commentary    → AI-generated market analysis            │
│  ├── GET /api/health        → Health check with DB connectivity       │
│  └── POST /api/pipeline/trigger → Manual pipeline execution           │
│                                                                       │
│  ML PIPELINE (Daily @ 02:00 Istanbul)                                 │
│  ├── News ingestion: 16 strategic queries → Google News RSS           │
│  ├── LLM sentiment scoring (MiMo Flash) with FinBERT fallback         │
│  ├── 200+ feature engineering across 14 symbols                       │
│  ├── XGBoost training with hyperparameter optimization                │
│  ├── Daily sentiment aggregation (time-weighted decay)                │
│  └── AI commentary generation (MiMo Flash via OpenRouter)             │
└───────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                           │
│  ├── Supabase PostgreSQL (managed, connection pooling)                │
│  │   ├── news_articles       (raw news storage)                       │
│  │   ├── news_sentiments     (LLM scores + reasoning)                 │
│  │   ├── daily_sentiments    (aggregated daily index)                 │
│  │   ├── analysis_snapshots  (cached predictions)                     │
│  │   ├── ai_commentaries     (generated market commentary)            │
│  │   └── model_metadata      (XGBoost artifacts + influencers)        │
│  ├── yfinance (OHLCV price data, 14 symbols)                          │
│  ├── TwelveData (backup live copper price)                            │
│  └── Google News RSS (sentiment source, 16 topic queries)             │
└───────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.x | UI framework |
| TypeScript | 5.x | Type safety |
| Vite | 5.x | Build tool, dev server |
| Recharts | 2.x | Interactive charts |
| Framer Motion | 11.x | Animations |
| TailwindCSS | 3.x | Styling |
| Lucide React | - | Icons |
| clsx | - | Conditional classes |
| @vercel/speed-insights | - | Performance monitoring |

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.115.x | REST API framework |
| Python | 3.11 | Runtime |
| Uvicorn | 0.32.x | ASGI server |
| SQLAlchemy | 2.x | ORM |
| APScheduler | 3.x | Scheduled pipeline jobs |
| httpx | - | Async HTTP client |
| psycopg2 | - | PostgreSQL driver |

### Machine Learning
| Technology | Purpose |
|------------|---------|
| XGBoost | Price prediction model (regression) |
| Transformers (FinBERT) | Fallback sentiment scoring |
| MiMo Flash (OpenRouter) | Primary LLM sentiment scoring |
| pandas/numpy | Data manipulation |
| scikit-learn | Feature preprocessing |

### Infrastructure
| Service | Purpose |
|---------|---------|
| Vercel | Frontend hosting, CDN, auto-deploy |
| HuggingFace Spaces | Backend hosting (Docker) |
| Supabase | PostgreSQL database, connection pooling |
| GitHub Actions | CI/CD, HF sync workflow |
| OpenRouter | LLM API gateway (MiMo Flash) |

## Tracked Symbols (14)

| Category | Symbols | Description |
|----------|---------|-------------|
| **Target** | HG=F | COMEX Copper Futures |
| **Macro** | DX-Y.NYB, CL=F | USD Index, Crude Oil |
| **ETFs** | FXI, COPX, COPJ | China Large-Cap, Global Miners, Junior Miners |
| **Titans** | BHP, FCX, SCCO, RIO | Major copper producers |
| **Regional** | TECK, IVN.TO, LUN.TO, 2899.HK | Regional mining companies |

## LLM Sentiment Analysis

The system uses a sophisticated LLM prompt for copper-specific sentiment scoring:

### Evaluation Framework
1. **Supply availability** - strikes, shutdowns, grade decline, logistic constraints
2. **Demand outlook** - China stimulus, EV/grid buildout, recession risk
3. **Inventories** - LME/COMEX/SHFE drawdowns, backwardation/contango
4. **Macro FX/rates** - USD strength/weakness, rate impact on metals
5. **Substitution/policy** - copper-to-aluminum substitution, electrification policy

### Magnitude Calibration
| Score Range | Signal Strength |
|-------------|-----------------|
| 0.05–0.20 | Weak/indirect/uncertain linkage |
| 0.25–0.45 | Moderately copper-relevant |
| 0.50–0.70 | Direct driver with clear mechanism |
| 0.75–1.00 | Major shock (supply cut, sharp inventory move) |

### Model Fallback
- **Primary**: MiMo Flash (`xiaomi/mimo-v2-flash:free`) via OpenRouter
- **Fallback**: FinBERT (`ProsusAI/finbert`) when LLM fails

## Feature Engineering (200+)

For each of the 14 tracked symbols, the following features are computed:

| Feature Type | Features Generated |
|--------------|-------------------|
| Returns | 1-day return (`ret1`) |
| Lagged returns | 1, 2, 3, 5 day lags (`lag_ret1_N`) |
| Moving averages | SMA 5, 10, 20 / EMA 5, 10, 20 |
| Momentum | RSI 14-day |
| Volatility | 10-day rolling std (`vol_10`) |
| Price ratios | Price-to-SMA ratio |
| **Sentiment** | Daily aggregated index + news count |

**Total**: 14 symbols × 14 features + 2 sentiment = **198+ features**

## XGBoost Configuration

```python
{
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.5,      # L1 regularization
    "reg_lambda": 2.0,     # L2 regularization
    "n_estimators": 200,
    "early_stopping_rounds": 30
}
```

## Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL database (or Supabase account)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp ../env.example .env
# Edit .env with your credentials

# Run development server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

```env
# ═══════════════════════════════════════════════════════════
# DATABASE (Required)
# ═══════════════════════════════════════════════════════════
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# ═══════════════════════════════════════════════════════════
# LLM APIs (Required for AI features)
# ═══════════════════════════════════════════════════════════
OPENROUTER_API_KEY=sk-or-v1-xxxx

# Commentary model (supports reasoning)
OPENROUTER_MODEL=xiaomi/mimo-v2-flash:free

# Sentiment scoring model (high-volume, fast)
LLM_SENTIMENT_MODEL=xiaomi/mimo-v2-flash:free

# ═══════════════════════════════════════════════════════════
# MARKET DATA (Optional - has fallbacks)
# ═══════════════════════════════════════════════════════════
TWELVEDATA_API_KEY=xxxx           # Backup live price source

# ═══════════════════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════════════════
SCHEDULER_ENABLED=true
SCHEDULE_TIME=02:00               # Daily pipeline execution
TZ=Europe/Istanbul

# ═══════════════════════════════════════════════════════════
# FEATURE CONFIGURATION
# ═══════════════════════════════════════════════════════════
YFINANCE_SYMBOLS=HG=F,DX-Y.NYB,CL=F,FXI,COPX,COPJ,BHP,FCX,SCCO,RIO,TECK,LUN.TO,IVN.TO,2899.HK
NEWS_LOOKBACK_DAYS=30
SENTIMENT_DECAY_HALF_LIFE=7.0
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

### POST /api/pipeline/trigger

Manually trigger the ML pipeline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| fetch_data | boolean | true | Fetch new news and prices |
| train_model | boolean | true | Retrain XGBoost model |

## Project Structure

```
terra-rara/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app, endpoints, scheduler
│   │   ├── ai_engine.py      # XGBoost training, LLM sentiment scoring
│   │   ├── inference.py      # Live prediction with feature matching
│   │   ├── features.py       # Technical indicator computation
│   │   ├── data_manager.py   # News ingestion, price fetching
│   │   ├── commentary.py     # AI commentary generation
│   │   ├── models.py         # SQLAlchemy ORM models
│   │   ├── db.py             # Database connection
│   │   └── settings.py       # Pydantic settings configuration
│   ├── Dockerfile            # HuggingFace Spaces deployment
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main dashboard (lazy loading)
│   │   ├── api.ts            # API client functions
│   │   ├── types.ts          # TypeScript interfaces
│   │   └── components/
│   │       └── MarketMap.tsx # Real-time market grid
│   ├── index.html            # Preconnect hints, meta tags
│   ├── vite.config.ts
│   └── package.json
├── data/
│   └── models/               # Local model artifacts (git-ignored)
├── .github/
│   └── workflows/
│       └── hf-sync.yml       # GitHub → HuggingFace sync
├── env.example               # Environment template
└── README.md
```

## Performance Optimizations

### Frontend
- **Lazy loading**: MarketMap and TradingView widget loaded after initial render
- **Code splitting**: MarketMap as separate chunk (3.76 KB)
- **Skeleton loaders**: Perceived performance improvement
- **Preconnect hints**: API, TradingView, fonts
- **Silent refresh**: No UI flash during 60s data updates

### Backend
- **Chunked processing**: 20 articles per LLM batch
- **Fallback mechanism**: FinBERT when LLM fails
- **Connection pooling**: Supabase pooler for DB connections
- **Snapshot caching**: Predictions cached, refreshed on demand

## Daily Pipeline Flow

```
02:00 Istanbul (23:00 UTC)
     │
     ▼
┌─────────────────────┐
│ 1. Fetch News       │  16 strategic queries → Google News RSS
│    ~1200 articles   │  Language filter, fuzzy dedup
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ 2. LLM Scoring      │  Chunk 20 articles → MiMo Flash
│    ~106 batches     │  FinBERT fallback on failure
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ 3. Fetch Prices     │  14 symbols → yfinance (2yr history)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ 4. Aggregate        │  Daily sentiment (time-weighted decay)
│    Sentiment        │  Half-life: 7 days
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ 5. Train XGBoost    │  200+ features, 80/20 split
│                     │  Early stopping, validation
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ 6. Generate         │  AI commentary via OpenRouter
│    Commentary       │  Stance determination (BULLISH/NEUTRAL/BEARISH)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ 7. Save Snapshot    │  Cache prediction for API
└─────────────────────┘
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ for copper market intelligence**
