# CopperMind

AI-powered copper futures price prediction platform combining machine learning, sentiment analysis, and real-time market data.

[![Live Demo](https://img.shields.io/badge/demo-terra--rara.vercel.app-0969da)](https://terra-rara.vercel.app)
[![API Docs](https://img.shields.io/badge/api-docs-10b981)](https://ifieryarrows-copper-mind.hf.space/api/docs)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## Overview

CopperMind predicts next-day copper futures (HG=F) closing prices using an XGBoost regression model trained on:
- **Technical indicators** from copper and correlated assets (USD Index, Crude Oil, China ETF)
- **News sentiment** scored by FinBERT with time-weighted aggregation
- **Cross-asset features** including lagged returns and volatility measures

## Live Demo

| Service | URL |
|---------|-----|
| Dashboard | [terra-rara.vercel.app](https://terra-rara.vercel.app) |
| API | [ifieryarrows-copper-mind.hf.space/api/docs](https://ifieryarrows-copper-mind.hf.space/api/docs) |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  FRONTEND (Vercel)          React + TypeScript + Vite        │
│  ├── Price & Sentiment Chart (Recharts)                      │
│  ├── Tomorrow's Prediction Card                              │
│  └── Market Intelligence Map (14 symbols, live)              │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  BACKEND (HuggingFace Spaces)     FastAPI + Python 3.11      │
│  ├── /api/analysis      → Live XGBoost prediction            │
│  ├── /api/history       → Historical price & sentiment       │
│  ├── /api/market-prices → Real-time quotes (yfinance)        │
│  └── /api/commentary    → AI-generated analysis (OpenRouter) │
│                                                              │
│  ML PIPELINE                                                 │
│  ├── FinBERT sentiment scoring                               │
│  ├── 60+ feature engineering                                 │
│  └── XGBoost model training                                  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                  │
│  ├── Supabase PostgreSQL (persistence)                       │
│  ├── yfinance (price data)                                   │
│  └── Google News RSS (sentiment source)                      │
└──────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | React 18, TypeScript, Vite, Recharts |
| Backend | FastAPI, Python 3.11, Uvicorn |
| ML | XGBoost, FinBERT (transformers) |
| Database | Supabase PostgreSQL |
| Hosting | Vercel (frontend), HuggingFace Spaces (backend) |

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
# Edit .env with your DATABASE_URL and API keys

# Run development server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Configuration

Create a `.env` file in the backend directory:

```env
# Required
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Optional - News API
NEWSAPI_KEY=your_newsapi_key

# Optional - AI Commentary
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=xiaomi/mimo-v2-flash:free

# Scheduler
SCHEDULER_ENABLED=true
SCHEDULE_TIME=09:00
TZ=Europe/Istanbul
```

## API Reference

### GET /api/analysis

Returns current prediction with live price.

```json
{
  "symbol": "HG=F",
  "current_price": 4.2500,
  "predicted_price": 4.3137,
  "predicted_return": 0.0150,
  "sentiment_index": 0.35,
  "sentiment_label": "Bullish",
  "top_influencers": [...],
  "generated_at": "2026-01-10T09:00:00Z"
}
```

### GET /api/history

Returns historical price and sentiment data for charting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| symbol | string | HG=F | Trading symbol |
| days | integer | 180 | Days of history (7-730) |

### POST /api/pipeline/trigger

Manually trigger the data pipeline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| fetch_data | boolean | true | Fetch new news and prices |
| train_model | boolean | true | Retrain XGBoost model |

## Project Structure

```
copper-mind/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI endpoints
│   │   ├── ai_engine.py     # XGBoost training
│   │   ├── inference.py     # Live prediction
│   │   ├── features.py      # Feature engineering
│   │   ├── data_manager.py  # Data ingestion
│   │   ├── models.py        # SQLAlchemy models
│   │   └── settings.py      # Configuration
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main dashboard
│   │   ├── api.ts           # API client
│   │   └── components/
│   └── package.json
├── data/
│   └── models/              # Trained model files
└── README.md
```

## Model Details

### Features (60+)
- **Technical indicators**: SMA, EMA, RSI (5, 10, 14, 20 day periods)
- **Price ratios**: Price-to-SMA, volatility measures
- **Lagged returns**: 1, 2, 3, 5 day lags
- **Cross-asset**: USD Index, Crude Oil, China ETF features
- **Sentiment**: Daily aggregated news sentiment index

### XGBoost Configuration

```python
{
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0
}
```

### Tracked Symbols

| Category | Symbols |
|----------|---------|
| Target | HG=F (Copper Futures) |
| Macro | DX-Y.NYB (USD Index), CL=F (Crude Oil) |
| ETFs | FXI (China), COPX (Miners), COPJ (Junior Miners) |
| Miners | BHP, FCX, SCCO, RIO, TECK, LUN.TO, IVN.TO |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.
