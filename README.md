# 🔮 CopperMind - AI-Powered Copper Price Intelligence

**Real-time copper futures prediction platform using XGBoost ML, sentiment analysis, and live market data.**

![Live Demo](https://img.shields.io/badge/demo-terra--rara.vercel.app-blue)
![Backend](https://img.shields.io/badge/backend-HuggingFace%20Spaces-orange)
![Database](https://img.shields.io/badge/database-Supabase-green)

---

## 📊 Live Demo

- **Frontend:** [https://terra-rara.vercel.app](https://terra-rara.vercel.app)
- **Backend API:** [https://ifieryarrows-copper-mind.hf.space/api/docs](https://ifieryarrows-copper-mind.hf.space/api/docs)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Vercel)                        │
│                    React + TypeScript + Vite                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Price Chart │  │ Predictions │  │ Market Intelligence Map │  │
│  │  (Recharts) │  │    Card     │  │   (Live yfinance data)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND API (HuggingFace Spaces)              │
│                         FastAPI + Python                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  /api/analysis   │  │ /api/market-prices│  │/api/commentary│  │
│  │  Live Prediction │  │  yfinance Live   │  │  OpenRouter AI│  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ML Pipeline                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │   │
│  │  │  FinBERT    │  │  XGBoost    │  │  Feature Engine │   │   │
│  │  │  Sentiment  │  │   Model     │  │  (60+ features) │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Supabase      │  │   yfinance      │  │  Google News    │  │
│  │   PostgreSQL    │  │   Price Data    │  │  RSS Feeds      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🎯 Live Predictions
- **Real-time model inference** on every request
- Current price from yfinance (15-min delayed)
- XGBoost predicts next-day close
- Sentiment-adjusted confidence bands

### 📰 News Sentiment Analysis
- 16 strategic copper-related news queries
- FinBERT sentiment scoring
- Exponential decay aggregation (τ = 12h)
- Fuzzy duplicate detection

### 🗺️ Market Intelligence Map
- 14 tracked symbols across 5 categories
- Auto-refresh every 30 seconds
- Flash animations on price changes
- Live yfinance data

### 🤖 AI Market Commentary
- OpenRouter API integration
- Daily AI-generated market analysis
- Context-aware insights

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18, TypeScript, Vite, Recharts |
| Backend | FastAPI, Python 3.11, Uvicorn |
| ML Model | XGBoost (regression) |
| Sentiment | FinBERT (transformers) |
| Database | Supabase PostgreSQL |
| Hosting | Vercel (frontend), HuggingFace Spaces (backend) |
| AI Commentary | OpenRouter (mimo-v2-flash) |

---

## 📈 Tracked Symbols

```python
yfinance_symbols = [
    # Core Indicators
    "HG=F",      # Copper Futures (target)
    "DX-Y.NYB",  # US Dollar Index
    "CL=F",      # Crude Oil
    
    # ETFs
    "FXI",       # China Large-Cap ETF
    "COPX",      # Global Copper Miners
    "COPJ",      # Junior Copper Miners
    
    # Titans
    "BHP", "FCX", "SCCO", "RIO",
    
    # Regional
    "TECK", "IVN.TO", "2899.HK",
    
    # Juniors
    "LUN.TO"
]
```

---

## 🚀 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analysis` | GET | Live prediction with current price |
| `/api/history` | GET | Historical price & sentiment data |
| `/api/market-prices` | GET | Live prices for all symbols |
| `/api/commentary` | GET | AI-generated market analysis |
| `/api/health` | GET | System health check |
| `/api/pipeline/trigger` | POST | Trigger data pipeline |

### Pipeline Parameters
```bash
# Full pipeline (fetch + train)
POST /api/pipeline/trigger?fetch_data=true&train_model=true

# Quick update (no training)
POST /api/pipeline/trigger?fetch_data=true&train_model=false

# Just refresh snapshot
POST /api/pipeline/trigger?fetch_data=false&train_model=false
```

---

## 🧠 ML Model Details

### XGBoost Parameters
```python
params = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 5,
    "reg_alpha": 0.5,      # L1 regularization
    "reg_lambda": 2.0,     # L2 regularization
}
```

### Feature Engineering
- **60+ features** per prediction
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands
- Cross-asset correlations
- Sentiment aggregation
- Lagged returns (1d, 5d, 10d, 20d)

---

## 🔐 Environment Variables

```env
# Database
DATABASE_URL=postgresql://...

# News API (optional)
NEWSAPI_KEY=your_key

# AI Commentary
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=xiaomi/mimo-v2-flash:free

# Scheduler
SCHEDULER_ENABLED=true
SCHEDULE_TIME=09:00
TZ=Europe/Istanbul
```

---

## 📦 Project Structure

```
copper-mind/
├── backend/
│   └── app/
│       ├── main.py           # FastAPI app & endpoints
│       ├── ai_engine.py      # XGBoost training
│       ├── inference.py      # Live predictions
│       ├── features.py       # Feature engineering
│       ├── data_manager.py   # Data ingestion
│       ├── sentiment.py      # FinBERT scoring
│       ├── commentary.py     # AI commentary
│       ├── models.py         # SQLAlchemy models
│       └── settings.py       # Configuration
├── frontend/
│   └── src/
│       ├── App.tsx           # Main dashboard
│       ├── api.ts            # API client
│       ├── types.ts          # TypeScript types
│       └── components/
│           └── MarketMap.tsx # Live market grid
├── data/
│   └── models/               # Trained model files
└── README.md
```

---

## 🔄 Data Flow

1. **Pipeline Trigger** → Fetch news + prices
2. **Sentiment Scoring** → FinBERT analyzes articles
3. **Feature Generation** → 60+ technical features
4. **Model Training** → XGBoost learns patterns
5. **Live Prediction** → Real-time inference on request
6. **AI Commentary** → OpenRouter generates insights

---

## 📊 Frontend Display

### Prediction Card
```
Tomorrow's Prediction
━━━━━━━━━━━━━━━━━━━━
$5.99

🐂 +1.67% expected
Data: Fri Jan 10 → Predicting: Mon Jan 13
```

### Sentiment-Adjusted Returns
```javascript
// Sentiment index: -1 (bearish) to +1 (bullish)
sentimentNorm = (sentiment_index + 1) / 2;  // 0 to 1

// Adjust prediction display
if (isBullish) {
  adjustedReturn = baseBullish * sentimentNorm;
} else {
  adjustedReturn = baseBearish * (1 - sentimentNorm);
}
```

---

## 🛠️ Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

---

## 📝 Recent Updates (Jan 2026)

- ✅ Live yfinance price on every request
- ✅ Real-time model prediction (no stale cache)
- ✅ Market Map with 30s auto-refresh
- ✅ Flash animations on price changes
- ✅ Sentiment-adjusted prediction display
- ✅ AI commentary via OpenRouter
- ✅ XGBoost tuning for reduced overfitting

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built with ❤️ for copper market intelligence**
