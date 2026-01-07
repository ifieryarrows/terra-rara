# =============================================================================
# 🔶 CopperMind: Copper Price Prediction Model Training
# =============================================================================
# Bu script Kaggle'da çalıştırılmak üzere tasarlanmıştır.
#
# Adımlar:
# 1. Fiyat verisi çekme (yfinance)
# 2. GERÇEK haber çekme (NewsAPI + Google News RSS)
# 3. FinBERT ile sentiment scoring
# 4. Feature engineering
# 5. XGBoost eğitimi
# 6. Model artefaktlarını kaydetme
#
# Çıktılar: /kaggle/working/ altında model dosyaları oluşur
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus
import requests

import yfinance as yf
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# RSS için feedparser (Kaggle'da yüklü, yoksa: pip install feedparser)
try:
    import feedparser
except ImportError:
    print("⚠️ feedparser yükleniyor...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'feedparser', '-q'])
    import feedparser

# Tarih parse için dateutil
try:
    from dateutil import parser as dateutil_parser
except ImportError:
    print("⚠️ python-dateutil yükleniyor...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'python-dateutil', '-q'])
    from dateutil import parser as dateutil_parser

print("✅ Temel kütüphaneler yüklendi")

# =============================================================================
# 🔑 API KEY - KAGGLE SECRETS
# =============================================================================
# Kaggle Secrets'a ekle:
#   Label: NEWS_API_KEY
#   Value: (newsapi.org'dan aldığın key)
# =============================================================================

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
NEWS_API_KEY = secrets.get_secret("NEWS_API_KEY")

if NEWS_API_KEY:
    print("✅ NEWS_API_KEY alındı!")
else:
    print("⚠️ NEWS_API_KEY bulunamadı, sadece RSS kullanılacak")

# =============================================================================

# =============================================================================
# AYARLAR
# =============================================================================

TARGET_SYMBOL = "HG=F"  # Copper Futures
SYMBOLS = ["HG=F", "DX-Y.NYB", "CL=F", "FXI"]
LOOKBACK_DAYS = 365
VALIDATION_DAYS = 30
EARLY_STOPPING_ROUNDS = 10

# Çıktı dizini
OUTPUT_DIR = Path("/kaggle/working") if os.path.exists("/kaggle") else Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"📁 Output directory: {OUTPUT_DIR}")

# =============================================================================
# 1. HABER VERİSİ ÇEKME (GERÇEK!)
# =============================================================================

def fetch_newsapi(query: str = "copper", days: int = 30) -> list:
    """NewsAPI'den gerçek haberler çek."""
    print(f"  🌐 NewsAPI sorgulanıyor: '{query}'")
    
    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=min(days, 30))
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": NEWS_API_KEY,
        "q": query,
        "language": "en",
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
        "sortBy": "publishedAt",
        "pageSize": 100,
    }
    
    print(f"  📅 Tarih aralığı: {from_date.date()} → {to_date.date()}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"  📡 HTTP Status: {response.status_code}")
        
        data = response.json()
        print(f"  📦 API Status: {data.get('status')}, Total Results: {data.get('totalResults', 0)}")
        
        if data.get("status") != "ok":
            print(f"  ❌ API Error: {data.get('message', data.get('code', 'Unknown'))}")
            return []
        
        articles = []
        for item in data.get("articles", []):
            try:
                pub_str = item.get("publishedAt", "")
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00")) if pub_str else datetime.now(timezone.utc)
                
                articles.append({
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "published_at": pub_dt,
                    "source": item.get("source", {}).get("name", "NewsAPI"),
                })
            except:
                continue
        
        print(f"  ✅ NewsAPI: {len(articles)} haber alındı")
        return articles
    
    except Exception as e:
        print(f"  ❌ Hata: {type(e).__name__}: {e}")
        return []


def fetch_google_news_rss(query: str = "copper price") -> list:
    """Google News RSS'den haberler çek (API key gerektirmez!)."""
    print(f"  🌐 Google News RSS'den haberler çekiliyor...")
    
    encoded_query = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
    
    try:
        feed = feedparser.parse(url)
        
        articles = []
        for entry in feed.entries[:100]:
            try:
                title = entry.get("title", "")
                
                # Google News başlığından kaynağı ayır
                source = "Google News"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    if len(parts) == 2:
                        title, source = parts
                
                # Tarih parse
                pub_str = entry.get("published", "")
                try:
                    pub_dt = dateutil_parser.parse(pub_str)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                except:
                    pub_dt = datetime.now(timezone.utc)
                
                articles.append({
                    "title": title.strip(),
                    "description": entry.get("summary", ""),
                    "published_at": pub_dt,
                    "source": source,
                })
            except:
                continue
        
        print(f"  ✅ Google News RSS: {len(articles)} haber çekildi")
        return articles
    
    except Exception as e:
        print(f"  ❌ RSS hatası: {e}")
        return []


def fetch_all_news() -> pd.DataFrame:
    """Tüm kaynaklardan haberleri çek ve birleştir."""
    print("\n📰 HABER VERİSİ ÇEKİLİYOR...")
    
    all_articles = []
    
    # NewsAPI
    newsapi_articles = fetch_newsapi(query="copper")
    all_articles.extend(newsapi_articles)
    
    # Google News RSS (her zaman çalışır)
    rss_articles = fetch_google_news_rss("copper price futures")
    all_articles.extend(rss_articles)
    
    # Ek RSS sorguları
    rss_articles2 = fetch_google_news_rss("copper mining supply")
    all_articles.extend(rss_articles2)
    
    if not all_articles:
        print("  ⚠️ Hiç haber bulunamadı, simüle veri kullanılacak")
        return None
    
    # DataFrame'e çevir
    df = pd.DataFrame(all_articles)
    
    # Dedup: aynı başlığı tekrar ekleme
    df['title_hash'] = df['title'].apply(lambda x: hashlib.md5(x.lower().encode()).hexdigest()[:16])
    df = df.drop_duplicates(subset=['title_hash'])
    df = df.drop(columns=['title_hash'])
    
    print(f"\n✅ TOPLAM: {len(df)} benzersiz haber toplandı!")
    print(f"   Tarih aralığı: {df['published_at'].min().date()} → {df['published_at'].max().date()}")
    
    return df


# =============================================================================
# 2. FİYAT VERİSİ ÇEKME
# =============================================================================

def fetch_prices(symbols: list, lookback_days: int = 365) -> dict:
    """Fetch OHLCV data for multiple symbols."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    data = {}
    for symbol in symbols:
        print(f"  📊 Fetching {symbol}...", end=" ")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d"
            )
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                data[symbol] = df
                print(f"{len(df)} bars")
            else:
                print("No data!")
        except Exception as e:
            print(f"Error: {e}")
    
    return data

print("\n🔄 Fetching price data...")
price_data = fetch_prices(SYMBOLS, LOOKBACK_DAYS)
print(f"\n✅ Loaded {len(price_data)} symbols")

for symbol, df in price_data.items():
    print(f"   {symbol}: {df.index.min().date()} to {df.index.max().date()} ({len(df)} bars)")

# =============================================================================
# 3. GERÇEK HABERLERİ ÇEK
# =============================================================================

news_df = fetch_all_news()

# =============================================================================
# 4. FİNBERT İLE SENTIMENT SCORING
# =============================================================================

def load_finbert():
    """FinBERT modelini yükle."""
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"  🧠 FinBERT yükleniyor... (Device: {'GPU' if device == 0 else 'CPU'})")
    
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=device,
        max_length=512,
        truncation=True
    )
    print("  ✅ FinBERT hazır!")
    return pipe


def score_with_finbert(pipe, texts: list, batch_size: int = 16) -> list:
    """FinBERT ile metinleri skorla."""
    scores = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [t[:1000] if t else "neutral" for t in batch]
        
        try:
            results = pipe(batch)
            for result in results:
                probs = {r['label'].lower(): r['score'] for r in result}
                score = probs.get('positive', 0.33) - probs.get('negative', 0.33)
                scores.append(score)
        except Exception as e:
            scores.extend([0.0] * len(batch))
        
        if (i + batch_size) % 50 == 0 or i + batch_size >= len(texts):
            print(f"    Skorlandı: {min(i+batch_size, len(texts))}/{len(texts)}")
    
    return scores


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Günlük sentiment agregasyonu (recency-weighted)."""
    news_df['date'] = pd.to_datetime(news_df['published_at']).dt.normalize()
    
    daily_data = []
    for date, group in news_df.groupby('date'):
        # Gün içi saate göre ağırlıklandırma
        hours = (group['published_at'] - date).dt.total_seconds() / 3600
        weights = np.exp(hours / 12)  # Geç haberler daha ağırlıklı
        weights = weights / weights.sum()
        
        weighted_sentiment = (group['sentiment_score'] * weights).sum()
        
        daily_data.append({
            'date': date,
            'sentiment_index': weighted_sentiment,
            'news_count': len(group)
        })
    
    df = pd.DataFrame(daily_data)
    df = df.set_index('date')
    df.index = df.index.tz_localize(None)
    
    return df


def generate_simulated_sentiment(price_df: pd.DataFrame, noise_level: float = 0.3) -> pd.DataFrame:
    """Simüle sentiment (haber yoksa fallback)."""
    returns = price_df['Close'].pct_change()
    base_sentiment = np.tanh(returns * 20)
    
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, len(returns))
    sentiment = np.clip(base_sentiment + noise, -1, 1)
    
    volatility = returns.abs().rolling(5).mean().fillna(0.01)
    news_count = (volatility * 500 + np.random.randint(5, 20, len(returns))).astype(int)
    
    df = pd.DataFrame({
        'date': price_df.index,
        'sentiment_index': sentiment.values,
        'news_count': news_count.values
    })
    df = df.set_index('date')
    df = df.fillna(0)
    
    return df


# SENTIMENT HESAPLAMA
print("\n🧠 SENTIMENT ANALİZİ...")

if news_df is not None and len(news_df) > 0:
    # GERÇEK HABERLER VAR - FinBERT kullan!
    print(f"  📰 {len(news_df)} gerçek haber bulundu, FinBERT ile skorlanıyor...")
    
    finbert = load_finbert()
    
    # Başlık + açıklama birleştir
    texts = (news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')).tolist()
    
    print(f"  🔄 {len(texts)} metin skorlanıyor...")
    news_df['sentiment_score'] = score_with_finbert(finbert, texts)
    
    # Günlük agregasyon
    daily_sentiment = aggregate_daily_sentiment(news_df)
    
    print(f"\n✅ GERÇEK SENTIMENT: {len(daily_sentiment)} gün")
    print(f"   Ortalama skor: {daily_sentiment['sentiment_index'].mean():.4f}")
    print(f"   Aralık: [{daily_sentiment['sentiment_index'].min():.3f}, {daily_sentiment['sentiment_index'].max():.3f}]")
    
    USE_REAL_SENTIMENT = True
else:
    # Haber yok - simüle veri kullan
    print("  ⚠️ Gerçek haber bulunamadı, simüle sentiment kullanılıyor...")
    target_prices = price_data[TARGET_SYMBOL]
    daily_sentiment = generate_simulated_sentiment(target_prices)
    print(f"  ✅ Simüle sentiment: {len(daily_sentiment)} gün")
    
    USE_REAL_SENTIMENT = False

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def compute_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    return prices.pct_change(periods)

def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=1).mean()

def compute_ema(prices: pd.Series, span: int) -> pd.Series:
    return prices.ewm(span=span, adjust=False).mean()

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_volatility(returns: pd.Series, window: int = 10) -> pd.Series:
    return returns.rolling(window=window, min_periods=1).std()

def generate_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Generate technical features for a symbol."""
    features = pd.DataFrame(index=df.index)
    prefix = f"{symbol.replace('=', '_').replace('-', '_').replace('.', '_')}_"
    
    close = df['Close']
    ret1 = compute_returns(close)
    
    features[f"{prefix}ret1"] = ret1
    for lag in [1, 2, 3, 5]:
        features[f"{prefix}lag_ret1_{lag}"] = ret1.shift(lag)
    
    for w in [5, 10, 20]:
        features[f"{prefix}SMA_{w}"] = compute_sma(close, w)
        features[f"{prefix}EMA_{w}"] = compute_ema(close, w)
    
    features[f"{prefix}RSI_14"] = compute_rsi(close)
    features[f"{prefix}vol_10"] = compute_volatility(ret1)
    
    sma_20 = compute_sma(close, 20)
    features[f"{prefix}price_sma_ratio"] = close / sma_20.replace(0, np.nan)
    
    return features

print("\n🔄 Building feature matrix...")

target_df = price_data[TARGET_SYMBOL].copy()
target_index = target_df.index

all_features = pd.DataFrame(index=target_index)

for symbol, df in price_data.items():
    print(f"  Processing {symbol}...")
    aligned = df.reindex(target_index).ffill(limit=3)
    features = generate_features(aligned, symbol)
    all_features = all_features.join(features, how='left')

# Sentiment features
print("  Adding sentiment features...")
sentiment_aligned = daily_sentiment.reindex(target_index).ffill(limit=3)
all_features['sentiment__index'] = sentiment_aligned['sentiment_index'].fillna(0)
all_features['sentiment__news_count'] = sentiment_aligned['news_count'].fillna(0)

print(f"\n✅ Feature matrix: {all_features.shape}")

# =============================================================================
# 4. TARGET DEĞİŞKENİ
# =============================================================================

# Next-day return (veri sızıntısını önler)
target_ret = compute_returns(target_df['Close'])
y = target_ret.shift(-1)
y.name = 'target_ret'

X = all_features.copy()

# NaN temizliği
valid_mask = ~y.isna()
X = X[valid_mask]
y = y[valid_mask]
X = X.dropna()
y = y.loc[X.index]

print(f"\n✅ Final dataset: {len(X)} samples, {X.shape[1]} features")
print(f"   Date range: {X.index.min().date()} to {X.index.max().date()}")

feature_names = X.columns.tolist()

# =============================================================================
# 5. XGBOOST EĞİTİMİ
# =============================================================================

# Time-based split
split_date = X.index.max() - timedelta(days=VALIDATION_DAYS)

train_mask = X.index <= split_date
val_mask = X.index > split_date

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"\n📊 Train set: {len(X_train)} samples")
print(f"📊 Val set:   {len(X_val)} samples")

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 42,
}

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

evals = [(dtrain, "train"), (dval, "val")]

print("\n🔄 Training XGBoost model...\n")

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=evals,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=50
)

print(f"\n✅ Training complete! Best iteration: {model.best_iteration}")

# Evaluate
y_pred_train = model.predict(dtrain)
y_pred_val = model.predict(dval)

train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
val_mae = mean_absolute_error(y_val, y_pred_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

print("\n📈 Model Performance:")
print(f"   Training   - MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}")
print(f"   Validation - MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")

# Feature importance
importance = model.get_score(importance_type="gain")
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
total_importance = sum(v for _, v in sorted_importance)

normalized_importance = [
    {"feature": k, "importance": v / total_importance}
    for k, v in sorted_importance
]

print("\n🏆 Top 10 Feature Influencers:")
print("-" * 50)
for i, item in enumerate(normalized_importance[:10], 1):
    print(f"  {i:2d}. {item['feature']:30s} {item['importance']*100:5.2f}%")

# =============================================================================
# 6. MODEL KAYDETME
# =============================================================================

symbol_safe = TARGET_SYMBOL.replace("=", "_").replace("-", "_")

# Model
model_path = OUTPUT_DIR / f"xgb_{symbol_safe}_latest.json"
model.save_model(str(model_path))
print(f"\n✅ Model saved: {model_path}")

# Metrics
metrics = {
    "target_symbol": TARGET_SYMBOL,
    "trained_at": datetime.now().isoformat(),
    "train_samples": len(X_train),
    "val_samples": len(X_val),
    "train_mae": float(train_mae),
    "train_rmse": float(train_rmse),
    "val_mae": float(val_mae),
    "val_rmse": float(val_rmse),
    "best_iteration": model.best_iteration,
    "feature_count": len(feature_names),
}

metrics_path = OUTPUT_DIR / f"xgb_{symbol_safe}_latest.metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Metrics saved: {metrics_path}")

# Features
features_path = OUTPUT_DIR / f"xgb_{symbol_safe}_latest.features.json"
with open(features_path, "w") as f:
    json.dump(feature_names, f, indent=2)
print(f"✅ Features saved: {features_path}")

# Importance
importance_path = OUTPUT_DIR / f"xgb_{symbol_safe}_latest.importance.json"
with open(importance_path, "w") as f:
    json.dump(normalized_importance, f, indent=2)
print(f"✅ Importance saved: {importance_path}")

# =============================================================================
# ÖZET
# =============================================================================

print("\n" + "="*60)
print("🎉 EĞİTİM TAMAMLANDI!")
print("="*60)
print(f"\n📊 Model: {TARGET_SYMBOL}")
print(f"📈 Validation MAE: {val_mae:.6f}")
print(f"📈 Validation RMSE: {val_rmse:.6f}")

print(f"\n🧠 Sentiment Verisi: {'✅ GERÇEK (FinBERT)' if USE_REAL_SENTIMENT else '⚠️ SİMÜLE'}")
if USE_REAL_SENTIMENT:
    print(f"   Toplam haber: {len(news_df)}")
    print(f"   Günlük sentiment: {len(daily_sentiment)} gün")

print(f"\n🏆 Top 3 Influencers:")
for item in normalized_importance[:3]:
    print(f"  • {item['feature']}: {item['importance']*100:.1f}%")

print(f"\n📁 Çıktı dosyaları: {OUTPUT_DIR}")
print("\n" + "-"*60)
print("📥 SONRAKI ADIMLAR:")
print("-"*60)
print("1. Kaggle'da: Sağdaki 'Output' sekmesinden 4 dosyayı indir")
print("2. Dosyaları projenin data/models/ klasörüne kopyala")
print("3. Backend API'yi başlat ve dashboard'u aç!")
print("-"*60)

if not NEWS_API_KEY:
    print("\n💡 İPUCU: Kaggle Secrets'a NEWS_API_KEY ekleyerek daha fazla haber çekebilirsin!")
    print("   newsapi.org'dan ücretsiz key al ve Secrets'a ekle.")

