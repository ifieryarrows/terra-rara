# CopperMind v1.0.0

Bakır piyasası için **haber duyarlılığı + fiyat tahmini** üreten, tamamen otomatik çalışan ve web arayüzünde anlaşılır biçimde sunan uçtan uca bir sistem.

## Mimari

```
NewsAPI/RSS → Python Otomasyonu → FinBERT → XGBoost → FastAPI → React Dashboard
     ↓              ↓                ↓          ↓          ↓
  Haberler       Fiyatlar      DailySentiment  Tahmin    JSON API
```

## Hızlı Başlangıç

### 1. Ortam Değişkenlerini Ayarla

```bash
cp env.example .env
# .env dosyasını düzenle (en azından NEWSAPI_KEY ekle veya boş bırak RSS için)
```

### 2. Sistemi Başlat

```bash
make reset    # İlk kurulum veya fabrika ayarlarına dönüş
make seed     # Haber ve fiyat verilerini çek
make train    # FinBERT + XGBoost eğitimi
make health   # Sistem durumunu kontrol et
```

### 3. Erişim Noktaları

| Servis | URL |
|--------|-----|
| Dashboard | http://localhost:5173 |
| API Docs (Swagger) | http://localhost:5173/api/docs |
| Health Check | http://localhost:5173/api/health |
| Backend Direct | http://localhost:8000/api/docs |

## Makefile Komutları

| Komut | Açıklama |
|-------|----------|
| `make up` | Servisleri başlat |
| `make down` | Servisleri durdur |
| `make logs` | Container loglarını izle |
| `make reset` | Fabrika ayarlarına dön (veri silinir) |
| `make seed` | Online kaynaklardan veri çek |
| `make seed-csv` | CSV dosyalarından veri yükle |
| `make train` | ML pipeline çalıştır |
| `make health` | Sistem sağlık kontrolü |
| `make inspect` | Veritabanı özeti |
| `make test` | Testleri çalıştır |

## API Endpoint'leri

### GET /api/analysis

Güncel analiz raporu döndürür.

```json
{
  "symbol": "HG=F",
  "current_price": 4.2534,
  "predicted_return": 0.012345,
  "predicted_price": 4.3057,
  "confidence_lower": 4.1892,
  "confidence_upper": 4.4222,
  "sentiment_index": 0.1523,
  "sentiment_label": "Bullish",
  "top_influencers": [
    {
      "feature": "DX-Y.NYB_ret1",
      "importance": 0.23,
      "description": "US Dollar Index Return"
    },
    {
      "feature": "sentiment__index",
      "importance": 0.18,
      "description": "Market Sentiment Index"
    },
    {
      "feature": "HG=F_RSI_14",
      "importance": 0.12,
      "description": "Copper RSI (14)"
    }
  ],
  "data_quality": {
    "news_count_7d": 45,
    "missing_days": 0,
    "coverage_pct": 100
  },
  "generated_at": "2026-01-02T09:00:00Z"
}
```

### GET /api/history

Geçmiş fiyat ve duygu verilerini döndürür. `sentiment_index` değeri `0.0` olabilir (nötr duygu); bu `null`'dan farklıdır.

```json
{
  "symbol": "HG=F",
  "data": [
    {
      "date": "2025-12-01",
      "price": 4.1523,
      "sentiment_index": 0.05,
      "sentiment_news_count": 12
    },
    {
      "date": "2025-12-02",
      "price": 4.1834,
      "sentiment_index": 0.0,
      "sentiment_news_count": 5
    },
    {
      "date": "2025-12-03",
      "price": 4.2101,
      "sentiment_index": -0.12,
      "sentiment_news_count": 8
    }
  ]
}
```

### GET /api/health

Sistem durumu kontrolü.

```json
{
  "status": "healthy",
  "db_type": "sqlite",
  "models_found": 1,
  "pipeline_locked": false,
  "timestamp": "2026-01-02T10:00:00Z",
  "news_count": 1250,
  "price_bars_count": 1460
}
```

## Cold Start (CSV Seed)

NewsAPI geçmiş limitleri nedeniyle ilk kurulumda tarihsel haber verisi eksik olabilir. Bu durumda:

1. Haber CSV'nizi `./data/seed/` klasörüne koyun
2. `make seed-csv` çalıştırın

CSV formatı:
```csv
title,published_at,url,source,description
"Copper prices rise...",2025-06-15T10:00:00Z,https://...,Reuters,"..."
```

## Konfigürasyon

Tüm ayarlar `.env` dosyasından yönetilir. Önemli değişkenler:

- `NEWSAPI_KEY`: NewsAPI anahtarı (boşsa RSS kullanılır)
- `YFINANCE_SYMBOLS`: Takip edilen semboller
- `SCHEDULE_TIME`: Günlük otomasyon saati
- `ANALYSIS_TTL_MINUTES`: Cache süresi

## Mimari Detaylar

### Veri Katmanı
- SQLite (WAL mode) ile eşzamanlı okuma/yazma desteği
- NewsAPI + Google News RSS ile haber toplama
- yfinance ile çoklu sembol fiyat verisi
- Fuzzy deduplication ile RSS gürültü azaltma

### Zeka Katmanı
- FinBERT ile haber başına duygu skoru
- Günlük agregasyon ile Market Sentiment Index
- XGBoost ile next-day return tahmini
- Feature importance ile açıklanabilirlik

### Sunum Katmanı
- FastAPI ile RESTful API
- Snapshot + TTL cache ile stabil yanıt
- Pipeline lock ile race condition koruması
- React + Recharts ile interaktif dashboard
- Nginx reverse proxy ile `/api` yönlendirme

## Lisans

MIT

