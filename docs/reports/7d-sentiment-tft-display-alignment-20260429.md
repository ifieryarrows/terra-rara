# 7D Sentiment ve TFT Display Hizalama

| Alan | Deger |
| --- | --- |
| Rapor Tarihi | 29 Nisan 2026 |
| Rapor No | NEWS-TFT-UI-2026-001 |
| Proje | CopperMind |
| Bilesen | `frontend/src/pages/OverviewPage.tsx`, `frontend/src/features/news/NewsIntelligencePanel.tsx`, `backend/app/main.py` |
| Durum | Kod duzeltmesi uygulandi |

---

## 1. Problem Ozeti

Frontend ana ekranda sentiment gostergesi XGBoost analiz payload'undaki `analysis.sentiment_index` alanindan besleniyordu. Bu alan haber yuzeyi icin dogrudan 7 gunluk DB-backed haber sentimentini temsil etmiyordu.

News Intelligence panelinde haber penceresi ve istatistik penceresi de farkli davranabiliyordu: feed varsayilani 48 saat, stats varsayilani 24 saat uzerinden calisiyordu.

TFT tarafinda ana kartin T+1 sinyal olarak kalmasi, haftalik bilginin ise `weekly_trend` / T+5 ozet seviyesinde kullanilmasi karari korundu.

---

## 2. Uygulanan Cozum

### 2.1 Overview header sentiment

`OverviewPage` header sentiment degeri artik `useSentimentSummary(7, 6)` sonucundan besleniyor.

- Header label'i `7D News Sentiment` olarak guncellendi.
- Renk secimi `Bullish`, `Bearish`, `Neutral` label'ina baglandi.
- XGBoost model ici sentiment adjustment mantigina dokunulmadi.

### 2.2 News Intelligence 7D pencere

News Intelligence varsayilan penceresi 168 saate alindi.

- Feed varsayilani 7 gun oldu.
- `useNewsStats` aktif feed penceresine baglandi.
- 24h hard-coded tooltip metinleri aktif pencere label'i ile degistirildi.

### 2.3 Sentiment summary API

`/api/sentiment/summary` endpoint'inde `recent_articles` sorgusu artik secilen pencere baslangicina gore filtreleniyor:

```text
NewsRaw.published_at >= window_start
```

`data_freshness` alanina geriye uyumlu opsiyonel metadata eklendi:

- `window_start`
- `window_days`
- `article_count_window`

---

## 3. Degisen Dosyalar

```text
backend/app/main.py
backend/tests/test_api.py
frontend/src/api.ts
frontend/src/features/news/NewsIntelligencePanel.tsx
frontend/src/hooks/useNews.ts
frontend/src/pages/OverviewPage.tsx
```

---

## 4. Dogrulama

Calistirilan kontroller:

```text
py -m py_compile backend/app/main.py backend/tests/test_api.py
py -m pytest backend/tests/test_api.py -q
npm run build
git diff --check
```

Sonuclar:

- `backend/tests/test_api.py -q`: 26 test gecti.
- `frontend` build basarili tamamlandi.
- `git diff --check`: whitespace hatasi yok; yalniz CRLF uyarilari goruldu.

Manuel prod API kontrolleri:

```text
/api/sentiment/summary?days=7&recent_limit=6
/api/news/stats?since_hours=168
/api/analysis/tft/HG=F
```

Prod ortamda endpointler yanit verdi. Yeni `data_freshness.article_count_window` ve `window_days` alanlari deploy sonrasi gorunur olacak.
