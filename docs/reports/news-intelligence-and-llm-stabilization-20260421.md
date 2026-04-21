# News Intelligence (UI + API) & LLM Sentiment Stabilizasyonu — Sonuç Raporu


| Alan             | Değer                                                 |
| ---------------- | ----------------------------------------------------- |
| **Rapor Tarihi** | 21 Nisan 2026                                         |
| **Rapor No**     | NEWS-INTEL-2026-001                                   |
| **Proje**        | CopperMind — Copper Intelligence Platform             |
| **Bileşen**      | `backend/app/` · `backend/worker/` · `frontend/src/`  |
| **Bağlam**       | `news-frontend-and-hybrid-sentiment_45e0d464.plan.md` |
| **Hazırlayan**   | Gökhan Soytürk                                        |
| **Durum**        | ✅ Tamamlandı                                          |
| **Öncelik**      | P1 — Prod kararlılığı + görünürlük                    |


---

## 1. Yönetici Özeti

Bu çalışma iki ana problemi hedefledi:

- **Haberlerin frontend’de görünür ve kullanılabilir hale getirilmesi**: DB’de zaten bulunan `news_raw / news_processed / news_sentiments_v2` verisi için dedicated API endpoint’leri oluşturuldu ve Overview sayfasında sağda **sticky News Intelligence** paneli eklendi (filtre, arama, detay drawer, infinite scroll).
- **LLM tabanlı sentiment skorlamanın kararsızlığının düşürülmesi**: OpenRouter free-tier / provider farklılıkları nedeniyle oluşan JSON parse hataları ve rate-limit “tüm günü kapatan” davranışlar stabilize edildi; LLM kotası daha verimli kullanıldı.

Çalışma sonunda:

- Backend’de `GET /api/news`, `GET /api/news/{processed_id}`, `GET /api/news/stats` endpoint’leri eklendi (in-memory TTL cache ile).
- Frontend’de Overview sağ sidebar’da sticky News paneli eklendi (NewsCard + NewsDetailDrawer).
- LLM scoring hattı, provider uyumsuzluklarına ve kota/rate-limit durumlarına karşı daha dayanıklı hale getirildi.
- Test kapsamı genişletildi ve değişiklikler doğrulandı (backend pytest + frontend tsc).

---

## 2. Kapsam ve Kapsam Dışı

### 2.1 Kapsam

- **News/Sentiment uçtan uca**:
  - Backend: `/api/news`* endpoint’leri + Pydantic contract’ları
  - Frontend: Overview’da sağ sticky panel + data-layer (React Query)
  - LLM kararlılığı: parse/retry/rate-limit ve maliyet/quotaya duyarlı iyileştirmeler

### 2.2 Kapsam Dışı (bilgilendirme)

- TFT tarafında (6.03→6.27, 17-04, `train_model` semantiği) **kod değişikliği yapılmadı**; yalnızca tanı/açıklama plan dokümanında netleştirildi.

---

## 3. Backend Değişiklikleri

### 3.1 LLM Sentiment Kararlılığı (`backend/app/ai_engine.py`)

#### 3.1.1 `response_format` kaldırıldı

Bazı OpenRouter provider’larında `response_format={"type":"json_object"}` ipucunun:

- JSON array yerine “wrapped object” döndürme,
- Array’i reddetme,
- veya parse tutarsızlığı

üretmesi sebebiyle V2 scoring çağrılarında bu ipucu devre dışı bırakıldı. Prompt + post-processing zaten `markdown fence`, `wrapped results`, `preamble` gibi durumları normalize ediyor.

#### 3.1.2 Rate-limit scope model bazlı yapıldı

Önceki davranış: fast model rate-limit olursa, global flag ile **tüm gün LLM kapatılıyordu**.

Yeni davranış:

- Rate-limit state **model bazlı** tutulur (`_rate_limited_by_model`).
- Sadece fast tükenmişse reliable denenebilir (veya tersi).
- `score_batch_with_llm_v2` bundle’ı `rate_limited_fast` / `rate_limited_reliable` bayraklarını döndürür.
- Backward-compat için `rate_limited` yalnızca **ikisi birden** tükenirse `true`.

#### 3.1.3 LLM çağrılarını “boşa gitmeyecek” örneklere odaklama

LLM maliyeti/kotası korumak için:

- **Non-English** veya **çok kısa** cleaned_text içeren haberler (eşik: 80 char) LLM’e gönderilmez,
- Doğrudan FinBERT + rule fallback ile skorlanır.

### 3.2 OpenRouter retry/backoff iyileştirmeleri (`backend/app/openrouter_client.py`)

- 429 durumunda **minimum bekleme** eklendi (30s floor, 300s cap) — tight-retry ile retry haklarını yakmayı önler.
- `X-Ratelimit-`* header’ları (varsa) loglanır; prod gözlemi kolaylaşır.

### 3.3 Konfigürasyon güncellemesi (`backend/app/settings.py`)

- `max_llm_articles_per_run`: **100 → 60**
  - chunk_size=12 ile run başına istek sayısı ve günlük limit tüketimi daha kontrollü hale gelir.
- `openrouter_fallback_models` için dokümantif açıklama güçlendirildi (comma-separated model slug listesi).

### 3.4 News API endpoint’leri (`backend/app/main.py` + `backend/app/schemas.py`)

#### 3.4.1 Yeni endpoint’ler

- `GET /api/news`
  - Parametreler: `limit, offset, since_hours, label, event_type, min_relevance, channel, publisher, search`
  - Default: `since_hours=48`, `limit=20`
  - In-memory cache TTL: 60s
- `GET /api/news/stats`
  - Rolling window default: 24h
  - Label / event_type / channel distribution + top 5 publishers
  - In-memory cache TTL: 120s
- `GET /api/news/{processed_id}`
  - Tek haber detayı (reasoning/finbert vb dahil)

#### 3.4.2 “Channel” vs “Publisher” ayrımı

- **channel**: ingestion kanalı (`NewsRaw.source` → `google_news` / `newsapi`)
- **publisher**: orijinal yayıncı (`raw_payload.source`) — Reuters/Bloomberg/Mining.com vb.

DB backend-agnostic kalmak için publisher extraction/filtreleme endpoint seviyesinde güvenli şekilde ele alındı.

---

## 4. Frontend Değişiklikleri

### 4.1 Tipler ve API client (`frontend/src/types.ts`, `frontend/src/api.ts`)

- Backend contract’ına uygun yeni tipler:
  - `NewsItem`, `NewsSentimentBlock`, `NewsListResponse`, `NewsStatsResponse`, `NewsFeedFilters`
- Yeni API çağrıları:
  - `fetchNews(filters)`, `fetchNewsById(processedId)`, `fetchNewsStats(sinceHours)`

### 4.2 React Query hooks (`frontend/src/hooks/useNews.ts`)

- `useNewsFeed(filters)`: infinite query, 90s polling
- `useNewsStats()`: 120s polling
- `useNewsDetail(id)`: drawer açılınca lazy fetch
- `flattenNewsPages(...)`: UI performansı için yardımcı

### 4.3 UI Bileşenleri (`frontend/src/features/news/`)

- `NewsIntelligencePanel.tsx`
  - Sticky sidebar panel
  - Search + filtre (label, time window, min relevance, channel)
  - Top publisher chip’leri (stats’ten)
  - Infinite scroll sentinel + refresh butonu
- `NewsCard.tsx`
  - Publisher + channel micro-badge (GN/NA)
  - Label chip + event_type
  - Relevance mini-bar + confidence
- `NewsDetailDrawer.tsx`
  - Sağdan açılır drawer
  - FinBERT prob bar’ları + LLM rationale + metrikler
  - Orijinal link (external)

### 4.4 Overview layout (`frontend/src/pages/OverviewPage.tsx`)

- Desktop: `lg:grid-cols-[minmax(0,1fr)_360px]`
  - Sol: mevcut 12-col dashboard bozulmadan
  - Sağ: `lg:sticky lg:top-24` News Intelligence panel
- Mobile: sidebar otomatik olarak alta stack olur

---

## 5. Test ve Doğrulama

### 5.1 Backend testleri

- `backend/tests/test_ai_engine.py`
  - `_clean_json_content` edge-case regresyon testleri (wrapped object, markdown fence, preamble, single object)
  - V2 scoring bundle’ının per-model rate-limit bayraklarını doğrulayan test
- `backend/tests/test_api.py`
  - News schemas şekil testleri + publisher/reasoning extraction helper testleri
- `backend/tests/test_openrouter_client.py`
  - 429 backoff floor (küçük Retry-After) + büyük Retry-After saygısı testleri

Çalıştırma sonucu (lokal):

- Backend: `pytest backend/tests/` → ✅ geçti (ingestion testleri hariç)

### 5.2 Frontend doğrulama

- `npx tsc --noEmit` → ✅ geçti

---

## 6. Operasyonel Notlar (Prod Davranışı)

- News paneli **LLM çağırmaz**; DB’ye yazılmış skorları okur.
- LLM scoring yalnızca worker/pipeline tarafında yeni haberler için devreye girer.
- Rate-limit durumunda:
  - fast model tükenirse reliable model hala çalışabilir,
  - ikisi de tükenirse LLM kısmı pause olur, FinBERT+rule fallback devam eder.

---

## 7. Takip / Önerilen İyileştirmeler

- `/api/news` publisher filtrelemesini DB-native hale getirmek (Postgres JSONB `->>` ile) ve sqlite için degrade path bırakmak (performans).
- Sidebar listesi için gerçek virtualization (`tanstack-virtual`/`react-window`) eklemek (çok yüksek item sayısı).
- Publisher typeahead (stats endpoint genişletmesi ile) ve event_type multi-select chip UI iyileştirmesi.

