# Terra Rara UI Yorum Duzeltmeleri (5 Madde)

| Alan | Deger |
| --- | --- |
| Rapor Tarihi | 4 Mayis 2026 |
| Rapor No | TERRA-UI-2026-005 |
| Proje | CopperMind |
| Bilesen | `backend/app/main.py`, `backend/tests/test_api.py`, `frontend/src/api.ts`, `frontend/src/hooks/useNews.ts`, `frontend/src/features/news/NewsIntelligencePanel.tsx`, `frontend/src/pages/OverviewPage.tsx` |
| Durum | Uygulandi |

---

## 1. Problem Ozeti

PR yorumlarinda 5 UI sorunu isaretlendi:

1. News panel ust sayaclarinin `min relevance` ve diger filtrelerle senkron olmamasi.
2. Market Drivers satirlarinda etiketlerin `...` ile kesilmesi.
3. Neural Analysis model rozeti (MIMO) gereksizligi.
4. Copper Futures delta miktarinda fazla ondalik.
5. 7D News Sentiment'in sayi odakli ve user-friendly olmamasi.

---

## 2. Uygulanan Cozum

### 2.1 News stats filtre senkronu

- `/api/news/stats` endpoint'i su opsiyonel filtreleri alacak sekilde genisletildi:
  - `label`, `event_type`, `min_relevance`, `channel`, `publisher`, `search`
- Stats cache key'i `since_hours` yerine filtre kombinasyonundan uretilecek tuple yapisina gecirildi.
- Stats sorgusu, `/api/news` feed endpoint'indeki filtreleme mantigi ile hizalandi.
- Frontend tarafinda `fetchNewsStats` ve `useNewsStats` imzalari `NewsFeedFilters` tabanli hale getirildi.
- `NewsIntelligencePanel` artik stats sorgusuna `effectiveFilters` gonderiyor.

### 2.2 Market Drivers okunabilirlik

- Tek satir `truncate` kaldirildi.
- Label metni `whitespace-normal` + `break-words` ile cok satirli gosterime alindi.
- Satir duzeni `items-start` ve `flex-1` ile carpisma/ellipsis azaltacak sekilde guncellendi.

### 2.3 Neural Analysis sadelestirme

- MIMO rozet elementi tamamen kaldirildi.
- `generated_at` zamani bagimsiz sekilde korunmaya devam ediyor.

### 2.4 Copper Futures delta formati

- Miktar alani `toFixed(5)` -> `toFixed(2)` olarak degistirildi.
- Yuzde alani `toFixed(2)%` olarak korundu.

### 2.5 7D News Sentiment user-friendly sunum

- Ana sunum etiket + ikon yapisina cevrildi (Positive/Negative/Balanced).
- Sayisal sentiment degeri ikincil bilgi olarak kucuk monospaced satirda korunuyor.
- Renk tonu sentiment label haritalamasi ile tutarli sekilde surduruldu.

---

## 3. Degisen Dosyalar

```text
backend/app/main.py
backend/tests/test_api.py
frontend/src/api.ts
frontend/src/hooks/useNews.ts
frontend/src/features/news/NewsIntelligencePanel.tsx
frontend/src/pages/OverviewPage.tsx
```

---

## 4. Dogrulama Plani

Calistirilacak kontroller:

```text
py -m pytest backend/tests/test_api.py -q
npm.cmd run build
```

Beklenen sonuc:

- News stats regression testi filtre etkilerini dogrular.
- Frontend build TypeScript/JSX seviyesinde yeni imzalari ve UI degisikliklerini dogrular.
