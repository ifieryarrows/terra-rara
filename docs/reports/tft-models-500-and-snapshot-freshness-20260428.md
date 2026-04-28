# TFT Models 500 ve Snapshot Freshness Duzeltmesi

| Alan | Deger |
| --- | --- |
| Rapor Tarihi | 28 Nisan 2026 |
| Rapor No | TFT-API-2026-006 |
| Proje | CopperMind |
| Bilesen | `backend/app/main.py` (`/api/models/tft/summary`, `/api/analysis/tft/{symbol}`) |
| Durum | Kod duzeltmesi uygulandi |

---

## 1. Problem Ozeti

Kullanici tarafinda iki semptom goruldu:

1. `Models` page 500 hatasi (`/api/models/tft/summary`)
2. Yeni model egitilmis olmasina ragmen `Close Date` alaninda eski tarih (`2026-04-24`) gorunmesi

Egitim loglari 27 Nisan 2026 kosusunda yeni checkpoint'in olustugunu ve kalite metriklerinin guncel oldugunu gosteriyordu.

---

## 2. Kok Neden

### 2.1 Models page 500

`/api/models/tft/summary` endpoint'i `quality_gate.metrics` icinde `None` degerleri de donuyordu:

- `tail_capture`
- `quantile_crossing_rate`
- `median_sort_gap_max`

Sema `Dict[str, float]` oldugu icin bu `None` degerler response validation'da 500'e dusurebiliyordu.

### 2.2 Stale Close Date

Frontend varsayilan olarak `source=snapshot` kullaniyor.  
Snapshot kaydi yeni model train edilmeden once olusmussa (veya reference close tarihi stale ise), API eski payload'i servis etmeye devam ediyor.

Bu nedenle yeni model train edilmis olsa bile UI'da eski `reference_price_date` gosterilebiliyordu.

---

## 3. Uygulanan Cozum

### 3.1 Summary endpoint sanitize

`/api/models/tft/summary` icin:

- `config_json` ve `metrics_json` guvenli parse edildi
- `metrics` icin yalnizca finite numeric degerler response'a alindi
- `quality_gate.metrics` icinde `None` alanlar response'a eklenmedi

Sonuc: response modeliyle uyumsuz `None` degerleri 500'e neden olmuyor.

### 3.2 Snapshot bypass kurali

`/api/analysis/tft/{symbol}` (`source=snapshot`) akisina stale kontrol eklendi:

- `trained_at > snapshot.generated_at` ise snapshot bypass
- `reference_price_date` >= 3 gun stale ise snapshot bypass

Bu kosullarda endpoint otomatik olarak live inference'a dusup guncel payload donuyor.

---

## 4. Degisen Dosyalar

```text
backend/app/main.py
docs/reports/tft-models-500-and-snapshot-freshness-20260428.md
```

---

## 5. Dogrulama

Calistirilan kontroller:

```text
py -m py_compile backend/app/main.py
py -m pytest backend/tests/test_quality_gate.py -q
```

Sonuc:

```text
py_compile passed
2 passed
```

---

## 6. T+1 Yuzde Hesaplama Notu

UI'daki `T+1 Forecast` yuzdesi backend payload'indaki:

```text
prediction.daily_forecasts[0].daily_return
```

degerinden gelir ve ekranda `*100` ile gosterilir.

Bu deger `format_prediction()` icinde medyan quantile'in (gerekiyorsa anomaly bound sonrasinda) T+1 return'u olarak uretilir.  
Ayni satirin fiyat karsiligi da `daily_forecasts[0].price_median` oldugu icin yuzde ve fiyat ayni kaynaktan gelir.
