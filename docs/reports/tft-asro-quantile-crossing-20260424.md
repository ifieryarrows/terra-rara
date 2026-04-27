# TFT-ASRO Quantile Crossing & Yanıltıcı Tahmin Hatası — Müdahale Raporu

| Alan                   | Değer                                                                                                   |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| **Rapor Tarihi** | 24 Nisan 2026                                                                                            |
| **Rapor No**     | TFT-REG-2026-003                                                                                         |
| **Proje**        | CopperMind — Bakır Vadeli İşlem Tahmin Platformu                                                     |
| **Bileşen**     | `deep_learning/models/tft_copper.py` — `format_prediction()` + ASRO Loss                                |
| **Bağlamlar**   | [REG-2026-002](./tft-asro-directional-accuracy-fix-20260415.md) · [IMP-2026-001](./tft-asro-sprint1-kapsamli-iyilestirme-20260420.md) |
| **Hazırlayan**  | AI Engineering Team                                                                                      |
| **Durum**        | 🔴 Açık — Üretim modeli yanıltıcı tahmin gösteriyor                                                   |
| **Öncelik**     | P0 — Kullanıcıya yanlış sinyal iletiliyor                                                               |
| **Endpoint**     | `GET /api/analysis/tft/HG=F`                                                                            |

---

## 1. Yönetici Özeti

24 Nisan 2026 tarihinde `/api/analysis/tft/HG=F` endpoint'inin döndürdüğü tahmin verilerinin denetimi sırasında üretim TFT-ASRO modelinin **matematiksel olarak imkansız quantile dağılımları** ürettiği tespit edildi. Modelin medyan tahmini (q0.50 = +%4.59) kendi 90. yüzdeliğinden (q0.90 = +%0.57) yüksek, 96% güven bandının ise tamamen dışında. Bu durum frontend'de BULLISH sinyal, +%25 haftalık hedef fiyat ve MEDIUM risk olarak gösterilmektedir — **kullanıcılar yanlış bilgiyle işlem kararı alabilir**.

Kök neden, İt.4 eğitiminde (20 Nisan 2026) ASRO loss fonksiyonunun `lambda_quantile=0.25` ayarıyla medyan quantile head'ini agresif bir yönsel optimizasyona zorlarken diğer quantile head'lerinin kalibre kalmasına neden olmasıdır. Quantile crossing cezası bulunmadığı için model bu çelişkiyi öğrenmeye devam etmiştir.

---

## 2. Tespit Edilen Sorunlar

### 2.1 Quantile Sıralama İhlali (KRİTİK)

Quantile değerleri tanım gereği monoton artmalıdır: q0.02 < q0.10 < q0.25 < q0.50 < q0.75 < q0.90 < q0.98.

**Üretim modeli çıktısı (24 Nisan 2026, 00:08 UTC):**

| Quantile | Değer (return) | Beklenen Sıra | Durum |
|----------|---------------|---------------|-------|
| q0.02 | +0.024% | EN DÜŞÜK olmalı | ❌ q0.10'dan yüksek |
| q0.10 | −2.15% | — | ❌ q0.02'den düşük |
| q0.25 | −1.40% | — | ✅ q0.10'dan yüksek |
| q0.50 | **+4.59%** | — | ❌ q0.75, q0.90, q0.98'den yüksek |
| q0.75 | +0.50% | — | ❌ Medyandan düşük |
| q0.90 | +0.57% | — | ❌ Medyandan düşük |
| q0.98 | +1.71% | EN YÜKSEK olmalı | ❌ Medyandan düşük |

```
Doğru dağılım (monoton):
q02 ──── q10 ──── q25 ──── q50 ──── q75 ──── q90 ──── q98
 ↓        ↓        ↓        ↓        ↓        ↓        ↓
düşük ─────────────────────────────────────────── yüksek

Üretim modeli çıktısı (BOZUK):
q10 ──── q25 ──── q02 ──── q75 ── q90 ── q98 ─────── q50
 ↓        ↓        ↓        ↓       ↓      ↓           ↓
−2.2%  −1.4%   +0.02%  +0.50% +0.57% +1.71%      +4.59%
                                                    ↑ MEDYAN
                                              medyan dağılımın
                                              en ucunda — imkansız
```

### 2.2 Medyan Kendi Güven Bandının Dışında (KRİTİK)

| Metrik | Değer |
|--------|-------|
| Medyan fiyat (q0.50) | **$6.401** |
| 96% güven bandı (q0.02–q0.98) | **$6.109 – $6.217** |

Model eşzamanlı olarak:
- "En olası fiyat $6.40" diyor
- "Fiyatın $6.22'yi geçme olasılığı %2" diyor

> [!CAUTION]
> Bu iki ifade birlikte doğru olamaz. Kullanıcı güven bandına baksa "dar ve düşük volatilite" görür; medyana baksa "+%4.59 günlük return" görür. İkisi tamamen çelişkili.

### 2.3 Gerçekçi Olmayan Tahmin Büyüklüğü (YÜKSEK)

| Gün | Günlük Return | Kümülatif | Fiyat |
|-----|--------------|-----------|-------|
| T+1 | +4.59% | +4.59% | $6.40 |
| T+2 | +4.66% | +9.47% | $6.70 |
| T+3 | +4.82% | +14.74% | $7.02 |
| T+4 | +4.19% | +19.56% | $7.32 |
| T+5 | +4.66% | **+25.12%** | **$7.66** |

- Bakır günlük σ ≈ %2.4
- Tek bir +%4.59 return ≈ 1.9σ → olası ama nadir
- Üst üste 5 gün +%4-5 → kümülatif olasılık < %0.003
- `ANOMALY_DAILY_RET = 0.12` eşiği tetiklenmemiş (%4.59 < %12)

### 2.4 Kullanıcıya İletilen Yanıltıcı Sinyaller

| Sinyal | Gösterilen | Gerçek Durum |
|--------|-----------|-------------|
| `direction` | **BULLISH** | Bozuk q0.50'ye dayalı — güvenilmez |
| `weekly_trend` | **BULLISH** (+%25.1) | 5 ardışık +2σ hareket → gerçekçi değil |
| `risk_level` | **MEDIUM** | Quantile'lar non-monoton → risk hesaplanamaz |
| `confidence_band_96` | $6.11–$6.22 | Medyanı bile içermiyor — anlamsız |

---

## 3. Kök Neden Analizi

### 3.1 ASRO Loss Ağırlık Dengesizliği

İt.4 (20 Nisan) eğitiminde Optuna'nın seçtiği `lambda_quantile = 0.25`:

```python
# tft_copper.py L110-113 — ASROPFLoss.loss()
w_directional = 1.0 - self.lambda_quantile        # = 0.75
calibration = q_loss + self.lambda_vol * (vol_loss + amplitude_loss)
directional = sharpe_loss + self.lambda_madl * madl_loss
return self.lambda_quantile * calibration + w_directional * directional
#      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           0.25 × kalibrasyon                  0.75 × yönsel optimizasyon
```

**Sonuç:** Loss bütçesinin %75'i `sharpe_loss + madl_loss` üzerinden **sadece `median_pred`** (q0.50 head) üzerinde çalışıyor. Diğer 6 quantile head (q0.02, q0.10, q0.25, q0.75, q0.90, q0.98) sadece %25 ağırlıklı pinball loss ile eğitiliyor.

Medyan head, pinball loss'tan bağımsız olarak Sharpe + MADL ile "agresif yönsel predictor" olmayı öğrenirken, diğer head'ler standart quantile regression yapmaya devam ediyor → **quantile crossing**.

### 3.2 Quantile Crossing Cezası Yok

```python
# losses.py — AdaptiveSharpeRatioLoss.forward()
# MEVCUT DURUM: crossing kontrolü veya cezası YOK

# Olması gereken:
# for i in range(n_quantiles - 1):
#     crossing_penalty += relu(y_pred[..., i] - y_pred[..., i+1]).mean()
# total_loss += lambda_crossing * crossing_penalty
```

Pytorch-forecasting'in standart `QuantileLoss`'u da crossing penalty uygulamaz — bu sorun küçük model + agresif directional loss kombinasyonunda ortaya çıkar.

### 3.3 `format_prediction()` Sorunu Gizliyor

```python
# tft_copper.py L342-346
# Spread'ler medyana göre hesaplanıyor:
spread_q10 = float(pred[0, 1]) - raw_med_0    # = -2.15% - 4.59% = -6.74%
spread_q90 = float(pred[0, -2]) - raw_med_0   # = 0.57% - 4.59% = -4.02%

# Fiyat hesabı:
price_q90 = cum_price_med * (1 + spread_q90)   # = $6.40 × (1 - 0.0402) = $6.14
```

Spread'ler medyana göre alındığı için q90 fiyatı ($6.14) q50 fiyatından ($6.40) düşük çıkıyor. Fonksiyon **raw quantile sırasını kontrol etmiyor** — çapraz quantile'ları olduğu gibi geçiriyor.

---

## 4. Etki Analizi

### 4.1 Etkilenen Bileşenler

```
format_prediction()  →  predictor.predict()  →  generate_tft_analysis()
       ↓                                              ↓
  [quantiles bozuk]                          [direction: BULLISH]
       ↓                                    [weekly_trend: BULLISH]
       ↓                                    [risk_level: MEDIUM]
       ↓                                              ↓
  /api/analysis/tft/HG=F  ←──────────────── API response
       ↓
  OverviewPage (TFT kartı)  +  ensemble_directional_vote()
       ↓                              ↓
  [kullanıcı görüyor:           [XGBoost ile voting:
   "+4.59% günlük",             tft_return = 0.0459
   "$7.66 haftalık hedef"]      → BULLISH → 0.6× ağırlık]
```

### 4.2 Risk Değerlendirmesi

| Risk | Açıklama | Seviye |
|------|---------|--------|
| **Kullanıcı yanıltma** | +%25 haftalık hedef fiyat gerçekçi değil | 🔴 KRİTİK |
| **Güven kaybı** | Güven bandı medyanı içermiyor | 🔴 KRİTİK |
| **Ensemble bozulması** | TFT return +%4.59 → voting'de ağırlıklı BULLISH bias | 🟡 YÜKSEK |
| **Veri tutarlılığı** | Quantile sırası bozuk → türetilmiş metriklerin tamamı tutarsız | 🟡 YÜKSEK |

---

## 5. Çözüm Planı

### 5.1 Faz 1: Acil — `format_prediction()` Monotonicity Enforcement (POST-HOC)

| Alan | Değer |
|------|-------|
| **Hedef** | Kullanıcıya iletilen quantile'ların her zaman monoton olmasını garanti et |
| **Dosya** | `deep_learning/models/tft_copper.py` — `format_prediction()` |
| **Yöntem** | Her gün için raw quantile vektörünü `np.sort()` ile sırala |
| **Ek** | `quantile_crossing_detected` flag'i response'a ekle |
| **Etki** | Yönsel sinyal (direction, weekly_trend) **düzeltilmiş medyana** göre hesaplanacak |
| **Risk** | Sort edilen medyan ile raw medyan farklı olacak — raw'ı `raw_predicted_return_median` field'ında koru |

```python
# Önerilen değişiklik — format_prediction() başında
for d in range(n_days):
    day_quantiles = pred[d, :]               # (n_quantiles,)
    if not np.all(np.diff(day_quantiles) >= 0):
        crossing_detected = True
        pred[d, :] = np.sort(day_quantiles)  # monotonicity enforcement
```

**Önemli:** Bu, modelin kendisini düzeltmez — sadece çıktıyı güvenli hale getirir. "Bandaj" çözümüdür.

### 5.2 Faz 2: Kısa Vadeli — ASRO Loss'a Quantile Crossing Penalty

| Alan | Değer |
|------|-------|
| **Hedef** | Eğitim sırasında quantile crossing'i cezalandır |
| **Dosyalar** | `deep_learning/models/losses.py`, `deep_learning/models/tft_copper.py` |
| **Yöntem** | Ardışık quantile çiftleri arasındaki "geriye geçişi" penalize et |
| **Config** | `ASROConfig.lambda_crossing: float = 1.0` |

```python
# Önerilen — losses.py
def _quantile_crossing_penalty(y_pred: torch.Tensor) -> torch.Tensor:
    """
    Penalise non-monotonic quantile predictions.
    y_pred shape: (..., n_quantiles)
    """
    diffs = y_pred[..., 1:] - y_pred[..., :-1]  # should all be >= 0
    violations = torch.relu(-diffs)               # positive where crossing
    return violations.mean()
```

```python
# ASROPFLoss.loss() içine ekleme
crossing_loss = _quantile_crossing_penalty(y_pred)
calibration = q_loss + self.lambda_vol * (vol_loss + amplitude_loss) \
              + self.lambda_crossing * crossing_loss
```

### 5.3 Faz 3: Orta Vadeli — `lambda_quantile` Alt Sınır Güvencesi

| Alan | Değer |
|------|-------|
| **Hedef** | Optuna'nın `lambda_quantile`'ı 0.25'in altına çekmesini engelle |
| **Dosya** | `deep_learning/training/hyperopt.py` |
| **Değişiklik** | `lambda_quantile` arama aralığı `[0.20, 0.40]` → `[0.35, 0.50]` |

Gerekçe: `lambda_quantile < 0.30` olduğunda quantile kalibrasyon bileşeni toplam loss'un %30'undan azına düşüyor ve yönsel optimizasyon head'leri birbirinden koparıyor. Alt sınır 0.35 ile en az %35 kalibrasyon ağırlığı garanti edilir.

### 5.4 Faz 4: İzleme — Anomaly Detection Eşiği İyileştirme

| Alan | Değer |
|------|-------|
| **Hedef** | +%4.59 gibi mantıksız ama teknik olarak `ANOMALY_DAILY_RET = 0.12` altında kalan tahminleri yakala |
| **Dosya** | `deep_learning/models/tft_copper.py` — `format_prediction()` |
| **Değişiklik** | Anomaly algılamaya quantile coherence kontrolü ekle |

```python
# Mevcut (sadece büyüklük kontrolü)
anomaly_detected = abs(raw_median_0) > ANOMALY_DAILY_RET  # 12%

# Önerilen (büyüklük + coherence)
quantile_coherent = np.all(np.diff(pred[0, :]) >= 0)
median_in_iqr = pred[0, 1] <= pred[0, median_idx] <= pred[0, -2]
anomaly_detected = (
    abs(raw_median_0) > ANOMALY_DAILY_RET
    or not quantile_coherent
    or not median_in_iqr
)
```

---

## 6. Uygulama Öncelik Sırası

```
Faz 1 (ACİL — bugün)        Faz 2 (eğitim öncesi)      Faz 3 (eğitim)       Faz 4 (izleme)
┌──────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐  ┌──────────────────┐
│ format_prediction()  │    │ ASRO Loss'a         │    │ Optuna           │  │ anomaly_detected │
│ np.sort() post-hoc   │    │ crossing penalty    │    │ lambda_quantile  │  │ quantile         │
│ + flag               │───▶│ lambda_crossing=1.0 │───▶│ aralık [0.35,   │──▶│ coherence check  │
│                      │    │                     │    │ 0.50]            │  │                  │
│ Kullanıcı hemen      │    │ Model doğru         │    │ Kalibrasyon      │  │ Continuous       │
│ güvenli çıktı görür │    │ öğrensin            │    │ korunur          │  │ monitoring       │
└──────────────────────┘    └─────────────────────┘    └──────────────────┘  └──────────────────┘
    Commit + Push              Config + Loss code         hyperopt.py          format_prediction
    → Vercel redeploy          → Yeniden eğitim           → Yeniden eğitim     → Logları izle
```

---

## 7. Başarı Kriterleri

### 7.1 Faz 1 Sonrası (Acil)

| Kriter | Eşik |
|--------|------|
| Quantile'lar monoton mu? | ✅ `q[i] <= q[i+1]` tüm i için |
| Medyan güven bandı içinde mi? | ✅ `q02 <= q50 <= q98` |
| `quantile_crossing_detected` flag | Response'da mevcut |
| Mevcut metriklerde regresyon yok | DA, Sharpe, VR değişmez (sadece output formatting) |

### 7.2 Faz 2+3 Sonrası (Yeniden Eğitim)

| Metrik | Mevcut (İt.4) | Hedef | Rollback Eşiği |
|--------|--------------|-------|----------------|
| **DA** | 52.31% | ≥ 52% (koru) | < 49% |
| **Sharpe** | +0.5285 | ≥ 0.30 (koru) | < −0.30 |
| **VR** | 0.6725 | 0.70–1.20 | < 0.50 veya > 1.80 |
| **Quantile Crossing Rate** | ~%100 (her tahmin) | ≤ %5 | > %20 |
| **|q50 − q50_sorted|** | ~0.04 (4pp) | ≤ 0.005 | > 0.01 |

---

## 8. İlgili Dosya Haritası

```
Etkilenen Dosyalar (Faz 1–4)
├── backend/deep_learning/
│   ├── models/
│   │   ├── tft_copper.py        ← format_prediction() [Faz 1, 4]
│   │   │                          ASROPFLoss.loss() [Faz 2]
│   │   └── losses.py            ← _quantile_crossing_penalty() [Faz 2]
│   ├── config.py                ← ASROConfig.lambda_crossing [Faz 2]
│   │                              ASROConfig.lambda_quantile [Faz 3]
│   ├── training/
│   │   └── hyperopt.py          ← lambda_quantile aralığı [Faz 3]
│   └── inference/
│       └── predictor.py         ← generate_tft_analysis() [sinyal düzeltme]
└── docs/reports/
    └── tft-asro-quantile-crossing-20260424.md  ← bu rapor
```

---

*Rapor oluşturma: 24 Nisan 2026*
*Durum: Açık — Faz 1 uygulanmayı bekliyor*
