# TFT-ASRO Horizon Alignment, Quantile Coherence ve Promotion Safety Düzeltmesi

| Alan | Değer |
| --- | --- |
| **Rapor Tarihi** | 27 Nisan 2026 |
| **Rapor No** | TFT-REG-2026-004 |
| **Proje** | CopperMind — Bakır Vadeli İşlem Tahmin Platformu |
| **Bileşen** | `backend/deep_learning/` + TFT training workflow |
| **Bağlamlar** | [IMP-2026-001](./tft-asro-sprint1-kapsamli-iyilestirme-20260420.md) · [REG-2026-003](./tft-asro-quantile-crossing-20260424.md) |
| **Durum** | ✅ Kod düzeltmeleri uygulandı, yeniden eğitim bekleniyor |
| **Öncelik** | P0/P1 — Yanlış model seçimi ve hatalı checkpoint promote riskini kapatma |

---

## 1. Yönetici Özeti

27 Nisan 2026 02:00 eğitiminde model kalibrasyon tarafında iyi, yön tarafında ise ciddi şekilde başarısız sonuç verdi:

```text
mae=0.0363, rmse=0.0405
directional_accuracy=0.4377
tail_capture_rate=0.3962
sharpe_ratio=-2.4054
sortino_ratio=-4.4435
pred_std=0.0153, actual_std=0.0162, variance_ratio=0.9424
```

Bu tablo, önceki raporlardaki "doğru volatilite, yanlış yön" patolojisinin geri döndüğünü gösterdi. VR neredeyse ideal olmasına rağmen DA ve Sharpe çok kötüydü. Kök neden yüzeyde yalnızca hiperparametre seçimi gibi görünse de kod incelemesi üç yapısal kırılma gösterdi:

1. **Metric horizon hizalama hatası:** T+1 tahminleri, `flatten()` edilmiş 5 günlük target matrisiyle karşılaştırılıyordu.
2. **Quantile coherence eksikliği:** Eğitim loss'unda quantile crossing cezası yoktu; inference tarafı da crossing'i kullanıcıdan önce yakalamıyordu.
3. **Promotion sırası hatası:** Trainer HF Hub upload'ını quality gate'ten önce yapabiliyordu; workflow gate failure durumunda gerçek bir promote blokajı sağlamıyordu.

Bu raporda belgelenen kod değişiklikleri bu üç kök nedeni birlikte kapatır. Amaç tek bir metriği yama ile iyi göstermek değil; model seçim sinyalini, output matematiğini ve production promote akışını aynı anda güvenli hale getirmektir.

---

## 2. Kök Neden Zinciri

### 2.1 Kalibrasyon-yön dengesinin yanlış okunması

27 Nisan'da Optuna şu loss ağırlıklarını seçti:

```text
lambda_vol=0.4
lambda_quantile=0.4
lambda_madl=0.2
```

Bu seçim, 20 Nisan İt.4'e göre kalibrasyon ve volatilite tarafını güçlendirirken MADL'nin efektif yönsel katkısını düşürdü:

```text
İt.4 efektif MADL:       (1 - 0.25) * 0.40 = 0.30
27 Nisan efektif MADL:   (1 - 0.40) * 0.20 = 0.12
```

Sonuç beklenen yönde gerçekleşti: VR ve MAE iyileşti; DA, Sharpe ve Sortino çöktü.

### 2.2 Asıl ölçüm hatası: T+1 vs flatten edilmiş target

Training ve hyperopt değerlendirmesinde prediction tarafı T+1 horizon'a sabitlenmişti:

```python
y_pred_median = pred_np[:, 0, median_idx]
```

Ancak actual tarafı tüm horizon'ları düzleştiriyordu:

```python
y_actual = torch.cat(y_actual_parts).cpu().numpy().flatten()
```

`max_prediction_length=5` olduğu için bu, T+1 tahminlerinin T+1/T+2/T+3/T+4/T+5 karışık target dizisiyle eşleşmesine yol açabilir. Bu durumda DA, Sharpe, Tail Capture ve hyperopt objective sinyali güvenilir değildir.

### 2.3 Quantile crossing borcu

24 Nisan raporu quantile crossing'i P0 olarak açmıştı. Kodda crossing penalty ve post-hoc coherence guard henüz yoktu. Bu nedenle model iki uç arasında salınıyordu:

```text
Directional baskı yüksek  → q50 agresifleşir, quantile crossing çıkar
Calibration baskı yüksek  → quantile/VR düzelir, yön sinyali zayıflar
```

Kalıcı çözüm, yön sinyalini kısmak değil; quantile monotonicity'yi ayrı bir matematiksel constraint olarak loss ve output kontratına eklemektir.

### 2.4 Quality gate gerçek promote kapısı değildi

`trainer.py` içinde HF Hub upload metadata yazıldıktan hemen sonra çalışıyordu. Workflow'da quality gate bunun ardından geliyordu. Dolayısıyla kötü model gate'ten kalsa bile production artifact daha önce değişmiş olabiliyordu.

Bu, "quality gate" kavramını alarm seviyesine indiriyordu; gerçek promote engeli değildi.

---

## 3. Uygulanan Düzeltmeler

### 3.1 Horizon-aligned metric kontratı

Yeni yardımcı fonksiyon:

```python
select_prediction_horizon(values, horizon_idx=0)
```

Etkilenen dosyalar:

```text
backend/deep_learning/training/metrics.py
backend/deep_learning/training/trainer.py
backend/deep_learning/training/hyperopt.py
```

Artık final test metrikleri ve hyperopt fold metrikleri T+1 tahminleri yalnızca T+1 actual değerleriyle karşılaştırır.

### 3.2 Quantile crossing penalty

Yeni loss bileşeni:

```python
quantile_crossing_penalty(y_pred)
```

ASRO calibration bundle artık şunu içerir:

```python
calibration = (
    q_loss
    + lambda_vol * (vol_loss + amplitude_loss)
    + lambda_crossing * crossing_loss
)
```

Yeni config alanı:

```python
ASROConfig.lambda_crossing = 1.0
```

Etkilenen dosyalar:

```text
backend/deep_learning/config.py
backend/deep_learning/models/losses.py
backend/deep_learning/models/tft_copper.py
```

### 3.3 Inference coherence guard

`format_prediction()` artık raw quantile vektörünü kontrol eder. Crossing varsa:

- public `quantiles` alanını monoton sıralanmış değerlerden üretir,
- `raw_quantiles` alanında ham modeli audit için korur,
- `quantile_crossing_detected`, `quantile_crossing_rate`, `median_sort_gap` alanlarını response'a ekler,
- `anomaly_detected=True` döndürür.

Bu, modelin kendisini düzeltmez; ancak kullanıcıya matematiksel olarak imkansız band gitmesini engeller ve sorunu saklamaz.

### 3.4 Hyperopt güvenlik bantları

Arama uzayı bilinen 27 Nisan patolojisine göre sıkılaştırıldı:

```text
lambda_vol:      0.25–0.45 → 0.30–0.45
lambda_quantile: 0.20–0.40 → 0.25–0.40
lambda_madl:     0.10–0.50 → 0.30–0.50
```

Ek olarak hyperopt fold score artık quantile incoherence penalty de içerir:

```text
crossing_penalty = 2.0 * max(0, crossing_rate - 0.05)
median_gap_penalty = 5.0 * max(0, median_gap - 0.005)
```

Fold ortalamasında `crossing_rate > 0.20` veya `median_gap > 0.01` olan trial prune edilir.

### 3.5 Legacy Optuna artifact guard

`run_hyperopt=false` ile eski `optuna_results.json` yeniden kullanılmak istenirse, trainer bilinen riskli ASRO parametrelerini güvenli alt sınırlara çeker:

```text
lambda_vol >= 0.30
0.25 <= lambda_quantile <= 0.40
lambda_madl >= 0.30
```

Bu, 27 Nisan'daki `lambda_madl=0.2` gibi zayıf-yön konfigürasyonunun sessizce tekrar uygulanmasını engeller.

### 3.6 Promote-before-gate hatası kapatıldı

`train_tft_model()` artık default olarak HF Hub upload yapmaz:

```python
train_tft_model(..., upload_to_hub=False)
```

Workflow sırası değiştirildi:

```text
Train → GitHub artifact upload → Quality gate → HF Hub upload
```

Quality gate failure durumunda rollback step `exit 1` ile job'ı durdurur. HF Hub upload yalnızca gate success olduğunda çalışır.

### 3.7 Quality gate genişletildi

Quality gate artık DA, Sharpe, VR yanında opsiyonel olarak şunları da değerlendirir:

```text
tail_capture_rate < 0.35          → FAIL
quantile_crossing_rate > 0.20     → FAIL
median_sort_gap_max > 0.01        → FAIL
```

Metadata dosyası yoksa gate artık pass/skip dönmez; fail döner.

---

## 4. Test Sonuçları

Çalıştırılan komutlar:

```text
py -m py_compile backend/deep_learning/training/metrics.py backend/deep_learning/models/losses.py backend/deep_learning/models/tft_copper.py backend/deep_learning/training/trainer.py backend/deep_learning/training/hyperopt.py backend/app/quality_gate.py backend/scripts/tft_quality_gate.py backend/app/main.py

py -m pytest backend/tests/deep_learning/test_losses.py backend/tests/deep_learning/test_metrics.py backend/tests/deep_learning/test_config.py backend/tests/deep_learning/test_tft_format_prediction.py backend/tests/test_quality_gate.py -q

py -m pytest backend/tests/deep_learning -q
```

Sonuç:

```text
36 passed, 2 warnings
71 passed, 1 skipped, 2 warnings
```

Uyarılar mevcut pytest config ve dateutil deprecation kaynaklıdır; bu değişiklik setinden doğan test hatası yoktur.

---

## 5. Değişen Dosya Haritası

```text
.github/workflows/tft-training.yml
backend/app/main.py
backend/app/quality_gate.py
backend/deep_learning/config.py
backend/deep_learning/models/losses.py
backend/deep_learning/models/tft_copper.py
backend/deep_learning/training/hyperopt.py
backend/deep_learning/training/metrics.py
backend/deep_learning/training/trainer.py
backend/scripts/tft_quality_gate.py
backend/tests/deep_learning/test_config.py
backend/tests/deep_learning/test_losses.py
backend/tests/deep_learning/test_metrics.py
backend/tests/deep_learning/test_tft_format_prediction.py
backend/tests/test_quality_gate.py
frontend/src/types.ts
```

---

## 6. Yeniden Eğitim Kabul Kriterleri

Bir sonraki TFT training koşusunun promote edilebilmesi için minimum eşikler:

| Metrik | Minimum | Not |
| --- | ---: | --- |
| Directional Accuracy | >= 0.49 | Gate minimumu; hedef >= 0.52 |
| Sharpe Ratio | >= -0.30 | Gate minimumu; hedef >= 0.30 |
| Variance Ratio | 0.20–2.50 | Gate minimumu; hedef 0.70–1.20 |
| Tail Capture | >= 0.35 | Gate minimumu; hedef >= 0.50 |
| Quantile Crossing Rate | <= 0.20 | Gate minimumu; hedef <= 0.05 |
| Median Sort Gap Max | <= 0.01 | Gate minimumu; hedef <= 0.005 |

27 Nisan sonucu bu gate'ten geçemez:

```text
DA=0.4377 < 0.49
Sharpe=-2.4054 < -0.30
```

---

## 7. Kalan Riskler

1. Bu değişiklikler uzun eğitim koşusunun metriklerini garanti etmez; modelin gerçekten iyileştiği yeniden eğitimle doğrulanmalıdır.
2. Crossing penalty yeni checkpoint'lerde yapısal çözüm sağlar; eski checkpoint'ler için inference tarafındaki monotonic sorting güvenlik katmanıdır.
3. Hyperopt fold'ları hâlâ küçük veri rejiminde çalışıyor. Horizon alignment bu sinyali temizler, fakat küçük örneklem istatistiksel gürültüyü tamamen ortadan kaldırmaz.
4. HF Hub upload artık gate sonrasına taşındı; ancak manuel `--upload-hub` kullanımı operatör sorumluluğundadır.

---

## 8. Sonuç

Bu müdahale, tek bir metrik yamasından ziyade TFT-ASRO training hattının üç temel güvenlik katmanını düzeltir:

```text
Ölçüm doğruluğu      → T+1 tahmin yalnızca T+1 actual ile ölçülür
Dağılım tutarlılığı  → quantile crossing eğitimde cezalanır, inference'ta flaglenir
Promotion güvenliği  → HF Hub update yalnızca quality gate sonrası yapılır
```

Bu nedenle 27 Nisan'daki başarısız koşunun aynı mekanizma ile tekrar production'a sızması engellenmiştir. Nihai doğrulama için bir sonraki eğitim koşusu bu rapordaki kabul kriterleriyle değerlendirilmelidir.

---

*Son güncelleme: 27 Nisan 2026 — Kod düzeltmeleri ve hedefli testler tamamlandı.*
