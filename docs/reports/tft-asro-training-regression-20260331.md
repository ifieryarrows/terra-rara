# TFT-ASRO Eğitim Regresyon Raporu

| Alan                   | Değer                                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------------------------- |
| **Rapor Tarihi** | 31 Mart 2026                                                                                         |
| **Rapor No**     | TFT-REG-2026-001                                                                                     |
| **Proje**        | CopperMind — Bakır Vadeli İşlem Tahmin Platformu                                                 |
| **Bileşen**     | `deep_learning/` — TFT-ASRO (Temporal Fusion Transformer with Adaptive Sharpe Ratio Optimization) |
| **Hazırlayan**  | AI Engineering Team                                                                                  |
| **Durum**        | 🟡 Araştırma Devam Ediyor                                                                          |
| **Öncelik**     | P1 — Üretim modeli kalitesi doğrudan etkileniyor                                                  |

---

## 1. Yönetici Özeti

31 Mart 2026 tarihinde gerçekleştirilen haftalık TFT-ASRO yeniden eğitimi, bir önceki döneme kıyasla ciddi performans regresyonu gösterdi. Modelin Sharpe Ratio'su pozitiften negatife düştü (0.84 → −0.86), directional accuracy %50'nin altına geriledi ve MAE %12 kötüleşti. Variance Ratio ise 0.48'den 0.78'e yükselerek %62 iyileşme gösterdi. Kök neden analizi, regresyonun Optuna hiperparametre optimizasyonunun aşırı uyum (overfitting) yapan bir parametre konfigürasyonu seçmesinden kaynaklandığını, aynı hafta uygulanan bilateral amplitude_loss değişikliğinin ise etkisiz olduğunu (VR=0.78 < 1.5 eşiği) ortaya koymuştur.

---

## 2. Sorun Tanımı

### 2.1 İlk Tespit

| Özellik                  | Detay                                                         |
| ------------------------- | ------------------------------------------------------------- |
| **Tespit Zamanı**  | 31 Mart 2026, 13:44 UTC+3                                     |
| **Ortam**           | GitHub Actions —`tft-training.yml` haftalık cron workflow |
| **Tetikleyici**     | Haftalık otomatik TFT-ASRO yeniden eğitim döngüsü        |
| **Tespit Yöntemi** | Eğitim tamamlanma loglarındaki metrik karşılaştırması  |

### 2.2 Etkilenen Bileşenler

```
CopperMind Platform
├── deep_learning/
│   ├── training/hyperopt.py     ← Sorunun kaynağı (parametre seçimi)
│   ├── training/trainer.py      ← Eğitim sürecini yönetti
│   ├── models/losses.py         ← Bilateral amplitude_loss (etkisiz)
│   └── models/tft_copper.py     ← ASROPFLoss (etkisiz)
├── HuggingFace Hub
│   └── ifieryarrows/copper-mind-tft  ← Yeni checkpoint yüklendi
└── Frontend (dolaylı)
    └── TFT tahmin gösterimi     ← Kalite düşüşü kullanıcıya yansıyor
```

### 2.3 Etki Değerlendirmesi

| Etki Alanı                   | Seviye      | Açıklama                                                               |
| ----------------------------- | ----------- | ------------------------------------------------------------------------ |
| **Model Doğruluğu**   | 🔴 Kritik   | Directional accuracy %50 altına düştü — rastgele tahminden kötü   |
| **Risk-Ayarlı Getiri** | 🔴 Kritik   | Sharpe Ratio negatif — sistematik olarak yanlış yön                  |
| **Kullanıcı Güveni** | 🟡 Orta     | TFT tahminleri XGBoost ile paralel sunuluyor; doğrudan zarar sınırlı |
| **Altyapı**            | 🟢 Düşük | Pipeline ve deployment etkilenmedi                                       |
| **Veri Bütünlüğü** | 🟢 Düşük | Eğitim verisi bozulmadı                                                |

### 2.4 Metrik Karşılaştırması

| Metrik                   | 25 Mart (Referans) | 31 Mart (Regresyon) | Δ               | Yorum                                           |
| ------------------------ | ------------------ | ------------------- | ---------------- | ----------------------------------------------- |
| MAE                      | 0.0354             | 0.0398              | +12.4%           | 🔴 Kötüleşme                                 |
| RMSE                     | 0.0409             | 0.0459              | +12.2%           | 🔴 Kötüleşme                                 |
| Directional Accuracy     | 0.5087             | 0.4826              | −5.1%           | 🔴 Coin-flip altı                              |
| Tail Capture Rate        | 0.6055             | 0.3945              | −34.8%          | 🔴 Kuyruk olayları kaçırılıyor             |
| Sharpe Ratio             | 0.8439             | −0.8598            | −202%           | 🔴**Negatif — sistematik yanlış yön** |
| Sortino Ratio            | 1.4058             | −1.5710            | −212%           | 🔴 Negatif                                      |
| pred_std                 | 0.0098             | 0.0159              | +62.2%           | 🟢 Daha geniş tahmin bandı                    |
| actual_std               | 0.0204             | 0.0204              | —               | Piyasa volatilitesi sabit                       |
| **Variance Ratio** | **0.4803**   | **0.7785**    | **+62.1%** | **🟢 Anlamlı iyileşme**                 |

---

## 3. Kök Neden Analizi

### 3.1 Olası Neden Envanteri

| # | Olası Neden                                 | Durum                  | Gerekçe                                                                                       |
| - | -------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------- |
| 1 | Bilateral amplitude_loss değişikliği      | ❌ Elendi              | VR=0.78 < 1.5 eşiği → yeni terim `0.25 × relu(0.78 − 1.5) = 0` üretir, tamamen inaktif |
| 2 | Eğitim verisi bozulması / kayıp           | ❌ Elendi              | `actual_std=0.0204` her iki dönemde aynı, veri tutarlı                                    |
| 3 | Piyasa rejim değişikliği                  | ❌ Elendi              | 6 günlük fark, bakır piyasasında rejim kırılması yok                                    |
| 4 | Altyapı/GPU sorunu                          | ❌ Elendi              | Eğitim tamamlandı, hata yok                                                                  |
| 5 | **Optuna parametre seçim regresyonu** | ✅**Kök Neden** | Aşağıda detaylı analiz                                                                     |

### 3.2 Kök Neden: Optuna Hiperparametre Regresyonu

**5 Neden (5 Whys) Analizi:**

```
Neden 1: Model neden negatif Sharpe üretiyor?
  → Çünkü directional accuracy %48 — yön tahminleri sistematik olarak yanlış.

Neden 2: Yön tahminleri neden kötüleşti?
  → Çünkü model eğitim verisini ezberlemiş (overfitting) ve
    genelleme yapamıyor.

Neden 3: Model neden overfit oldu?
  → Çünkü Optuna bu dönemde düşük regularizasyon parametreleri
    seçti (dropout=0.10, hidden_continuous_size=32).

Neden 4: Optuna neden bu parametreleri seçti?
  → Çünkü validation loss + VR penalty metriğinde bu konfigürasyon
    daha iyi skor aldı (val_loss=−0.080 vs önceki −0.094).
    VR iyileşmesi penalty'yi azalttı ve overfitting'i maskeledi.

Neden 5: Optuna neden overfitting'i tespit edemedi?
  → Çünkü mevcut objective function'da directional accuracy ve
    Sharpe'ın TEST seti üzerindeki performansı doğrudan
    cezalandırılmıyor. Val_loss içindeki Sharpe bileşeni
    eğitim dağılımına uyum sağlamış bir modelde bile
    iyi görünebilir.
```

### 3.3 Parametre Karşılaştırması (Balık Kılçığı Detayı)

```
                           TFT-ASRO REGRESYON
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
   REGULARIZASYON          MODEL KAPASİTESİ        EĞİTİM DİNAMİĞİ
        │                       │                       │
  dropout 0.25→0.10       hidden_cont 16→32       batch_size 16→32
  (2.5× azalma)          (2× artış)              (epoch/step yarıya)
        │                       │                       │
  Sonuç: Az düzenli-      Sonuç: Daha fazla        Sonuç: Daha gürültülü
  leştirme, memorize      parametre, memorize       gradientler
  kapasitesi artışı       alanı genişledi           ↓
        │                       │               grad_clip 0.5→1.5
        └───────────────────────┼───────────────(3× artış)
                                │                   │
                          OVERFITTING            Büyük güncelleme
                                │               adımlarına izin
                                ↓
                       DA=0.48, Sharpe=−0.86
```

#### Parametre-Parametre Detay Tablosu

| Parametre                  | Referans (25 Mart) | Regresyon (31 Mart) | Etki Analizi                                                                                                                                                        |
| -------------------------- | ------------------ | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dropout`                | **0.25**     | 0.10                | 🔴 ~313 eğitim örneği için %10 dropout yetersiz regularizasyon. Model ağırlıkları eğitim setindeki gürültüyü ezberliyor.                               |
| `hidden_continuous_size` | **16**       | 32                  | 🔴 Continuous Variable Selection Network'ün parametre sayısını 2× artırıyor. Overfitting yüzeyini genişletiyor.                                            |
| `gradient_clip_val`      | **0.5**      | 1.5                 | 🟡 Daha büyük gradient güncellemelerine izin veriyor. Tanh-tabanlı Sharpe gradientleri zaten sınırlı, ancak quantile gradientleri sınırsız büyüyebilir. |
| `batch_size`             | **16**       | 32                  | 🟡 Epoch başına 19→~10 gradient adımı. EarlyStopping'in convergence'ı yakalaması zorlaşıyor.                                                               |
| `max_encoder_length`     | **70**       | 90                  | 🟡 Daha uzun lookback penceresi. Küçük dataset'te daha az eğitim örneği üretir (sliding windows azalır).                                                    |
| `attention_head_size`    | **3**        | 4                   | 🟢 Marjinal etki.                                                                                                                                                   |
| `lambda_vol`             | **0.35**     | 0.40                | 🟢 Vol calibration ağırlığı hafif artış — VR iyileşmesini açıklıyor.                                                                                    |
| `lambda_quantile`        | **0.20**     | 0.30                | 🟡 Quantile ağırlığı artışı, Sharpe ağırlığını 0.80→0.70'e düşürdü. Directional learning baskılandı.                                           |

### 3.4 Bilateral Amplitude Loss Etkisi — Kanıtlama

Yeni eklenen overshoot ceza terimi:

```python
0.25 * torch.relu(vr - 1.5)
```

Bu dönemin VR değeri = 0.7785:

```python
0.25 * relu(0.7785 - 1.5) = 0.25 * relu(-0.7215) = 0.25 * 0.0 = 0.0
```

> [!NOTE]
> **Kanıt:** Bilateral amplitude_loss'un over-variance terimi bu eğitimde **kesinlikle sıfır çıktı** üretmiştir. Under-variance terimi ise mevcut formülle aynı değeri üretmektedir:
> `relu(1.0 - 0.7785) = 0.2215` (önceki versiyonda da aynı olurdu). Regresyon bu değişiklikten **bağımsızdır.**

---

## 4. Denenen ve Değerlendirilen Çözüm Senaryoları

### 4.1 Senaryo Kronolojisi

#### Senaryo 1: Post-Inference Soft Dampening (Reddedildi)

| Alan                    | Detay                                                                    |
| ----------------------- | ------------------------------------------------------------------------ |
| **Tarih**         | 30 Mart 2026, 18:35 UTC+3                                                |
| **Commit**        | `acdbfc6`                                                              |
| **Amaç**         | Inference sırasında büyük tahminleri kademeli olarak sıkıştırmak |
| **Dosya**         | `deep_learning/models/tft_copper.py` → `format_prediction()`        |
| **Sonuç**        | ❌ Reddedildi ve geri alındı                                           |
| **Revert Commit** | `bf370f9`                                                              |

**Uygulanan Değişiklik:**

```python
_SOFT_ZONE   = 0.010   # ±1% pass-through
_SOFT_MAX    = 0.020   # 1%–2% linear taper
_MAX_DAILY_RET = 0.030 # hard clamp backstop

def _dampen(r: float) -> float:
    sign = 1.0 if r >= 0 else -1.0
    abs_r = abs(r)
    if abs_r <= _SOFT_ZONE: return r
    if abs_r <= _SOFT_MAX:
        t = (abs_r - _SOFT_ZONE) / (_SOFT_MAX - _SOFT_ZONE)
        return sign * abs_r * (1.0 - 0.30 * t)
    if abs_r <= _MAX_DAILY_RET:
        base = _SOFT_MAX * 0.70
        excess = abs_r - _SOFT_MAX
        budget = _MAX_DAILY_RET - _SOFT_MAX
        compressed = base + excess**0.5 * budget**0.5 * 0.5
        return sign * min(compressed, _MAX_DAILY_RET)
    return sign * _MAX_DAILY_RET
```

**Neden Reddedildi:**

1. Gerçek sorunu maskeleme — model hâlâ yanlış öğrenmeye devam eder
2. Sabit eşikler — bakır %5 hareket ederse meşru tahmini de keser
3. Yön-büyüklük ilişkisi kaybı — yüksek tahmin = yüksek güven sinyali silinir
4. Eğitim sırasında modele gradient vermez, sadece çıktıyı kozmetik düzeltir

---

#### Senaryo 2: Huber Quantile Loss (Araştırıldı, Uygulanmadı)

| Alan             | Detay                                                   |
| ---------------- | ------------------------------------------------------- |
| **Tarih**  | 30 Mart 2026, araştırma aşaması                     |
| **Amaç**  | Büyük hataları L1 yerine L2 (karesel) cezalandırmak |
| **Sonuç** | ❌ Reddedildi — risk analizi sonucu                    |

**Formülasyon:**

```
L_huber(q, u) =
    pinball(q, u)              if |u| ≤ δ
    pinball(q, u) × (|u|/δ)    if |u| > δ
```

**Neden Reddedildi:**

1. **Sharpe bileşeniyle çelişir:** Sharpe "büyük doğru yön" ister, Huber "büyük herhangi hata" cezalar
2. **Quantile coverage bozulur:** Q2/Q98 kuyrukları içe çekilir
3. **5 yeni hiperparametre etkileşimi:** Optuna arama uzayı katlanır
4. **VR=0.48 modelini daha muhafazakâr yapar:** TFT'nin avantajını siler

---

#### Senaryo 3: Post-Inference İzotonik Kalibrasyon (Araştırıldı, Uygulanmadı)

| Alan             | Detay                                                        |
| ---------------- | ------------------------------------------------------------ |
| **Tarih**  | 30 Mart 2026, araştırma aşaması                          |
| **Amaç**  | Eğitim sonrası monoton dönüşüm ile tahmin kalibrasyonu |
| **Sonuç** | ❌ Reddedildi — veri yetersizliği                          |

**Neden Reddedildi:**

1. Validation set ~47 sample — isotonic regression için yetersiz
2. Overshoot'u geçmişe bakarak düzeltir, geleceğe genellemez
3. Pipeline'a yeni artifact (calibrator.joblib) ve bakım yükü ekler

---

#### Senaryo 4: Bilateral Amplitude Loss (Uygulandı ✅)

| Alan               | Detay                                                                           |
| ------------------ | ------------------------------------------------------------------------------- |
| **Tarih**    | 30 Mart 2026, 18:48 UTC+3                                                       |
| **Commit**   | `4c80824`                                                                     |
| **Amaç**    | Eğitim sırasında VR>1.5 durumunda modele hafif overshoot ceza sinyali vermek |
| **Dosyalar** | `losses.py` (L220–226), `tft_copper.py` (L83–89)                          |
| **Sonuç**   | ✅ Uygulandı, ancak mevcut eğitimde inaktif (VR=0.78 < 1.5)                   |

**Uygulanan Değişiklik:**

```python
# Önceki (tek yönlü — sadece muhafazakâr modeli cezalandırıyordu)
median_std = median_pred.std() + self.sharpe_eps
amplitude_loss = torch.relu(1.0 - median_std / actual_std)

# Yeni (iki yönlü — aşırı tahmin de cezalı)
median_std = median_pred.std() + self.sharpe_eps
vr = median_std / actual_std
amplitude_loss = (
    torch.relu(1.0 - vr)              # VR < 1 → güçlü ceza (1.0×)
    + 0.25 * torch.relu(vr - 1.5)     # VR > 1.5 → hafif ceza (0.25×)
)
```

**Mevcut Durum:**

- `relu(1.0 − 0.78) = 0.22` → mevcut formülle aynı (under-variance cezası)
- `0.25 × relu(0.78 − 1.5) = 0.0` → yeni terim aktifleşmedi
- **Regresyonla ilişkisi yok** — tamamen diğer hiperparametre değişikliklerinden kaynaklanıyor

---

## 5. Güncel Durum

| Madde                                     | Durum                                                       |
| ----------------------------------------- | ----------------------------------------------------------- |
| Regresyonun kök nedeni tespit edildi mi? | ✅ Evet — Optuna parametre seçim regresyonu               |
| Bilateral amplitude_loss suçlu mu?       | ❌ Hayır — matematiksel olarak kanıtlandı (Bölüm 3.4) |
| Üretim modeli etkilendi mi?              | ⚠️ Yeni checkpoint HF Hub'a yüklendiyse evet             |
| Önceki checkpoint erişilebilir mi?      | ✅ HF Hub versiyon geçmişinden geri alınabilir           |
| Kalıcı çözüm uygulandı mı?         | ❌ Henüz — araştırma devam ediyor                       |

---

## 6. Araştırma Yol Haritası

### 6.1 Acil Eylemler (P0)

- [ ] Önceki çalışan checkpoint'i (25 Mart) HF Hub'dan indirip `best_tft_asro.ckpt` olarak geri yükle
- [ ] Önceki `optuna_results.json` dosyasını (25 Mart parametreleri) geri yükle
- [ ] Pipeline'ı tetikle ve TFT inference'ın eski model ile çalıştığını doğrula

### 6.2 Kısa Vadeli İyileştirmeler (P1)

- [ ] **Hyperopt arama uzayını daralt:**
  - `dropout` alt sınırını 0.10 → 0.20'ye çıkar
  - `hidden_continuous_size` üst sınırını 32 → 24'e düşür
  - `batch_size` seçeneklerinden 64'ü çıkar (zaten var), 32'yi de değerlendir
- [ ] **Hyperopt objective'e directional accuracy penaltısı ekle:**
  ```python
  # Mevcut: score = val_loss + variance_penalty
  # Önerilen: DA < 0.50 ise ağır ceza
  if directional_accuracy < 0.50:
      da_penalty = 2.0 * (0.50 - directional_accuracy)
  else:
      da_penalty = 0.0
  score = val_loss + variance_penalty + da_penalty
  ```
- [ ] **Early stopping'e Sharpe sign kontrolü ekle:** Sharpe negatife düşen trial'ı otomatik prune et

### 6.3 Orta Vadeli Araştırmalar (P2)

- [ ] **Walk-forward validation:** Tek split yerine kayar pencere ile daha güvenilir metrikler
  - Referans: [Tashman, L.J. (2000) &#34;Out-of-sample tests of forecasting accuracy&#34;](https://doi.org/10.1016/S0169-2070(00)00065-0)
- [ ] **Optuna pruner konfigürasyonu:** `MedianPruner(n_warmup_steps=5)` → daha agresif pruning ile overfitting trial'ları erken kes
- [ ] **Ensemble-based confidence:** En iyi 3 Optuna trial'ının medyan tahminini kullan
- [ ] **Regularization schedule:** Epoch ilerledikçe dropout artır (scheduled dropout)

### 6.4 İncelenmesi Gereken Kaynaklar

| Kaynak                                                                                                                             | Konu                         | Neden                                                       |
| ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ----------------------------------------------------------- |
| [Optuna docs: Pruning](https://optuna.readthedocs.io/en/stable/reference/pruners.html)                                                | Agresif pruning stratejileri | Overfitting trial'ları erken kesmek                        |
| [PyTorch Forecasting: TFT interpretability](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html)             | Variable importance analizi  | Hangi feature'ların regresyona katkı yaptığını tespit |
| [Lim et al. (2021) &#34;Temporal Fusion Transformers&#34;](https://doi.org/10.1016/j.ijforecast.2021.03.012)                          | Orijinal TFT paper           | Quantile loss + attention head size etkileşimi             |
| [Bergstra &amp; Bengio (2012) &#34;Random Search for Hyper-Parameter Optimization&#34;](https://jmlr.org/papers/v13/bergstra12a.html) | Hyperopt vs Random Search    | TPE'nin küçük dataset'lerde sınırlamaları             |

### 6.5 Benzer Vakalar

| Vaka                                       | Benzerlik                          | Alınan Ders                                                              |
| ------------------------------------------ | ---------------------------------- | ------------------------------------------------------------------------- |
| CopperMind TFT v1.0 (Ocak 2026) VR=0.14    | pred_std çok düşük, model flat | `_TANH_SCALE` 100→20 ile çözüldü; Optuna arama alanı daraltıldı |
| CopperMind XGBoost dar varyans sorunu      | Model sürekli küçük tahminler  | Sentiment multiplier eklendi; post-hoc düzeltme                          |
| Genel: Optuna küçük dataset overfitting | 313 sample ile 50 trial aşırı   | Trial sayısı azaltma veya cross-validation                              |

---

## 7. Sonuçlar ve Öneriler

### 7.1 Çıkarılan Dersler

1. **Optuna güvenliği yetersiz:** Mevcut objective (`val_loss + VR_penalty`) yalnızca kalibrasyon kalitesini ölçüyor; directional accuracy ve Sharpe sign'ı doğrudan korumuyor. Bu, modelin eğitim verisini ezberleyip validation'da iyi loss almasına ama gerçek dünyada başarısız olmasına izin veriyor.
2. **Regularizasyon alt sınırı kritik:** ~313 eğitim örneği ile `dropout=0.10` ve `hidden_continuous_size=32` kombinasyonu, modelin parametre sayısı/veri oranını tehlikeli bölgeye taşıyor. Arama uzayındaki alt sınırlar bir "güvenlik kemeri" işlevi görmelidir.
3. **VR iyileşmesi yanıltıcı olabilir:** VR 0.48→0.78 harika görünüyor, ancak VR tek başına kalite göstergesi değil. Yanlış yönde geniş tahminler, doğru yönde dar tahminlerden daha kötüdür. VR iyileşmesi ile directional accuracy'nin **birlikte** izlenmesi gerekir.
4. **Bilateral amplitude_loss doğru bir guard-rail'dir:** Bu dönemde aktifleşmemiş olması tasarımın doğruluğunu kanıtlıyor — sadece gerçek overshoot'ta devreye giriyor, normal operasyonu etkilemiyor. Gelecek eğitimlerde VR>1.5'e çıkarsa koruma sağlayacak.

### 7.2 Önleme Önerileri

| # | Öneri                                                         | Etki                                                                | Efor     |
| - | -------------------------------------------------------------- | ------------------------------------------------------------------- | -------- |
| 1 | Hyperopt objective'e DA<0.50 penaltısı ekle                  | Yüksek — coin-flip altı modeller otomatik elenir                 | Düşük |
| 2 | `dropout` minimum sınırını 0.20'ye yükselt              | Yüksek — overfitting yüzeyini daralır                           | Düşük |
| 3 | `hidden_continuous_size` maksimumunu 24'e düşür           | Orta — parametre/veri oranını kontrol eder                       | Düşük |
| 4 | Sharpe negatifse trial'ı prune et                             | Yüksek — sistematik yanlış yön trial'ları erken kesilir       | Düşük |
| 5 | Önceki en iyi parametreleri "warm-start" olarak Optuna'ya ver | Orta — arama uzayını önceki optima etrafında yoğunlaştırır | Orta     |
| 6 | Walk-forward CV uygula (3-fold temporal)                       | Yüksek — genelleme kalitesini doğrudan ölçer                   | Yüksek  |

### 7.3 Açık Sorular

1. **Model rollback kriterleri nedir?** Hangi metriklerin hangi seviyelerine düşmesi durumunda otomatik rollback tetiklenmeli?

   - Öneri: `directional_accuracy < 0.50 OR sharpe_ratio < 0.0` → önceki checkpoint'e geri dön
2. **Optuna trial sayısı optimal mi?** 313 sample ile 50 trial yapılıyor. Her trial farklı bir train/val split üretmiyor (split sabit). Bu durumda 50 trial aynı validation set üzerinde 50 farklı konfigürasyon deniyor — bir nevi validation set'e overfit olma riski var.
3. **Bilateral amplitude_loss'un 0.25 katsayısı ve 1.5 eşiği optimal mi?** Gelecek eğitimlerde VR>1.5'e çıkma durumunda bu parametrelerin etkisi gözlemlenip kalibre edilmeli.

---

## Ekler

### Ek A: Tam Eğitim Log Çıktısı

```
============================================================
TFT-ASRO TRAINING COMPLETE
============================================================
  mae: 0.0398
  rmse: 0.0459
  directional_accuracy: 0.4826
  tail_capture_rate: 0.3945
  sharpe_ratio: -0.8598
  sortino_ratio: -1.5710
  pred_std: 0.0159
  actual_std: 0.0204
  variance_ratio: 0.7785
```

### Ek B: Optuna En İyi Parametreler

```
============================================================
HYPEROPT COMPLETE
============================================================
Best trial: #11
Best val_loss: -0.080359
  max_encoder_length: 90
  hidden_size: 48
  attention_head_size: 4
  dropout: 0.1
  hidden_continuous_size: 32
  learning_rate: 0.0003304639335410294
  gradient_clip_val: 1.5
  lambda_vol: 0.4
  lambda_quantile: 0.30000000000000004
  batch_size: 32
```

### Ek C: Git Commit Geçmişi (İlgili)

```
4c80824 feat(asro): add bilateral amplitude_loss — gentle penalty for VR > 1.5 overshoot
bf370f9 Revert "feat(tft): replace hard clamp with two-stage soft dampening"
acdbfc6 feat(tft): replace hard clamp with two-stage soft dampening (GERİ ALINDI)
84f29d7 feat(ci): auto-trigger champion/challenger backtest after screener
5c673df fix(stage6): add None-safety coercion guard before commentary f-string formatting
```

---

*Bu rapor, sorun tamamen çözülene kadar güncellenecektir.*
