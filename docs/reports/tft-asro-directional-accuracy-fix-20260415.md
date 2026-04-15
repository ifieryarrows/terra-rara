# TFT-ASRO Yön Doğruluğu Çıkmazı — Müdahale Raporu

| Alan                   | Değer                                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------------------------- |
| **Rapor Tarihi** | 15 Nisan 2026                                                                                         |
| **Rapor No**     | TFT-REG-2026-002                                                                                     |
| **Proje**        | CopperMind — Bakır Vadeli İşlem Tahmin Platformu                                                 |
| **Bileşen**     | `deep_learning/` — TFT-ASRO (Temporal Fusion Transformer with Adaptive Sharpe Ratio Optimization) |
| **Bağlamlar**   | [REG-2026-001](./tft-asro-training-regression-20260331.md) · [P2 Geçiş](./tft-reg-2026-001-p0p1-analysis-p2-transition.md) |
| **Hazırlayan**  | AI Engineering Team                                                                                  |
| **Durum**        | 🟡 İterasyon 3 Eğitimi Bekleniyor                                                                  |
| **Öncelik**     | P1 — Üretim modeli yönsel sinyal veremiyor                                                         |

---

## 1. Yönetici Özeti

13–15 Nisan 2026 tarihlerinde gerçekleştirilen TFT-ASRO eğitim döngüsü, P2 çözümlerinin (Walk-Forward CV, Snapshot Ensemble) uygulanmasından sonra bile modelin **"doğru volatilite, yanlış yön" paradoksuna** düştüğünü ortaya koydu. Model kalibrasyon metriklerinde mükemmel performans gösterirken (VR=1.10), yön doğruluğu yazı-tura seviyesinde kaldı (DA=49.57%). Bu rapor, üç ardışık eğitim iterasyonunu, her birinin kök neden analizini ve uygulanan cerrahi müdahaleleri belgelemektedir.

İlk iterasyonda sorun teşhis edildi; ikinci iterasyonda uygulanan Direction BCE çözümü anti-korelasyona neden olarak durumu kötüleştirdi (DA=43.91%); üçüncü iterasyonda BCE kaldırılıp büyüklük-ağırlıklı yönsel ödül (magnitude-weighted directional reward) ile değiştirildi. Sonuçlar beklenmektedir.

---

## 2. Altyapı Düzeltmeleri (Ön Koşullar)

Bu çalışma başlamadan önce iki kritik CI/CD sorunu çözüldü:

### 2.1 GitHub Actions 3 Saat Timeout Sorunu

| Alan | Detay |
|---|---|
| **Tespit** | 13 Nisan 2026, 12:10 UTC |
| **Sorun** | Tek job olarak çalışan hyperopt (~108dk) + final training (~50dk) toplamı 3 saatlik GitHub Actions limitini aştı |
| **Commit** | `2040e7d` |
| **Çözüm** | `tft-training.yml` iki bağımsız job'a bölündü |

```
Eski (1 job, ~158 dk, 3h limitine takıldı)
┌─────────────────────────────────────────────┐
│  train-tft  [timeout: 190 dk]               │
│  ├─ hyperopt  (~108 dk)                     │
│  └─ final training  (~50 dk) ──── ×FAIL     │
└─────────────────────────────────────────────┘

Yeni (2 bağımsız job, her biri kendi 3h bütçesinde)
┌───────────────────────────┐  artifact   ┌────────────────────────────┐
│  hyperopt  [timeout: 175] │─────────────▶│  train  [needs: hyperopt]  │
│  ├─ optuna search (~108m) │  optuna_    │  ├─ download artifact      │
│  └─ upload artifact       │  results    │  ├─ final training (~50m)  │
└───────────────────────────┘  .json      └────────────────────────────┘
```

### 2.2 Screener Pipeline Workflow Dispatch Hatası

| Alan | Detay |
|---|---|
| **Sorun** | `HTTP 403: Resource not accessible by integration` — `gh workflow run` backtest tetikleyemiyordu |
| **Kök Neden** | `screener-schedule.yml` job permissions'da `actions: write` eksikti |
| **Commit** | `ed2ec57` |
| **Ek Fix** | `git pull --rebase` eklendi — concurrent push rejection önlendi (`a9f2fc4`) |

---

## 3. İterasyon Kronolojisi

### 3.1 Karşılaştırmalı Metrik Tablosu

| Metrik | İt.1 (13 Nisan) | İt.2 (14 Nisan) | Δ İt.1→2 | İt.3 (bekleniyor) | Hedef Eşik |
|---|---|---|---|---|---|
| **MAE** | 0.0455 | **0.0360** | 🟢 −20.9% | — | ≤ 0.040 |
| **RMSE** | 0.0523 | **0.0439** | 🟢 −16.1% | — | ≤ 0.045 |
| **DA** | 0.4957 | **0.4391** | 🔴 −11.4% | — | ≥ 0.520 |
| **Tail Capture** | 0.4444 | **0.5455** | 🟢 +22.7% | — | ≥ 0.500 |
| **Sharpe** | −0.7038 | **−0.1264** | 🟡 +82.0% | — | ≥ 0.30 |
| **Sortino** | −1.2788 | **−0.2313** | 🟡 +81.9% | — | ≥ 0.50 |
| **pred_std** | 0.0203 | **0.0335** | ⚠️ +65.0% | — | 0.010–0.025 |
| **actual_std** | 0.0184 | 0.0184 | — | — | — |
| **VR** | 1.1043 | **1.8229** | 🔴 +65.1% | — | 0.50–1.50 |
| **Ensemble** | 3 | 3 | — | — | ≥ 2 |

### 3.2 İlerleme Yörüngesi

```
DA Trendi
──────────────────────────────────────────
52.0% ┤ · · · · · · · · · · · · · hedef (52%)
50.0% ┤ · · · · · · · · · · · · · coin-flip sınırı
      │ ■ İt.1 (49.57%)
48.0% ┤
      │
46.0% ┤
      │
44.0% ┤                     ■ İt.2 (43.91%)
──────────────────────────────────────────
       İterasyon 1      İterasyon 2      İterasyon 3
       (teşhis)         (BCE)            (büyüklük-ağırlıklı)

VR Trendi
──────────────────────────────────────────
 2.0  ┤                     ■ İt.2 (1.82)
      │ · · · · · · · · · · · · · üst sınır (1.50)
 1.5  ┤
      │
 1.0  ┤ ■ İt.1 (1.10)                          ← ideal bölge
      │ · · · · · · · · · · · · · alt sınır (0.50)
 0.5  ┤
──────────────────────────────────────────
       İterasyon 1      İterasyon 2      İterasyon 3
```

---

## 4. İterasyon 1: Teşhis (13 Nisan)

### 4.1 Gözlem

İlk eğitim run'ı (hyperopt 15 trial × 3 fold CV) tamamlandı. Pipeline 2-job split sayesinde timeout almadı.

```
Best trial: #12, val_loss: 0.347188
  hidden_size: 48, attention_head_size: 4, dropout: 0.25
  hidden_continuous_size: 8, lambda_vol: 0.3, lambda_quantile: 0.5

Test: DA=49.57%, Sharpe=−0.70, VR=1.10
```

### 4.2 Paradoksun Tanımı

> [!IMPORTANT]
> **"Doğru Volatilite, Yanlış Yön" Paradoksu:** Model VR=1.10 ile neredeyse mükemmel kalibrasyon gösteriyor (pred_std ≈ actual_std), ama DA=49.57% ile yazı-turada — yön tahmin edemiyor. Kalibrasyon ile yönsel doğruluk birbirinden tamamen kopuk.

### 4.3 Kök Neden Analizi

Dört yapısal kök neden tespit edildi:

**Kök Neden 1: Batch-Level Sharpe Loss**

```python
# losses.py L191-207 (önceki)
signal = torch.tanh(median_pred * 20.0)
strategy_returns = signal * y_actual_f
sharpe_loss = -(strategy_returns.mean() / (strategy_returns.std() + eps))
```

Batch boyutu=32, prediction_length=5 → `strategy_returns` tensörü 160 değer. `.mean()` ve `.std()` bu shuffle edilmiş 160 değer üzerinden → çok noisy bir Sharpe tahmini. Model bireysel örnekler için yön sinyali vermek yerine batch ortalamasını sıfırda tutmayı öğrendi.

**Kök Neden 2: Hyperopt Objective Yönü Ödüllendirmiyor**

```python
# hyperopt.py L250 (önceki)
fold_score = fold_val_loss + fold_vr_penalty
```

DA (yönsel doğruluk) sadece çok geç ve çok zayıf bir penalty olarak giriyordu. DA=0.495 neredeyse sıfır ceza alıyordu.

**Kök Neden 3: Lambda Quantile = 0.5 (Kalibrasyon Baskınlığı)**

Hyperopt'un seçtiği `lambda_quantile: 0.5` → %50 kalibrasyon, %50 Sharpe ağırlığı. Ama kalibrasyon bileşeni magnitude olarak genelde Sharpe'tan büyük olduğu için EarlyStopping efektif olarak kalibrasyona optimize ediyordu.

**Kök Neden 4: Feature-to-Sample Oranı (Boyutluluk Laneti)**

~375 eğitim örneğine karşı 200+ feature (32 PCA embedding boyutu dahil). VSN bu oranla başa çıkamıyordu.

---

## 5. İterasyon 2: Beş Fazlı Loss Reformu (14 Nisan)

### 5.1 Uygulanan Değişiklikler

| Faz | Değişiklik | Dosya | Commit |
|---|---|---|---|
| **1** | Hyperopt fold_score'a DA reward eklendi | `hyperopt.py` | `3306e33` |
| **2** | Direction BCE (Binary Cross-Entropy) terimi eklendi | `losses.py`, `tft_copper.py` | `3306e33` |
| **3** | `lambda_quantile` arama aralığı [0.2, 0.6] → [0.2, 0.4] | `hyperopt.py` | `3306e33` |
| **4** | PCA embedding boyutu 32 → 8 | `config.py`, `tft-training.yml` | `3306e33` |
| **5** | Fold-level Sharpe < −0.5 pruning | `hyperopt.py` | `3306e33` |

Faz 2 — Direction BCE implementasyonu:

```python
# Per-sample directional cross-entropy
actual_sign = torch.sigmoid(y_actual_f * 100.0)  # hard sigmoid → near 0/1
pred_prob = torch.sigmoid(median_pred * 20.0)
direction_bce = F.binary_cross_entropy(pred_prob, actual_sign.detach())
sharpe_loss = sharpe_loss + 0.3 * direction_bce
```

### 5.2 Sonuçlar

```
Best trial: #4, val_loss: 0.559377
  hidden_size: 48, attention_head_size: 1, dropout: 0.35
  hidden_continuous_size: 8, lambda_vol: 0.45, lambda_quantile: 0.30
  batch_size: 32

Test: DA=43.91%, Sharpe=−0.13, VR=1.82
```

### 5.3 Direction BCE Başarısızlık Analizi

> [!CAUTION]
> **Kritik Bulgu:** Direction BCE, DA'yı 49.57%'den 43.91%'e düşürdü — model aktif olarak anti-korelasyon geliştirdi. İşaretini terslesek %56 accuracy olurdu.

**Neden Başarısız Oldu:**

Copper günlük getirilerinin çoğu ±0.01 aralığında. `sigmoid(actual * 100)` bu aralıkta belirsiz label'lar üretiyor:

| Actual Return | `sigmoid(actual × 100)` | Label Kalitesi |
|---|---|---|
| +0.024 (1σ) | 0.917 | ✅ Net pozitif |
| +0.010 | 0.731 | 🟡 Belirsiz |
| +0.005 | 0.622 | 🔴 Neredeyse kararsız |
| −0.005 | 0.378 | 🔴 Neredeyse kararsız |
| −0.010 | 0.269 | 🟡 Belirsiz |
| −0.024 (1σ) | 0.083 | ✅ Net negatif |

Günlük bakır getirilerinin çoğunluğu ±1σ içinde oluyor. Bu bölgede BCE label'ları 0.37–0.62 arasında — modele "aynı anda hem pozitif hem negatif ol" diyen çelişkili sinyaller. Model bu noise'u ezberleyerek anti-korelasyon geliştirdi.

**Ek Sorun:** VR 1.10'dan 1.82'ye fırladı (pred_std = 2× actual_std). Over-variance cezası 0.25× katsayı ile çok zayıftı:
```python
0.25 × relu(1.82 − 1.5) = 0.25 × 0.32 = 0.08  # neredeyse etkisiz
```

---

## 6. İterasyon 3: Büyüklük-Ağırlıklı Yönsel Ödül (15 Nisan)

### 6.1 Uygulanan Değişiklikler

| Değişiklik | Dosya | Commit |
|---|---|---|
| BCE → Magnitude-Weighted Directional Bonus | `losses.py`, `tft_copper.py` | `c14392f` |
| Over-variance cezası 0.25× → 1.0× | `losses.py`, `tft_copper.py` | `c14392f` |

### 6.2 Büyüklük-Ağırlıklı Yönsel Ödül — Tasarım

```python
# BCE yerine: her örneğin yönsel katkısını |actual_return| ile ağırlıkla
abs_actual = y_actual_f.abs()
magnitude_weight = abs_actual / (abs_actual.mean() + eps)
weighted_directional = (signal * y_actual_f * magnitude_weight).mean()
sharpe_loss = sharpe_loss - 0.3 * weighted_directional
```

**Tasarım Gerekçesi:**

1. **Büyük hareketler** (|return| >> mean) yüksek ağırlık alır → model büyük hareketlerin yönünü öğrenmeye zorlanır
2. **Küçük hareketler** (|return| << mean) düşük ağırlık alır → tahmin edilemez günlük noise'tan gradyan gelmez
3. BCE'nin çelişkili label problemi ortadan kalkar — label yok, doğrudan `sign(pred) × |actual|` ödülü
4. Tail Capture Rate'i doğrudan hedefler — kuyruk olaylarında yön doğruluğu ödüllendirilir

**BCE vs Magnitude-Weighted — Gradyan Karşılaştırması:**

```
Actual = +0.005 (küçük hareket, yönü tahmin edilemez):
  BCE:                 güçlü gradyan (label=0.62, zorla "pozitif" öğret) → NOISE
  Magnitude-weighted:  zayıf gradyan (weight ≈ 0.3) → doğru şekilde ignore

Actual = +0.030 (büyük hareket, yönü tahmin edilebilir):
  BCE:                 güçlü gradyan (label=0.95, "pozitif" öğret) → OK
  Magnitude-weighted:  çok güçlü gradyan (weight ≈ 1.8) → büyük hareket yönü ÖNCELİKLİ
```

### 6.3 Over-Variance Ceza Güçlendirmesi

```python
# Önceki (İt.2'de VR=1.82 ile etkisiz kaldı)
amplitude_loss = (
    torch.relu(1.0 - vr)              # VR < 1 → 1.0× ceza
    + 0.25 * torch.relu(vr - 1.5)     # VR > 1.5 → 0.25× ceza (çok zayıf)
)

# Yeni (simetrik)
amplitude_loss = (
    torch.relu(1.0 - vr)              # VR < 1 → 1.0× ceza
    + 1.0 * torch.relu(vr - 1.5)      # VR > 1.5 → 1.0× ceza (simetrik)
)
```

İt.2'deki VR=1.82 ile karşılaştırma:
- Eski: `0.25 × relu(1.82 − 1.5) = 0.08` → modeli frenleyemedi
- Yeni: `1.0 × relu(1.82 − 1.5) = 0.32` → 4× güçlü frenleme

### 6.4 Beklenen Etki

| Metrik | İt.2 Sonucu | İt.3 Beklenti | Gerekçe |
|---|---|---|---|
| **DA** | 43.91% | ≥ 52% | Küçük hareket noise'u eliminasyonu + büyük hareket yön ödülü |
| **Sharpe** | −0.13 | ≥ 0.0 | Yön doğruluğu artışı + anti-korelasyon düzeltmesi |
| **VR** | 1.82 | 0.8–1.3 | Simetrik 1.0× over-variance cezası |
| **Tail Capture** | 54.55% | ≥ 55% | Korunacak — magnitude weighting kuyruk tahminini güçlendirir |
| **MAE** | 0.0360 | ~0.036 | Korunacak |

---

## 7. Tüm Değişikliklerin Dosya Haritası

```
CopperMind Platform (13–15 Nisan 2026)
├── .github/workflows/
│   ├── tft-training.yml       ← 1 job → 2 job split (2040e7d)
│   │                            pca-dim 32→8 (3306e33)
│   └── screener-schedule.yml  ← actions:write izni (ed2ec57)
│                                 git pull --rebase (a9f2fc4)
├── backend/deep_learning/
│   ├── config.py              ← pca_dim 32→8 (3306e33, f6b30df)
│   ├── models/
│   │   ├── losses.py          ← Sample-level Sharpe + BCE (3306e33)
│   │   │                        BCE → magnitude-weighted (c14392f)
│   │   │                        VR cezası 0.25→1.0 (c14392f)
│   │   └── tft_copper.py      ← ASROPFLoss mirror (3306e33, c14392f)
│   └── training/
│       └── hyperopt.py        ← DA reward + lambda cap + fold pruning (3306e33)
└── backend/tests/
    └── deep_learning/
        └── test_config.py     ← pca_dim assertion 32→8 (f6b30df)
```

### Commit Geçmişi

```
c14392f fix(asro): replace direction BCE with magnitude-weighted reward, strengthen VR cap
f6b30df fix(config): repair corrupted pca_dim field definition and update test assertion
3306e33 fix(tft): break directional accuracy deadlock with 5-phase loss reform
a9f2fc4 ci: add git pull --rebase before pushing to prevent race conditions
ed2ec57 ci: fix screener pipeline workflow dispatch permissions
2040e7d ci: split TFT training into hyperopt + train jobs (3h limit fix)
```

---

## 8. Denenen ve Reddedilen Çözümler

### 8.1 Direction BCE (Uygulandı, Geri Alındı)

| Alan | Detay |
|---|---|
| **Commit** | `3306e33` (eklendi), `c14392f` (kaldırıldı) |
| **Ömür** | 1 eğitim döngüsü (24 saat) |
| **Amaç** | Her örneğe "yönün doğru mu?" binary sinyali vermek |
| **Sonuç** | ❌ DA %50'den %44'e düştü — anti-korelasyon |

**Neden Reddedildi:**
1. Copper günlük getirilerinin çoğunluğu sigmoid'de belirsiz bölgede (0.37–0.62)
2. Belirsiz label'lar → noise ezberleme → anti-korelasyon
3. VR kontrolsüz artış (1.10 → 1.82) — model agresifleşti ama yanlış yönde

---

## 9. Güncel Durum ve Sonraki Adımlar

| Madde | Durum |
|---|---|
| Pipeline altyapısı sağlıklı mı? | ✅ 2-job split, permissions, rebase |
| İt.1 kök nedeni tespit edildi mi? | ✅ Batch-level Sharpe + kalibrasyon baskınlığı |
| İt.2 başarısızlık nedeni tespit edildi mi? | ✅ BCE noisy label → anti-korelasyon |
| İt.3 çözümü uygulandı mı? | ✅ Magnitude-weighted directional + VR cap |
| İt.3 eğitim sonuçları geldi mi? | ❌ Bekleniyor |

### 9.1 Başarı Kriterleri (İt.3)

| Metrik | Minimum Kabul | Hedef | Kritik Sınır (Geri Dönüş) |
|---|---|---|---|
| **DA** | ≥ 0.510 | ≥ 0.530 | < 0.490 → rollback |
| **Sharpe** | ≥ 0.00 | ≥ 0.30 | < −0.30 → rollback |
| **VR** | 0.70–1.50 | 0.80–1.20 | < 0.50 veya > 1.80 → rollback |
| **Tail Capture** | ≥ 0.450 | ≥ 0.550 | < 0.350 → rollback |

### 9.2 Eskalasyon (İt.3 Başarısız Olursa)

İt.3 başarısız olması durumunda aşağıdaki eskalasyon seçenekleri değerlendirilecektir:

| # | Seçenek | Efor | Gerekçe |
|---|---|---|---|
| 1 | Magnitude-weighted ağırlığını 0.3 → 0.5'e artır | Düşük | Yönsel sinyal hâlâ zayıfsa |
| 2 | Sharpe bileşenini tamamen kaldır, sadece quantile + magnitude | Orta | Sharpe loss'un kendisi sorunlu olabilir |
| 3 | Feature pruning: `interpret_output()` ile top-20 feature seç | Orta | 200+ feature'ın çoğu noise |
| 4 | Lookback penceresi 730→1095 gün (daha fazla eğitim örneği) | Düşük | Feature/sample oranını iyileştirir |

---

*Bu rapor, İterasyon 3 sonuçları geldikten sonra güncellenecektir.*
*Son güncelleme: 15 Nisan 2026*
