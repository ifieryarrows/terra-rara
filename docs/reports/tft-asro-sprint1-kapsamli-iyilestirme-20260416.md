# TFT-ASRO Kapsamlı İyileştirme Sprint'i — Sonuç Raporu

| Alan                   | Değer                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| **Rapor Tarihi**       | 16 Nisan 2026                                                                                       |
| **Rapor No**           | TFT-IMP-2026-001                                                                                    |
| **Proje**              | CopperMind — Bakır Vadeli İşlem Tahmin Platformu                                                    |
| **Bileşen**            | `deep_learning/` — TFT-ASRO + CI/CD Pipeline                                                       |
| **Bağlam**             | [REG-2026-002](./tft-asro-directional-accuracy-fix-20260415.md) · İt.3 sonuçlarına yanıt            |
| **Hazırlayan**         | AI Engineering Team                                                                                  |
| **Durum**              | 🟢 Sprint 1 Tamamlandı — İt.4 Eğitimi Çalışıyor                                                   |
| **Öncelik**            | P1 — Sharpe ve DA hedeflerine ulaşma                                                                |

---

## 1. Yönetici Özeti

15 Nisan 2026 tarihinde tamamlanan İterasyon 3 (magnitude-weighted directional reward) modeli kritik eşiği geçirdi: **Sharpe -0.70'ten +0.068'e**, **DA %49.57'den %51.15'e** yükseldi. Ancak VR=0.39 ile model düşük varyans tuzağına girdi — tahminler gerçek oynaklığın yalnızca %39'unu yansıtıyor.

16 Nisan 2026'da kapsamlı bir iyileştirme sprint'i uygulandı. Bu sprint, önceki raporlarda ([REG-2026-001](./tft-asro-training-regression-20260331.md), [REG-2026-002](./tft-asro-directional-accuracy-fix-20260415.md)) tespit edilen tüm köken nedenleri hedefleyen altı fazlı bir plan olarak tasarlandı:

1. **FAZ 1** — Feature mühendisliği ve boyut azaltma (MRMR, lookback)
2. **FAZ 2** — Kayıp fonksiyonu reformu (MADL, curriculum learning)
3. **FAZ 3** — Eğitim metodolojisi (Purged CV, SWA, weight decay, augmentation)
4. **FAZ 4** — Model boyutu optimizasyonu
5. **FAZ 5** — Ensemble stratejileri (XGBoost+TFT oylama, Theta baseline)
6. **FAZ 6** — Pipeline kalitesi (otomatik rollback, sentiment QC)

Sprint tamamlandıktan sonra ilk eğitim döngüsü **val_loss=−0.009** ile başarıyla tamamlandı.

---

## 2. İterasyon 3 Sonuçları (Bağlam)

```
İt.3 Eğitim (15 Nisan 2026 gece)
Best trial: #14, val_loss: −0.009069
  max_encoder_length: 50, hidden_size: 24, attention_head_size: 2
  dropout: 0.30, hidden_continuous_size: 8, lr: 9.28e-4
  lambda_vol: 0.3, lambda_quantile: 0.4, lambda_madl: 0.4
  batch_size: 16, weight_decay: 6.23e-4

Test Metrikleri:
  DA           = 51.15%   ↑ (+1.6pp — 50% bariyerini geçti)
  Sharpe       = +0.068   ↑ (−0.70'ten pozitife döndü ✅)
  VR           = 0.394    ⚠ (hedef 0.5–1.5; düşük varyans tuzağı)
  Tail Capture = 47.06%   ↑ (+2.6pp)
  MAE          = 0.0462   ➖
  pred_std     = 0.0078   (actual_std = 0.0199)
  Ensemble     = 3 model
```

**Kritik Gözlem:** `lambda_madl: 0.4` Optuna tarafından yüksek seçildi — MADL'nin yön doğruluğuna katkısı doğrulandı. Ancak yüksek MADL ağırlığı, modelin medyan tahminlerini küçük tutarak kalibrasyon yerine yön sinyaline odaklanmasına yol açtı → VR=0.39.

---

## 3. Sprint Değişiklikleri

### 3.1 FAZ 1: Feature Mühendisliği

#### 1.1 Lookback Penceresi Genişletme

| Alan | Değer |
|---|---|
| **Değişiklik** | `lookback_days: 730 → 1095` (~3 yıl) |
| **Dosya** | `config.py` |
| **Etki** | ~313 örnek → ~564 örnek (İt.3 loglarında doğrulandı: "565 bars for HG=F") |

Eğitim örneği sayısı %80 arttı. Feature/sample oranı ~200:313'ten MRMR sonrası ~80:500'e indi.

#### 1.2 MRMR Ön-Filtre

| Alan | Değer |
|---|---|
| **Yeni Dosya** | `backend/deep_learning/data/feature_selection.py` |
| **Yöntem** | Mutual Information relevance − Pearson correlation redundancy |
| **Konfigürasyon** | `FeatureStoreConfig.mrmr_top_k = 80` |

İt.3 loglarında görüldüğü gibi: `MRMR selected 80/320 features (top relevance=0.0911, bottom=0.0059)` — 320 feature'dan 80'e indirdi; feature/sample oranı ~1:6 hedefine ulaşıldı.

**İki aşamalı budama zinciri:**
```
Ham features (320)
  → MRMR ön-filtre (top 80 by MI relevance − redundancy)
    → Eğitim sonrası VSN interpret_output() budama (top 40-60)
```

**Referans:** Ding & Peng (2005) "Minimum Redundancy Feature Selection from Microarray Gene Expression Data"

#### 1.3 Sentiment QC

`feature_store.py`'e sentiment index'in hedef değişkenle gecikme korelasyonunu raporlayan `_sentiment_qc()` fonksiyonu eklendi. İt.3 loglarında `corr(sent, ret_t+1)=nan` görüldü — bu embedding karışıklığından kaynaklandı (aşağıda Bölüm 5.1).

---

### 3.2 FAZ 2: Kayıp Fonksiyonu Reformu

#### 2.1 MADL (Mean Absolute Directional Loss)

| Alan | Değer |
|---|---|
| **Yeni Sınıf** | `MeanAbsoluteDirectionalLoss` in `losses.py` |
| **Formülasyon** | `MADL = mean(-tanh(pred×20) × actual × |actual|)` |
| **Config** | `ASROConfig.lambda_madl = 0.25` (Optuna: 0.1–0.5) |

```python
# losses.py — MeanAbsoluteDirectionalLoss.forward()
soft_sign = torch.tanh(y_pred_median * self.tanh_scale)  # differentiable sign
direction_match = soft_sign * y_actual
madl = (-direction_match * y_actual.abs()).mean()
```

BCE ile karşılaştırma:

| Kriter | BCE (İt.2) | MADL (İt.3+) |
|---|---|---|
| Küçük hareket (±0.005) | Güçlü gradient (noise) | Zayıf gradient (doğru) |
| Büyük hareket (±0.030) | Güçlü gradient | Çok güçlü gradient |
| Anti-korelasyon riski | Yüksek (ambiguous labels) | Düşük |
| Gradient sürekliliği | ✅ | ✅ (tanh) |

**Optuna'nın seçimi `lambda_madl: 0.4`** — MADL'nin etkili olduğunu gösteriyor.

**Referans:** Kisiel & Gorse (2023) "Mean Absolute Directional Loss" (ScienceDirect); Kisiel & Gorse (2024) "Generalized MADL" (arXiv:2412.18405)

#### 2.2 Curriculum Loss Scheduler

| Alan | Değer |
|---|---|
| **Yeni Dosya** | `backend/deep_learning/training/callbacks.py` — `CurriculumLossScheduler` |
| **Davranış** | Epoch 0: λ_q=0.65 (kalibrasyon önce), warmup sonrası λ_q=0.35 (hedef) |

```
Epoch 0-10 (warmup): lambda_quantile = 0.65 → model önce doğru ölçekte tahmin etsin
Epoch 10+:           lambda_quantile = 0.35 → yönsel öğrenme ön plana çıksın
```

**Referans:** Bengio et al. (2009) "Curriculum Learning" (ICML)

---

### 3.3 FAZ 3: Eğitim Metodolojisi

#### 3.1 Purged Walk-Forward CV (embargo=5)

```python
# dataset.py — build_cv_folds()
val_start_pos = train_end_pos + purge_gap  # purge_gap=5 gün
val_start_idx = master_df["time_idx"].iloc[val_start_pos]
```

Train ve validation arasına 5 günlük boşluk eklendi. Bu, autocovariance yoluyla oluşabilecek veri sızıntısını önler.

**Referans:** de Prado (2018) "Advances in Financial Machine Learning" Bölüm 7

#### 3.2 Stochastic Weight Averaging (SWA)

| Alan | Değer |
|---|---|
| **Yeni Sınıf** | `SWACallback` in `callbacks.py` |
| **Başlangıç** | Son %75 epoch'tan itibaren ağırlık ortalaması |

**Referans:** Izmailov et al. (2018) "Averaging Weights Leads to Wider Optima and Better Generalization" (UAI)

#### 3.3 Weight Decay

```python
# tft_copper.py — configure_optimizers wrapper
for pg in opt.param_groups:
    if pg.get("weight_decay", 0.0) == 0.0:
        pg["weight_decay"] = _weight_decay
```

`optimizer_kwargs` pytorch_forecasting tarafından desteklenmediği için `configure_optimizers`'ı post-construction wrap ederek uygulandı.

#### 3.4 Veri Artırma

| Alan | Değer |
|---|---|
| **Yeni Dosya** | `backend/deep_learning/data/augmentation.py` |
| **Yöntemler** | Jittering (σ=0.5%), Magnitude Warping (%2 bant) |
| **Oran** | Eğitim setinin %15'i kadar sentetik örnek |

---

### 3.4 FAZ 4: Model Boyutu Optimizasyonu

Optuna arama uzayı MRMR sonrası (~80 feature) veri boyutuna göre daraltıldı:

| Parametre | Eski Aralık | Yeni Aralık | Gerekçe |
|---|---|---|---|
| `hidden_size` | 32–64 (step 16) | 24–48 (step 8) | 80 feature için 24 yeterli |
| `attention_head_size` | 1–4 | 1–2 | Tek seri, tek grup |
| `hidden_continuous_size` | 8–24 (step 8) | 8–16 (step 8) | Paired reduction |

İt.3'te Optuna `hidden_size: 24` seçti — küçük modelin daha iyi genellediği doğrulandı.

---

### 3.5 FAZ 5: Ensemble Stratejileri

#### 5.1 XGBoost + TFT Yön Oylama

| Alan | Değer |
|---|---|
| **Yeni Fonksiyon** | `ensemble_directional_vote()` in `inference/predictor.py` |
| **Kural** | Her iki model aynı yön → yüksek güven (1.0×); farklı yön → nötr (0.0×) |
| **Not** | XGBoost aşırı nadir negatif gösterdiği için `xgb_bias_correction` parametresi eklendi |

#### 5.2 Theta Model Baseline

| Alan | Değer |
|---|---|
| **Yeni Dosya** | `backend/deep_learning/baselines/theta_model.py` |
| **İçerik** | `theta_forecast()`, `theta_backtest()` |

**Referans:** Assimakopoulos & Nikolopoulos (2000) "The theta model: a decomposition approach to forecasting" (IJF)

---

### 3.6 FAZ 6: Pipeline Kalitesi

#### 6.1 Otomatik Quality Gate

| Alan | Değer |
|---|---|
| **Yeni Dosya** | `backend/scripts/tft_quality_gate.py` |
| **Kural** | DA < 0.49 veya Sharpe < -0.30 veya VR ∉ [0.2, 2.5] → FAIL |
| **CI** | `tft-training.yml` sonunda `continue-on-error: true` ile çalışır |

#### 6.2 Eğitim Metadata Kaydı

```python
# trainer.py — _persist_tft_metadata sonrası
meta_json_path = Path(cfg.training.best_model_path).parent / "tft_metadata.json"
meta_json_path.write_text(json.dumps(result, indent=2, default=str))
```

Quality gate bu dosyayı okur.

---

## 4. Hata Düzeltmeleri

### 4.1 Embedding Shape Uyuşmazlığı

| Alan | Detay |
|---|---|
| **Hata** | `ValueError: all input arrays must have the same shape` |
| **Kök Neden** | DB'de eski 32-dim PCA vektörleri ile yeni 8-dim vektörler karışmış |
| **Dosya** | `embeddings.py` — `bytes_to_embedding()` |
| **Commit** | `98b27fd` |

```python
# Önceki (dim parametresini yoksayıyordu)
return np.frombuffer(data, dtype=np.float32).copy()

# Yeni (truncate veya zero-pad ile normalleştir)
arr = np.frombuffer(data, dtype=np.float32).copy()
if len(arr) > dim: return arr[:dim]
if len(arr) < dim:
    padded = np.zeros(dim, dtype=np.float32)
    padded[:len(arr)] = arr
    return padded
return arr
```

### 4.2 `optimizer_kwargs` TypeError

| Alan | Detay |
|---|---|
| **Hata** | `TypeError: BaseModel.__init__() got an unexpected keyword argument 'optimizer_kwargs'` |
| **Kök Neden** | `TemporalFusionTransformer.from_dataset()` bu parametreyi desteklemiyor |
| **Etki** | Tüm hyperopt trial'ları `val_loss=inf` döndürdü (model oluşturmada çöktü, catch bloğu `inf` döndürdü) |
| **Commit** | `38ec858` |

```python
# Yeni yaklaşım: configure_optimizers post-construction wrap
_orig = model.configure_optimizers
def _wd_configure_optimizers():
    result = _orig()
    for opt in [result] if not isinstance(result, list) else result:
        for pg in opt.param_groups:
            pg.setdefault("weight_decay", _weight_decay)
    return result
model.configure_optimizers = _wd_configure_optimizers
```

### 4.3 YAML Sözdizimi Hatası (satır 160)

| Alan | Detay |
|---|---|
| **Hata** | `Invalid workflow file: .github/workflows/tft-training.yml#L160` |
| **Kök Neden** | `run: |` bloğu içindeki Python kodu YAML girintisini kaybetti |
| **Çözüm** | Python mantığı `backend/scripts/tft_quality_gate.py`'e taşındı |
| **Commit** | `8693282` |

---

## 5. Metrik Karşılaştırma Tablosu

| Metrik | İt.1 (13 Nis) | İt.2 (14 Nis) | İt.3 (15 Nis) | Δ İt.2→3 | Hedef | Durum |
|---|---|---|---|---|---|---|
| **DA** | 49.57% | 43.91% | **51.15%** | +7.2pp | ≥ 52% | 🟡 Yaklaşıyor |
| **Sharpe** | −0.70 | −0.13 | **+0.068** | +0.20 | ≥ 0.30 | 🟡 Pozitife döndü |
| **VR** | 1.10 | 1.82 | **0.394** | −1.43 | 0.5–1.5 | 🔴 Düşük varyans |
| **Tail Capture** | 44.4% | 54.6% | **47.1%** | −7.5pp | ≥ 50% | 🟡 |
| **MAE** | 0.0455 | 0.0360 | **0.0462** | +0.010 | ≤ 0.040 | 🔴 |
| **pred_std** | 0.0203 | 0.0335 | **0.0078** | −0.026 | 0.010–0.025 | 🔴 Çok düşük |
| **actual_std** | 0.0184 | 0.0184 | **0.0199** | — | — | — |
| **Ensemble** | 3 | 3 | **3** | — | ≥ 2 | ✅ |

```
DA Trendi
──────────────────────────────────────────────
52.0% ┤ · · · · · · · · · · · · · hedef
51.2% ┤                              ■ İt.3
50.0% ┤ · · · · · · · · coin-flip sınırı
      │ ■ İt.1 (49.57%)
      │
44.0% ┤         ■ İt.2 (43.91%)
──────────────────────────────────────────────

VR Trendi
──────────────────────────────────────────────
1.82 ┤         ■ İt.2 (aşırı volatil)
     │ · · · · · · · · · üst sınır (1.50)
1.10 ┤ ■ İt.1
     │ · · · · · · · · · alt sınır (0.50)
0.39 ┤                              ■ İt.3 (düşük varyans tuzağı)
──────────────────────────────────────────────
```

---

## 6. VR=0.394 Düşük Varyans Tuzağı — Analiz ve Düzeltme

### 6.1 Kök Neden

İt.3'te `lambda_madl: 0.4` seçildi. MADL yüksek ağırlıkla **yön sinyaline** optimize ederken, **medyan tahminlerini küçük tutmayı** ikincil bıraktı. Model doğru yönde küçük değerler tahmin ederek hem MADL ödülü aldı hem kalibrasyon cezasından kaçındı.

```
Eski amplitude_loss (3.6.3 raporu ile güncellenmişti):
  relu(1 - VR)         → VR=0.39 için: relu(0.61) = 0.61
  1.0 × relu(VR - 1.5) → 0 (VR < 1.5)
  Toplam = 0.61

lambda_vol=0.3 ile efektif gradient: 0.3 × 0.61 = 0.183
```

Bu gradient, `lambda_madl: 0.4` yön ödülünü dengelemek için yetersiz kaldı.

### 6.2 Uygulanan Düzeltme

VR < 0.5 için iki kademeli ceza eklendi:

```python
# losses.py — amplitude_loss (İt.4)
under_severe   = 2.0 * torch.relu(0.5 - vr)   # VR < 0.5 → ekstra 2× ceza
under_moderate = torch.relu(1.0 - vr)          # VR < 1.0 → mevcut 1× ceza
over_variance  = 1.0 * torch.relu(vr - 1.5)    # VR > 1.5 → değişmedi
amplitude_loss = under_severe + under_moderate + over_variance
```

VR=0.39 için karşılaştırma:

| Durum | Eski Ceza | Yeni Ceza | Değişim |
|---|---|---|---|
| VR=0.39 | `relu(0.61) = 0.61` | `2×relu(0.11) + relu(0.61) = 0.22+0.61 = 0.83` | +36% |
| VR=0.70 | `relu(0.30) = 0.30` | `0 + relu(0.30) = 0.30` | — (etkilenmedi) |
| VR=1.82 | `relu(0.32) = 0.32` | `0 + 0 + relu(0.32) = 0.32` | — (etkilenmedi) |

---

## 7. Tüm Değişikliklerin Dosya Haritası

```
CopperMind Platform (16 Nisan 2026)
├── .gitignore                         ← .cursor/ eklendi
├── .github/workflows/
│   └── tft-training.yml               ← Quality gate adımı (8693282)
├── backend/
│   ├── scripts/
│   │   └── tft_quality_gate.py        ← Yeni: CI quality gate scripti
│   └── deep_learning/
│       ├── config.py                  ← lookback 730→1095, lambda_madl, weight_decay,
│       │                                  mrmr_top_k
│       ├── baselines/
│       │   └── theta_model.py         ← Yeni: Theta baseline
│       ├── data/
│       │   ├── augmentation.py        ← Yeni: Jitter + MagnitudeWarp
│       │   ├── dataset.py             ← Purged CV gap=5
│       │   ├── embeddings.py          ← bytes_to_embedding shape normalisation (98b27fd)
│       │   ├── feature_selection.py   ← Yeni: MRMR + VSN budama
│       │   └── feature_store.py       ← MRMR entegrasyonu, sentiment QC
│       ├── models/
│       │   ├── losses.py              ← MeanAbsoluteDirectionalLoss, iki kademeli VR cezası
│       │   └── tft_copper.py          ← ASROPFLoss MADL + weight_decay wrap (38ec858)
│       ├── training/
│       │   ├── callbacks.py           ← Yeni: CurriculumLossScheduler, SWACallback
│       │   ├── hyperopt.py            ← lambda_madl, weight_decay arama uzayı, küçük model
│       │   └── trainer.py             ← Curriculum+SWA callbacks, metadata JSON
│       ├── inference/
│       │   └── predictor.py           ← ensemble_directional_vote()
│       └── validation/
│           └── backtest.py            ← Yeni: Walk-forward backtest + Theta karşılaştırma
├── tests/deep_learning/
│   ├── test_backtest.py               ← Yeni
│   ├── test_callbacks.py              ← Yeni
│   ├── test_ensemble.py               ← Yeni
│   ├── test_feature_selection.py      ← Yeni
│   ├── test_config.py                 ← lookback, mrmr_top_k, weight_decay, lambda_madl
│   └── test_losses.py                 ← MADL testleri
```

### Commit Geçmişi

```
(güncel)   fix(losses): strengthen under-variance penalty for VR < 0.5
38ec858    fix(tft): remove unsupported optimizer_kwargs from from_dataset()
98b27fd    fix(embeddings): normalize PCA vector shapes to prevent stack crash
8693282    fix(workflow): yaml syntax error l160, create new folder scripts
7d3a0bd    feat(tft-asro): improve direction learning and generalization
```

---

## 8. Test Sonuçları

```
64 passed, 1 skipped (callbacks — lightning yerel kurulu değil, CI'da çalışır)

tests/deep_learning/test_losses.py::test_madl_correct_direction_is_negative PASSED
tests/deep_learning/test_losses.py::test_madl_wrong_direction_is_positive  PASSED
tests/deep_learning/test_losses.py::test_madl_large_moves_dominate         PASSED
tests/deep_learning/test_losses.py::test_madl_has_gradients                PASSED
tests/deep_learning/test_losses.py::test_asro_includes_madl_component      PASSED
tests/deep_learning/test_ensemble.py::test_both_agree_bullish              PASSED
tests/deep_learning/test_ensemble.py::test_disagreement_yields_neutral     PASSED
tests/deep_learning/test_feature_selection.py::test_mrmr_select_reduces_features PASSED
```

---

## 9. Güncel Durum ve Sonraki Adımlar

| Madde | Durum |
|---|---|
| MRMR ön-filtresi çalışıyor mu? | ✅ `80/320 features selected` (İt.3 log) |
| MADL Optuna tarafından anlamlı bulundu mu? | ✅ `lambda_madl: 0.4` seçildi |
| Embedding shape hatası düzeltildi mi? | ✅ `98b27fd` |
| YAML sözdizimi düzeltildi mi? | ✅ `8693282` |
| `optimizer_kwargs` hatası düzeltildi mi? | ✅ `38ec858` |
| VR düşük varyans düzeltmesi uygulandı mı? | ✅ İki kademeli ceza |
| İt.4 eğitimi başlatıldı mı? | 🔄 Bekleniyor |

### 9.1 İt.4 Başarı Kriterleri

| Metrik | Minimum Kabul | Hedef | Geri Dönüş |
|---|---|---|---|
| **DA** | ≥ 51% | ≥ 53% | < 49% → rollback |
| **Sharpe** | ≥ 0.05 | ≥ 0.25 | < −0.20 → rollback |
| **VR** | 0.50–1.50 | 0.70–1.20 | < 0.40 veya > 2.0 → rollback |
| **Tail Capture** | ≥ 47% | ≥ 52% | < 40% → rollback |

### 9.2 İt.4 Başarısız Olursa

| # | Seçenek | Gerekçe |
|---|---|---|
| 1 | `lambda_vol` Optuna aralığını 0.4–0.6'ya yükselt | VR hâlâ düşükse daha güçlü volatilite sinyali |
| 2 | Curriculum warmup'ı 15 epoch'a çıkar | Kalibrasyon temeli daha sağlam kurulsun |
| 3 | MRMR `top_k` 80 → 60 | Daha agresif feature budama |
| 4 | Alternatif mimari: N-HiTS veya PatchTST | Küçük veri setlerinde TFT'den üstün olabilir |

---

*Bu rapor, İterasyon 4 sonuçları geldikten sonra güncellenecektir.*
*Son güncelleme: 16 Nisan 2026*
