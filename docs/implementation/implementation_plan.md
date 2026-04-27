# TFT-ASRO Tekrarlayan Eğitim Başarısızlığı — Kök Neden Analizi ve Kalıcı Çözüm Planı

| Alan | Değer |
| --- | --- |
| **Rapor Tarihi** | 27 Nisan 2026 |
| **Bağlam** | 27 Nisan son eğitim koşusu + 9 geçmiş rapor analizi |
| **Durum** | 📝 Plan — kullanıcı onayı bekleniyor |

---

## 1. Yönetici Özeti

27 Nisan 2026 son eğitim koşusu aşağıdaki sonuçları üretti:

```text
HYPEROPT: no_finite_completed_trials (pruned=15, completed=0)
TRAINING: DA=0.4340, Sharpe=-2.3472, VR=1.6173, Tail=0.3636
QUALITY GATE: FAILED → rollback
```

Bu, projenin **31 Mart'tan bu yana 5. ciddi eğitim başarısızlığı**. Rapor kronolojisi:

| Tarih | Sorun | Sonuç |
|---|---|---|
| 31 Mart | Optuna overfitting | Sharpe -0.86, DA 48% |
| 13-15 Nisan | Yön paradoksu + BCE felaketi | DA 43.9% (anti-korelasyon) |
| 20 Nisan | Sprint 1 İt.4 | ✅ İlk başarılı sonuç (Sharpe +0.53, DA 52%) |
| 24 Nisan | Quantile crossing | Production'da yanıltıcı tahmin |
| **27 Nisan** | **Tüm trial'lar prune** | **Hiç geçerli trial yok** |

> [!CAUTION]
> **Kritik gözlem:** Başarılı olan İt.4 (20 Nisan) bir **istisna**, başarısızlıklar ise **kural**. 5 koşunun 4'ü fail etti. Bu, tek tek hataları düzeltmekle değil, sistemin yapısal kırılganlığıyla yüzleşmekle çözülecek bir sorundur.

---

## 2. Kök Neden Zinciri — Gerçek Sorun Nerede?

### 2.1 Yüzeysel Sorun: "15 trial hep prune edildi"

Hyperopt çıktısı `no_finite_completed_trials` — 15 trial'ın hepsi prune. Bu neden oldu?

### 2.2 İlk Kök Neden: Aşırı Agresif Prune Kuralları

[hyperopt.py](file:///d:/dev/projects/copper-mind/backend/deep_learning/training/hyperopt.py) analizi:

```python
# L357-362: Per-fold Sharpe pruning
if fold_sharpe < -0.5 and fold_idx >= 1:
    raise optuna.exceptions.TrialPruned()

# L399-404: Cross-fold avg Sharpe pruning  
if avg_sharpe < 0.0:
    raise optuna.exceptions.TrialPruned()

# L406-411: Quantile incoherence pruning
if avg_crossing > 0.20 or avg_median_gap > 0.01:
    raise optuna.exceptions.TrialPruned()
```

**Sorunun mekaniği:**
1. Sharpe < 0 olunca trial prune ediliyor (L399)
2. Crossing > 0.20 olunca trial prune ediliyor (L406)
3. Fold Sharpe < -0.5 olunca trial prune ediliyor (L357)
4. MedianPruner running average üzerinden de prune ediyor (L367)

Bu **dört ayrı prune kapısı** birlikte, erken fazda hiçbir trial'ın tamamlanmasına izin vermiyor. Sorun: bu kurallar İt.4'ün **başarılı** sonuçlarına göre kalibre edilmiş, ama modelin **başlangıç performansı** bu eşikleri karşılayamıyor.

> [!IMPORTANT]
> **Paradoks:** Prune kuralları "kötü modeli elemeyi" hedefliyor, ama **tüm modelleri** eliyor. Optuna'nın kendi TPE algoritmasına hiç şans verilmiyor — 15 trial'ın hiçbiri hiperparametre uzayını keşfetme fırsatı bulamıyor.

### 2.3 İkinci Kök Neden: "No Trial → Default Config" Geri Dönüş Mekanizması

[hyperopt.py L72-83](file:///d:/dev/projects/copper-mind/backend/deep_learning/training/hyperopt.py#L72-L83) ve [trainer.py L348-412](file:///d:/dev/projects/copper-mind/backend/deep_learning/training/trainer.py#L348-L412) analizi:

Tüm trial'lar prune edildiğinde:
1. `_build_result_payload` → `status: "no_finite_completed_trials"` döner
2. `optuna_results.json` → `best_params: {}` (boş)
3. `trainer.py._apply_optuna_results` → `params = {}` → default config kullanılır

Default config [config.py L96-125](file:///d:/dev/projects/copper-mind/backend/deep_learning/config.py#L96-L125):
```text
lambda_quantile: 0.4   → %40 kalibrasyon, %60 yön
lambda_vol: 0.35
lambda_madl: 0.25
lambda_crossing: 1.0
```

Bu config İt.4'te Optuna'nın bulduğu config'den farklı:
```text
İt.4 (başarılı):   lambda_quantile=0.25, lambda_madl=0.40, lambda_vol=0.30
Default:           lambda_quantile=0.40, lambda_madl=0.25, lambda_vol=0.35
```

Default config **kalibrasyon ağırlıklı** (%40), İt.4 **yön ağırlıklı** (%75). Default config ile eğitim, 27 Nisan raporunda (REG-2026-004) belgelenen "doğru volatilite, yanlış yön" patolojisini tekrar üretiyor.

### 2.4 Üçüncü Kök Neden: Veri Rejimi Değişkenliği

~500 eğitim örneği (1095 gün lookback) ile 3-fold CV'de her fold ~35-40 validation sample içeriyor. Bu küçük örneklem:
- Fold'lar arası metrik varyansını çok yüksek tutuyor
- Bir fold'da Sharpe +0.3 olan bir config, diğer fold'da Sharpe -0.8 olabiliyor
- `avg_sharpe < 0` kuralı ortalamada çoğu trial'ı prune ediyor

### 2.5 Kök Neden Ağacı (Özet)

```
27 Nisan başarısızlığı
    ↓
Tüm 15 trial prune → default config ile eğitim
    ↓
Default config yön-zayıf → DA=0.43, Sharpe=-2.35
    ↓
Quality gate fail → rollback
    ↓
NEDEN tüm trial'lar prune?
    ↓
4 katmanlı prune sistemi (Sharpe<0 + crossing>0.20 + fold_sharpe<-0.5 + MedianPruner)
    ↓
Küçük validation fold'larında (~35 sample) yüksek metrik varyansı
    ↓
Agresif prune eşikleri + küçük veri = 0 tamamlanan trial

NEDEN default config kötü?
    ↓
Default config İt.4'ün Optuna-optimized config'i değil
    ↓
lambda_quantile=0.4 kalibrasyon ağırlıklı → yön öğrenimi zayıf
    ↓
"Doğru volatilite, yanlış yön" patolojisi tekrar ediyor
```

---

## 3. Önerilen Yapısal Değişiklikler

### Değişiklik 1: Hyperopt Pruning'de "Guaranteed Completion" Mekanizması

> **Amaç:** Optuna'nın en az N trial'ı tamamlamasını garantilemek; sıfır-tamamlanan-trial senaryosunu yapısal olarak ortadan kaldırmak.

#### [MODIFY] [hyperopt.py](file:///d:/dev/projects/copper-mind/backend/deep_learning/training/hyperopt.py)

**Değişiklik A — Startup trial koruması:**
```python
# İlk N trial'da agresif pruning kapatılır
MIN_COMPLETED_TRIALS = 3  # En az 3 trial tamamlanmadan prune yapma

def _objective(trial, base_cfg, master_data):
    ...
    # Kaç trial şu ana kadar tamamlandı?
    study = trial.study
    n_completed = sum(
        1 for t in study.trials
        if t.state.name == "COMPLETE" and t.value is not None and np.isfinite(t.value)
    )
    protect_trial = (n_completed < MIN_COMPLETED_TRIALS)
    
    ...
    # Per-fold Sharpe pruning — startup korumalı
    if fold_sharpe < -0.5 and fold_idx >= 1 and not protect_trial:
        raise optuna.exceptions.TrialPruned()
    
    ...
    # Cross-fold avg Sharpe pruning — startup korumalı
    if avg_sharpe < 0.0 and not protect_trial:
        raise optuna.exceptions.TrialPruned()
    
    # Quantile incoherence — startup korumalı
    if (avg_crossing > 0.20 or avg_median_gap > 0.01) and not protect_trial:
        raise optuna.exceptions.TrialPruned()
```

**Değişiklik B — Pruning eşiklerini gevşetme:**

Mevcut `avg_sharpe < 0.0` eşiği çok katı. İt.4 öncesindeki İt.3 bile avg_sharpe = +0.068 idi — 0'ın hemen üstü. 15 trial'ın tümünün başarısız olması, eşiğin gerçekçi olmadığını gösteriyor.

```python
# Daha gerçekçi prune eşikleri
SHARPE_PRUNE_THRESHOLD = -0.3  # 0.0 yerine -0.3 (quality gate minimumu ile aynı)
FOLD_SHARPE_PRUNE_THRESHOLD = -1.0  # -0.5 yerine -1.0

if avg_sharpe < SHARPE_PRUNE_THRESHOLD and not protect_trial:
    raise optuna.exceptions.TrialPruned()

if fold_sharpe < FOLD_SHARPE_PRUNE_THRESHOLD and fold_idx >= 1 and not protect_trial:
    raise optuna.exceptions.TrialPruned()
```

**Değişiklik C — MedianPruner startup artırma:**
```python
# n_startup_trials=5 → n_startup_trials=max(5, n_trials//3) 
# 15 trial ile en az ilk 5 trial MedianPruner'dan korunur
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=max(5, n_trials // 3),
    n_warmup_steps=1,  # 5→1: fold bazında prune daha erken başlasın ama trial bazında değil
)
```

---

### Değişiklik 2: Default Config'i Bilinen-İyi Config ile Değiştirme

> **Amaç:** Hiçbir trial tamamlanmadığında kullanılan "default config"in, kalibrasyon-ağırlıklı bir config değil, kanıtlanmış yön-ağırlıklı bir config olmasını sağlamak.

#### [MODIFY] [config.py](file:///d:/dev/projects/copper-mind/backend/deep_learning/config.py)

İt.4'ün Optuna-optimized parametrelerini default olarak ayarla:

```python
@dataclass(frozen=True)
class ASROConfig:
    # İt.4 (20 Nisan) Optuna-optimized — başarı kanıtlanmış config
    lambda_quantile: float = 0.25   # was 0.4; 0.25 gives 75% directional weight
    lambda_vol: float = 0.30        # was 0.35
    lambda_madl: float = 0.40       # was 0.25; MADL proven effective at 0.4
    lambda_crossing: float = 1.0    # unchanged
    risk_free_rate: float = 0.0
    sharpe_window: int = 20

@dataclass(frozen=True)
class TFTModelConfig:
    # İt.4 model parametreleri
    hidden_size: int = 48           # was 32; İt.4 used 48
    attention_head_size: int = 2    # unchanged
    dropout: float = 0.30           # was 0.20; İt.4 used 0.30
    hidden_continuous_size: int = 16  # unchanged
    learning_rate: float = 2e-4     # was 3e-4; İt.4 used 2.04e-4
    weight_decay: float = 5e-5      # was 1e-4; İt.4 used 5.03e-5
```

**Gerekçe:** Default config, "tüm trial'lar prune olduğunda" otomatik olarak kullanılan güvenlik ağıdır. Bu güvenlik ağının, kanıtlanmış en iyi config'e yakın olması, en kötü senaryoda bile kabul edilebilir sonuç üretme olasılığını artırır.

#### [MODIFY] [trainer.py](file:///d:/dev/projects/copper-mind/backend/deep_learning/training/trainer.py)

`_apply_optuna_results` fonksiyonuna "bilinen-iyi config" geri dönüş mekanizması:

```python
# Optuna sonuç yok → default config PLUS bilinen-iyi override
KNOWN_GOOD_CONFIG = {
    "hidden_size": 48,
    "attention_head_size": 2,
    "dropout": 0.30,
    "hidden_continuous_size": 16,
    "learning_rate": 2e-4,
    "weight_decay": 5e-5,
    "lambda_vol": 0.30,
    "lambda_quantile": 0.25,
    "lambda_madl": 0.40,
    "batch_size": 32,
}
```

---

### Değişiklik 3: Hyperopt Warm-Start — Son Bilinen-İyi Config'den Başla

> **Amaç:** Optuna'nın her koşuda sıfırdan aramaya başlamak yerine, önceki başarılı koşunun parametrelerini başlangıç noktası olarak kullanmasını sağlamak.

#### [MODIFY] [hyperopt.py](file:///d:/dev/projects/copper-mind/backend/deep_learning/training/hyperopt.py)

```python
def _enqueue_known_good_trial(study, base_cfg):
    """
    Optuna'nın ilk trial olarak İt.4'ün bilinen-iyi parametrelerini
    denemesini sağla. Bu, TPE'nin iyi bir başlangıç noktası almasını
    ve boşa prune edilen trial sayısını azaltmasını sağlar.
    """
    known_good = {
        "max_encoder_length": 50,
        "hidden_size": 48,
        "attention_head_size": 2,
        "dropout": 0.30,
        "hidden_continuous_size": 16,
        "learning_rate": 2e-4,
        "gradient_clip_val": 1.0,
        "weight_decay": 5e-5,
        "lambda_vol": 0.30,
        "lambda_quantile": 0.25,
        "lambda_madl": 0.40,
        "batch_size": 32,
    }
    study.enqueue_trial(known_good)
```

Bu sayede ilk trial'ın tamamlanma şansı çok yüksek olur; TPE de bu başarılı trial etrafında arama yapar.

---

### Değişiklik 4: Eğitim Diagnostik Metadata — Ölü Trial Otopsisi

> **Amaç:** "Neden tüm trial'lar prune edildi?" sorusunu otomatik yanıtlayan diagnostik bilgi toplamak.

#### [MODIFY] [hyperopt.py](file:///d:/dev/projects/copper-mind/backend/deep_learning/training/hyperopt.py)

`_build_result_payload` fonksiyonuna detaylı diagnostik ekleme:

```python
def _build_result_payload(study) -> dict:
    trial_state_counts = _trial_state_counts(study)
    best = _best_finite_completed_trial(study)

    # Prune nedenlerini topla
    prune_reasons = {"sharpe_prune": 0, "crossing_prune": 0, "median_prune": 0, "fold_sharpe_prune": 0, "error": 0}
    fold_metrics = []
    for trial in study.trials:
        if trial.state.name == "PRUNED":
            # User attributes'dan prune nedenini çıkar
            reason = trial.user_attrs.get("prune_reason", "median_prune")
            prune_reasons[reason] = prune_reasons.get(reason, 0) + 1
        # Fold metriklerini topla (başarılı veya başarısız)
        for key in ("avg_variance_ratio", "avg_directional_accuracy", "avg_val_sharpe",
                     "avg_quantile_crossing_rate", "avg_median_sort_gap"):
            if key in trial.user_attrs:
                fold_metrics.append({
                    "trial": trial.number,
                    "state": trial.state.name,
                    key: trial.user_attrs[key],
                })

    if best is None:
        return {
            "status": "no_finite_completed_trials",
            "best_trial": None,
            "best_value": None,
            "best_params": {},
            "n_trials": len(study.trials),
            "trial_state_counts": trial_state_counts,
            "prune_reasons": prune_reasons,
            "fold_diagnostics": fold_metrics,
            "message": (
                "No Optuna trials completed with a finite objective value; "
                "final training will use the known-good fallback config "
                "(İt.4 parameters: lambda_quantile=0.25, lambda_madl=0.40)."
            ),
        }
    ...
```

Ayrıca prune noktalarında `trial.set_user_attr("prune_reason", ...)` eklenmeli:

```python
# Per-fold Sharpe prune noktasında:
trial.set_user_attr("prune_reason", "fold_sharpe_prune")
raise optuna.exceptions.TrialPruned()

# Cross-fold avg Sharpe prune noktasında:
trial.set_user_attr("prune_reason", "sharpe_prune")
raise optuna.exceptions.TrialPruned()

# Quantile incoherence prune noktasında:
trial.set_user_attr("prune_reason", "crossing_prune")
raise optuna.exceptions.TrialPruned()
```

---

### Değişiklik 5: Curriculum Warmup'ı Hyperopt İçinde de Aktifleştir

> **Amaç:** Hyperopt trial'larında curriculum warmup olmaması, fold metriklerini bozuyor. Model ilk epoch'lardan itibaren yön sinyali almaya çalışıyor ama henüz kalibre olmamış — bu, erken fold'larda Sharpe'ı yapay olarak düşürüyor ve prune tetikliyor.

#### [MODIFY] [hyperopt.py](file:///d:/dev/projects/copper-mind/backend/deep_learning/training/hyperopt.py)

Hyperopt trial'larına da curriculum callback ekle:

```python
from deep_learning.training.callbacks import CurriculumLossScheduler

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=trial_cfg.training.early_stopping_patience,
        mode="min",
    ),
    # Hyperopt trial'larında da curriculum warmup — fold metriklerinin
    # erken-epoch noise'dan etkilenmesini önler
    CurriculumLossScheduler(
        warmup_epochs=5,  # Trial'lar kısa, warmup da kısa
        initial_lambda_quantile=0.55,
        target_lambda_quantile=trial_cfg.asro.lambda_quantile,
        initial_lambda_madl=0.10,
        target_lambda_madl=trial_cfg.asro.lambda_madl,
    ),
]
```

**Gerekçe:** Final trainer'da curriculum warmup var ama hyperopt trial'larında yok. Bu tutarsızlık, hyperopt'un fold metriklerini farklı bir rejimde ölçmesine neden oluyor. Hyperopt sonrası seçilen config, final trainer'da farklı davranıyor.

---

## 4. Değişen Dosya Haritası

```text
backend/deep_learning/training/hyperopt.py   ← Değişiklik 1, 3, 4, 5
backend/deep_learning/config.py              ← Değişiklik 2
backend/deep_learning/training/trainer.py    ← Değişiklik 2
backend/tests/deep_learning/test_hyperopt.py ← Yeni: prune koruması testleri
```

---

## 5. Değişikliklerin Etkileşim Tablosu

| | Dğş.1 (Prune koruması) | Dğş.2 (Default config) | Dğş.3 (Warm-start) | Dğş.4 (Diagnostik) | Dğş.5 (Curriculum) |
|---|---|---|---|---|---|
| 15/15 prune engeller | ✅ Doğrudan | ✅ Dolaylı | ✅ Doğrudan | ❌ İzleme | ✅ Dolaylı |
| Kötü default config | ❌ | ✅ Doğrudan | ✅ Dolaylı | ❌ | ❌ |
| TPE keşif yetersizliği | ✅ Dolaylı | ❌ | ✅ Doğrudan | ❌ | ❌ |
| Erken-epoch metric noise | ❌ | ❌ | ❌ | ❌ | ✅ Doğrudan |
| Post-mortem analiz | ❌ | ❌ | ❌ | ✅ Doğrudan | ❌ |

> [!IMPORTANT]
> **Değişiklik 1 + 3 birlikte "sıfır tamamlanan trial" senaryosunu yapısal olarak ortadan kaldırır.** Değişiklik 2 ise "her şey başarısız olsa bile en kötü senaryoyu kabul edilebilir tutan" güvenlik ağıdır. Değişiklik 5 ise hyperopt-trainer tutarsızlığını giderir.

---

## 6. Risk Değerlendirmesi

| Risk | Olasılık | Etki | Azaltma |
|---|---|---|---|
| Korumalı trial'lar zayıf config seçebilir | Orta | TPE zayıf noktayı "en iyi" olarak seçer | `MIN_COMPLETED_TRIALS=3` ile en az 3 trial karşılaştırılır |
| Warm-start İt.4 config'i artık optimal olmayabilir | Düşük | TPE lokal minimuma takılır | Warm-start sadece 1 trial; kalan 14 trial serbestçe arar |
| Default config değişikliği mevcut testleri kırabilir | Düşük | Test assertion hataları | Testler de güncellenir |
| Curriculum warmup trial süresini uzatır | Düşük | CI süre artışı ~5 dk | Warmup sadece 5 epoch (trial toplam 25 epoch) |

---

## 7. Doğrulama Planı

### Otomatik Testler

```bash
# 1. Mevcut testlerin geçtiğini doğrula
py -m pytest backend/tests/deep_learning -q

# 2. Yeni testler: prune koruması
py -m pytest backend/tests/deep_learning/test_hyperopt.py -q

# 3. Syntax kontrolü
py -m py_compile backend/deep_learning/training/hyperopt.py
py -m py_compile backend/deep_learning/config.py
py -m py_compile backend/deep_learning/training/trainer.py
```

### Manuel Doğrulama (CI Koşusu)

Değişiklikler uygulandıktan sonra:
```bash
gh workflow run tft-training.yml -f run_hyperopt=true -f hyperopt_trials=15
```

Beklenen sonuç:
- `no_finite_completed_trials` **artık oluşmamalı**
- En az 3 trial tamamlanmalı
- Quality gate PASS veya anlamlı bir FAIL nedeni (DA/Sharpe/VR)

---

## Açık Sorular

> [!IMPORTANT]
> 1. **`MIN_COMPLETED_TRIALS` değeri kaç olmalı?** 3 mü 5 mi? 3 daha güvenli (daha az süre), 5 daha güvenilir (daha iyi karşılaştırma).
> 2. **Warm-start config'ini statik mi kodlayalım, yoksa son başarılı koşunun `optuna_results.json`'ını cache'leyip oradan mı okuyalım?** Cache yaklaşımı daha esnek ama ek dosya yönetimi gerektirir.
> 3. **Default config'i İt.4'e güncellemek mevcut production checkpoint'i etkiler mi?** Checkpoint'in kendisi modelin ağırlıklarını içerir, config sadece yeni eğitimleri etkiler — sorun olmaz.
