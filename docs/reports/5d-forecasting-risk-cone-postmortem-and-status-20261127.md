# 5D Tahminleme, Risk Konisi ve Piyasa Stresi Katmanı: Sorunlar, Benchmarklar, Kod Değişiklikleri ve Durum Raporu

**Tarih:** 2026-11-27  
**Kapsam:** COMEX Bakır (`HG=F`) 5 günlük (`5D`) ana tahmin hattı (`65% Chronos-2 + 35% TFT-ASRO`), Risk Konisi / Oynaklık / Stres katmanı (`volatility_stress.py`, `conformal.py`) araştırması, V4 ve V5 Colab L4 benchmarkları ve ürün gerçekliği değerlendirmesi.  
**Kod Tabanı Durumu:** Son commit `b61c336`; üzerine **henüz commit edilmemiş (uncommitted)** V2 `EWMA-0.94 + HAR-YZ-Rough-F1 + FF-NexCP` motoru, mutlak volatilite (`vol5_p75 / vol5_p90`) rejim etiket koruması ve `5D` gürültü bandı (`±%0.5` `NO CLEAR EDGE` + `±1σ` salınım / risk bütçesi) dürüstleştirmesi working tree'de hazır bekliyor (`937/937` offline pytest ve `272/272` frontend test geçiyor).

---

## 1. Yönetici Özeti ve Gelinen Son Durum

Bu süreç `5D` ana tahmin hattındaki **"küçük nokta tahmin (`0.04%–0.3%`) ve kör boğa/gerçeklik uyuşmazlığı"** şikâyetiyle başladı, ardından:
1. Günlük (`T+1`) XGBoost'un UI/cron hattındaki kafa karıştırıcı sürücü rolü ve gereksiz rutin snapshotları emekli edildi; `5D` yön motoru **%65 `Chronos-2` + %35 `TFT-ASRO`** ve **nedensel `HAR-GJR` v1 + `Split CQR`** yapısına çekildi (`b61c336`).
2. Risk konisi ve stres katmanı için **önce klasik/ML odaklı Benchmark V4 (Colab L4, `223.8s`)**, ardından 2022–2026 kurumsal bankacılık/fon standartlarını (`FRTB`, `Acerbi-Székely Z2`, `Bayer-Dimitriadis ESR`, `Du-Escanciano`, `twCRPS`, `DtACI`, `Conformal PID`, `GINN`, `Mamba-Lite SSM`, `XGBoostLSS`, `Rough Volatility`, `DSR`, `CSCV PBO`) ve bağlı `65/35` portföy testini içeren **Benchmark V5 (Colab L4, `790.0s`)** tamamlandı.
3. Matematiksel olarak 4 sütunda da rakiplerini ve eski `v1` sistemi yenen **V2 `EWMA-0.94 + HAR-YZ-Rough-F1 + FF-NexCP` ("saf JSON")** mimarisi `backend/` kod tabanına entegre edildi ve **934 offline testin tamamından geçti**.
4. **Ancak** entegrasyon sonrası `2,000` OOS günü üzerinde tekil tarih örneklerini (`2019–2026`) yan yana koyup ürün gözüyle denetlediğimizde kritik bir gerçekle yüzleştik: **Ekonometrik/istatistiksel skorlar (`Winkler-80`, `FRTB`, `Sharpe`) iyileşse bile, 5 günlük fiyat-bazlı bir sistemin ürettiği `q50 = ±%0.26` ve `PI96 = [-%10, +%12]` çıktısı son kullanıcı için hâlâ yanıltıcı bir "ekonomist tahmini" görüntüsü veriyor.**

Bu rapor yaşanan tüm sorunları, yakalanan metodolojik hataları, V4/V5 benchmark sonuçlarını, working tree'de yapılan somut kod değişikliklerini ve bundan sonraki yol haritası hakkındaki net mühendislik görüşünü özetler.

---

## 2. Yaşanan Sorunlar ve Kök Nedenler

### 2.1. `q50` Genliğinin Küçük Kalması (`%0.04–%0.3` vs Gerçek `±%2–%11`)
- **Sorun:** Model `+%0.1` / `+%0.3` gibi çok küçük haftalık getiriler tahmin ederken, gerçek piyasada bakır haftalıkta düzenli olarak `-%3`, `-%9`, `+%5`, `+%11` gibi hareketler yapıyor.
- **Kök Neden:** Test ettiğimiz tüm modeller (`Chronos-2`, `TFT-ASRO`, `TimeXer`, `XGBoost`, `CatBoost`, `BiGRU`, `GINN`, `Mamba`) 5 günlük bakır getirilerinde **`%52–%55` yön isabeti** (`R² ≈ 0.01–0.03`) duvarına çarpıyor. Bu gürültü/sinyal oranında MAE/Pinball kaybını minimize eden **her model**, beklenen medyanı sıfır çevresine (`|q50|` medyanı **`%0.26`**, 90. yüzdeliği **`%0.91`**) çeker. Yani bu bir "bug" değil, yalnızca fiyat serisinden 5 gün sonrasını tahmin etmenin matematiksel sınırıdır.
- **Önemli Ayırt Edici Bulgu:** Modelin yön gücü (`|q50|`) büyüdüğünde isabet gerçekten yükselmektedir (`2,000` OOS günü):
  - `|q50| < %0.3` (1,099 gün — günlerin %55'i): Yön isabeti **`%52.0`** (neredeyse yazı-tura).
  - `%0.3 <= |q50| < %1.0` (731 gün): Yön isabeti **`%53.9`**.
  - `|q50| >= %1.0` (170 gün — günlerin %8.5'i): Yön isabeti **`%55.3`**, tahmin yönünde ortalama getiri `+%0.25`.

### 2.2. Günlük XGBoost ile Haftalık TFT Çelişkisi
- **Sorun:** Günlük `T+1` XGBoost ile `5D` TFT-ASRO'nun sürücü açıklamaları ve yönleri birbirinden kopup kullanıcıyı yanıltıyordu.
- **Çözüm (`b61c336`):** Rutin günlük XGBoost snapshot/retrain ve çelişkili Overview UI sürücüleri emekli edildi; `5D` (`%65 Chronos-2 + %35 TFT-ASRO`) tek ana yön ufku olarak netleştirildi.

### 2.3. Benchmark V4 Sırasında Yakalanan Metodolojik Hatalar
V4 ilk koşusu audit edildiğinde 4 önemli tuzak tespit edildi:
1. **Açık uç satır `NaN` sızıntısı:** Fold 7'nin son 5 günü (`2026-11-12..18`) için ileri 5D Yang-Zhang varyansı henüz kapanmadığından `NaN` kalmış, maskelenmediği için özet skorları `NaN`'a düşürmüştü (düzeltildi).
2. **`arch_model` yeniden kurma hatası:** GARCH ailesi (`GJR-GARCH-t`, `EGARCH-t`, `FIGARCH-t`) instance parametresi geçildiği için sessizce sabit varyans fallback'ine düşüyordu (düzeltildi).
3. **Medyan Hizalama (Jensen Eşitsizliği) Yanılgısı:** Sağa-çarpık varyansta (`mean = 0.001255`, `median = 0.000758`) tahminleri `val` **medyanına** hizalamak GARCH/GBDT/NN modellerinin ortalamasını ~%40 aşağı bastırdı ve QLIKE kaybını suni olarak `+0.13..+0.18` şişirdi. V5'te tüm modeller `val` üzerinde **QLIKE-optimal ortalama oranı (`c* = mean(y_va / v_va)`)** ile hizalanarak gerçek performans ortaya çıkarıldı.
4. **`Split CQR` Çift Ölçekleme vs Körleşme:**
   - `raw_already_vol_scaled=True` iken `65/35`'in ham bant genişliği volatiliteyle hiç çarpılmadığı için iyi bir oynaklık modelinin etkisi bastırılıyordu (`HiV = %58.1`).
   - Tersine eski üretimdeki tam çarpma (`gamma = 1.0`) zaten `Chronos-2` içinde bulunan adaptif genişliği ikinci kez çarptığı için volatiliteyi çift sayıyor ve `65/35` portföyünün Sharpe'ını `0.539`'dan `0.257`'ye düşürüyordu.

### 2.4. "Benchmarkmaxxing" Öz-Eleştirisi ve Kullanıcı Gözünden Ürün Gerçekliği
V5 sonrası gerçek OOS tarih örnekleri masaya yatırıldığında 3 temel ürün zayıflığı net görüldü:
1. **`Winkler-80: 0.1288 → 0.1280` (%0.6) ve `PI96` yanılsaması:** Kağıt üzerinde tüm düzenleyici testleri (`FRTB GREEN`, `Acerbi-Székely Z2`, `Bayer-Dimitriadis ESR`, `%90 MCS`) geçmesine rağmen, kullanıcıya `q50 = +%0.17` ile birlikte `PI96 = [-%10.2, +%12.6]` gibi devasa bir bant sunup gerçekleşen `+%11` hareketi "bakın aralıkta tuttu" diye pazarlamak pratikte işe yaramıyor.
2. **Büyük Şok Haftalarında Koni Gecikmesi:** Bakırın `|R_5d| >= %4` oynadığı sert haftalarda `PI80` yalnızca **`%19.4`** kapsama sağlayabildi. Yani ani makro şoklarda (`2020-03-12 -%11.33`, `2022-06-15 -%9.81`, `2025-04-03 -%10.18`) koni ilk vuruşu önceden bilmiyor; ancak şok yaşandıktan **1–5 gün sonra** (`2020-03-19` `%14.1`, `2025-04-08` `%13.5`) açılıp sonraki dalgaları yakalayabiliyor.
3. **Rejim Etiketi Çelişkisi (`stress_ratio` tuzağı):** Mevcut etiketleme sadece `stress_ratio = hat_vol_5d / rv20_vol5` oranına baktığı için, oynaklık zaten çok yükseldiği `2020-03-19` gününde (`hat_vol_5d = %4.19`, koni genişliği `%14.1`) 20 günlük taban da yükselmiş olduğundan `stress_ratio = 0.94` çıkmış ve ekrana yanıltıcı şekilde **`CALM`** etiketi basmıştır.

---

## 3. Yapılan Benchmarklar ve Sayısal Bulgular

Tüm değerlendirmeler `fold_manifest.json` üzerindeki **8-fold walk-forward (`2018-12-10` → `2026-11-18`, `2,000` OOS işlem günü, `400` çakışmasız haftalık blok)** üzerinde, ileri bakış sızıntısı olmadan (`idx=2000` mutasyon testinde `max_diff = 0.00e+00`) yürütülmüştür.

### 3.1. Benchmark V4 (Colab L4, `223.8s`) — Temel Bulgular
- **Test edilen aileler:** `Static RV20`, `EWMA-0.94`, `GJR-GARCH-t`, `EGARCH-t`, `FIGARCH-t`, `HAR-RV F0`, `HAR-GJR F0/F1/F2`, `LightGBM QLIKE F1/F2`, `CatBoost LogVol F2`, `XGBoost QLIKE F2`, `GARCH-LSTM`, `BiGRU-Attention-t`, `DeepVol Dilated TCN-t`, `PatchTST-Vol` ve `Split CQR / NexCP / Delayed ACI / Asym-CQR / Student-t Asym-CQR`.
- **Sonuç:** `Chronos-2 + TFT-ASRO`'nun ham kantil dağılımını tamamen atıp (`lam=0.0`) yerine saf Student-t bandı koymak sol-kuyruk `FZ0` puanını ciddi biçimde bozdu (`-1.75 → -0.76`). Buna karşılık `Chronos-2 + TFT-ASRO` ile Student-t'yi harmanlamak + sönümlü ölçek (`gamma=0.40..0.55`) hem `Winkler-80`'i hem kuyruk skorunu iyileştirdi.

### 3.2. Benchmark V5 (Colab L4, `790.0s`) — Kurumsal 4 Sütun ve Makale Düzeltmeleri
V5'te `v5_dataset.parquet` (`3507 x 118`, `F1_rough=46`, `F2_rough=71`) kuruldu: günlük OHLC'den nedensel **Rough Volatility eğimi (`rough_hurst_proxy`)**, gece sıçrama oranı, **SHAR semivaryans asimetrisi**, **HARQ 4. moment** ve çoklu-ölçek **EWMA** eklendi; `realized_vol_20d` ölçeği düzeltildi (`true_rv20_daily mean = 0.01375`).

#### Sütun 1 — Volatilite Tahmin Gücü (`N = 1,995`, Patton QLIKE)
| Model | `QLIKE_YZ` ↓ | `QLIKE_CC` ↓ | Vol `MZ-R2` ↑ | 90% MCS |
| :--- | :---: | :---: | :---: | :---: |
| **`f1_ewma_har_70_30` (70% EWMA-0.94 + 30% HAR-YZ-Rough-F1)** | **0.3761** | **0.4960** | 0.049 | — |
| **`ewma_094`** | **0.3776** | 0.5075 | 0.039 | **Evet (1.00)** |
| **`f1_ewma_har_50_50` (50% EWMA-0.94 + 50% HAR-YZ-Rough-F1)** | **0.3879** | **0.5043** | **0.053** | — |
| `true_rv20` | 0.4010 | 0.5572 | 0.045 | **Evet (1.00)** |
| `prod_har_gjr_f0` (Eski V1 üretim) | 0.4433 | 0.5595 | 0.039 | **Evet (1.00)** |
| `v5_ens_full_ginn_mamba` (L4 GPU Ansambl) | 0.4593 | 0.5724 | 0.035 | **Evet (1.00)** |
| `v5_ens_econ_gbdt` (Sıfır NN Ansambl) | 0.4596 | 0.5659 | 0.038 | **Evet (1.00)** |
| `ginn_bigru_skewt_f2` / `mamba_ssm_vol_t_f2` / `xgb_lss_studentt_f2` | 0.4771 / 0.5078 / 0.5324 | 0.6118 / 0.6291 / 0.6408 | 0.028 / 0.026 / 0.025 | **Evet** |
| `gjr_garch_t` | 0.5375 | 0.6649 | 0.021 | **Hayır (0.01)** |

*Not:* L4 GPU'da 3-seed eğitilen `GINN-BiGRU-SkewT` + `Mamba-Lite SSM` + `XGBoostLSS Student-t` gibi ağır modeller, saf JSON **`50/50 EWMA-0.94 + HAR-YZ-Rough-F1`** kompozitini ne QLIKE'da ne $R^2$'de geçebilmiştir.

#### Sütun 2 — Düzenleyici FRTB / Expected Shortfall / `twCRPS` / Conformal
| Mimari (`Vol + Conformal`) | `W80` ↓ | `FZ0` ↓ | `PI80` | `HiV` | `T4` | `FRTB` | `Z2` (AS) | `ESRp` (BD) | `CCp` | MCS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `RAW 65/35 (Uncalibrated)` | 0.1272 | -1.681 | 0.785 | 0.581 | 0.140 | GREEN (22) | +2.50 | **0.00 (KALDI)** | 0.12 | Y/Y |
| `PROD: har_gjr_f0 + split_cqr` | 0.1288 | -1.736 | 0.801 | 0.602 | 0.197 | GREEN (16) | +2.23 | 0.34 | 0.45 | N/Y |
| `v5_ens_econ_gbdt \| vanilla_dtaci` (Gibbs & Candès 2024) | 0.1312 | -1.707 | 0.787 | 0.634 | 0.249 | GREEN (22) | +2.29 | **0.00 (KALDI)** | 0.17 | N/Y |
| `v5_ens_econ_gbdt \| vanilla_pid` (Angelopoulos 2023) | 0.1346 | -1.683 | 0.798 | 0.642 | 0.277 | GREEN (14) | +2.19 | **0.05** | 0.74 | N/Y |
| `v5_ens_econ_gbdt \| ff_nexcp` | **0.1266** | -1.740 | 0.785 | 0.612 | 0.181 | GREEN (16) | +2.22 | **0.65** | 0.11 | **Y/Y** |
| **`50/50 EWMA+HAR_F1 \| ff_nexcp` (Birleşik V2)** | **0.1280** | **-1.743** | **0.790** | **0.624** | **0.194** | **GREEN (15)** | **+2.18** | **0.51** | **0.28** | **Y*** |

*\*`l=0.75, g=0.40, sb=0.40` konfigürasyonunda `%90 MCS p=1.00`; birleşik `cqr_vol_scale (sb=0.55)` ile aynı `W80=0.1280, ESRp=0.51, CCp=0.28, FRTB=GREEN(15/400)` korunmuştur.*

#### Sütun 3 — `65/35` Yön Motoru × Risk Konisi Bağlı Sizing Backtesti (`400` Çakışmasız Hafta, Net 4–8 bps)
| Strateji | `Sharpe (4bp)` | `Sharpe (8bp)` | `Sortino` | `Calmar` | `CAGR` | `MDD` | `DSR` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Unscaled 65/35 Sign` | 0.722 | 0.722 | 1.06 | 0.42 | +16.1% | **-38.7%** | 0.902 |
| `Moreira-Muir RV20 Vol-Target` | 0.539 | 0.472 | 0.89 | 0.27 | +4.6% | -17.1% | 0.807 |
| `PROD (eski V1)` | 0.257 | 0.179 | 0.36 | 0.10 | +1.4% | -14.5% | 0.519 |
| **`har_yz_rough_f1 \| ff_pid`** | **0.572** | **0.495** | **0.91** | **0.40** | +3.7% | **-9.2%** | **0.826** |
| **`50/50 EWMA+HAR_F1 \| ff_nexcp` (Entegre V2)** | **0.508** | **0.429*** | **0.74*** | **0.29*** | ~+3.1%* | **-11.9%** | ~0.77 |

*21 strateji genelinde **CSCV PBO (`S=16`):** `0.2570` (`*sb=0.40` varyantı `Sh4=0.505, Sh8=0.429, Cal=0.29, MDD=-11.7%`). Dürüst not: Risk-ayarlı Sharpe eski üretimin 2 katına çıkmış olsa da mutlak getiri (`~%3.1/yıl`) mütevazıdır ve kazanç yön tahmininin keskinleşmesinden değil, yüksek volatilitede risk kısılmasından gelir.*

#### Sütun 4 — Stres / Şok Erken Uyarı (`true_rv20` Düzeltmeli, `N = 2,000`)
- Göreli Şok (`|R_5d| >= 1.75*RV20_5d`, prev=`%10.2`): `prod_har_gjr_f0` `AUC=0.6580 (Lift=2.36x)`, `har_yz_rough_f1` `AUC=0.6410 (Lift=1.97x)`.
- 5D Volatilite Patlaması (`Vol-Surge >= 1.35*YZ22`, prev=`%14.1`): `har_yz_rough_f1` `AUC=0.5996 (Top10% Lift=2.02x)` vs eski üretim `0.5667 (1.28x)`.

### 3.3. Üretim V2 Kodunun Somut Tarihsel OOS Örnekleri (`N = 2,000`, Genel `DA = %52.95`, `PI80 = %78.30`, `PI96 = %95.85`)
| Senaryo & Tarih | Rejim / 5D Vol | `q50` | `PI80` (Genişlik) | `PI96` | **Gerçek 5D** | Gözlem |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Sakin Yatay** (`2019-07-15`) | `CALM` / `%2.41` | `-0.21%` | `[-2.6%, +2.3%]` (`%4.9`) | `[-5.3%, +5.3%]` | **`+0.26%`** | Dar ve temiz koni (`Hit80=Evet`) |
| **Sakin Yatay** (`2023-10-16`) | `CALM` / `%2.65` | `+0.06%` | `[-2.7%, +3.0%]` (`%5.7`) | `[-5.3%, +5.6%]` | **`-0.39%`** | Gereksiz genişleme yok (`Hit80=Evet`) |
| **COVID İlk Patlama** (`2020-03-12`) | `HIGH_STRESS` / `%2.54` (`1.27x`) | `+0.17%` | `[-3.3%, +3.5%]` (`%6.8`) | `[-6.1%, +6.4%]` | **`-11.33%`** | Stress alarmı yanmış ama şoku kestiremez (`Hit=Hayır`) |
| **COVID Zirve** (`2020-03-19`) | `CALM`* / `%4.19` | `-0.35%` | `[-7.9%, +6.1%]` (`%14.1`) | `[-12.9%, +11.3%]` | **`+0.36%`** | Koni ~3× açılmış; ancak etiket `sr=0.94` ile hatalı `CALM` yazmış |
| **2022 Fed/Çin** (`2022-06-15`) | `CALM` / `%3.76` | `+0.93%` | `[-2.8%, +5.5%]` (`%8.2`) | `[-7.0%, +10.5%]` | **`-9.81%`** | Ani makro kırılmada hem yön hem bant ıska |
| **2024 Mayıs Sıkışma** (`2024-05-15`) | `CALM` / `%2.96` | `-0.26%` | `[-4.3%, +4.1%]` (`%8.3`) | `[-6.6%, +6.6%]` | **`-2.05%`** | Dönüş yönü ve bant tutarlı (`Dir=Evet, Hit80=Evet`) |
| **2024 Sonrası Satış** (`2024-05-21`) | `CALM` / `%3.07` | `-0.23%` | `[-4.3%, +4.1%]` (`%8.4`) | `[-6.7%, +6.6%]` | **`-6.12%`** | Yön doğru, satış `PI80` dışı / `PI96` içi |
| **2025 Tarife Şoku** (`2025-04-03`) | `CALM` / `%3.68` | `-0.99%` | `[-6.1%, +4.4%]` (`%10.6`) | `[-9.9%, +8.7%]` | **`-10.18%`** | Yön güçlü doğru; şok `PI96` sınırını da az farkla aşmış |
| **2025 Tarife Tepkisi** (`2025-04-08`) | `ELEVATED` / `%5.95` | `+0.02%` | `[-6.3%, +7.3%]` (`%13.5`) | `[-11.3%, +13.5%]` | **`+11.71%`** | Tepki bandı açık, `PI96` içinde kalmış |
| **2026 Son Dönem** (`2026-11-05`) | `HIGH_STRESS` / `%4.75` | `+0.45%` | `[-2.8%, +4.0%]` (`%6.8`) | `[-13.1%, +16.3%]` | **`-1.06%`** | `PI80` dar kalırken `PI96` kuyruğu `[-13%, +16%]` açılmış |

---

## 4. Kod Tabanında Oluşan Değişiklikler (Uncommitted Working Tree)

Bu iyileştirmelerde **hiçbir ekstra `.pt` ağırlık dosyası veya `^VIX/^GVZ/^OVX` dış veri bağımlılığı eklenmedi**. "Saf JSON" yaklaşımıyla modelin tüm öğrenilmiş katsayıları düz sayısal listeler halinde JSON artefaktlarına yazılır ve canlı inference sadece NumPy + 1 satırlık `EWMA-0.94` ile çalışır.

Çalışma ağacındaki 16 değiştirilmiş dosya (backend kalibrasyon, şema, worker, yorum motoru, testler ve frontend `OverviewPage`):
1. [`backend/deep_learning/data/feature_store.py`](file:///d:/dev/projects/copper-mind/backend/deep_learning/data/feature_store.py): `PriceBar`'daki `target_open`, `target_high`, `target_low`, `target_volume` sütunları `master_df` içine güvenli aktarıldı; `non_feature_cols` listesine eklenerek `TFT` encoder/MRMR girdi kümesi tamamen korundu.
2. [`backend/deep_learning/calibration/volatility_stress.py`](file:///d:/dev/projects/copper-mind/backend/deep_learning/calibration/volatility_stress.py): `version: 2` (`ewma_har_yz_rough_v2_json`) motoru — %50 nedensel `EWMA-0.94` + %50 `val` QLIKE-ortalama hizalı `HAR-YZ-SHAR-HARQ-Rough (F1)` + çift stres başlığı (`p_shock` + `p_vol_surge`) + mutlak volatilite yüzdelik koruması (`vol5_p75`, `vol5_p90`, `vol_percentile` — uzayan krizlerde `CALM` etiketi çelişkisini önler) + birleşik `cqr_vol_scale`, `risk_budget_mult` (`1.0 / 0.85 / 0.70`); OHLC yoksa close-only fail-open ve eski `version: 1` artefaktlarıyla %100 geriye uyumlu.
3. [`backend/deep_learning/calibration/conformal.py`](file:///d:/dev/projects/copper-mind/backend/deep_learning/calibration/conformal.py) & [`backend/deep_learning/training/trainer.py`](file:///d:/dev/projects/copper-mind/backend/deep_learning/training/trainer.py): `FF-NexCP` (`prior_blend_lambda=0.75`, `prior_student_t_df=6.0`, `base_vol_power=0.40`, `vol_clip=[0.45, 2.80]`, `nexcp_gamma=0.985`, `left_tail_extra=0.15`) kalibrasyonu ve `conformal_calibration.json` serileştirmesi; `q50` değişmezliği ve eski V1 uyumu korundu.
4. [`backend/deep_learning/inference/predictor.py`](file:///d:/dev/projects/copper-mind/backend/deep_learning/inference/predictor.py), [`backend/app/schemas.py`](file:///d:/dev/projects/copper-mind/backend/app/schemas.py), [`backend/worker/tasks.py`](file:///d:/dev/projects/copper-mind/backend/worker/tasks.py) & [`backend/app/commentary.py`](file:///d:/dev/projects/copper-mind/backend/app/commentary.py): `5D` gürültü bandı (`neutral_band_pct=0.5`, `signal_state`, `has_directional_edge`) ile düşük genlikli günlerde zorlama boğa/ayı yorumu üretilmesi engellendi; `±1σ` beklenen 5D salınım ve `risk_budget_mult` yorum/telemetri hattına bağlandı.
5. [`frontend/src/pages/OverviewPage.tsx`](file:///d:/dev/projects/copper-mind/frontend/src/pages/OverviewPage.tsx) & [`frontend/src/types.ts`](file:///d:/dev/projects/copper-mind/frontend/src/types.ts): `|q50| < %0.5` iken `NO CLEAR EDGE` nötr görünümü eklendi, `cuprumTelemetry` rejim okuması `tftSignalDir` ile hizalandı ve üst şeritte `T+1 TFT Step 1` yerine `5D Risk & Stress` (`±1σ` salınım + pozisyon bütçesi) karosu devreye alındı.
6. Test dosyaları — [`test_har_gjr_volatility_stress.py`](file:///d:/dev/projects/copper-mind/backend/tests/test_har_gjr_volatility_stress.py), [`test_conformal_calibration.py`](file:///d:/dev/projects/copper-mind/backend/tests/deep_learning/test_conformal_calibration.py), [`test_pipeline_overhaul_regressions.py`](file:///d:/dev/projects/copper-mind/backend/tests/test_pipeline_overhaul_regressions.py), [`test_tft_priority_commentary.py`](file:///d:/dev/projects/copper-mind/backend/tests/test_tft_priority_commentary.py), [`test_worker_tft_explain_wiring.py`](file:///d:/dev/projects/copper-mind/backend/tests/test_worker_tft_explain_wiring.py), [`OverviewPage.test.tsx`](file:///d:/dev/projects/copper-mind/frontend/src/pages/OverviewPage.test.tsx): V2 nedensellik, uzayan krizde `CALM` koruması, `v1` geriye uyumluluk ve `NO CLEAR EDGE` UI/yorum dürüstlüğü doğrulandı (`937 passed` backend, `272 passed` frontend, `0` lint uyarısı, bundle bütçeleri yeşil).

---

## 5. Mühendislik Görüşü ve Önerilen Sonraki Adımlar

1. **5D Fiyat-Bazlı Model Kovalamayı Durdurmak:** `Chronos-2`, `TFT-ASRO`, `TimeXer`, `GINN`, `Mamba`, `XGBoost` ve `CatBoost` ile ulaşılan tavan nettir — yalnızca geçmiş fiyatlarla 5 günlük bakır getirisi **~%53–55 yön isabetini** ve **±%0.26 medyan genliği** aşamamaktadır. Aynı 5D fiyat verisi üzerinde yeni mimari veya ekstra 4. ondalık benchmark aramanın getirisi sıfıra yakındır.
2. **5D Ekranının Dürüstleştirilmesi (Working Tree'de Tamamlandı):**
   - Düşük güvenli günlerde (`|q50| < %0.5` veya nötr rejim) zorlama boğa/ayı rengi yerine açıkça **`NO CLEAR EDGE` / Nötr** gösterimi devreye alındı;
   - Üst şeritte ve ana kartta **"Beklenen Haftalık Normal Salınım (`±1σ`)"** ve pozisyon risk bütçesi çarpanı (`risk_budget_mult`) öne çıkarıldı;
   - Rejim etiketindeki `CALM` çelişkisi mutlak `hat_vol_5d` eğitim yüzdeliği (`vol5_p75 / vol5_p90`) kontrolüyle kapatıldı.
3. **Asıl Katma Değer Katmanlarına Geçmek:** Bakırda `±%5–%11` gibi büyük ve karar değiştirici hareketleri 5 günlük grafik değil;
   - **Anlık Katalizör / Olay-Şok Katmanı** (haberlerden fiyat regresyonu uydurmadan; tarife şokları, Çin teşvikleri, maden arz kesintileri ve stok çekilmelerini `24s/72s` olay kümesiyle yakalayıp rejim uyarısı üreten *Nowcast* katmanı) ve
   - **Orta-Uzun Vade Makro & Fiziksel Katmanı (`long_term/`, 20D/60D)** (mevcut `Macro Ridge + Damped Theta` iskeletine ücretsiz CFTC COT pozisyonlanması, `HG/GC` oranı ve gerçek vade yapısı eklenmesi) belirler.
4. **Açık Karar:** Mevcut working tree'deki V2 `EWMA + HAR-YZ-Rough + FF-NexCP` iyileştirmeleri, `vol5_p75/p90` rejim koruması ve `5D` Nötr/Risk-Bütçesi dürüstleştirmesi test ve bundle bütçelerinden eksiksiz geçmektedir; doğrudan commit edilip ardından **Katalizör Şok Radarı** ve **Uzun Vade Fiziksel/COT** fazına geçilebilir.
