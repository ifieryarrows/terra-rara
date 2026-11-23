# TFT-ASRO Kök Neden Onarımı ve Uzun Vadeli Görünüm (`20D / 1M` & `60D / 1Q`) Mimarisi — 2026-11-21

| Alan | Değer |
| --- | --- |
| **Rapor Tarihi** | 21–22 Kasım 2026 |
| **Kapsam** | TFT-ASRO (`5D`) Uçtan Uca Onarım + Yeni Uzun Vadeli Görünüm (`20D / 1M` & `60D / 1Q`) + Dashboard Horizon/Gate Hizalaması |
| **Başlangıç Tabanı** | `main` @ `6759205` (ve canlı DB aktif satırı `dfb3b96`, `id=1`) |
| **Kalite Kapısı Sözleşmesi** | [`backend/app/quality_gate.py`](../../backend/app/quality_gate.py) — **sıfır eşik gevşetmesi / sıfır bypass** |
| **Canlı DB / HF Hub Durumu** | SALT-OKUNUR doğrulama (`0` uzak DB mutasyonu, `0` HF Hub push) |
| **Doğrulama Durumu** | ✅ Backend `918 passed` · Frontend `269 passed`, `0` lint uyarısı, bütçeler geçti · Canlı DB `3/3` seed `QUALITY GATE: PASSED` |

---

## 1. Yönetici Özeti ve Önce / Sonra Karşılaştırması

### Neden TFT-ASRO Kırıktı?

CopperMind'ın 5 işlem günlük (`5D`) birincil modeli **TFT-ASRO**, 8 aylık geçmiş boyunca (`Mart–Kasım 2026`, son 25 koşunun **22'sinde başarısızlık**, zamanlanmış **13/13 Pazar cron koşusunun tamamında kırılma**) dört patoloji arasında salınmaktaydı: tek yönlü boğa sapması, işaret terslenmesi, büyüklük patlaması veya sıfıra çökme (`WeeklyMagnitudeCollapse`) ile PI80 kalibrasyon bozukluğu. İnceleme başlangıcındaki durum iki katmanlı bir kırılmayı ortaya çıkardı:

1. **`HEAD` (`6759205`, CI run `44797154947`) — Çalışma Zamanı Çöküşü ve Sıfır-Çökme:**
   - [`backend/deep_learning/training/trainer.py`](../../backend/deep_learning/training/trainer.py) içindeki çoklu-seed validasyon döngüsünde (`rec_idx >= 1`, seed `137`) `_predict_quantiles_to_np(..., return_y=False)` tek bir `np.ndarray` döndürürken kod 2 elemanlı tuple açmaya çalışarak `ValueError: too many values to unpack (expected 2)` fırlattı. Geniş `except Exception` bloğu bu hatayı yuttuğu için **seed filtreleme, checkpoint slot uzlaştırması, Theta hibrit harmanı ve Split CQR kalibrasyonunun tamamı sessizce atlandı**.
   - Aynı anda `WeeklyASROPFLoss` içindeki L1+L2 `lambda_drift_prior = 4.0` sıfır-ortalama cezası (Optuna uzayı `[2.5, 4.0, 6.0]` ile kilitli) gradyanları bastırarak tüm tahminleri sıfıra çökertti: `WeeklyMagnitudeCollapse=0.0082 < 0.02` (`weekly_raw_pred_min=-0.0003`, `max=+0.0005`, `weekly_variance_ratio=0.0067`, `T+1 DA=0.3802`, `T+1 Sharpe=-2.3974`, kalibresiz `weekly_pi80_coverage=0.8678`).
2. **Canlı Üretim DB'sindeki Aktif Model (`dfb3b96`, CI run `44752204561`, `tft_model_metadata.id=1`) — Negatif Öngörü Yeteneği:**
   - Yalnızca `4cbe855` commit'inde gevşetilmiş kapı koşulları altında geçebilmiş bu tek-seed (`ensemble_size=1`) model, temel referansların gerisinde kalıyordu: `weekly_directional_accuracy=0.5108 < weekly_actual_positive_rate=0.5269` (her zaman yükseliş tahmininden kötü), `weekly_pred_positive_rate=0.8548` (%85.5 boğa yanlılığı), `weekly_sign_correlation=0.0124`, `weekly_mae_vs_naive_zero=1.0039` (sıfır tahmininden kötü), `weekly_pinball_skill_vs_empirical=-0.0489`, `weekly_sharpe_ratio_hac=0.1643`, `weekly_variance_ratio=0.1187`, `weekly_magnitude_ratio=0.2352`.
   - VSN önem ağırlıklarının **%52'den fazlası** işaretsiz volatilite ve sabit zaman indeksine kilitlenmişti (`CEW_vol_10` %28.5, `relative_time_idx` %14.3, `DX_Y_NYB_vol_10` %5.3, `HG=F_vol_10` %4.2).
   - Canlı `backtest_reports` (`id=10`, `kind=tft_vs_xgb`, `n=260`): TFT `DA=54.62%` (= `baseline_always_up`), `Sharpe=0.6430`, `MCC=0.0384` iken Theta baseline `DA=60.77%`, `Sharpe=1.4318` ile öne geçiyordu (`verdict=MIXED`).

### Temiz Onarımlar Sonrası Önce / Sonra Karşılaştırması

Hiçbir kalite kapısı eşiği düşürülmeden veya atlanmadan yapılan düzeltmelerin ardından, canlı (salt-okunur) veritabanı verisi üzerinde geçici dizin (`TemporaryDirectory`) kullanılarak çalıştırılan uçtan uca 3-seed eğitim, kalite kapısı ve yürüyen ileri (`TFTBacktestRunner`) doğrulama sonuçları aşağıdaki gibidir:

| Metrik / Durum | Önce (`HEAD 6759205` — CI `44797154947`) | Önce (Canlı DB `dfb3b96` — `id=1` / `backtest id=10`) | Sonra (Onarılmış Stack — Canlı DB Salt-Okunur Doğrulama) |
| --- | --- | --- | --- |
| **Çoklu-Seed Kalibrasyon & Filtre** | Seed 2'de unpack çöküşü (CQR/Hybrid/Filtre atlandı) | `ensemble_size = 1` | **`3/3` seed kabul edildi** (`val_mae_vs_naive_zero`: `0.9482`, `0.9771`, `0.9850`) |
| **Kalite Kapısı (`quality_gate.py`)** | **FAIL** (`WeeklyMagnitudeCollapse=0.0082`, `PI80=0.8678`) | Yalnız gevşetilmiş kapı ile geçti; negatif beceri | **`PASS`** (`passed=True`, `reasons=[]`, sıfır eşik gevşetmesi) |
| **Holdout Weekly DA (`5D`)** | — (çökmüş / anlamsız sıfır-yakını) | `0.5108` (`actual_pos=0.5269` altında) | **`0.5690`** (`n=116`, `2026-06-08..2026-11-20`) |
| **Holdout Weekly Sharpe (HAC)** | — | `+0.1643` (düşük beceri / yanlı) | **`+0.1180`** (unadjusted `weekly_sharpe_ratio = +0.1881`) |
| **Holdout PI80 / PI96 Kapsama** | `0.8678` (ham kalibresiz RSS) | — | **`0.8534`** (`PI80`) / **`0.9655`** (`PI96`) |
| **Quantile Crossing (Post-CQR)** | — | Tautolojik çift-softplus ile maskelenmiş | **`0.0000`** (`ordered_pred` uzayı + post-CQR gate) |
| **Holdout T+1 Teşhis (DA / Sharpe)** | `0.3802` / `-2.3974` | — | **`0.5862`** / **`+2.3916`** |
| **Holdout `magnitude_ratio` / `MAE/0`** | `0.0082` (`Collapse`) | `0.2352` / `1.0039` | **`0.0713`** / **`1.0034`** (`pred_pos=0.8103`) |
| **Walk-Forward (`TFTBacktestRunner`)** | Çalıştırılamadı / uyumsuz CQR | `n=260`: `DA=54.62%`, `Sharpe=0.6430`, `MCC=0.0384` (`MIXED`) | Tam OOS (`n=216`): **`DA=0.5972`**, **`Sharpe=+0.6962`** (`TFT_SUPERIOR` 3–0 vs Theta) |
| **Strict Post-Val OOS (`TFTBacktestRunner`)** | — | Yok | **`n=119`** (düşüş rejimi `mean=-0.004846`): **`DA=0.5630`**, **`Sharpe=+0.0900`**, **`MCC=+0.1793`** (**`TFT_SUPERIOR` 2–1** vs Theta `DA=0.5378, Sharpe=-0.1342`) |
| **Uzun Vadeli Görünüm (`1M / 1Q`)** | Yok (`5D` ötesi tahmin yok) | Yok | **Yeni `backend/long_term/` motoru** (`20D DA=0.5174 / non-ovl 0.5401`, `60D DA=0.5363`, `PI80=82.8%/84.4%`, `Crossing=0.0`) |

---

## 2. Kök Neden Analizi (Faz 1 Bulguları)

Faz 1 denetimi, TFT-ASRO'nun tek bir hatadan değil, birbirini güçlendiren **beş katmanlı bir hata zincirinden** çöktüğünü kanıtladı:

### Katman 1: Veri Bütünlüğü, `price_bars` Çiftleme, `auto_adjust` Kırılma Dikişi ve Zaman Dilimi Bozulması
- **Üretim `price_bars` kirliliği:** Canlı PostgreSQL tablosunda (`45,102` satır, `23` sembol, `2021-11-22..2026-11-20`) her aktif sembol için aynı takvim gününe ait **`535–954` çift satır** (`00:00 UTC` taze unadjusted kayıtlar ile eski `04:00/05:00/16:00 UTC` `auto_adjust=True` kayıtları) birikmişti. Toplam **`16,459` duplicate grup**, **`73` hafta sonu hayalet barı** (`14–15`/vadeli sembol) ve `SCCO` üzerinde `1` adet `IEEE NaN` close (`2026-08-11 04:00 UTC`) bulundu (`16,532` silinecek bozuk satır).
- **`_rows_to_price_df` ve `auto_adjust` kırılma dikişi ([`backend/app/features.py`](../../backend/app/features.py)):** Fonksiyon satırları `_raw_ts`'e göre sıralayıp `last` değerini tuttuğu için, `2024-01-16/22` tarihinden itibaren eski `04:00/05:00 UTC` (`auto_adjust=True`) barlar taze `00:00 UTC` (`auto_adjust=False`) barların üzerine yazıldı. Üstelik `generate_symbol_features` `adj_close` yerine ham `close` kullandı (`adj_close != close` durumu hisse/ETF'lerde `1,146–1,227 / 1,254` barda mevcut) ve `_sanitize_close_spikes` yalnızca hedef sembole uygulandı. Sonuç olarak eğitim verisine sahte 1 günlük çöküşler enjekte edildi: `VALE -24.11%` (`2024-01-22`), `RIO -17.83%`, `BHP -11.39%` (MRMR top-12 adayı), `CEW -7.68%`, `PICK -7.20%`, `SCCO -6.75%`, `GXC -6.10%` (MRMR top-12 adayı) ve `2021–2023` arasında `30+` ex-dividend düşüşü.
- **Veri çekme ve backfill uyumsuzluğu ([`backend/app/data_manager.py`](../../backend/app/data_manager.py)):** Toplu `yf.download(..., auto_adjust=False)` ile tekli retry `yf.Ticker.history()` (`auto_adjust=True`) uyumsuzdu; ayrıca `.env` içindeki `LOOKBACK_DAYS=365`, TFT'nin `1825d` (5 yıllık) penceresiyle çelişiyordu.
- **`dedup_price_bars.py` yıkıcı saat dilimi ve unique-constraint hatası ([`backend/scripts/dedup_price_bars.py`](../../backend/scripts/dedup_price_bars.py)):** `_utc_midnight()`, `2899.HK`'nin `16:00 UTC` (= ertesi gün `00:00 HKT`) barlarını bir önceki UTC gününe indirgeyerek `--apply` çalıştırılsaydı **140 geçerli Hong Kong Pazartesi barını Pazar hayalet barı sanıp silecek** ve 560 Salı–Cuma barını yanlış güne taşıyacaktı. Ayrıca `max(r.date)` ile `04:00/05:00 UTC`'yi tercih ediyor ve silme flush'ından önce `canonical.date` güncelleyerek Postgres `uq_price_symbol_date` `UniqueViolation` riski doğuruyordu.

### Katman 2: Özellik Mühendisliği, MRMR Sızıntısı, Sahte Fiziksel Proxy'ler ve Bozuk Core Prior'lar
- **Son 5 OOS gününün kaybı ve encoder körlüğü ([`backend/deep_learning/data/feature_store.py`](../../backend/deep_learning/data/feature_store.py)):** `drop_missing_target=True` iken `target_5d_log_return.notna()` filtresi uygulanması, `TimeSeriesDataSet`'in zaten 5 adımlı decoder ile temiz çözebildiği son 5 işlem gününü (`2026-11-16..20`) kesiyordu. İlk 59 ısınma satırı sıfır ile doldurularak `StandardScaler`'ı bozuyor ve `HG=F_ret1` encoder'a verilmiyordu (`"target" in tft.encoder_variables == False`).
- **Hyperopt CV sızıntısı, bozuk grup sınırı ve OOS IC terslenmesi ([`backend/deep_learning/data/feature_selection.py`](../../backend/deep_learning/data/feature_selection.py), [`backend/deep_learning/training/hyperopt.py`](../../backend/deep_learning/training/hyperopt.py)):** MRMR `master.iloc[:1030]` üzerinde tek sefer çalıştırıldığı için Fold 1–2 validasyonunun **%100'ü** ve Fold 3 validasyonunun **~%40'ı** özellik seçimine sızıyordu (`selection_df` son 5 satırı `VAL` hedefine taşmaktaydı). `group_prefixes` `HG=F`'i içermediği için hedef varlık `max_per_group=2` sınırını aşıyor, `forbidden` kolonları `sel_df`'de kalıyor ve sağlıklı 341 özellikte `Corr(sp_tr, sp_te) = -0.1066` ölçülüyordu (`CL=F_risk_adj_ret10` `-0.166 → +0.265` gibi 4/12 özellik yön değiştiriyordu).
- **Ölü / sahte fiziksel ve eğri özellikleri ([`backend/deep_learning/data/lme_warehouse.py`](../../backend/deep_learning/data/lme_warehouse.py), [`backend/deep_learning/data/futures_curve.py`](../../backend/deep_learning/data/futures_curve.py), [`backend/deep_learning/data/regime_features.py`](../../backend/deep_learning/data/regime_features.py)):** `lme_warehouse_data` tablosunda `0` satır vardı ve `feature_store.py` tabloyu sorgulamıyordu. Vadeli eğri modülü ileri vade kontratı çekmek yerine `HG=F` MA farklarını `futures_curve_slope` (`r=-0.923`) ve `backwardation_flag` (`MA5 < MA20`) olarak etiketliyordu; `proxy_vol_spike` ise `HG=F` hacmindeki `90×` kontrat rollover sıçramalarını (`med=660`, `max=60,833`) ölçüyordu.
- **Zorunlu Core-5 içindeki iki matematiksel olarak bozuk sinyal ([`backend/deep_learning/calibration/hybrid.py`](../../backend/deep_learning/calibration/hybrid.py)):**
  - `macro_fair_value_zscore`: Standardize edilmemiş 20g log getirilerine `ridge_alpha=1.0` uygulanınca varyansı ~70× küçük olan `DX-Y.NYB` ağırlığı **%99.0**, `GXC` ağırlığı **%97.8** ezildi; sinyal fiilen `HG=F_ret20_zscore60` ile `r = +0.9136` korelasyonlu kendi getirisine dönüştü (`sp_tr=+0.012`, `sp_te=-0.333`).
  - `HG=F_theta_5d_prior`: Bar `0`'dan başlayan expanding-origin OLS eğimi kullandığı için tanımı 2022'de 60g eğiminden 2026'da 1,250g eğimine kaydı; `TEST` setinde **`te_zshift = +0.84`** OOD kayması ve düşüş piyasasında %73.6–%94.2 tek yönlü pozitif sapma üretti.
- **Ek özellik ve veri kümesi hijyen hataları:** `close_sma_20_ratio` ve `price_sma_ratio` birebir kopyaydı (`r = 1.0000`); `compute_rsi` kesintisiz yükseliş serilerini `100` yerine `50`'ye atıyordu; yabancı borsa tatillerinde `ffill` sahte `0.0%` getiriler (`CEW` 72 gün) üreterek `vol_20` ve `risk_adj_ret*` değerlerini bastırıyordu; `future_frame.py` ABD/COMEX tatillerini yok sayıyordu; `build_datasets` içinde `purge_gap=0` idi ve tüm ensemble seed'leri tek bir stateful `train_generator` paylaşıyordu.

### Katman 3: Model, `WeeklyASROPFLoss`, Başlık Parametrizasyonu ve Eğitim Motoru Çöküşleri
- **`lambda_drift_prior = 4.0` (L1+L2) matematiksel sıfır-çöküşü ([`backend/deep_learning/models/tft_copper.py`](../../backend/deep_learning/models/tft_copper.py)):** `|mean(pred_w)/scale_w| + (mean(pred_w)/scale_w)^2` formülü (`4.0` ağırlığı ve `[2.5, 4.0, 6.0]` Optuna kilidi ile), tam sıfır ortalamalı mükemmel bir tahmincide bile `B=32` minibatch örnekleme varyansı nedeniyle $\approx 0.564\,\sigma_{\hat{m}}/s_w$ sabit L1 vergisi yarattı. Kontrollü PyTorch mikro-deneyi (`B=32`, `scale_w=0.032`, `corr=+0.1819` sentetik sinyal): `alpha=0.00` (tüm sıfır) **`total=0.33341`** ile kesin global minimum olurken, Bayes-optimal `alpha=0.20` **`0.44532` (%33.6 daha kötü)** çıktı. `drift_prior` gradyan normu **`49.886`**, `weekly_q` `2.859` (**17.4× küçük**), `directional` `1.159` (**43× küçük**) ve 6 dış quantile kanalının toplamı `0.00812` (**6,143× küçük**) düzeyinde kaldı; `gradient_clip_val=0.5` altında gerçek öğrenme sinyali ~100× ezildi.
- **Çoklu-seed validasyon döngüsü çöküşü (`trainer.py:1156`):** `rec_idx >= 1` için `_predict_quantiles_to_np(..., return_y=False)` tek dizi döndürdüğünde tuple açma `ValueError` fırlattı ve `trainer.py:1346`'daki `except Exception` yakalaması her çoklu-seed koşuda seed filtrelemeyi, checkpoint slot eşleştirmeyi, Theta harmanını ve Split CQR'ı iptal etti.
- **Çıkış başlığı yayılım donması, ham logit sızıntısı ve Jensen büzülmesi:** `_differential_configure_optimizers` 7 bias kanalının tümüne `head_bias_lr = 2.5e-5` uygulayınca ve `enforce_monotonic_quantiles` `0.03 * softplus(raw - 3.0)` Jacobian'ıyla (`~0.0075`) birleşince dış quantile spread bias'ları fiilen dondu (<%0.026 değişim). Ayrıca `WeeklyASROPFLoss`, `to_quantiles()`'ı override etmediği için ham kısıtsız delta logitleri yayıyordu; `trainer.py:1244, 1371` çoklu-seed ortalamasını konveks `softplus` öncesinde ham logitlerde alarak Jensen eşitsizliği nedeniyle PI80 genişliğini **%13.7–%44.4** daraltıyordu.
- **Validasyon yön terslenmesi, decoder nullspace ve ölü iskeleler:** Unshuffled `val_dl` (`shuffle=False`) üzerinde batch içi `actual_weekly` ortalaması çıkarıldığı için ralli batch'lerinde ortalamanın altındaki pozitif getiriler negatif yön hedefine dönüşüyordu. `h=1..4` adımları denetimsiz 3D nullspace (`daily_forecasts[1..4]` belirsizliği) taşıyordu. `SWACallback` `ModelCheckpoint` sonrasında ağırlık ortaladığı için ölü koddu; top-2 checkpoint soup'ta epoch yakınlık sınırı yoktu; `WeeklyLossComponentLogger` `sanity_checking` korumasına sahip değildi; `hyperopt.py` ise son fold'da (`fold_idx=2`) trial buduyor, `effective_da = max(avg_da, avg_weekly_da)` ile T+1 DA sızdırıyor ve `TFT_FEATURE_SNAPSHOT_PATH` yazımını kazanan `mrmr_top_k` seçilmeden önce yapıyordu.

### Katman 4: Kalibrasyon, Metrik, Quality Gate ve 4-Yol Execution Parite Kırılmaları
- **`conformal_calibration.json` içinden `base_vol_power` kaybı ([`backend/deep_learning/calibration/conformal.py`](../../backend/deep_learning/calibration/conformal.py), `trainer.py:1793-1804`):** `fit_weekly_cqr` içinde `base_vol_power = 1.0` ile ölçeklenip test değerlendirmesinde geçilen CQR parametreleri JSON'a yazılırken `base_vol_power` anahtarı unutuluyordu. `apply_weekly_cqr` eksik anahtarı `0.0` varsaydı; canlı/backtest yolları yüksek oynaklıkta (`2×` medyan vol) **%46.5** daha dar (`0.535×`), düşük oynaklıkta (`0.5×`) **%93** daha geniş (`1.930×`) aralıklar üretti. Ayrıca `q25/q75` iç çeyrekleri ölçeklenmiyor ve `_split_conformal_quantile` `MA(4)` örtüşme düzeltmesi (`n_eff`) yapmıyordu.
- **Çifte `monotonic_quantiles_np` totolojisi ve `AssertionError` 500'ü ([`backend/deep_learning/training/metrics.py`](../../backend/deep_learning/training/metrics.py), `quality_gate.py`):** `compute_all_metrics` zaten sıralanmış quantile dizisine ikinci kez `monotonic_quantiles_np` uygulayarak %100 ters çevrilmiş (`crossing=1.0`) girdilerde bile `quantile_crossing_rate=0.0` ve `median_sort_gap_max=0.0` döndürüyordu. `quality_gate.py` post-CQR `weekly_ordered_quantile_crossing_rate`'i okumuyor, eksik `weekly_mae_vs_naive_zero` değerini atlıyor ve crossing durumunda `(False, reasons)` yerine `AssertionError` fırlatarak `/api/models/tft/summary` endpoint'ini HTTP 500 ile düşürüyordu.
- **`sortino_ratio` `1e-9` ve RMS formül hatası:** `1e-9 < 1e-9` kontrolü `False` olduğundan `0–1` negatif getiride **`~4.74e8`** değerine patlıyor; negatif getirilerin hedef-altı RMS sapması yerine kendi ortalaması etrafındaki `std()` değerini alarak `63×` bozulmuş sonuç (`-853.1` vs `-13.49`) üretiyordu.
- **4-Yol Parite Kırılmaları (Train / Gate / Backtest / Live API & Worker):**
  - `format_prediction`: `±0.12` medyan kırpmasını `enforce_monotonic_quantiles` *sonrasında* uygulayarak gizli crossing (`q10=0.284 > q50=0.128`) yaratıyor ve skaler `conformal_adjustment` dalında `daily_forecasts[-1]` fiyat bantlarını güncellemiyordu.
  - `TFTBacktestRunner` ([`backend/backtest/runner.py`](../../backend/backtest/runner.py)): `weekly_median_cap` uygulamasını atlıyor, eksik özellikleri sessizce `0.0` ile dolduruyor, `needs_news` bilgisini taşımıyor ve eksik seed checkpoint'lerini `DEBUG` logu ile sessizce düşürüyordu.
  - [`backend/deep_learning/models/hub.py`](../../backend/deep_learning/models/hub.py): Ensemble küçüldüğünde eski `seed1`/`seed2` dosyalarını HF Hub'dan (`CommitOperationDelete` yok) veya yerel staging'den silmiyordu.
  - [`backend/app/main.py`](../../backend/app/main.py) & [`backend/worker/tasks.py`](../../backend/worker/tasks.py): `/api/analysis/consensus` T+1 XGBoost ile T+1 TFT'yi `xgb_bias_correction=0.001` ile karıştırıp `prediction=None` durumunda çöküyordu; Stage 5.5 `artifact_set_valid` değerini yok sayıp hatalarda pipeline'ı `"ok"` raporluyordu; `.github/workflows/tft-training.yml` adım atlamalarında bile HF Hub upload başarısı basıyordu.

### Katman 5: Küçük Örneklem (`Small-N`) Rejim Kırılganlığı ve Dashboard Horizon / Eşik Uyumsuzluğu
- **Parametre / efektif örneklem dengesizliği:** `1,006` örtüşen eğitim günü (~`201` bağımsız 5D hafta) üzerinde `12` bilinmeyen özellikle bile TFT `14,451` eğitilebilir parametreye (= **bağımsız hafta başına 71.8 parametre**) sahipti. Tekil `val_ratio=0.08` (~`19.2` hafta) ile `test_ratio=0.10` (~`24.2` hafta) bölünmesinde DA binomial standart hatası **`±10.2pp`**, yıllıklaştırılmış Sharpe standart hatası **`~1.47`** mertebesindedir. Üstelik aynı ~19 haftalık `val_dl` penceresi 6 ardışık seçim/kalibrasyon aşamasında tekrar kullanılıyordu.
- **Dashboard horizon karmaşası ve uzun vade boşluğu:** `OverviewPage.tsx` `T+1` günlük XGBoost sürücülerini (`include_raw_ma=True` ile eğitilen "Market drivers") 5D birincil TFT tahmininin yanında net horizon ayrımı olmadan sergiliyordu. `ModelReliability.tsx` ise `quality_gate.py` ile çelişen eski sabit eşikleri (`DA ≥ 53%`, `MR [0.65, 1.35]`, `PI80 [74%, 86%]`, `Tail ≥ 45%`, T+1 `dailySharpe`) kullanıyordu. Ayrıca 180 günlük fiyat grafiğinin aksine `20D (1M)` / `60D (1Q)` için uygun düşük-varyanslı bir yapısal tahmin motoru mevcut değildi.

---

## 3. Uygulanan Çözümler ve Değiştirilen Dosyalar (Faz 2)

> **Sözleşme Güvencesi:** Tüm düzeltmeler `weekly_log_v1` 5-adımlı günlük log-getiri ve RSS quantile mimarisini koruyarak yapılmıştır. [`backend/app/quality_gate.py`](../../backend/app/quality_gate.py) içindeki **hiçbir promotion eşiği düşürülmemiş veya bypass edilmemiştir**; aksine eksik/NaN `weekly_mae_vs_naive_zero` koruması ve post-CQR crossing kontrolü ile kapı sertleştirilmiştir.

### 3.1 Veri ve Özellik Katmanı
| Dosya | Temel Semboller / Değişiklikler ve Gerekçe |
| --- | --- |
| [`backend/app/price_utils.py`](../../backend/app/price_utils.py) | `exchange_timezone`, `exchange_local_date` (`.HK`, `.L`, `.TO`, `America/New_York`), `is_canonical_utc_midnight`, `is_weekend_bar`, ortak `sanitize_close_spikes` ve hafta sonu/`NaN`/pozitif-olmayan kapanışları eleyen `latest_finite_price_bar` eklendi. |
| [`backend/app/features.py`](../../backend/app/features.py) | `_rows_to_price_df` borsa yerel tarihinde tekilleştirip taze `00:00 UTC` + geçerli `adj_close` barlarını `04:00/05:00/16:00 UTC` barlarına tercih etti; `generate_symbol_features` pozitif `adj_close` kullandı ve `include_raw_ma=False` iken `price_sma_ratio` kopyasını bastırdı; `generate_aligned_symbol_features` göstergeleri aktif işlem günlerinde hesaplayıp tatil `ffill` sıfır-getiri bozulmasını önledi; `compute_rsi` uç durumları (`100.0` / `0.0` / `50.0`) düzeltildi. |
| [`backend/app/data_manager.py`](../../backend/app/data_manager.py) | Tekli retry `yf.Ticker.history(..., auto_adjust=False)` ile `yf.download` uyumu sağlandı; borsa yerel hafta sonu filtresi ve `tft_enabled=True` iken 5 yıllık varsayılan backfill eklendi. |
| [`backend/scripts/dedup_price_bars.py`](../../backend/scripts/dedup_price_bars.py) | `exchange_local_date` ile `2899.HK` `16:00 UTC` barlarının HK Pazartesi olarak korunması sağlandı (`0` yanlış silme); kanonik `00:00 UTC` seçimi ve `UniqueViolation` önlemek için silmelerin `session.flush()` ile canonical tarih güncellemesinden önce uygulanması eklendi. |
| [`backend/deep_learning/data/feature_store.py`](../../backend/deep_learning/data/feature_store.py) | `build_tft_dataframe` yalnızca `target.notna()` filtreleyerek son 5 OOS gününü geri kazandı; baştaki 59 warmup satırı kesildi (`row0 zero_frac=0.000`); `HG=F_ret1` eklendi; `lme_warehouse_data` boşken sahte `PROXY_LME_FEATURES` / `SYNTHETIC_PROXY_CURVE_FEATURES` / `regime_inventory_tightness` MRMR havuzundan bastırıldı; `known` girdiler `["day_of_week"]` ile sınırlandı; `target_prefix` (`HG=F`) `group_prefixes`'e eklendi; `apply_mrmr=False` veya `use_snapshot=False` iken snapshot dosyasına dokunulmadı. |
| [`backend/deep_learning/data/feature_selection.py`](../../backend/deep_learning/data/feature_selection.py) | `HG=F` ve fiziksel/eğri grup etiketleme düzeltildi (`max_per_group=2`); `forbidden_features` `filtered.columns` dışına çıkarıldı; `_is_stationary_forced_feature` koruması ve zero-leakage CV için `build_fold_scoped_features` (`selection_end = train_end_pos - horizon`) eklendi. |
| [`backend/deep_learning/data/dataset.py`](../../backend/deep_learning/data/dataset.py) | `compute_dataset_split_bounds` ve `build_datasets(..., purge_gap=5)` ile `[train \| gap=5 \| val \| gap=5 \| test]` ayrımı sağlandı; `make_dataloader_generator(seed)` / `create_train_dataloader(..., seed=...)` ile seed başına bağımsız RNG tanımlandı; `build_cv_folds` içine per-fold `mrmr_top_k` bağlandı. |
| [`backend/deep_learning/data/lme_warehouse.py`](../../backend/deep_learning/data/lme_warehouse.py), [`futures_curve.py`](../../backend/deep_learning/data/futures_curve.py), [`regime_features.py`](../../backend/deep_learning/data/regime_features.py), [`future_frame.py`](../../backend/deep_learning/data/future_frame.py) | `load_lme_from_db` bağlandı ve durağan olmayan `NON_STATIONARY_LME_FEATURES` ayrıldı; `future_frame.py` `USMarketHolidayCalendar` (`CustomBusinessDay`) ile ABD/COMEX tatillerine duyarlı hale getirildi. |

### 3.2 Model, Kayıp Fonksiyonu, Kalibrasyon ve Hyperopt Katmanı
| Dosya | Temel Semboller / Değişiklikler ve Gerekçe |
| --- | --- |
| [`backend/deep_learning/config.py`](../../backend/deep_learning/config.py) | `FeatureStoreConfig.mrmr_top_k = 5` (kompakt Core-5 + `HG=F_ret1`, ~`11.4K` parametre — temiz DB ablasyonu gürültülü `mrmr >= 8` çapraz varlıkların test holdout'unu bozduğunu kanıtladı), `TFTModelConfig.weight_decay = 2e-2`, `WeeklyLossConfig.lambda_drift_prior = 1.00`, `lambda_subhorizon_quantile = 0.15` varsayılanları ayarlandı. |
| [`backend/deep_learning/models/tft_copper.py`](../../backend/deep_learning/models/tft_copper.py) | **1)** `WeeklyASROPFLoss`: L1+L2 kink yerine pürüzsüz L2-only `(mean / scale_w)^2` (`lambda_drift_prior=1.00`), kümülatif RSS alt-ufuk quantile denetimi (`lambda_subhorizon_quantile=0.15` ile decoder nullspace giderimi), `directional_loss` içinde batch demeaning'in `if self.training` ile sınırlandırılması, `emits_ordered_quantiles = True` ve `enforce_monotonic_quantiles` kullanan `to_quantiles()` override'ı. **2)** `_DecoupledQuantileHead`: yalnızca `median_bias` kanalı `0.05x` LR ile sınırlanırken `spread_bias` ve çıkış ağırlıkları `output_head_lr_mult * lr` ile serbest bırakıldı (`weight`/`bias` native `state_dict` uyumlu). **3)** `ensure_ordered_quantiles_from_model_output` ve `format_prediction`: `±0.12` güvenlik kırpması monotonlaştırma öncesi (veya sıralı çıktıda tüm kanallara kohezif shift olarak) taşındı; skaler konformal `daily_forecasts[-1]` güncellemesi eklendi. |
| [`backend/deep_learning/training/trainer.py`](../../backend/deep_learning/training/trainer.py), [`callbacks.py`](../../backend/deep_learning/training/callbacks.py) | `rec_idx >= 1` tuple unpack çöküşü düzeltildi ve kalibrasyonu gizleyen geniş `except Exception` kaldırıldı; `_predict_quantiles_to_np(..., ordered=True)` ile çoklu-seed ensemble ortalaması kesinlikle sıralı quantile uzayında (post-softplus, Jensen büzülmesi olmadan) alındı; ölü `SWACallback` kaldırıldı; `_build_uniform_checkpoint_soup` içine `max_epoch_gap=5` sınırı eklendi; seed başına izole `create_train_dataloader(..., seed=...)` ve `compute_dataset_split_bounds` bağlandı; `WeeklyLossComponentLogger` `trainer.sanity_checking` koruması aldı. |
| [`backend/deep_learning/calibration/conformal.py`](../../backend/deep_learning/calibration/conformal.py) | `conformal_calibration.json` içine `base_vol_power` (`1.0`), `n_eff` ve `overlap_horizon` (`5`) kalıcı olarak yazıldı (`_write_conformal_calibration_artifact`); `_split_conformal_quantile` `MA(4)` `n_eff = max(n / 5.0, 1.0)` + üst quantile sınırı kullandı; `apply_weekly_cqr` içinde `q25` ve `q75` `[q10, q90]` içinde orantılı ölçeklendi. |
| [`backend/deep_learning/calibration/hybrid.py`](../../backend/deep_learning/calibration/hybrid.py) | `compute_causal_macro_fair_value_zscore` pencere içi predictor standardizasyonu kazandı (`DX-Y.NYB` ve `GXC` %99 ezilmeden gerçek makro artığı ölçer); `compute_causal_theta_5d_series` sabit kayan pencere (`lookback_window=126`) kullandı; `fit_theta_hybrid_blend` anti-skew korumaları (`max_prior_pos_skew=0.20`, `max_prior_excess_skew=0.15`, dengeli validasyonda `[0.22, 0.78]` bandı) ile yalnızca validasyon bileşik skorunu MAE bozmadan iyileştirdiğinde devreye girdi. |
| [`backend/deep_learning/training/hyperopt.py`](../../backend/deep_learning/training/hyperopt.py) | Arama uzayı `mrmr_top_k ∈ [5, 6, 8, 12]`, `lambda_drift_prior ∈ [0.15, 0.35, 0.75, 1.00]`, `weight_decay ∈ [1e-2, 2e-2, 3e-2]` olarak güncellendi; `build_cv_folds` ile per-fold sıfır-sızıntılı MRMR kullanıldı; `effective_da` yalnızca birincil `avg_weekly_da` okudu; son fold'da pruning kapatıldı; `#63` düzeltmesiyle `TFT_FEATURE_SNAPSHOT_PATH` çalışma öncesi yerine `study.optimize` sonrasında kazanan `best_params["mrmr_top_k"]` ile yazıldı. |
| [`backend/deep_learning/models/hub.py`](../../backend/deep_learning/models/hub.py) | `upload_tft_artifacts` içinde eksilen opsiyonel dosyalar (`seed1`, `seed2`, `pca_finbert.joblib`) için `CommitOperationDelete` eklendi; `sync_tft_artifacts_if_stale` eski yerel seed checkpoint'lerini temizledi. |

### 3.3 Metrik, Quality Gate, 4-Yol Parite, API, Worker ve CI Katmanı
| Dosya | Temel Semboller / Değişiklikler ve Gerekçe |
| --- | --- |
| [`backend/deep_learning/training/metrics.py`](../../backend/deep_learning/training/metrics.py), [`validation/backtest.py`](../../backend/deep_learning/validation/backtest.py) | `compute_all_metrics` içindeki çifte `monotonic_quantiles_np` kaldırılarak gerçek crossing/sort-gap ölçümü sağlandı; `sortino_ratio` hedef-altı RMS formülü ve `<= 1e-9` koruması ile düzeltildi; `sharpe_ratio_hac(..., annualisation=52.0, max_lag=4)` sözleşmesi korundu; `weekly_non_overlapping_{sharpe,da,n}` ve `already_ordered: bool` parametresi eklendi; `run_backtest` çok günlük koşularda `p75` `tail_threshold` ve `max_lag=4` kullandı. |
| [`backend/app/quality_gate.py`](../../backend/app/quality_gate.py), [`backend/scripts/tft_quality_gate.py`](../../backend/scripts/tft_quality_gate.py) | Tüm eşikler korunurken eksik/NaN `weekly_mae_vs_naive_zero` hard fail'e çevrildi; tüm `AssertionError` dalları `(False, reasons)` dönüşüne çevrildi (HTTP 500 giderildi); post-CQR `weekly_ordered_quantile_crossing_rate > 0.001` (`WeeklyPostCQRQuantileCrossing`) eklendi; CLI `weekly_sharpe_ratio_hac == 0.0` `or` hatası (`is None`) düzeltildi. |
| [`backend/backtest/runner.py`](../../backend/backtest/runner.py), [`deep_learning/inference/predictor.py`](../../backend/deep_learning/inference/predictor.py), [`app/tft_explain.py`](../../backend/app/tft_explain.py) | `TFTBacktestRunner` ve `TFTPredictor` sıralı quantile uzayında ensemble ortalaması, eksiksiz kalibrasyon zinciri (`hybrid_blend`, `weekly_median_cap`, `apply_weekly_cqr` dahil), eksik/bozuk ikincil seed veya eksik özellikte fail-closed davranış ve atomik staging sync ile eşitlendi; `TFTPredictor` `get_model_metadata` `.order_by(trained_at.desc())` ve hafta sonu/sonlu fiyat koruması (`_latest_weekday_price_bar`) ekledi; `tft_explain.py` `aggregate_interpretations` ile tüm aktif ensemble üyelerini kapsadı ve `focus_basis = "pre_calibration_raw_decoder_step"` işaretledi. |
| [`backend/app/main.py`](../../backend/app/main.py), [`models.py`](../../backend/app/models.py), [`db.py`](../../backend/app/db.py), [`schemas.py`](../../backend/app/schemas.py), [`worker/tasks.py`](../../backend/worker/tasks.py) | `long_term_router` ve `LongTermOutlookSnapshot` bağlandı; `_latest_price_bar_date` hafta sonu ve geçersiz close'ları atladı; `/api/analysis/tft/{symbol}` tazelik/geçerlilik denetimi kazandı; `/api/analysis/consensus` 5D birincil sinyale, `xgb_bias_correction=0.0`'a ve `prediction: None` korumasına geçirildi; `/api/models/tft/summary` `weekly_sharpe_ratio_hac`, `weekly_mae_vs_naive_zero`, `weekly_pred_positive_rate`, `weekly_actual_positive_rate`, `weekly_pinball_loss` döndürdü; `/api/models/tft/backtest/latest` yalnızca TFT rapor türlerine (`tft_vs_xgb`, `tft_walk_forward`) kilitlendi; Stage 5.5 `artifact_set_valid` zorunlu tuttu, Stage 5.7 fail-open `run_long_term_outlook_stage` çalıştırdı. |
| [`.github/workflows/tft-training.yml`](../../.github/workflows/tft-training.yml) | Hyperopt timeout yorumu (`355 min`), `if-no-files-found: error` artifact kilidi, `TOKENIZERS_PARALLELISM: "false"`, doğru rollback CLI (`-f optuna_results_run_id=...`) ve yalnızca `steps.quality_gate.outcome == 'success'` için HF Hub başarı mesajı düzeltildi. |

### 3.4 Uzun Vadeli Görünüm Backend Katmanı
- [`backend/long_term/`](../../backend/long_term/) (`__init__.py`, `schemas.py`, `features.py`, `model.py`, `backtest.py`, `service.py`), [`backend/app/long_term_models.py`](../../backend/app/long_term_models.py), [`backend/app/long_term_routes.py`](../../backend/app/long_term_routes.py), [`backend/migrations/005_long_term_outlook_snapshots.sql`](../../backend/migrations/005_long_term_outlook_snapshots.sql) — ayrıntılı mimari Bölüm 4'tedir.

### 3.5 Frontend UI ve Horizon / Gate Hizalama Katmanı
| Dosya | Temel Semboller / Değişiklikler ve Gerekçe |
| --- | --- |
| [`frontend/src/types.ts`](../../frontend/src/types.ts), [`api.ts`](../../frontend/src/api.ts), [`hooks/useQueries.ts`](../../frontend/src/hooks/useQueries.ts) | `LongTermOutlookResponse` şeması 1:1 tanımlandı, `TFTMetrics` genişletildi, `fetchLongTermOutlook` (`GET /api/outlook/long-term/{symbol}`) ve `useLongTermOutlook` eklendi. |
| [`frontend/src/features/forecast/ModelReliability.tsx`](../../frontend/src/features/forecast/ModelReliability.tsx), [`frontend/src/pages/ModelsPage.tsx`](../../frontend/src/pages/ModelsPage.tsx) | Eski sabit eşikler yerine arka uç [`backend/app/quality_gate.py`](../../backend/app/quality_gate.py) ile birebir eşleşen `computeQualityGateThresholds`, `hasWeeklySignCollapse` ve `evaluateClientQualityGate` implemente edildi (`weekly_sharpe_ratio_hac`, `weekly_mae_vs_naive_zero`, `MA(4)` `n_eff` binomial PI80 bantları ve arka uç `quality_gate.passed` durumu). |
| [`frontend/src/features/forecast/LongTermOutlookPanel.tsx`](../../frontend/src/features/forecast/LongTermOutlookPanel.tsx), [`frontend/src/pages/OverviewPage.tsx`](../../frontend/src/pages/OverviewPage.tsx) | `LongTermOutlookPanel` + `ConnectedLongTermOutlookPanel` lazy bileşeni olarak `#long-term-outlook` (`04 // 1M & 1Q STRATEGIC OUTLOOK`) ve `1M / 1Q Outlook` lens sekmesi eklendi; `OverviewPage.tsx` `5D` birincil TFT tahminini (`#weekly-outlook`) ve 5D XAI odağını (`#neural-analysis`), `T+1` XGBoost teşhis panelinden (`#market-drivers`) net ayrıştırdı. |

---

## 4. Yeni Uzun Vadeli Görünüm Mimarisi (`20D / 1M` & `60D / 1Q`)

`5D` taktiksel sinir ağı tahmininin uzun ufuklara mekanik olarak uzatılması istatistiksel olarak geçersiz olduğu için, 5D TFT checkpoint'lerinden tamamen izole çalışan düşük-varyanslı bir yapısal motor ([`backend/long_term/`](../../backend/long_term/)) inşa edildi:

1. **Veri Hijyeni, Haftalık Örnekleme ve Nedensel FRED Gecikmeleri ([`backend/long_term/features.py`](../../backend/long_term/features.py)):**
   - `clean_price_rows_to_series`: `00:00 UTC` barlarını eski `04:00/05:00 UTC` barlarına tercih eder, hafta sonu (`dayofweek >= 5`) barlarını eler, `adj_close` -> `close` yedeklemesi yapar ve `W-FRI` haftalık serilere örnekler (`ffill(limit=1)`).
   - 15 yıllık 11 makro/emtia/madenci enstrümanı (`LONG_TERM_SYMBOLS`: `HG=F`, `DX-Y.NYB`, `CL=F`, `GXC`, `COPX`, `FCX`, `SCCO`, `RIO`, `BHP`, `FXA`, `XME`) ile fail-open FRED CSV serilerini birleştirir.
   - Yayın gecikmesi (`FRED_SERIES_LAG_DAYS`) sıfır-sızıntılıdır: `DGS10` (10Y getiri) **`+2d`**, `T10YIE` (10Y breakeven enflasyon) **`+2d`**, `DTWEXBGS` (geniş USD endeksi) **`+7d`** kaydırılarak haftalık gözleme bağlanır.
2. **Düşük Varyanslı Yapısal Ensemble ([`backend/long_term/model.py`](../../backend/long_term/model.py)):**
   - İki ufuk tanımı (`HORIZONS`): **`20D`** (`1 Month (20D)`, `4` hafta, yıllıklaştırma `√12`) ve **`60D`** (`1 Quarter (60D)`, `12` hafta, yıllıklaştırma `√4`).
   - 10 durağan, ölçek-bağımsız faktör (`MODEL_FEATURES`): `hg_sma10w_ratio`, `hg_sma40w_ratio`, `hg_dd52`, `dxy_ret26`, `real_yield_chg26`, `breakeven_10y_z`, `fxa_ret26`, `cl_ret12`, `miners_vs_hg_12w`, `copx_vs_hg_12w`.
   - **Standardize `Ridge` Makro-Trend:** Volatilite-normalize ileri log getiriler $z_H = r_{t \to t+H} / (\hat{\sigma}_{w,t}\sqrt{H})$ üzerinde `ridge_window=260` hafta (~5 yıl) kayan pencere ve ufuk örtüşmesine göre ölçeklenen L2 cezası $\alpha_{\text{eff}} = 3000 \cdot \max(1, H_w / 4)^{0.8}$ ile eğitilir (`ridge_weight = 0.55`, `60D` için `0.50`).
   - **Damped-Trend Theta (`Theta-0 / Theta-2`):** `theta_lookback = 64` hafta ve `theta_phi = 0.82` sönümleme katsayısı ile yapısal trend bileşenini verir.
   - **L2 Multinomial Rejim Sınıflandırıcısı (`LogisticRegression`, `C=0.002`):** Gürültü dışı ekonomik hareket eşiği (`|r| > 0.5 * sigma_H`) üzerinde eğitilen sınıf olasılıkları Gaussian-implied dağılım ile harmanlanır (`logit_weight = 0.25`); `prob_bullish + prob_neutral + prob_bearish == 1.0`, `stance ∈ {BULLISH, NEUTRAL, BEARISH}` ve `conviction` üretilir.
   - **Purged Split Conformal Bantları (`QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)`):** Purged OOF artık mutlak değerlerinden kesintisiz monotonik (`hw80 >= hw50 + 1e-3`, `crossing_rate == 0.0`) fiyat ve getiri aralıkları üretir.
3. **4 Sütunlu Makro Scorecard (`compute_macro_scorecard`):**
   - `trend_structure` (Trend Yapısı), `usd_macro` (USD & Makro), `china_industrial_demand` (Çin & Sanayi Talebi) ve `cross_asset_confirmation` (Çapraz Varlık Teyidi) sütunlarından `composite_score` ve `composite_stance` hesaplanır.
4. **15 Yıllık Purged Walk-Forward Ampirik Doğrulama Sonuçları ([`backend/long_term/backtest.py`](../../backend/long_term/backtest.py)):**
   `HG=F` (`2011-11-25..2026-11-20`, `783` haftalık bar, `as_of=2026-11-20` `$5.477`, FRED aktif) üzerinde `purged_walk_forward_splits` (`tr <= te[0] - H`) ile elde edilen OOS sonuçları:
   - **`20D (1M)` (`H=4w`, `purge_gap=4w`, `ann=12.0`, 5 fold, `n=545` OOS / `n_no=137` örtüşmesiz):** `DA=0.5174` (`DA_no=0.5401`), `MAE/Naive=0.9994`, `MAE/Theta=0.9980`, `HAC Sharpe=+0.0358` (`SR_no=+0.0302`), `PI80=0.8275`, `PI50=0.5450`, `Brier=0.6595` (prior `0.6615`), `RegimeAcc=0.4440`, `Crossing=0.0`.
   - **`60D (1Q)` (`H=12w`, `purge_gap=12w`, `ann=4.0`, 5 fold, `n=537` OOS / `n_no=45` örtüşmesiz):** `DA=0.5363` (`Theta DA=0.5400`), `MAE/Naive=1.0005`, `MAE/Theta=1.0052`, `HAC Sharpe=+0.0584`, `PI80=0.8436`, `PI50=0.5493`, `Brier=0.6649` (prior `0.6739`), `RegimeAcc=0.4320`, `Crossing=0.0`.
   - **Canlı Nokta Snapshot (`as_of=2026-11-20`, `$5.477`):** `20D` hedef `$5.5187` (`+0.76%`, `NEUTRAL`), `60D` hedef `$5.5328` (`+1.02%`, `NEUTRAL`), 4-sütun makro skor kartı `-0.8628 BEARISH` (Kasım 2026 geri çekilmesi + güçlü USD/reel faiz baskısını doğru yansıtarak).
5. **Veritabanı, Snapshot-Only API ve Frontend Paneli:**
   - [`backend/migrations/005_long_term_outlook_snapshots.sql`](../../backend/migrations/005_long_term_outlook_snapshots.sql) ve [`backend/app/long_term_models.py`](../../backend/app/long_term_models.py): `long_term_outlook_snapshots` tablosu (`symbol`, `as_of_date`, `payload_json`, `validation_json`, `model_version`, `generated_at`, `run_id`) ve 3 indeks.
   - [`backend/app/long_term_routes.py`](../../backend/app/long_term_routes.py): `GET /api/outlook/long-term/{symbol}` — tamamen **snapshot-only** çalışır (HTTP işleyicisi içinde asla eğitim veya ağ çağrısı yapmaz), snapshot yoksa temiz `HTTP 200` `status="unavailable"` döner.
   - [`backend/worker/tasks.py`](../../backend/worker/tasks.py) & [`backend/long_term/service.py`](../../backend/long_term/service.py): Stage 5.7 `run_long_term_outlook_stage` (fail-open, DB hatasında rollback ile ana pipeline'ı düşürmez).
   - [`frontend/src/features/forecast/LongTermOutlookPanel.tsx`](../../frontend/src/features/forecast/LongTermOutlookPanel.tsx): `1 Month (20D)` / `1 Quarter (60D)` `SegmentedTabs`, hedef fiyat ve sıfır-baskılanmış yüzdelik değişim, kalibre `Q10–Q90` / `Q25–Q75` bant grafiği, 3-yönlü rejim olasılık barı ve güven skoru (`conviction`), 4-sütun Macro Factor Scorecard, Purged WF doğrulama rozeti ve loading/error/unavailable görünümleri.

---

## 5. Doğrulama Kanıtları (Post-Final-Edit)

Tüm son kod düzenlemelerinin (`#63` `hyperopt.py` post-study `feature_snapshot.pkl` düzeltmesi dahil) ardından bağımsız doğrulama ve test sonuçları aşağıdadır:

### 5.1 Otomatik Test Süitleri ve Bütçe Kontrolleri
- **Backend Tam Çevrimdışı Test Süiti (`backend/`, son düzenleme sonrası):**
  ```powershell
  py -3 -m pytest tests/ -v -m "not online"
  ```
  **Sonuç:** **`918 passed, 12 warnings in 45.61s`** (çıkış kodu `0`, `0` hata, `0` skip; `860` taban test üzerine `+58` yeni regresyon ve sözleşme testi).
- **Hedefli Derin Öğrenme & Hyperopt Testleri (`backend/`, son düzenleme sonrası):**
  - `py -3 -m pytest tests/deep_learning/test_hyperopt.py -v -m "not online"` → **`20 passed in 17.60s`**
  - `py -3 -m pytest tests/deep_learning tests/test_pipeline_overhaul_regressions.py -v -m "not online"` → **`333 passed, 12 warnings in 32.14s`**
  - `py -3 -m pytest tests/test_long_term_outlook.py -v` → **`7 passed in 37.43s`** (`#32` doğrulaması: `37.18s`)
- **Frontend Tam Süit (`frontend/`):**
  - `npm test` → **`28 test files, 269 passed (269)`** (`52.81s`, `long-term-outlook.test.tsx` ve `OverviewPage.test.tsx` dahil)
  - `npm run lint` → **`0 warnings, 0 errors`**
  - `npm run build` → Başarılı (`tsc && vite build && node scripts/prerender.mjs`, `2954` modül; `LongTermOutlookPanel` lazy chunk **`4.37 kB`** gzip)
  - `npm run check:budgets` → **PASSED** (`initialJS = 135,015 / 190,000 B`, `css = 14,355 / 15,000 B`, `dashboardJS = 299,453 / 300,000 B`)

### 5.2 Salt-Okunur Canlı Veritabanı ve Uçtan Uca Eğitim / Backtest Kanıtları (`0` DB / HF Hub Yazımı)
- **`dedup_price_bars` Dry-Run Denetimi (salt-okunur DB):**
  - `dedup_price_bars(session, dry_run=True, drop_weekends=True)` → `total_rows_scanned=45102`, `symbols_scanned=23`, `duplicate_groups=16459`, `weekend_rows_to_delete=73`, `rows_to_delete=16532`, `rows_normalized=21`, `rows_deleted=0`.
- **Feature Store & Leak-Free CV Parite Kontrolü (salt-okunur DB, son düzenleme sonrası):**
  - `build_tft_dataframe(session, cfg, apply_mrmr=True, use_snapshot=False)` → şekil `(1201, 14)`, `unknown=['HG=F_close_sma_20_ratio', 'HG=F_ret1', 'HG=F_ret5', 'HG=F_signed_vol_ratio', 'HG=F_theta_5d_prior', 'macro_fair_value_zscore']`, `known=['day_of_week']`, `last_close=5.477`, `weekends=0`, `row0 zero_frac=0.000`.
  - `compute_dataset_split_bounds(df, cfg, purge_gap=5)` → `train_size=975 (idx<=974)`, `purge_gap=5`, `val_size=96 (idx 980..1075)`, `test_size=120 (idx 1081..1200)`.
  - `build_cv_folds(..., n_folds=3, purge_gap=5, mrmr_top_k=5)` → fold başına sızıntısız MRMR boyutları `[(626, 139), (769, 139), (912, 136)]`.
- **FastAPI `TestClient` Smoke Kontrolü (son düzenleme sonrası):**
  - `GET /api/outlook/long-term/HG=F` → `200 OK`
  - `GET /api/models/tft/summary?symbol=HG=F` → `200 OK`
  - `GET /api/analysis/consensus?symbol=HG=F` → `200 OK`
- **Canlı DB Üzerinde Uçtan Uca TFT-ASRO Eğitim, Kapı & Backtest Koşusu (`TemporaryDirectory`, `#58`):**
  *(Not: Model, kayıp fonksiyonu, kalibrasyon ve veri modüllerinin tümü final halindeyken çalıştırılmış olup, sonrasında yalnızca `hyperopt.py` post-study snapshot yazımı `#63` ile düzeltilmiş ve ardından tüm `918` testlik suite ile salt-okunur smoke kontrolleri tekrar geçilmiştir.)*
  - **3-Fold Purged CV:** Fold 1 `DA=0.6691, HAC=+1.4765, MAE/0=0.9737`; Fold 2 (single seed `43`) `DA=0.4173, HAC=-0.0647, MAE/0=1.0116`; Fold 3 `DA=0.5588, HAC=+0.0715, MAE/0=0.9899`, `crossing=0.0`.
  - **3-Seed Holdout (`2026-06-08..2026-11-20`, `n=116`):** `3/3` seed kabul; Val `DA=0.6413, HAC=+1.4069`; Test `weekly_directional_accuracy=0.5690`, `weekly_sharpe_ratio=+0.1881`, `weekly_sharpe_ratio_hac=+0.1180`, `weekly_mae_vs_naive_zero=1.0034`, `weekly_magnitude_ratio=0.0713`, `weekly_pi80_coverage=0.8534`, `weekly_pi96_coverage=0.9655`, `crossing=0.0000`, `T+1 DA=0.5862, Sharpe=+2.3916`, **`evaluate_quality_gate_metrics(tm)` → `passed=True, reasons=[]`**.
  - **`TFTBacktestRunner`:** Tam OOS (`n=216`) `DA=0.5972, Sharpe=+0.6962` (`TFT_SUPERIOR` 3–0 vs Theta); Strict Post-Val OOS (`n=119`, `mean_actual=-0.004846`) **`DA=0.5630, Sharpe=+0.0900, MCC=+0.1793`** (`TFT_SUPERIOR` 2–1 vs Theta `DA=0.5378, Sharpe=-0.1342`).
- **Depo Hijyeni (`git status --short`):**
  - `backend/checkpoints` ve `backend/lightning_logs` dizinlerinin mevcut olmadığı ve geride hiçbir geçici scratch dosyası kalmadığı doğrulandı (`backend/app/`, `backend/deep_learning/`, `backend/long_term/`, `backend/migrations/`, `backend/scripts/`, `backend/worker/`, `backend/backtest/`, `.github/workflows/tft-training.yml`, `backend/tests/`, `frontend/src/` — ek olarak çalışma ağacında bu iyileştirmeden bağımsız önceden mevcut Stage-03 Venus/Galaxy UI değişiklikleri bulunmaktadır).

---

## 6. Operasyonel Sonraki Adımlar (Production Checklist)

> [!IMPORTANT]
> Bu çalışma kapsamında **uzak production veritabanında hiçbir yazma/silme işlemi yapılmamış** ve **Hugging Face Hub'a hiçbir model veya kalibrasyon artefaktı push edilmemiştir**. Aşağıdaki adımlar canlı ortamda operatör tarafından sırayla uygulanmalıdır.

1. **Adım 1 — `price_bars` Veritabanı Yedeği ve Tekilleştirme Temizliği:**
   - Önce tam DB yedeği (veya `price_bars` snapshot'ı) alın.
   - `backend/` dizininden `py -3 scripts/dedup_price_bars.py --dry-run` komutunu çalıştırıp temizlenecek sayının beklenen düzeyde olduğunu (`~16,532` satır: `16,459` çift grup + `73` hafta sonu hayalet barı; `2899.HK` Pazartesi barları korunarak) doğrulayın.
   - Doğrulama sonrası temizliği uygulayın:
     ```powershell
     cd backend
     py -3 scripts/dedup_price_bars.py --apply
     ```

2. **Adım 2 — Uzun Vadeli Görünüm Veritabanı Migrasyonu:**
   - [`backend/migrations/005_long_term_outlook_snapshots.sql`](../../backend/migrations/005_long_term_outlook_snapshots.sql) betiğini production PostgreSQL üzerinde çalıştırarak `long_term_outlook_snapshots` tablosunu ve üç indeksini oluşturun (veya uygulama başlangıcındaki `init_db()` akışının tabloyu oluşturduğunu doğrulayın).

3. **Adım 3 — Temiz Veri Üzerinden TFT-ASRO Yeniden Eğitimi ve HF Hub Promotion:**
   - Temizlenmiş `price_bars` üzerinde yeni Optuna hiperparametre taramasının (`feature_snapshot.pkl` kazanan `mrmr_top_k` ile) çalışması, 3-seed modelin `quality_gate.py`'yi geçmesi ve yeni checkpoint + `conformal_calibration.json` (`base_vol_power` dahil) paketinin HF Hub'a promote edilmesi için [`.github/workflows/tft-training.yml`](../../.github/workflows/tft-training.yml) iş akışını **`run_hyperopt=true`** ile tetikleyin:
     ```powershell
     gh workflow run tft-training.yml -f run_hyperopt=true
     ```

4. **Adım 4 — Günlük Pipeline ile Canlı Snapshot'ların Doldurulması:**
   - Eğitim ve HF Hub promotion tamamlandıktan sonra [`.github/workflows/daily-pipeline.yml`](../../.github/workflows/daily-pipeline.yml) iş akışını tetikleyerek hem güncellenmiş `5D` TFT tahmin snapshot'larının (`tft_prediction_snapshots`) hem de `20D / 1M` & `60D / 1Q` Uzun Vadeli Görünüm snapshot'larının (`long_term_outlook_snapshots`) canlı veritabanına yazılmasını sağlayın:
     ```powershell
     gh workflow run daily-pipeline.yml
     ```
