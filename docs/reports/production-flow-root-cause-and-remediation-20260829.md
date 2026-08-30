# Production Flow Root Cause and Remediation — 2026-08-29

## Kapsam ve sınır

Bu çalışma news sentiment, AI commentary, günlük pipeline, XGBoost training/artefact lifecycle, Price Forecast ve TFT-ASRO inference/API/UI sınırını uçtan uca izledi. TFT-ASRO training, loss, calibration ve quality-gate kodu bu remediation tarafından değiştirilmedi. İncelemenin başladığı `9360011` tabanında 15–29 Ağustos arasında news/LLM günlük runtime dosyalarına dokunan tek commit `4e82e07` (`backend/worker/tasks.py`, Market Drivers onarımı) oldu; LLM regresyonunu açıklayan bir LLM kod/dependency commit’i bulunmadı. Rollout öncesi merge edilen TFT quality-gate PR #4 rebase ile korundu ve birleşik suite yeniden çalıştırıldı.

## Çıkarılan akışlar

- News: Google News/NewsAPI → `news_raw` → canonical preprocessing/dedup → OpenRouter + FinBERT → `news_sentiments_v2` → horizon-aware günlük aggregate → `/api/news*` → Overview news panel.
- XGBoost: finite `price_bars` + daily sentiment → ortak feature builder → kronolojik train/validation split → atomik `model_artifacts` candidate/promotion → aynı bundle’dan inference → `analysis_snapshots` → `/api/analysis` → Price Forecast/commentary.
- TFT-ASRO: haftalık training workflow’unda üretilen artefact → günlük pipeline’da yalnız inference → `tft_prediction_snapshots` → `/api/analysis/tft/{symbol}` → frontend. Günlük pipeline içinde ayrı bir deep-learning training adımı yoktur.
- Pipeline: authenticated enqueue → ARQ worker ve advisory lock → stage sonuçları → snapshot/commentary → terminal evaluator → authenticated run-status polling.

## Doğrulanmış root cause’lar

| Alan | Kanıt | Kök neden | Etki |
| --- | --- | --- | --- |
| LLM fallback | 29 Ağustos canlı exact-contract probe: `minimax/minimax-m2.5:free` scoring ve commentary çağrıları HTTP 404, `model_unavailable`; provider mesajı free sürümün artık mevcut olmadığını söylüyor. | Fast, reliable ve commentary rollerinin kaldırılan tek slug’a bağlanması; 404 ve parse nedenlerinin gözlenebilir olmaması. | Normal LLM yolu tamamen `deterministic_fallback` oldu. |
| Price Forecast/TFT fiyatları | Deploy-öncesi production: XGB `current_price=6.697`, `predicted_return=0.002593`, `predicted_price=6.6041`, fakat baseline alanı yok; TFT reference/price alanları null; commentary `$0.0000` içeriyor. DB’de son `HG=F` close `NaN`. | `NaN` close ingest edildi ve latest sorgularında geçerli sayıldı; commentary missing price’ı sıfıra çevirdi. | TFT fiyatları null, commentary sahte sıfır, health yanlış freshness. |
| XGB price basis | Production değeri `6.6041`, `6.587 × (1 + 0.002593)` ile eşleşiyor; canlı `6.697` ile eşleşmiyor. | Tahmin DB close baseline’ına göre doğru hesaplanıyor, fakat schema baseline metadata’sını düşürüyordu. | UI aynı raporda iki farklı fiyat temelini açıklamasız gösteriyordu. |
| Pipeline green false-positive | Eski workflow yalnız enqueue HTTP cevabını kontrol edip saniyeler içinde bitiyordu; DB’de `snapshot_generated=false`, failed stage ve kalıcı `running` kayıtları bulunuyordu. | Worker terminal state’i GitHub job’a bağlanmamıştı; kritik stage’ler catch edilip job başarılı dönebiliyordu. | Başarısız/stale production run’ı yeşil görünüyordu. |
| XGB artefact split-brain | Model dosyası ve DB metadata’sı ayrı yazılıyor, ortak version/hash taşımıyordu. | Restart/kısmi yazmada model ile feature/target metadata’sı bağımsız yükleniyordu. | Eski model + yeni feature listesi veya yanlış target yorumu mümkündü. |
| News doğruluğu | Wrapper URL temelli dedup, küçülen query üzerinde offset, `language="en"`, horizon filtresiz join ve publisher filtresinden önce 500-row limit kodda reproduce edildi. | İçerik kimliği ve pagination yanlış katmanda uygulanıyordu. | Batch atlama, duplicate scoring, yanlış dil/horizon/count/pagination. |
| Cache/staleness | News cache anahtarı pipeline sürümünü taşımıyordu; Overview toplu promise failure’ında endpoint’ler birbirini etkiliyor ve eski veri gözlenebilir olmadan kalabiliyordu. | Cache invalidation ve endpoint-bazlı error state yoktu. | Yeni run sonrası stale feed/dashboard. |
| News cut-off silent failure | İlk production canary’de `news_cutoff_time` yerine `cutoff_error` oluştuğu halde run başarılıydı. | Worker fonksiyonundaki iç `from datetime import datetime` import’u outer scope’taki `datetime` adını gölgeledi; evaluator cut-off’u kritik saymıyordu. | Snapshot yanlış haber zaman penceresiyle üretilebilirdi. |
| FinBERT stale-work amplification | İlk canary yalnız 24 yeni haber varken 22 embedding üretip 1.034 satırı skip etti. | Sorgu önce tüm mevcut skorları çekiyor, embedding varlığını Python döngüsünde eliyordu. | Her günlük run gereksiz şekilde birkaç dakika uzuyordu. |
| Commentary input loss | Duplicate-only canary’de news `stale` olunca commentary report tüm model/fiyat girdilerini de temizliyordu. | Haber freshness’i ile model availability aynı `quality_state` dalında birleştirilmişti. | LLM çalışsa bile XGB/TFT için yanlış “unavailable” yorumu üretilebiliyordu. |
| Advisory lock leak | İlk lock düzeltmesi sonrasında terminal success’e rağmen production `pg_locks` aynı lock id’sini idle pooled connection üzerinde tuttu. | Session-level PostgreSQL advisory lock, stage transaction’larını da taşıyan pooled ORM session’ına bağlıydı; rollback/connection dönüşü ownership’i belirsizleştiriyordu. | Sonraki GitHub run HTTP 409 aldı ve health kalıcı locked göründü. |
| Heatmap provider noise | HF container loglarında `COP.AX`, `NOVR`, `NI=F`, `PB=F` için tekrarlı Yahoo “quote not found” 404’leri görüldü. | Ocak ayından kalan seed ticker’ları provider’da artık yok veya hiç desteklenmiyordu. | Heatmap refresh’i gereksiz provider çağrısı/log hatası üretiyordu; bu dört seed zaten veri gösteremiyordu. |

## Uygulanan kalıcı düzeltmeler

### News ve LLM

- Raw haber audit kaynağı olarak korunuyor. Publisher normalize edilip kolonda tutuluyor; processed content kimliği `canonical title + publisher + UTC publication date`. Duplicate satırlar silinmeden `duplicate_of_id` ve `dedup_version` ile işaretleniyor; scoring/aggregation/API yalnız canonical satırları kullanıyor.
- Shrinking unprocessed query’de offset kaldırıldı; gerçek language detection processing yoluna alındı. Dry-run varsayılanlı `scripts/repair_production_flow.py` dedup etkisini raporluyor; `--apply` yalnız ilişkileri ve yeniden üretilebilir aggregate’i değiştiriyor.
- OpenRouter client hata sınıfları: `auth`, `model_unavailable`, `rate_limit`, `timeout`, `network`, `provider_5xx`, `context_limit`, `unsupported_contract`, `empty_response`; parser ayrıca `parse_invalid` üretir.
- 404 retrysiz sonraki modele geçer; 401/403 zinciri durdurur; 408/429/5xx/network model başına en fazla bir retry yapar. `Retry-After` 30 saniyede, tüm scoring batch’i 120 saniyede sınırlıdır.
- Roller ayrıldı: fast M2.7, reliable/commentary M3; Gemma 4 ve GLM 5.2 kontrollü ikincil zincirdir. Parser eksik/ek alan, invalid enum ve aralık dışı değerleri uydurmaz/clamp etmez. Repair farklı modelle bir kez denenir.
- Haber metni untrusted data olarak delimite edilir; içerikteki talimatlar sistem/output contract’ını değiştiremez, tool erişimi yoktur.
- Sentiment/commentary payload’ları actual model, generation/scoring mode, fallback reason ve attempt özetini taşır. Missing predicted/current price artık sıfıra çevrilmez.

### Price, preprocessing ve model lifecycle

- Ortak finite-price validator ingest, history, health, XGB ve TFT inference/freshness sınırlarında kullanılır. Non-finite/pozitif olmayan close yazılmaz ve latest sayılmaz.
- Training ve inference aynı symbol seti, limited forward-fill, sentiment fill ve final non-finite→0 preprocessing fonksiyonunu kullanır.
- XGBoost model blob, SHA-256, features, metrics, importance ve data-window fingerprint tek `ModelArtifact` satırında tutulur. Candidate reload, exact feature-name ve finite smoke inference geçmeden transaction içinde active olmaz.
- Inference model ve metadata’yı aynı active row’dan yükler. Additive rollout için `auto` modu önce DB artefact’ını, yoksa feature-equality doğrulanmış legacy file bundle’ını kullanır.
- XGB response `current_price` ile matematiksel `baseline_price/date`, target type, source, equation ve artefact version’ı ayırır. Tek denklem `predicted_price = baseline_price × (1 + predicted_return)` olur.
- TFT-ASRO output’unda nullable price alanları, explicit reference price/date, return basis ve backend-authoritative forecast dates bulunur. Frontend tarih/fiyat üretmez; yalnız backend değerini formatlar.

### Pipeline, API ve frontend

- Run enqueue anında `queued`; worker başlangıcı, stage JSON, LLM fallback sayıları, commentary mode ve requested/promoted/used artefact sürümleri persist edilir. İki saatten eski queued/running satırlar `worker_interrupted` ile kapatılır.
- Kritik stage/snapshot/training/artefact mismatch ARQ exception ve terminal `failed`; tüm operational LLM fallback veya commentary fallback `degraded`; language/short-text policy skip ayrı sayılır.
- `GET /api/pipeline/runs/{run_id}` aynı bearer auth ile terminal durumu döndürür. Daily workflow 30 saniyede bir en fazla 60 dakika poll eder; failed/degraded/locked/timeout kırmızıdır. Günlük ve Pazar-weekly cron ayrıdır; training kararı trigger edilen cron ifadesinden gelir.
- News API current horizon join, DB publisher filter/count, `published_at DESC, processed_id DESC`, sabit `as_of`, `data_as_of` ve completed-pipeline-version cache key kullanır.
- Frontend nullable contract ile eşlendi; actual/fallback badge gösterir; endpoint’leri `Promise.allSettled` ile bağımsız yükler ve endpoint-bazlı unavailable state gösterir.
- CI backend testlerine ek olarak frontend `npm ci`, lint, Vitest ve type-check/build koşar.
- Heatmap seed listesindeki canlı provider tarafından kalıcı 404 döndürülen dört sonuçsuz ticker çıkarıldı; çalışan bir heatmap öğesi veya forecast sembolü değiştirilmedi.

## Değişen dosya grupları ve amaçları

- LLM/news contract ve veri akışı: `backend/app/openrouter_client.py`, `backend/app/ai_engine.py`, `backend/app/commentary.py`, `backend/pipelines/{ingestion,processing}/news.py`, `backend/app/main.py`, `backend/app/models.py`, `backend/app/schemas.py`.
- Price ve model lifecycle: `backend/app/price_utils.py`, `backend/app/data_manager.py`, `backend/app/features.py`, `backend/app/inference.py`, `backend/app/db.py`.
- Günlük worker/observability: `backend/worker/tasks.py`, `backend/app/lock.py`, `backend/app/scheduler.py`, `.github/workflows/daily-pipeline.yml`.
- TFT-ASRO yalnız inference/API sınırı: `backend/deep_learning/inference/predictor.py`, `backend/deep_learning/models/tft_copper.py`; training loss, calibration ve quality-gate değiştirilmedi.
- Frontend sözleşmesi/rendering: `frontend/src/types.ts`, `frontend/src/api.ts`, `frontend/src/pages/OverviewPage.tsx`, `frontend/src/hooks/useNews.ts`, `frontend/src/features/news/NewsCard.tsx`, `frontend/src/utils/forecast.ts`.
- Rollout/CI/onarım: `.github/workflows/{tests,hf-sync}.yml`, `backend/scripts/{probe_openrouter_contracts,repair_production_flow}.py`, `env.example`, `docker-compose.yml` ve regression testleri.

## Matematiksel Price Forecast doğrulaması

- Canlı örnek kaynak output: baseline DB close `6.587`, XGB simple return `0.002593`.
- Beklenen: `6.587 × 1.002593 = 6.604079...` → API rounding ile `6.6041`.
- Canlı quote `6.697` yalnız current/display price’tır; tahmin temelinde kullanılsaydı `6.71436...` çıkardı. Regression testi iki değerin karışmasını engeller.
- TFT log-return yolu ayrı tutulur: her horizon fiyatı `reference_price × exp(cumulative_log_return)`; frontend inverse-transform yapmaz.

## Doğrulama

- Backend offline suite (TFT PR #4 ile rebase ve canary bulguları sonrası): **519 passed, 15 skipped**, failure yok. Skip’ler online/credential bağımlı testlerdir.
- Frontend: ESLint geçti; Vitest **2/2** geçti; TypeScript + Vite production build geçti. Yalnız mevcut >500 kB chunk warning’i kaldı.
- OpenRouter live exact-contract:
  - M2.5 free: 404 / `model_unavailable`, başarısız (doğrulanmış root cause).
  - M2.7 fast: HTTP 200, strict sentiment contract repairsiz geçti.
  - M3 reliable: HTTP 200, strict sentiment contract repairsiz geçti.
  - M3 commentary: HTTP 200, structured commentary repairsiz `generation_mode=llm`.
  - Gemma 4 ve GLM 5.2 secondary probe: upstream 429; canlı contract başarısı kanıtlanmadı.
- Mock testler canlı provider başarısı olarak değerlendirilmedi. Primaries için ayrıca gerçek credential çağrısı yapıldı.

## Production rollout ve canlı kanıt

- Additive migration’dan sonra non-destructive news backfill uygulandı: 13.026 processed row tarandı, 3.168 duplicate ilişkisi, 13.026 dedup version ve 10.959 publisher alanı güncellendi; 607 günlük aggregate canonical kayıtlardan 530 satır olarak yeniden üretildi. Tekrarlanan dry-run update sayılarının tamamını sıfır verdi. Sonraki pipeline’larla aggregate 531 güne çıktı.
- Fiyat overlap ingest ilk onarımda 23 invalid historical close’u 9’a indirdi. Sonraki normal overlap run’ları geçerli barları aynı tarih üzerine yazdığı için 30 Ağustos denetiminde yalnız iki pre-deploy audit satırı (`SCCO` 11 Ağustos, `ALI=F` 28 Ağustos) non-finite kaldı; latest/history/health/training/inference sorgularının hiçbiri bunları kullanmıyor.
- Kontrollü `train_model=true` canary `c8563002-3803-4d71-b5f4-449ee64b0d20` normal LLM scoring/commentary ile tamamlandı ve `xgb-HG_F-20260829T223338Z-0a52473b0672` artefact’ını atomik promote etti. Cut-off, embedding ve commentary quality edge-case’leri bu canary’den türetilen regression testleriyle düzeltildi.
- İkinci training canary `dd358406-6eaa-44b3-b331-c4d829b0fe02`, aynı data fingerprint/model SHA için `xgb-HG_F-20260829T225133Z-0a52473b0672` sürümünü promote etti. Production’da bu sürüm tek active artefact; SHA-256 `0a52473b0672633a1aadb6599f74d95798473c96b7e8d2aceaade80674c16748`. Snapshot ve active version eşleşti.
- Son lock canary `c45c8e32-6986-4240-9907-303af90f55c7` `success/ok` oldu: 3/3 haber LLM success, 0 parse/operational/policy fallback, commentary `minimax/minimax-m3:free` ile `generation_mode=llm`, snapshot active artefact ile üretildi. GitHub run `33280721767` terminal success aldı.
- HF container logu aynı canary için advisory lock acquire’ı `23:20:03Z`, pipeline completion’ı `23:25:35Z` ve release’i `23:25:36Z` olarak gösterdi. Ardından health `pipeline_locked=false`; production DB’de iki saatten eski queued/running run bulunmadı. Sonraki bağımsız run `eb97097e-edac-4809-8ce5-801add7bd437` da lock’ı acquire/release edip terminal success oldu.
- Canlı XGB response: baseline `6.562` (2026-08-28 DB close), public rounded return `0.002678`, predicted `6.5796`, current/live display `6.659`. Public yuvarlanmış alanlarla denklem farkı yalnız `0.000027`; hesap içte yuvarlanmamış return ile tek price-basis fonksiyonunda yapılır.
- Canlı TFT response: reference `6.56199979782104` (2026-08-28), `return_basis=daily_log_return_path`; ilk horizon 2026-08-31 ve `6.56199979782104 × exp(-0.050546199...) = 6.238558865...`. Frontend bu fiyatı/tarihi yeniden hesaplamıyor.
- Canlı news pagination’da aynı `as_of` ile iki ardışık 5’li sayfa arasında id çakışması sıfırdı. Commentary normal LLM modunda ve fallback reason null kaldı.
- Son uygulama commit’indeki CI `33280521186`, HF deploy `33280626051` ve final daily smoke `33280721767` başarılıdır.

## Kalan riskler

- Free-tier model availability/rate limit dış bağımlılıktır. Primary exact-contract ve production run canlı geçti; Gemma 4/GLM 5.2 secondary zinciri upstream 429 nedeniyle henüz canlı contract başarısı kanıtlamadı. Attempt/fallback telemetry bu ayrımı görünür yapar.
- İki non-finite historical PriceBar audit satırı bilerek silinmedi. Geçerli provider barı gelirse overlap ingest bunları idempotent overwrite eder; tüm consumer’lar şimdiden finite-only’dir.
- TFT-ASRO raw quantile crossing production logunda observable olmaya devam ediyor (`raw_crossing_rate≈0.63`); public structural monotonic transform crossing’i sıfıra indiriyor. Training/loss/calibration/quality-gate bu işin açık kapsam sınırı nedeniyle değiştirilmedi.
- Legacy filesystem XGB bundle additive compatibility yolunda kalır; production source of truth DB active artefact’tır. `db_required` sıkılaştırması ayrı rollout kararıdır.
- Vite build’in büyük main chunk warning’i işlevsel bloklayıcı değildir; ayrı performans/code-splitting işi olarak ele alınmalıdır.
