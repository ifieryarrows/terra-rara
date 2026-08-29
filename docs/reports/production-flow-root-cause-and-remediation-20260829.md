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

## Matematiksel Price Forecast doğrulaması

- Canlı örnek kaynak output: baseline DB close `6.587`, XGB simple return `0.002593`.
- Beklenen: `6.587 × 1.002593 = 6.604079...` → API rounding ile `6.6041`.
- Canlı quote `6.697` yalnız current/display price’tır; tahmin temelinde kullanılsaydı `6.71436...` çıkardı. Regression testi iki değerin karışmasını engeller.
- TFT log-return yolu ayrı tutulur: her horizon fiyatı `reference_price × exp(cumulative_log_return)`; frontend inverse-transform yapmaz.

## Doğrulama

- Backend offline suite (TFT PR #4 ile rebase sonrası): **514 passed, 15 skipped**, failure yok. Skip’ler online/credential bağımlı testlerdir.
- Frontend: ESLint geçti; Vitest **2/2** geçti; TypeScript + Vite production build geçti. Yalnız mevcut >500 kB chunk warning’i kaldı.
- OpenRouter live exact-contract:
  - M2.5 free: 404 / `model_unavailable`, başarısız (doğrulanmış root cause).
  - M2.7 fast: HTTP 200, strict sentiment contract repairsiz geçti.
  - M3 reliable: HTTP 200, strict sentiment contract repairsiz geçti.
  - M3 commentary: HTTP 200, structured commentary repairsiz `generation_mode=llm`.
  - Gemma 4 ve GLM 5.2 secondary probe: upstream 429; canlı contract başarısı kanıtlanmadı.
- Mock testler canlı provider başarısı olarak değerlendirilmedi. Primaries için ayrıca gerçek credential çağrısı yapıldı.

## Rollout ve kalan riskler

- Rollout sonucu bu rapora deploy/canary tamamlanınca eklenecektir.
- Free-tier model availability/rate limit dış bağımlılıktır. Primary exact-contract canlı geçti; secondary zincir 429 nedeniyle henüz doğrulanmadı. Attempt/fallback telemetry bu ayrımı production run’da görünür yapar.
- Historical invalid PriceBar satırları audit için silinmez; tüm okuma yolları bunları dışlar ve overlap ingest yeniden geçerli bar almaya çalışır.
- Legacy filesystem XGB bundle yalnız additive geçiş içindir. İlk kontrollü `train_model=true` canary sonrasında DB active artifact source of truth olur; `db_required` sıkılaştırması ayrı rollout kararıdır.
- Vite build’in büyük main chunk warning’i işlevsel bloklayıcı değildir; ayrı performans/code-splitting işi olarak ele alınmalıdır.
