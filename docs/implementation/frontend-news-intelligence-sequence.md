# CopperMind Terra Rara — News → Intelligence sequence

Güncelleme: 2026-09-10. Önceki landing/performance/particle çalışma ağacının checkpoint'i: `839e930` (`chore: checkpoint cinematic landing before news sequence`). Bu belge, o checkpoint'ten sonra yapılan News Intelligence landing değişikliğinin uygulama planı ve veri-doğrulama kaydıdır.

## Ürün kararı

Market heatmap'te çalışan “ürünü açıklama; scroll sırasında çalışırken göster” deseni News Intelligence'a taşınır. Market → News geçişi aynı şirket/ticker bağlamını korur: örnek heatmap hücresindeki `FCX` seçimi, sonraki sahnede `FCX / Freeport-McMoRan` haber bağlamının giriş işareti olur. Bu görsel continuity, haber skorunun forecast'e doğrudan girdiği iddiası değildir.

Landing preview deterministiktir ve `/api/news`, `/api/commentary` ya da OpenRouter çağırmaz. `frontend/src/features/landing/preview-data.ts` içindeki fixture yalnızca production veri sözleşmesinin alan adlarını ve sırasını öğretmek için kullanılır; haber, skor ve tarih canlı veri değildir.

## Production pipeline doğrulaması

Kod incelemesi `backend/pipelines/ingestion/news.py`, `backend/pipelines/processing/news.py`, `backend/app/ai_engine.py`, `backend/app/main.py`, `backend/worker/tasks.py` ve `backend/app/commentary.py` üzerinden yapıldı.

1. **Ingestion:** Günlük worker önce Google News RSS sorgularını (copper supply deficit, price forecast, mining production, China demand, EV/battery ve büyük üreticiler) `news_raw` tablosuna yazar; `NEWSAPI_KEY` varsa sınırlı NewsAPI sorguları da çalışır. URL normalize edilip `url_hash` ile duplicate ingest engellenir. Başlık, açıklama, kaynak, publisher, yayın/fetch zamanı ve raw payload saklanır.
2. **Preprocessing / dedupe:** `news_raw → news_processed` aşaması başlık ve açıklamayı temizler, canonical title/language üretir ve `canonical_title + publisher + publication date` içeriğiyle `content_v2` dedup anahtarı kurar. Duplicate satırlar audit için tutulur ve `duplicate_of_id` ile işaretlenir; scoring yalnızca canonical kayda gider.
3. **Article scoring:** V2 LLM prompt'u yalnızca başlık + açıklama/metin ve 1–5 işlem günü HG=F etkisi bağlamını alır. Geçerli cevapta zorunlu alanlar `label`, `impact_score` [-1,1], `confidence`, `relevance`, `event_type` ve en fazla 160 karakterlik `reasoning`'dir. Event vocabulary gerçek sistemdeki supply/demand, inventory, policy, USD, cost-push, `mixed_unclear` ve `non_copper` kovalarıyla sınırlıdır.
4. **Tone + ensemble:** FinBERT `pos / neu / neg` olasılıkları üretir; tone yönü `pos - neg` olarak alınır. `final_score`, LLM impact, FinBERT polarity, event rule sign/strength ve relevance/confidence üzerinden deterministik ensemble ile hesaplanır. Frontend'in `confidence` alanı `confidence_calibrated`, `relevance` ise `relevance_score` projection'ıdır.
5. **Fallback:** İngilizce olmayan/kısa metin, kota, auth, rate-limit veya parse sorunu olan satır FinBERT + event-rule fallback'e gidebilir. API bunu `scoring_mode=deterministic_fallback` ve `fallback_reason` ile açıklar. Landing bunu canlı fallback gibi göstermez; preview yalnızca production-shaped bir örnektir.
6. **Frontend contract:** `/api/news` `title`, `description`, `publisher`, `channel`, `published_at` ve `sentiment` bloğunu döndürür. Sentiment projection'ında `label`, `final_score`, `impact_score_llm`, calibrated `confidence`, `relevance`, `event_type`, FinBERT olasılıkları, kısa `reasoning`, `scoring_mode`, model/fallback metadata'sı vardır.
7. **Commentary sınırı:** `/api/commentary` makale bazlı bir commentary üretmez. Worker Stage 6, forecast snapshot'ı, agregat sentiment index/label'ı, haber sayısını ve top influencers'ı birleştirip `AICommentary` tablosuna sembol başına cached commentary yazar. Bu nedenle landing final adımı makale için gerçek API'de bulunan `reasoning` alanını **LLM rationale / article read** olarak adlandırır; global forecast commentary'sini haber makalesinin sonucuymuş gibi birleştirmez.

## Scroll choreography

`CinematicLanding` mevcut tek `scrollYProgress` MotionValue'ını korur. `NewsPreview` içindeki katmanlar bu timeline'ın local `[0,1]` türeviyle scrub edilir; progress React state'e yazılmaz.

| Local progress | Görsel katman | Kullanıcıya kalan tek fikir |
| --- | --- | --- |
| 0.00–0.30 | `SOURCE HEADLINE` | Haber önce okunabilir, sade bir editorial yüzeydir. |
| 0.13–0.46 | `SEMANTIC EMPHASIS` | Şirket, ticker ve olay ifadesi ayrışır; marketteki `FCX` bağlamı korunur. |
| 0.27–0.67 | `TONE SIGNAL` | FinBERT olasılıkları bar değil, negative–neutral–positive spectrum üzerinde yön kazanan Copper Signal olarak görünür. |
| 0.44–0.82 | `IMPACT SCORE` | `label`, `impact_score_llm`, `final_score`, calibrated confidence/relevance ve gerçek event type sırayla görünür. |
| 0.68–1.00 | `INTERPRETATION` | Başlığı tekrar etmeyen kısa production-shaped `reasoning` cümlesi compose olur; bu makale-level rationale'dır. |
| 0.90–1.00 | Exit | Haber katmanı sonraki forecast sahnesine çözülür; news'in forecast girdisi olduğu söylenmez. |

Katmanlar overlap eder, fakat aynı anda tüm skorlar öne çıkarılmaz. Back-scroll aynı MotionValue aralıklarını tersine çalıştırır; hızlı scroll'da discrete slide/index state yoktur. `MarketPreview` ile NewsPreview aralığındaki mevcut crossfade, origin strip'teki `FCX → source → intelligence` işaretiyle neden-sonuç hissini taşır.

## Uygulama yüzeyi

- `frontend/src/features/landing/preview-data.ts`: production-shaped, deterministic news fixture.
- `frontend/src/features/landing/Previews.tsx`: headline, entity, tone SVG spectrum, score orbit ve rationale katmanları; static/reduced-motion görünümü.
- `frontend/src/features/landing/CinematicLanding.tsx` ve `ResearchStory.tsx`: news copy'si “tone scoring + LLM rationale” olarak güncellendi; global scene/timeline korunuyor.
- `frontend/src/features/landing/landing.css`: compositor-friendly opacity/transform/clip-path katmanları, mobil stage sıkıştırması ve static flow. Runtime LLM, network polling veya yeni WebGL yok.
- `frontend/scripts/check-experience.mjs` ve `frontend/src/experience.test.tsx`: yeni preview'nin render edilmesi, forbidden internal labels, no-API landing kuralı ve route regresyonları korunacak; local run sonrası evidence raporuna eklenecek.

## Responsive, accessibility ve performance

- Desktop'ta pinned surface üzerinde spatial overlap; mobilde aynı bilgi sırası daha kısa stage ve staged/static flow ile korunur.
- `prefers-reduced-motion`, Save-Data ve düşük CPU fallback'i mevcut policy üzerinden static preview'yi kullanır; headline, entity, tone, score ve rationale içerikleri kaybolmaz.
- Signal SVG tek küçük çizimdir; LLM çağrısı, canvas allocation, filter-heavy layer veya per-frame React render eklenmez. `transform`, `opacity`, `clip-path` ve sınırlı SVG path reveal kullanılır.
- Preview dekoratif cinematic surface içinde `aria-hidden` olabilir; reduced/static flow'da gerçek açıklama ve caption DOM'da kalır. Üretim News panelinin keyboard/drawer davranışına dokunulmaz.

## Kabul ölçütleri

- Heatmap → News geçişi aynı `FCX` bağlamını görsel olarak taşır.
- Kullanıcı haber → entity → tone → score → rationale sırasını açıklama kartı okumadan izler.
- Scoring/tonality ile interpretation/rationale ayrı yüzeylerdir.
- Yalnızca production'da bulunan alan adları/kategoriler kullanılır; article-level commentary iddiası yapılmaz.
- Landing API'siz, deterministic ve mevcut route/lazy boundary'leri bozmadan çalışır.
- Build, lint, test, experience QA ve reduced-motion/mobile kontrolleri geçer; gerçek cihaz FPS/field Web Vitals ölçülmediyse raporda “ölçüm bulunamadı” denir.
