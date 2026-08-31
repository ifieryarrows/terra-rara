# Stock Heatmap Uçtan Uca Yenileme — 2026-08-30

## Kapsam ve yöntem

Bu çalışma mevcut heatmap’i frontend component/state akışı, D3 treemap yerleşimi, Overview/News yerleşimi, backend snapshot üretimi, API/cache sözleşmesi, provider çağrıları, news context, logo yükleme ve kullanıcı etkileşimi boyunca yeniledi. Finviz yalnızca bilgi hiyerarşisi ve etkileşim yoğunluğu için referans alındı; CopperMind renkleri, tipografisi, copper focus/selection semantiği ve mevcut runtime stack’i korundu.

Karşılaştırma; 1536×864 masaüstü, 768 px tablet, 390 px mobil, gerçek 194 araç ve sentetik 1.000 araç üzerinde yapıldı. API baseline üretim ortamından, yeni UI/layout ölçümleri yerel uygulamadan alındı. Aynı ortamda deploy-sonrası ölçüm olmayan metrikler başarı gibi sunulmadı.

## İncelenen mevcut mimari ve commit geçmişi

- Frontend akışı: `OverviewPage` → lazy `HeatmapPanel` → filtre/state → D3 hierarchy/treemap → DOM stock/category hücreleri → tek portal tooltip.
- Backend akışı: `broad_universe.csv` → yfinance quote/history → tema snapshot’ı → `heatmap_cache` → `/api/market-heatmap` → React Query.
- Eski görünüm CopperMind group/subgroup taksonomisini kullanıyor ve Overview’daki News sütunu nedeniyle masaüstünde yaklaşık 772 px genişliğe sıkışıyordu.
- `c73ba07`, `5ef28e4`, `c074cc9`, `07d349d`, `4f9a4f2`, `c80a661`, `4d5a9a5` ve `31ee3fc` heatmap geçmişi incelendi. `c80a661` pointer konumunu rAF ile hafifletmişti; sonraki bileşen değişimlerinde bu davranış kaybolmuştu. Yeni tooltip bu optimizasyonu imperative ref + rAF ile açık biçimde geri getiriyor.
- Başlangıçta 194 araç, 19 üst grup, 62 alt grup; heatmap içinde 286, tüm Overview’da 1.299 DOM descendant vardı.
- Başlangıç frontend build ve 2 Vitest testi ile 29 API testi geçiyordu; heatmap’e özgü regression testi yoktu.

## Uygulanan veri ve API mimarisi

### İki geriye uyumlu görünüm

- Parametresiz `/api/market-heatmap` eski `themes` ağacını döndürmeye devam ediyor.
- Frontend varsayılan olarak `view=market` kullanıyor.
- Equity: `sector → industry → stock`.
- ETF/fon: `Funds & ETFs → mevcut theme/type → instrument`.
- Future/emtia: `Commodities & Futures → mevcut theme/type → instrument`.
- Currency, digital asset ve index için ayrı asset-class dalları var.
- Eksik equity metadata’sı `Other Equities → Unclassified Equities` altında güvenli biçimde kalıyor.
- Yol bazlı SHA-1-derived category ID’leri görünür addan bağımsız ve deterministik. Leaf ID’si ticker kimliğine bağlı.

Leaf sözleşmesine opsiyonel `id`, `instrumentType`, `sector`, `industry`, `exchange`, `logoTicker`, `sparkline` ve `asOf` alanları eklendi. Eski alanlar ve tema ağacı korunuyor.

### Sparkline ve news context

- Refresh başında bütün semboller için tek batched son-üç-aylık günlük history çağrısı yapılıyor; dönemin ilk ve son kapanışını koruyan, üç ay boyunca eşit aralıklı en fazla 10 finite close ilk değere göre 100 tabanına normalize ediliyor.
- History çağrısı başarısızsa quote heatmap yayımlanmaya devam ediyor.
- `/api/market-heatmap/context` yalnızca mevcut cached tree’de bulunan category ID’lerini kabul ediyor.
- Son yedi günlük, en fazla 250 canonical news satırında ticker ve şirket adı deterministik olarak eşleştiriliyor.
- Aynı context response’u geriye uyumlu category `news` alanına ek olarak category içindeki eşleşen hisseler için `stockNews` haritası döndürüyor. Her haber yalnız bir kez tokenize edilip bütün ticker/company adayları aynı geçişte puanlanıyor.
- Kart aktif ticker’ın gerçek cached haberini seçiyor; o hisse için eşleşme yoksa kompakt `No recent news is available for {TICKER}.` durumu gösteriliyor. Stock-to-stock geçişte yeni API isteği açılmıyor.
- Hover hot path’inde provider veya LLM çağrısı yok. Eşleşme yoksa news alanı gösterilmiyor.

## Cache ve gözlenebilirlik

### Ölçülen sorunlar

Başlangıç üretim fresh-cache ölçümü (10 istek): p50 748 ms, p95 1.691 ms; response yaklaşık 50,4 KB ve `gzip`, ETag, `Cache-Control` yoktu.

Yeni read-only `backend/scripts/benchmark_heatmap_snapshot.py` ile mevcut 194 sembollük snapshot 10 kez ölçüldü:

| Aşama | p50 | p95 |
| --- | ---: | ---: |
| DB snapshot read | 272,91 ms | 1.922,67 ms |
| Dynamic market hierarchy | — | 2,03 ms |
| JSON serialization | — | 1,05 ms |

DB p95 100 ms eşiğini açık biçimde geçtiği için yalnızca bir fresh snapshot tutan process-local memo katmanı eklendi. Unit test, ilk istekte tek DB query yapıldığını ve aynı TTL içindeki ikinci ETag isteğinin DB’ye gitmeden `X-Heatmap-Memo: hit` döndüğünü doğruluyor. Memo fresh TTL dışına veri taşımıyor ve başarılı refresh commit’inden sonra invalidate ediliyor.

### Yeni davranış

- `Server-Timing`: DB, hierarchy, serialization ve total süreler.
- `X-Heatmap-Cache`: `fresh`, `stale`, `refreshing`, `empty`.
- `X-Heatmap-Cache-Age` ve `X-Heatmap-Memo`.
- Snapshot zamanından türetilen ETag; eşleşen istekte 304 ve boş body.
- Fresh response’ta kalan TTL kadar `max-age` ve 60 saniye `stale-while-revalidate`; stale/empty response’ta kısa revalidation.
- FastAPI GZip middleware.
- Empty-cache marker background task kuyruğa alınmadan commit ediliyor.
- PostgreSQL advisory lock refresh’i process’ler arasında deduplike ediyor.
- Yeni snapshot coverage sağlıksızsa payload overwrite edilmiyor; son başarılı payload, hata metadata’sıyla korunuyor.
- Profil sırasında provider error sonrasında her 5 saniyede yeniden refresh başlatan retry fırtınası bulundu. Error yoluna 60 saniyelik backoff eklendi; payload stale kalıyor, `next_refresh_at` backoff sonunu gösteriyor ve React Query tek zamanlayıcıyla bunu izliyor.
- 31 Ağustos HF log incelemesinde tekrar eden advisory-lock dedup mesajlarının kök neden değil, eşzamanlı stale isteklerinin sonucu olduğu doğrulandı. Asıl hata yfinance’ın HF/runtime üzerinde varsayılan SQLite cookie/timezone cache’ini açamamasıydı: history ve `ticker.info` çağrıları `OperationalError: unable to open database file` ile 0/202 coverage üretiyordu.
- yfinance cache’i API trafik kabul etmeden, FastAPI startup aşamasında yazılabilir runtime temp altındaki `coppermind-yfinance-cache` dizinine yönlendirildi. Böylece `live-price` gibi başka bir endpoint varsayılan read-only yolu önce initialize edemiyor. İlk 202-symbol threaded download’dan önce timezone ve cookie SQLite cache’leri seri initialize edilerek lazy-init yarışı ve `database is locked` yolu kaldırıldı.
- Zaten batch indirilen history’nin son iki geçerli kapanışı fiyat ve günlük değişim fallback’i olarak kullanılıyor; bu yol yeni provider isteği açmıyor. `ticker.info` başarısız olsa da sağlıklı chart/history coverage snapshot yayımlayabiliyor.
- Yavaş metadata hot path’i price refresh’inden ayrıldı: metadata 24 saat TTL ile en fazla 12 sembollük incremental batch halinde yenileniyor; kalan leaf’ler son başarılı isim/sector/industry/exchange/weight metadata’sını koruyor. Provider sayaçları (`info_success/error/skipped`, `history_price_fallback`, `missing_price`, `metadata_scheduled`) structured refresh log’una eklendi.

Yeni snapshot market-tree serialization’ı 58.806 byte raw, 11.546 byte gzip oldu. Gerçek HTTP ölçümünde `Content-Encoding: gzip` ve yaklaşık 11.315 byte wire response görüldü. Başlangıçtaki sıkıştırılmamış 50,4 KB wire payload’a göre azalma yaklaşık %77,5; 20 KB hedefinin altında.

Yerel yeni endpoint için beş fresh wall-clock örneği p50 yaklaşık 386 ms, p95 yaklaşık 907 ms verdi. Bu yerel backend + uzak DB ölçümüdür ve üretim baseline’ıyla aynı deployment yolu değildir; deploy gerileme/geçiş kanıtı olarak kullanılmamalıdır.

## Frontend ve rendering

### Ekran kullanımı

- News panel üst dashboard grid’inde kaldı; heatmap bu grid’in dışına taşındı ve tam content width aldı.
- 1536×864’te section 1.216 px, treemap 1.214 px: eski yaklaşık 772 px’e göre %57,3 daha geniş.
- Normal map yüksekliği `clamp(560px, 72vh, 820px)`; fullscreen ve Escape davranışı korundu.
- Zoom=1’de map iç scroll üretmiyor. Video geri bildirimi sonrası görünür zoom butonları kaldırıldı; mouse wheel imleç merkezli zoom, zoomed map üzerinde drag ise pan yapıyor.
- Responsive grid’in auto min-content track’i 390 px görünümde masaüstü treemap genişliğini koruyordu. Grid `minmax(0,1fr)` ve section/scroller `min-width:0` ile düzeltildi.

| Viewport | Treemap genişliği | Treemap yüksekliği | Zoom=1 iç X/Y overflow |
| --- | ---: | ---: | ---: |
| 1536×864 | 1.214 px | 560 px | 0 / 0 |
| 768×864 | 703 px | 560 px | 0 / 0 |
| 390×844 | 342 px | 560 px | 0 / 0 |

390 px’te category panel content viewport genişliğinde bottom sheet oldu. Overview üstündeki mevcut live-price/sentiment şeridi ayrıca 248 px global page overflow üretiyor; kaynak heatmap değil, mevcut `min-width:360px` kart. Heatmap yenilemesi bu pre-existing üst şeridi değiştirmedi.

### Treemap ve LOD

- D3 hierarchy yalnızca data değiştiğinde kuruluyor; resize’da aynı hierarchy üzerinde `treemapResquarify` çalışıyor.
- Stable IDs React key olarak kullanılıyor.
- Eksik tam-ABD evreninde tek bir mega-cap’in alanı aşırı domine etmesini azaltmak için Weight görünümündeki leaf ağırlıkları görünür evren ortalamasına doğru %10 çekiliyor: `w' = 0,9w + 0,1μ`. Toplam ağırlık korunuyor ve herhangi iki araç arasındaki ağırlık farkı matematiksel olarak tam %10 azalıyor. Performance görünümü bu dönüşümden etkilenmiyor.
- Industry içindeki projected-area eşiğinin altındaki en az iki küçük leaf, toplam ağırlığı ve weighted change’i koruyan `+N` hücresinde birleşiyor. Category panel kaynak ağacı kullandığı için bütün hisseler erişilebilir.
- LOD seviyeleri: color-only → ticker → ticker/change → logo/ticker/change. Fiyat heatmap hücresinden kaldırıldı; ayrıntı kartında tek yerde gösteriliyor.
- Ticker ve yüzde fontları sabit Tailwind boyutları yerine hücre eni, boyu ve LOD seviyesinden hesaplanıyor. Büyük hücrede ticker 44 px, değişim 28 px ile sınırlandırılıyor; daha küçük hücrelerde oran ve üst sınırlar kademeli düşüyor. Böylece ticker her seviyede yüzde değerinden daha güçlü kalırken içerik hücreye sığıyor.
- Sürekli, yüksek doygunluklu kırmızı/nötr/yeşil finansal renk skalası, işaretli yüzde metni ve ayrıntılı ARIA label birlikte kullanılıyor. Bu revizyon videodaki Finviz kontrastını CopperMind tasarımına taşırken önceki sönük rose/teal görünümü kaldırdı.
- Sector ve industry header’ları ayrı padding, zemin, sınır ve tipografiyle ayrılıyor.
- Gerçek legacy snapshot’ta aggregation sonrası 120 rendered leaf ve 277 heatmap descendant ölçüldü; kaynak payload 194 araç olarak kaldı. Cached snapshot eski metadata taşıdığı için yerel `market` görünümü fallback branch gösterdi; yeni refresh sonrası dinamik sector/industry dalları backend testleriyle doğrulandı.

### Hover, tooltip ve panel

- Category açılışı 90 ms hover intent, kapanışı 180 ms close delay.
- Panel pointer enter close timer’ı iptal ediyor; pointer geçişinde flicker yok.
- Click/Enter/Space pinliyor, Escape kapatıyor.
- Sarı selection yerine copper outline ve hafif iç vurgu var.
- Aktif category geometry’si stock katmanının altında tutuluyor; vurgu stock hover/focus’u yakalamıyor.
- Industry Peers kartı 380 px genişliğe indirildi. Başlık `Sector - Industry`; varsa aynı industry category ID’siyle eşleşen cached haber/tarih/sentiment; ardından güçlü seçili-hisse satırı ve peers listesi geliyor. Haber yoksa veya stale geniş kategori haberi aktif industry ile eşleşmiyorsa blok hiç yer kaplamıyor.
- Seçili hisse ticker + şirket adı + sparkline + fiyat + yüzde değişimle listenin üzerinde sabit ve yalnızca bir kez gösteriliyor. Aynı hisse peers listesinden stable ID/ticker ile çıkarılıyor.
- Peers aktif hissenin sector/industry metadata’sına göre yerelde filtreleniyor; geniş/fallback cached category altında başka endüstriler karta karışmıyor. Satırlar `Ticker | Sparkline | Price | % Change` hizasında, 40 px ve alternating subtle background kullanıyor.
- 40 satır üzeri listede 40 px fixed-row, 9 görünür satır ve 6 satır overscan ile dependency-free virtualization var. Yalnız liste en fazla 360 px içinde scroll oluyor; seçili hisse görünür kalıyor.
- Panel category değiştiğinde scroll sıfırlanıyor; scroll state rAF ile batch ediliyor.
- Video-referanslı revizyonda category panel pointer koordinatını ref üzerinden izliyor; X/Y konumu `requestAnimationFrame` ve `translate3d` ile React commit üretmeden akıcı güncelleniyor. Aktif stock hücresi sağ/sol açılma tarafını kararlı seçiyor; kart hücrenin dışında kalırken imlecin hücre içindeki yatay ilerlemesini %45 parallax oranıyla takip ediyor. Böylece panel hover’ı çalmadan Finviz-benzeri iki eksenli hareket, edge flip ve viewport clamp birlikte korunuyor.
- Video geri bildirimi sonrası paralel stock tooltip kaldırıldı. Hover edilen stock, industry peer panelinin üst satırına logo/ticker/şirket/fiyat/değişim/sparkline olarak birleşiyor; aynı anda ikinci kart oluşturulmuyor.
- 30 aynı-stock pointer hareketinde React commit sayısı değişmedi. Unit test normal wheel’in doğrudan zoom yaptığını, zoomed map’in drag ile pan edildiğini ve pan sonunda category click oluşmadığını doğruluyor.
- Browser testinde panel içine geçişten 350 ms sonra panel açık kaldı; pin sonrası alan dışına çıkınca kapanmadı ve Escape ile kapandı.
- Son 1280×720 browser doğrulamasında NVDA hücresi sağında kart `x=434,8`, hücre sağ kenarı `x=416,8`; BABA hücresi solunda kart sağ kenarı `x=1138,8`, hücre sol kenarı `x=1156,8` ölçüldü. Her iki tarafta boşluk 18 px, overlap 0 ve kart viewport içinde kaldı.
- İki eksenli follow doğrulamasında aynı NVDA hücresi içinde pointer X=110’dan X=300’e ilerlediğinde panel X=432,19’dan X=517,69’a taşındı; 190 px pointer hareketi 85,5 px yatay panel hareketi üretti ve kart hücrenin dışında kaldı.
- Aynı-industry NVDA → AVGO geçişinde seçili satır 120 ms kontrol noktasında günceldi, seçili ticker DOM’da bir kez bulundu ve `/market-heatmap/context` request sayısı artmadı. Stale geniş category altında NVDA kartı yalnız QCOM/AVGO/AMD/INTC/TXN peer’larını gösterdi; alakasız broad-category haberini göstermedi.
- Stock-news browser kontrolünde FCX için cache’teki GuruFocus haberi, tarih ve sentiment gösterildi. Aynı category içinde NVDA’ya 120 ms geçişte `No recent news is available for NVDA.` mesajı göründü ve context request sayısı 1’de kaldı. 152 hisselik fallback category response’unda 24 ticker haberi toplu eşleşti; refactor sonrası tek yerel context örneği 622,51 ms sürdü (production latency kanıtı değildir).
- Wheel testi content genişliğini 1.199 px’den 1.487 px’e çıkardı ve cursor anchor’ını scroll offset ile korudu; drag testi scroll offset’i `(188,90)` → `(288,134)` taşıdı.
- Ticker double-click yeni sekmede quote detayını açıyor. Industry peer sparkline batch penceresi Finviz referansıyla uyumlu olarak 3 ay kullanıyor; eşit aralıklı normalize nokta sayısı sabit kaldığı için payload büyümedi.
- Zoomed map pointer-down’da native selection/drag engelleniyor; map ve logo katmanlarında `user-select:none`, logo görsellerinde `draggable=false` uygulanıyor. Browser drag testinde pan offset’i değişirken selection metni boş kaldı.
- Legacy/stale cache içindeki category ID ile render ID ayrışırsa peer listesi stabil ad üzerinden ilk eşleşmeye güvenli fallback yapıyor. Browser kontrolünde Semiconductors paneli yeniden 6 peer gösterdi.
- Category-level hover başlığı ilk leaf’in `group/subgroup` değerinden türetilmiyor; doğrudan hover edilen node adını kullanıyor. Gerçek browser kontrolünde `Basic Materials` hover kartı `Basic Materials` başlığıyla açıldı; ilk stock metadata’sı başlığa sızmadı. Stock hover’da `Sector - Industry` hiyerarşisi korunuyor.

## Logo.dev entegrasyonu

- Runtime secret eklenmedi. Publishable browser token yalnız `VITE_LOGO_DEV_PUBLISHABLE_KEY` ile okunuyor ve `env.example` içinde boş bırakılıyor.
- Cell, tooltip ve category row aynı ticker için sabit 128 px normalize URL kullanıyor; browser/CDN cache tek asset’i paylaşabiliyor.
- Yalnız LOD eşiğini geçen hücreler logo component’i oluşturuyor. IntersectionObserver, 48 px root margin, `loading="lazy"` ve `decoding="async"` kullanılıyor.
- Başarısız URL session-level set’e alınarak tekrar denenmiyor; broken image yerine ticker initials gösteriliyor.
- Eski/stale snapshot `logoTicker` taşımıyorsa equity/ETF/mutual-fund hücreleri normalize ticker’ı fallback olarak kullanıyor; future/currency gibi uygun olmayan araçlar için gereksiz logo isteği başlatılmıyor.
- `instrumentType` ve `logoTicker` enrichment artık yalnız Market dalında değil hierarchy oluşturulmadan önce ortak uygulanıyor. Yerel API doğrulamasında hem Market hem Themes response’u 194/194 `instrumentType`, logo-uygun 175/194 `logoTicker` taşıdı.
- Token yokken hiçbir Logo.dev request’i başlatılmıyor; mevcut test profilinde 0 logo request ölçüldü.
- Footer’da görünür “Logos provided by Logo.dev” linki var; attribution linkinde `noreferrer` kullanılmıyor.
- Logo self-host edilmiyor ve endorsement ima edilmiyor. Uygulama notları resmi [Stock Logo API](https://www.logo.dev/products/stock-ticker-logos), [attribution](https://www.logo.dev/docs/platform/attribution), [caching](https://www.logo.dev/docs/platform/caching), [fair-use](https://www.logo.dev/docs/platform/fair-use) ve [terms](https://www.logo.dev/legal/terms) belgelerine bağlı.

Logo.dev publishable token kullanıcı tarafından Vercel Production config’ine ve Git tarafından ignore edilen yerel `frontend/.env.local` dosyasına eklendi. Yerel browser kontrolünde LOD-eligible Market hücrelerinde 16 Logo.dev görseli / 25 logo holder; Themes görünümünde 10 görsel / 15 holder oluştu. Sağlayıcıdan logo dönmeyen holder’lar broken image yerine initials fallback gösterdi. Kod henüz deploy edilmediği için canlı CDN cache-hit ve production fallback oranı için **Ölçüm bulunamadı**.

Vite env değerlerini process başlangıcında okuduğu için `.env.local` değişikliğinden sonra Vite süreci yeniden başlatıldı; servis edilen modülün configured key’i içerdiği değer açığa çıkarılmadan doğrulandı.

## Performans sonuçları

| Ölçüt | Başlangıç | Yenilenen sürüm | Sonuç |
| --- | ---: | ---: | --- |
| Desktop treemap width | ~772 px | 1.214 px | +%57,3 |
| Heatmap DOM descendants | 286 | 277 | Daha zengin panel/tooltip’e rağmen azaldı |
| HTTP wire payload | ~50,4 KB, identity | ~11,3 KB gzip | ~%77,5 azalma |
| Read-only DB p50 / p95 | Ölçüm yok | 272,91 / 1.922,67 ms | Memo eşiği aşıldı |
| Backend hierarchy p95 | Ölçüm yok | 2,03 ms | 10 gerçek snapshot koşusu |
| Backend serialize p95 | Ölçüm yok | 1,05 ms | 10 gerçek snapshot koşusu |
| Frontend layout p95, 194 | Ölçüm yok | 2,75 ms | Son tam frontend koşusu; ≤8 ms hedefi geçti |
| Frontend layout p95, 1.000 | Ölçüm yok | 4,96 ms | Son tam frontend koşusu; ≤12 ms hedefi geçti |
| NVDA hücre genişliği, aynı viewport | 382 px | 345 px | Ağırlık-spread sıkıştırması sonrası −%9,7 |
| NVDA ticker / değişim fontu | 16 / 12 px | 44 / 28 px | Finviz-esintili responsive hierarchy |
| İlk open heatmap request | Gözlenebilir değildi | 1 | Hedef geçti |
| Warm SPA dönüşü request count | Gözlenebilir değildi | 1’de kaldı | Full-body tekrar yok |
| Aynı-stock 30 pointer move | Önceki rAF regresyon adayı | 0 ek React commit | Hedef geçti |
| Browser layout p95 | Ölçüm yok | 0,30 ms, 14 sample | Gerçek cached payload |
| Heatmap kaynaklı >50 ms long task | Ölçüm yok | Ölçüm bulunamadı | Page-scope observer max 70 ms; kaynak heatmap’e atanamadı |
| Etkileşim frame p95/FPS | Ölçüm yok | Ölçüm bulunamadı | Karşılaştırılabilir frame trace yok |
| İlk görünür heatmap süresi | Baseline yok | Ölçüm bulunamadı | %20 iyileşme iddiası yapılmadı |
| Deploy fresh p50/p95 | 748 / 1.691 ms | Ölçüm bulunamadı | Kod henüz deploy edilmedi |

DOM ölçeği ve 1.000 araç layout p95 hedefleri geçtiği için Web Worker/canvas eklenmedi. Erişilebilir DOM ve logo kullanımını koruyan pruning/virtualization yeterli kaldı.

31 Ağustos gerçek provider tanısında yazılamayan yfinance cache ile history 0/202 ve `ticker.info` 0/202 kaldı. Cache yönlendirmesi ve seri initialization sonrası aynı 202-symbol history batch’i 194 geçerli sembol üretti; yalnız sekiz provider tarafından veri bulunamayan/delist edilmiş sembol dışarıda kaldı. Eski tüm-symbol metadata yolu yaklaşık 177 saniyede tamamlandığı için metadata 24 saatlik, en fazla 12 sembollük incremental cadence’e ayrıldı. Başarılı snapshot 194 leaf ile yayımlandı; `refresh_error=null`, cache state `fresh` ve `Last refresh failed` durumu temizlendi. Bu doğrulama yerel yeni kodun paylaşılan DB cache’ine yaptığı provider refresh’tir; HF runtime kod deploy’u sonrası aynı structured sayaçlarla ayrıca izlenecektir.

## Test kapsamı ve doğrulama

Backend heatmap/API testleri:

- Dynamic hierarchy, stable ID, missing sector/industry ve non-equity branch.
- Sparkline normalization ve history provider fallback.
- Coverage guard ve last-good snapshot koruması.
- Fresh, empty, stale, refresh-error/backoff cache yolları.
- Empty marker ordering, process memo hit, DB request dedup, PostgreSQL advisory-lock dedup.
- ETag/304, gzip, cache ve timing header’ları.
- Cached category ID doğrulaması ve ticker/company news eşleşmesi.
- Category response içinde per-stock news eşleşmesi ve eşleşmeyen ticker’ın map dışında kalması.

Frontend testleri:

- Layout determinism, aggregate weight, LOD ve breadth.
- Toplam ağırlığı koruyan tam %10 pairwise weight-gap sıkıştırması.
- Büyük/küçük hücrelerde bounded responsive ticker/change tipografi oranları.
- 194/1.000 araç p95 bütçesi.
- Tooltip/panel viewport clamp.
- 380 px compact panel, selected-stock dedup, long company name, missing sparkline/news ve industry-scoped peer filtering.
- Gerçek stock-news seçimi, broad-category news izolasyonu ve açık no-news durumu.
- 40 satır üstü listede sabit selected stock + 360 px virtualized peer list.
- Delegated hover, keyboard pinning, hover intent/close delay ve panel geçişi.
- Pointer-follow panel, direct wheel zoom, cursor anchor ve drag-pan.
- Ticker double-click detail navigation.
- Logo lazy loading, normalize URL ve initials fallback.
- Mobil/tablet width ve map iç overflow browser doğrulaması.

Son kapılar (2026-08-31 follow-up dahil):

- Backend heatmap: **15 passed** (`test_heatmap.py`). GitHub workflow ile aynı offline seçiminde yerel backend paketi: **534 passed, 15 skipped**. Sabit 29 Ağustos fixture’ı nedeniyle zamanla 48 saatlik news-stats penceresinin dışına çıkan regression testi, çalışma anına bağlı kararlı `as_of` kullanacak şekilde düzeltildi.
- Frontend: **21 passed**.
- ESLint: warning/error yok.
- TypeScript + Vite production build: geçti.
- Heatmap lazy chunk: 33,92 KB raw / 12,28 KB gzip.
- Mevcut ana bundle >500 KB Vite warning’i devam ediyor; heatmap zaten ayrı lazy chunk olduğundan bu çalışmanın runtime regresyonu değil.

## Deployment sonrası tekrar ölçülmesi gerekenler

1. Aynı production bölgesinde 10+ fresh ve warm istekle p50/p95, `Server-Timing`, memo hit ve 304 oranı.
2. Performance panel/trace ile first-visible heatmap, interaction frame p95, FPS ve long-task attribution.
3. Gerçek Logo.dev publishable token ile LOD-eligible request sayısı, CDN/browser cache hit ve fallback oranı.
4. HF deploy sonrası beş normal provider refresh’te history/quotes stage dağılımı ile incremental metadata sayaçları.
5. Deployed `view=market` ağacında gerçek sector/industry sayıları, placeholder fallback sayısı ve category news hit oranı.
