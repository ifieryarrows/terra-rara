# CopperMind Performance Audit ve Regression Watch Değerlendirmesi

| Alan | Değer |
| --- | --- |
| Tarih | 15 Eylül 2026 |
| Kapsam | Landing rendering, production consensus endpoint, embedding backfill Stage 3.5 |
| Karşılaştırma | `388a7e4` çalışma ağacı kontrolü ve `cbffd61` son remote aday build'i |
| Durum | Ölçüldü; görsel sözleşme korunarak düşük riskli optimizasyonlar uygulandı |

## Önceliklendirilmiş sonuç

1. **Production consensus latency — en yüksek doğrulanmış risk.** İlk gözlenen istek 28,601 s sürdü. Ardışık 10 istekte tüm yanıtlar `200`, `x-vercel-cache: MISS` ve `p50 6,575 s`, `p95 6,809 s` oldu. Bu akış snapshot-first olacak şekilde düzeltildi; production yeniden ölçümü deploy sonrasına bırakıldı.
2. **Landing particle yoğunluğu — yerel regression uyarısı.** `cbffd61`, 4× CPU senaryolarında aynı makinedeki `388a7e4` kontrolüne göre FPS'i düşürdü ve p95 frame süresini artırdı. Görsel risk almamak için particle bütçesi değiştirilmedi; yalnızca aynı render çıktısını koruyan mikro optimizasyonlar uygulandı.
3. **Embedding backfill SQL maliyeti — doğrulanmış sorgu fazlalığı.** Ön filtre eksik kayıtları doğru seçiyordu; sonrasında satır başına varlık kontrolü yapılıyordu. Akış toplu, unique-conflict-safe upsert kullanacak şekilde düzeltildi.

Bu sonuçlar production RUM veya gerçek mobil/GPU cihaz ölçümü değildir. Görsel davranışta değişiklik yapılmadı; backend optimizasyonları response verisini ve mevcut freshness/degraded kurallarını korur.

## 1. Landing runtime profili

Her senaryo üç kez, aynı makinede local production build + Vite preview + Edge headless ile çalıştırıldı. Tablodaki değerler medyandır. `cbffd61` build'i particle bütçesini high kalite için `420 → 640`, balanced kalite için `280 → 420` olarak artırıyor.

| Senaryo | FPS kontrol | FPS `cbffd61` | FPS değişimi | p95 frame kontrol | p95 frame `cbffd61` | Yorum |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Desktop 1536×900, 1× CPU | 49,2 | 51,2 | +4,1% | 33,6 ms | 33,4 ms | Anlamlı regression yok |
| Desktop 1280×720, 4× CPU | 63,2 | 47,8 | **−24,4%** | 25,1 ms | **33,4 ms** | Regression uyarısı |
| Mobile 390×844, 4× CPU | 77,5 | 61,3 | **−20,9%** | 16,8 ms | **25,1 ms** | Regression uyarısı |

Her iki build de başarılı oldu; `cbffd61` latest profilde ölçülen koşullarda long task gözlenmedi. CPU throttling uygulandı, fakat kesin GPU throttling ve gerçek cihaz ölçümü yapılamadı. Bu nedenle sonuç, remote aday build için **yerel ve tekrarlanabilir bir performans sinyali** olarak değerlendirilmeli; production regression iddiası olarak sunulmamalıdır.

### Görsel sözleşme korunarak yapılan landing optimizasyonu

Particle sayısı, kalite eşikleri, shader formülleri, renkler, pozisyonlar, timeline ve fallback davranışı değiştirilmedi. WebGL renderer'da kurulumda zaten bağlı olan programın her frame yeniden bağlanması kaldırıldı; Canvas2D fallback'te frame dışı sabit hesaplar loop dışına taşındı.

Post-change profil medyanları aşağıdaki gibidir. Bu tek makine laboratuvar koşusunda FPS farkı gürültülü olduğu için net bir performans kazanımı iddia edilmemiştir; amaç aynı çıktıyı koruyarak gereksiz work'u kaldırmaktır.

| Senaryo | FPS post-change | p95 frame | Aktif particle | Lifecycle |
| --- | ---: | ---: | ---: | --- |
| Desktop 1536×900, 1× CPU | 55,6 | 33,4 ms | 420 | Canvas `0`, pointer draw `0` |
| Desktop 1280×720, 4× CPU | 73,0 | 25,0 ms | 420 | Canvas `0`, pointer draw `0` |
| Mobile 390×844, 4× CPU | 56,1 | 25,1 ms | 280 | Canvas `0`, pointer draw `0` |

Görsel/lifecycle QA sonucu: **7/7 landing viewport, 3/3 adaptive fallback, 6/6 workspace route**, overflow `0`, landing API isteği `0`, page error `0`. Ekran görüntüleri ve sonuçlar `experience-qa/` altında saklandı.

Ham profiller:

- [`landing-profile-388a7e4.json`](evidence/landing-performance-20260915/landing-profile-388a7e4.json)
- [`landing-profile-cbffd61.json`](evidence/landing-performance-20260915/landing-profile-cbffd61.json)
- [`landing-profile-optimized-20260915.json`](evidence/landing-performance-20260915/landing-profile-optimized-20260915.json)
- [`experience-qa/results.json`](evidence/landing-performance-20260915/experience-qa/results.json)

## 2. Production consensus endpoint

Ölçüm, `https://terra-rara.vercel.app/api/analysis/consensus?symbol=HG%3DF` adresine GET isteğiyle yapıldı. Yanıt gövdeleri saklanmadı; yalnızca durum, süre ve cache başlıkları kaydedildi.

| Ölçüm | Sonuç |
| --- | ---: |
| `/api/health` | `200`, 3,466 s |
| İlk gözlenen consensus isteği | `200`, 28,601 s |
| Ardışık warm sequence | 10/10 `200` |
| Warm sequence p50 | 6,575 s |
| Warm sequence p95 | 6,809 s |
| Warm sequence ortalama | 6,707 s |
| Cache | 10/10 `x-vercel-cache: MISS` |

İlk 28,601 saniyelik çağrının cold start olup olmadığı bu ölçümle kesinleştirilemedi. Ancak warm sequence boyunca her isteğin origin'e gittiği ve yaklaşık 6–8 saniyelik latency ürettiği görüldü. Bu ölçüm optimizasyon öncesi production baseline'dır; yeni kodda `/api/analysis/consensus`, TFT endpoint'inin snapshot-first sözleşmesini çağırır ve yalnızca snapshot geçersiz/eksik olduğunda mevcut live fallback'e iner.

Production deploy sonrasında aynı 10× seri yeniden çalıştırılmalı; geçerli snapshot durumunda inference çağrısının yapılmadığı ve latency'nin DB snapshot okumasına indiği doğrulanmalıdır. `available=false`/degraded freshness davranışı değiştirilmemelidir.

## 3. Embedding backfill Stage 3.5

Production verisine yazmadan, in-memory SQLite üzerinde sentetik News kayıtlarıyla ölçüm yapıldı. Embedding extractor ve PCA yolu monkeypatch edilerek FinBERT model indirme/çalıştırma maliyeti dışarıda bırakıldı; bu test yalnızca sorgu ve persistence desenini ölçer.

| Eksik News satırı | Önce süre / SQL | Sonra süre / SQL | Embedded |
| ---: | ---: | ---: | ---: |
| 250 | 0,2534 s / 501 | 0,0556 s / 1 `SELECT` + 2 batch `INSERT` | 250 |
| 1.000 | 0,8035 s / 2.001 | 0,1146 s / 1 `SELECT` + 5 batch `INSERT` | 1.000 |

Kodun ilk outer join filtresi eksik embedding'leri seçiyor. Yeni persistence akışında:

- 1 başlangıç seçimi,
- 200 satırlık batch başına 1 adet conflict-safe insert,
- `news_processed_id` unique conflict durumunda database-level skip

çalışıyor. Sentetik SQLite ölçümünde backfill sırasında 250 satır için `1 + 2`, 1.000 satır için `1 + 5` ifade gözlendi. Gerçek Postgres latency'si ve FinBERT CPU süresi için **Ölçüm bulunamadı**.

`missing-row` davranışı ve duplicate koruması regression testleriyle sabitlendi.

## Uygulanan değişiklikler

- `/api/analysis/consensus`, geçerli persisted TFT snapshot'ını yeniden hesaplama yapmadan kullanıyor; yeni bar/model freshness kontrolleri ve live fallback korunuyor.
- Embedding backfill, PostgreSQL/SQLite için `ON CONFLICT DO NOTHING` kullanan 200 satırlık batch persistence'a geçti; diğer dialect'lerde batch başına tek existence sorgusu fallback olarak tutuldu.
- Landing WebGL/Canvas2D yolu aynı görsel parametrelerle çalışıyor; yalnızca redundant state bind ve loop içindeki sabit hesap kaldırıldı.
- QA harness, başlangıçta gizli olan signal layer'ı DOM'dan, aktif renderer canvas'ını `data-renderer` üzerinden ve MotionValue yüzeylerini stabil hale geldikten sonra doğruluyor.

## Doğrulama

- Backend: **541 passed, 15 skipped**
- Frontend: **49 passed**
- Frontend lint: geçti
- Production build: geçti
- Bundle budgets: geçti; initial JS gzip `120.773 / 190.000`, CSS gzip `13.327 / 14.000`
- Landing/workspace QA: `7 + 3 + 6` senaryo geçti
- `git diff --check`: geçti

## Sınırlar ve sonraki sıra

- Production consensus ölçümü gerçek endpoint davranışını gösterir; ancak tek sembol ve tek seri ölçümdür, kullanıcı trafiği/RUM değildir.
- Landing profili headless Edge + CPU throttling'dir; fiziksel cihaz, gerçek GPU ve field RUM için **Ölçüm bulunamadı**.
- Embedding ölçümü DB sorgu desenini kanıtlar; gerçek model inference süresi için **Ölçüm bulunamadı**.
- Production consensus için yeni ölçüm deploy sonrasında yapılmalı. Particle sayısını azaltan veya görsel kompozisyonu değiştiren tuning bu kapsamda uygulanmadı; mevcut renderer profili net kazanç göstermediği için bu alanı ölçümle izlemek gerekir.
