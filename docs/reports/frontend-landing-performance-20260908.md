# CopperMind Landing Performance Profili

| Alan | Değer |
| --- | --- |
| Tarih | 8 Eylül 2026 |
| Baseline | `b09248e` (`feat(frontend): add continuous cinematic landing scene`) |
| Kapsam | Terra Rara landing cinematic scene ve landing → dashboard yaşam döngüsü |
| Durum | Ölçüldü, optimize edildi ve aynı sahne akışıyla yeniden ölçüldü |

## Sonuç

Mevcut native vertical scroll, `position: sticky` pinned world, kesintisiz `0–1` MotionValue timeline, scroll-driven yatay yüzey hareketleri, particle morph, global atmosphere, background typography ve scrubbed text reveal korunmuştur. Slide/chapter index’e yuvarlama eklenmemiştir.

Üç tekrarlı laboratuvar profilinin medyanına göre:

| Senaryo | FPS önce | FPS sonra | p95 frame önce | p95 frame sonra | Main thread / frame önce | Main thread / frame sonra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Desktop 1536×900, 1× CPU | 6,5 | 56,6 | 233,4 ms | 33,3 ms | 6,41 ms | 2,92 ms |
| Desktop 1280×720, 4× CPU | 10,4 | 62,3 | 125,0 ms | 25,1 ms | 25,05 ms | 14,67 ms |
| Mobile 390×844, 4× CPU | static fallback | 71,1 cinematic | karşılaştırılamaz | 24,9 ms | karşılaştırılamaz | 13,59 ms |

Mobil baseline particle/cinematic sahneyi hiç çalıştırmadığı için mobil FPS değerleri before/after performans kazancı olarak yorumlanmamıştır. Final mobil ölçüm, istenen görsel kompozisyonun 280 parçacık ve DPR 1 ile çalıştığı yeni davranışı belgeler.

## Yöntem

- Local production build ve Vite preview kullanıldı.
- Microsoft Edge headless, cache kapalı, 4.200 ms native document scroll kullanıldı.
- CDP `Performance`, `Tracing`, `LayerTree` ve `SystemInfo` ile main-thread, layout, paint, composite, raster ve GPU özellikleri toplandı.
- Her desktop, 4× CPU desktop ve 4× CPU mobile senaryosu üç kez çalıştırıldı; tabloda medyan kullanıldı.
- React DevTools hook ile scroll sırasındaki commit sayısı; prototype instrumentation ile Canvas2D/WebGL çağrıları ölçüldü.
- Görsel sözleşme 7 viewport, 5 timeline örneği, reduced-motion, Save-Data, düşük CPU ve WebGL2-yok fallback’lerinde Playwright ile doğrulandı.
- Pipeline darboğazı kontrollü CSS ablation ile ayrıca ölçüldü.

Ham kanıtlar:

- `docs/reports/evidence/landing-performance-20260908/baseline.json`
- `docs/reports/evidence/landing-performance-20260908/optimized-final.json`
- `docs/reports/evidence/landing-performance-20260908/pipeline-ablation-final.json`
- `docs/reports/evidence/landing-performance-20260908/lighthouse-final-all.json`
- `docs/reports/evidence/landing-performance-20260908/visual-baseline/`
- `docs/reports/evidence/landing-performance-20260908/visual-final/results.json`

## Darboğaz teşhisi

Baseline scroll sırasında React commit sayısı `0`, trace long task sayısı `0` ve desktop main-thread toplamı yalnızca yaklaşık 179 ms idi. Buna rağmen 1536×900 ölçüm 6,5 FPS kaldı. Viewport alanı büyüdükçe düşen hız ve kontrollü ablation testi sorunun React state veya scroll listener değil, fill-rate/raster overdraw olduğunu gösterdi.

Belirleyici A/B sonucu:

| Aynı final build üzerinde katman | FPS | Ortalama frame | p95 frame |
| --- | ---: | ---: | ---: |
| Optimize edilmiş statik organik grain | 47,7 | 21,0 ms | 49,9 ms |
| Eski `feTurbulence`, `inset:-50%`, soft-light grain geri enjekte edildi | 9,4 | 106,3 ms | 166,6 ms |

Eski grain katmanı viewport’un yaklaşık dört katı yüzeyde runtime SVG noise ve blend çalıştırıyordu. En büyük darboğaz bu katmandı. Particle alanını tek başına kapatmak önceki ara ablation’da çok küçük kazanç vermişti; bu nedenle particle yoğunluğunu rastgele düşürmek yerine önce gerçek overdraw kaynağı giderildi.

## Rendering pipeline karşılaştırması

Maliyetler toplam yerine render edilen frame başına normalize edilmiştir; final build çok daha fazla frame ürettiği için ham toplam süre tek başına yanıltıcıdır.

| Desktop 1536×900 | Önce | Sonra |
| --- | ---: | ---: |
| Style / frame | 0,43 ms | 0,30 ms |
| Layout / frame | 0,29 ms | 0,11 ms |
| Paint / frame | 0,76 ms | 0,36 ms |
| Raster / frame | 6,54 ms | 1,64 ms |
| Composite / frame | 0,47 ms | 0,24 ms |
| React scroll commit | 0 | 0 |
| DOM node | 376 | 375 |

4× CPU desktop’ta initial long task medyanı `7 → 5`, toplam initial long-task süresi `900 → 587 ms` oldu. Scroll trace long-task medyanı her iki build’de de `0` kaldı. Final 4× CPU koşularından birinde 4 trace long task outlier’ı görüldü; diğer iki koşuda `0` idi ve ham veri saklandı.

## Uygulanan optimizasyonlar

### Particle ve shader

- Canvas2D’de frame başına yaklaşık 420 ayrı shadow-blurred arc çizen yol, WebGL2 `GL_POINTS` renderer’a taşındı.
- Şekiller module-level `Float32Array` keyframe’lerinde tutuluyor; frame içinde object/array allocation yapılmıyor.
- İki pozisyon buffer’ı yeniden kullanılıyor ve yalnızca keyframe segmenti değiştiğinde `bufferSubData` çağrılıyor.
- Final scroll medyanında 238 frame için 237 `drawArrays`, 12 `bufferSubData`, 0 `drawElements`, 0 texture upload ölçüldü.
- Triangle count `0`; particle field nokta primitive kullanıyor. Texture count `0`.
- Shader’da texture sampling, noise fonksiyonu ve runtime branch bulunmuyor. Shader compile/link ilk kurulumda bir kez çalışıyor; desktop ölçümlerde warm compile yaklaşık 2,5 ms, cold ilk koşu 21,7 ms idi.
- High kalite GPU buffer bütçesi yaklaşık 11.760 byte; balanced kalite yaklaşık 7.840 byte’tır.
- WebGL2 yoksa aynı timeline 280 parçacıklı, DPR 1, typed-array Canvas2D fallback ile devam ediyor.

### Atmosphere, visibility ve scroll

- Görsel grain kaldırılmadı. Runtime `feTurbulence` yerine bir kez rasterize edilen düzensiz, düşük opaklıklı SVG micro-noise tile kullanıldı.
- Görünmez dashboard surface, background word ve hero signal katmanlarına progress bağlı `visibility` gating eklendi.
- Pointer koordinatları React handler yerine passive native listener ile alınıp `requestAnimationFrame` içinde tek style write olarak gruplanıyor.
- Mevcut `useScroll` MotionValue merkezi timeline’ı korundu. Scroll değeri React state’e yazılmıyor.
- Canvas resize ölçümü `ResizeObserver`, görünürlük `IntersectionObserver`, tab visibility ve route unmount ile sınırlandı.

### Adaptive quality ve yaşam döngüsü

- Desktop: 420 particle, DPR en fazla 1,5.
- Mobile/tablet: aynı cinematic composition, 280 particle, DPR en fazla 1; grain, shadow, backdrop ve atmosphere blur kontrollü azaltılıyor.
- `prefers-reduced-motion`, Save-Data veya en fazla 2 logical CPU: tam okunabilir static fallback ve particle loop yok.
- WebGL2 yoksa balanced Canvas2D fallback doğrulandı.
- Unmount sırasında rAF, MotionValue subscription, observer, listener, WebGL buffer/program ve context temizleniyor.
- Landing’den SPA ile `/dashboard` geçişinden sonra canvas sayısı `0`; pointer sonrasında Canvas/WebGL draw artışı `0` ölçüldü.

## Network ve bundle

| Kaynak | Önce | Sonra |
| --- | ---: | ---: |
| Kritik entry JS transfer | 121.037 byte | 118.121 byte |
| Kritik entry CSS transfer | 14.255 byte | 12.463 byte |
| Lazy cinematic JS | entry içindeydi | 6.702 byte |
| Lazy cinematic CSS | entry içindeydi | 3.803 byte |
| Landing toplam transfer | 206.494 byte | 212.291 byte |

Landing toplam transferi renderer ve adaptive fallback kodu nedeniyle 5.797 byte arttı. Buna karşılık kritik entry küçüldü ve cinematic JS/CSS capability policy sonrasında lazy yükleniyor. Direct workspace route QA’sı `/assets/CinematicLanding-*` isteği yapılmadığını doğrular; dashboard landing particle/render loop’u başlatmıyor.

Build budget sonucu:

- Initial JS gzip: 117.821 byte / 190.000 limit.
- Initial CSS gzip: 12.163 byte / 14.000 limit.
- Dashboard-before-news-and-heatmap JS gzip: 263.903 byte / 280.000 limit.

## Lighthouse ve erişilebilirlik

Final local Lighthouse 12.8.2 sonucu:

| Kategori/metrik | Sonuç |
| --- | ---: |
| Performance | 66 |
| Accessibility | 100 |
| Best Practices | 79 |
| SEO | 91 |
| FCP | 3,3 s |
| LCP | 4,1 s |
| TBT | 460 ms |
| CLS | 0,094 |

Bu Lighthouse koşusu HTTP local preview, production compression olmadan ve local ortamda 404 dönen Vercel Speed Insights script’iyle çalıştı. Bu nedenle production skoru olarak sunulmamıştır. Karşılaştırılabilir baseline Lighthouse ölçümü bulunamadı; Lighthouse için before/after kazanç iddiası yapılmamıştır.

Playwright final doğrulaması:

- 7/7 landing viewport geçti: 1536×900, 1280×600, 1024×650, 768×800, 320×650, 390×844 ve reduced-motion.
- Her cinematic viewport’ta 5 scroll örneğinde sticky top `0`, doğru surface/background word, tek canvas ve yatay transform akışı doğrulandı.
- 3/3 adaptive fallback geçti: Save-Data, düşük CPU, WebGL2-yok.
- 6/6 workspace fixture route geçti; desktop/mobile overflow `0`.
- Landing hiçbir finansal API isteği yapmadı.
- `Cu / 29`, `naive scroll continous signal` ve `native scroll / continuous signal` görünür body text içinde bulunmadı.
- Mevcut Geist Sans fontu korundu; League Gothic eklenmedi.
- Keyboard route navigation, focus target ve reduced-motion davranışı 49/49 unit/integration test ile korundu.

## Ölçüm sınırları

- Gerçek telefon/tablet donanımı ve field RUM: **Ölçüm bulunamadı**. Mobil sonuç Edge emulation + 4× CPU throttling laboratuvar ölçümüdür.
- Kesin GPU memory ve texture memory: **Ölçüm bulunamadı**. CDP GPU process `privateMemory` alanı `null` döndü. Uygulama seviyesinde particle texture çağrısı `0` ve buffer byte bütçesi yukarıda verilmiştir.
- Kesin overdraw pixel count: **Ölçüm bulunamadı**. Darboğaz kontrollü layer ablation ve raster/frame metriğiyle doğrulanmıştır.
- Stabil LayerTree memory/layer count: **Ölçüm bulunamadı**; CDP bu koşulda 0–4 arasında kararsız event verdi. Paint, raster ve composite süreleri trace üzerinden ölçülmüştür.
- Gerçek GPU throttling: **Ölçüm bulunamadı**. CDP ile CPU throttling uygulandı; GPU feature durumu ve renderer bilgisi kaydedildi.
