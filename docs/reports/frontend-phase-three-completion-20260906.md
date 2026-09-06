# Faz 3 — Dashboard teslimi ve yerel kabul

Tarih: 2026-09-06. Kapsam: [3. faz planı](../implementation/frontend-experience-phase-three.md). Önceki teslim: [ilk dilim](./frontend-phase-three-first-slice-20260906.md).

## Karar

3. fazın dashboard geliştirme ve yerel kabul kapsamı tamamlandı. İlk dilimdeki Models, Validation, News ve System işleri üzerine grafik kontrolleri, Overview veri durumları, mobil haber kullanımı ve heatmap kontrolleri tamamlandı. Landing kompozisyonu bu turda değiştirilmedi. Commit, yayın ve gerçek kullanıcı performans kabulü yapılmadı.

## Tamamlanan kullanıcı akışları

| Alan | Teslim | Kanıt |
| --- | --- | --- |
| Fiyat grafiği | 30/90/180 gözlenen kapanış seçimi; medyan ve Q10–Q90 kontrolleri; dolgu belirsizlik bandı; son kapanış tarihi; USD tooltip | `features/forecast/PriceForecastChart.tsx`, `chart-data.ts`; dönem seçimi API isteği üretmiyor |
| Grafik erişimi | Klavye oklarıyla tooltip; mobil/klavye için açılır semantik tablo; tablo yalnızca açıkken mount; kendi scroll alanında en fazla 180 kapanış + 5 tahmin | DOM testi ve 5 viewport browser senaryosu |
| Tahmin sözleşmesi | Dönem değişince baz tarih/fiyat değişmiyor; eski tarihli ve degraded tahmin grafiğe bağlanmıyor; eksik fiyat sıfır yapılmıyor; geçersiz quantile çifti doldurulmuyor | 7 yeni grafik/metrik testi; stale/degraded/empty browser durumları |
| Overview | Bölüm bağlantıları; manuel yenileme; eşzamanlı yenileme koruması; arka plan isteği başarısızsa önceki değerler uyarıyla görünür | Normal ve refresh-failure browser senaryoları |
| Metrik hiyerarşisi | Haftalık doğruluk ve günlük Sharpe açıkça ayrıldı; eksik Sharpe'dan sıfır/HEALTHY üretilmiyor; T+1 tanılaması açılır bölümde | `ModelReliability.tsx`; haftalık eşikler korunuyor |
| Haberler | Filtreler ve liste tek sınırlı scroll alanında; dar ekranlarda ulaşılabilir; veri yokken sayaçlar sıfırmış gibi görünmüyor | Arama → boş sonuç → reset → detay aç → Escape → focus/scroll geri yükleme |
| Heatmap | Ortak seçim alanları; kategori/boyut reset; zoom in/out/reset; mobilde ulaşılabilir kontroller | Kategori, boyut, reset ve zoom browser senaryoları |
| Heatmap tam ekran | Portal ile üst katman; viewport'a uyan esnek harita; klavye odağını içeride tutma; Escape/çıkışta focus ve body scroll geri yükleme | 320×650 dahil 5 viewport'ta kontrol edildi |
| Yerel build/dev | Prerender'ın optimizer cache'i çalışan dev cache'inden ayrıldı | Dev açıkken build alındı; ardından 10 dashboard senaryosu geçti |

Grafik state'inin Overview'dan ayrılması, dönem/seri/tablo etkileşimlerinde tüm dashboard'un yeniden render edilmesini önler. Bu mimari değişiklik tek başına genel hızlanma iddiası değildir; ölçüm aşağıdadır.

## Bulunan yerel çalışma sorunu

`scripts/prerender.mjs`, production HTML üretirken dev sunucusundan farklı `optimizeDeps` ayarlarıyla ikinci bir Vite instance'ı açıyordu. Ortak `.vite` cache'i nedeniyle çalışan dev sunucusunda yeni route/dependency yüklemesi bozulabiliyordu. `cacheDir: 'node_modules/.vite-prerender'` ile ayrıldı. Dosya silme veya GitHub sürümüne dönme kullanılmadı.

## Kabul sonuçları

| Kontrol | Sonuç |
| --- | --- |
| Vitest | 8 dosyada **49 test geçti** |
| Heatmap yerleşim benchmark'ı | Son testte p95 gerçek boyut 1,55 ms; 1.000 enstrüman 7,34 ms; mevcut 8/12 ms bütçeleri içinde |
| ESLint | Sıfır hata/uyarı |
| TypeScript + Vite + prerender | Geçti |
| `git diff --check` | Geçti |
| Paket bütçesi | Landing JS 116.768 / 190.000 gzip byte; ortak CSS 11.591 / 14.000; dashboard JS 262.851 / 280.000 |
| Dashboard toplam CSS | 14.063 gzip byte; bütçe betiğindeki 14.000 sınırı **ortak/landing CSS** içindir |
| Dashboard, production build preview | 10 senaryo geçti |
| Dashboard, build sırasında açık kalan dev | Aynı 10 senaryo geçti |
| Landing ve diğer workspace regresyonları | Production build üzerinde 6 landing + 6 Models/Validation/System senaryosu geçti |

Dashboard viewport'ları: 1536×900, 1280×600, 768×900, 390×844 ve 320×650. Son ikisinde reduced-motion etkin. Beş ek durum: stale tahmin, degraded tahmin, boş geçmiş, servis hatası ve arka plan yenileme hatası. Test edilen sayfalarda yatay taşma yok; normal etkileşim akışlarında runtime exception yok. Haber ve market API'leri kontrollü fixture'lardır; canlı veri veya model başarısı kanıtı değildir.

## Yerel performans: önce / sonra

`scripts/profile-dashboard.mjs`: headless Edge, 1440×900, 4× CPU yavaşlatma, kapalı HTTP cache, reduced-motion, yerel API fixture'ları, üç ayrı sayfa yüklemesi. Aynı iş yükü: grafik tablosunu aç/kapat, tooltip üzerinde ilerle, haritaya kaydır. Grafik ve harita dolu; karşılaştırma boyunca haber listesi boş. Bu kısa laboratuvar akışı yaklaşık 6–8 saniye etkileşim içerir; tam 10–15 saniyelik cihaz/GPU trace kabulü değildir.

| Ölçüm — 3 koşunun medyanı | Önce | Sonra |
| --- | ---: | ---: |
| Lab LCP | 4.332 ms | 3.840 ms |
| Layout shift toplamı, son kullanıcı girdisi dışı | 0,0633 | 0,0762 |
| Etkileşimde en uzun long task | 130 ms | 104 ms |
| Koşu başına en yüksek Event Timing süresi | 64 ms | 56 ms |
| Son JS heap, GC zorlanmadan | 10,90 MB | 11,41 MB |

Sonuçlar karışık: bu örnekte LCP/etkileşim süreleri azaldı, layout shift ve ham heap arttı. Üç koşu ve kontrolsüz GC ile istatistiksel hızlanma, bellek sızıntısı yokluğu veya field INP başarısı iddia edilmez. 4× CPU koşulundaki lab LCP, üretim p75 LCP hedefinin karşılandığını göstermiyor.

Kalıcı ham kanıt: [önce](./evidence/frontend-phase-three-20260906/profile-before.json), [sonra](./evidence/frontend-phase-three-20260906/profile-after.json), [dashboard dev](./evidence/frontend-phase-three-20260906/dashboard-dev.json), [dashboard preview](./evidence/frontend-phase-three-20260906/dashboard-preview.json), [regresyonlar](./evidence/frontend-phase-three-20260906/experience-preview.json), [kaynak hash'leri](./evidence/frontend-phase-three-20260906/source-hashes.json).

## Korunan çalışma ve yayın sınırı

Başlangıçtaki commit edilmemiş 32 dosya geçici ZIP arşiviyle karşılaştırıldı: 27 dosya byte düzeyinde aynı; yalnızca bu turda hedeflenen 5 mevcut değişiklik dosyası genişletildi. Ayrıca heatmap kontrol dosyaları ve prerender betiği düzenlendi; yeni grafik/test/rapor dosyaları eklendi. Önceki landing dosyaları korundu. Backend, model eğitimi, finansal hesaplar, kalite eşikleri, otomatik polling aralıkları, treemap yerleşimi ve hover-card geometri kodu değiştirilmedi. Yeni paket bağımlılığı yok.

Kod teslimi tamamlandı. Yayın sonrası ayrı doğrulamalar: canlı sağlayıcı akışı, gerçek kullanıcı p75 LCP/INP/CLS, fiziksel cihaz FPS/GPU trace ve uzun oturum bellek davranışı. Bunlar için **ölçüm bulunamadı**; yerel kabul bu sonuçları yerine koymaz. Commit/push/deploy yapılmadı.
