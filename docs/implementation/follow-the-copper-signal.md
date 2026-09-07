# Follow the copper signal — uygulama

Kaynak: 7 Eylül canlı incelemesi, `321f824`. Tasarım: Cu → market bağlamı → kaynak/gerekçe → belirsizlik → kanıt → workspace.

## Kapsam ve sıradaki değişiklikler

1. Hero'nun mevcut kontur geometrisini koru; tek vurgulu çizgi ve aşağı devam eden iz ile kompozisyonu büyüt. Intro'yu kısa bir geçiş başlığına indir.
2. ResearchStory'de full-card crossfade'i kaldır. Tek kalıcı Cu/bağlantı sahnesi, ayrık metin pencereleri, tamamlanmış okuma aralıkları ve aktif chapter navigation oluştur. Native scroll ve mevcut Motion kullanılır; yeni runtime bağımlılığı yok.
3. Market bağlam diyagramı ve açılabilir gerekçe; haber kaynağı/yorum ayrımı; sabit tarihsel çizgi ve açılan forecast aralığı. Önizlemeler sentetik; canlı fiyat veya korelasyon iddiası yok.
4. Büyük boş metrik tablosunu kompakt forecast passport ile değiştir. Evidence bağlantılarını tam genişlik satırına taşı. CTA'da bakır izi sonlandır.
5. Mobil, reduced-motion ve kısa ekranlarda aynı sahnelerin normal akışta tam görünür versiyonunu sun. Gizlenen sahnelerde focusable kontrol bırakma; erişilebilir metni dekorasyon wrapper'ına gizleme.
6. Quote kontrol zamanı/fiyat zamanı ayrımı, nötr sıfır/fallback delta ve model risk/kalite etiketlerini netleştir. Tahmin veya backend hesaplaması değiştirilmez.

## Dosya ve sorumluluklar

- `story-timeline.ts`: saf progress/okuma pencereleri; transition regresyon testi.
- `SignalStage.tsx`: kalıcı SVG referans işareti ve semantic scene içerikleri; küçük yerel disclosure.
- `ResearchStory.tsx`: native scroll, chapter state, anchor/focus, static fallback.
- `Hero.tsx`, `CopperSignal.tsx`, `LandingPage.tsx`, `landing.css`: kompozisyon ve anlatı devamlılığı.
- `useExperiencePolicy.ts`: okunabilir viewport ve motion tercihi; basit sticky için bellek/pointer koşulu zorunlu değil.
- `OverviewPage.tsx`: yalnızca quote ve risk açıklaması; finansal hesaplar/heatmap korunur.

## Kabul

Her progress noktasında en fazla bir metin sahnesi görünür; reverse scroll aynı state'e döner. Her chapter tamamlanmış okuma alanına sahip olur. Anchor'lar, direct dashboard route ve semantic içerik korunur. Gerekçe keyboard/tap ile açılır. Static prerender faydalı kalır. Test/lint/build/prerender ve mevcut paket bütçeleri çalıştırılır. Önceki ortamda yerel preview URL'si policy ile engellendi; bu kısıt aşılmaya çalışılmaz, canlı görsel kabul yapılmadıysa raporda açık bırakılır. PR incelemeye açılır; main merge veya production deploy yapılmaz.
