# 2. faz — Plan ve uygulama denetimi

Tarih: 2026-09-06. Kaynaklar: “Teknik UI UX Analizi” konuşması; `docs/implementation/frontend-experience-plan.md`; `docs/implementation/frontend-experience-phase-two.md`; mevcut yerel kaynak kodu.

## Karar

2. fazın yönü ilk öneriyle uyumlu. Cu çizimi tek kazanım değil; üç scroll bölümü ve workspace bileşenleri de uygulanmış. Bununla birlikte önceki durum raporunun tüm davranışların doğrulandığı ve yalnızca yayınlamanın kaldığı izlenimi fazla genişti. 35 testin çoğu 1. fazdan kalma route, reduced-motion, heatmap ve drawer testleridir; yeni haber filtrelerini veya kısa ekran sticky geometrisini doğrulamıyordu.

## Hedef / kanıt

| Hedef | Mevcut uygulama | Değerlendirme |
| --- | --- | --- |
| CopperMind'e özgü hero | `Hero.tsx`, `CopperSignal.tsx`; SVG konturlar, ürün açıklaması, doğrudan dashboard CTA | Uygulanmış |
| Heatmap'in scroll ile oluşması | `Previews.tsx / AnimatedMarketCell`; hücre sırasına bağlı opacity ve transform | Uygulanmış |
| Haber anlatımının açılması | `NewsStep`; üç workflow adımı, ortak scroll progress | Uygulanmış; gerçek haber taklidi yapılmıyor |
| Forecast çizimi ve belirsizlik | `ForecastPreview`, `ForecastRange`; path reveal ve aralık opacity | Uygulanmış; örnek veri olarak etiketli |
| Bölümler arası bağ | `ResearchStory`; katman crossfade, chapter bağlantıları, dashboard anchor'ları | Uygulanmış |
| Mobil / reduced-motion | Tek sütun statik anlatı, tüm üç bölümün HTML'de bulunması | 390/768 px ve reduced-motion tarayıcı kontrolü geçti |
| Workspace ortak bileşenleri | `PageHeader`, `ViewState`, `DataTable`, `FilterChip`, `RefreshButton` | Uygulanmış; tüm dashboard modernizasyonu bitmiş değil |
| Kısa masaüstü scroll | Önceki `height:auto` grid stretch nedeniyle 1280×600'de 2304 px stage | Bu denetimde düzeltildi |
| Production kullanıcı deneyimi | Yeni deployed field ölçümü yok | Açık; yerel testle kapatılamaz |

## Bulunan ve giderilen scroll hatası

1280×600 Edge ölçümünde `.cm-story-stage` 2304 px yüksekliğe geriliyordu. Grid öğesine `align-self:start` verildi, stage viewport yüksekliğinde tutuldu ve kısa ekranda preview iç boşlukları daraltıldı. Düzeltme sonrası stage 600 px, içeriği yaklaşık 594 px; üç katman sırasıyla görünür ve stage bölüm boyunca sabit.

`frontend/scripts/check-experience.mjs` bu kabulü tekrarlanabilir yapar: 1536×900, 1280×600, 1024×650, 768×800, 390×844 ve 1280×720 reduced-motion senaryolarının altısı geçti. Kontroller: yatay taşma yok, doğru static/enhanced modu, üç chapter görünürlüğü, stage geometrisi, landing'de finans API isteği ve runtime hatası yok. Bu geometrik doğrulama tüm cihazlarda performans veya tasarım beğenisi garantisi değildir.

## Faz sınırı

Son doğrulama: aynı altı landing senaryosu hem yeni Vite dev sunucusunda (5184), hem `npm run build` çıktısının preview sunucusunda (5185) geçti. Kanıt çıktıları geçici dizinde `coppermind-experience-qa/results.json` ve `coppermind-experience-preview-qa/results.json`; kontrol betiği repoda tutuluyor. Workspace'in altı ek fixture senaryosu ve 42 testin kapsamı [3. faz ilk dilim raporunda](./frontend-phase-three-first-slice-20260906.md).

Dar 2. faz implementasyon planının ana teslimleri mevcut; bulunan kısa ekran hatası kapatıldı. Tüm ürünün premium tasarım hedefi bitmedi. Sonraki çalışma `frontend-experience-phase-three.md` ile kapsamlandırıldı. Commit/deploy ve gerçek production LCP/INP/CLS/FPS ölçümü ayrı açık işlerdir. Bu denetimde GitHub'dan dosya alınmadı, mevcut dosyalar silinmedi.
