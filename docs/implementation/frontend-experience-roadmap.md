# CopperMind — Güncel frontend yol haritası

Güncelleme: 2026-09-06. İncelenen kaynak: `52030fe` ve faz planları/teslim raporları. Bu belge fazlar arası güncel sıradır; geçmiş raporların yerine tamamlanmamış işi tamamlanmış göstermez.

## Mevcut durum

| Aşama | Durum | Kaynak |
| --- | --- | --- |
| 1 — Tasarım temeli, routing ve ilk landing | Geliştirme tamamlandı | [İlk plan](./frontend-experience-plan.md) |
| 2 — Cu hero, scroll anlatısı, ortak workspace bileşenleri | Geliştirme ve yerel denetim tamamlandı | [Denetim](../reports/frontend-phase-two-audit-20260906.md) |
| 3 — Dashboard grafik ve çalışma akışları | Geliştirme ve yerel kabul tamamlandı | [Kapanış](../reports/frontend-phase-three-completion-20260906.md) |
| Tipografi — Geist Sans | Bu güncellemede uygulandı | Tüm metin ve sayılar için ortak aile; yerel değişken WOFF2; `tabular-nums` korunur |
| 4 — Landing'de ürün özelliklerini dengeli gösterme | Geliştirildi; yerel kabul tamamlandı | [4. faz planı](./frontend-experience-phase-four.md) |
| 5 — Market → News Intelligence product demonstration | Pipeline doğrulandı; sequence uygulanıyor | [News sequence planı](./frontend-news-intelligence-sequence.md) |
| Yayın kabulü | Açık; son sürüm üzerinde yapılmalı | Canlı servisler, gerçek cihaz ve kullanıcı ölçümleri |

Önceki belgelerde ayrıntılı bir 4. faz yoktu. 3. faz sonrası kalan iş yayın/üretim doğrulamasıydı. Kullanıcının yeni isteğiyle landing ürün anlatımı geliştirme sırasının sonuna eklendi; nihai yayın kabulü bu değişiklikler üzerinde tekrar yapılacak.

## Uygulama öncesi landing incelemesi

Kullanıcının gözlemi kodla örtüşüyor: `MarketPreview` görsel hücreler, semboller ve değişimleri gösteriyor. `NewsPreview` ise üç metinsel süreç adımıyla sınırlı. `ForecastPreview` fiyat çizgisi/aralık sunuyor ancak haftalık sonuç ile günlük tanılamanın kullanıcıya ne sağladığını yeterince göstermiyor. Landing'in evidence bölümü Models/Validation/System bağlantılarından oluşuyor; doğrulama deneyimini somutlaştırmıyor. Yalnızca yazıları veya fontu değiştirmek bu farkı kapatmaz.

## Faz 4 teslim edilen geliştirme

1. Haber/sentiment, tahmin ve doğrulama için küçük ürün sahneleri tasarla; mevcut heatmap'i görsel ayrıntı referansı olarak kullan.
2. Haber kaynağı → sentiment gerekçesi → bağlam; geçmiş fiyat → 5D sonuç → günlük yol ve aralık; model sonucu → kanıt ve veri tarihi ilişkilerini göster.
3. Her sahneden ilgili gerçek workspace bölümüne geçiş ver. Mobil ve reduced-motion'da anlamı koru.
4. Yerel kabulden sonra son build için yayın kontrolünü yap. Eski sürümün lab değerlerini yeni font veya yeni landing için geçerli sayma.

## Tipografi kararı

Ana arayüz, landing metinleri ve tüm sayısal alanlar Geist Sans kullanır; hizalama gereken değerlerde `tabular-nums` korunur. League Gothic, yalnızca kısa landing başlıkları için 4. fazda görsel karşılaştırma adayıdır; gövde metnine, tabloya ve dashboard'a uygulanmadı. İkinci font ancak somut kompozisyonda fayda sağlarsa eklenir.

Geist'in resmi [Vercel dağıtımı](https://vercel.com/font) kullanıldı: v1.7.2, 69.760 byte WOFF2, yerel barındırma, `font-display: swap`, lisans dosyasıyla birlikte. Yeni npm/Next.js bağımlılığı yok. Font boyutu JS/CSS bütçesi dışında ayrı transfer maliyetidir.

Tipografi kabulü: build, lint ve mevcut paket bütçeleri geçti. Production build preview üzerinde 12 landing/workspace + 10 dashboard senaryosu geçti; 320 px mobil ve 1280×600 kısa ekran dahil. Browser font incelemesi Türkçe 12 karakterin yüklenen Geist ile çizildiğini doğruladı; font isteği engellendiğinde ana başlık okunabilir ve sayfa taşmasız kaldı. Dashboard kontrol betiğinde Escape'in önce açık kategori panelini kapatması dikkate alındı; uygulamanın Escape davranışı değiştirilmedi. Yeni JS/CSS gzip değerleri: landing JS 116.767 byte, ortak CSS 11.647 byte, dashboard JS 262.846 byte. Bunlar gerçek kullanıcı performans ölçümü değildir.

## Açık yayın kabulü

2026-09-06 canlı entegrasyon kontrolü: aday build gerçek API ile desktop/mobile toplam 10 sayfa kontrolünü geçti. Canlı yayın hâlâ eski sürüm; walk-forward raporu mevcut değil. Saha performansı ve fiziksel cihaz kabulü açık. [Yayın kabul raporu](../reports/frontend-release-acceptance-20260906.md).

Canlı API akışı, cache/freshness, gerçek kullanıcı p75 LCP/INP/CLS, fiziksel cihaz scroll/GPU, uzun oturum bellek davranışı ve ürün anlaşılabilirliği değerlendirmesi açık. Yerel fixture sonuçları bunların yerine geçmez. Bu roadmap güncellemesi commit/push/deploy talebi değildir.

Faz 4 tamamlanma kaydı: [yerel teslim ve kontrol raporu](../reports/frontend-phase-four-completion-20260906.md). Sıradaki iş son build için yayın kabulüdür. League Gothic görsel karşılaştırması yapılmadı; mevcut Geist kompozisyonu korundu.
