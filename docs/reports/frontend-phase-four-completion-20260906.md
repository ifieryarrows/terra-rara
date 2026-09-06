# Faz 4 — Landing ürün anlatımı teslimi

Tarih: 2026-09-06. Yerel geliştirme ve kabul tamamlandı; production yayını yapılmadı.

## Teslim

- Hero: Cu görseli ve dashboard CTA korundu. Market, News, Forecasts ve Validation için doğrudan bölüm bağlantıları ve daha açık ürün açıklaması eklendi.
- News: genel adım listesi, örnek kaynak/tarih içeren bir araştırma kartına dönüştürüldü. Hipotetik arz senaryosu, koşullu sentiment gerekçesi ve okunabilir supply/inventory/demand bağlamı birlikte gösteriliyor. Gerçek haber veya canlı sentiment iddiası yok.
- Forecast: günlük T+1…T+5 yolu, son kapanış, medyan/Q10–Q90 aralığı ve ayrı ana 5D açıklaması görünür. T+1 tanılaması haftalık sonuçla eşitlenmiyor; sayılar sentetik örnektir.
- Evidence: haftalık yön doğruluğu, MAE/RMSE, aynı örneklemde baseline karşılaştırması, değerlendirme dönemi, veri tarihi ve eksik sonuç açıklaması içeren rapor önizlemesi eklendi. Gerçek performans değerleri veya güncellik tarihi uydurulmadı.
- Models, Validation, System ve dashboard bölüm bağlantıları korundu. Mevcut üç chapter, native scroll, Cu ve heatmap yapısı korunuyor.
- Geist Sans ve önceki yerel değişiklikler korundu. League Gothic A/B görsel karşılaştırması yapılmadı; ikinci font eklenmedi.

## Kontroller

| Kontrol | Sonuç |
| --- | --- |
| ESLint | Geçti |
| Vitest | 8 dosyada 49 test geçti |
| TypeScript, Vite build, prerender | Geçti |
| Paket bütçeleri | Geçti |
| Landing tarayıcı senaryoları | 7 geçti: 1536×900, 1280×600, 1024×650, 768×800, 320×650, 390×844, reduced-motion 1280×720 |
| Workspace fixture senaryoları | Models/Validation/System; 1536 ve 390 px, 6 geçti |
| Font başarısızlığı | 320 ve 1280 px: taşma yok; klavyeyle Validation bağlantısı ve sahne görüntüleri kontrol edildi |
| Scroll | Üç sahnede doğru görünür katman; kısa ekranda stage içeriği 593.86 px / 600 px |
| Landing ağ/JS | Finans API isteği ve pageerror yok |

Yeni ekran görüntülerinde kısa ekran forecast, mobil news ve desktop evidence incelendi. Tarayıcı çıktıları yerel geçici `coppermind-experience-qa/results.json` dosyasında; fixture sonuçları canlı servis kanıtı değildir. Mevcut dashboard etkileşim betiği bu tur tekrar çalıştırılmadı; dashboard uygulama kodu değiştirilmedi, ilgili birim testleri geçti.

Gzip: ilk JS **117.583 byte**, ortak CSS **12.113 byte**, dashboard JS **263.662 byte**. Sınırlar sırasıyla 190.000 / 14.000 / 280.000 byte. Geist ayrı **69.760 byte** WOFF2; ek font, raster medya veya npm bağımlılığı yok.

## Sonraki adım

Son sürüm için yayın kabulü: canlı API/cache/freshness, fiziksel cihaz scroll/GPU, uzun oturum belleği, ürün anlaşılabilirliği ve gerçek kullanıcı p75 LCP/INP/CLS. Bu teslim yerel kabulü kapsar; production performans ölçümü değildir. Commit, push ve deploy yapılmadı.
