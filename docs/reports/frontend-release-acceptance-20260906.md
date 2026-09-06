# Frontend yayın kabulü — 2026-09-06

Karar: son yerel build'in canlı API ile entegrasyon kontrolü geçti. Nihai production kabulü açık; aday sürüm henüz yayınlanmamış. Kontrol zamanı yaklaşık 19:38–19:41 Türkiye saati.

## Doğrulananlar

- Yerel production build, Headless Edge üzerinde 1280×800 ve 390×800 boyutlarında gerçek API yanıtlarıyla çalıştırıldı. Landing, dashboard, models, validation ve system için 10 sayfa kontrolünde yatay taşma ve JavaScript pageerror görülmedi. Reduced-motion kullanıldı; bu kontrol fiziksel cihaz testi değildir.
- Tarayıcıdaki `/api/` istekleri değiştirilmeden `https://terra-rara.vercel.app/api/` adresine iletildi. Fixture kullanılmadı. Analysis, TFT, history, live-price, sentiment, commentary, news, news stats, heatmap, model summary, backtest ve health yanıtları HTTP 200 döndü.
- Landing finans API isteği yapmadı. Dashboard'da heatmap ve haber bölümleri scroll ile açıldı.
- Health: healthy, Redis açık, pipeline kilidi kapalı, son pipeline durumu ok. Snapshot yaşı ölçüm anında yaklaşık 16,24 saat; bu değer sıfır veya anlık veri olarak sunulmamalı.
- TFT snapshot referansı, son PriceBar ve history'nin son tarihi aynı: **2026-09-04**. Son history kapanışı 6,597. TFT kaynak türü snapshot, ana ufuk 5D.
- XGB hedefi baseline × (1 + return) sözleşmesiyle yuvarlama toleransı içinde eşleşiyor; mutlak fark yaklaşık 0,0000122.
- Commentary `generation_mode=llm`, `error=null`, `fallback_reason=null` döndürdü. Bu, metnin tüm iddialarının doğruluk denetimi değildir.
- Canlı backtest `available=false` döndürüyor. Son arayüz her iki genişlikte doğru biçimde “No backtest report available yet” gösterdi; eksik rapor başarı olarak sunulmadı.

## Açık kabul maddeleri

1. **Yayın sürümü farklı.** Canlı JS/CSS `index-ZUfC_bxI.js` / `index-D7uPPXwM.css`; aday build `index-B9xySxsm.js` / `index-YM1GoN5_.css`. Canlı HTML Geist preload içermiyor. Canlı font yolu HTTP 200 ile HTML workspace fallback döndürüyor; bu eski yayında fontun bulunmadığını gösterir, aday font dosyasının doğrulandığı anlamına gelmez. Aday yayınlandıktan sonra gerçek font MIME/byte ve route kontrolleri tekrarlanmalı.
2. **Walk-forward raporu yok.** Eksik durum arayüzü doğru; gerçek out-of-sample sonuç ve baseline içeriği açısından kabul açık. Bu UI turunda backtest/eğitim tetiklenmedi veya model değiştirilmedi.
3. **Gerçek kullanıcı p75 LCP/INP/CLS: Ölçüm bulunamadı.** Kodda SpeedInsights bulunması ölçüm kanıtı değildir. Son yayın kimliğine bağlı yeterli trafik/pencere ile saha sonuçları gerekir.
4. Fiziksel mobil cihaz scroll/GPU, uzun oturum bellek davranışı ve bağımsız kullanıcıyla ürün anlaşılabilirliği henüz ölçülmedi. Kısa headless kontroller bu maddeleri kapatmaz.

## Kanıt ve kapsam

[Kontrol özeti JSON](./frontend-release-acceptance-20260906.json). Önceki aday build testleri: [Faz 4 teslimi](./frontend-phase-four-completion-20260906.md).

Bu tur rapor ve kanıt dosyaları eklendi; uygulama/model kodu değiştirilmedi. Mevcut yerel değişiklikler korundu. Commit, push veya deploy yapılmadı. Entegrasyon sonucu olumlu olsa da yukarıdaki açık maddeler nedeniyle tam production kabulü verilmedi.
