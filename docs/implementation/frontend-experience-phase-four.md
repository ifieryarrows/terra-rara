# Faz 4 — Landing'de ürünün tamamını görünür kılma

Durum: geliştirildi ve yerel kabul kontrolleri tamamlandı. [Teslim raporu](../reports/frontend-phase-four-completion-20260906.md). Kaynak: kullanıcının 2026-09-06 tarihli landing/heatmap karşılaştırması ve [güncel yol haritası](./frontend-experience-roadmap.md).

## Amaç

Ziyaretçi CopperMind'i yalnızca bir heatmap olarak algılamamalı. İlk viewport ürünün piyasa bağlamı, haber analizi, tahmin ve doğrulama yeteneklerini anlaşılır biçimde tanıtmalı; sayfanın devamı bunları somut ürün örnekleriyle göstermeli.

## Görsel teslimler

| Bölüm | Şu anki eksik | Planlanan sahne / kullanıcı kazanımı |
| --- | --- | --- |
| Hero | Cu kimliği güçlü; diğer ürün işlevleri soyut | Mevcut Cu ve doğrudan dashboard CTA korunarak dört yeteneğe kısa, okunabilir giriş; dekoratif görselin yanında açık ürün vaadi |
| Market | En somut mevcut ürün önizlemesi | Korunacak referans; yeni efekt veya yeniden tasarım hedefi değil |
| News intelligence | Metinsel workflow listesi | Açıkça örnek olarak işaretlenmiş haber kartı, kaynak/tarih, sentiment gerekçesi ve detay okuma görünümü; metin yerine okuyucunun yaptığı işi göster |
| Forecast | Çizgi/aralık var, kullanım bağlamı zayıf | Son kapanıştan çıkan günlük yol ile ayrı 5D özet; Q10–Q90 belirsizliği; geçmiş/günlük/haftalık ayrımını anlaşılır etiketle |
| Drivers / commentary | Genel ifadelerin içinde kayboluyor | Bir etkinin insan tarafından okunabilir adı ve açıklaması; yorumun dayandığı bağlamı göster; nedensellik veya al/sat sonucu uydurma |
| Validation / freshness | Çoğunlukla bağlantı kartları | Kanıt inceleme sahnesi: metrik açıklaması, dönem/baseline, veri tarihi ve eksik sonuç hali. Uydurma doğruluk yüzdesi veya başarı rozeti kullanma |

Sahneler mevcut özelliklerin tanıtımıdır; yeni model veya yeni backend fonksiyonu değildir. Gerçek ekran görüntüsü kullanılacaksa tarih/kaynak açık olmalı; sentetik örnekler görünür biçimde etiketlenmeli. Landing canlı haber/finans servislerini çağırmaz.

## Uygulama sırası

1. Mevcut desktop/mobile kompozisyonunu Geist ile kaydet; News/Forecast/Evidence sahnelerinin statik taslağını oluştur. League Gothic'i yalnızca kısa başlıklarda bu taslak üzerinde karşılaştır.
2. News ve Forecast önizlemelerini gerçek ürün kullanımına benzet. İlk etapta mevcut üç chapter ve native scroll düzenini koru; önceki kısa ekran sticky düzeltmesini bozma.
3. Evidence bölümünü görsel kanıt okuma örneğiyle geliştir; Market Drivers ve yorum bağlantısını anlatıya yerleştir. İlgili gerçek sayfa/bölüm linklerini doğrula.
4. Geçişleri mevcut MotionValue ile bağla; görsel hareket metnin okunmasını bekletmesin. Mobil/reduced-motion tam statik anlatı sunsun.
5. Aşağıdaki kabul kontrollerini tamamla; sonra yayın kabulü için son build'i değerlendir.

## Kabul ölçütleri

- [x] Heatmap, News, Forecast ve Validation tek başına anlaşılabilir bir görsel örneğe sahip.
- [x] Her bölüm “Ne görüyorum, ne işime yarıyor, nereden açarım?” sorularını yanıtlıyor.
- [x] İlk viewport ürün açıklaması ve doğrudan dashboard CTA içeriyor; zorunlu scroll yok.
- [x] Haftalık sonuç, T+1 tanılama ve örnek veriler birbirine karışmıyor.
- [x] Sahte live veri, uydurma haber/başarı iddiası, landing API polling'i yok.
- [x] 320/390 px mobil ve 1280×600 kısa masaüstünde içerik/CTA kırpılmıyor; reduced-motion ve klavye kullanılabilir.
- [x] Font yüklü ve font isteği başarısızken okunabilir fallback ile kontrol edildi.
- [x] Mevcut route, heatmap, model ve kalite kapısı davranışları korunuyor.
- [x] Lint/build/test, mevcut paket bütçeleri ve ayrı font/medya transfer maliyeti raporlandı.
- [x] Yerel ölçüm ile gerçek production kabulü ayrı raporlandı.

Kapsam dışı: landing'i sıfırdan silip yeniden kurmak, dashboard'un tekrar tasarımı, yeni 3D/scroll kütüphanesi, model eğitimi ve kalite eşiklerini değiştirmek.

## Uygulama kararı

Geist Sans korundu. League Gothic için ayrı görsel A/B karşılaştırması bu teslimde yapılmadı; ikinci font eklenmedi. Yeni haber sahnesi gerçek yayın taklidi yerine tarihli, açıkça etiketlenmiş hipotetik araştırma senaryosu kullanır. Doğrulama önizlemesi gerçek sonuç/tarih uydurmadan rapor alanlarını ve eksik sonuç anlamını gösterir.
