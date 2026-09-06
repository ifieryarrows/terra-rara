# Faz 3 — Finansal çalışma ekranları

Tarih: 2026-09-06. Başlangıç: commit edilmemiş 2. faz çalışma ağacı; `5e032d0` yalnızca merge edilmiş temel sürümdür.

Durum: geliştirme ve yerel kabul tamamlandı. Sonuçlar ve yayın sınırları [kapanış raporunda](../reports/frontend-phase-three-completion-20260906.md).

## Kapsamın kaynağı

“Teknik UI UX Analizi” konuşmasının son önerisi, landing görsel anlatımının ardından dashboard tabloları, filtreler, formlar, grafikler ve durum ekranlarının iyileştirilmesidir. Konuşmada ayrı, ayrıntılı bir 3. faz sprinti tanımlanmamıştı. Bu belge önce ilk çalışma dilimini tanımladı; kullanıcının dashboard ve grafik iyileştirmelerini tamamlayarak 3. fazı bitirme isteğiyle aşağıdaki kapanış kapsamına genişletildi.

## İlk dilim

1. Models: ortak bölüm başlıkları; haftalık metrikler öncelikli, T+1 tanılaması kullanıcı açtığında görünür. Hesaplar ve eşikler korunur.
2. Validation: Theta karşılaştırmasını okunabilir metrik tablosuna dönüştür. Backend'in mevcut `mean_*` özet ve `da`/`sharpe` pencere alanlarını da göster; eksik sayıdan sıfır üretme. Tanınmayan rapor detaylarını erişilebilir bırak.
3. News: seçili yayıncıyı tekrar tıklayarak kaldırabilme; filtre daraldığında seçili kanalı kaldırabilme; eski sonuçlar güncellenirken görünür durum bilgisi.
4. System: bilinmeyen servis/model/kilit bilgilerini başarılı veya sıfırmış gibi göstermeme; ortak bölüm başlıkları ve responsive servis grupları.
5. Davranış testleri ve kontrollü API fixture'larıyla gerçek tarayıcı doğrulaması. Fixture ekranları canlı veri olarak raporlanmaz.

## Kapanış kapsamı

- Grafik: 30/90/180 gözlenen kapanış seçimi; medyan ve Q10–Q90 görünürlüğü; belirsizlik bandı; anlamlı son kapanış etiketi; USD tooltip; klavye ve mobil veri tablosu.
- Veri sınırı: dönem seçimi yeni API isteği üretmez. Tahmin fiyatları/tarihleri yeniden hesaplanmaz. Eski baz tarihli veya degraded tahmin yeni geçmişe bağlanmaz; eksik veri sıfır yapılmaz.
- Overview: açık haftalık/günlük metrik grupları; bölüm bağlantıları; manuel yenileme; başarısız arka plan yenilemesinde önceki verinin uyarıyla korunması; boş yorum/model/veri durumları.
- News: kısa ekranlarda filtreler ve haberler ortak kaydırma alanında; drawer klavye odağı, Escape ve sayfa scroll geri yükleme doğrulaması.
- Heatmap: ortak filtre kontrolleri, reset, dokunma/klavye zoom kontrolleri; tam ekran viewport/focus/scroll yaşam döngüsü; hata ve retry. Mevcut geometrik yerleşim ve hover-card yoğunluğu korunur.
- Yerel çalışma: prerender ve dev optimizer önbellekleri ayrılır; dev açıkken build sonrasında sayfalar çalışır.
- Kabul: davranış testleri, lint, build, bundle bütçesi; mobil/masaüstü/reduced-motion browser kontrolü; mevcut landing ve workspace regresyonları; karşılaştırılabilir yerel performans örnekleri ve sınırlarıyla raporlama.

## Sınırlar ve yayın kabulü

Yeni bağımlılık, otomatik polling aralığı değişikliği, backend/model/kalite eşiği değişikliği yok. Manuel yenileme kullanıcı eylemidir; eşzamanlı overview yenilemeleri birleştirilir. Heatmap geometri ve pointer optimizasyonları korunur. Landing'in kabul edilen kompozisyonu korunur. Mevcut çalışma ağacı GitHub sürümüyle değiştirilmez.

Bu faz mevcut finansal araştırma akışlarını tamamlar; yeni grafik ürünleri, yeni model fonksiyonları veya tüm utility class'ların toplu yeniden yazımı kapsamda değildir. Gerçek kullanıcı p75 LCP/INP/CLS, fiziksel cihaz FPS, uzun oturum bellek profili ve canlı sağlayıcı doğrulaması yayın kabulüdür. Yerel fixture/lab kontrolleri bunları tamamlanmış göstermez. Kod teslimi commit veya deployment anlamına gelmez.
