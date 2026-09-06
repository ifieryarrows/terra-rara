# Faz 3 — İlk çalışma dilimi teslim raporu

Tarih: 2026-09-06. Kapsam: [3. faz planı](../implementation/frontend-experience-phase-three.md). Başlangıç kararı: [2. faz denetimi](./frontend-phase-two-audit-20260906.md).

## Sonuç

2. fazın ana uygulamaları korundu ve kısa ekran scroll hatası giderildi. 3. fazın finansal çalışma ekranlarına yönelik ilk dilimi tamamlandı. Tüm ürünün görsel dönüşümü veya production performans kabulü tamamlanmış sayılmıyor.

## Bu turda değişen davranışlar

| Alan | Önce / sorun | Teslim |
| --- | --- | --- |
| Models | Haftalık strateji ve günlük tanılama aynı ağırlıkta görünüyordu | Haftalık metrikler öncelikli; T+1 native details içinde, klavyeyle açılabilir. MAE/RMSE açıklamaları ve ortak bölüm başlıkları. Mevcut hesaplar/eşikler aynı |
| Validation | Karşılaştırma ham JSON'du; rolling raporun `mean_*`, `da`, `sharpe` alanları boş görünebiliyordu | Raporlanan TFT/Theta değerleri semantik tabloda; mevcut alan adlarına uyum. Sıfır korunur, eksik veri çizgidir; diğer rapor biçimleri ham detayda erişilebilir |
| News | Seçili yayıncı tekrar tıklanınca kalkmıyordu; kanal dağılımı daralınca seçimi kaldırma kontrolü kaybolabiliyordu | Yayıncı toggle; seçili kanal ve All channels korunur; eski sonuçlar yenilenirken görünür durum metni |
| System | Eksik model sayısı sıfır, eksik kilit bilgisi free gibi görünebiliyordu | Bilinmeyen alanlar nötr ve açık; servis/veri grupları masaüstünde yan yana, mobilde alt alta |
| Ortak yapı | Bölüm hiyerarşisi sayfa bazında tekrarlanıyordu | `SectionHeader`, responsive metrik ve servis grid'leri |

Kaynak dosyalar: `frontend/src/pages/{ModelsPage,ValidationPage,SystemPage}.tsx`, `frontend/src/features/validation/ReportComparison.tsx`, `frontend/src/features/news/NewsIntelligencePanel.tsx`, `frontend/src/components/ui/SectionHeader.tsx`, `frontend/src/design/workspace.css`. Scroll düzeltmesi: `frontend/src/features/landing/landing.css`.

Validation alan uyumunun kaynak kanıtı: `backend/deep_learning/validation/backtest.py` içindeki rolling summary ve `compare_with_baseline`. Backend dosyası değiştirilmedi. Tanınmayan karşılaştırma biçimlerinden Theta sonucu veya yeni kazanan çıkarılmıyor.

## Doğrulama

| Kontrol | Sonuç ve kapsam |
| --- | --- |
| `npm.cmd run test` | 7 dosyada 42 test geçti; 7 yeni davranış testi |
| Yeni workspace testleri | Haftalık/günlük ayrımı; rapor alan uyumu, gerçek sıfır ve eksik metrik; Theta olmayan rapor; hata/yenileme; bilinmeyen health alanları |
| Yeni haber testleri | Yayıncı toggle, daralan kanal seçimini kaldırma; 300 ms arama debounce ve reset |
| `npm.cmd run lint` | Geçti |
| `npm.cmd run build` | TypeScript, Vite ve prerender geçti |
| `npm.cmd run check:budgets` | Landing JS 116.783 / 190.000 gzip byte; ortak CSS 11.736 / 14.000; dashboard JS 261.916 / 280.000. Dashboard toplam CSS 13.532 byte |
| Headless Edge, dev | 6 landing + 6 workspace senaryosu geçti |
| Headless Edge, production build preview | Aynı 12 senaryo geçti |

Landing senaryoları: 1536×900, 1280×600, 1024×650, 768×800, 390×844 ve reduced-motion 1280×720. Üç katmanın sırası, kısa ekranda sticky geometrisi, statik fallback, yatay taşma, runtime hataları ve landing finans API istekleri denetlendi. Landing'de finans API isteği yok.

Workspace senaryoları: Models, Validation ve System; 1536 ve 390 px. Taşma yok; dört navigasyon bağlantısı erişilebilir. Models günlük bölümü Enter ile açıldı; validation özet ve karşılaştırma değerleri kontrol edildi. Masaüstü Models ve mobil Validation ekran görüntüleri ayrıca görsel olarak incelendi. Tablolar mobilde kendi yatay scroll alanlarını kullanır.

Workspace browser testleri açıkça sentetik API fixture'ları kullanır. Bunlar canlı veri kalitesinin veya sağlayıcı erişilebilirliğinin kanıtı değildir. Haber etkileşimleri DOM davranış testleriyle doğrulandı; bu turdaki browser matrisi News'i kapsamaz. Tam ekran okuyucu veya tüm tarayıcılar için erişilebilirlik kabulü yapılmadı.

## Tekrarlama ve kanıt

`frontend/scripts/check-experience.mjs` haricen kurulu Playwright ve varsayılan olarak Edge kullanır; projeye bağımlılık eklemez. `PLAYWRIGHT_MODULE` paket yolu, `EXPERIENCE_URL` hedef yerel sunucu, `EXPERIENCE_OUTPUT` JSON/PNG çıktı dizinidir. Production kontrolünden önce frontend build ve preview çalıştırılır.

Bu çalışmanın çıktıları Windows geçici dizininde `coppermind-experience-qa` (dev) ve `coppermind-experience-preview-qa` (production build preview) altındadır. Her dizinde `results.json` ve ekran görüntüleri bulunur. Geçici dosyalar kalıcı CI artifact'i değildir; kontrol betiği yeniden üretim kaynağıdır.

## Korunanlar ve açık işler

- Kullanıcının önceki commit edilmemiş dosyaları korundu. Başlangıçtaki değişiklikler için geçici dizinde `coppermind-phase-two-before-audit-20260906.zip` arşivi oluşturuldu. GitHub'dan sürüm alınmadı; dosya silinmedi; commit/push/deploy yapılmadı.
- Backend, model eğitimi, kalite eşikleri, veri sorguları/polling, heatmap geometrisi ve package bağımlılıkları değişmedi.
- Daha kapsamlı grafik araçları, kalan legacy UI alanları ve üretim ortamındaki ölçümler sonraki 3. faz dilimleridir.
- Gerçek cihaz FPS, p75 LCP/INP/CLS ve bellek profili: **Ölçüm bulunamadı**. Paket bütçesi ve headless geometrik test bunların yerine geçmez.
