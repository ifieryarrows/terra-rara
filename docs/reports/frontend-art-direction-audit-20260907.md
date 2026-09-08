# CopperMind — Canlı deneyim incelemesi ve art direction önerisi

İnceleme: 7 Eylül 2026, yaklaşık 21:45–21:50 UTC. Tarayıcıdaki yerelleştirilmiş tarihler Türkiye saatinde 8 Eylül olarak görünebilir.

İncelenen yayın: https://terra-rara.vercel.app/ . Kaynak kodu: `321f8244e04f81c1fa8dadbc8aa332f018d265c5` (`main`). Canlı HTML script yolu `assets/index-D3qMiocK.js`, stylesheet `assets/index-CBNzy2QM.css`. Canlı içerik güncel Cu hero ve faz 4 önizlemeleriyle örtüşüyor; deployment–commit eşleşmesi ayrıca doğrulanmadı.

Bu teslimat bir canlı inceleme ve uygulanabilir tasarım önerisidir. Uygulama kodu değişikliği, yeni tasarımın tamamlandığına dair beyan veya production performans kabulü değildir. Önceki DOCX teknik prensip kaynağıdır; içindeki araçlar zorunlu teslim listesi değildir.

## 1. Karar

Mevcut site markalı, tutarlı ve anlaşılır bir ürün tanıtımına dönüşmüş. Ancak özgün Cu görseliyle başlayan anlatı daha sonra aynı dikdörtgen panel içinde değişen önizlemelere dönüyor. Eksiklik, tek başına animasyon miktarı değil: **görsel süreklilik, sahne kurgusu, okuma ritmi ve kullanıcının araştırmada bir adım ilerlediğini hissetmesi**.

Sticky çalışıyor. Cinematic etki sınırlı. Bunun üzerine Lenis veya Three.js eklemek mevcut kompozisyon problemini kendiliğinden çözmez. Daha iddialı yaklaşım, Cu motifini tüm anlatının görsel öznesi yapmak ve bir bölümde görülen öğenin sonraki bölümde anlamlı biçimde devam etmesini sağlamak.

## 2. Kanıt kapsamı

| Kanıt türü | Yapılan | Sınır |
| --- | --- | --- |
| Canlı görsel inceleme | 1363×936 viewport; hero, intro, üç sticky sahne, geçiş ortası, evidence, CTA/footer | Bu tur mobil viewport ve fiziksel cihaz görsel kabulü yapılmadı |
| Canlı etkileşim | Scroll, PageDown/End, chapter anchor'ları, dashboard CTA, 90 closes kontrolü, Models/Validation/System navigasyonu | Tüm kontrollerin regresyon testi değil |
| DOM/CSS | Sticky konumu, sahne boyutları, katman opacity/transform, chapter navigation, script/style yolları, computed font | Network waterfall, GPU trace veya kaynak haritası incelemesi değil |
| Repository | Landing componentleri, motion policy, token'lar, OverviewPage, önceki faz planları ve raporları | Son main ile yayın commit'i birebir kanıtlanmış değil |
| DOCX | Terminoloji, finans ürününe aktarım ve son prensipler yeniden okundu | Referans siteler bu tur tekrar denetlenmedi |

Tarayıcı scroll çağrılarının bazısı komut zaman aşımı bildirdi ancak sonraki DOM ve ekran görüntüleri scroll'un gerçekleştiğini gösterdi. Bu araç davranışı site donması veya düşük FPS kanıtı sayılmadı. Tarayıcı uzantısından gelen console hataları uygulamaya atfedilmedi. Fiziksel mobil, reduced-motion'ın canlı görsel sonucu, LCP/INP/CLS, FPS ve bellek bu tur ölçülmedi. Önceki yerel test raporları tarihli kanıtlardır; yeni ölçüm gibi sunulmaz.

## 3. Gerçekte mevcut teknikler

| Teknik | Doğrulama | Değerlendirme |
| --- | --- | --- |
| CSS sticky | `.cm-story-stage { position: sticky; top: 0; height: 100svh }`; farklı scroll konumlarında canlı `top: 0` | Eksik değil, çalışıyor |
| Scroll-linked timeline | `useScroll`, `offset: ['start start','end end']`, `useTransform` | Framer Motion üzerinden var |
| Yerel sahne animasyonları | Heatmap hücrelerinde x/y/opacity; haber adımlarında x/opacity; fiyat yolunda pathLength | Varlıkları, iyi sahne kurgusu anlamına gelmiyor |
| Hero hareketi | Scroll'a bağlı 40 px y, −3 derece rotate; küçük grafikte pathLength | Ölçülü parallax benzeri hareket; sahneler arası aktarım yok |
| Hero çizimi | `CopperSignal.tsx`: 26 kontur, her biri 97 örnek nokta, SVG gradient | Shader veya 3D model değil; üretken 2D SVG |
| Native scroll | Scroll konumu browser'a ait; mevcut landing'de wheel interception/Lenis yok | Korunabilir; akıcılık için değiştirmek zorunlu değil |
| Kinetic typography | Landing başlıklarında split/mask/scroll reveal controller yok | Yer yer uygulanabilir; tüm metne eklenmemeli |
| WebGL/3D/shader | Canlı landing DOM'unda canvas yok; landing kaynaklarında bu renderer'lar yok; package.json'da ilgili ek bağımlılıklar yok | Bu yayının landing'ine atfedilmez |
| Responsive enhancement | ≥1024 px, fine pointer, reduced motion kapalı, Save-Data kapalı, bellek ipucu >4 GB veya bilinmiyor | Ölçülmüş GPU tier sistemi değil |
| Typography | Canlı computed font Geist Sans; kaynakta yerel variable WOFF2 | Font ailesini tekrar değiştirmek ilk öncelik değil |

## 4. Bölüm bölüm değerlendirme

### Hero: Marka öğesi var, anlatısal görevi eksik

Cu ve bakır konturlar doğru bir başlangıç. Başlık okunuyor, açıklama bakır/market/news/forecast kapsamını veriyor, uygulamaya geçiş açık. Ancak kompozisyon hâlâ solda yazı–sağda görsel düzeninde. Konturların biçimi ile alttaki küçük fiyat grafiği arasında anlamlı hareket bağı yok. Scroll başlayınca görsel az miktarda dönüyor ve sayfadan çıkıyor; hikâyede bir daha görünmüyor.

Markanın en ayırt edici parçası yalnızca ilk ekrana ait kalıyor. “The metal. The signal.” ifadesi için gereken görsel dönüşüm henüz yaşanmıyor.

Canlı hero ekran görüntüsü yerel inceleme kaydında tutuldu; PR içine yüklenmedi.

Öneri: Konturun tek bir çizgisini anlatının devam eden öznesi yap. Hero başlığının ölçeğini ve konturun kadrajla ilişkisini daha cesur kur; küçük not, caption ve capability linklerinin rekabetini azalt. İlk ekrandaki ürün cümlesi ve CTA hemen kullanılabilir kalsın.

### Intro: İkinci kez vaat, henüz yeni kanıt değil

“A price is a point…” iyi bir köprü cümlesi. Fakat hero'dan sonra yaklaşık 553 px daha ürün vaadi ve dört capability etiketi geliyor. İlk story bölümü bu viewport'ta sayfanın yaklaşık 1496. pikselinde başlıyor. Bu, otomatik olarak fazla uzun değildir; ancak bu mesafede yeni bir görsel fikir veya ürün kanıtı az.

Öneri: Bağımsız manifesto bloğunu, Cu konturundan market sahnesine dönüşümün kısa başlığına dönüştür. Capability listesini bir kez kullanılan, ilerlemeyi gösteren chapter navigation ile birleştir. Buradaki scroll mesafesi yeni bir ilişki açığa çıkarmalı.

### Sticky research: Sahne sabit, görsel kimlik devam etmiyor

Bu viewport'ta section 2808 px, üç chapter 936'şar px. Sticky pin alanının kendisi doğru çalışıyor. Yerel progress hesabının etkin scroll mesafesi yaklaşık `2808 − 936 = 1872 px`.

`StoryLayer` market için `.18–.32`, news→forecast için `.68–.82` aralığında tüm kartları crossfade ediyor. İlk geçiş yaklaşık 262 px sürüyor. Kullanıcı ortada durduğunda bu, geçici bir video karesi gibi geçip gitmiyor; okunamayan karışım ekranda kalıyor.

Canlı kayıt: `scrollY=1985`, stage top `0`, market opacity `0.419872`, news opacity `0.580128`. İki kartın başlıkları, çerçeveleri, açıklamaları ve dipnotları birbirinin üstünde. [DOM ölçümü](./evidence/art-direction-20260907/transition.json).

Duraklatılmış geçişin ekran görüntüsü yerel inceleme kaydında tutuldu; PR içine yüklenmedi.

**P1:** Full-card crossfade yerine sabit bir ortak çerçeve, hareket eden grafik katmanı ve birbiriyle çakışmayan metin katmanları kullan. Dekoratif çizgiler blend olabilir; iki bağımsız metin aynı koordinatlarda blend olmamalı. Metin çıkışını bitir, sonra yeni metni aç; aradaki kısa aralıkta kalıcı Cu/etiket sahneyi anlaşılır tutar.

Okuma süresi de dengeli değil. Market yerel animasyonu global progress `.17` civarında tamamlıyor, kart fade-out'u `.18`'de başlıyor. Bütün market kompozisyonunun tam parlaklıkta, tam yerleşmiş bekleme mesafesi yalnızca yaklaşık 19 px. News `.5–.68` arasında çok daha geniş bir bekleme alanına sahip. Forecast çizgisi `.97`'de tamamlanıyor, pin bitimine yaklaşık 56 px kalıyor. Bu değerler koddan türetilmiş; fiziksel okuma süresi ölçümü değil. Yeni timeline'da her sahneye bilinçli bir tamamlanmış okuma aralığı ayrılmalı.

Chapter nav tıklanabiliyor fakat aktif chapter stili ve `aria-current` yok. Aynı yerdeki üç etiket kullanıcının nerede olduğunu yeterince göstermiyor. `aria-current="step"` ve ilerlemeye bağlı görünür vurgu eklenebilir; ekran okuyucuya her frame duyuru yapılmamalı.

### Market preview: İlk güçlü ürün görüntüsü, fakat pasif

Renkli hücreler ve HG=F odağı ürünü somutlaştırıyor. Fakat copy “relationships” diyor; görüntü yalnızca günlük değişim ve göreli alan sunuyor. Bu veriler korelasyon kanıtı değildir. Hücreler animasyonla yerleşiyor ancak neden birlikte incelendikleri açılmıyor. Enhanced sahne `aria-hidden`, katmanlar `pointer-events:none`; kullanıcının seçebileceği bir araştırma öğesi yok.

Öneri: İki veya üç bağlam öğesini açık etiketlerle göster: bakır vadeli işlem, üretici hisseleri, makro bağlam. Bağlantı çizgileri “araştırma bağlamı” anlamına gelsin; ölçülmemiş korelasyon veya nedensellik iddiası taşımasın. HG=F düğümü sonraki sahnede devam etsin. İsteğe bağlı “Bu öğe neden burada?” kontrolü aynı açıklamayı hover, focus ve tap ile açabilsin.

### News preview: Görünüşü haber kartı, deneyimi metin listesi

Hipotetik örnek olduğu açıkça yazılıyor; bu korunmalı. Fakat geniş panelde üç paragrafın sırayla belirmesi gerçek news intelligence ürününün araştırma değerini yeterince hissettirmiyor. “Kaynak → sentiment → driver” ilişkisi metinde anlatılıyor; görsel olarak kurulmuş değil.

Öneri: Başlıktaki “supply” ifadesinden ilgili gerekçe satırına bir annotation aç. Kullanıcı kaynak ve yorum ayrımını aynı sahnede görsün. Otomatik olarak anlamı değişen bir sentiment göstergesi yerine örneğin koşullu yorumunu sabit tut. “Read the reasoning” gerçek butonuyla açıklama açılabilir. Bu demo canlı LLM çağrısı gerektirmez.

### Forecast preview: İyi kavram, daha güçlü açıklama fırsatı

Geçmiş, son kapanış, gelecek yol ve aralık ayrılıyor; sentetik olduğu belirtiliyor. Ancak tarih çizgisinin sıfırdan çizilmesi araştırma sorusuna çok az şey ekliyor. Kullanıcı tüm çizginin ortaya çıkmasını beklemek yerine, son gözlemden sonra neden tek bir kesin sonuç gösterilmediğini anlamalı.

Öneri: Geçmiş çizgiyi görünür tut. Son kapanışta bakır çizgi geleceğe uzanırken tek çizginin çevresinde belirsizlik bandı açılsın. Ana 5D sonuç etiketi, günlük T+1…T+5 yolundan açıkça ayrılmış kalsın. Bandın reveal'i yalnızca çizimi açsın; quantile değerlerini değiştirmesin. Pointer yalnızca zaten tanımlı örnek tarihleri incelesin; sahte model/senaryo hesaplaması üretmesin.

### Evidence: Fazla uzun, boş alan ve eksik metrik hissi

Bu viewport'ta yaklaşık 1459 px. Soldaki başlık ve açıklama bittikten sonra sağda rapor kartı ve üç bağlantı devam ediyor; uzun bir sol boşluk oluşuyor. Bütün metriklerin “—” olduğu büyük kart, dürüst olmasına rağmen ürünün henüz tamamlanmadığı hissini kuvvetlendiriyor. Buraya uydurma başarı oranı koymak çözüm değildir.

Öneri: “Forecast passport” adlı kompakt bir kanıt özeti: instrument, horizon, reference close, model version, evaluation availability. Yalnızca var olan alanlar ve gerçek durumlar gösterilsin. Üç evidence linkini sonraki tam genişlik satırına al; iki kolonun dengesini düzelt. Rapor yoksa bunu kısa ve açık belirt, kullanıcıyı boş sonuç yerine mevcut model metriklerine yönlendiren ikincil bağlantı sun.

### Kapanış: Net CTA, görsel sonuç yok

CTA görünür ve dashboard'a geçiş çalışıyor. Ancak hero'daki motif veya araştırma sırasında kurulan ilişki burada geri dönmüyor. Yeni sahnede aynı bakır çizgi sade bir workspace çerçevesine yerleşebilir; kullanıcı ürünün içinde çalışmaya geçeceğini görür. Bu bağ görsel olarak CTA'dan önce kurulmalı; tıklama sonrası zorunlu film oynatılmamalı.

## 5. Dashboard: Güven ve bilgi hiyerarşisi

Canlı dashboard açıldı, veriler yüklendi, `90 closes` kontrolü `aria-pressed=true` oldu. Görsel dil landing ile uyumlu. Bununla birlikte şu işler motion'dan daha yüksek öncelikli:

| Öncelik | Gözlem / kod kanıtı | Önerilen çözüm |
| --- | --- | --- |
| P1 | Aynı TFT kartında unusual-output uyarısı ve yeşil LOW RISK | Model kalite durumu ile piyasa/risk sınıflandırmasını ayrı başlıklarla açıkla. Riskin ufku ve tanımı veri sözleşmesinden doğrulansın; yeşil onay işareti genel güvenilirlik onayı gibi kullanılmasın |
| P1 | Quote “As of” alanı `lastLiveUpdateAt`; fetch başarılı olunca `new Date()` atanıyor | Kaynak fiyat zamanı ile son kontrol zamanını ayır. Kaynak zamanı yoksa “Last checked” de; yeni fetch eski fiyatı yeni gözlem gibi göstermesin |
| P1 | `quoteDelta=quotePrice-latestHistoryPrice`; fallback'te iki değer aynı. Sıfır `>=0` ile yeşil | Gerçek sıfırı nötr göster. Fallback'in kendi kendisiyle farkını piyasa değişimi gibi sunma. Delta referansını “vs last stored close” gibi doğru etiketle |
| P1 | TFT ana kartında negatif 5D görünüm; Neural Analysis metninde pozitif ayrı tahmin anlatımı | Bunun model hatası olduğu varsayılmamalı. Kaynak model, horizon ve reference date her çıktı yanında açık olsun; yorumun hangi analysis snapshot'ını açıkladığı sözleşmeyle doğrulansın |
| P1 | Validation canlı sayfası “No backtest report available yet” gösteriyor | Eksik rapor UI'sı doğru; ürün kanıtı gerçekten eksik. Model metadata metrikleri ile bağımsız walk-forward raporunu eşitleme |
| P2 | AI commentary tek uzun metin; kaynakta bir `<p>` içinde gösteriliyor | Kaynak metni koruyarak özet/gerekçe/risk bölümlerini yapısal veri varsa ayır. LLM metninden kanıtlanmamış sayıları yeniden türetme |
| P2 | Kısa etiketler ve çok sayıda küçük açıklama üç dar kolonda yarışıyor | Ana tahmin ve price chart için daha belirgin hiyerarşi; ikincil eğitim açıklamalarını disclosure'a al. Haber panelinin iç scroll'u ve ana scroll birlikte klavye/touch testinden geçsin |

Models sayfasında Quality gate: Passed yanında Stability Warnings görülmesi tek başına bug değildir; farklı kabul boyutları olabilir. Görsel sunum bunu açıklamalı. System sayfasında healthy, bir model ve yaklaşık 21 saatlik snapshot yaşı görüldü; servis sağlığı veri yeniliğiyle aynı şey değildir. Bunlar inceleme anının UI çıktılarıdır, finansal doğruluk denetimi değildir.

## 6. Art direction: “Bakırın izini sür” / Follow the copper signal

**Tasarım tezi:** CopperMind, tek bir fiyat noktasını daha geniş bir araştırma bağlamında okuma aracıdır. Görsel dünya, metalin fiziksel konturlarından ölçülebilir bir araştırma yüzeyine geçer. Bakır rengi anlatının bağlantı öğesi, mavi belirsizlik, yeşil/kırmızı yalnızca verinin yönü için kullanılır.

Bu önerinin cesur yanı yeni bir arka plan efekti değil, sayfanın bütün kompozisyonunu aynı öznenin dönüşümüne bağlamasıdır. Düz feature kartlarının yerini “aynı şeyi başka bir katmanıyla görme” deneyimi alır.

### Özgün görsel dil

- Hero'da büyük, kadrajı kısmen aşan Cu konturu. Teknik çizim gibi ince ölçü/annotation işaretleri; metin üzerinde parlayan efekt yok.
- Konturlardan bir bakır çizgi ayrılır. Hero'da markanın izi, market'te seçilen enstrümanın işareti, news'te gerekçe annotation'ı, forecast'te son gözlem referansı olur.
- Sahne genişlediğinde koyu panelin içinde panel oluşturmak yerine grafiğin kendisi kompozisyonun ana yüzeyi olur. Kenar çerçevesi yalnızca bilgi gruplamak gerektiğinde görünür.
- Tipografide Geist korunur; hero'nun kısa vurgusu daha büyük ölçek ve kontrollü mask reveal alabilir. Ana ürün cümlesi başlangıçta okunur. Tablolar, sayılar ve body copy hareket etmez.
- Bakır çizgi bilimsel veri sanılmamalı: kavramsal bağlantı ile eksen/seri çizgisinin stilleri ve etiketleri ayrılır. Veri koordinatlarına girildiğinde ortak veri ölçeği kullanılır.

### Storyboard ve teknik karşılık

| An | Görsel olay | Kullanıcının anladığı | Uygulama |
| --- | --- | --- | --- |
| 0 — Metal | Cu konturu ve güçlü ürün cümlesi | Bakır odaklı araştırma ürünü | Mevcut SVG, yerel font, içerik hemen görünür |
| 1 — Market | Tek kontur çizgisi ayrılır; HG=F çevresindeki bağlam alanları yerleşir | Bir fiyat tek başına okunmaz | Ortak SVG koordinat sistemi, CSS sticky, transform; istatistiksel ilişki iddiası yok |
| 2 — Context | HG=F işareti kalır; haber kaynağı ve ilgili gerekçe alanı açılır | Kaynak ile yorum farklıdır | Ayrı grafik/metin katmanları, ince connector ve disclosure |
| 3 — Possibilities | Son gözlem sabitlenir; gelecek yol ve aralık aynı referanstan açılır | Tahmin belirsizlik içerir | Gerçekçi sabit fixture, clip/reveal; veri değerleri sabit |
| 4 — Evidence | Sahne sadeleşir; horizon/reference/model/availability bilgisi öne çıkar | Sonuç sorgulanabilir olmalıdır | Normal flow'da kompakt passport; ek uzun pin yok |
| 5 — Workspace | Çizgi workspace panelinin sınırına oturur; CTA | Araştırmaya devam edebilirim | Hafif DOM/SVG bitişi, doğrudan link |

İlk prototip yalnızca **market → context aktarımını** üretmeli. Bağlantı orada inandırıcı değilse bütün sayfayı aynı sistemle büyütmek gereksizdir. İki versiyonun side-by-side tasarım karşılaştırması, daha fazla efekt eklemekten daha fazla öğrenme sağlar.

## 7. Uygulanabilir mimari ve timing

Mevcut React/Vite/Framer Motion korunabilir. GSAP'e geçiş için bu incelemede bir gereklilik bulunmadı. Yeni öneri dosyaları henüz oluşturulmuş componentler değildir:

| Component / modül | Girdi | Sorumluluk ve çıktı |
| --- | --- | --- |
| `ResearchStory` | Chapter içeriği, policy | Semantic section/anchor ve static fallback; mevcut gövde refactor edilir |
| `useStoryTimeline` | Section ref, ölçülen chapter anchor'ları | `progress`, `chapterIndex`, yerel arrive/read/exit progress; frame başına React state yok |
| `SignalStage` | Timeline MotionValue'ları | Tek sticky sahne, kalıcı arka plan ve Cu işareti; her chapter için kart mount döngüsü yok |
| `SignalTrace` | Normalized stage progress | SVG connection, transform ve reveal; sahte fiyat/korelasyon üretmez |
| `StoryCaption` | Aktif chapter, metin geçiş aralığı | Başlık/metinlerin birbirinin üstüne binmesini önler |
| `ContextDisclosure` | Fixture veya onaylı snapshot | Kaynak/gerekçe açıklaması; pointer, keyboard, touch aynı bilgiye erişir |
| `ForecastPassport` | Instrument/horizon/reference/model/availability | Kanıt özetini kurar; eksik alanı açıkça ifade eder |
| `StoryChapterNav` | chapterIndex | Link hedefleri korunur, aktif adım görsel ve semantic olarak belirtilir |
| `useExperiencePolicy` | Motion tercihleri ve kullanılabilir alan | Pahalı efekt tercihi ile basit sticky tercihi ayrılır |

### Timeline prensibi

`p = clamp((scrollY − sectionStart) / (sectionHeight − viewportHeight), 0, 1)` mevcut mekanizmanın kavramsal karşılığıdır. Resize, font yüklenmesi ve kısa viewport'ta gerçek anchor konumları yeniden ölçülmeli. Her frame `getBoundingClientRect` okuma–style yazma döngüsü kurulmamalı. Kullanılan offset/MotionValue yaklaşımı için [Motion useScroll dokümantasyonu](https://motion.dev/docs/react-use-scroll).

Her chapter için başlangıç taslağı: yerel progress `0–.15` geliş, `.15–.75` tamamlanmış okuma, `.75–1` aktarım. Bunlar final sabitler değil, prototipte kalibre edilecek oranlardır. Yeni scroll mesafesi “ne kadar film oynatacağız” sorusundan değil, okunacak anlam ve mobil karşılığından türetilmeli. İlk hedef mevcut yaklaşık 3 viewport story uzunluğunu büyütmemek.

Bir aktarımın kendi `t` progress'inde eski caption `.0–.4` arası çıkar; yenisi `.6–1` arası girer. Ortada Cu işareti ve bölüm çerçevesi görünür kalır. Grafik eşleşmesi devam edebilir; uzun metin blokları aynı yerde crossfade edilmez. Reverse scroll deterministik olarak aynı görüntüyü geri üretmeli. Herhangi bir noktada durmak kabul edilebilir bir kompozisyon bırakmalı.

React state yalnızca chapter değişimi veya disclosure click gibi semantic olaylarda güncellenir. Progress CSS/SVG'ye MotionValue ile gider. Geometry path verisi render başına yeniden üretilmez. `transform`/`opacity` ucuz başlangıçtır; SVG stroke/mask çizimleri paint üretebilir, ücretsiz GPU animasyonu olarak varsayılmaz. Her path'i her frame morph etmek yerine az sayıda hareketli bağlantı, sabit konturlar ve wrapper transform tercih edilir.

### Native scroll ve smoothing

Native scroll bu art direction'ı uygulamaya engel değil. Önce doğrudan scroll-linked sürüm denenmeli. Hafif damping yalnızca dekoratif bakır çizgide denenebilir; caption, focus, anchor ve chapter durumu native konumdan gecikmemeli. Aynı progress'e birden fazla spring/lerp katmanı eklenmemeli. Sayfaya sırf premium his amacıyla ikinci scroll otoritesi eklemek bu taslakta önerilmiyor.

### Mobil ve erişilebilirlik

Mobilde ekran sabitleyerek masaüstü filmi küçültmek yerine aynı bakır işaretiyle üç kısa editorial scene göster. İsteğe bağlı bölüm linkleri scroll'u doğal hedefe taşısın; hareket bilgiyi saklamasın. Reduced-motion'da son kompozisyon ve tüm içerik açık olmalı. Küçük viewport, büyütülmüş font veya sahne yüksekliği kullanılabilir alana sığmıyorsa sticky yerine normal akışa geçilmeli.

Mevcut bellek `>4 GB` koşulu pahalı renderer'ı korumak için anlaşılabilir bir ipucu; yalnızca altı DOM tile ve birkaç SVG'lik sahneyi tamamen statik yapmanın faydası ayrıca ölçülmeli. Basit sticky/okunabilir sahne ile pahalı görsel kalite aynı boolean'a bağlanmamalı. Touch pointer'a sahip olmak tek başına bütün anlamlı motion'u kapatmak için yeterli gerekçe değildir.

Enhanced sahne şu an `aria-hidden`. İnteraktif kontroller eklenecekse bu wrapper içine erişilebilir button koyup görünmez bırakma. Semantic metin ve kontrol katmanı gizlenmeyen DOM'da; sadece yinelenen dekoratif çizim `aria-hidden` olmalı. Focus'u scroll nedeniyle taşıma, her frame canlı duyuru yapma. Kısa ekranda nav ve caption dahil gerçek stage yüksekliği ölçülmeli.

## 8. WebGL ve yaratıcı risk kararı

Marka hissi de meşru bir tasarım faydasıdır; 3D'nin mutlaka bir sayısal grafik açıklaması olması gerekmez. Ancak “Cu yüzeyi materyal olarak ne kazandırıyor?” sorusuna somut görsel yanıt gerekir.

İkinci bir prototip adayı: hero'daki konturu düşük poligonlu bakır kabartma yüzeye dönüştürmek; scroll sırasında yüzeyin çizgisel araştırma görünümüne düzleşmesi. Tek seferlik bu malzeme değişimi marka hafızasına katkı sağlayabilir. Mevcut SVG storyboard ile karşılaştırılmadan production kararı verilmemeli.

İlk WebGL spike yapılırsa: yalnızca hero'da dinamik import, tek renderer, tek yüzey, mümkünse prebaked environment, shadow map/post-processing yok, sınırlı DPR; hidden/offscreen durumda render durur. Pointer koordinatı material uniform'una sınırlı ışık kayması olarak gider, metin etkilenmez. Shader derleme ve cihaz ısınması gerçek cihazda ölçülür. Static SVG eşdeğeri hazır olur. Dashboard'a geçerken context/geometry/material/texture/listener cleanup test edilir. Bu belge WebGL'i sıradaki zorunlu faz yapmaz.

Particles, devamlı parlayan sayılar, cursor takip eden tüm arka plan, scroll velocity blur ve otomatik kayan ticker şu anki anlatı problemini çözmüyor. Önce tek bir imza anı güçlü yapılmalı; bütün ekranlarda efekt yoğunluğu eşitlenmemeli.

## 9. Uygulama sırası ve kabul

| İş paketi | Değişiklik | Tamamlanma ölçütü |
| --- | --- | --- |
| A — Anlam ve okunabilirlik | Crossfade metin çakışması, aktif chapter, quote zamanı/delta, risk etiketlerinin ayrımı | Geçiş ortasında iki caption üst üste değil; quote kaynak/refresh ayrımı doğru; finansal hesaplama davranışı korunmuş |
| B — İmza prototipi | Cu → market → context bağlantısı, daha cesur hero/intro kompozisyonu | Statik üç ana kare güçlü; scroll durdurma/geri sarma ve doğrudan anchor aynı anlamı koruyor |
| C — Hikâyeyi tamamlama | Forecast aralığı reveal'i, kompakt passport, evidence düzeni ve CTA devamlılığı | Kullanıcı gözlem–yorum–tahmin–kanıt ayrımını yapabiliyor; boş rapor bir başarı iddiasına dönüşmüyor |
| D — Cihaz ve yayın kabulü | Mobil art direction, reduced motion, Safari/Chrome, gerçek API ve aynı cihaz performans karşılaştırması | Yeni yayın kimliğiyle kanıt; önceki rapor sayıları yeniden kullanılarak kabul verilmez |

Korunacaklar: route/deep link sözleşmesi, gerçek finans API sözleşmeleri, heatmap geometri/LOD/zoom davranışı, chart hesapları, ortak token'lar ve örnek veri açıklığı. Refactor odağı `Hero`, `CopperSignal`, `ResearchStory`, `Previews`, `landing.css`; dashboard değişiklikleri ayrı küçük teslimlerle yapılmalı.

Mevcut kontrol script'inde toplam initial JS/CSS ve dashboard bütçeleri yeniden çalıştırılmalı. Önceki faz 4 raporunun 117583 byte initial JS gzip, 12113 byte CSS gzip ve 263662 byte dashboard JS değerleri yalnızca tarihli referanstır; bu tur canlı transfer ölçümü değildir. Yeni art-direction değişikliği için aynı build/cihaz altında delta raporla. Başlangıç deneysel hedef: yeni runtime bağımlılığı yok, landing artışı ≤15 KB gzip; kalıcı bir sınır olarak kabul edilmeden önce prototipte değerlendir.

Performance kabulünde 60 Hz için 16.7 ms toplam frame aralığı temel alınır. Sadece ortalama FPS değil p95 frame süreleri, kaçırılan frame dağılımı, long task'lar ve paint maliyeti incelenir. Gerçek kullanıcı p75 LCP ≤2.5 s, INP ≤200 ms, CLS ≤0.1 hedefleri mevcut release yaklaşımıyla doğrulanır; bu doküman hedefleri sağlandı diye raporlamaz. Eşikler ve mobil/desktop p75 ayrımı: [Google Web Vitals](https://web.dev/articles/vitals).

Tasarım kabulü için üç kullanıcıya kısa görev: ürün ne yapıyor, hangi görüntü örnek veri, tahmin neden aralık içeriyor, gerçek kanıta nereden gidilir? Önce yanıtları mevcut sayfada, sonra prototipte karşılaştır. Üç kişi istatistiksel başarı kanıtı değil, yanlış anlama tespiti için nitel başlangıçtır. Geçişi daha güzel bulmak ile ürünü daha iyi anlamayı ayrı sor.

## 10. Kaynaklar ve belge ilişkileri

- [Canlı landing](https://terra-rara.vercel.app/) ve [Validation](https://terra-rara.vercel.app/validation): bu incelemedeki canlı UI gözlemleri.
- [DOCX teknik rehberi](../design/immersive-web-case-study-tr.docx): terminoloji, finansal kullanım ilkeleri ve minimum runtime prensibi.
- [Frontend engineering guide](../design/frontend-engineering-guide.md): uygulamanın ortak tasarım/motion kuralları.
- [Önceki roadmap](../implementation/frontend-experience-roadmap.md): tamamlanan fazların tarihli kapsamı.
- [Faz 4 raporu](./frontend-phase-four-completion-20260906.md) ve [yayın kabulü](./frontend-release-acceptance-20260906.md): önceki kontrollerin kapsamı ve kalan saha kabulü.
- Kaynak dosyaları: `frontend/src/features/landing/{Hero,CopperSignal,ResearchStory,Previews,useExperiencePolicy}.tsx/ts`, `landing.css`, `frontend/src/design/tokens.css`, `frontend/src/pages/OverviewPage.tsx`.

Yol haritasındaki “geliştirildi” durumu önceki kapsamın tamamlanmasıdır. Bu yeni art-direction değerlendirmesi o kapsamı geriye dönük değiştirmez; kullanıcının daha özgün ve bütünlüklü bir deneyim hedefi için yeni öneri oluşturur.
