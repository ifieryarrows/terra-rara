# CopperMind Terra Rara — Kesintisiz anlatı araştırması ve uygulama planı

Araştırma: 7–8 Eylül 2026. Durum: **tasarım/engineering önerisi; uygulama yapılmadı**.

## 1. Karar

Eksik olan şey yeni bir animasyon kütüphanesi değil, sahneler arasında korunacak bir görsel nesne ve onun değişmeyen anlamıdır. Öneri: **Copper, Under Examination — Bakırı izlemekten sinyali sınamaya**. Başlangıçta bakırın konturu olan çizgi, araştırma boyunca referans izine dönüşür; son sahnede gösterilen iddiaların denetlenebilirliğini görünür kılar.

Kullanıcı metalden piyasaya, kaynaklardan olasılıklara ve oradan kanıta ilerler. Her adım aynı dünyada gerçekleşir. Bakır atmosferi ve iz devam eder; veri katmanları aynı görsel koordinat sistemini paylaşır. Finansal anlamı farklı veriler birbirine matematiksel olarak dönüşmüş gibi sunulmaz: haber sentiment'i fiyat değildir, belirsizlik bandı ölçülmüş model başarısı değildir.

İlk uygulanacak sürüm React/Vite + mevcut Framer Motion + native scroll + SVG/CSS ile yapılmalı. Yeni GSAP/Lenis/Three.js paketi başlangıç gereksinimi değildir. Cesur kompozisyon, büyük League Gothic kelimeler, occlusion, ölçek değişimi ve kontrollü sessizlik bu stack ile mümkündür. WebGL, aynı atmosferin CSS/SVG karşılığı ölçülebilir biçimde yetersiz kalırsa ayrı deney olur.

## 2. Sürüm, yöntem ve kanıt sınırları

| Kaynak | Doğrulanan durum | Bu raporda nasıl kullanıldı? |
|---|---|---|
| `origin/main` | `321f8244e04f81c1fa8dadbc8aa332f018d265c5`; son görsel commit `424769f` | Canlı sayfanın component/kod karşılığı |
| [PR #6](https://github.com/ifieryarrows/terra-rara/pull/6) | Açık, draft, merge edilmemiş; `eac823c109ffff79108f2c28bb86481aab97a18d` | Ayrı kaynak incelemesi; canlıymış gibi ele alınmadı |
| [Canlı landing](https://terra-rara.vercel.app/) | “Read the market. See the structure.”; `index-D3qMiocK.js` | Desktop DOM, computed style, screenshot, anchor ve klavye ile bölüm ilerleyişi |
| Önceki DOCX | `docs/design/immersive-web-case-study-tr.docx` | Terminoloji, progressive enhancement ve geçmiş bundle bulguları yeniden okundu |
| Mevcut browser | 1363×936 CSS px | Fiziksel cihaz/FPS ölçümü yerine geçmez |

Kanıt etiketleri: **Canlı** = bu oturumdaki DOM/görünür sonuç; **Kod** = belirtilen committeki kaynak; **Arşiv** = önceki araştırmadan saklanan bundle; **Üretici** = işi yapan ekibin açıklaması; **Öneri** = Coppermind için tasarlanan sistem.

Mobil viewport/emülasyon ve Performance/Network kayıt API'si bu tarayıcı yüzeyinde sunulmadı. Gerçek mobil kullanım, HAR waterfall, CPU/GPU trace, INP, LCP ve FPS ölçülmedi. Mobil bulgular CSS/policy incelemesidir. Lusion ve Orage'da WebGL context hataları loglarda doğrulandı; görselleri loader/boş yüzeyde kaldı. Lando hero da bu oturumda tamamlanmış görsel sahne sunmadı. Bu sitelerin başarılı GPU hareketleri deneyimlenmiş gibi anlatılmıyor. Astra içindeki bir oyun demosunun WebGL hatası, Astra ana arka planının WebGL kullandığını kanıtlamaz.

Lando'nun güncel DOM script URL'si eski araştırmanın kaydettiği isimle uyumlu; ancak güncel bundle indirme denemesi geçerli JS üretmedi. Dolayısıyla arşiv kodundaki süreler bugün değişmemiş kabul edilmedi. Ekran görüntüleri GitHub'a eklenmedi.

## 3. Mevcut deneyimin neden parçalı hissettirdiği

### 3.1 Başlangıçtan sona canlı bölüm anatomisi

Aşağıdaki geometriler aynı desktop oturumundandır; font, viewport ve içerikle değişir.

| Bölüm | Başlangıç / yükseklik | Davranış ve teşhis |
|---|---:|---|
| Navigation | 0 / yaklaşık 86 px | Ürün adı ve doğrudan dashboard erişimi açık. Korunmalı. |
| Hero | 86 / 857 px | Sol mesaj, sağ Cu konturları. Tek nesne etrafında hiyerarşi kuruluyor. Alt küçük forecast aynı viewport'a üçüncü hikâye ekliyor. |
| Research intro | 943 / 553 px | Ayrı büyük başlık, paragraf ve dört etiket; hero ile asıl hareketli sahne arasına ikinci açıklama ekranı koyuyor. |
| Market–News–Forecast | 1496 / 2808 px | Sol üç adet viewport yüksekliğinde yazı; sağ `position:sticky; top:0` sahne. Sticky çalışıyor. |
| Evidence | 4304 / 1459 px | Sağ rapor okuyucu ve bağlantılar uzun; soldaki kısa metnin altında büyük boşluk. Boşluk anlatısal değil, grid dengesizliği. |
| CTA | 5762 / 599 px | Hafif bakır diffusion ilk kez belirgin. Ortalanmış mesaj ve dashboard linki; önceki nesne buraya taşınmıyor. |
| Footer | yaklaşık 6362 sonrası | Kısa kimlik ve doğrudan uygulama linki. Korunabilir. |

### 3.2 Cu neden daha bütünsel?

Cu merkezli konturların ortak kökeni, aynı renk ailesi ve çevresine konan kısa açıklamalar tek bir odak yaratıyor. Kullanıcı önce nesneyi görüyor, sonra ilişkileri okuyor. Negatif alanın bir merkezi var. Ancak sonraki bölümde bu nesne sayfadan çıkıyor: `Hero.EnhancedSignal` kendi `useScroll` hedefini kullanıyor; `ResearchStory` ayrı progress hesaplıyor. Geometrik veya koordinat düzeyinde bir handoff yok.

Canlı ana dalda hero dönüşü 0→−3 derece, y hareketi 0→40 px. Bu, gerçek kamera perspektifi değil, DOM düzleminin 2D rotasyonu. PR #6 bunu −5 derece/64 px'e taşıyor; hareketin anlamını değiştirmiyor. Daha büyük tilt tek başına kopukluğu çözmez.

### 3.3 Gerçek bug, eksik davranış ve tasarım tercihini ayırma

| Bulgu | Kanıt | Sonuç / düzeltme yönü |
|---|---|---|
| Sticky yok sanılması | Canlı sahne top=0 ve computed position=sticky | Yeni pin paketi eklemek gereksiz; sahne içeriğini ve handoff'u düzelt |
| Kart metinlerinin üst üste gelmesi | Ana dal `StoryLayer` Market çıkış ve News girişini aynı `.18–.32` aralığına bağlıyor | p=.25'te iki opaklık da .5. Gerçek bir kompozisyon hatası; metinler crossfade edilmemeli |
| Anchor ani atlıyor | Canlı HTML scroll-behavior `auto`; News tıklaması `#news`, y=2320; kaynak plain href | Kısa native smooth + hedef okuma state'i + history/focus sözleşmesi |
| Navigation ve sahne ilerleyişi eşleşmiyor | Hedef chapter'a 7rem scroll padding uygulanıyor, timeline ayrı section üzerinden normalize | Nav hedefi article başlangıcından değil tamamlanmış okuma anından türetilmeli |
| Kartların her sahneyi ayırması | `Previews.tsx` ayrı figure, yüzey, border ve başlıklar; `cm-evidence` border-top | Border'ları azaltmak yetmez; ortak trace ve katman sahipliği kurulmalı |
| Global atmosfer eksik | Bakır radial gradient esasen `cm-enter` üzerinde | Tüm landing'i kapsayan ortam; metin arkasında sabit sakin ink alanı |
| Evidence kanıt göstermiyor | Canlı `—` metrikleri ve okuma rehberi; `/validation` boş rapor durumu | Placeholder grid'i başarı payoff'u gibi kullanma |
| FOLLOW THE COPPER SIGNAL canlıda yok | Canlı bölüm adı “THE CONNECTED VIEW”; söz konusu ifade PR #6'da | Kullanıcının taslak değerlendirmesini canlı bug diye etiketleme |

### 3.4 PR #6'nın kendi eksikleri

`SignalStage.tsx` ortak SVG içindeki Cu işaretini Market/News/Forecast boyunca koruyor; non-overlap metinler olumlu. Ancak hero, kısa research intro, signal stage, forecast passport ve CTA ayrı nesne yaşam döngülerinde. Aynı sembolün tekrarı, nesnenin gerçekten el değiştirmesi değil.

`ResearchStory.tsx` içindeki `focusWithin`, readings alanına herhangi bir focus geldiğinde `readingProgress` ve chapter güncellemelerini durduruyor. Bu klavye focus'unu korumak için yazılmış; mouse ile details açıp sayfayı kaydırınca da stage'in eski yerde kalması mümkündür. **Koddan türetilmiş test edilecek risk**, canlıda üretilmiş bug değildir. Öneri: uzun açıklamayı pin dışında normal flow'da tut; kullanıcının açık okuması sürerken dekoratif timeline'ı durdurma. Focus'un bulunduğu düğümü unmount etme; odak taşıyan disclosure'ı ayrı, kalıcı bir içerik alanında tut.

`cm-scene-readings` için max-height + overflow-y:auto, viewport içinde ikinci scroll alanı oluşturuyor. Kısa desktop'ta grafik ve yazı sığdırma çabası mimarinin sınırını gösteriyor. Sahnedeki anlatımı kısaltıp ayrıntıyı dışarı taşımak bu sorunu daha iyi çözer.

PR #6'daki `ForecastPassport` gerçek validation sunmuyor. Onu yeni Evidence'ın merkezine koymak aynı problemi daha küçük bir kartta tekrarlar.

## 4. Referans araştırması: aktarılacak prensipler

### Astra — ortak ortam, farklı içerik yoğunlukları

**Canlı:** Next asset yolları, fixed `data-astra-backdrop`, statik poster fallback, üç scroll cue ve klavye/drag etiketli kontrol bulundu; ana DOM'da canvas sayısı sıfırdı. Bunlar tek arka plan üzerinde içerik akışını doğrular; shader kütüphanesini doğrulamaz. Metin, karşılaştırmalar ve medya örnekleri ortamdan ayrı içerik katmanlarıdır. [Kaynak](https://openai.com/index/gpt-6-astra/).

**Coppermind yorumu:** boşluğu bir öğenin yokluğu olarak değil, sinyalin seçilmesini beklediğimiz aralık olarak kullan. Koyu ve sakin okuma alanları, büyük tek kelimeler ve birkaç veri anı arasında ritim kur. Yıldız estetiğini taşımak yerine bakır yüzey/ışık yayılımı üret. Finansal tabloya gelindiğinde ortam sakinleşsin.

### Lando — metin bir sahne olayı, nesne bir devamlılık taşıyıcısı

**Canlı:** Webflow CSS/JS yolları, çok sayıda canvas, `split-text="lines"`, `.line`, `high-line-reveal` öğeleri ve `clip-path:inset(...)` bulundu. Görsel display metni Brier, yardımcı metin Mona Sans; erişilebilir heading kopyaları ve `aria-hidden` görsel başlıklar var. League Gothic kullanılmıyor. [Site](https://landonorris.com/).

**Arşiv kod:** GSAP SplitText satırlarını üreten kod, yönlü clip açılması ve üst örtünün `scaleX` ile çekilmesi birlikte çalışıyor. Kaydedilmiş ayarlarda iki hareket .6 saniye, satır farkı .15 saniye; trigger `top 90%`, bir kez oynatma. Bu particular reveal scroll-triggered'dır; tüm başlıkları scroll scrub diye adlandırmak yanlış. Kask sahnesinin paused timeline'ı ise merkezi helmet progress tarafından sürülüyor. Arşivde Three.js, GSAP/ScrollTrigger, Lenis ve model/texture loader izleri var; güncel çalışmanın tamamı tekrar bundle düzeyinde doğrulanamadı.

**Coppermind yorumu:** kısa kelimeleri bakır izinin geçtiği yüzeyden çıkar. Bütün paragrafı karakterlere bölme. Ortak nesnenin sahneler arasında konum/ölçek değiştirerek sürmesi, herhangi bir kask estetiğinden daha değerlidir.

### OFF+BRAND — deneyimi tekrar çalıştırmadan gösterme

**Canlı:** kask dönüşü, ana case video, 404, takvim ve Rive örnekleri MP4 video öğeleri olarak yüklü; bu anda canvas yok. Arşivde ajansın ayrı Three.js orb kodu bulunması, case içindeki her hareketin canlı 3D olduğu anlamına gelmez. [Üretici case study](https://www.itsoffbrand.com/our-work/lando-norris).

**Coppermind yorumu:** dashboard'un pahalı interactive heatmap'ini landing'de ikinci kez mount etmek zorunda değiliz. Kontrollü, açıkça etiketlenmiş veri snapshot'ı üzerinden küçük SVG preview daha tutarlı olur. Ekran kaydı alternatifinde poster, oynatma kontrolü ve lazy medya gerekir; video scrubbing mobil seek maliyeti nedeniyle ilk tercih değil.

### Orage — tek bir sistemin farklı görsel yoğunlukları

**Üretici:** Beaucoup, grid'de shader/canvas masking, görsel kart ve yoğun liste modları; WordPress, GSAP, Taxi.js, Lenis, Three.js açıklıyor. **Canlı:** WordPress theme bundle yolu ve Three renderer context hatası doğrulandı; başarılı grid etkileşimi görülmedi. [Üretici tarafından yazılan case](https://www.awwwards.com/case-study-orage.html), [site](https://orage.studio/).

**Coppermind yorumu:** landing ve dashboard aynı materyalin iki yoğunluğu olmalı. Etkileşimi sistemin konusu belirlesin. Boot terminali, rastgele glitch, draggable navigation ve sahte teknik durum metinleri Coppermind'e aktarılmamalı. Heatmap veri yapısını tile estetiği için bozma.

### Lusion — bir nesne etrafında bütün marka dünyası

**Üretici:** Lusion kendi çalışmalarında tasarım, 3D, motion ve yazılımı projeye özel bir sistem olarak ele aldığını; Oryzo'da tek nesne etrafında bütün lansman dünyası kurduğunu anlatıyor. **Canlı:** `_astro/hoisted...js` ve `THREE.WebGLRenderer` hata kaydı var; loader 000'da kaldı. Astro asset izi güçlü framework göstergesi; Three renderer doğrudan hata kanıtı. Başarılı pointer fiziği gözlemlendi iddiası yok. [Birinci el anlatı](https://tympanus.net/codrops/2026/04/13/lusion-where-digital-craft-meets-ambitious-experimentation/), [site](https://lusion.co/).

**Coppermind yorumu:** Cu'nun hero dekorasyonu olmaktan çıkıp bütün anlatının maddesi olması. Aynı zamanda loader'a bağımlı tasarımın fallback gereksinimini ciddiye almak.

### The Pudding — okunabilirliği kaybetmeden grafiği elde tutma

**Birinci el rehber ve canlı demo DOM:** adımlar, ortadaki trigger, tutulan grafik ve tekrar normal flow'a dönüş; kaynak örneğinde grafik yalnızca adım değişince güncelleniyor. Güncel Scrollama anlatısı IntersectionObserver kullanıyor. Tarihî rehberdeki ScrollMagic/jQuery tavsiyeleri yeni stack tercihi olarak alınmadı. [Yöntem](https://pudding.cool/process/how-to-implement-scrollytelling/), [sticky yaklaşımı](https://pudding.cool/process/scrollytelling-sticky/), [Scrollama](https://pudding.cool/process/introducing-scrollama/).

**Coppermind yorumu:** Evidence okurun anlayacağı bir karşılaştırma olmalı. Büyük bir grafik aynı yerde dururken yalnızca değerlendirme sorusu değişsin. Güzel bir hareketten sonra kullanıcı hâlâ eksenleri, dönemi ve karşılaştırılan şeyi bilmelidir.

## 5. Yeni art direction: Copper, Under Examination

### Görsel gramer

Üç temel katman: derin ink ortam, bakır kontur/iz, yüzeyde net ölçüm. İlkinde soyutluk yüksek; sonuna doğru ölçülebilirlik artar. Ortam sıcaklığı yükseldi diye piyasa yükseliyor anlamı oluşmamalı. Pozitif/negatif veri renkleri atmosferden ayrı semantik token'larda kalır.

League Gothic yalnızca büyük display kelimelerde: COPPER, CONTEXT, RANGE, EVIDENCE. Geist body, navigation ve finansal değerlerde korunur; sayılarda tabular numerals. League Gothic açık kaynak/OFL; dağıtılan dosyanın lisansı repo asset'ine eşlik etmeli. Değişken font ekseni dosya metadata'sından kontrol edilmeden animasyon planlanmaz. [Font üreticisi](https://www.theleagueofmoveabletype.com/league-gothic).

Temel kural: her sahnede bir ana cümle, bir görsel tez, isteğe bağlı bir ayrıntı. Küçük legal/provenance metni hariç aynı viewport'ta yaklaşık 35–55 kelime hedefi. Bu editoryal hedef; erişilebilir açıklamaları silmek için limit değildir.

Trace bir “her şeyi bilen AI kablosu” olmayacak. Düğüm seçer, kaynak işaretler, tahminin kesim tarihini tutar ve Evidence'ta karşılaştırma cetveline dönüşür. Sonunda kullanıcıya bırakılan bir inceleme çizgisidir.

### Sahne storyboard'u

Aşağıdaki p değerleri başlangıç koreografi taslağıdır. Tasarım testinde değişebilir; tek kalıcı scene manifest'te tutulur. Desktop sahne aralığı yaklaşık 650–800svh; içerik ve erişilebilirlikten bağımsız devasa boş pin alanı yaratılmaz.

| Sahne / p | Giriş ve progression | Çıkış ve bir sonraki sahneye devir | Teknik / component / maliyet |
|---|---|---|---|
| **Cu / 0–.14** | Hero'nun Cu konturu korunur. Büyük COPPER arka düzlemde; ön kontur bir kısmını örter. 0–.05 hazır okunur. .05–.11'de konturlar açılır, tek bakır iz öne çıkar. | Nesneyi sadece yana yatırmak yerine ölçekle yaklaş; seçilen kontur alt-sağdan açılarak ortak iz olur. Diğer konturlar uzaklaşır. | `Hero` + `CopperTrace` + `SpatialHeading`; SVG grup transform, en çok 24–32 kontur. Mask alanı sınırlı. |
| **Turning point / .14–.23** | Kısa “FOLLOW THE COPPER SIGNAL”. İz ekranı fiziksel olarak geçer; arka kelimeyi kesip açar. Aralık ortasında yalnızca iz ve Cu; açıklama yok. | İz bir odak halkasının kenarına yerleşir. Market nesneleri bu referansa bağlanır. | `SignalHandoff`; aynı path sahibi, aynı viewport koordinatı. 0.3–0.5 viewport nefes; loading benzeri sonsuz bekleme yok. |
| **Market / .23–.39** | Bir fiyatın çevresindeki bağlam ortaya çıkar. Cu ana düğüm; üretici/ilişkili enstrümanlar 4–6 sade alan halinde açılır. Gerçek snapshot varsa etiketli değişim/ölçek. | İlişkilerden birinin açıklama işareti büyür, kaynak okuma penceresine dönüşür. Düğümler kaybolmadan önce bağlam başlığı sabitlenir. | `MarketScene`; küçük SVG, statik d3 yerleşimi önceden hesaplanır. “Moves together” için korelasyon hesabı yoksa iddia kullanılmaz. |
| **News / .39–.54** | Piyasa işaretinden kaynak lensi açılır. Bir haber başlığı, tarih, kaynak; ardından yorum ve belirsizlik ayrımı. Arka CONTEXT kelimesi lensin arkasında kalır. | Kaynak çizgisi zaman eksenine hizalanır. Haber anlamı fiyat serisine morph edilmez; kısa etiket değişimi “context → model outlook” ayrımı yapar. | `ContextLens`; DOM source + SVG bağlantı. Kaynak yoksa açık illustrative senaryo. Clip yalnız küçük lens bölgesinde. |
| **Forecast / .54–.72** | Önce historical path tamamlanır; kesim tarihi görünür. Sonra 5D median ve Q10–Q90 açılır. RANGE kelimesinin önünden band geçer; eksen/legend üst düzlemde okunur. | Kesim tarihini taşıyan bakır çizgi sabit kalır. Parlaklık azalır; aynı çizgi Evidence inceleme cetveli olur. | `ForecastScene`; tek çizim, reveal mask/stroke. Gerçek quantile yoksa olasılık bandı uydurulmaz. 300–500 nokta üstünde decimation. |
| **Evidence / .72–.91** | “WHAT HELD UP?” sorusu. Aşağıdaki veri yeteneğine göre gerçekleşen seri, karşılaştırılabilir pencere özeti veya açık eksik-kanıt açıklaması gelir. Hareket azalıp dikkat veriye geçer. | İnceleme cetveli sağ kenara çekilir, workspace frame'inin sınırına dönüşür. Rapor tarihi/provenance görünür kalır. | `EvidenceScene` + `EvidenceAdapter`; detay/uzun tablo pin dışında. Per-frame query ve React state yok. |
| **Workspace / .91–1** | Atmosferdeki bakır diffusion CTA'nın arkasında toplanır. “Your next question starts here.” Tek ana CTA; incelemeye devam linkleri ikincil. | Click normal `/dashboard` navigation; sahne kaynakları dispose. Gerçek route verisi hazır değilken sahte dashboard morph gösterilmez. | `WorkspaceThreshold`; opacity/transform 180–240 ms isteğe bağlı. Tıklamayı animasyon sonuna kadar kilitleme. |

Sahne bağlantılarının kabul kriteri: sınırın hemen öncesinde ve hemen sonrasında trace konumu, genişliği ve yönü tutarlı; reverse scroll aynı yolu geri kuruyor; rapid skip sonraki scene'i yarım hazırlık durumunda bırakmıyor. Birbirine geçen iki farklı veri setinin eksen/etiketleri aynı anda görünmüyor.

## 6. Mimari ve dosya kararları

Tüm aşağıdaki frontend yolları `frontend/src/` altındadır. Yeni isimler öneridir; bugün varmış gibi okunmamalı.

| Mevcut dosya/component | Karar | Gerekçe |
|---|---|---|
| `features/landing/LandingPage.tsx` | Refactor | Hero'dan CTA'ya ortak world sahipliği; semantic section'lar korunur |
| `Hero.tsx` | Metin/CTA korunur, yerel controller değiştirilir | Global manifest ve shared trace'e bağlanır |
| `CopperSignal.tsx` | Geometri üretimi yeniden kullanılır | Cu kimliğinin güçlü başlangıcını koru |
| `ResearchStory.tsx` | Enhanced düzen değiştirilir | Sağ kart/sol uzun metin modeli yerine full-bleed ortak stage |
| `Previews.tsx` | Verisel semantik/grafik parçaları seçilerek kullanılır | Bütün card wrapper'ları ve tekrarlanan başlıklar taşınmaz |
| `preview-data.ts` | Açık illustrative fixture olarak korunur | Gerçek veriden ayrı provenance |
| `useExperiencePolicy.ts` | Refactor | Viewport/pointer/RAM boolean'ı yerine layout-fit + reduce + measured quality ayrımı |
| `landing.css` | Kuralları konsolide et | Üste override ekleme zincirini bırak; yalnız landing scope |
| PR6 `SignalStage.tsx`, `story-timeline.ts` | Ortak trace/transition fikirlerini yeniden kullan | Üç sahne sınırını tüm narrative'e genişlet; PR merge varsayma |
| PR6 `ForecastPassport.tsx` | Merkez sahne olarak değiştir | Kanıt yeteneğine bağlı Evidence composition |
| `design/tokens.css`, `design/chart-tokens.ts` | Geliştir | Atmosfer/display/motion token'ları ile finansal semantik ayrı tutulur |
| `App.tsx`, route boundary, mevcut shell | Koru; yalnız gerekirse lazy sınırı düzenle | URL, error boundary, deep link, dashboard ayrımı |
| `api.ts`, `hooks/useQueries.ts` | Adapter tarafından kullan | Landing için polling davranışını aynen devralma |
| `pages/ValidationPage.tsx`, `features/validation/ReportComparison.tsx` | İlk milestone'da değiştirme | API alanları ve anlamlandırma referansı; chart primitive çıkarımı sonradan |
| Dashboard, news, heatmap business logic | Koru | Bu çalışma landing anlatısına odaklanır; ilk plan dashboard redesign'ını yeniden başlatmaz |

### Önerilen sahiplik

| Yeni birim | Girdi | Çıktı / sorumluluk |
|---|---|---|
| `NarrativeWorld.tsx` | policy, snapshot, manifest | Tek stage ve semantic içeriğin sahibi; mount/unmount lifecycle |
| `useNarrativeProgress.ts` | scroll, ölçülmüş sınırlar, resize | raw progress MotionValue; yalnız scene değişiminde React state |
| `narrative-manifest.ts` | sabit scene id ve interval tanımları | Nav hedefleri, okuma hold'ları, transition aralıkları için tek kaynak |
| `CopperAtmosphere.tsx` | p, quality, visibility | 2–3 düşük kontrast ortam katmanı |
| `CopperTrace.tsx` | p, scene anchors, geometry | Tek iz ve odak noktası; DOM/SVG koordinatları |
| `SpatialHeading.tsx` | text, reveal phase, occlusion contract | Semantic heading + dekoratif görsel katman |
| `NarrativeNavigation.tsx` | manifest, current scene | Native href, hedef hesaplama, focus/history yönetimi |
| `EvidenceAdapter.ts` | gerçek API cevabı / fixture | discriminated union: series, summary, unavailable, error |
| `EvidenceScene.tsx` | adapter çıktısı, p | Veri yeteneğine uygun karşılaştırma, provenance, rapor linki |
| `StaticNarrative.tsx` | aynı içerik/adapter | Mobile/reduced-motion/no enhancement sürümü |

Context içine değişen p sayısı koyma. Context gerekirse sabit MotionValue referanslarını taşısın. Scene birimleri bunlara subscribe olur. İş verisi TanStack Query'de; pointer/time/progress finansal state'e yazılmaz. Mevcut import yüzeyi `framer-motion` kalabilir; güncel belgelerdeki `motion/react` örnekleri körlemesine kopyalanmaz. [Motion useScroll](https://motion.dev/docs/react-use-scroll).

## 7. Scroll, navigation ve hareket sözleşmesi

Ölçülen world başlangıcı S, bitişi E, viewport yüksekliği V ise `p = clamp((scrollY-S)/(E-S-V), 0, 1)`; payda pozitif değilse statik moda geç. Scene aralığı [a,b] için yerel ilerleme `q=clamp((p-a)/(b-a),0,1)`. Mesafeler her frame DOM ölçümüyle alınmaz; resize, font yerleşimi ve içerik değişimi sonrası bir read batch ile güncellenir.

Bir scene'in ilk yaklaşık %20'si giriş, ortası okunabilir hold, son %20'si devir. Metin açık halde durur; scroll durunca yarı saydam iki paragraf bırakılmaz. Dekorasyon interpolation'ı ease ile şekillenebilir; fiyat değerlerinin interpolasyonu sadece chart'ın çizilme biçimidir, yeni veri örneği değildir.

**Native scroll ilk tercih.** Anchor için kısa mesafede CSS smooth yeterli olabilir; istenen hedef scene'in hold noktasını manifest'ten bulmak ayrıca gerekir. Native smooth süresi browser'a aittir. Çok uzak atlamada bütün uzun filmi oynatmak yerine kısa hedef geçişi veya doğrudan hedef tercih edilir. Scroll-jacking ve zorunlu snap yok.

`href="#market"` gibi gerçek linkler kalır. Normal primary click gerekirse hedef ofseti hesaplar; modifier/middle click engellenmez. Bir tıklama bir history entry üretir; her frame URL değişmez. `popstate/hashchange`, ilk deep link ve font sonrası layout hesaplaması aynı hedef çözümleyicisini kullanır. Sonunda ilgili semantic heading'e `preventScroll` ile focus taşınır; focus'un yeni bir ikinci scroll başlatması önlenir. Kullanıcı wheel/touch/keyboard ile müdahale ederse programatik hareket iptal edilir.

Lenis ancak native prototipte ölçülen belirgin senkron problemi kalırsa ayrı karar. Kullanılırsa scroll sahibi tek olur, native touch varsayılan kalır, nested scroll alanları explicit ele alınır. GSAP eklenirse aynı p'yi iki ayrı raf döngüsü üretmez; Lenis raf'ı ve ScrollTrigger update aynı zaman biriminde senkronize edilir. Lenis README bu entegrasyonu açıklıyor. [Resmî kaynak](https://github.com/darkroomengineering/lenis).

Damping yalnız ortam/pointer takip katmanında. Frame-rate bağımsız örnek `alpha=1-exp(-lambda*dt)`; smoothed += alpha*(target-smoothed). Raw scroll ile content/nav kilitli; ortam 80–140 ms civarı algısal gecikmeyle izleyebilir. İki kez smoothing uygulamak input lag yaratır. Velocity yalnız clamp edilmiş düşük genlikli ışık/derinlik tepkisi üretir; rota veya fiyat değiştirmez.

## 8. Typography, spatial layering ve Copper Atmosphere

**Typography:** League Gothic için font yüklenene kadar içerik görünür fallback. Başlık satır kırılımlarını mümkünse editoryal olarak tanımla. Responsive wrap gerektiğinde font-ready ve ResizeObserver sonrası tekrar ölç; harfleri her render yeniden bölme. Metni okuyucuya bir kez sun; dekoratif kopya `aria-hidden`, asıl heading semantic ve seçilebilir olsun. Büyük arka kelime bilgi için zorunluysa önündeki objenin ardında kaybolmasına izin verme.

Üç hareket ailesi: (1) trace yönünde line reveal; (2) bakır düzleminin gerisinden açılan tek kelime; (3) Evidence'ta bütün hareketin durduğu net tipografi. Bütün başlıklar için aynı karakter stagger kullanılmaz. Başlangıç değerleri reveal 450–650 ms, satır farkı 70–110 ms, hover 120–180 ms. Bunlar Lando sürelerinin kopyası değil yeni motion token önerisidir.

**Layer sırası:** ink background → atmosphere → büyük display word → copper geometry → veri/legend → etkileşim ve focus. World isolation kullan; transform/filter ancestor'ın fixed/sticky containing block etkisini kontrol et. SVG clip küçük bölgede; tüm ekran backdrop-filter/animated blur yok. CSS perspective ile 2.5D grup ayrımı önce denenir. Occlusion depth cue için yeterliyse gerçek 3D kamera kurulmaz.

**Atmosphere ilk sürüm:** iki büyük statik radial gradient katmanı + küçük tekrar eden grain texture. p yalnız bu katmanların transform/opacity'sini değiştirir; her frame gradient string'i üretmez. Sıcaklık geçişi önceden hazırlanmış katman crossfade'i. Metin alanına sakin ink scrim; pointer-events:none; viewport dışı ortam update'i durur. Continuous time loop zorunlu değil: kullanıcı durduğunda dünya sakin kalabilir.

**İsteğe bağlı shader deneyi:** tek fullscreen triangle, düşük çözünürlüklü field, `uProgress/uPointer/uTime/uResolution/uIntensity`. 2–3 octave noise ile bakır diffusion, smoothstep ile lokal trace glow; düşük frekans. Her pikselde uzun raymarch, gerçek volumetric ve bloom başlangıç kapsamında yok. Aynı look'u CSS katmanları üretiyorsa shader shipping gerekmez. Bu shader Coppermind için öneridir; Astra'ya atfedilmez.

Gerçek bakır mesh gerekirse PBR material ve küçük environment map yeterli olabilir. GLB yalnız gerekli geometri; Meshopt/Draco ancak transfer kazancı decoder maliyetini karşılıyorsa. KTX2 textures uygun destek testiyle; mobile 512–1024, desktop en fazla 2048 ile başla. Hareketli shadow maps yerine ortam aydınlatması veya baked/fake contact shadow. Tek camera/renderer, lazy scene, context-lost fallback; geometry/material/texture/render target ve listener dispose. Üç boyutlu model bu planın teslim bağımlılığı değildir.

## 9. Evidence: gösterinin denetlenebilir sonuca dönüşmesi

Doğrulanan mevcut yol: `api.ts::fetchLatestBacktest()` → `/models/tft/backtest/latest`; backend `backend/app/main.py` altındaki `/api/models/tft/backtest/latest`. Cevapta `summary_metrics`, `window_metrics`, `theta_comparison` bulunabilir. `ValidationPage` bunları özet ve pencere tablosuna çeviriyor. Endpoint'in varlığı verinin yayında mevcut olduğunu garanti etmiyor; canlı sayfa bu araştırmada rapor bulunmadığını gösterdi.

**A — Pointwise kayıt gerçekten varsa:** aynı instrument, forecast issue date, model/checkpoint, horizon, trading calendar ve birimle eşleşmiş tahmin/gerçekleşen çiftleri. Reveal önce tahmini sabitler, sonra gerçekleşeni açar. Bir forecast'i sonradan güncellenmiş modelle yeniden hesaplayıp geçmiş performans gibi sunma. Cutoff'tan sonrası gözlemler ve quantile coverage yalnız gerçek kayıtlarla gösterilir. İlk API audit'i bu serilerin mevcut olup olmadığını doğrular; mevcut summary endpoint'in yeterli olduğu varsayılmıyor.

**B — Yalnız summary/window report varsa:** döneme göre hizalanmış error/baseline karşılaştırması veya window işaretleri. Forecast çizgisinin yerine çizilmiş sahte actual seri koyma. Theta karşılaştırmasında aynı horizon/sample/birim sağlanmıyorsa yan yana başarı oranı üretme. Report date, window sayısı ve karşılaştırmanın sınırı sahnede görünür.

**C — Rapor yoksa (şimdiki canlı durum):** çizgi boş metrik kartına dönüşmez; inceleme cetveli üzerinde “Published validation is not available yet” net görünür. Altında neyin şu an incelenebildiği: model metadata ve veri tazeliği bağlantıları. Bu bir kalite zaferi değil şeffaflık payoff'udur. Error ise unavailable'dan ayrı anlatılır, retry sunulur. Placeholder dashboard veya sahte passed badge yok.

Etkileyici Evidence için backend raporunun yayınlanması ayrı ürün bağımlılığıdır. Frontend tasarımı bunu gizleyemez. Narrative renderer A/B/C yeteneğine göre aynı alanı doldurmalı; loading sırasında sahne yüksekliği değişmemeli.

## 10. Responsive, erişilebilirlik ve yükleme

Mobile, full desktop pin'in küçültülmüş hali olmayacak. Normal flow'da kısa Cu → market → context → range → evidence; kenarda ince bakır iz aynı continuity'yi korur. Büyük type ekran dışına taşmaz. Dokunma hareketleri browser'a ait; hover'a saklanmış bilgi yok. Touch target tasarım hedefi en az 44×44 CSS px.

Statik mod: reduced-motion, save-data, layout'a sığmayan yazı veya başarısız enhancement. Bütün içerik erişilebilir sırada; crossfade/tilt/parallax kaldırılır. 200% zoom ve büyütülmüş font yüzünden clipping varsa statik moda düş. Sadece cihaz RAM değerine bakarak mobile/GPU kararı verme; bu bilgiler her browser'da yoktur.

Quality tier önerisi: Q0 statik SVG/CSS; Q1 native scroll + küçük transform/opacity + trace; Q2 geniş ekranda full spatial composition; Q3 yalnız deney başarılıysa shader. WebGL varsa DPR Q1=1, Q2 en fazla1.5; full cihaz DPR'sini otomatik kullanma. Birkaç saniyelik p95 frame-time eşiği aşılırsa tek kademe düş; anlık FPS dalgalanmasıyla sürekli kalite değiştirme. Orientation/font resize'da anchor ölçümü yeniden yapılır.

Initial HTML'de heading, value proposition ve CTA mevcut; prerender korunur. Loader overlay yok, sahte yüzde yok. Critical font tek display subset, Geist korunur; en fazla gerekli font preload. Ağır preview/renderer dynamic import; dashboard route'a landing bağımlılıkları sızmaz. Snapshot isteği ilgili sahneye yaklaşınca bir kez, uygun cache ile; landing'de dashboard polling hook'ları doğrudan kullanılmaz. Hash'li asset'ler immutable cache, HTML revalidate; bundle graph ile doğrula.

## 11. Performans bütçeleri ve ölçüm planı

Bütçe hedefleri ölçüm değildir. Önceki PR6 teslim raporunun aynı ortam ölçümleri: initial JS gzip 117580→116544 byte; initial CSS 12111→13145; dashboard JS closure 263660→262785. Yeni plana uygulanmış sonuçlar değildir. Mevcut script sınırları initial JS190000, initial CSS14000, dashboard JS280000 byte; CSS boşluğu sınırlı. Yeni atmosferi override yığarak eklemek yerine eski kuralları sadeleştir.

| Alan | İlk hedef / gate | Ölçüm |
|---|---|---|
| Core Web Vitals | p75 LCP≤2.5s, INP≤200ms, CLS≤.1 | Vercel Speed Insights/RUM; yeterli gerçek trafik. Lighthouse INP yerine geçmez |
| Görsel yerleşim | İç lab hedefi CLS≤.05 | Performance layout shift attribution; font/resize/hydration |
| JS/CSS | Mevcut repo gate'lerini aşma; dashboard closure büyümesin | `npm run build` + `npm run check:budgets`; route chunk grafiği |
| Font/ilk medya | Ek display WOFF2 hedef≤40KB; critical transfer toplam hedef≤500KB | Network cold cache waterfall; transfer ve decoded boyut ayrımı |
| Frame | 60Hz için16.7ms bütçe; desktop p95≤16.7ms hedefi | DevTools Frames/Main/Rendering; 10–20sn forward/reverse/anchor kayıt |
| Main thread | Sürekli scroll kaynaklı >50ms long task yok; animation JS hedef≤3ms/frame | Performance flame chart, layout/paint events |
| GPU | Büyük filter/mask/overdraw azalmalı | Layers/paint flashing; WebGL varsa Spector.js frame capture |
| Geometri | İlk SVG≤32 kontur, chart≤500 örnek; aynı anda 1–2 scene görseli | DOM/SVG sayımı, Profiler; canvas'a geçiş ancak ölçümle |
| Opsiyonel3D | Başlangıç≤30 draw call,≤50k visible triangle, tek renderer | `renderer.info`; bu sayılar kalite garantisi değil deney tavanı |
| Opsiyonel GPU bellek | Yaklaşık toplam≤64MB hedef; model transfer≤500KB | Texture/render-target hesabı + araçlar; browser kesin GPU belleğini her zaman vermez |
| Route disposal | 5 giriş/çıkış sonrası sürekli artan heap/context yok | Heap snapshot, allocation timeline, listener/context kontrolü |
| Mobile | Orta sınıf Android + iOS Safari; cold/warm, düşük güç, orientation | Gerçek cihaz; desktop CPU throttle gerçek mobile GPU'yu taklit etmez |

RGBA8 2048² texture yaklaşık16MiB; mip zinciriyle yaklaşık21.3MiB. Sıkıştırılmış transfer boyutu GPU belleği değildir. Fullscreen render target alanı DPR² ile büyür. Shader maliyeti draw-call düşükken bile fill-rate/overdraw yüzünden yüksek olabilir.

Main-thread React commit'leri baskınsa state subscription sınırını düzelt; layout baskınsa ölçüm/yazma sırasını ve geometri animasyonlarını düzelt; CPU boşken frame kaçırıyorsa raster/GPU/texture maliyetine bak. Long task'ları sadece debounce ile saklama; scroll ilerleyişini debouncing ile geciktirmek akıcılığı bozar. [CWV eşikleri](https://web.dev/articles/vitals).

## 12. Milestone ve kabul kriterleri

| Sıra | İş | Görsel/teknik kabul |
|---|---|---|
| M0 — Kaynak ve veri sözleşmesi | PR6'nın merge durumunu yeniden kontrol et; baseline; Evidence A/B/C alan audit'i | Sonuçların tarih/horizon/model eşleşmesi; mevcut URL ve fonksiyon envanteri |
| M1 — Continuity iskeleti | World/manifest/shared trace, static fallback ve global atmosphere | Hero'dan CTA'ya tek nesne; ileri/geri scroll'da kopmayan devir; no-JS içerik; font/zoom sınaması |
| M2 — Görsel dil | League Gothic, spatial layers, üç reveal ailesi | Her büyük kelime okunur; occlusion hiyerarşiye hizmet eder; font swap CLS yok |
| M3 — Kritik koreografi | Cu açılımı, FOLLOW turning point, Market→News→Forecast | .25/.50/.75 progress'te durup okunabilirlik; hiçbir boundary'de çift metin; hızlı atlamada tamamlanmış state |
| M4 — Evidence ve navigation | Capability adapter, A/B/C layout, anchors/history/focus, CTA | Gerçek rapor yokken dürüst sonuç; klavye focus kaybı yok; back/deep link/reduced motion |
| M5 — Mobile ve performans | Fit policy, asset/code split, long-task/heap/route QA | Gerçek mobile ve 200% zoom; CWV lab+RUM ayrımı; repo bütçeleri; dashboard regression yok |
| M6 — Polish / opsiyonel deney | Hover, kontrollü velocity, gerekirse shader A/B prototipi | CSS sürümüne göre görünür fayda ve ölçülebilir bütçe uyumu; yoksa shader shipping dışı |

Kodlama başladığında her milestone sonunda ilgili test/lint/build/budget gate'i; görsel kabul için desteklenen preview'da desktop ve gerçek mobile review. Saf timeline sınırları/reverse/anchor mapping ve Evidence normalization için meaningful test; her CSS değerini aynalayan test yazma.

**Tasarım kabul soruları:** İlk10 saniyede ürünün ne olduğu anlaşılıyor mu? Kullanıcı trace'i ikinci sahnede yeniden aramak zorunda mı? Market/News/Forecast ayrımı metinsiz hareketten anlaşılıyor mu? Evidence kullanıcıya gerçekten neyin bilindiğini söylüyor mu? Returning user intro'ya takılmadan çalışmaya başlayabiliyor mu?

Bu plan PR6'daki üç sahneli düzeni nihai çözüm kabul etmez. PR6 merge edilmeden de M0 başlayabilir; merge edilirse o commit yeniden baseline alınır. Eski DOCX teknik başvuru olarak korunur; uygulanacak art direction ve yeni kararlar için bu plan kullanılır. Araştırma tamamlandıktan sonra frontend koduna geçilmedi.
