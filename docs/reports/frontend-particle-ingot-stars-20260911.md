# CopperMind Landing Particle Ingot and Space Atmosphere

| Alan | Değer |
| --- | --- |
| Tarih | 11 Eylül 2026 |
| Referans | `388a7e44` cinematic landing checkpoint |
| Kapsam | Hero perspektif hareketi, CTA öncesi ingot morph'u ve global yıldız atmosferi |
| Durum | Kod ve browser smoke doğrulandı; runtime FPS/GPU profili bekliyor |

## Uygulanan akış

- İlk C formuna `.02–.24` timeline aralığında WebGL ve Canvas2D için aynı pseudo-perspective tilt uygulandı. Y ekseni sıkışması, derinlik kaynaklı yatay skew ve küçük aşağı offset parçacıkların 3D yüzey gibi yatmasını sağlar.
- Forecast/evidence akışının sonuna deterministic `copperIngot` keyframe'i eklendi (`.80–.93`). Form, sığ trapezoid front face, top face ve lower lip dağılımından oluşur; ayrı bir model veya yeni veri iddiası değildir.
- CTA eşiğinde (`.93–1`) ingot parçacıkları `researchSpreadField` ile iki yana ve dikey alana açılır. Parçacık renderer'ı CTA görünürken çalışmaya devam eder.
- Global atmosphere'a compositor-friendly, CSS radial-gradient yıldız katmanı eklendi. Katman aynı scroll timeline'ından düşük genlikli x/y drift ve opacity alır; pseudo-element nefes hareketi reduced-motion modunda kapanır.
- Ingot aralığında mavi karışım azaltılarak formun copper olarak okunması güçlendirildi; spread aşamasında mevcut sinyal renkleri geri gelir.

## Doğrulama

- `npm test -- --run --reporter=dot`: 52/52 başarılı (ısı haritası p95: real `3.28ms`, large `9.52ms`; benchmark ısınma ve batch ortalamasıyla paralel suite dalgalanmasına dayanıklı hale getirildi).
- `npm run build`: başarılı; landing CSS gzip `3.91 kB`, cinematic JS gzip `7.58 kB`.
- `npm run lint`: başarılı.
- `npm run check:budgets`: başarılı; initial JS `120.779` gzip byte, CSS `13.327` gzip byte, dashboard JS `267.100` gzip byte.
- `EXPERIENCE_URL=http://127.0.0.1:5173 node scripts/check-experience.mjs`: başarılı; 7 landing viewport, 3 adaptive fallback ve 6 workspace route kontrol edildi. Script’in stale hidden-canvas, background-word ve hidden text selector'ları düzeltildi; StrictMode cleanup'ında canvas metadata temizleniyor.
- Bu browser smoke koşusu FPS/GPU profili değildir; yeni ölçülmemiş performans kazanımı iddia edilmiyor.
