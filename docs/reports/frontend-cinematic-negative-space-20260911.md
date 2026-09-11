# CopperMind Cinematic Negative-Space Handoff

| Alan | Değer |
| --- | --- |
| Tarih | 11 Eylül 2026 |
| Kapsam | Forecast çıkışı, copper ingot particle morph'u ve research CTA atmosferi |
| Durum | Uygulandı; test, build, lint, budget ve browser smoke geçti |

## Uygulanan düzenlemeler

- Forecast preview ve `03 / THE POSSIBILITIES` metni `.74–.75` civarında kapanacak şekilde öne alındı. Böylece ingot geçişi artık dashboard yüzeyi veya forecast açıklamasıyla üst üste binmeden gerçek bir negative-space aşamasında çalışıyor.
- Particle timeline forecast path'ten evidence mark'a `.70–.74`, ardından copper ingot'a `.74–.84` arasında geçiyor; ingot `.84–.93` arasında korunuyor ve research CTA başladığında `.93–1` aralığında iki yana açılıyor. Bu, önceki kısa `.87–.93` morph'una göre oluşum/settle süresini belirgin biçimde uzatıyor.
- Büyük `FORECAST` arka plan kelimesi forecast çıkışıyla birlikte kapanıyor; global `RESEARCH` kelimesi CTA yaklaşırken `.92` sonrasında görünür oluyor. Bu sayede negative-space sırasında yalnızca particle world ve ortak atmosfer kalıyor.
- Cinematic `YOUR RESEARCH STARTS HERE` yüzeyi artık ayrı radial/linear panel arka planı veya yerel `RESEARCH` pseudo-element'i üretmiyor; alttaki tek global atmosphere, yıldız alanı ve ortak typography katmanını kullanıyor.

## Doğrulama

- `npm test -- --run --reporter=dot`: 9 dosya / 52 test geçti; heatmap p95 real `1.61ms`, large `3.12ms`.
- `npm run build`: başarılı; 2.909 modül dönüştürüldü.
- `npm run lint`: başarılı.
- `npm run check:budgets`: başarılı; initial JS `120.777` gzip byte, CSS `13.327` gzip byte, dashboard JS `267.101` gzip byte.
- `npm run check:experience`: landing `7`, adaptive `3`, workspace `6` senaryosu geçti; API isteği ve page error yok.
- Browser smoke sırasında `.74` sonrasında tüm dashboard surface'lerinin kapandığı ve `.96` civarında CTA'nın ortak atmosphere üzerinde göründüğü doğrulandı.
- Bu kontroller runtime FPS/GPU profili değildir; ölçülmemiş bir FPS kazanımı iddia edilmiyor.
