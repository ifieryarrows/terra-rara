# CopperMind Landing Particle Tuning

| Alan | Değer |
| --- | --- |
| Tarih | 11 Eylül 2026 |
| Referans | `388a7e44` cinematic landing checkpoint |
| Kapsam | Landing particle görünürlüğü, yoğunluğu ve scroll fiziği |
| Durum | Kod ve otomatik kontroller doğrulandı; yeni runtime FPS profili bekliyor |

## Uygulanan ayarlar

- High kalite particle sayısı `420 → 640`, balanced/fallback sayısı `280 → 420` oldu.
- High/balanced WebGL point size `13 / 12` olarak ayarlandı; CSS canvas opacity `.94` oldu.
- WebGL fragment halo/core alpha değerleri `.38 / .66`; Canvas2D yarıçap ve alpha değerleri kaliteye göre artırıldı.
- `MAX_PARTICLES` artışı son evidence formundaki halka ve trace dağılımına oranlı uygulandı; yeni parçacıklar form dışına taşmıyor.
- Başlangıçtaki `.04–.30` timeline aralığına düşük genlikli yatay/dikey `earlyTide` eklendi. Bu hareket, ana scatter geçişinden ayrıdır.
- Idle drift hızı ve genliği artırıldı; scroll başlamadan önceki alan tamamen statik kalmıyor.
- Scroll impulse gain `.72 → .48`, damping `.90 → .965`, particle response `120ms → 150ms` oldu. Dağılım sonrasında hareket daha uzun ve daha düşük enerjili sönüyor.

## Doğrulama

- `npm test -- --run --reporter=dot`: 52/52 başarılı.
- `npm run build`: başarılı.
- `npm run lint`: başarılı.
- `npm run check:budgets`: başarılı; initial JS `120.779` gzip byte, CSS `13.327` gzip byte, dashboard JS `267.100` gzip byte.
- Heatmap benchmark izole tekrarında `3.61ms / 9.10ms`, tam suite tekrarında `3.79ms / 10.90ms` ölçüldü; particle tuning ile ilişkili bir regresyon gözlenmedi.
- Yeni browser FPS/GPU profili için `check-experience.mjs` çalıştırılamadı; çalışma ortamında `playwright` paketi kurulu değil. Bu nedenle bu tuning için yeni FPS kazanımı iddiası yapılmıyor.
