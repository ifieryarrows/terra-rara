# CopperMind Frontend Test Stability

| Alan | Değer |
| --- | --- |
| Tarih | 11 Eylül 2026 |
| Kapsam | Heatmap benchmark ve cinematic browser QA |
| Durum | Çözüldü; yerel suite ve browser smoke geçti |

## Kök nedenler ve düzeltmeler

- Heatmap p95 testi, parallel Vitest worker yükü altında 1.000 öğelik tekil layout çağrılarını ölçüyordu. Ölçüm aralığına `layout.leaves()` ve assertion maliyeti de giriyordu; bu durum gerçek layout değişikliği olmadan p95’in 29 ms’e sıçramasına neden olabiliyordu. Dört çağrılık warm-up, üçlü batch başına ortalama ve 20 örnekle saf layout ölçümü kullanıldı. `8 / 12 ms` eşikleri değiştirilmedi.
- StrictMode effect cleanup ilk WebGL canvas’ının `data-renderer` metadata’sını bırakıyordu. Cleanup artık iki canvas’ın renderer/quality/count/compile metadata’sını temizliyor.
- Browser QA, başlangıçta görünürlüğü kapalı scroll katmanlarını `innerText` ile arıyor, gizli fallback canvas’ı `first()` ile seçiyor ve mevcut beş background word yerine eski dört kelimeyi bekliyordu. DOM metin kontrolü `textContent`, aktif canvas seçimi `data-renderer`, typography beklentisi ise mevcut continuous story ile hizalandı.
- `check:experience` npm script’i eklendi ve varsayılan hedef normal Vite portu olan `127.0.0.1:5173` oldu. Playwright frontend dev dependency olarak lockfile’a alındı.

## Son doğrulama

- `npm test -- --run --reporter=dot`: 9 dosya / 52 test geçti; heatmap p95 real `0.79ms`, large `2.60ms`.
- `npm run check:experience`: landing `7`, adaptive `3`, workspace `6` senaryosu geçti; API istekleri ve page error yok.
- `npm run build`: başarılı.
- `npm run lint`: başarılı.
- `npm run check:budgets`: başarılı; initial JS `120.779` gzip byte, CSS `13.327` gzip byte, dashboard JS `267.104` gzip byte.
- `npm audit --omit=dev --audit-level=moderate`: production dependency vulnerability yok.
