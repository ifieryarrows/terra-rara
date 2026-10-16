# Code Review — `codex/tft-xai-explain` — Muse Spark 1.3

* Reviewer model: **Muse Spark 1.3 Free** (`opencode/muse-spark-1.3-contributor-free`)
* Date: 2026-10-16
* Range: `main 7f4a49b` .. `origin/codex/tft-xai-explain 283d25f` (10 commits)
* Plan: `docs/superpowers/plans/2026-10-16-tft-xai-explain.md` (branch)
* Invariants: `GEMINI.md`
* Mode: koordineli review — Reviewer-A (backend), Reviewer-B (frontend/API/test), + bu dosya (sentez + kendi doğrulamam). Read-only, working tree mutate edilmedi.

## Kapsam

TFT-ASRO XAI: `interpret_batch` + `get_variable_importance` fix, `app/tft_explain.py` builder,
predictor/worker wiring + `explain_json` snapshot kolonu, `GET /api/explain/{symbol}`,
VSN importance'ın metadata'ya yazımı + summary tercihi, `ModelFocusPanel` + Overview,
README/GEMINI docs.

## Hüküm

**Ready to merge — With fixes.** Critical yok. Merge öncesi 2 Important (worker fail-open deliği,
`_importance_entries` NaN filtresi) + predictor unpack fixlenmeli. Budget headroom bir sonraki
dashboard işinden önce ele alınmalı.

## Strengths (doğrulandı)

* `interpret_batch` (`backend/deep_learning/models/tft_copper.py:845`) pinned `pytorch-forecasting 1.8.0`
  `interpret_output(reduction="none")` şekilleriyle uyumlu; eski `predict()+dict` varsayımı gitmiş,
  `get_attention_weights` silinmesi güvenli (referans yok).
* Fail-open 3 katmanda gerçek: `try_build_explain_payload -> None`,
  `_latest_window_explain -> None`, `get_variable_importance -> {}`; endpoint snapshot-only,
  canlı inference yok; UI boş durum var.
* Review Focus 6 maddesinin testi var: length-mismatch `ValueError`, NaN/Inf drop,
  unknown-feature fallback, malformed `importance_json` -> MRMR fallback, stale -> 404.
* `_tft_snapshot_invalid_reasons` refactor'ı behavior-preserving; `quality_gate.py` untouched;
  quality-gate filtreleri korunmuş; horizon etiketleri (5-day primary + t+1 + not causal) doğru.
* Migration'lar idempotent; trainer -> metadata -> summary zinciri uçtan uca tamam.

## Issues

### Important (merge öncesi)

1. **Worker persist tek transaction'da `explain_json`'ı da yazıyor — explain hatası snapshot'ı düşürür**
   * `backend/worker/tasks.py:720-756`
   * Fail-open sözleşmesine aykırı. Explain'e özel hata (serialize/payload/eksik kolon) tüm forecast
     snapshot'ını `rollback` eder.
   * Fix: except'te `explain_json=None` ile bir retry, downgrade'i logla.

2. **`_importance_entries` non-finite filtresi yok**
   * `backend/app/main.py:_importance_entries` — bare `float(imp)`; komşu `_numeric_map`'te
     `isnan/isinf` filtresi var, burada yok. `total_weight <= 0` NaN'i yakalayamaz (NaN karşılaştırma
     hep False) -> `json.dumps(NaN)` -> `/api/models/tft/summary` invalid JSON, `JSON.parse` patlar.
     MRMR fallback sadece malformed'da tetiklenir, NaN'da değil.
   * Fix: `math.isfinite` filtresi (`_numeric_map` ile aynı).

3. **Predictor `x, _ = next(iter(dataloader))` tuple varsayıyor**
   * `backend/deep_learning/inference/predictor.py:650`; trainer tarafındaki
     `batch[0] if isinstance(batch,(tuple,list)) else batch` defansı yok. Bare-dict batch ->
     ValueError -> fail-open yutar -> explain prod'da sessizce hiç üretilmez, testler green kalır.
   * Fix: aynı defansif unpack. Test yok — wiring testi eklenmeli.

4. **Dashboard JS budget headroom bitti**
   * `frontend/src/pages/OverviewPage.tsx:~561` + `frontend/scripts/check-budgets.mjs` —
     `279402/280000` byte, 598 byte kaldı. Bu branch green ama sonraki dashboard değişimi CI'ı
     kızartır.
   * Fix: `ModelFocusPanel`'i `HeatmapPanel` gibi lazy-load et veya limiti bilinçli artır.

### Minor

5. `frontend/src/components/ModelFocusPanel.tsx` — `horizon: 5-day primary` UI'da yok, sadece
   `Attention · forecast day 1 (t+1)` var. GEMINI horizon-separation açısından yanıltıcı.
   Fix: `data.horizon`'dan tek satır.
6. `backend/deep_learning/training/trainer.py:1562` `_importance_metadata_payload` — `result`'ta
   `trained_at` key'i hiç yok, stored `trained_at` hep null. Fix: timestamp'i promotion'da enjekte
   et veya field'ı kaldır.
7. `frontend/src/pages/OverviewPage.tsx:explainResult` — optional kartın 500/network hatası global
   `loadErrors['Model focus']` banner'ını kirletiyor. Fix: reject'te `setExplain(null)`, banner'a dokunma.
8. `ModelFocusPanel.tsx` attention barlar sadece `title`'da — keyboard/SR erişilemez.
   Fix: container `role="img"+aria-label` / sr-only özet.
9. `tft_copper.py:865-867` — slice sonrası renormalize yok. `horizon=0`'da no-op, gelecek caller için
   trap. Fix: renormalize veya assert/document.

## Test-gap önerisi

Predictor `include_explain` pop/move, worker Stage 5.5 `run_id` stamp + persist, `fetchExplain`
404->null, Overview kart render — bu seam'ler sadece dolaylı testli. Eklenmeli.

## Diğer agent ile kıyas notu

* Reviewer-A: With fixes — worker retry vurgusu.
* Reviewer-B: Yes + non-blocking follow-ups — 610 backend / 107 frontend pass iddiası, budget + 5 hardening.
* Bu dosya: ikisiyle de hemfikir; severity'de 2+3+4'ü merge-öncesi sayıyorum (B'de Minor olan NaN
  ve unpack'ı sözleşme deldiği için yükselttim). Kıyas için diğer agent dosyasıyla diff'lenebilir.

## Declined to judge

* Gerçek checkpoint ile e2e explain üretimi (checkout'ta checkpoint yok; plan da checkpoint makinesine bırakıyor).
* Canlı PgBouncer pooler altında migration load-testi.
* Commentary/chat faz-2 tüketimi, `as_of` tarihsel sorgu, live-inference fallback — plan-dışı.
* Offline pytest/vitest/budget'ları bu oturumda koşmadım (read-only review); B'nin ampirik sonuçları rapor olarak alındı.

---

## Post-Remediation & Final Verification Addendum (2026-10-16)

Yukarıdaki inceleme `283d25f` commit'ini kapsamaktaydı. Takip eden `b391591` ve son sertleştirme
commit'i sonrasında yukarıdaki tüm Important ve ilgili Minor bulgular kapatılmıştır:

1. **Worker `explain_json=None` Retry (Important #1 — Çözüldü):** `worker/tasks.py:_persist_tft_report_snapshot`
   içinde `explain_json` ile yazım hata verdiğinde `session.rollback()` çağrılıp `explain_json=None` ile
   ikinci kez yazım denenmektedir. Böylece açıklama kaydı hatası tahmin snapshot'ını düşürmez
   (`test_persist_retries_with_explain_none_when_explain_write_fails`).
2. **`_importance_entries` ve `get_variable_importance` Non-Finite Filtresi (Important #2 — Çözüldü):**
   Hem VSN havuzlamasında (`np.isfinite` + `math.isfinite`) hem de `/api/models/tft/summary`
   `_importance_entries` okumasında non-finite değerler elenmektedir.
3. **Predictor Savunmacı Batch Unpack (Important #3 — Çözüldü):** `predictor.py:_latest_window_explain`
   içinde `batch[0] if isinstance(batch, (tuple, list)) else batch` savunmacı açılımı uygulanmış ve
   `test_latest_window_explain_accepts_bare_dict_batch` ile test edilmiştir.
4. **UI Horizon Etiketi & `trained_at` & Eval-Mode (Minor #5, #6, A-01 — Çözüldü):**
   `ModelFocusPanel.tsx` birincil `5-day cumulative log return (primary)` ufkunu ve `attention.step.horizon_index`
   bilgisini göstermekte; `_run_interpretation` eval modunu zorlayıp eski durumu geri yüklemektedir.
5. **Son Doğrulama Sonuçları:**
   * Backend offline testleri: **631 passed** (`-m "not online"`)
   * Frontend lint: **0 warnings**
   * Vitest: **14 test files, 110 passed**
   * Build & Budgets: **OK** (`initialJS: 127,036 / 190,000 B`, `css: 13,990 / 14,000 B`,
     `dashboardJS: 279,564 / 280,000 B`)

