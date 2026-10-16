# Code Review — `codex/tft-xai-explain` — DeepSeek V4.1 Flash

* Reviewer model: **DeepSeek V4.1 Flash Free** (`opencode/deepseek-v4.1-flash-free`)
* Date: 2026-10-16
* Range: `main 7f4a49b` .. `codex/tft-xai-explain 283d25f` (10 commit)
* Plan: `docs/superpowers/plans/2026-10-16-tft-xai-explain.md` (branch)
* Invariants: `GEMINI.md`, `AGENTS.md`
* Mode: bağımsız review — read-only; backend offline suite + frontend lint/test/build/budgets çalıştırıldı. İkinci bağımsız inceleme ile karşılaştırma adımı kullanıcı talebiyle ayrıca yapılacaktır.

## Kapsam

TFT-ASRO XAI: `interpret_batch` + `get_variable_importance`, `app/tft_explain.py` payload builder,
predictor/worker wiring + `explain_json` snapshot kolonu, `GET /api/explain/{symbol}`,
VSN importance'ın metadata'ya yazımı + summary tercihi, `ModelFocusPanel` + Overview entegrasyonu,
migration blokları, README/GEMINI docs. Diff'in tamamı (~2923 satır) ve dokunulan tüm dosyalar okundu.

## Hüküm

**Merge edilebilir.** Critical/Major bulgu yok. Önerilen tek pre-merge düzeltmesi: **A-01** (tek satırlık
eval-mode capture/restore). Kalanlar kapsam/öneri/nitpick seviyesinde; hiçbiri bloklamıyor.

## Doğrulama (kanıt)

* Backend offline suite (`py -3 -m pytest tests/ -q -m "not online"`): **610 passed** (39.2s).
  Yeni/değişen 35 test ayrıca birebir koşuldu: **35 passed** (ilk koşudaki 3 setup hatası Windows
  `pytest-of-LENOVO` tmpdir contention'ıydı — izole `--basetemp` ile temiz geçti; kod hatası değil).
* `interpret_output`/`forward` kontratı kurulu **pytorch-forecasting 1.8.0** kaynağıyla çapraz doğrulandı
  (`_tft.py:706-855`): `reduction="none"` per-sample semantiği, renormalize attention, grup isim
  property'leri, `encoder_lengths` çıktısı — wrapper ile tutarlı.
* Lightning mod davranışı kurulu kaynaktan doğrulandı (`evaluation_loop.py:281-352`,
  `prediction_loop.py:194-206,348-353`, `model_helpers.py:64-84`); pytorch-forecasting bu hook'ları
  override etmiyor → A-01'in temeli.
* Promotion sırası: CI gate → HF Hub upload başarısı → `persist_promoted_tft_metadata`; içeride gate
  **yeniden** değerlendirilip fail-closed (`trainer.py:1137-1157, 1594-1601`). AGENTS.md kontratı korunmuş.
* Worker Stage 5.5 (`worker/tasks.py:677-793`): `explain` payload'dan pop edilip `explain_json`'a yazılıyor,
  `run_id` enjekte ediliyor; explain hatası non-critical, snapshot persist'i düşürmüyor (farazi ayrışma:
  aynı transaction — bkz. A-06 notu).
* Endpoint (`main.py:1412-1471`): snapshot-only, canlı inference yok (test bunu `pytest.fail` ile zorluyor);
  freshness `_tft_snapshot_invalid_reasons` ile paylaşımlı; refactor eski inline kodla satır satır behavior-preserving.
* Migration: SQLite round-trip + idempotent ikinci geçiş + missing-table toleransı + gerçek hata re-raise
  test edilmiş (`test_tft_snapshot_explain_roundtrip.py`).
* Frontend: `npm run lint` 0 uyarı; hedefli vitest 18/18; tam vitest 106/107 (tek hata: ilgisiz heatmap
  perf benchmark'ı yük altında 12.08ms>12ms — izole koşuda 13/13 geçiyor, flake); `npm run build`
  (tsc+vite+prerender) OK; `npm run check:budgets` **exit 0** (initial JS 127.038/190.000,
  CSS 13.990/14.000, dashboard 279.406/280.000).
* Hizalama: `history = master_df.tail(max_encoder_length)` ile attention uzunluğu ve forecast batch'i
  (`predictor.py:447-450, 470-478, 650`) aynı pencere — pozisyonel hiza doğru.

## Strengths (doğrulandı)

* Fail-open 3 katmanda gerçek: `try_build_explain_payload -> None`, `_latest_window_explain -> None`,
  `get_variable_importance -> {}`; worker non-critical; endpoint 404'e düşüyor; UI boş durumu var.
* Review Focus 6 maddesinin testi var: length-mismatch `ValueError`, NaN/Inf drop + attention zero,
  unknown-feature fallback, malformed `importance_json` -> MRMR fallback, numarasız/stale -> 404.
* Non-finite güvenliği uçtan uca: `finite_number` + `json.dumps(..., allow_nan=False)` round-trip testi.
* VSN importance zinciri: trainer -> `importance_json` (gate-first) -> summary tercihi (MRMR fallback'li);
  `quality_gate.py` dokunulmamış; hedef/metrik değişikliği yok.
* UI: `fetchExplain` 404 -> `null`; `Promise.allSettled` hata atıfı; partial payload'a karşı defansif
  render (testli); `driverCategories` gerçek duplikasyonu kaldırıyor; tek `./api` mock'u güncellenmiş.
* Docs: README endpoint/404 semantiği ve "not causality" çerçevesiyle uyumlu; GEMINI'ye fail-open invariant'ı.

## Issues

### Minor (merge bloklamaz; A-01 için düzeltme önerilir)

1. **A-01 — `interpret_batch` eval-mode garantisi yok; trainer fallback yolunda dropout'lu VSN ağırlığı**
   * `backend/deep_learning/models/tft_copper.py:845-885` (`torch.no_grad()` var, `model.eval()` yok),
     çağrı `trainer.py:1046`, fallback `trainer.py:794-809`; üretim yolu zaten eval
     (`predictor.py:649`, `load_tft_model` `tft_copper.py:823`).
   * `Trainer.fit()` sonrası in-memory modül train modundadır (Lightning validation sonunda yakaladığı
     moda geri döner); `best_path` boşsa veya checkpoint reload hata verirse `evaluation_model = model`
     olur ve `get_variable_importance` dropout aktifken çalışır → `importance_json` / `tft_metadata.json`
     gürültülü ve non-deterministik olabilir. Ortalama/renormalize edildiği için etki düşük; yalnızca
     bilgilendirici promotion metadata'sı. Canlı explain payload'ı etkilenmez.
   * Fix: `was_training = model.training; model.eval()` + `finally: if was_training: model.train()`;
     stub testte `model.training is False` assert'i.

2. **A-05 — "model trained after snapshot ⇒ 404" dalı iki endpoint için de testsiz**
   * `backend/app/main.py:1227-1231`; testli olan yalnızca PriceBar dalı
     (`test_tft_explain_endpoint.py:164-183`). Mantık önceden vardı ve behavior-preserving taşındı;
     kapsam boşluğu, bug değil.
   * Fix: `quality_gate_passed=True` + daha yeni `trained_at` meta'sı ile 404; `quality_gate_passed=False`
     negatif vakada snapshot'ın servis edilmeye devam ettiğini doğrulayan tek test.

### Info

3. **A-02 — Interpretasyon testleri yalnızca stub; gerçek kütüphane hiç koşmuyor**
   * `tests/deep_learning/test_tft_interpretation.py:11-76`. `requirements.txt:27`
     `pytorch-forecasting>=1.0.0` (aralık pin) → sürüm kayması bu testlerle yakalanmaz.
   * Öneri: opsiyonel, `online`-dışı ama yavaş işaretli gerçek mini-TFT smoke testi.

4. **A-06 — Worker Stage 5.5 explain persist bloğu testsiz**
   * `worker/tasks.py:720-756` (pop / `run_id` / `explain_json` / `tft_explain_generated`); repo genelinde
     `include_explain` geçen test yok. Bayrak sırası ve rollback davranışı yalnızca okuma ile doğrulandı.
   * Öneri: mock `generate_tft_analysis` + in-memory session ile küçük bir worker-stage testi
     (`explain_json` doluyken `payload_json`'da `explain` yok; persist hatası → rollback → endpoint 404).

5. **A-07 — Frontend bütçe marjı çok ince (uyarı)**
   * Bu branch sonrası: CSS 13.990/14.000 B (10 B), dashboard JS 279.406/280.000 B (594 B);
     `check:budgets` exit 0. Sonraki `workspace.css`/dashboard dokunuşu CI'ı kırabilir.

### Nitpick

6. **A-03 — `pct` özellikte ilk değer 0 ise tüm `recent` bloğu düşüyor** (`tft_explain.py:142-149`);
   mutlak `last_value`/`change` hâlâ tanımlı — `change_pct: null` ile blok korunabilir.
7. **A-04 — Horizon etiket/gün sayısı "5-day" hardcode** (`tft_explain.py:32-34`); config'ten türetilmeli
   ya da schema sabiti olarak belgelenmeli (bugün doğru).

## Review Focus kontrol listesi (plan)

1. Stale snapshot -> 404 / canlı inference yok: **PASS** (endpoint testleri + `_forbid_live_inference`).
2. Fail-open: **PASS** (3 katman + worker + UI boş durumu; GEMINI invariant'ı eklendi).
3. Date/length mismatch: **PASS** (`ValueError` + hot-path `try` sarmalı; testler).
4. Non-finite değerler: **PASS** (weight drop / attention zero; `allow_nan=False` testleri).
5. Bilinmeyen feature: **PASS** (`describe_feature` fallback; kolon yoksa `recent` yok).
6. Bozuk `importance_json`: **PASS** (`_safe_json_load` -> `{}` + MRMR fallback; testler).

## Sınırlar

* Gerçek checkpoint + gerçek DB ile uçtan uca explain üretimi bu incelemede koşulmadı (kontrat kaynak
  düzeyinde ve testlerle doğrulandı).
* Postgres migration yolu runtime'da denenmedi; SQLite yolu ve PG `ADD COLUMN IF NOT EXISTS` deseni
  mevcut repo standardıyla aynı.

## Özet

Disiplinli, iyi test edilmiş bir değişiklik: fail-open katmanları, snapshot/freshness kontratı,
promotion-gate sırası, migration idempotency ve frontend entegrasyonu bağımsız olarak doğrulandı.
Önerilen aksiyon sırası: **A-01** (pre-merge), sonra opsiyonel kapsam testleri (A-05, A-06) ve
farkındalık notu (A-07).

---

## Post-Remediation & Final Verification Addendum (2026-10-16)

Yukarıdaki ilk inceleme `283d25f` commit'i üzerinde yapılmıştır. Sonrasında uygulanan iki sertleştirme
commit'i (`b391591` + son fail-open / savunmacı unpack tamamlaması) ile tüm açık maddeler kapatılmıştır:

* **A-01 (`interpret_batch` / `_run_interpretation` eval-mode + stub testi):** `_run_interpretation`
  içinde `was_training` yakalanıp `model.eval()` zorlanmakta ve `finally` bloğunda `model.train()`
  restore edilmektedir; `test_tft_interpretation.py` içinde `training_during_call is False` ve
  sonrasında `model.training is True` açıkça doğrulanmaktadır.
* **A-03 (`pct` özellikte `first_value == 0.0` durumu):** `tft_explain.py:_recent_block` içinde tüm
  `recent` bloğunu düşürmek yerine `change_pct: None` ile `last_value` ve `change` korunmaktadır
  (`test_pct_feature_with_zero_first_value_keeps_recent_with_null_change_pct`).
* **A-05 (`trained_at > snapshot.generated_at` ve `quality_gate_passed` kapsamı):**
  `test_tft_explain_endpoint.py` ve `test_tft_model_summary_promotion.py` ile hem staleness 404 hem
  de reddedilen aday modelin (`quality_gate_passed=False`) snapshot'ı geçersiz kılmadığı test edilmiştir.
* **A-06 & Worker Fail-Open Retry (`_persist_tft_report_snapshot`):** `worker/tasks.py` içinde
  `explain_json` yazımı hata verirse `session.rollback()` sonrası `explain_json=None` ile otomatik
  tekrar yazma (downgrade retry) uygulanmıştır (`test_persist_retries_with_explain_none_when_explain_write_fails`).
  Böylece isteğe bağlı açıklamadaki bir serileştirme/yazım hatası tahmin snapshot'ını asla düşürmez.
* **Predictor Savunmacı Batch Unpack (`predictor.py:_latest_window_explain`):** `x, _ = next(iter(dataloader))`
  yerine `batch = next(iter(dataloader)); x = batch[0] if isinstance(batch, (tuple, list)) else batch`
  kullanılarak bare-dict batch uyumluluğu sağlanmıştır (`test_latest_window_explain_accepts_bare_dict_batch`).
* **Güncel Doğrulama Damgası (Son Commit):**
  * Backend offline suite (`py -3 -m pytest tests/ -q -m "not online"`): **631 passed**
  * Frontend lint (`npm run lint`): **0 warnings**
  * Frontend unit/integration (`npx vitest run`): **14 files, 110 passed**
  * Frontend build (`npm run build`): **OK** (`tsc && vite build && prerender`)
  * Frontend budgets (`npm run check:budgets`): **OK** (`initialJS: 127,036 / 190,000 B`,
    `css: 13,990 / 14,000 B`, `dashboardJS: 279,564 / 280,000 B`)

