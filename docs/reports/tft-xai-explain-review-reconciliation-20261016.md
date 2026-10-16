# Multi-Agent Review Reconciliation & Final Verification — `codex/tft-xai-explain`

* **Date:** 2026-10-16
* **Branch:** `codex/tft-xai-explain`
* **Reviewers:** DeepSeek V4.1 Flash, Muse Spark 1.3, GLM, GPT, Gemini
* **Invariants Checked:** `GEMINI.md` (§1–§5), `AGENTS.md`

## 1. Uzlaşma ve Kapanış Tablosu

İlk inceleme turu `283d25f` commit'i üzerinde yürütülmüş; ardından `b391591` ve son fail-open sertleştirme commit'i ile tüm pre-merge ve sözleşme bulguları kapatılmıştır:

| Madde | Kaynak | Durum | Uygulanan Çözüm & Kanıt |
| :--- | :--- | :--- | :--- |
| **A-01: `_run_interpretation` eval-mode garantisi** | DeepSeek, GLM | ✅ Kapandı | `tft_copper.py:_run_interpretation` içinde `model.eval()` zorlanıp `finally` bloğunda `model.train()` restore edildi; `test_tft_interpretation.py` içinde `training_during_call is False` ve restore testi eklendi. |
| **Worker `explain_json=None` fail-open retry** | Muse (#1), GPT | ✅ Kapandı | `worker/tasks.py:_persist_tft_report_snapshot` içinde `explain_json` yazımı hata verirse `session.rollback()` sonrası `explain_json=None` ile tekrar yazma (downgrade retry) eklendi (`test_persist_retries_with_explain_none_when_explain_write_fails`). |
| **`get_variable_importance` & `_importance_entries` NaN/Inf filtresi** | Muse (#2), GLM | ✅ Kapandı | VSN ağırlık havuzlamasında örnek/özellik bazlı sonluluk (`np.isfinite` / `math.isfinite`) ve `/api/models/tft/summary` okumasında tekil bozuk değer toleransı sağlandı. |
| **Predictor savunmacı batch unpack** | Muse (#3), GPT | ✅ Kapandı | `predictor.py:_latest_window_explain` içinde `batch = next(iter(dataloader)); x = batch[0] if isinstance(batch, (tuple, list)) else batch` uygulandı (`test_latest_window_explain_accepts_bare_dict_batch`). |
| **A-03: `pct` özellikte `first_value == 0.0` durumu** | DeepSeek, Muse | ✅ Kapandı | `tft_explain.py:_recent_block` içinde `first_value == 0.0` olduğunda tüm `recent` bloğunu düşürmek yerine `change_pct: None` ile `last_value` ve `change` korundu (`test_pct_feature_with_zero_first_value_keeps_recent_with_null_change_pct`). |
| **A-05: Stale / `quality_gate_passed` kapsamı** | DeepSeek, GLM | ✅ Kapandı | `test_tft_explain_endpoint.py` ve `test_tft_model_summary_promotion.py` ile staleness 404 ve reddedilen aday model (`quality_gate_passed=False`) izolasyonu doğrulandı. |
| **UI horizon etiketi & `attention.step` başlığı** | Muse (#5), GLM | ✅ Kapandı | `ModelFocusPanel.tsx` içinde `5-day cumulative log return (primary)` referans notu ve `attention.step.horizon_index` tabanlı dinamik gün başlığı eklendi. |
| **`trained_at` metadata alanı** | Muse (#6) | ✅ Kapandı (FP) | `trainer.py` promotion akışında alan mevcuttur; `None` görülen yer yalnızca test fixture'ıydı (`test_trainer_importance.py` ile sabitlendi). |

## 2. Son Doğrulama Damgası (Final Tree)

| Kontrol | Komut | Sonuç |
| :--- | :--- | :--- |
| **Backend Offline Suite** | `py -3 -m pytest tests/ -q -m "not online"` | **631 passed** (27.6s) |
| **Frontend Lint** | `npm run lint` | **0 warnings** ✓ |
| **Frontend Unit / Integration** | `npx vitest run` | **14 files, 110 passed** ✓ |
| **Frontend Build & Prerender** | `npm run build` | **OK** (`tsc && vite build && prerender`) ✓ |
| **Frontend Bundle Budgets** | `npm run check:budgets` | `initialJS`: **127,036 / 190,000 B** · `css`: **13,990 / 14,000 B** (10 B marj) · `dashboardJS`: **279,564 / 280,000 B** (436 B marj) ✓ |

## 3. İleriki Faz / Follow-Up Havuzu (Non-Blocking)

* **Gerçek mini-TFT smoke testi (A-02):** `pytorch-forecasting` üst sınır kaymalarına karşı opsiyonel slow smoke testi.
* **Frontend bütçe marjı (A-07):** `workspace.css` (10 B marj) ve `dashboardJS` (436 B marj) bir sonraki dashboard/CSS değişikliğinde lazy-load veya bütçe revizyonu gerektirecektir.
* **Erişilebilirlik & Konfigürasyon (A-04 / a11y):** `ModelFocusPanel` attention bar'ları için `sr-only` özet ve `_HORIZON_DAYS` değerinin model konfigürasyonundan türetilmesi.
