# TFT-ASRO XAI (`/explain`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** TFT-ASRO'nun her tahmin için VSN ağırlıklarını ve attention dağılımını (`explain_json`) snapshot'a yazıp `GET /api/explain/{symbol}` ile JSON olarak sunmak; aynı çekirdekle kırık `get_variable_importance` bug'ını düzeltmek; Overview'da "Model focus" kartında göstermek.

**Architecture:** Çıkarım çekirdeği `deep_learning/models/tft_copper.py`'de (forward + `interpret_output` + isim eşleme); zengin JSON üretimi `app/tft_explain.py`'de (tarih/recent/etiket); hesaplama worker Stage 5.5'te, API yalnızca saklanan snapshot'ı okur (canlı inference yok). Malzeme snapshot başına saklanır; commentary entegrasyonu ve karakter/chat bu turun dışında — veri onların zeminine göre tasarlandı.

**Tech Stack:** Python 3.12 / FastAPI / SQLAlchemy / pytorch-forecasting **1.8.0** (pinli) / React 18 + Vite + Tailwind / vitest.

**Spec:** Bu sohbette onaylanan tasarım (kullanıcının son 5 maddesi dahil); ayrı spec dosyası yok.

## Global Constraints

- `pytorch-forecasting==1.8.0` sabit (bkz. `backend/constraints-tft.txt`); sürüm yükseltilmez.
- API handler'ları TFT inference **çalıştırmaz**; XAI worker Stage 5.5'te hesaplanır (snapshot-only).
- `explain_json` yazımı **opsiyoneldir**: yorumlama hatası tahmini, snapshot kaydını ve quality-gate'i asla düşürmez.
- `variable_importance` yanıt sözleşmesi `[{feature, importance}]` olarak aynen korunur; VSN yoksa MRMR artifact fallback'i çalışır.
- `explain_json` şemasında `schema_version`, `model.checkpoint`, `reference_price_date`, `generated_at`, `run_id` alanları bulunur; horizon etiketi "5-day (primary)"; `notes` "not causal" içerir.
- Offline CI filtresi: `py -3 -m pytest tests/ -v -m "not online"` (bu Windows makinede `python` PATH'te yok; `py -3` kullanılır).
- Frontend: `npm test`, `npm run lint` (0 warning), `npm run build`, `npm run check:budgets` (dashboard JS ≤280 KB gzip).
- Migration'lar idempotent: Postgres `ADD COLUMN IF NOT EXISTS`, SQLite `_sqlite_columns` kontrolü (`_ensure_columns` deseni).
- `app/quality_gate.py` değiştirilmez.
- Commit kuralı: her implementer **yalnızca görevde adı geçen dosyaları** `git add` etmeli; `git add -A` / `git add .` **yasak** (worktree kökünde izlenmeyen `AGENTS.md` var, sweep edilmemeli).

## Review Focus

1. **Bayat snapshot** (yeni PriceBar veya yeni promote edilen model) → `/explain` eski vintage'ı da canlı inference'ı da **sunmaz**, 404 + gerekçe. (Task 4 testi)
2. **Fail-open**: extraction hatası veya eski snapshot'ta `explain_json=null` → snapshot aynen yazılır; endpoint 404, UI boş durum. (Task 2 + Task 4 testleri)
3. **Tarih/uzunluk uyumsuzluğu**: attention serisi ile pencere günleri birebir değilse payload üretilmez (yanlış gün etiketi basılmaz). (Task 2 testleri)
4. **Non-finite değerler**: NaN/Inf ağırlık düşürülür, JSON geçerli kalır; sıfır attention serisi yine üretilir. (Task 2 testi)
5. **Bilinmeyen özellik adları veya frame'de olmayan sütun** → `describe_feature` fallback etiketi; `recent` atlanır, çökme yok. (Task 2 testi)
6. **Malformed/eksik `importance_json`** → summary MRRR fallback'ine düşer, sözleşme bozulmaz. (Task 5 testleri)

---

### Task 1: Interpretation çekirdeği + `get_variable_importance` fix

**Files:**
- Modify: `backend/deep_learning/models/tft_copper.py` (~825-876: `get_variable_importance` yeniden yaz, `interpret_batch` ekle; `get_attention_weights` **sil** — grep edildi, başka referansı yok)
- Test: `backend/tests/deep_learning/test_tft_interpretation.py` (yeni)

**Interfaces (Produces):**
- `interpret_batch(model, x: dict, attention_prediction_horizon: int = 0) -> dict` → `{"encoder": [{"feature","weight"}], "decoder": [...], "static": [...], "attention": [float,...], "encoder_length": int}`; attention `[: model.hparams.max_encoder_length]` ile encoder kısmına kırpılır; grup adları `model.encoder_variables/decoder_variables/static_variables` sırasıyla eşlenir; uzunluk uyuşmazsa `ValueError`.
- `get_variable_importance(model, val_dataloader=None) -> Dict[str, float]` → batch'ler üzerinde encoder ağırlık ortalaması, toplamı 1'e normalize, azalan sıralı; loader yok/boşsa `{}`.

**Compatibility note:**
- `backend/deep_learning/training/trainer.py:1046` çağrısı: `get_variable_importance(evaluation_model, val_dataloader=val_dl)` — imza birebir uyumlu kalmalı.
- Gerçek `interpret_output` davranışını doğrulamak için kurulu pytorch-forecasting 1.8.0 kaynağını oku (`model.interpret_output(out, reduction="none", attention_prediction_horizon=...)`; çıktı anahtarları `encoder_variables`, `decoder_variables`, `static_variables`, `attention`). Not: `reduction="none"` attention'ı örnek başına yeniden normalize eder.

- [ ] **Step 1: Failing testleri yaz** — `interpret_batch` named-group mapping + attention + `encoder_length`; uzunluk uyuşmazlığında `ValueError`; iki batch ortalaması → `{"A": ~0.6, "B": ~0.4}` + sıralama; loader yoksa `{}`. Stub model: `encoder_variables`/`decoder_variables`/`static_variables` listeleri, `hparams = SimpleNamespace(max_encoder_length=3)`, `__call__` → `{"encoder_lengths": torch.tensor([3])}`, `interpret_output` → tensor sözlüğü (`encoder_variables` `(1,N)`, `attention` `(1,3)`).

Örnek test iskeleti:

```python
class _StubModel:
    encoder_variables = ["HG=F_close_sma_5", "HG=F_sentiment_mean"]
    decoder_variables = []
    static_variables = ["symbol_id"]
    hparams = SimpleNamespace(max_encoder_length=3)

    def __call__(self, x):
        return {"encoder_lengths": torch.tensor([3]), "decoder_lengths": torch.tensor([5])}

    def interpret_output(self, out, reduction="none", attention_prediction_horizon=0):
        return {
            "encoder_variables": torch.tensor([[0.7, 0.3]]),
            "decoder_variables": torch.zeros((1, 0)),
            "static_variables": torch.tensor([[1.0]]),
            "attention": torch.tensor([[0.2, 0.3, 0.5]]),
        }
```

- [ ] **Step 2: Fail doğrula** — Run: `py -3 -m pytest tests/deep_learning/test_tft_interpretation.py -v` → **FAIL** (ImportError).
- [ ] **Step 3: Implement et** — `out = model(x)` `torch.no_grad()` içinde; `model.interpret_output(out, reduction="none", attention_prediction_horizon=...)`; tensorlar `[0].detach().cpu().numpy()`; zip kontrolü `ValueError("...names/weights mismatch...")`; `encoder_length` `out["encoder_lengths"][0]`. Eski `get_variable_importance`'ın `predict()`+dict varsayımı tamamen gider. `get_attention_weights` silinir.
- [ ] **Step 4: Pass doğrula** — aynı komut → **PASS**; ardından `py -3 -m pytest tests/ -q -m "not online"` (tam suite yeşil kalmalı).
- [ ] **Step 5: Commit** — `git add deep_learning/models/tft_copper.py tests/deep_learning/test_tft_interpretation.py` · `git commit -m "fix(tft): repair interpret_output-based variable importance; add interpretation core"`

### Task 2: Explain payload builder

**Files:**
- Create: `backend/app/tft_explain.py`
- Test: `backend/tests/test_tft_explain_payload.py` (yeni)

**Interfaces (Produces):**
- `EXPLAIN_SCHEMA_VERSION = 1`, `TOP_RECENT = 5`, `RECENT_BASIS_SESSIONS = 5`
- `build_explain_payload(*, interpretation, window_dates, feature_frame, reference_price_date, symbol, checkpoint, trained_at=None, run_id=None, generated_at=None) -> dict` — `len(window_dates) == len(attention)` değilse `ValueError`; `len(feature_frame) == len(window_dates)` kontrolü; çıktı: `{schema_version, symbol, reference_price_date, generated_at, run_id, model{checkpoint, trained_at}, horizon{label:"5-day cumulative log return (primary)", days:5}, layout_hash{encoder,decoder,static: sha1[:12]}, variable_selection{method, encoder[], decoder[], static[]}, attention{method, step{kind:"first_forecast_day", horizon_index:0}, window{start,end,days}, top_days[](5), series[]}, notes}`.
- Entry: `{feature, weight(round 6), label, description, category, time_horizon}`; yalnız `TOP_RECENT` içindeki encoder girdilerine `recent` eklenir: `{basis:"5 sessions", kind:"pct"|"abs", last_value, change, change_pct?|None, as_of}`.
- `kind` kuralı: `Embedding` kategorisi veya `emb_` öneki → `recent` yok; level token'lar (`price,sma,ema,close,open,high,low,stock,tonnes,inventory,warehouse,volume`) → `"pct"`; diğerleri → `"abs"`.
- Non-finite ağırlık düşürülür; `recent` değerleri non-finite ise eklenmez; etiketler `app.features.describe_feature`.
- `try_build_explain_payload(**kwargs) -> dict | None` — tüm istisnaları yakalar, `logger.warning`, `None` döner.

- [ ] **Step 1: Failing testler** — (a) happy path: 3 encoder (biri `HG=F_emb_03`), 1 decoder, 1 static, 3 tarih; assert: `schema_version==1`, gruplar ayrı, encoder azalan sıralı, `top["recent"]["kind"]=="pct"` ve `basis=="5 sessions"`, emb girdisinde `"recent" not in entry`, `series[-1]=={"date": son tarih, "weight": ~0.5}`, `top_days[0]["date"]==son tarih`, `step.kind=="first_forecast_day"`, `notes` "not causal" içerir; (b) NaN ağırlık → listeden düşer; (c) uzunluk uyuşmazlığı → `build_...` `ValueError` / `try_...` `None`; (d) frame'de olmayan sütun → `recent` anahtarı yok.

Örnek happy-path iskeleti:

```python
interp = {
    "encoder": [
        {"feature": "HG=F_close", "weight": 0.5},
        {"feature": "HG=F_sentiment_mean", "weight": 0.3},
        {"feature": "HG=F_emb_03", "weight": 0.2},
    ],
    "decoder": [{"feature": "dow_sin", "weight": 1.0}],
    "static": [{"feature": "symbol_id", "weight": 1.0}],
    "attention": [0.2, 0.3, 0.5],
    "encoder_length": 3,
}
frame = pd.DataFrame(
    {"HG=F_close": [6.0, 6.1, 6.061], "HG=F_sentiment_mean": [0.1, 0.2, 0.15], "HG=F_emb_03": [0.0, 1.0, 2.0]},
    index=["2026-10-10", "2026-10-13", "2026-10-14"],
)
payload = build_explain_payload(
    interpretation=interp, window_dates=["2026-10-10", "2026-10-13", "2026-10-14"],
    feature_frame=frame, reference_price_date="2026-10-14",
    symbol="HG=F", checkpoint="best.ckpt", generated_at=datetime(2026, 10, 15, tzinfo=timezone.utc),
)
```

- [ ] **Step 2: Fail doğrula** — `py -3 -m pytest tests/test_tft_explain_payload.py -v` → **FAIL**.
- [ ] **Step 3: Implement et** — `_recent_kind(feature, category)`, `_sanitize_weight`, `_layout_hash(names) -> sha1[:12]` yardımcıları; `generated_at` default `datetime.now(timezone.utc)`; tüm dönüş değerleri JSON-safe (float/round/isoformat).
- [ ] **Step 4: Pass** — aynı komut + tam suite.
- [ ] **Step 5: Commit** — `git add app/tft_explain.py tests/test_tft_explain_payload.py` · `git commit -m "feat(tft-xai): add explain payload builder"`

### Task 3: Predictor + worker wiring, `explain_json` saklama

**Files:**
- Modify: `backend/deep_learning/inference/predictor.py` (`TFTPredictor.predict` ~330: `include_explain: bool = False` param; `generate_tft_analysis` ~903: paramı geçir ve `explain`'i `prediction` sözlüğünden **pop'layıp** `raw_result["explain"]`'e taşı)
- Modify: `backend/worker/tasks.py` (Stage 5.5: `generate_tft_analysis(session, "HG=F", include_explain=True)`; persist bloğunda `explain_payload = tft_report.pop("explain", None)`, dict ise `run_id` ekle, `TFTPredictionSnapshot(..., explain_json=explain_payload)`; `result["tft_explain_generated"] = isinstance(explain_payload, dict)`)
- Modify: `backend/app/models.py` (`TFTPredictionSnapshot.explain_json = Column(JSON, nullable=True)`)
- Modify: `backend/app/db.py` (`TFT_MODEL_METADATA_COLUMNS` yanına `TFT_PREDICTION_SNAPSHOT_COLUMNS = [("explain_json", "JSON", "JSON")]` + `_run_migrations` içinde `_ensure_columns(conn, "tft_prediction_snapshots", ...)` çağrısı, tablo-yok hatası için diğer bloklar gibi try/except)
- Test: `backend/tests/test_tft_snapshot_explain_roundtrip.py` (yeni)

**Interfaces:** Consumes Task 1 `interpret_batch`, Task 2 `try_build_explain_payload`. Predictor'da açıklama yolu: encoder penceresi frame'i (predict içinde `master_df.tail(encoder_length)`'ten üretilen `history` — **değişken adını kaynakta doğrula**, plan metni `history` diyor), `x, _ = next(iter(dl))`; `model.eval()`; `try_build_explain_payload(interpretation=interpret_batch(model, x), window_dates=[self._date_label(d) for d in history.index], feature_frame=history, reference_price_date=reference_price_date, symbol=symbol, checkpoint=str(self.cfg.training.best_model_path), trained_at=(self.get_model_metadata(session) or {}).get("trained_at"), generated_at=datetime.now(timezone.utc))`; sadece `include_explain` iken; hata halinde `None`.

- [ ] **Step 1: Failing test** — sqlite in-memory: `TFTPredictionSnapshot.__table__.create(engine)`; `explain_json={...}` ile insert edip query ile aynı dict'i geri oku (JSON kolon round-trip).
- [ ] **Step 2: Fail doğrula** — `py -3 -m pytest tests/test_tft_snapshot_explain_roundtrip.py -v` → **FAIL** (no such column).
- [ ] **Step 3: Implement et** (yukarıdaki dört dosya).
- [ ] **Step 4: Pass + suite** — hedef test + `py -3 -m pytest tests/ -q -m "not online"`.
- [ ] **Step 5: Commit** — `git add deep_learning/inference/predictor.py worker/tasks.py app/models.py app/db.py tests/test_tft_snapshot_explain_roundtrip.py` · `git commit -m "feat(tft-xai): persist per-forecast explanation with TFT snapshots"`

### Task 4: `GET /api/explain/{symbol}` + tazelik helper refactorı

**Files:**
- Modify: `backend/app/main.py` (typing importuna `Dict, Any`; `_as_utc`/`_parse_ref_date` handler-içi fonksiyonlarını modül seviyesine çıkar — isim çakışması kontrol et; yeni `_tft_snapshot_invalid_reasons(session, symbol, snapshot) -> list[str]` — mevcut snapshot-geçerlilik mantığı; `/api/analysis/tft/{symbol}` bu helper'ı kullanacak şekilde refactor; yeni route `/api/analysis/tft/{symbol}` yakınına)
- Test: `backend/tests/test_tft_explain_endpoint.py` (yeni; fixture desenini `tests/test_tft_api_snapshot_freshness.py`'den kopyala)

**Interfaces (Produces):** `GET /api/explain/{symbol}` → `response_model=Dict[str, Any]`; `explain_json` dict'ini aynen döner. 404 detayları: `"No TFT snapshot for {symbol}..."` / `"TFT snapshot is stale: {reasons}"` / `"Explanation not available ... next pipeline refresh."` Canlı inference **yok**.

- [ ] **Step 1: Failing testler** — (a) taze snapshot + PriceBar aynı gün + `explain_json` dolu → 200 ve gövde eşitliği; (b) `explain_json=None` → 404; (c) snapshot'tan yeni PriceBar → 404 ve canlı inference çağrısının `pytest.fail` ile monkeypatch edilmesi (main.py'nin gerçekte hangi yolu import ettiğini kaynaktan oku ve o hedefi patch'le); (d) snapshot yok → 404.
- [ ] **Step 2: Fail doğrula** — `py -3 -m pytest tests/test_tft_explain_endpoint.py -v` → **FAIL**.
- [ ] **Step 3: Implement et** (helper + refactor + route; TFT endpoint davranışı birebir korunur).
- [ ] **Step 4: Pass + regresyon** — hedef test + `py -3 -m pytest tests/test_tft_api_snapshot_freshness.py tests/test_tft_explain_endpoint.py -v`.
- [ ] **Step 5: Commit** — `git add app/main.py tests/test_tft_explain_endpoint.py` · `git commit -m "feat(api): expose /api/explain/{symbol} for TFT model focus"`

### Task 5: Global importance'ın metadata'ya yazımı + summary tercihi

**Files:**
- Modify: `backend/deep_learning/training/trainer.py` (~1592-1613 promotion bloğu; yeni saf fonksiyon `_importance_metadata_payload(result) -> str` → `{"schema_version":1, "encoder": result["variable_importance"], "trained_at", "checkpoint"}`, hem update hem insert dalında `importance_json=` yaz)
- Modify: `backend/app/models.py` (`TFTModelMetadata.importance_json = Column(Text, nullable=True)`)
- Modify: `backend/app/db.py` (`TFT_MODEL_METADATA_COLUMNS` listesine `("importance_json", "TEXT", "TEXT")`)
- Modify: `backend/app/main.py` (`/api/models/tft/summary` ~1746: önce `meta.importance_json` → `encoder` dict → azalan sıralı, cap 20, mevcut zenginleştirme şekliyle; yoksa mevcut MRMR artifact bloğu aynen)
- Test: `backend/tests/test_tft_model_summary_promotion.py` (genişlet) + `backend/tests/deep_learning/test_trainer_importance.py` (yeni, `_importance_metadata_payload` için)

- [ ] **Step 1: Failing testler** — (a) `_importance_metadata_payload` çıktısı parse edilince `encoder == result["variable_importance"]`; (b) summary: `SimpleNamespace` metadata + FakeSession (mevcut dosya stili) `importance_json=json.dumps({"schema_version":1,"encoder":{"a":0.4,"b":0.6}})` → `[(feature, importance)] == [("b",0.6),("a",0.4)]`; (c) `importance_json="not json"` + tmp MRMR dosyası (`tmp_path/"latest"/"mrmr_results.json"`, `config_json={"feature_store":{"artifact_dir":str(tmp_path)}}`) → MRMR listesi; (d) importance yok + artifact yok → `[]`.
- [ ] **Step 2: Fail doğrula** — `py -3 -m pytest tests/test_tft_model_summary_promotion.py tests/deep_learning/test_trainer_importance.py -v`.
- [ ] **Step 3: Implement et.**
- [ ] **Step 4: Pass + suite.**
- [ ] **Step 5: Commit** — `git add deep_learning/training/trainer.py app/models.py app/db.py app/main.py tests/test_tft_model_summary_promotion.py tests/deep_learning/test_trainer_importance.py` · `git commit -m "fix(tft): persist VSN importance to metadata and prefer it in model summary"`

### Task 6: Frontend — `ModelFocusPanel` + Overview kartı

**Files:**
- Create: `frontend/src/utils/driverCategories.ts` — `driverCategoryTone`/`driverCategoryLabel` haritalarını `frontend/src/pages/OverviewPage.tsx` satır ~52-66'dan buraya taşı (export et); OverviewPage ve panel ikisi de buradan içe aktarır.
- Modify: `frontend/src/types.ts` (`ExplainRecent`, `ExplainFeatureEntry`, `ExplainAttentionPoint`, `ExplainResponse`)
- Modify: `frontend/src/api.ts` (`fetchExplain(symbol): Promise<ExplainResponse | null>` — `validateStatus: (s) => s === 200 || s === 404`, 404 → `null`)
- Create: `frontend/src/components/ModelFocusPanel.tsx` + `frontend/src/components/ModelFocusPanel.test.tsx`
- Modify: `frontend/src/pages/OverviewPage.tsx` (state + `loadData`'daki `Promise.allSettled`'a `fetchExplain`; `loadErrors['Model focus']`; Neural analysis'in **yanına** `<GlassCard id="tft-focus" title="Model focus (TFT-ASRO)" icon={Brain} colSpan={4}>`)
- Modify: `./api` mock'layan mevcut testler (`grep -l "vi.mock('./api'"` ile bul; `fetchExplain: vi.fn(() => new Promise(() => {}))` ekle)

**Panel içeriği (pinned metinler):** boş durum `"Model focus will appear after the next model refresh."`; top-5 encoder: etiket + kategori chip'i (`driverCategories.ts`) + ağırlık barı + recent satırı (`pct`: `"+2.10% over last 5 sessions"`, `abs`: `"+0.120 over last 5 sessions"`); attention başlığı `"Attention · forecast day 1 (t+1)"`, 60 çubuk (`title="{date}: {pct}%"`, `data-testid="model-focus-attention"`); meta `"Reference {reference_price_date}"`; dipnot olarak `data.notes`; `<details>Raw JSON</details>` içinde `JSON.stringify(data, null, 2)`.

- [ ] **Step 1: Failing test** — fixture payload ile render: sürücü etiketi, `forecast day 1`, `not causal` metni, `Last 5 sessions` satırı görünür; `data={null}` → boş durum metni.
- [ ] **Step 2: Fail doğrula** — `npx vitest run src/components/ModelFocusPanel.test.tsx` → **FAIL**.
- [ ] **Step 3: Implement et** (util taşıma + panel + types/api/OverviewPage + mock güncellemeleri).
- [ ] **Step 4: Pass + kalite** — `npx vitest run src/components/ModelFocusPanel.test.tsx` → PASS; `npm test`; `npm run lint`; `npm run build`; `npm run check:budgets`.
- [ ] **Step 5: Commit** — `git add` (görevde adı geçen tüm dosyalar) · `git commit -m "feat(overview): add TFT model focus panel"`

### Task 7: Dokümantasyon

**Files:** Modify `README.md` (API bölümüne `/api/explain/{symbol}` + XAI semantiği: ham model odağı, nedensellik değil, snapshot-only, 404 davranışı) · Modify `GEMINI.md` (tek invariant satırı: explain optional; snapshot'ı asla düşürmez; ham/pre-calibration çerçeve; 5 günlük birincil horizon etiketi).

- [ ] **Step 1: Yaz**
- [ ] **Step 2: Commit** — `git add README.md GEMINI.md` · `git commit -m "docs: document TFT explain endpoint and XAI semantics"`

---

**Not:** `/api/explain` bu checkout'ta **yeni** bir uçtur. İlk pipeline çalışmasına kadar panel boş durumda kalır; checkpoint bulunan bir makinede tek seferlik `generate_tft_analysis(..., include_explain=True)` çalıştırıp JSON'u birlikte inceleyebiliriz.

**Out of scope (bu plana dahil değil):** commentary prompt entegrasyonu (faz 2), SVG karakter/chat arayüzü, `/explain` için canlı inference fallback'i, `as_of` tarihsel sorgu.
