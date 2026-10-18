# Cuprum XAI & Companion — Agent Orders (2026-10-18)

- **Çalıştırma Şekli:** İkisini **aynı anda (paralel)** ve doğrudan **BUILD modunda** çalıştırabilirsin.
- **Git / Branch İzolasyonu:** Her iki ajana da kendi özel branch'i (`feat/xai-commentary-core` ve `feat/cuprum-companion-contract`), ortak klasörde çalışıyorlarsa çakışmamaları için otomatik `git worktree` izolasyonu, sadece kendi dosyalarını `git add` etme kuralı ve iş bitince kendi branch'ini commit + push etme görevi eklendi. İkisi bitince ben (Antigravity) branch'lerini denetleyip birleştireceğim.

---

## ORDER #1 — DeepSeek V4.1 Flash (BUILD Mode)

```markdown
# AGENT ORDER: Backend XAI-Commentary Integration (Phase 2 Core)
**Assigned Agent:** DeepSeek V4.1 Flash (Run directly in BUILD mode — plan is finalized)
**Lead Architect & Decision Maker:** Antigravity (owns visual assets, frontend UI, and final branch merge/review; GLM 5.3 Flash runs in parallel on a separate branch).
**Assigned Branch:** `feat/xai-commentary-core`

## 1. Git, Branch & Worktree Isolation Protocol (CRITICAL — Run First)
Another agent (GLM 5.3 Flash) and Lead Architect (Antigravity) are working in parallel. Follow these exact Git rules:
1. Check your current working directory:
   - If you are already inside an isolated OpenCode/Codex worktree (i.e. path is NOT `D:/dev/projects/copper-mind`), sync and switch to your dedicated branch:
     ```bash
     git fetch origin main
     git checkout -B feat/xai-commentary-core origin/main
     ```
   - If your working directory IS the shared root `D:/dev/projects/copper-mind`, DO NOT run `git checkout` in the root directory (that would disrupt parallel agents). Instead, create an isolated git worktree for your branch and run all edits/tests/commits inside that worktree directory:
     ```bash
     git fetch origin main
     git worktree add -B feat/xai-commentary-core ../copper-mind-deepseek origin/main
     ```
     (Then perform all file edits, pytest runs, and git operations inside `../copper-mind-deepseek`).
2. **Strict File Boundaries:** You may ONLY modify these 4 files:
   - `backend/app/tft_explain.py`
   - `backend/app/commentary.py`
   - `backend/worker/tasks.py`
   - `backend/tests/test_commentary.py`
   DO NOT touch `backend/app/main.py`, `backend/app/companion.py`, `backend/app/quality_gate.py`, or any `frontend/` files.
3. **Forbidden Git Commands:** NEVER run `git add .`, `git add -A`, `git commit -a`, or push to `main`.

## 2. Architectural Invariants (`GEMINI.md` & `AGENTS.md`)
- **Primary vs. Diagnostic Horizon Separation (`GEMINI.md` §4):** XGBoost inputs represent the T+1 daily simple-return horizon; TFT-ASRO inputs represent the 5-day cumulative log-return (primary) horizon. Never mix or conflate their horizons in prompts or fallback text.
- **Non-Causal Explainability & Fail-Open Contract (`GEMINI.md` §4):** TFT-ASRO `explain_json` (VSN weights + encoder attention) is optional, pre-calibration model focus — NEVER causality. If `tft_explain` or `tft_summary` is `None` or malformed, commentary generation must degrade gracefully and never fail a pipeline run.

## 3. Implementation Tasks

### Task A: Add `summarize_explain_for_commentary` in `backend/app/tft_explain.py`
Add a pure, fail-open helper function:
```python
def summarize_explain_for_commentary(explain_payload: Optional[dict]) -> Optional[dict]:
```
- If `explain_payload` is not a `dict`, return `None` (never raise).
- Extract and sanitize:
  - `horizon_label`: `str` (default `"5-day cumulative log return (primary)"`)
  - `reference_price_date`: `Optional[str]`
  - `top_encoder_signals`: top 5 items from `variable_selection.encoder`, each formatted as `{feature, label, category, weight_pct, recent_summary}` where `weight_pct` is `round(weight * 100, 1)` and `recent_summary` formats `recent` if present (e.g. `"+2.10% over 5 sessions"` for `kind == "pct"` with finite `change_pct`, `"+0.120 over 5 sessions"` for `kind == "abs"` with finite `change`, or `None`).
  - `memory_anchors`: top 3 items from `attention.top_days`, each formatted as `{date, weight_pct}` where `weight_pct` is `round(weight * 100, 1)`.
  - `window_span`: `{start, end, days}` from `attention.window`.

### Task B: Embed XAI Context & Section Parser in `backend/app/commentary.py`
1. Extend `_generate_commentary_and_stance`, `generate_commentary`, `generate_and_save_commentary`, and `_build_commentary_template_fallback` with optional parameters:
   - `tft_explain: Optional[dict] = None`
   - `tft_summary: Optional[dict] = None` (optional dict with keys `primary_forecast_return`, `primary_forecast_q10`, `primary_forecast_q90`, `direction`, `risk_level`, `reference_price_date`)
   All existing call sites without these arguments MUST continue to work unchanged.
2. In `_generate_commentary_and_stance`:
   - Call `summarize_explain_for_commentary(tft_explain)`.
   - Append a dedicated `TFT-ASRO 5-DAY PRIMARY HORIZON & XAI FOCUS (PRE-CALIBRATION, NON-CAUSAL):` block to `user_prompt` when `tft_summary` or summarized `tft_explain` is available:
     - Explicitly distinguish XGBoost (T+1 next-day simple return) from TFT-ASRO (5-day cumulative log return, primary horizon).
     - List the top 5 VSN encoder signals (with their weight % and 5-session recent delta) and top 3 historical attention memory anchor dates (`memory_anchors`).
     - Instruct the LLM in `ANALYTICAL FRAMEWORK` to weave the TFT 5-day outlook, VSN focus signals, and historical memory anchor dates into `RISKS`, `OPPORTUNITIES`, and `SYNTHESIS`, while framing VSN/attention strictly as model focus ("where the network looked"), never as causal proof.
3. In `_build_commentary_template_fallback`:
   - When `summarize_explain_for_commentary(tft_explain)` is available, include a concise non-causal `XAI Focus:` line referencing the top encoder signal(s) and top attention anchor date before the disclaimer.
4. Add a pure helper function in `backend/app/commentary.py`:
   ```python
   def parse_commentary_sections(commentary: Optional[str], tft_explain: Optional[dict] = None) -> dict:
   ```
   - Parses any generated or fallback `commentary` text into a structured dict for the UI companion:
     - `synthesis`: `str` (extracted from `SYNTHESIS:` or `Summary:` section; falls back to first non-empty paragraph)
     - `risks`: `list[str]` (up to 3 clean bullet strings from `RISKS`)
     - `opportunities`: `list[str]` (up to 3 clean bullet strings from `OPPORTUNITIES`)
     - `xai_brief`: `Optional[str]` (a 1-2 sentence human-readable summary built from `summarize_explain_for_commentary(tft_explain)` describing the top VSN signals and top attention memory dates with non-causal framing, or `None` if `tft_explain` is unavailable)
     - `bias_warning`: `Optional[str]` (extracted from `BIAS ACKNOWLEDGMENT:` or `Bias warning:`)

### Task C: Wire Stage 5.5 `explain_payload` into Stage 6 in `backend/worker/tasks.py`
- In Stage 5.5 (`backend/worker/tasks.py`), preserve `explain_payload` (when `is_forecast_healthy and persisted`) and a compact `tft_commentary_summary` dict (`primary_forecast_return`, `primary_forecast_q10`, `primary_forecast_q90`, `direction`, `risk_level`, `reference_price_date`).
- Pass `tft_explain=explain_payload` and `tft_summary=tft_commentary_summary` into `generate_and_save_commentary(...)` in Stage 6.
- If TFT snapshot or `explain_payload` failed or was degraded, pass `None` so Stage 6 remains 100% fail-open.

### Task D: Unit Tests in `backend/tests/test_commentary.py`
Add tests verifying:
1. `summarize_explain_for_commentary` handles full payloads, missing `recent`, and `None`/malformed input safely.
2. `_generate_commentary_and_stance` includes the TFT 5-day & XAI VSN/attention block in the OpenRouter prompt when `tft_explain` is provided, and still works when `tft_explain=None`.
3. `parse_commentary_sections` cleanly extracts `synthesis`, `risks`, `opportunities`, `xai_brief`, and `bias_warning` from both LLM-formatted and deterministic-fallback commentary strings.

## 4. Verification, Commit & Push (Deliverable to Lead Architect)
1. Run tests from `backend/` (note: on this Windows machine use `py -3`, not `python`):
   ```bash
   py -3 -m pytest tests/test_commentary.py tests/test_tft_explain_payload.py tests/test_worker_tft_explain_wiring.py -v
   ```
2. Stage and commit ONLY your 4 assigned files on `feat/xai-commentary-core`:
   ```bash
   git add backend/app/tft_explain.py backend/app/commentary.py backend/worker/tasks.py backend/tests/test_commentary.py
   git commit -m "feat(commentary): embed TFT XAI focus and section parser into commentary pipeline"
   git push -u origin feat/xai-commentary-core
   ```
3. Report the commit hash and pytest summary back for Antigravity's final integration review.
```

---

## ORDER #2 — GLM 5.3 Flash (BUILD Mode)

```markdown
# AGENT ORDER: Cuprum Companion State Machine & `/api/commentary` Enriched Contract
**Assigned Agent:** GLM 5.3 Flash (Run directly in BUILD mode — plan is finalized)
**Lead Architect & Decision Maker:** Antigravity (owns visual assets, frontend UI, and final branch merge/review; DeepSeek V4.1 Flash runs in parallel on a separate branch).
**Assigned Branch:** `feat/cuprum-companion-contract`

## 1. Git, Branch & Worktree Isolation Protocol (CRITICAL — Run First)
Another agent (DeepSeek V4.1 Flash) and Lead Architect (Antigravity) are working in parallel. Follow these exact Git rules:
1. Check your current working directory:
   - If you are already inside an isolated OpenCode/Codex worktree (i.e. path is NOT `D:/dev/projects/copper-mind`), sync and switch to your dedicated branch:
     ```bash
     git fetch origin main
     git checkout -B feat/cuprum-companion-contract origin/main
     ```
   - If your working directory IS the shared root `D:/dev/projects/copper-mind`, DO NOT run `git checkout` in the root directory (that would disrupt parallel agents). Instead, create an isolated git worktree for your branch and run all edits/tests/commits inside that worktree directory:
     ```bash
     git fetch origin main
     git worktree add -B feat/cuprum-companion-contract ../copper-mind-glm origin/main
     ```
     (Then perform all file edits, pytest runs, and git operations inside `../copper-mind-glm`).
2. **Strict File Boundaries:** You may ONLY create or modify these 3 files:
   - `backend/app/companion.py` (NEW FILE)
   - `backend/app/main.py` (ONLY the `GET /api/commentary` endpoint around lines 1118-1155)
   - `backend/tests/test_companion_contract.py` (NEW FILE)
   DO NOT touch `backend/app/commentary.py`, `backend/app/tft_explain.py`, `backend/worker/tasks.py`, `backend/app/quality_gate.py`, or any `frontend/` files.
3. **Forbidden Git Commands:** NEVER run `git add .`, `git add -A`, `git commit -a`, or push to `main`.

## 2. Architectural Invariants (`GEMINI.md` & `AGENTS.md`)
- **Snapshot-Only & Freshness Isolation (`GEMINI.md` §3 & §4):** `GET /api/commentary` must NEVER trigger live TFT inference. When reading `TFTPredictionSnapshot` to enrich `/api/commentary` with XAI focus and companion state, it must check `_tft_snapshot_invalid_reasons(session, symbol, snapshot)` from `app/main.py`. If the TFT snapshot is missing or stale, `xai_highlights` must degrade to `None` without failing `/api/commentary`.
- **Backward Compatibility:** Existing top-level keys on `GET /api/commentary` (`symbol`, `commentary`, `error`, `generated_at`, `ai_stance`, `generation_mode`, `model_name`, `fallback_reason`) must remain 100% intact.

## 3. Implementation Tasks

### Task A: Create `backend/app/companion.py` ("Cuprum / Külçe Ruh" Deterministic State Engine)
Create `backend/app/companion.py` with no external I/O or LLM calls:
1. `extract_commentary_sections(commentary: Optional[str], explain_payload: Optional[dict] = None) -> dict`:
   - Try importing `parse_commentary_sections` from `app.commentary` inside a `try/except ImportError`; if available, delegate to it, otherwise use a built-in fallback parser in `companion.py` so `companion.py` works independently during parallel execution.
   - Must return a dict with keys:
     - `synthesis`: `str` (from `SYNTHESIS:` or `Summary:` block, or first non-empty paragraph, or `""`)
     - `risks`: `list[str]` (up to 3 clean bullet strings under `RISKS`)
     - `opportunities`: `list[str]` (up to 3 clean bullet strings under `OPPORTUNITIES`)
     - `xai_brief`: `Optional[str]` (1-2 sentence non-causal summary of top VSN encoder features and top attention memory dates from `explain_payload`, or `None`)
     - `bias_warning`: `Optional[str]` (from `BIAS ACKNOWLEDGMENT:` or `Bias warning:`)
2. `build_companion_state(*, symbol: str, commentary_record: Optional[dict], explain_payload: Optional[dict] = None, tft_payload: Optional[dict] = None) -> dict`:
   - Determine directional stance: prefer `commentary_record.get("ai_stance")` if valid (`BULLISH`/`BEARISH`/`NEUTRAL`), fallback to `tft_payload.get("direction")`, default `"NEUTRAL"`.
   - Map `mood`:
     - `"BULLISH"` -> `"molten_bull"`
     - `"BEARISH"` -> `"patina_bear"`
     - `"NEUTRAL"` -> `"tempered_neutral"`
   - Map `energy` from `(tft_payload or {}).get("risk_level")`:
     - `"HIGH"` -> `"volatile"`
     - `"MEDIUM"` -> `"charged"`
     - otherwise -> `"calm"`
   - Map `confidence_band`:
     - `"high"` if `commentary_record` has `generation_mode in ("llm", "llm_repaired")` and `isinstance(explain_payload, dict)`
     - `"medium"` if `commentary_record` is present or `isinstance(explain_payload, dict)`
     - `"low"` otherwise
   - Build `xai_highlights` (when `isinstance(explain_payload, dict)`, else `None`):
     - `horizon_label`: `explain_payload.get("horizon", {}).get("label", "5-day cumulative log return (primary)")`
     - `reference_price_date`: `explain_payload.get("reference_price_date")`
     - `top_signals`: top 5 entries from `explain_payload.get("variable_selection", {}).get("encoder", [])`, keeping `{feature, label, category, weight, recent}`.
     - `memory_anchors`: top 3 entries from `explain_payload.get("attention", {}).get("top_days", [])`, keeping `{date, weight}`
     - `window`: `explain_payload.get("attention", {}).get("window")`
     - `notes`: `explain_payload.get("notes")`
   - Build `headline_quote`: first 1-2 sentences of `sections["synthesis"]` (or `"Cuprum is awaiting the next pipeline refresh."` when commentary is unavailable).
   - Return:
     ```python
     {
         "character": "Cuprum",
         "mood": mood,
         "stance": stance,
         "energy": energy,
         "confidence_band": confidence_band,
         "headline_quote": headline_quote,
         "xai_highlights": xai_highlights,
     }
     ```

### Task B: Enrich `GET /api/commentary` in `backend/app/main.py`
In `get_commentary` (`backend/app/main.py` around line 1118):
- Inside `with SessionLocal() as session:`, also query the latest `TFTPredictionSnapshot` for `symbol` (`order_by(TFTPredictionSnapshot.generated_at.desc()).first()`).
- Check `_tft_snapshot_invalid_reasons(session, symbol, snapshot)`:
  - Only if `snapshot is not None and not _tft_snapshot_invalid_reasons(session, symbol, snapshot)`, extract `explain_payload = snapshot.explain_json if isinstance(snapshot.explain_json, dict) else None` and `tft_payload = snapshot.payload_json if isinstance(snapshot.payload_json, dict) else None`.
  - Otherwise set `explain_payload = None` and `tft_payload = None` (never run live inference).
- Compute `sections = extract_commentary_sections(result["commentary"] if result else None, explain_payload)` and `companion_state = build_companion_state(symbol=symbol, commentary_record=result, explain_payload=explain_payload, tft_payload=tft_payload)`.
- Include `"sections": sections` and `"companion_state": companion_state` in both the available and unavailable return dicts of `GET /api/commentary`.

### Task C: Create `backend/tests/test_companion_contract.py`
Write unit and endpoint tests covering:
1. `build_companion_state` mood mapping (`molten_bull`, `patina_bear`, `tempered_neutral`), energy mapping (`calm`, `charged`, `volatile`), and `xai_highlights` extraction.
2. `extract_commentary_sections` parsing for LLM commentary, fallback commentary, and `None`.
3. `GET /api/commentary` returns `sections` and `companion_state` with `xai_highlights` when a fresh `TFTPredictionSnapshot` exists.
4. `GET /api/commentary` drops `xai_highlights` (`None`) when the `TFTPredictionSnapshot` is stale (e.g., newer `PriceBar` date or newer promoted `TFTModelMetadata.trained_at`), and never calls live TFT inference.

## 4. Verification, Commit & Push (Deliverable to Lead Architect)
1. Run tests from `backend/` (note: on this Windows machine use `py -3`, not `python`):
   ```bash
   py -3 -m pytest tests/test_companion_contract.py tests/test_tft_explain_endpoint.py -v
   ```
2. Stage and commit ONLY your 3 assigned files on `feat/cuprum-companion-contract`:
   ```bash
   git add backend/app/companion.py backend/app/main.py backend/tests/test_companion_contract.py
   git commit -m "feat(api): add Cuprum companion state machine and enrich /api/commentary"
   git push -u origin feat/cuprum-companion-contract
   ```
3. Report the commit hash and pytest summary back for Antigravity's final integration review.
```
