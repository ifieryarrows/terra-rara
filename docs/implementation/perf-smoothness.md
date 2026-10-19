# Performance & smoothness pass — working plan (2026-10-19)

Branch: `perf/smoothness`. Basis: three-part read-only audit (interaction jank, chart load, backend memory) performed 2026-10-19; findings reference exact files/lines below.

**Goal:** the app feels immediate and fluid ("oh, what a smooth app") without sacrificing the visual language. Memory wins where they are real; budgets stay green (CSS headroom ≈ 11 B, dashboard JS ≈ 1.2 KB gzip — keep changes size-neutral; landing JS has ~60 KB headroom).

**Invariants:** motion guide durations (160–240 ms UI, 650 ms reveal); GEMINI.md metric/quality-gate rules untouched; API serves snapshots and never runs pipeline work inline; no new dependencies.

## Workstreams

### L — Landing scene pacing (replaces wheel dampener)
- [x] L1 Replaced `createCinematicWheelDampener` (`frontend/src/features/landing/cinematic-scroll.ts`) with a gesture **scene stepper**: one wheel/keyboard gesture advances at most one narrative rest point (`REST = 0 / .24 / .45 / .66 / .97` of the pinned story); native `scrollTo({behavior:'smooth'})` per step; further input during the glide or within the same momentum burst is swallowed (no queueing) until `scrollend`/timeout; at bounds (before hero rest / after CTA rest) input is released to native scrolling. Ctrl-zoom, horizontal-dominant deltas, edits and modified keys pass through. No rAF loop.
- [x] L2 Wired in `CinematicLanding.tsx` (wheel on the scene, keydown on window; guarded by editable focus, shift+arrow selection, bounds, dispose on unmount).
- [x] L3 Rewrote `cinematic-scroll.test.ts` for the new contract (one-step guarantee, swallow during glide, pass-through cases, lock release) — 9/9.
- [x] L4 Verified: `check-experience.mjs` green (programmatic scroll cases untouched); wheel probe (tmp/, not shipped): old dampener sent one burst of three flicks 0 → 2700 (story end, every scene skipped); stepper lands 648 → 1215 → 1782 (exact rests). Rest values derive from the copy windows; screenshots at rests skipped (probe values match the same plateaus check-experience already samples).

### O — Overview entry latency (charts show up sooner)
- [x] O1 Decoupled chart from slowest endpoint: skeleton and `data-cm-dashboard-ready` gate on `history + analysis`; TFT/explain hydrate in the background; separate in-flight guards; core polls 60 s, enrichment 300 s (weekly cadence). Probe with a 9 s-delayed TFT fixture: dashboard-ready 9.71 s → **1.76 s**, chart 9.92 s → **1.87 s**.
- [x] O2 Entry choreography: the ceremony timings stay (they are the visual language), but the overlay no longer blocks the app while it is visually gone — `.cm-logo-transition--reveal` drops `pointer-events`, so the workspace accepts clicks from the start of the reveal phase instead of ~2 s later.
- [x] O3 Chart refresh churn: animate on first mount only — entry draw plays through the forecast landing, later refreshes redraw in place (`PriceForecastChart.tsx`); unchanged payloads keep their object identity so polls skip the diff (`OverviewPage.tsx`, `isSamePayload`).
- [ ] O4 Repeat visits: stale-while-revalidate cache (React Query) for the four overview fetches. Deferred — O1 already removed the slowest-endpoint block; revisit only if remount latency still reads slow in production.

### S — Selection & pointer smoothness
- [x] S1 `TerraCursor.tsx` — pointer moves with any button held (`event.buttons !== 0`) no longer spawn dust; trail resets instead of chaining through a selection drag.
- [x] S2 `TerraCursor.css` — native I-beam restored over prose (`p`, `h1–h3`, `li`, `td`, `th`, form fields); copper cursor stays everywhere else. Byte-neutral edit (removed redundant var fallbacks).
- [x] S3 News rail (`NewsIntelligencePanel.tsx`) — drag arms on pointer-down but only captures the pointer once the gesture is horizontal and past a 6 px threshold; vertical gestures are released to native text selection; track width measured once per drag instead of per move. Contract test updated + new vertical-gesture test.
- [x] S4 `workspace.css` — dropped permanent `will-change` on the sticky header (kept the glass blur).

### M — Mount & polling discipline
- [x] M1 Defer below-fold panels (Heatmap/News/Companion) until the section approaches the viewport (`OverviewPage.tsx`, IntersectionObserver, 600 px margin, first-Tab fallback for keyboard-only walkthroughs). The loading placeholders hold the loaded heights (news 540, map 705/812 mobile) via `workspace.css` — dashboard-only rules stay out of the landing-critical global sheet, and the CLS the deferral briefly introduced (0.14–0.25) is gone (0.0098).
- [ ] M2 Heatmap: mount only the visible tier layer per leaf (`HeatmapTiles.tsx`) — ~500 fewer nodes.
- [x] M3 Pause core/enrichment intervals while `document.hidden`; memoize last-history-price.

### B — Backend latency & memory
- [x] B1 `market-prices` + `live-price` (`backend/app/main.py`): both ran blocking yfinance (`.info`, 15-min-delayed sweep) inside async endpoints, stalling every concurrent request; both now run in the threadpool — `market-prices` behind a 60 s single-flight TTL cache so dashboard mounts share one upstream sweep.
- [x] B2 TFT live path (`main.py`): threadpool + a single-flight lock (concurrent cache misses cannot run the model twice); cache-hit path unchanged. Follow-up (not done here): move the refresh to the worker and serve stale snapshots, so a cold live inference never occupies an API worker at all.
- [x] B3 Caches bounded: `_tft_cache` gets a 128-entry cap with oldest-first eviction on top of the per-symbol live-key purge; `_news_stats_cache` bounded like the list cache (128, oldest first).
- [x] B4 One FinBERT per process: `deep_learning/data/embeddings.py` exposes the shared classifier loader (`load_finbert_classifier`), the embedding encoder is its base-model view (same checkpoint weights, no head copy) and `app/ai_engine.py` builds the sentiment pipeline on the same instance — roughly one ~440 MB model copy reclaimed in the worker. Structural equivalence is covered by the offline suite (mocked models); no numeric embedding run was possible locally (no transformers in the offline env).

## Verification (before/after)
- [x] V1 Baselines (4× CPU where applicable): landing profile captured on the pre-change build; wheel + TFT-gate probes captured on `main` build. Note: `profile-dashboard.mjs` was stale (waited for a "Market Heatmap" heading that no longer exists) — fixed to `#market-map h2` before any run.
- [ ] V2b Landing rest screenshots at each stepper plateau (probe values match the copy-window plateaus; visual capture deferred).
- [x] V2 After-change runs on the same lab/build: TFT-gate probe 9.71 s → 1.01 s; dashboard CLS back to baseline (0.0052 → 0.0098) with heap and long-task counts unchanged; landing profile unchanged (programmatic-scroll workload identical by design). Numbers in the log below.
- [x] V3 `npm test`, `npm run lint`, `npm run build`, `npm run check:budgets`, `node scripts/check-experience.mjs`, `node scripts/check-dashboard.mjs` — all green on the final build (136/136 tests; budgets 13,968/14,000 CSS, 279,153/280,000 dashboard JS).
- [x] V4 Backend: `py -3 -m pytest tests/ -m "not online"` → 681 passed. (Run with `--basetemp` here: `%TEMP%\pytest-of-<user>` was permission-locked on this box — environmental, unrelated to the suite.)

## Verification log

### 2026-10-19 — baseline (pre-change dist)
- `profile-landing` (3 runs × desktop/4×CPU/mobile): scripted scroll at 4× CPU shows `mainThreadTaskMs` ≈ 3.12–3.36 s busy during a ~4 s scroll window; fps 108–119; longTasks 0; domNodes 425. Baseline for V2 comparison.
- `profile-dashboard` initially failed: script waited for a “Market Heatmap” heading that the current UI does not render; the page itself loaded (chart surface present). Script fixed to `#market-map h2` (same commit as V1).
- TFT-gate probe (fixture delays `/api/analysis/tft/**` by 9 s): `skeletonGoneMs = 9706`, `chartMs = 9922` — the old single `Promise.allSettled` blocked the whole dashboard on TFT.
- Wheel probe (old dampener): a momentum burst of three heavy flicks travelled 0 → 2700 (story end) — every scene skipped; the dampener only scaled the distance, it did not guarantee scene pacing.

### 2026-10-19 — after changes (same lab, identical build)
- TFT-gate probe: `skeletonGoneMs = 1758`, `chartMs = 1866` with the same 9 s-delayed TFT fixture (chart renders from history + analysis while the forecast hydrates).
- Wheel probe: `burst-of-three → 648`, `after-pause → 1215`, `after-pause-2 → 1782` — one narrative rest per deliberate gesture, no scene skipping, glide stays native.
- `npm run build`: TFT-modules unchanged; budgets: initial CSS **13,989 / 14,000 B** (byte-neutral S2 edit), initial JS 126,971 / 190,000 B, dashboard JS 278,809 / 280,000 B.
- `check-experience` → 7 landing + 3 adaptive + 6 workspace scenarios pass, page errors `[]`, landing API requests `[]`.
- `check-dashboard` → 10/10 scenario-state checks pass.
- `npx vitest run` → 16 files; news contract test intentionally updated (restores “capture after horizontal drag” + new vertical-selection case); heatmap perf test is load-sensitive on this box (passes in isolation).

### 2026-10-19 — final sweep (all workstreams on one build)
- Dashboard entry probe (TFT fixture delayed 9 s): skeleton 9,706 → **1,009 ms**; chart 9,922 → **1,197 ms**.
- Dashboard profile (3 runs, same script/lab): CLS 0.0052 → **0.0098**; long tasks 8/9/7 → 8/7/7 (max ≈130 ms both); heap ≈11.8 MB both; LCP noise-dominated (fixtures). The first after-run measured CLS 0.13–0.25 because the deferred-panel skeletons were far shorter than the panels they stand in for — aligning them (news 540, map 705/812) removed the shift, and the dashboard-only heights live in `workspace.css` so the landing-critical global sheet stays under budget.
- Deferred panels mount when their section is within 600 px of the viewport and never unmount; the first Tab press mounts everything for keyboard users; the QA scripts scroll the sections into view before asserting them.
- Landing profile: 4× CPU main-thread 3.07–3.23 s (baseline 3.07–3.36 s), fps 118–120, long tasks 0, DOM 425; wheel stepper unchanged (648/1,215/1,782).
- Budgets: initial JS 126,975/190,000; initial CSS **13,968/14,000**; dashboard JS 279,153/280,000.
- `npm run lint` clean; `npx vitest run` 136/136; `check-experience` green (7 landing + 3 adaptive + 6 workspace); `check-dashboard` 10/10.
- Backend: 681 offline tests pass. `market-prices` and `live-price` run in the threadpool (market-prices behind a 60 s single-flight TTL cache); the TFT live path is threadpooled behind a single-flight lock with a 128-entry cache cap; the news stats cache is capped like the list cache; FinBERT loads once per process (the embedding encoder is the sentiment classifier's base-model view).

### Lab note
- Any plain static server (incl. `vite preview`) must answer 404 for missing files; its HTML fallback turns `/_vercel/speed-insights/script.js` into a page error ("Unexpected token '<'") that QA scripts correctly flag. The tmp QA server (`tmp/qa-server.mjs`, untracked) mirrors production routing (`/` → landing shell, deep links → workspace shell, missing files → 404).
