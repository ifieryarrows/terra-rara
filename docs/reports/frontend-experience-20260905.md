# CopperMind introduction and frontend foundation — 2026-09-05

Baseline: `a5a0233cb12074effe529a97e48a9485b8ca6349`. Scope: first implementation slice from the user's long-term frontend redesign brief.

## Delivered

- Root introduction with typographic copper/ink identity, bounded chart reveal/perspective, native scroll storytelling, sticky preview composition, static mobile/reduced-motion mode and immediate/final dashboard entry.
- `/dashboard` for Overview; existing Models/Validation/System URLs retained; root symbol queries forwarded, other campaign queries retained on landing; a useful missing-route state.
- Shared semantic CSS/TS tokens, brand, financial panels and metric cards; full mobile workspace navigation; responsive quote/header and card columns.
- Clear illustrative previews with no market API polling, no fabricated live status and no performance claims. The marketing route contains no WebGL/model/video runtime.
- Removed changing-number blur and width-based bar animation. News typography, contrast and modal keyboard behavior improved. Forecast chart has a lazily expanded data table.
- Lazy workspace pages and chart dependencies; prerendered introduction HTML, independent workspace shell, cache headers, route error/focus handling and Speed Insights across both modes.
- Research DOCX copied into `docs/design`; implementation plan and Coppermind-specific engineering guide added. CI now runs delivery budget checks.

## Verification

| Check | Before | After / interpretation |
| --- | --- | --- |
| Frontend tests | 27 pass | 35 pass; existing heatmap and forecast tests retained |
| Lint | Pass | Pass |
| Typecheck/build | Pass | Pass, including landing prerender |
| Initial entry JS, gzip | 254.10 kB | 115.50 kB, approximately 54.5% lower |
| Shared CSS, gzip | 8.58 kB | 11.94 kB |
| Dashboard initial static-import closure | 254.10 kB entry before lazy panels | 259.80 kB before lazy news/heatmap; approximately 2.2% higher, within the 280 kB budget |
| News lazy chunk, gzip | 8.12 kB | 8.37 kB |
| Heatmap lazy chunk, gzip | 12.56 kB | 12.58 kB |
| Reference D3 layout p95 | 0.47 ms | 0.46 ms in final local test run |
| 1,000-instrument D3 layout p95 | 3.69 ms | 2.45 ms in final local test run; normal test variability, no claimed algorithmic speedup |
| Delivery budgets | None | Pass: entry 190 kB, CSS 14 kB, dashboard 280 kB gzip limits |
| First HTML content | Empty React root | Full introduction and CTA prerendered; workspace shell separate |

The initial-entry reduction compares entry payloads, not all bytes required to use the dashboard. A user entering the workspace still downloads the chart and data modules. Budget tooling follows the production manifest's static import graph and counts each file once. Compressed build sizes use gzip bytes, not HTTP transfer timing.

## Known verification limits and release gate

The supervised local preview started, but the cloud browser denied its URL (`ERR_BLOCKED_BY_CLIENT`, then an explicit URL-policy rejection). No alternate browser surface or network workaround was used. Therefore no visual screenshots, actual mobile layout measurements, browser frame traces, LCP/INP/CLS, GPU/heap measurement or production API E2E result is claimed. Passing DOM/unit tests is not equivalent to these checks.

Review as a draft before merging. In a normal browser, test desktop 1536×864, tablet 768 px, mobile 390 px and 200% text zoom. Inspect sticky transitions, viewport overflow, keyboard navigation, reduced motion, hash anchors, Back/Forward restoration and API loading/error/empty states. Run cold/warm production-equivalent traces with fixed network/CPU throttling and compare dashboard interactions to the baseline. The first slice is a foundation; full component migration and any justified WebGL scene remain subsequent work.

## Preservation

No backend, model, API, forecast calculation, quality-gate threshold, treemap layout, data-cache or heatmap interaction algorithm changes. Its hierarchy memoization, stable identity, LOD, weight compression and pointer rAF paths remain in place. Financial fixture previews are isolated from real application data.
