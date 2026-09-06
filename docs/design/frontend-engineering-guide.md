# CopperMind frontend engineering and design guide

The research source is [the Turkish technical case study](./immersive-web-case-study-tr.docx). Coppermind-specific decisions live here and in the [implementation plan](../implementation/frontend-experience-plan.md). The executable token definitions in `frontend/src/design/` are authoritative; keep this guide synchronized when they change. The DOCX is a reference, not a dependency checklist.

## Product and content

The brand has two densities: editorial introduction and financial workspace. Both use the same copper accent, ink surface, IBM Plex Sans typography, system monospace numbers, borders and focus treatments. Landing promises only code-backed capabilities. A model being present in the repository does not prove that production has a trained checkpoint or a current snapshot.

Landing sequence: hero → connected workflow → market/news/forecast story → model evidence → dashboard entry. All CTAs are real links. Returning visitors can enter from the first viewport. There is no unlock timer, compulsory scroll, intro replay state or automatic returning-user redirect.

Use explicit dates, units and horizon labels in financial views. Positive/negative colors mean data direction or a clearly labeled status; copper is brand/selection, blue is forecast, gray is neutral. Never replace missing values with zero, add optimistic model claims, or blend weekly accuracy and daily Sharpe under an unlabeled aggregate. Existing financial transformations and quality gates are outside visual redesign scope.

Preview fixtures in `features/landing/preview-data.ts` are deterministic and illustrative. Each numeric preview names this fact visibly and in chart alternatives. They are not claimed to be live, sampled from production or actual model forecasts. The news preview describes the real workflow without inventing articles. The landing does not call financial services, poll quotes, run training or trigger commentary generation.

## Design tokens and components

| Token group | Implementation | Rule |
| --- | --- | --- |
| Surfaces | `--cm-bg`, `--cm-surface`, `--cm-surface-raised` | Opaque reading surfaces; no continuous effects behind tables |
| Text | `--cm-text`, `--cm-muted` | Readable heading/body/metadata hierarchy |
| Brand | `--cm-copper`, `--cm-copper-strong` | CTA, focus-related accents, brand and selection |
| Financial | `--cm-positive`, `--cm-negative`, `--cm-neutral`, `--cm-forecast` | Labels/signs accompany color; current treemap scale retained |
| Space | `--cm-space-panel`, content width, rem spacing | Responsive panels, consistent gutters, bounded text widths |
| Motion | `design/motion.ts`, CSS duration/ease tokens | 160–240 ms UI; 650 ms optional reveal; no random easing |
| Charts | `design/chart-tokens.ts` | Charts consume semantic colors rather than duplicating hex values |

Body text is normally 16 px; controls and regularly read labels 14 px; secondary metadata 12 px. Hero uses fluid `clamp` with a readable mobile minimum. Financial values use `tabular-nums` and system monospace. Existing treemap area-driven LOD typography is an explicit exception: tiny tiles preserve their labels through accessible names and category details, not through forced 12 px content overflowing the cell.

`Brand`, `FinancialPanel`, `MetricCard` and `RouteBoundary` establish reusable primitives. Existing glass-panel classes map to the same surface tokens. New screens should use these or extend their contract rather than copy Overview's prior local card implementations. Models and Validation now share one metric card. Business queries stay in existing hooks/API modules.

The first migration covers shared surfaces, shell/navigation, Overview panels/header/chart alternatives, Models/Validation metrics and news readability/drawer behavior. It does not claim every legacy table, input and utility class has been replaced by primitives. Consolidate additional components when a screen requires work, keeping functionality tests intact.

Workspace pages share `PageHeader`, `SectionHeader`, `ViewState`, `RefreshButton`, `DataTable` and `FilterChip`. Weekly model metrics are primary; native `details` keeps T+1 diagnostics available on demand. Validation accepts both API metric names and the rolling report's `mean_*`, `da` and `sharpe` aliases without recomputing metrics. The Theta table uses reported comparison values; unrecognized report shapes retain their raw details. Missing service values remain unknown rather than implying success.

`features/forecast/PriceForecastChart` owns display window, series visibility and table disclosure state, independent of Overview updates. Window labels count observed closes, not calendar days. Reuse the exact forecast-reference-date guard; period selection never triggers an API request or recalculates the model path. Date-only chart labels use UTC to avoid shifting a market date in another timezone. A crossed/incomplete Q10–Q90 pair is not repaired into a filled interval. Keep the published values available in the table.

News filters and headlines share one bounded, keyboard-scrollable viewport so expanded controls cannot squeeze the feed out of short screens. Fullscreen heatmaps use a portal, contain keyboard focus, restore the trigger and body scroll, and reserve flexible map height beneath the controls. Shared controls do not change treemap geometry, category hover positioning, peers or query cadence.

## Motion architecture

Native scroll is the input source. Framer Motion is already installed; this slice adds no GSAP, Lenis, Three.js, Rive, shader, model or decoder dependency. CSS handles static layout and sticky positioning. The research sequence uses one `scrollYProgress` MotionValue; preview crossfades, translation and the progress line derive from that value. High-frequency progress never enters React state. Hero has its own local chart-reveal scope because it is not synchronized with the later story.

Viewport capability is low-frequency state. Fine pointer + width ≥1024 + no reduced-motion + no Save-Data + available memory hint >4 GB enables enhancement. Missing memory hints do not block desktop animation. Coarse-pointer/mobile, reduced-motion, Save-Data and low-memory paths retain fully visible, stacked content. This is conservative progressive enhancement, not a measured GPU-tier or runtime-FPS detector.

Hooks own subscription disposal. Leaving landing unmounts its scroll consumers. No manual infinite rAF, wheel interception, inertia, snap or scroll-jacking exists. `MotionConfig reducedMotion="user"`, reduced-motion CSS, static previews and Recharts' explicit animation flag cooperate. Numeric updates are immediate; removed blur/re-key animations do not repeatedly obscure current prices. Progress/importance bars use transform scale rather than layout width animation.

SVG path reveal has paint cost even when its wrapper transform is composited. Keep it bounded to one small chart; do not apply path morph/blur across hundreds of visible cells. Crossfade layers are noninteractive and decorative; all substantive explanations remain in HTML. The reduced/static flow includes the complete preview captions.

The enhanced story stage must keep `align-self:start` and a viewport-bounded height. Grid stretch combined with `height:auto` breaks sticky behavior on short desktop screens. Compact preview spacing below 650 px height instead; the 1280×600 and 1024×650 browser cases verify that the stage and its inner content fit.

## Routing, loading and SEO

`/` introduces the platform. `/dashboard` opens Overview. `/models`, `/validation`, `/system` retain their URLs. `/overview` is a compatibility alias. Root links with a symbol query enter `/dashboard` while retaining the query/hash; ordinary campaign parameters keep the landing. Unknown routes show a useful fallback. No localStorage flag controls navigation.

React Router remains in place. Dashboard pages and feature panels are lazy imports. The light introduction is part of the entry; no heavy marketing runtime or assets are shared into the dashboard. QueryClient remains outside route boundaries. Route navigation focuses the main landmark and keeps a bounded in-memory history of scroll positions. Browser QA of scroll restoration remains required because DOM unit tests cannot validate viewport geometry.

The build prerenders the complete static landing into `dist/index.html` via React's server renderer. The client mounts React after its code arrives; this is build-time prerendering, not runtime SSR or RSC. `workspace.html` preserves an independent SPA shell for direct application links. Vercel/Nginx route configuration distinguishes root from other pages. Critical copy, navigation and preview explanations exist without JavaScript; the financial app still requires JavaScript.

The prerender Vite instance uses `node_modules/.vite-prerender` for its optimizer cache. Its configuration differs from the development server, so sharing the default `.vite` cache can invalidate a running dev server's dependency URLs during a build.

No new font/image/video/model fetches are introduced. Existing IBM Plex Sans remains; mono uses system fonts. Hashed `/assets/*` are immutable. HTML remains revalidatable. Never apply immutable caching to API responses or the HTML shell. Vercel Speed Insights mounts for both landing and dashboard; actual field data depends on the deployment's service configuration and traffic.

## Accessibility and resilience

- Preserve semantic main/nav/header/footer, heading hierarchy and visible focus; skip links work on both modes.
- All four workspace routes remain accessible below 640 px; touch controls target at least 44 px where practical.
- News detail has dialog semantics, focus containment, Escape, close-button target and focus restoration; body scroll is restored on cleanup.
- The actual forecast chart has keyboard accessibility and an expandable HTML table. Table rows mount only when expanded.
- Loading, missing model, degraded forecast, stale snapshot and API errors retain their existing meaning. Route chunk failures offer a reload and dashboard path.
- No WebGL failure can block this implementation because it has no WebGL dependency.

## Future optional WebGL gate

Before adding 3D, write the explanatory benefit and a cheaper SVG/CSS comparator. Profile both on real mobile hardware. Load a scene only after critical content, through its own dynamic import. Keep one canvas owner, capped DPR (start 1–1.5), bounded texture/geometry memory, visibility suspension and context-loss fallback. Model compression is selected from actual asset needs (Draco/Meshopt); KTX2 may reduce GPU texture memory. None is useful as a checkbox when no model is loaded.

Provide low/mid/high tiers only when a scene exists. Low uses the static product figure; mid removes postprocessing/shadows and reduces texture/particle resolution; high is enabled only with frame headroom. Use downgrade/upgrade hysteresis. Dispose or reference-count renderer, render targets, textures, materials, geometries, observers, timelines and events on route exit. Financial tables, filters and charts must never depend on scene readiness.

## Validation and budgets

Run `npm run test`, `npm run lint`, `npm run build`, `npm run check:budgets`. Budget checks recursively count static manifest imports; dynamic imports are excluded from landing and included for the chosen dashboard route. Limits: landing JS 190,000 gzip bytes; shared CSS 14,000 bytes; dashboard before lazy news/heatmap 280,000 JS bytes. These are project budgets, not web standards.

Production targets remain p75 LCP ≤2.5 s, INP ≤200 ms, CLS ≤0.1; agreed-device scroll p95 frame ≤16.7 ms at 60 Hz. Use identical production builds, viewport, network/CPU throttling, cache mode, API fixtures and interaction scripts before/after. Capture DevTools network/long tasks/heap and a 10–15 s interaction trace. Lighthouse lab results are not field INP or measured GPU time. Keep tests of reference and 1,000-instrument D3 layouts independent of browser measurements.

The initial session could not run a cloud-browser preview. On 2026-09-06, `scripts/check-experience.mjs` passed six landing and six fixture-backed workspace cases in headless Edge against both a fresh Vite dev server and the production build preview. These establish the tested geometry, chapter transitions and keyboard disclosure behavior. Real-device FPS, field LCP/INP/CLS, memory behavior and deployed provider availability remain unmeasured; see the phase-two audit and phase-three first-slice delivery report.

Phase-three closure adds `scripts/check-dashboard.mjs`: five viewport interaction flows and five data/error states passed on dev and production preview, including chart/table keyboard access, news drawer focus restoration and fullscreen map lifecycle. `scripts/profile-dashboard.mjs` records a short, repeatable three-run local lab comparison. Its event durations are not field INP and its end heap is not a leak test. See [the completion report](../reports/frontend-phase-three-completion-20260906.md) for raw evidence, mixed performance results and remaining release validation.
