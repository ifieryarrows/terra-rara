# CopperMind Terra Rara — frontend experience plan

Date: 2026-09-05. Starting commit: `a5a0233cb12074effe529a97e48a9485b8ca6349`.

## Product decision

One product, two modes: a short, cinematic introduction followed by a quiet financial workspace. Visual thesis: **copper signal on an ink-black research surface**. Large editorial typography and a layered market chart establish the brand; precise grids, tabular numbers and restrained status colors carry into the dashboard.

The research reference is `docs/design/immersive-web-case-study-tr.docx`. It is evidence and technique guidance, not a requirement to reproduce the reference sites' dependencies. Keep React 18 / Vite 7 / React Router / TanStack Query / Framer Motion / Recharts / D3. Do not migrate to Next.js, add GSAP/Lenis, or add WebGL merely for stylistic parity.

## Audit and invariants

- All four primary pages are eagerly imported by `App.tsx`. Baseline entry: 854.00 kB raw / 254.10 kB gzip; CSS 48.90 / 8.58 kB. News and heatmap are already lazy chunks (8.12 / 12.56 kB gzip).
- 27 tests and lint pass before editing. D3 layout p95: 0.47 ms for the reference fixture; 3.69 ms for 1,000 instruments in this local run. These are CPU test timings, not browser FPS.
- Root currently opens Overview; `/models`, `/validation`, `/system` are existing deep links. Vercel rewrites API before SPA fallback.
- Main navigation disappears below 640 px. Overview's quote card has a 360 px minimum width and a non-wrapping header; the previous heatmap report already records mobile overflow from this component.
- Styling is dark-only, spread between Tailwind, literal colors and local components. Existing IBM Plex Sans + system monospace are intentional. Many metadata labels are 9–10 px.
- Overview uses local polling state alongside Query hooks. Preserve API contracts and polling semantics in this slice; do not combine a data-state rewrite with visual migration.
- Preserve heatmap hierarchy memoization, stable IDs, resquarify, projected-area LOD, 10% weight compression, pointer rAF, category cache, full-width layout, zoom/pan, keyboard and Escape behavior. Leave its geometry and color scale intact.
- Preserve forecast horizon/alignment logic and quality gates in `GEMINI.md`. Do not change predictions, risk logic or model metrics to make marketing claims.

## Routing

| Route | Responsibility |
| --- | --- |
| `/` | Intro and product story; primary CTA visible immediately and at the end |
| `/dashboard` | Existing Overview in shared workspace shell |
| `/models`, `/validation`, `/system` | Keep current URLs and functionality |
| `/overview` | Compatibility redirect to `/dashboard` |
| Unknown path | Useful not-found view with dashboard/home links |

The root changes intentionally from Overview to landing. Symbol-query-bearing old root links retain a direct workspace route. Returning users get a one-click dashboard link at the top; no forced replay, timed lock, scroll completion requirement, automatic redirect or storage-dependent routing. Preserve Back/Forward and hash anchors. Route-level lazy imports isolate Recharts and dashboard business code from landing.

## Storyboard and first implementation slice

1. **Hero / Read the market. See the structure.** Copper futures, news and quantitative forecasts in one research workspace. A large bounded SVG preview shows observed-path context and a clearly distinguished forecast range. This is labeled illustrative, never live.
2. **Signal strip.** Market context → news intelligence → forecast range → validation. These are product capabilities verified in code, not performance claims.
3. **Sticky research sequence.** Three short narrative chapters beside a shared preview: market heatmap, news/sentiment, forecast uncertainty. Native vertical scroll drives one MotionValue; transforms, opacity and path reveal derive from it. Mobile/reduced-motion use ordinary stacked sections with all meaning in HTML.
4. **Evidence before conviction.** Link to the real Models and Validation routes; explain weekly vs T+1 horizons, model availability and freshness. Do not claim model accuracy or returns.
5. **Enter CopperMind.** Direct dashboard CTA plus secondary documentation/navigation links already present in the product.

Preview data is deterministic and explicitly marked as illustrative. No randomized prices, fake live indicator, invented backtest results or frontend calls that trigger training/LLM refresh. Real data remains in the application. A future cached preview adapter may replace fixtures only after freshness/availability contracts and backend cost are verified.

## Component architecture

| Layer | Components / responsibility |
| --- | --- |
| App | Router, query client, route loading/error, motion preference, route focus/title |
| Shared UI | Brand, action link, financial panel, metric card, tokens |
| Landing | LandingPage, Hero, ResearchStory, MarketPreview, NewsPreview, ForecastPreview, Evidence, final CTA |
| Motion | MotionPolicy and one story `scrollYProgress`; no per-frame React state |
| Workspace | AppShell and existing Overview/Models/Validation/System |
| Data | Existing API/types/query hooks; landing fixture isolated under landing feature |
| WebGL | No runtime in the first slice; future dynamically imported optional scene with an explicit UX hypothesis |

Inputs to a story visual are `{ progress, reducedMotion }`, plus local fixture data. Output is DOM/SVG only. UI business state never depends on animation completion. Motion cleanup is owned by hooks; route unmount removes subscriptions. Chart/treemap engines stay in their current feature boundaries.

## Implementation phases

- [x] 1. Audit, baseline and source reference in repository.
- [x] 2. Shared semantic tokens, brand, motion policy and accessible workspace navigation.
- [x] 3. Landing with native scroll, controlled previews and lazy route boundaries.
- [x] 4. Dashboard visual migration: shared panels/metric components, responsive header, financial text and cheap transitions; keep heatmap behavior.
- [ ] 5. Test/build/lint, bundle comparison, route/accessibility regression checks and available browser QA.
- [ ] 6. After deployment: same-device production performance sampling, field Web Vitals and product comprehension validation. This cannot be claimed from local unit tests.

## Acceptance and performance budgets

Entry JS target ≤190 kB gzip for landing; dashboard data visualization stays deferred. Shared CSS target ≤14 kB gzip. No initial 3D/video/model requests. No marketing API polling. HTML heading/description and links survive JS failure. No horizontal page overflow at 390/768/1536 px; keyboard navigation always visible; all routes directly addressable.

Production targets: p75 LCP ≤2.5 s, INP ≤200 ms, CLS ≤0.1; sustained animation p95 frame interval ≤16.7 ms on an agreed 60 Hz desktop. These are targets, not measurements. Measure cold/warm runs, identical throttling, route, fixture and viewport. Unit layout p95 is a separate metric. No new >50 ms interaction tasks attributable to motion. Capture build gzip sizes and compare before/after; retain current heatmap tests.

Browser QA requested: hero/story/CTA; direct dashboard and legacy routes; Back/Forward; keyboard skip/focus; responsive layout; reduced motion; API error/empty states. The cloud preview initially returned `ERR_BLOCKED_BY_CLIENT`; record unresolved infrastructure limits rather than fabricating Lighthouse, INP, FPS or screenshots.

## Prompt refinements

The user's brief is the long-term specification. Add explicit first-slice deliverables, fixed baseline commit, dependency-selection rationale, metric measurement conditions, preview provenance and route compatibility. Keep the DOCX as the research reference, while version-controlled Markdown records Coppermind-specific decisions and CSS/TS tokens remain the executable source of truth. "Central motion" means one shared clock for coordinated visuals, not a global controller coupling every table hover to marketing.
