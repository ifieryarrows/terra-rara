# Follow the copper signal — delivery

Date: 2026-09-07 UTC. Base: `321f8244e04f81c1fa8dadbc8aa332f018d265c5`.

## Implemented

- Hero: retained the existing Cu contours, highlighted a contour and extended a copper trace; simplified the competing annotations. The heading now reads “Read the market. Follow the signal.”
- Intro: replaced the repeated capability strip with a shorter narrative bridge.
- Research: replaced the opaque preview-card stack with a persistent Cu reference marker, a shared SVG stage and separate semantic readings. Market categories turn into source/context annotations; the same marker travels to the forecast's last observation. The historical sample is visible and the illustrative future range is revealed without changing its values.
- Timing: non-overlapping caption windows; completed reading holds; reversible native-scroll progress; current-step chapter navigation. Focused disclosures remain mounted until focus leaves; inactive readings contain no hidden focus targets.
- Interaction: market context explanation and conditional news reasoning use native, keyboard/touch-accessible disclosures. No live finance API or LLM call is added to the landing.
- Responsive: all three scene diagrams and readings appear in normal flow for small/short viewports and reduced motion. The initial server-rendered experience remains complete. Save-Data also chooses the static version; simple sticky no longer depends on a RAM or fine-pointer assumption.
- Evidence: compact forecast passport replaces the large empty-metric preview. Models/Validation/System links use a full-width row; the Cu trace returns at the closing CTA.
- Dashboard: “Last checked” accurately labels browser fetch time; fallback quote deltas are omitted, displayed zeros are neutral, signed zero is suppressed. Model volatility classification is distinguished from quality warnings and investment safety.

## Verification

| Gate | Result |
| --- | --- |
| ESLint | Pass |
| Vitest | 57 tests across 10 files pass |
| TypeScript + Vite build + prerender | Pass |
| Existing bundle budgets | Pass |
| Transition regression | Forward/reverse samples never show two readings simultaneously; completed-scene holds verified |
| Interaction regression | Chapter navigation, inactive reading removal, disclosure focus preservation and static disclosure access pass |
| Quote regression | Missing/invalid observations, neutral and rounded zero, signed formatting and real change direction pass |
| Existing route, forecast, news and heatmap tests | Pass |

The original frontend baseline was built again in the same environment from the same lockfile/dependency installation. These are compressed artifact sizes, not measured network transfer or field speed:

| Static import closure | Baseline gzip bytes | New gzip bytes | Delta |
| --- | ---: | ---: | ---: |
| Initial JS | 117580 | 116544 | −1036 |
| Initial CSS | 12111 | 13145 | +1034 |
| Dashboard JS before lazy news/heatmap | 263660 | 262785 | −875 |
| Dashboard CSS including workspace styles | 14583 | 15617 | +1034 |

Existing limits: initial JS 190000, initial CSS 14000, dashboard JS 280000 bytes. The CSS limit in the existing script applies to initial CSS, not the larger dashboard closure. No limit was raised. No new package, font, model, texture or raster asset was added to the application.

## Acceptance still open

New visual screenshots and physical-device scroll/GPU acceptance were not performed. The earlier local browser preview was blocked by URL policy; this implementation did not attempt to bypass that restriction. The prior audit screenshots show the old live build and are not after-images. Passing unit tests does not establish final cinematic quality, text fit at every viewport, FPS or field LCP/INP/CLS.

Before merging, review the candidate in a permitted preview at 1536×900, 1280×900, 1280×720, 1024×650, 390×844 and 320 px, plus 200% text enlargement and reduced motion. Pause in both transfer windows, reverse scroll, jump directly to chapter anchors, open each disclosure with keyboard/touch, and enter/return from the dashboard. Check stage fit with disclosures expanded and complete static diagrams at narrow widths. Check native route/history scroll restoration after font loading.

WebGL material exploration remains optional, not delivered. This implementation provides conceptual continuity between the hero/intro/stage/footer, not a shared-element overlay across route navigation. A full source-model/horizon provenance adapter for AI commentary and missing walk-forward report generation remain separate product work; neither was fabricated or changed by this design patch. Backend predictions, heatmap algorithms and existing financial chart math were not modified.

The branch is intended as a draft PR for visual review, not a production release approval. See [implementation plan](../implementation/follow-the-copper-signal.md) and [live audit](./frontend-art-direction-audit-20260907.md).

Image evidence from the prior audit was excluded from GitHub publication after automatic approval review rejected uploading screenshots due to possible sensitive UI/financial data disclosure. The source code, tests and text documentation are the PR deliverables. Screenshots remain local review artifacts; they are not repackaged or uploaded through another channel.

Short desktop viewports (at least 42em tall) retain the sticky story with a smaller diagram. The disclosure area has a bounded native scrolling region so expanded reasoning cannot push the stage beyond the viewport. Smaller viewports use normal flow; keyboard focus still holds the active reading.
