# CopperMind frontend experience — phase-two status

> Follow-up: the [2026-09-06 audit](./frontend-phase-two-audit-20260906.md) checks these claims against the original conversation and actual browser behavior, fixes the short-desktop sticky defect, and defines the remaining verification limits. This document records the earlier status.

Date: 2026-09-05
Baseline: `5e032d0` (`Merge pull request #5 ... cinematic-entry-design-system`)

## Status

Phase one is complete and merged on `main`. Phase two is functionally implemented in the current working tree, but it is not formally closed yet: the changes are uncommitted and a deployed production verification has not been run after this work.

## What phase one delivered

- Established the CopperMind Terra Rara visual foundation: copper-on-ink tokens, shared motion primitives, brand mark and workspace shell.
- Added the landing page structure, deterministic preview fixtures and the introduction-to-workspace route flow.
- Added route boundaries, prerendering and bundle-budget checks.
- Preserved the existing forecast lifecycle, heatmap geometry and backend/model contracts.

The merged commit is `5e032d0`, with the first-phase implementation on its second parent `f6a3e03`.

## Phase-two goals and results

| Goal | Result | Status |
| --- | --- | --- |
| Give the landing page a clear product explanation and direct dashboard entry | Added the `Hero` and `CopperSignal` composition with copper contours, market/context/forecast labels and a visible dashboard CTA. | Complete |
| Turn the introduction into a connected native-scroll story | `ResearchStory` uses one Framer Motion scroll value for market, news and forecast layers. Chapter navigation and links lead to the real dashboard sections. | Complete |
| Keep the experience usable on mobile, reduced motion and conservative paths | Static story markup remains available; the desktop enhancement is opt-in and the local short-height scroll fallback was restored. | Complete |
| Improve workspace consistency | Added and applied `PageHeader`, `ViewState`, `DataTable`, `FilterChip` and `RefreshButton` across Overview, Models, Validation, System and News Intelligence. | Complete |
| Preserve product semantics and data behavior | Weekly reliability metrics, forecast alignment, financial calculations, heatmap behavior and API queries remain in place; the landing page uses deterministic sample data. | Complete by code review |
| Validate route, state and accessibility behavior | Loading, empty, error, table, filter, drawer and focus paths are covered by the existing suite and browser checks. | Complete locally |
| Close the delivery with production evidence | Build, lint, tests, budgets and diff checks pass. No new deployed trace or production p75 measurement exists yet. | Open |

## Concrete evidence

- `npm.cmd run lint` passed.
- `npm.cmd run test` passed: 5 files, 35 tests. The heatmap benchmark was 2.04 ms p95 for the real fixture and 5.45 ms for the 1,000-instrument fixture, under the existing 8/12 ms bounds.
- `npm.cmd run build` passed and prerendered the landing plus the workspace shell.
- `npm.cmd run check:budgets` passed: initial JS 116,766 gzip bytes, shared CSS 11,723 bytes, dashboard-before-news/heatmap JS 261,898 bytes and CSS 13,319 bytes.
- `git diff --check` passed.
- In the local browser at 1280×720, the enhanced story kept its stage sticky and switched visible layers across scroll positions 0, 2,200 and 3,000. Local and preview routes both loaded without API data being required by the landing page.

## Remaining closure work

1. Review the current working-tree diff and commit phase two as a focused change.
2. Run the deployed landing and workspace smoke check on the same viewport used for local comparison.
3. Capture production performance evidence separately; the local browser check does not establish production p75 LCP, INP, CLS, real-device FPS or GPU memory.
