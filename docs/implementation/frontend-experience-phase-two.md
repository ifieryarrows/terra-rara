# CopperMind experience — second implementation

Date: 2026-09-05. Baseline: `5e032d0` (merged introduction/design foundation).

## Scope and visual direction

Continue the existing copper-on-ink system with a compact editorial hero and a copper signal sculpture: bounded SVG contours connect the metal identity to market structure. Compose the real product concepts around it, with clearly illustrative market data. The first viewport must contain the product explanation and a direct dashboard link. Mobile uses its own stacked composition.

Keep the existing React/Motion/SVG stack. No new animation, WebGL, chart or font dependency. Native scroll controls a single story progress value. Reveal heatmap cells, connect the news workflow and draw the forecast/range in their respective chapters. All explanatory content remains available with reduced motion, small screens, Save-Data or JavaScript unavailable.

## Implementation sequence

1. Record the merged build and preserve a local baseline for comparison. Inspect desktop and mobile in the browser.
2. Extract `Hero` and `CopperSignal`; retain route entry/exit, deterministic preview fixtures and content provenance. Use transform/opacity for bounded layer composition, and only small SVG path reveals.
3. Refine `ResearchStory` and previews around one progress source. Add direct chapter navigation, aligned chapter transitions and useful links into the workspace. Keep static and enhanced layouts geometrically consistent at startup.
4. Add shared `PageHeader`, `ViewState`, `DataTable`, `FilterChip` and `RefreshButton` primitives. Migrate Models/Validation/System page framing and states, Overview chart states/table/tooltip, and News form controls. Keep queries, polling, financial values, thresholds, forecast alignment and heatmap geometry intact.
5. Verify route/focus behavior, keyboard filters and drawer, error/empty states, mobile geometry and desktop story composition. Run tests, lint, build and existing bundle gates. Record measurements and remaining production checks in a separate report.

## Boundaries and budgets

- Preserve `/dashboard`, `/models`, `/validation`, `/system`, `/overview` and legacy symbol links. Landing never fetches market services or generates commentary.
- Leave backend, TFT training/calibration/quality gates, forecast transforms and live heatmap implementation unchanged.
- Landing JS ≤190,000 gzip bytes, shared CSS ≤14,000 bytes, initial dashboard JS ≤280,000 bytes. No increased budgets.
- Compare production builds on the same local browser/viewport. Local timing is lab evidence, not production p75 Core Web Vitals or real-device GPU performance.

## Completion evidence

See the second-phase report for final checks and remaining production measurements.
