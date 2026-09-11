# Copper continuity implementation — 8 September 2026

Base: main after merged PR #6 (`bd48e17`) and PR #7 (`ab99edc` ancestry). The implementation was rebased onto the resulting main tip. This change implements the CSS/SVG direction from `docs/implementation/copper-continuity-research-plan-tr.md`; it does not merge the earlier three-card implementation.

## Delivered

- One desktop sticky world from Cu through signal, market, context, range, evidence and workspace. A single 65-point SVG path changes geometry across the seven states; the path owner remains mounted. Shared copper diffusion is driven by the same MotionValue.
- Local League Gothic display font (20,268-byte WOFF, upstream OFL included), large background words and directional clipped heading reveals. Geist remains the reading/data font. No new runtime dependencies.
- Semantic, normal-flow chapter content. Small/short screens, reduced motion and Save-Data retain the sequence as static scenes. Static graphics render only their relevant layer. No inner scrolling reading panel or focus-based animation freeze.
- Genuine hash anchors, native smooth scroll, reduced-motion instant navigation, destination focus and user-input cancellation. Existing dashboard/model/validation/system URLs and direct-entry behavior remain intact.
- Lazy Evidence module and one cached report request when approaching the section; no landing polling. Typed normalization separates absent reports from actual finite summary metrics. Zero remains a valid result. Request errors offer retry. Summary results retain report units/date availability and link to the full report; no invented pointwise replay, confidence claims or baseline comparison.
- Landing-only font preload in prerendered HTML; workspace shell stays free of landing content. Updated critical-content guards without changing budget thresholds. Old landing-only CSS consolidated.

## Design decisions and limits

Current backend evidence endpoint exposes aggregate/window reports; it does not establish a verified pointwise prediction/actual contract. The shipped Evidence reader therefore supports summaries and unavailable/error/loading states. A historical replay requires that separate data contract, not a fabricated animation. Market and news diagrams and forecast path are explicitly illustrative; they are not live market updates.

The first implementation uses a shared right-side graphic plane with normal-flow foreground readings and background typography. It is a reviewable implementation of continuity, not a claim that every proposed camera/spatial experiment has passed visual review. Full camera/mesh/shader exploration remains optional, gated by demonstrated benefit. No Three.js, Lenis, GSAP, video, model, texture decoder or continuous idle loop was added.

The font is WOFF rather than WOFF2: the available font encoder lacked Brotli support. Its 20KB transfer size meets the proposed font allowance. Responsive and reduced-motion behavior exist in code; physical mobile testing remains necessary.

## Verification

Existing suite plus six new tests passed; six new tests cover handoffs, finite geometry, reverse-scroll tolerance, overscroll and evidence normalization. One additional route test covers the reduced-motion hash/focus destination. TypeScript, lint, Vite build, prerender and unchanged bundle budgets were checked. This is not a Lighthouse, field CWV or GPU profile.

Final measured gzip sizes: initial JS 114654 bytes, initial CSS 10557 bytes, dashboard closure JS 260929 bytes. The final `check:budgets` gate passed; limits remain 190000 / 14000 / 280000 bytes. No claim of measured 60FPS is made.

## Required preview review before merge

Browser visual QA was not completed: the available local preview route had previously been policy-blocked, and no alternative route was used to evade it. Keep the PR draft for review on its authorized preview.

1. Desktop 1366×936 and 1440×900: each anchor, forward/reverse scroll, all boundary holds, heading fit and Evidence with summary/loading/error fixtures.
2. Mobile Safari/Android and 200% text/zoom: readable typography, normal flow, no horizontal overflow, usable CTA/focus and orientation changes.
3. Cold-load font/hydration and deep-link positions; browser Back/Forward after multiple anchor clicks; cancel smooth movement with wheel/touch/keyboard.
4. Performance trace during scroll, route transitions and five repeated enter/leave cycles. Confirm no growing listeners/heap and no dashboard interaction regression.
5. Real-data freshness/report interpretation review. Missing evidence must remain visibly missing.

PR #6 is already merged; this branch is the follow-up continuity implementation.
