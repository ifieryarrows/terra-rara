# Terra Rara brand assets

The canonical star paths live in `src/components/ui/brand-mark-geometry.json`. The `BrandMark` component and native cursor use those values through `brand-mark-geometry.ts`. The local asset renderer uses the same JSON to regenerate the favicon, Apple touch icon source, social preview source, and their PNG outputs:

```sh
npm run brand:render
```

## Usage policy

- `primary` is the expressive copper treatment. Use it for the cinematic landing signal and other spacious brand moments on dark backgrounds; copper shading and the highlight are most useful above 32px.
- `on-dark` is the warm-ivory, high-contrast treatment for persistent dark product chrome and compact dashboard identity. It keeps the workspace mark legible without competing with data. The navbar and Overview mark use this variant.
- `on-light` is reserved for actual cream or light surfaces. Do not use it on the current dark workspace; switch to it if a light theme or light document surface is introduced.
- `monochrome` is reserved for one-ink contexts such as constrained exports or print. It inherits its color from the parent.
- `small` uses only the enlarged star silhouette for tight labels and status indicators. Explicit `small` stays simplified at any size; every mark at 20px or below is simplified automatically.

Landing hero, entry reveal, and Copper Signal keep `primary`; eyebrow marks and route loading use `small`. Favicon, app icon, and social preview are generated as context-specific assets from the same canonical geometry, with separate backgrounds and contrast treatments.

The “Enter CopperMind” action uses the same mark for its route transition: the page covers to the dark ground, Canvas2D dust converges into the star and orbit, the resolved dashboard stays covered until its initial market requests settle, then the particles disperse to reveal it. Reduced-motion users see a short static-mark handoff.

The app UI uses Geist Sans for the wordmark and keeps the existing site type system.
