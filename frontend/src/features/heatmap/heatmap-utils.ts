const TOOLTIP_WIDTH = 292;
const TOOLTIP_HEIGHT = 154;
const TOOLTIP_GAP = 14;
const PANEL_WIDTH = 380;
const PANEL_GAP = 18;
const PANEL_POINTER_Y_OFFSET = 48;

interface ColorStop {
  pct: number;
  r: number;
  g: number;
  b: number;
}

// Finviz exact market heatmap color palette:
// Radiant green on the upside (+3% and above -> #30cc5a)
// Distinct dark charcoal neutral at 0.00% (#414554)
// Radiant red on the downside (-3% and below -> #f63538)
const FINVIZ_COLOR_STOPS: readonly ColorStop[] = [
  { pct: -3.0, r: 246, g: 53,  b: 56 },  // #f63538 (Finviz max red)
  { pct: -2.0, r: 199, g: 62,  b: 67 },  // #c73e43
  { pct: -1.0, r: 139, g: 68,  b: 78 },  // #8b444e
  { pct: -0.5, r: 100, g: 69,  b: 83 },  // #644553
  { pct:  0.0, r:  65, g: 69,  b: 84 },  // #414554 (Finviz neutral slate)
  { pct:  0.5, r:  55, g: 100, b: 78 },  // #37644e
  { pct:  1.0, r:  53, g: 118, b: 78 },  // #35764e
  { pct:  2.0, r:  46, g: 189, b: 89 },  // #2ebd59
  { pct:  3.0, r:  48, g: 204, b: 90 },  // #30cc5a (Finviz max green)
];

export function getColorForChange(change?: number): string {
  if (change == null || !Number.isFinite(change)) return '#414554';
  const first = FINVIZ_COLOR_STOPS[0];
  const last = FINVIZ_COLOR_STOPS[FINVIZ_COLOR_STOPS.length - 1];
  if (change <= first.pct) return `rgb(${first.r},${first.g},${first.b})`;
  if (change >= last.pct) return `rgb(${last.r},${last.g},${last.b})`;

  for (let i = 0; i < FINVIZ_COLOR_STOPS.length - 1; i += 1) {
    const s0 = FINVIZ_COLOR_STOPS[i];
    const s1 = FINVIZ_COLOR_STOPS[i + 1];
    if (change >= s0.pct && change <= s1.pct) {
      const t = (change - s0.pct) / (s1.pct - s0.pct);
      const r = Math.round(s0.r + (s1.r - s0.r) * t);
      const g = Math.round(s0.g + (s1.g - s0.g) * t);
      const b = Math.round(s0.b + (s1.b - s0.b) * t);
      return `rgb(${r},${g},${b})`;
    }
  }
  return '#414554';
}

export function clampTooltipPosition(
  x: number,
  y: number,
  viewportWidth: number,
  viewportHeight: number,
  width = TOOLTIP_WIDTH,
  height = TOOLTIP_HEIGHT,
) {
  const left = x + TOOLTIP_GAP + width <= viewportWidth - 8 ? x + TOOLTIP_GAP : x - width - TOOLTIP_GAP;
  const top = y + TOOLTIP_GAP + height <= viewportHeight - 8 ? y + TOOLTIP_GAP : y - height - TOOLTIP_GAP;
  return {
    left: Math.max(8, Math.min(left, viewportWidth - width - 8)),
    top: Math.max(8, Math.min(top, viewportHeight - height - 8)),
  };
}

interface RectBounds {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

export function computePanelPosition(
  anchor: RectBounds,
  bounds: RectBounds,
  viewportWidth: number,
  viewportHeight: number,
) {
  if (viewportWidth <= 640) {
    return { mode: 'sheet' as const, left: 0, top: 0, width: viewportWidth, maxHeight: Math.min(560, viewportHeight * 0.78) };
  }
  const margin = 10;
  const width = Math.min(PANEL_WIDTH, Math.max(300, bounds.width - margin * 2));
  const roomRight = bounds.right - anchor.right - margin;
  const roomLeft = anchor.left - bounds.left - margin;
  let left = roomRight >= width || roomRight >= roomLeft ? anchor.right + margin : anchor.left - width - margin;
  left = Math.max(bounds.left + margin, Math.min(left, bounds.right - width - margin));
  const maxHeight = Math.min(560, bounds.height - margin * 2, viewportHeight - margin * 2);
  let top = Math.max(bounds.top + margin, anchor.top);
  if (top + maxHeight > Math.min(bounds.bottom, viewportHeight) - margin) {
    top = Math.max(bounds.top + margin, Math.min(bounds.bottom, viewportHeight) - maxHeight - margin);
  }
  return { mode: 'float' as const, left, top, width, maxHeight };
}

export function computePointerPanelPosition(
  x: number,
  y: number,
  bounds: RectBounds,
  viewportWidth: number,
  viewportHeight: number,
  panelWidth = PANEL_WIDTH,
  panelHeight = 480,
  avoidRect?: RectBounds,
) {
  if (viewportWidth <= 640) {
    return { mode: 'sheet' as const, left: 0, top: 0, width: viewportWidth, maxHeight: Math.min(560, viewportHeight * 0.78) };
  }
  const margin = 10;
  const rightEdge = Math.min(bounds.right, viewportWidth);
  const bottomEdge = Math.min(bounds.bottom, viewportHeight);
  const width = Math.min(panelWidth, Math.max(300, bounds.width - margin * 2));
  const height = Math.min(panelHeight, Math.max(180, bottomEdge - bounds.top - margin * 2));
  // The stock rect chooses a stable opening side, while the pointer drives the
  // actual position at a 1:1 rate. Anchoring to the far edge of a large stock
  // made the card feel detached; scaling the cell into a short lane made it
  // visibly lag behind the pointer.
  const sideAnchor = avoidRect || { left: x, right: x };
  const roomRight = rightEdge - sideAnchor.right - PANEL_GAP;
  const roomLeft = sideAnchor.left - bounds.left - PANEL_GAP;
  const opensRight = roomRight >= roomLeft;
  let left = opensRight ? x + PANEL_GAP : x - width - PANEL_GAP;
  // Stock cards sit beside the pointer instead of diagonally below it. Keep
  // the legacy category positioning because its larger anchor is already
  // stable and intentionally flips at the viewport edge.
  let top = avoidRect ? y - PANEL_POINTER_Y_OFFSET : y + PANEL_GAP;
  if (!avoidRect && top + height > bottomEdge - margin) top = y - height - PANEL_GAP;
  left = Math.max(bounds.left + margin, Math.min(left, rightEdge - width - margin));
  top = Math.max(bounds.top + margin, Math.min(top, bottomEdge - height - margin));
  return { mode: 'float' as const, left, top, width, maxHeight: Math.min(560, bottomEdge - bounds.top - margin * 2) };
}

export function normalizeLogoTicker(ticker: string): string {
  return ticker.trim().toUpperCase().replace(/\./g, '-');
}

export function logoUrl(ticker: string): string | null {
  const token = import.meta.env.VITE_LOGO_DEV_PUBLISHABLE_KEY as string | undefined;
  if (!token || !ticker) return null;
  return `https://img.logo.dev/ticker/${encodeURIComponent(normalizeLogoTicker(ticker))}?token=${encodeURIComponent(token)}&size=128&retina=true`;
}
