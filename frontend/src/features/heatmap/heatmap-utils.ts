const TOOLTIP_WIDTH = 292;
const TOOLTIP_HEIGHT = 154;
const TOOLTIP_GAP = 14;
const PANEL_WIDTH = 380;
const PANEL_GAP = 18;

const rgb = (hex: string) => [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16));
const mix = (from: string, to: string, amount: number) => {
  const a = rgb(from);
  const b = rgb(to);
  return `rgb(${a.map((channel, index) => Math.round(channel + (b[index] - channel) * amount)).join(',')})`;
};

export function getColorForChange(change?: number): string {
  if (change == null || !Number.isFinite(change)) return '#3f4652';
  const strength = Math.min(1, Math.abs(change) / 4.5);
  if (change > 0) return mix('#315445', '#12c45b', 0.22 + strength * 0.78);
  if (change < 0) return mix('#60434a', '#ff3048', 0.22 + strength * 0.78);
  return '#414852';
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
  const horizontalAnchor = avoidRect || { left: x, right: x };
  const roomRight = rightEdge - horizontalAnchor.right - PANEL_GAP;
  const roomLeft = horizontalAnchor.left - bounds.left - PANEL_GAP;
  let left = roomRight >= width || roomRight >= roomLeft
    ? horizontalAnchor.right + PANEL_GAP
    : horizontalAnchor.left - width - PANEL_GAP;
  let top = y + PANEL_GAP;
  if (top + height > bottomEdge - margin) top = y - height - PANEL_GAP;
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
