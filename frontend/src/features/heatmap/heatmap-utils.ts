const TOOLTIP_WIDTH = 292;
const TOOLTIP_HEIGHT = 154;
const TOOLTIP_GAP = 14;
export const HEATMAP_PANEL_WIDTH = 300;
export const HEATMAP_PANEL_DEFAULT_HEIGHT = 316;
export const HEATMAP_PANEL_MAX_HEIGHT = 360;
const PANEL_GAP = 18;
const PANEL_POINTER_Y_OFFSET = 48;

export type MarketSession = 'regular' | 'pre' | 'after' | 'closed';

export interface MarketSessionInfo {
  session: MarketSession;
  isRegularOpen: boolean;
  label: string;
}

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] as const;
const NY_CLOCK = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/New_York',
  weekday: 'short',
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
});

function isNyseHoliday(year: number, month: number, day: number, dow: number): boolean {
  // Observed fixed holidays: New Year's (including Dec 31 if Jan 1 falls on Sat), Juneteenth, July 4, Christmas
  if (month === 12 && day === 31 && dow === 5) return true;
  for (const [m, d] of [[1, 1], [6, 19], [7, 4], [12, 25]] as const) {
    const baseDow = new Date(Date.UTC(year, m - 1, d)).getUTCDay();
    const obsDay = baseDow === 6 ? d - 1 : baseDow === 0 ? d + 1 : d;
    const obs = new Date(Date.UTC(year, m - 1, obsDay));
    if (obs.getUTCFullYear() === year && obs.getUTCMonth() + 1 === month && obs.getUTCDate() === day) return true;
  }
  // Floating weekday holidays: MLK (3rd Mon Jan), Presidents (3rd Mon Feb), Memorial (last Mon May), Labor (1st Mon Sep), Thanksgiving (4th Thu Nov)
  if (month === 1 && dow === 1 && day >= 15 && day <= 21) return true;
  if (month === 2 && dow === 1 && day >= 15 && day <= 21) return true;
  if (month === 5 && dow === 1 && day >= 25) return true;
  if (month === 9 && dow === 1 && day <= 7) return true;
  if (month === 11 && dow === 4 && day >= 22 && day <= 28) return true;
  // Good Friday (Easter Sunday - 2 days via Anonymous Gregorian algorithm)
  const a = year % 19;
  const b = Math.floor(year / 100);
  const c = year % 100;
  const d = Math.floor(b / 4);
  const e = b % 4;
  const f = Math.floor((b + 8) / 25);
  const g = Math.floor((b - f + 1) / 3);
  const h = (19 * a + b - d - g + 15) % 30;
  const i = Math.floor(c / 4);
  const k = c % 4;
  const l = (32 + 2 * e + 2 * i - h - k) % 7;
  const m = Math.floor((a + 11 * h + 22 * l) / 451);
  const easterMonth = Math.floor((h + l - 7 * m + 114) / 31);
  const easterDay = ((h + l - 7 * m + 114) % 31) + 1;
  const goodFriday = new Date(Date.UTC(year, easterMonth - 1, easterDay - 2));
  return goodFriday.getUTCMonth() + 1 === month && goodFriday.getUTCDate() === day;
}

/**
 * Classifies the US cash equity market clock (`America/New_York`, 09:30-16:00 regular session).
 * Extended / closed hours render the matte Milky Way palette; open hours boost tile vibrancy.
 */
export function getMarketSessionState(now: Date = new Date()): MarketSessionInfo {
  const parts = Object.fromEntries(NY_CLOCK.formatToParts(now).map((part) => [part.type, part.value]));
  const dow = WEEKDAYS.indexOf(parts.weekday as (typeof WEEKDAYS)[number]);
  const year = Number(parts.year);
  const month = Number(parts.month);
  const day = Number(parts.day);
  const hour = Number(parts.hour) % 24;
  const minute = Number(parts.minute);

  if (dow <= 0 || dow === 6 || isNyseHoliday(year, month, day, dow)) {
    return { session: 'closed', isRegularOpen: false, label: 'Market closed · Muted palette' };
  }

  const minutes = hour * 60 + minute;
  if (minutes >= 570 && minutes < 960) {
    return { session: 'regular', isRegularOpen: true, label: 'Market open · Live palette' };
  }
  if (minutes >= 240 && minutes < 570) {
    return { session: 'pre', isRegularOpen: false, label: 'Pre-market · Muted palette' };
  }
  if (minutes >= 960 && minutes < 1200) {
    return { session: 'after', isRegularOpen: false, label: 'After-hours · Muted palette' };
  }
  return { session: 'closed', isRegularOpen: false, label: 'Market closed · Muted palette' };
}

const rgb = (hex: string) => [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16));
const mix = (from: string, to: string, amount: number) => {
  const a = rgb(from);
  const b = rgb(to);
  return `rgb(${a.map((channel, index) => Math.round(channel + (b[index] - channel) * amount)).join(',')})`;
};

export function getColorForChange(change?: number, session: MarketSession = 'closed'): string {
  const live = session === 'regular';
  if (change == null || !Number.isFinite(change)) return live ? '#333d4b' : '#283140';
  const strength = Math.min(1, Math.abs(change) / 4.5);
  if (change > 0) {
    return live
      ? mix('#225c42', '#12c45b', 0.26 + strength * 0.74)
      : mix('#233a35', '#52b896', 0.24 + strength * 0.76);
  }
  if (change < 0) {
    return live
      ? mix('#612834', '#eb344b', 0.26 + strength * 0.74)
      : mix('#3f2630', '#c96279', 0.24 + strength * 0.76);
  }
  return live ? '#3a4452' : '#2d3645';
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
    return { mode: 'sheet' as const, left: 0, top: 0, width: viewportWidth, maxHeight: Math.min(HEATMAP_PANEL_MAX_HEIGHT, viewportHeight * 0.78) };
  }
  const margin = 10;
  const width = Math.min(HEATMAP_PANEL_WIDTH, Math.max(240, bounds.width - margin * 2));
  const roomRight = bounds.right - anchor.right - margin;
  const roomLeft = anchor.left - bounds.left - margin;
  let left = roomRight >= width || roomRight >= roomLeft ? anchor.right + margin : anchor.left - width - margin;
  left = Math.max(bounds.left + margin, Math.min(left, bounds.right - width - margin));
  const maxHeight = Math.min(HEATMAP_PANEL_MAX_HEIGHT, bounds.height - margin * 2, viewportHeight - margin * 2);
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
  panelWidth = HEATMAP_PANEL_WIDTH,
  panelHeight = HEATMAP_PANEL_DEFAULT_HEIGHT,
  avoidRect?: RectBounds,
) {
  if (viewportWidth <= 640) {
    return { mode: 'sheet' as const, left: 0, top: 0, width: viewportWidth, maxHeight: Math.min(HEATMAP_PANEL_MAX_HEIGHT, viewportHeight * 0.78) };
  }
  const margin = 10;
  const rightEdge = Math.min(bounds.right, viewportWidth);
  const bottomEdge = Math.min(bounds.bottom, viewportHeight);
  const width = Math.min(panelWidth, Math.max(240, bounds.width - margin * 2));
  const height = Math.min(panelHeight, Math.max(160, bottomEdge - bounds.top - margin * 2));
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
  return { mode: 'float' as const, left, top, width, maxHeight: Math.min(HEATMAP_PANEL_MAX_HEIGHT, bottomEdge - bounds.top - margin * 2) };
}

export function normalizeLogoTicker(ticker: string): string {
  return ticker.trim().toUpperCase().replace(/\./g, '-');
}

export function logoUrl(ticker: string): string | null {
  const token = import.meta.env.VITE_LOGO_DEV_PUBLISHABLE_KEY as string | undefined;
  if (!token || !ticker) return null;
  return `https://img.logo.dev/ticker/${encodeURIComponent(normalizeLogoTicker(ticker))}?token=${encodeURIComponent(token)}&size=128&retina=true`;
}
