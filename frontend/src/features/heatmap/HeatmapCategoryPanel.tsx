import React, { useLayoutEffect, useRef, useState } from 'react';
import { HeatmapData } from './heatmap-layout';

interface Anchor {
  /** Rect of the hovered / pinned category cell in viewport coordinates. */
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

interface HeatmapCategoryPanelProps {
  categoryName: string | null;
  leaves: HeatmapData[];
  onClose: () => void;
  /**
   * Rect (in viewport coords) of the category the panel should
   * float next to. When null we fall back to the bottom-right of
   * the container.
   */
  anchor?: Anchor | null;
  /**
   * Bounding rect of the heatmap body so the popover never escapes
   * the dashboard card. When omitted we clamp to the viewport.
   */
  containerRect?: DOMRect | null;
}

const PANEL_WIDTH = 300;
const MARGIN = 10;
const MAX_HEIGHT = 420;

/**
 * Category Inspector — renders as a compact floating popover that flips
 * to whichever side of the anchor has the most room, and collapses to a
 * full-width bottom sheet on narrow viewports.
 *
 * Critically, this component does NOT shrink the treemap (it is
 * position:fixed/absolute), preserving Finviz-style heatmap sizing.
 */
const HeatmapCategoryPanel: React.FC<HeatmapCategoryPanelProps> = ({
  categoryName,
  leaves,
  onClose,
  anchor,
  containerRect,
}) => {
  const panelRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{
    left: number;
    top: number;
    maxHeight: number;
    mode: 'float' | 'sheet';
  }>({ left: 0, top: 0, maxHeight: MAX_HEIGHT, mode: 'float' });

  useLayoutEffect(() => {
    if (!categoryName) return;

    const vpW = window.innerWidth;
    const vpH = window.innerHeight;

    // Bottom-sheet fallback on narrow viewports (<= 640px)
    if (vpW <= 640) {
      setPos({ left: 0, top: vpH - Math.min(MAX_HEIGHT, vpH * 0.75), maxHeight: Math.min(MAX_HEIGHT, vpH * 0.75), mode: 'sheet' });
      return;
    }

    const bound = containerRect || {
      left: 0,
      top: 0,
      right: vpW,
      bottom: vpH,
      width: vpW,
      height: vpH,
    };

    const a = anchor || {
      left: bound.right - PANEL_WIDTH - MARGIN,
      top: bound.top + MARGIN,
      right: bound.right,
      bottom: bound.top + MARGIN,
      width: PANEL_WIDTH,
      height: 0,
    };

    // Prefer placing on the side with more room
    const roomRight = bound.right - a.right - MARGIN;
    const roomLeft = a.left - bound.left - MARGIN;
    const placeRight = roomRight >= PANEL_WIDTH || roomRight >= roomLeft;

    let left = placeRight ? a.right + MARGIN : a.left - PANEL_WIDTH - MARGIN;
    // Clamp horizontally within container
    left = Math.max(bound.left + MARGIN, Math.min(left, bound.right - PANEL_WIDTH - MARGIN));

    let top = a.top;
    const maxH = Math.min(MAX_HEIGHT, bound.bottom - bound.top - MARGIN * 2);
    if (top + maxH > bound.bottom - MARGIN) {
      top = Math.max(bound.top + MARGIN, bound.bottom - maxH - MARGIN);
    }
    top = Math.max(bound.top + MARGIN, top);

    setPos({ left, top, maxHeight: maxH, mode: 'float' });
  }, [categoryName, anchor, containerRect]);

  if (!categoryName) return null;

  const sorted = [...leaves].sort(
    (a, b) => (b.changePercent ?? 0) - (a.changePercent ?? 0),
  );

  const up = sorted.filter((l) => (l.changePercent ?? 0) > 0).length;
  const down = sorted.filter((l) => (l.changePercent ?? 0) < 0).length;

  const floatingStyle: React.CSSProperties = pos.mode === 'sheet'
    ? {
        position: 'fixed',
        left: 0,
        right: 0,
        bottom: 0,
        maxHeight: pos.maxHeight,
        zIndex: 60,
      }
    : {
        position: 'fixed',
        left: pos.left,
        top: pos.top,
        width: PANEL_WIDTH,
        maxHeight: pos.maxHeight,
        zIndex: 60,
      };

  return (
    <aside
      ref={panelRef}
      style={floatingStyle}
      className="bg-[#0b1220] border border-slate-700 shadow-2xl flex flex-col rounded-md overflow-hidden"
      onMouseEnter={(e) => e.stopPropagation()}
    >
      <header className="flex items-center justify-between px-3 py-2.5 border-b border-slate-700">
        <div className="min-w-0">
          <p className="text-[10px] uppercase tracking-widest text-slate-500 leading-none">
            Category Inspector
          </p>
          <h3 className="text-sm font-semibold text-white truncate">{categoryName}</h3>
          <p className="text-[10px] text-slate-500 mt-0.5">
            {sorted.length} symbols · <span className="text-emerald-400">{up}↑</span>{' '}
            <span className="text-rose-400">{down}↓</span>
          </p>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-white text-xs px-2 py-1 border border-slate-700 rounded"
        >
          Close
        </button>
      </header>

      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {sorted.length === 0 ? (
          <div className="p-4 text-slate-500 text-xs">
            No symbols in this category have live data.
          </div>
        ) : (
          <ul className="divide-y divide-slate-800">
            {sorted.map((item) => {
              const ch = item.changePercent ?? 0;
              return (
                <li
                  key={item.name}
                  className="px-3 py-2 hover:bg-slate-800/50 transition-colors"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <div className="min-w-0">
                      <div className="text-sm font-semibold text-white truncate">
                        {item.name}
                      </div>
                      <div className="text-[10px] text-slate-500 truncate">
                        {item.shortName || item.subgroup || ''}
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="text-sm font-mono text-slate-200 tabular-nums">
                        {item.price !== undefined
                          ? `$${item.price.toFixed(item.price < 10 ? 4 : 2)}`
                          : '—'}
                      </div>
                      <div
                        className={`text-[11px] font-mono font-semibold tabular-nums ${
                          ch > 0
                            ? 'text-emerald-400'
                            : ch < 0
                            ? 'text-rose-400'
                            : 'text-slate-500'
                        }`}
                      >
                        {item.changePercent !== undefined
                          ? `${ch > 0 ? '+' : ''}${ch.toFixed(2)}%`
                          : '—'}
                      </div>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
};

export default HeatmapCategoryPanel;
