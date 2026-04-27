import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { buildTreemapLayout, HeatmapNode, HeatmapData } from './heatmap-layout';
import HeatmapTooltip from './HeatmapTooltip';

export interface CategoryAnchor {
  name: string;
  /** Rect of the category cell in viewport (client) coordinates. */
  rect: { left: number; top: number; right: number; bottom: number; width: number; height: number };
}

interface HeatmapTreemapProps {
  data: HeatmapNode;
  width: number;
  height: number;
  zoom: number;
  hoveredCategory: string | null;
  onCategoryHover: (anchor: CategoryAnchor | null) => void;
  onCategoryClick?: (anchor: CategoryAnchor) => void;
  onZoomDelta?: (delta: number) => void;
}

export function getColorForChange(change: number | undefined): string {
  if (change === undefined) return '#1e293b';
  if (change >= 2.0) return '#15803d';
  if (change > 0) return '#22c55e';
  if (change <= -2.0) return '#b91c1c';
  if (change < 0) return '#ef4444';
  return '#334155';
}

const HeatmapTreemap: React.FC<HeatmapTreemapProps> = ({
  data,
  width,
  height,
  zoom,
  hoveredCategory,
  onCategoryHover,
  onCategoryClick,
  onZoomDelta,
}) => {
  const [hoveredNode, setHoveredNode] = useState<{ node: any; x: number; y: number } | null>(
    null,
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const hoverFrameRef = useRef<number | null>(null);
  const pendingHoverRef = useRef<{ node: any; x: number; y: number } | null>(null);
  const pendingZoomFocusRef = useRef<{
    prevZoom: number;
    /** Pointer position in viewport coordinates relative to scroll container */
    viewportX: number;
    viewportY: number;
    /** Pointer position in content coordinates (includes current scroll offsets) */
    contentX: number;
    contentY: number;
  } | null>(null);

  // Non-passive wheel listener: only hijack the wheel when the pointer
  // is actually over the heatmap AND Ctrl is NOT pressed (Ctrl+wheel
  // is the browser zoom gesture, never steal it). This way scrolling
  // outside the heatmap behaves as usual — addressing the "page scroll
  // disappears when I reach the heatmap" complaint.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !onZoomDelta) return;
    const handler = (e: WheelEvent) => {
      if (e.ctrlKey) return;
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      const r = el.getBoundingClientRect();
      const viewportX = e.clientX - r.left;
      const viewportY = e.clientY - r.top;
      // Convert to content coordinates (so we can keep the same content point
      // under the cursor after the parent updates zoom and the layout resizes).
      const contentX = viewportX + el.scrollLeft;
      const contentY = viewportY + el.scrollTop;
      pendingZoomFocusRef.current = { prevZoom: zoom, viewportX, viewportY, contentX, contentY };
      onZoomDelta(delta);
    };
    el.addEventListener('wheel', handler, { passive: false });
    return () => el.removeEventListener('wheel', handler);
  }, [onZoomDelta, zoom]);

  // Cursor-focused zoom: after zoom changes (which resizes the layout),
  // adjust scroll offsets so the content point under the cursor stays put.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    const pending = pendingZoomFocusRef.current;
    if (!el || !pending) return;
    if (pending.prevZoom === zoom) return;

    const ratio = zoom / pending.prevZoom;
    const nextContentX = pending.contentX * ratio;
    const nextContentY = pending.contentY * ratio;
    const nextScrollLeft = nextContentX - pending.viewportX;
    const nextScrollTop = nextContentY - pending.viewportY;

    // Clamp scroll to valid range (layout might be smaller than viewport at zoom=1).
    const maxScrollLeft = Math.max(0, el.scrollWidth - el.clientWidth);
    const maxScrollTop = Math.max(0, el.scrollHeight - el.clientHeight);
    el.scrollLeft = Math.max(0, Math.min(maxScrollLeft, nextScrollLeft));
    el.scrollTop = Math.max(0, Math.min(maxScrollTop, nextScrollTop));

    pendingZoomFocusRef.current = null;
  }, [zoom]);

  const scaledWidth = Math.max(1, Math.round(width * zoom));
  const scaledHeight = Math.max(1, Math.round(height * zoom));

  const layout = useMemo(() => {
    return buildTreemapLayout(data, scaledWidth, scaledHeight);
  }, [data, scaledWidth, scaledHeight]);

  const leaves = useMemo(() => layout.leaves(), [layout]);
  const parentNodes = useMemo(
    () => layout.descendants().filter((d) => d.depth > 0 && d.children),
    [layout],
  );

  const handleLeafMove = useCallback((e: React.MouseEvent, leaf: any) => {
    pendingHoverRef.current = { node: leaf, x: e.clientX, y: e.clientY };
    if (hoverFrameRef.current !== null) return;

    hoverFrameRef.current = window.requestAnimationFrame(() => {
      hoverFrameRef.current = null;
      if (pendingHoverRef.current) {
        setHoveredNode(pendingHoverRef.current);
      }
    });
  }, []);

  const handleLeafLeave = useCallback(() => {
    pendingHoverRef.current = null;
    if (hoverFrameRef.current !== null) {
      window.cancelAnimationFrame(hoverFrameRef.current);
      hoverFrameRef.current = null;
    }
    setHoveredNode(null);
  }, []);

  useEffect(() => () => {
    if (hoverFrameRef.current !== null) {
      window.cancelAnimationFrame(hoverFrameRef.current);
    }
  }, []);

  const emitCategoryAnchor = (el: HTMLElement | null, name: string) => {
    if (!el) return onCategoryHover({ name, rect: { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 } });
    const r = el.getBoundingClientRect();
    onCategoryHover({
      name,
      rect: {
        left: r.left, top: r.top, right: r.right, bottom: r.bottom,
        width: r.width, height: r.height,
      },
    });
  };

  return (
    <div
      ref={scrollRef}
      style={{
        width: '100%',
        height,
        position: 'relative',
        overflow: zoom > 1 ? 'auto' : 'hidden',
        backgroundColor: '#0f172a',
      }}
      className="custom-scrollbar"
    >
      <div
        style={{
          width: scaledWidth,
          height: scaledHeight,
          position: 'relative',
        }}
      >
        {parentNodes.map((p, i) => {
          const nodeWidth = p.x1 - p.x0;
          const nodeHeight = p.y1 - p.y0;
          if (nodeWidth < 20 || nodeHeight < 20) return null;

          const name = (p.data as any).name as string;
          const isHighlighted = hoveredCategory === name;

          return (
            <div
              key={`parent-${i}`}
              onMouseEnter={(e) => emitCategoryAnchor(e.currentTarget, name)}
              onMouseLeave={() => onCategoryHover(null)}
              onClick={(e) => {
                if (!onCategoryClick) return;
                const r = e.currentTarget.getBoundingClientRect();
                onCategoryClick({
                  name,
                  rect: {
                    left: r.left, top: r.top, right: r.right, bottom: r.bottom,
                    width: r.width, height: r.height,
                  },
                });
              }}
              style={{
                position: 'absolute',
                left: p.x0,
                top: p.y0,
                width: nodeWidth,
                height: nodeHeight,
                border: isHighlighted
                  ? '2px solid #facc15'
                  : '1px solid #1e293b',
                boxShadow: isHighlighted
                  ? '0 0 0 2px rgba(250, 204, 21, 0.25), inset 0 0 0 1px rgba(250, 204, 21, 0.4)'
                  : undefined,
                cursor: onCategoryClick ? 'pointer' : 'default',
                zIndex: isHighlighted ? 5 : 1,
                transition: 'box-shadow 120ms ease',
              }}
            >
              {p.depth <= 2 && nodeWidth > 50 && (
                <div
                  className={`px-1 pt-1 truncate bg-[#0f172a] bg-opacity-90 ${
                    p.depth === 1
                      ? 'text-[11px] font-semibold text-slate-300 uppercase tracking-widest'
                      : 'text-[9px] font-medium text-slate-500 uppercase tracking-wider'
                  }`}
                  style={{ pointerEvents: 'none' }}
                >
                  {name}
                </div>
              )}
            </div>
          );
        })}

        {leaves.map((leaf, i) => {
          const cellData = leaf.data as HeatmapData;
          const cellWidth = leaf.x1 - leaf.x0;
          const cellHeight = leaf.y1 - leaf.y0;
          if (cellWidth < 8 || cellHeight < 8) return null;

          const bgColor = getColorForChange(cellData.changePercent);
          const isLarge = cellWidth > 60 && cellHeight > 40;
          const isMedium = cellWidth > 40 && cellHeight > 25;

          return (
            <div
              key={`leaf-${i}`}
              onMouseMove={(e) => handleLeafMove(e, cellData)}
              onMouseLeave={handleLeafLeave}
              style={{
                position: 'absolute',
                left: leaf.x0,
                top: leaf.y0,
                width: cellWidth,
                height: cellHeight,
                backgroundColor: bgColor,
                border: '1px solid #0f172a',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                transition: 'filter 0.1s ease',
                overflow: 'hidden',
                zIndex: 2,
              }}
              className="hover:brightness-125"
            >
              <span
                className={`font-bold text-white ${
                  isLarge ? 'text-sm' : 'text-[10px]'
                } leading-tight truncate px-1`}
              >
                {cellData.name}
              </span>
              {isMedium && cellData.changePercent !== undefined && (
                <span
                  className={`font-thin text-white ${
                    isLarge ? 'text-xs' : 'text-[9px]'
                  } leading-tight`}
                >
                  {cellData.changePercent > 0 ? '+' : ''}
                  {cellData.changePercent.toFixed(2)}%
                </span>
              )}
              {isLarge && cellData.price !== undefined && (
                <span className="font-thin text-white text-[10px] opacity-80 leading-tight">
                  ${cellData.price.toFixed(2)}
                </span>
              )}
            </div>
          );
        })}

        {hoveredNode && (
          <HeatmapTooltip data={hoveredNode.node} x={hoveredNode.x} y={hoveredNode.y} />
        )}
      </div>
    </div>
  );
};

export default HeatmapTreemap;
