import React, { memo, useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import {
  categoryHeaderPadding,
  createTreemapHierarchy,
  detailLevel,
  layoutTreemap,
  stockTextSizes,
  type HeatmapData,
  type HeatmapNode,
  type LayoutNode,
} from './heatmap-layout';
import { CompanyLogo } from './CompanyLogo';
import { heatmapMetrics, recordLayout } from './performance';
import { getColorForChange } from './heatmap-utils';

const LOGO_INSTRUMENT_TYPES = new Set(['equity', 'etf', 'mutualfund']);

export interface CategoryAnchor {
  id: string;
  name: string;
  depth: number;
  pointer?: { x: number; y: number };
  rect: { left: number; top: number; right: number; bottom: number; width: number; height: number };
  containerRect: { left: number; top: number; right: number; bottom: number; width: number; height: number };
}

interface Props {
  data: HeatmapNode;
  width: number;
  height: number;
  zoom: number;
  hoveredCategoryId: string | null;
  onCategoryHover: (anchor: CategoryAnchor | null) => void;
  onCategoryPointerMove?: (x: number, y: number) => void;
  onLeafHover?: (leaf: HeatmapData | null) => void;
  onCategoryClick?: (anchor: CategoryAnchor) => void;
  onZoomDelta?: (delta: number) => void;
}

function rectForNode(node: LayoutNode, scroller: HTMLDivElement): CategoryAnchor['rect'] {
  const bounds = scroller.getBoundingClientRect();
  const left = bounds.left + node.x0 - scroller.scrollLeft;
  const top = bounds.top + node.y0 - scroller.scrollTop;
  const width = node.x1 - node.x0;
  const height = node.y1 - node.y0;
  return { left, top, right: left + width, bottom: top + height, width, height };
}

const HeatmapTreemap = memo(function HeatmapTreemap({
  data,
  width,
  height,
  zoom,
  hoveredCategoryId,
  onCategoryHover,
  onCategoryPointerMove,
  onLeafHover,
  onCategoryClick,
  onZoomDelta,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const activeLeafRef = useRef<string | null>(null);
  const activeCategoryRef = useRef<string | null>(null);
  const pendingZoomRef = useRef<{ previous: number; x: number; y: number; contentX: number; contentY: number } | null>(null);
  const wheelFrameRef = useRef<number | null>(null);
  const wheelDeltaRef = useRef(0);
  const wheelPointerRef = useRef({ x: 0, y: 0 });
  const dragRef = useRef<{
    pointerId: number; startX: number; startY: number; scrollLeft: number; scrollTop: number; moved: boolean;
  } | null>(null);
  const suppressClickRef = useRef(false);
  const scaledWidth = Math.max(1, Math.round(width * zoom));
  const scaledHeight = Math.max(1, Math.round(height * zoom));

  // Hierarchy construction only follows data; resquarify reuses its topology on resize.
  const hierarchyRoot = useMemo(() => createTreemapHierarchy(data), [data]);
  const layout = useMemo(() => {
    const started = performance.now();
    const next = layoutTreemap(hierarchyRoot, scaledWidth, scaledHeight);
    recordLayout(performance.now() - started);
    heatmapMetrics().resizeLayouts += 1;
    return next;
  }, [hierarchyRoot, scaledHeight, scaledWidth]);
  const leaves = useMemo(() => layout.leaves(), [layout]);
  const parents = useMemo(
    () => layout.descendants().filter((node) => node.depth > 0 && node.children) as LayoutNode[],
    [layout],
  );
  const leafEntries = useMemo(() => {
    const occurrences = new Map<string, number>();
    return leaves.map((leaf) => {
      const item = leaf.data as HeatmapData;
      const parent = leaf.parent as LayoutNode | null;
      const parentId = parent ? String((parent.data as HeatmapNode).id || parent.data.name) : 'root';
      const baseId = String(item.id || item.name);
      const collisionKey = `${parentId}/${baseId}`;
      const occurrence = occurrences.get(collisionKey) || 0;
      occurrences.set(collisionKey, occurrence + 1);
      return {
        leaf,
        parentId,
        renderId: occurrence ? `${collisionKey}#${occurrence}` : collisionKey,
      };
    });
  }, [leaves]);
  const leafById = useMemo(
    () => new Map(leafEntries.map(({ leaf, renderId }) => [renderId, leaf])),
    [leafEntries],
  );
  const categoryById = useMemo(() => new Map(parents.map((node) => [String((node.data as HeatmapNode).id || node.data.name), node])), [parents]);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element || !onZoomDelta) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      const bounds = element.getBoundingClientRect();
      wheelPointerRef.current = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
      wheelDeltaRef.current += event.deltaY;
      if (wheelFrameRef.current !== null) return;
      wheelFrameRef.current = requestAnimationFrame(() => {
        wheelFrameRef.current = null;
        const { x, y } = wheelPointerRef.current;
        pendingZoomRef.current = {
          previous: zoom,
          x,
          y,
          contentX: x + element.scrollLeft,
          contentY: y + element.scrollTop,
        };
        const delta = Math.max(-0.24, Math.min(0.24, -wheelDeltaRef.current * 0.0015));
        wheelDeltaRef.current = 0;
        if (Math.abs(delta) >= 0.01) onZoomDelta(delta);
      });
    };
    element.addEventListener('wheel', wheel, { passive: false });
    return () => {
      element.removeEventListener('wheel', wheel);
      if (wheelFrameRef.current !== null) cancelAnimationFrame(wheelFrameRef.current);
    };
  }, [onZoomDelta, zoom]);

  useLayoutEffect(() => {
    const element = scrollRef.current;
    const pending = pendingZoomRef.current;
    if (!element || !pending || pending.previous === zoom) return;
    const ratio = zoom / pending.previous;
    element.scrollLeft = Math.max(0, Math.min(element.scrollWidth - element.clientWidth, pending.contentX * ratio - pending.x));
    element.scrollTop = Math.max(0, Math.min(element.scrollHeight - element.clientHeight, pending.contentY * ratio - pending.y));
    pendingZoomRef.current = null;
  }, [zoom]);

  const anchorFor = (
    id: string,
    pointer?: { x: number; y: number },
    rectOverride?: CategoryAnchor['rect'],
  ): CategoryAnchor | null => {
    const node = categoryById.get(id);
    const scroller = scrollRef.current;
    if (!node || !scroller) return null;
    const bounds = scroller.getBoundingClientRect();
    return {
      id,
      name: String(node.data.name),
      depth: node.depth,
      pointer,
      rect: rectOverride || rectForNode(node, scroller),
      containerRect: {
        left: bounds.left, top: bounds.top, right: bounds.right, bottom: bounds.bottom,
        width: bounds.width, height: bounds.height,
      },
    };
  };

  const targetData = (target: EventTarget | null) =>
    target instanceof Element ? target.closest<HTMLElement>('[data-hm-leaf-id],[data-hm-category-id]') : null;

  const showCategory = (
    id: string | undefined,
    x?: number,
    y?: number,
    rectOverride?: CategoryAnchor['rect'],
    forceAnchorUpdate = false,
  ) => {
    if (!id) return;
    if (x != null && y != null) onCategoryPointerMove?.(x, y);
    if (activeCategoryRef.current === id && !forceAnchorUpdate) return;
    activeCategoryRef.current = id;
    const anchor = anchorFor(id, x != null && y != null ? { x, y } : undefined, rectOverride);
    if (anchor) onCategoryHover(anchor);
  };

  const onPointerOver = (event: React.PointerEvent<HTMLDivElement>) => {
    const target = targetData(event.target);
    if (!target) return;
    const leafId = target.dataset.hmLeafId;
    if (leafId) {
      const node = leafById.get(leafId);
      const leafChanged = activeLeafRef.current !== leafId;
      if (node && leafChanged) {
        activeLeafRef.current = leafId;
        onLeafHover?.(node.data as HeatmapData);
      }
      const scroller = scrollRef.current;
      showCategory(
        target.dataset.hmParentId,
        event.clientX,
        event.clientY,
        node && scroller ? rectForNode(node, scroller) : undefined,
        leafChanged,
      );
    } else {
      if (activeLeafRef.current) {
        activeLeafRef.current = null;
        onLeafHover?.(null);
      }
      showCategory(target.dataset.hmCategoryId, event.clientX, event.clientY);
    }
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    const element = scrollRef.current;
    if (drag && element && drag.pointerId === event.pointerId) {
      const deltaX = event.clientX - drag.startX;
      const deltaY = event.clientY - drag.startY;
      if (!drag.moved && Math.hypot(deltaX, deltaY) > 3) {
        drag.moved = true;
        activeLeafRef.current = null;
        onLeafHover?.(null);
        onCategoryHover(null);
      }
      if (drag.moved) {
        element.scrollLeft = drag.scrollLeft - deltaX;
        element.scrollTop = drag.scrollTop - deltaY;
      }
      return;
    }
    if (activeCategoryRef.current) onCategoryPointerMove?.(event.clientX, event.clientY);
  };

  const onPointerOut = (event: React.PointerEvent<HTMLDivElement>) => {
    const next = targetData(event.relatedTarget);
    const nextLeaf = next?.dataset.hmLeafId || null;
    const nextCategory = next?.dataset.hmParentId || next?.dataset.hmCategoryId || null;
    if (nextLeaf !== activeLeafRef.current) {
      activeLeafRef.current = nextLeaf;
      if (nextLeaf) {
        const node = leafById.get(nextLeaf);
        if (node) {
          onLeafHover?.(node.data as HeatmapData);
          const scroller = scrollRef.current;
          showCategory(
            nextCategory || undefined,
            event.clientX,
            event.clientY,
            scroller ? rectForNode(node, scroller) : undefined,
            true,
          );
        }
      } else if (nextCategory) {
        onLeafHover?.(null);
      }
    }
    if (nextCategory !== activeCategoryRef.current) {
      activeCategoryRef.current = nextCategory;
      if (nextCategory) {
        const anchor = anchorFor(nextCategory, { x: event.clientX, y: event.clientY });
        if (anchor) onCategoryHover(anchor);
      } else {
        onCategoryHover(null);
      }
    }
  };

  const activateCategory = (target: HTMLElement | null) => {
    const id = target?.dataset.hmCategoryId;
    if (!id || !onCategoryClick) return;
    const anchor = anchorFor(id);
    if (anchor) onCategoryClick(anchor);
  };

  const finishDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    suppressClickRef.current = drag.moved;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    event.currentTarget.style.cursor = '';
  };

  return (
    <div
      ref={scrollRef}
      onPointerOver={onPointerOver}
      onPointerMove={onPointerMove}
      onPointerOut={onPointerOut}
      onPointerDown={(event) => {
        if (zoom <= 1 || event.button !== 0 || !scrollRef.current) return;
        event.preventDefault();
        dragRef.current = {
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          scrollLeft: scrollRef.current.scrollLeft,
          scrollTop: scrollRef.current.scrollTop,
          moved: false,
        };
        event.currentTarget.setPointerCapture(event.pointerId);
        event.currentTarget.style.cursor = 'grabbing';
      }}
      onPointerUp={finishDrag}
      onPointerCancel={finishDrag}
      onClick={(event) => {
        if (suppressClickRef.current) {
          suppressClickRef.current = false;
          return;
        }
        activateCategory(targetData(event.target));
      }}
      onDoubleClick={(event) => {
        const target = targetData(event.target);
        const leafId = target?.dataset.hmLeafId;
        const item = leafId ? leafById.get(leafId)?.data as HeatmapData | undefined : undefined;
        if (!item || item.aggregateCount) return;
        const ticker = item.name.trim().toUpperCase().replace(/\./g, '-');
        window.open(`https://finance.yahoo.com/quote/${encodeURIComponent(ticker)}`, '_blank', 'noopener,noreferrer');
      }}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          const target = targetData(event.target);
          if (target?.dataset.hmCategoryId) {
            event.preventDefault();
            activateCategory(target);
          }
        }
      }}
      onFocus={(event) => {
        const target = targetData(event.target);
        const leafId = target?.dataset.hmLeafId;
        const node = leafId ? leafById.get(leafId) : undefined;
        if (node && target) {
          activeLeafRef.current = leafId || null;
          onLeafHover?.(node.data as HeatmapData);
          const scroller = scrollRef.current;
          showCategory(
            target.dataset.hmParentId,
            undefined,
            undefined,
            scroller ? rectForNode(node, scroller) : undefined,
            true,
          );
        } else {
          const id = target?.dataset.hmCategoryId;
          if (id) showCategory(id);
        }
      }}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) {
          onLeafHover?.(null);
          onCategoryHover(null);
        }
      }}
      aria-label="Interactive market heatmap. Use Tab to inspect categories and instruments."
      onDragStart={(event) => event.preventDefault()}
      className="custom-scrollbar relative min-w-0 max-w-full select-none bg-slate-950 outline-none"
      style={{ width: '100%', height, overflow: 'hidden', touchAction: zoom > 1 ? 'none' : 'auto', userSelect: 'none', WebkitUserSelect: 'none' }}
    >
      <div className="relative" style={{ width: scaledWidth, height: scaledHeight }}>
        {parents.map((node) => {
          const nodeData = node.data as HeatmapNode;
          const id = String(nodeData.id || `${node.depth}-${nodeData.name}`);
          const nodeWidth = node.x1 - node.x0;
          const nodeHeight = node.y1 - node.y0;
          if (nodeWidth < 24 || nodeHeight < 20) return null;
          const headerPadding = categoryHeaderPadding(node.depth, nodeWidth, nodeHeight);
          const active = hoveredCategoryId === id;
          return (
            <div
              key={id}
              data-hm-category-id={id}
              role="button"
              tabIndex={0}
              aria-label={`${node.depth === 1 ? 'Sector or asset class' : 'Industry or theme'}: ${nodeData.name}`}
              aria-pressed={active}
              className={`absolute overflow-hidden outline-none focus-visible:ring-2 focus-visible:ring-copper-400 ${zoom > 1 ? 'cursor-grab' : 'cursor-pointer'}`}
              style={{
                left: node.x0, top: node.y0, width: nodeWidth, height: nodeHeight,
                border: active ? '2px solid #d99a5b' : node.depth === 1 ? '1px solid #334155' : '1px solid #1e293b',
                backgroundColor: active ? '#d99a5b' : '#020617',
                boxShadow: active ? '0 0 0 2px rgba(217,154,91,.22), inset 0 0 18px rgba(217,154,91,.08)' : undefined,
                // Category geometry stays below stock cells so the copper
                // highlight never intercepts stock hover/focus events.
                zIndex: 1,
              }}
            >
              {headerPadding > 1 && (
                <div
                  className={node.depth === 1
                    ? 'pointer-events-none truncate bg-slate-900/95 px-1.5 pt-0.5 text-[10px] font-bold uppercase tracking-wide text-slate-200'
                    : 'pointer-events-none truncate bg-slate-800/95 px-1 text-[8px] font-semibold uppercase tracking-wide text-slate-400'}
                  style={{ height: headerPadding - 1 }}
                >
                  {nodeData.name}
                </div>
              )}
            </div>
          );
        })}

        {leafEntries.map(({ leaf, parentId, renderId }) => {
          const item = leaf.data as HeatmapData;
          const cellWidth = leaf.x1 - leaf.x0;
          const cellHeight = leaf.y1 - leaf.y0;
          if (cellWidth < 4 || cellHeight < 4) return null;
          const level = detailLevel(cellWidth, cellHeight);
          const textSizes = stockTextSizes(cellWidth, cellHeight, level);
          const change = item.changePercent || 0;
          const showTicker = level !== 'color';
          const showChange = ['change', 'logo', 'price'].includes(level);
          const fallbackLogoTicker = LOGO_INSTRUMENT_TYPES.has((item.instrumentType || '').toLowerCase())
            ? item.name
            : null;
          const logoTicker = item.logoTicker || fallbackLogoTicker;
          const showLogo = ['logo', 'price'].includes(level) && !!logoTicker && !item.aggregateCount;
          return (
            <div
              key={renderId}
              data-hm-leaf-id={renderId}
              data-hm-parent-id={parentId}
              role="button"
              tabIndex={0}
              aria-label={`${item.aggregateCount ? item.shortName : `${item.name}, ${item.shortName || ''}`}. Price ${item.price ?? 'unavailable'}. Daily change ${change >= 0 ? 'plus ' : 'minus '}${Math.abs(change).toFixed(2)} percent.`}
              className={`absolute z-[2] flex flex-col items-center justify-center overflow-hidden text-center text-white outline-none transition-[filter] duration-75 hover:brightness-125 focus-visible:z-10 focus-visible:ring-2 focus-visible:ring-white ${zoom > 1 ? 'cursor-grab' : 'cursor-crosshair'}`}
              style={{
                left: leaf.x0,
                top: leaf.y0,
                width: cellWidth,
                height: cellHeight,
                backgroundColor: getColorForChange(item.changePercent),
              }}
            >
              {showLogo && (
                <CompanyLogo
                  ticker={logoTicker || item.name}
                  label={item.shortName}
                  size={level === 'price' ? Math.min(42, cellHeight * 0.34) : Math.min(28, cellHeight * 0.3)}
                  className="mb-1"
                />
              )}
              {showTicker && (
                <strong
                  className="max-w-full truncate px-1 font-bold tracking-[-0.02em]"
                  style={{ fontSize: textSizes.ticker, lineHeight: 1.04, textShadow: '0 1px 2px rgba(0,0,0,.45)' }}
                >
                  {item.name}
                </strong>
              )}
              {showChange && (
                <span
                  className="font-semibold tabular-nums tracking-[-0.015em]"
                  style={{ fontSize: textSizes.change, lineHeight: 1.08, textShadow: '0 1px 2px rgba(0,0,0,.42)' }}
                >
                  {change > 0 ? '+' : ''}{change.toFixed(2)}%
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
});

export default HeatmapTreemap;
