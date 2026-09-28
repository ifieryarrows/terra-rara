import React, { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import {
  createTreemapHierarchy,
  layoutTreemap,
  type HeatmapData,
  type HeatmapNode,
  type LayoutNode,
} from './heatmap-layout';
import { CategoryTiles, LeafTiles } from './HeatmapTiles';
import { heatmapMetrics, recordLayout } from './performance';

const MIN_ZOOM = 1;
const MAX_ZOOM = 4;

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

function rectForNode(node: LayoutNode, scroller: HTMLDivElement, scale: number): CategoryAnchor['rect'] {
  const bounds = scroller.getBoundingClientRect();
  const left = bounds.left + node.x0 * scale - scroller.scrollLeft;
  const top = bounds.top + node.y0 * scale - scroller.scrollTop;
  const width = (node.x1 - node.x0) * scale;
  const height = (node.y1 - node.y0) * scale;
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
  const zoomTargetRef = useRef(zoom);
  const displayZoomRef = useRef(zoom);
  const zoomAnchorRef = useRef<{ x: number; y: number; contentX: number; contentY: number } | null>(null);
  const zoomFrameRef = useRef<number | null>(null);
  const lastZoomFrameTimeRef = useRef<number | null>(null);
  const wheelFrameRef = useRef<number | null>(null);
  const wheelDeltaRef = useRef(0);
  const wheelPointerRef = useRef({ x: 0, y: 0 });
  const dragRef = useRef<{
    pointerId: number; startX: number; startY: number; scrollLeft: number; scrollTop: number; moved: boolean;
  } | null>(null);
  const suppressClickRef = useRef(false);
  const [displayZoom, setDisplayZoom] = useState(zoom);
  zoomTargetRef.current = zoom;
  const visualScale = displayZoom;
  const scaledWidth = Math.max(1, width * visualScale);
  const scaledHeight = Math.max(1, height * visualScale);

  // Hierarchy construction only follows data; resquarify reuses its topology on resize.
  const hierarchyRoot = useMemo(() => createTreemapHierarchy(data), [data]);
  const layout = useMemo(() => {
    const started = performance.now();
    const next = layoutTreemap(hierarchyRoot, width, height);
    recordLayout(performance.now() - started);
    heatmapMetrics().resizeLayouts += 1;
    return next;
  }, [hierarchyRoot, height, width]);
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
    if (zoomFrameRef.current !== null) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      displayZoomRef.current = zoomTargetRef.current;
      setDisplayZoom(zoomTargetRef.current);
      return;
    }

    const animate = (timestamp: number) => {
      const previous = lastZoomFrameTimeRef.current ?? timestamp;
      const elapsed = Math.min(48, Math.max(0, timestamp - previous));
      lastZoomFrameTimeRef.current = timestamp;
      const current = displayZoomRef.current;
      const target = zoomTargetRef.current;
      const next = current + (target - current) * (1 - Math.exp(-elapsed / 68));
      if (Math.abs(target - next) < 0.001) {
        displayZoomRef.current = target;
        setDisplayZoom(target);
        zoomFrameRef.current = null;
        lastZoomFrameTimeRef.current = null;
        return;
      }
      displayZoomRef.current = next;
      setDisplayZoom(next);
      zoomFrameRef.current = window.requestAnimationFrame(animate);
    };

    zoomFrameRef.current = window.requestAnimationFrame(animate);
  }, [zoom]);

  useEffect(() => () => {
    if (zoomFrameRef.current !== null) window.cancelAnimationFrame(zoomFrameRef.current);
    zoomFrameRef.current = null;
    lastZoomFrameTimeRef.current = null;
  }, []);

  useLayoutEffect(() => {
    const element = scrollRef.current;
    const anchor = zoomAnchorRef.current;
    if (!element || !anchor) return;
    element.scrollLeft = Math.max(0, Math.min(element.scrollWidth - element.clientWidth, anchor.contentX * displayZoom - anchor.x));
    element.scrollTop = Math.max(0, Math.min(element.scrollHeight - element.clientHeight, anchor.contentY * displayZoom - anchor.y));
    if (Math.abs(displayZoom - zoomTargetRef.current) < 0.001) zoomAnchorRef.current = null;
  }, [displayZoom]);

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
        const delta = Math.max(-0.24, Math.min(0.24, -wheelDeltaRef.current * 0.0015));
        wheelDeltaRef.current = 0;
        if (Math.abs(delta) < 0.01) return;

        const currentTarget = zoomTargetRef.current;
        const nextZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, +(currentTarget + delta).toFixed(2)));
        if (nextZoom === currentTarget) return;
        const currentScale = Math.max(0.001, displayZoomRef.current);
        zoomAnchorRef.current = {
          x,
          y,
          contentX: (x + element.scrollLeft) / currentScale,
          contentY: (y + element.scrollTop) / currentScale,
        };
        zoomTargetRef.current = nextZoom;
        onZoomDelta(nextZoom - currentTarget);
      });
    };
    element.addEventListener('wheel', wheel, { passive: false });
    return () => {
      element.removeEventListener('wheel', wheel);
      if (wheelFrameRef.current !== null) cancelAnimationFrame(wheelFrameRef.current);
    };
  }, [onZoomDelta]);

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
      rect: rectOverride || rectForNode(node, scroller, visualScale),
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
        node && scroller ? rectForNode(node, scroller, visualScale) : undefined,
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
            scroller ? rectForNode(node, scroller, visualScale) : undefined,
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
            scroller ? rectForNode(node, scroller, visualScale) : undefined,
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
        <div className="cm-heatmap-preview-content relative" style={{ width, height, transform: visualScale === 1 ? undefined : `scale(${visualScale})`, transformOrigin: 'top left' }}>
          <CategoryTiles parents={parents} hoveredCategoryId={hoveredCategoryId} zoom={zoom} />
          <LeafTiles leafEntries={leafEntries} zoom={zoom} />
        </div>
      </div>
    </div>
  );
});

export default HeatmapTreemap;
