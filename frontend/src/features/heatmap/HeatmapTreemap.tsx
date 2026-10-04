import React, { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import {
  createTreemapHierarchy,
  HEATMAP_MAX_ZOOM,
  layoutTreemap,
  type HeatmapData,
  type HeatmapNode,
  type LayoutNode,
} from './heatmap-layout';
import { CategoryTiles, LeafTiles } from './HeatmapTiles';
import { heatmapMetrics, recordLayout } from './performance';

const MIN_ZOOM = 1;

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
  onCategoryDrillDown?: (anchor: CategoryAnchor) => void;
  onNavigateBack?: () => void;
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
  onCategoryDrillDown,
  onNavigateBack,
  onZoomDelta,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const categoryElementsRef = useRef(new Map<string, HTMLElement>());
  const hoveredCategoryElementRef = useRef<HTMLElement | null>(null);
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
  const [layoutZoom, setLayoutZoom] = useState(zoom);
  const scaledWidth = Math.max(1, Math.round(width * zoom));
  const scaledHeight = Math.max(1, Math.round(height * zoom));
  const layoutWidth = Math.max(1, Math.round(width * layoutZoom));
  const layoutHeight = Math.max(1, Math.round(height * layoutZoom));
  const visualScale = zoom / layoutZoom;

  useEffect(() => {
    if (zoom === layoutZoom) return;
    const timer = window.setTimeout(() => setLayoutZoom(zoom), 120);
    return () => window.clearTimeout(timer);
  }, [layoutZoom, zoom]);

  // Hierarchy construction only follows data; resquarify reuses its topology on resize.
  const hierarchyRoot = useMemo(() => createTreemapHierarchy(data), [data]);
  const layout = useMemo(() => {
    const started = performance.now();
    const next = layoutTreemap(hierarchyRoot, layoutWidth, layoutHeight);
    recordLayout(performance.now() - started);
    heatmapMetrics().resizeLayouts += 1;
    return next;
  }, [hierarchyRoot, layoutHeight, layoutWidth]);
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
      const parentData = parent?.data as HeatmapNode | undefined;
      const parentId = parent
        ? String(parentData?.id || `${parent.depth}-${parentData?.name || parent.data.name}`)
        : 'root';
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
  const categoryById = useMemo(() => new Map(parents.map((node) => {
    const nodeData = node.data as HeatmapNode;
    return [String(nodeData.id || `${node.depth}-${nodeData.name}`), node] as const;
  })), [parents]);

  const setCategoryHoverVisual = (id: string | null) => {
    const previous = hoveredCategoryElementRef.current;
    const next = id ? categoryElementsRef.current.get(id) || null : null;
    if (previous === next) return;
    const restore = (element: HTMLElement) => {
      element.style.border = element.dataset.hmCategoryId === hoveredCategoryId
        ? '1px solid #d99a5b'
        : '0.5px solid #253244';
      element.style.backgroundColor = '#020617';
      element.style.boxShadow = 'none';
    };
    if (previous) restore(previous);
    hoveredCategoryElementRef.current = next;
    if (next && next.dataset.hmCategoryId !== hoveredCategoryId) {
      next.style.border = '1px solid #d99a5b';
      next.style.backgroundColor = '#020617';
      next.style.boxShadow = 'none';
    }
  };

  useLayoutEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller) return;
    categoryElementsRef.current = new Map(
      Array.from(scroller.querySelectorAll<HTMLElement>('[data-hm-category-id]'))
        .map((element) => [element.dataset.hmCategoryId || '', element] as const)
        .filter(([id]) => !!id),
    );
    if (hoveredCategoryElementRef.current && !scroller.contains(hoveredCategoryElementRef.current)) {
      hoveredCategoryElementRef.current = null;
    }
  }, [height, layoutZoom, leafEntries, parents, width]);

  useLayoutEffect(() => {
    const element = scrollRef.current;
    const pending = pendingZoomRef.current;
    if (!element || !pending || pending.previous === zoom) return;
    const ratio = zoom / pending.previous;
    element.scrollLeft = Math.max(0, Math.min(element.scrollWidth - element.clientWidth, pending.contentX * ratio - pending.x));
    element.scrollTop = Math.max(0, Math.min(element.scrollHeight - element.clientHeight, pending.contentY * ratio - pending.y));
    pendingZoomRef.current = null;
  }, [zoom]);

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
        const nextZoom = Math.max(MIN_ZOOM, Math.min(HEATMAP_MAX_ZOOM, +(zoom + delta).toFixed(2)));
        if (nextZoom === zoom) {
          if (delta < 0 && zoom <= MIN_ZOOM) onNavigateBack?.();
          return;
        }
        pendingZoomRef.current = {
          previous: zoom,
          x,
          y,
          contentX: x + element.scrollLeft,
          contentY: y + element.scrollTop,
        };
        onZoomDelta(nextZoom - zoom);
      });
    };
    element.addEventListener('wheel', wheel, { passive: false });
    return () => {
      element.removeEventListener('wheel', wheel);
      if (wheelFrameRef.current !== null) cancelAnimationFrame(wheelFrameRef.current);
      wheelFrameRef.current = null;
      wheelDeltaRef.current = 0;
    };
  }, [onNavigateBack, onZoomDelta, zoom]);

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
    setCategoryHoverVisual(id);
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
        setCategoryHoverVisual(null);
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
    setCategoryHoverVisual(nextCategory);
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
    delete event.currentTarget.dataset.panning;
  };

  return (
    <div
      ref={scrollRef}
      onPointerOver={onPointerOver}
      onPointerMove={onPointerMove}
      onPointerOut={onPointerOut}
      onPointerDown={(event) => {
        if (zoom <= MIN_ZOOM || event.button !== 0 || !scrollRef.current) return;
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
        event.currentTarget.dataset.panning = 'true';
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
        const categoryId = target?.dataset.hmCategoryId;
        if (categoryId && onCategoryDrillDown) {
          const anchor = anchorFor(categoryId, { x: event.clientX, y: event.clientY });
          if (anchor) {
            event.preventDefault();
            onCategoryDrillDown(anchor);
          }
          return;
        }
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
            if (event.shiftKey && onCategoryDrillDown) {
              const anchor = anchorFor(target.dataset.hmCategoryId);
              if (anchor) onCategoryDrillDown(anchor);
            } else activateCategory(target);
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
          setCategoryHoverVisual(null);
          onLeafHover?.(null);
          onCategoryHover(null);
        }
      }}
      aria-label="Interactive market heatmap. Use Tab to inspect categories and instruments."
      onDragStart={(event) => event.preventDefault()}
      className="cm-heatmap-map custom-scrollbar relative min-w-0 max-w-full select-none bg-slate-950 outline-none"
      data-zoomed={String(zoom > MIN_ZOOM)}
      style={{ width: '100%', height, overflow: 'hidden', touchAction: zoom > MIN_ZOOM ? 'none' : 'auto', userSelect: 'none', WebkitUserSelect: 'none' }}
    >
      <div className="relative" style={{ width: scaledWidth, height: scaledHeight }}>
        <div
          className="relative"
          style={{
            width: layoutWidth,
            height: layoutHeight,
            transform: visualScale === 1 ? undefined : `scale(${visualScale})`,
            transformOrigin: 'top left',
          }}
        >
          <CategoryTiles parents={parents} hoveredCategoryId={hoveredCategoryId} zoom={layoutZoom} />
          <LeafTiles leafEntries={leafEntries} zoom={layoutZoom} />
        </div>
      </div>
    </div>
  );
});

export default HeatmapTreemap;
