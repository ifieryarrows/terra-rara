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
  resetKey?: string | number;
}

interface ZoomCamera {
  scale: number;
  x: number;
  y: number;
}

function clampCamera(camera: ZoomCamera, scroller: HTMLDivElement, contentWidth: number, contentHeight: number): ZoomCamera {
  const minX = Math.min(0, scroller.clientWidth - contentWidth * camera.scale);
  const minY = Math.min(0, scroller.clientHeight - contentHeight * camera.scale);
  return {
    scale: camera.scale,
    x: Math.max(minX, Math.min(0, camera.x)),
    y: Math.max(minY, Math.min(0, camera.y)),
  };
}

function localPoint(scroller: HTMLDivElement, clientX: number, clientY: number) {
  const bounds = scroller.getBoundingClientRect();
  return {
    x: (clientX - bounds.left) * scroller.clientWidth / Math.max(bounds.width, 1),
    y: (clientY - bounds.top) * scroller.clientHeight / Math.max(bounds.height, 1),
  };
}

function rectForNode(node: LayoutNode, scroller: HTMLDivElement, camera: ZoomCamera): CategoryAnchor['rect'] {
  const bounds = scroller.getBoundingClientRect();
  const screenScaleX = bounds.width / Math.max(scroller.clientWidth, 1);
  const screenScaleY = bounds.height / Math.max(scroller.clientHeight, 1);
  const left = bounds.left + (node.x0 * camera.scale + camera.x) * screenScaleX;
  const top = bounds.top + (node.y0 * camera.scale + camera.y) * screenScaleY;
  const width = (node.x1 - node.x0) * camera.scale * screenScaleX;
  const height = (node.y1 - node.y0) * camera.scale * screenScaleY;
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
  resetKey,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const surfaceRef = useRef<HTMLDivElement>(null);
  const activeLeafRef = useRef<string | null>(null);
  const activeCategoryRef = useRef<string | null>(null);
  const zoomTargetRef = useRef(zoom);
  const previousZoomPropRef = useRef(zoom);
  const cameraRef = useRef<ZoomCamera>({ scale: zoom, x: 0, y: 0 });
  const zoomAnchorRef = useRef<{ x: number; y: number; contentX: number; contentY: number } | null>(null);
  const zoomFrameRef = useRef<number | null>(null);
  const lastZoomFrameTimeRef = useRef<number | null>(null);
  const startZoomAnimationRef = useRef<() => void>(() => {});
  const wheelFrameRef = useRef<number | null>(null);
  const wheelDeltaRef = useRef(0);
  const wheelPointerRef = useRef({ x: 0, y: 0 });
  const dragRef = useRef<{
    pointerId: number; startX: number; startY: number; x: number; y: number; moved: boolean;
  } | null>(null);
  const suppressClickRef = useRef(false);
  const [zoomed, setZoomed] = useState(zoom > MIN_ZOOM);
  const [detailZoom, setDetailZoom] = useState(zoom);
  const zoomedRef = useRef(zoom > MIN_ZOOM);
  const dimensionsRef = useRef({ width, height });
  dimensionsRef.current = { width, height };

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

  const applyCamera = (next: ZoomCamera) => {
    const scroller = scrollRef.current;
    const surface = surfaceRef.current;
    if (!scroller || !surface) return;
    const { width: contentWidth, height: contentHeight } = dimensionsRef.current;
    const camera = clampCamera(next, scroller, contentWidth, contentHeight);
    cameraRef.current = camera;
    // Transform the stable treemap geometry directly. A CSS `zoom` raster
    // layer made coordinates and text detail drift apart while the camera
    // moved, and forced the browser to resample a very large texture.
    surface.style.transform = `matrix(${camera.scale}, 0, 0, ${camera.scale}, ${camera.x}, ${camera.y})`;
  };

  const setZoomedTarget = (target: number) => {
    zoomTargetRef.current = target;
    const nextZoomed = target > MIN_ZOOM;
    if (nextZoomed !== zoomedRef.current) {
      zoomedRef.current = nextZoomed;
      setZoomed(nextZoomed);
    }
  };

  const startZoomAnimation = () => {
    if (zoomFrameRef.current !== null) return;
    const applyTarget = () => {
      const target = zoomTargetRef.current;
      const current = cameraRef.current;
      const anchor = zoomAnchorRef.current;
      applyCamera({
        scale: target,
        x: anchor ? anchor.x - anchor.contentX * target : current.x,
        y: anchor ? anchor.y - anchor.contentY * target : current.y,
      });
      setDetailZoom(target);
      zoomAnchorRef.current = null;
      lastZoomFrameTimeRef.current = null;
    };

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      applyTarget();
      return;
    }

    const animate = (timestamp: number) => {
      const previous = lastZoomFrameTimeRef.current ?? timestamp;
      const elapsed = Math.min(48, Math.max(0, timestamp - previous));
      lastZoomFrameTimeRef.current = timestamp;
      const current = cameraRef.current;
      const target = zoomTargetRef.current;
      const nextScale = current.scale + (target - current.scale) * (1 - Math.exp(-elapsed / 68));
      const finished = Math.abs(target - nextScale) < 0.001;
      const scale = finished ? target : nextScale;
      const anchor = zoomAnchorRef.current;
      applyCamera({
        scale,
        x: anchor ? anchor.x - anchor.contentX * scale : current.x,
        y: anchor ? anchor.y - anchor.contentY * scale : current.y,
      });
      if (finished) {
        setDetailZoom(target);
        zoomAnchorRef.current = null;
        zoomFrameRef.current = null;
        lastZoomFrameTimeRef.current = null;
        return;
      }
      zoomFrameRef.current = window.requestAnimationFrame(animate);
    };

    zoomFrameRef.current = window.requestAnimationFrame(animate);
  };
  startZoomAnimationRef.current = startZoomAnimation;

  const centerAnchor = () => {
    const scroller = scrollRef.current;
    if (!scroller) return null;
    const camera = cameraRef.current;
    const x = scroller.clientWidth / 2;
    const y = scroller.clientHeight / 2;
    return {
      x,
      y,
      contentX: (x - camera.x) / camera.scale,
      contentY: (y - camera.y) / camera.scale,
    };
  };

  useLayoutEffect(() => {
    applyCamera(cameraRef.current);
  }, [height, width]);

  useEffect(() => {
    if (zoom === previousZoomPropRef.current) return;
    previousZoomPropRef.current = zoom;
    if (Math.abs(zoom - zoomTargetRef.current) < 0.001) return;
    zoomAnchorRef.current = centerAnchor();
    setZoomedTarget(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom)));
    startZoomAnimationRef.current();
  }, [zoom]);

  const previousResetKeyRef = useRef(resetKey);
  useEffect(() => {
    if (Object.is(resetKey, previousResetKeyRef.current)) return;
    previousResetKeyRef.current = resetKey;
    if (zoomTargetRef.current === MIN_ZOOM) return;
    zoomAnchorRef.current = centerAnchor();
    setZoomedTarget(MIN_ZOOM);
    startZoomAnimationRef.current();
  }, [resetKey]);

  useEffect(() => () => {
    if (zoomFrameRef.current !== null) window.cancelAnimationFrame(zoomFrameRef.current);
    zoomFrameRef.current = null;
    lastZoomFrameTimeRef.current = null;
  }, []);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      wheelPointerRef.current = localPoint(element, event.clientX, event.clientY);
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
        const camera = cameraRef.current;
        zoomAnchorRef.current = {
          x,
          y,
          contentX: (x - camera.x) / camera.scale,
          contentY: (y - camera.y) / camera.scale,
        };
        setZoomedTarget(nextZoom);
        startZoomAnimationRef.current();
        onZoomDelta?.(nextZoom - currentTarget);
      });
    };
    element.addEventListener('wheel', wheel, { passive: false });
    return () => {
      element.removeEventListener('wheel', wheel);
      if (wheelFrameRef.current !== null) cancelAnimationFrame(wheelFrameRef.current);
      wheelFrameRef.current = null;
      wheelDeltaRef.current = 0;
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
      rect: rectOverride || rectForNode(node, scroller, cameraRef.current),
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
        node && scroller ? rectForNode(node, scroller, cameraRef.current) : undefined,
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
        const bounds = element.getBoundingClientRect();
        const scaleX = element.clientWidth / Math.max(bounds.width, 1);
        const scaleY = element.clientHeight / Math.max(bounds.height, 1);
        applyCamera({
          ...cameraRef.current,
          x: drag.x + deltaX * scaleX,
          y: drag.y + deltaY * scaleY,
        });
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
            scroller ? rectForNode(node, scroller, cameraRef.current) : undefined,
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
        if (zoomTargetRef.current <= MIN_ZOOM || event.button !== 0 || !scrollRef.current) return;
        event.preventDefault();
        dragRef.current = {
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          x: cameraRef.current.x,
          y: cameraRef.current.y,
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
            scroller ? rectForNode(node, scroller, cameraRef.current) : undefined,
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
      style={{ width: '100%', height, overflow: 'hidden', touchAction: zoomed ? 'none' : 'auto', userSelect: 'none', WebkitUserSelect: 'none' }}
    >
      <div
        ref={surfaceRef}
        className="relative"
        style={{
          width,
          height,
          transform: `matrix(${zoom}, 0, 0, ${zoom}, 0, 0)`,
          transformOrigin: 'top left',
        }}
      >
        <div className="relative" style={{ width, height }}>
          <CategoryTiles parents={parents} hoveredCategoryId={hoveredCategoryId} zoomed={zoomed} />
          <LeafTiles leafEntries={leafEntries} zoomed={zoomed} detailZoom={detailZoom} />
        </div>
      </div>
    </div>
  );
});

export default HeatmapTreemap;
