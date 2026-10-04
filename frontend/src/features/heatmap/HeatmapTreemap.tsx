import React, { memo, useEffect, useLayoutEffect, useMemo, useRef } from 'react';
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
const MAX_ZOOM = HEATMAP_MAX_ZOOM;

/** An element shown only while the camera scale is within [minScale, maxScale). */
interface LodEntry {
  element: HTMLElement;
  minScale: number;
  maxScale: number;
}

interface LodIndex {
  entries: LodEntry[];
  /** Every finite range boundary, sorted ascending, for incremental updates. */
  boundaries: Array<{ scale: number; entry: LodEntry }>;
}

function buildLodIndex(surface: HTMLElement): LodIndex {
  const entries = Array.from(surface.querySelectorAll<HTMLElement>('[data-hm-lod-min]')).map((element) => {
    const min = Number(element.dataset.hmLodMin);
    const max = element.dataset.hmLodMax == null ? Number.NaN : Number(element.dataset.hmLodMax);
    return {
      element,
      minScale: Number.isFinite(min) ? min : 1,
      maxScale: Number.isFinite(max) ? max : Number.POSITIVE_INFINITY,
    };
  });
  const boundaries: LodIndex['boundaries'] = [];
  for (const entry of entries) {
    boundaries.push({ scale: entry.minScale, entry });
    if (Number.isFinite(entry.maxScale)) boundaries.push({ scale: entry.maxScale, entry });
  }
  boundaries.sort((a, b) => a.scale - b.scale);
  return { entries, boundaries };
}

function applyLodVisibility(entry: LodEntry, scale: number) {
  const visibility = entry.minScale <= scale && scale < entry.maxScale ? 'visible' : 'hidden';
  if (entry.element.style.visibility !== visibility) entry.element.style.visibility = visibility;
}

function firstBoundaryAbove(boundaries: LodIndex['boundaries'], scale: number) {
  let low = 0;
  let high = boundaries.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (boundaries[middle].scale <= scale) low = middle + 1;
    else high = middle;
  }
  return low;
}

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
  resetKey?: string | number;
}

interface ZoomCamera {
  scale: number;
  x: number;
  y: number;
}

function clampCamera(camera: ZoomCamera, scroller: HTMLDivElement, contentWidth: number, contentHeight: number): ZoomCamera {
  const viewWidth = scroller.clientWidth || contentWidth;
  const viewHeight = scroller.clientHeight || contentHeight;
  const minX = Math.min(0, viewWidth - contentWidth * camera.scale);
  const minY = Math.min(0, viewHeight - contentHeight * camera.scale);
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
  const screenScaleX = scroller.clientWidth > 0 ? bounds.width / scroller.clientWidth : 1;
  const screenScaleY = scroller.clientHeight > 0 ? bounds.height / scroller.clientHeight : 1;
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
  onCategoryDrillDown,
  onNavigateBack,
  onZoomDelta,
  resetKey,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const surfaceRef = useRef<HTMLDivElement>(null);
  const lodRef = useRef<LodIndex>({ entries: [], boundaries: [] });
  const lodScaleRef = useRef<number | null>(null);
  const categoryElementsRef = useRef(new Map<string, HTMLElement>());
  const hoveredCategoryElementRef = useRef<HTMLElement | null>(null);
  const activeLeafRef = useRef<string | null>(null);
  const activeCategoryRef = useRef<string | null>(null);
  const zoomTargetRef = useRef(zoom);
  const previousZoomPropRef = useRef(zoom);
  const cameraRef = useRef<ZoomCamera>({ scale: zoom, x: 0, y: 0 });
  const zoomAnchorRef = useRef<{ x: number; y: number; contentX: number; contentY: number } | null>(null);
  const zoomFrameRef = useRef<number | null>(null);
  const lastZoomFrameTimeRef = useRef<number | null>(null);
  const zoomingRef = useRef(false);
  const lastPointerRef = useRef<{ x: number; y: number } | null>(null);
  const startZoomAnimationRef = useRef<() => void>(() => {});
  const wheelFrameRef = useRef<number | null>(null);
  const wheelDeltaRef = useRef(0);
  const wheelPointerRef = useRef({ x: 0, y: 0 });
  const dragRef = useRef<{
    pointerId: number; startX: number; startY: number; x: number; y: number; moved: boolean;
  } | null>(null);
  const suppressClickRef = useRef(false);
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
      const isSector = element.dataset.hmDepth === '1';
      element.style.border = isSector ? '1.5px solid #202432' : '1px solid #12151c';
      element.style.backgroundColor = '#020617';
      element.style.boxShadow = 'none';
    };
    if (previous) restore(previous);
    hoveredCategoryElementRef.current = next;
    if (next) {
      next.style.border = '2px solid #d99a5b';
      next.style.backgroundColor = '#d99a5b';
      next.style.boxShadow = '0 0 0 2px rgba(217,154,91,.22), inset 0 0 18px rgba(217,154,91,.08)';
    }
  };

  const updateTileVisibility = (scale: number) => {
    const { entries, boundaries } = lodRef.current;
    const previous = lodScaleRef.current;
    lodScaleRef.current = scale;
    if (previous === null) {
      for (const entry of entries) applyLodVisibility(entry, scale);
      return;
    }
    if (previous === scale) return;
    // Only elements with a [min, max) boundary in (low, high] can change.
    const low = Math.min(previous, scale);
    const high = Math.max(previous, scale);
    let index = firstBoundaryAbove(boundaries, low);
    while (index < boundaries.length && boundaries[index].scale <= high) {
      applyLodVisibility(boundaries[index].entry, scale);
      index += 1;
    }
  };

  const applyCamera = (next: ZoomCamera) => {
    const scroller = scrollRef.current;
    const surface = surfaceRef.current;
    if (!scroller || !surface) return;
    const { width: contentWidth, height: contentHeight } = dimensionsRef.current;
    const camera = clampCamera(next, scroller, contentWidth, contentHeight);
    cameraRef.current = camera;
    const isZoomed = camera.scale > MIN_ZOOM;
    const zoomedValue = String(isZoomed);
    if (scroller.dataset.zoomed !== zoomedValue) scroller.dataset.zoomed = zoomedValue;
    const touchAction = isZoomed ? 'none' : 'auto';
    if (scroller.style.touchAction !== touchAction) scroller.style.touchAction = touchAction;
    scroller.scrollLeft = Math.round(-camera.x);
    scroller.scrollTop = Math.round(-camera.y);
    surface.style.transform = `matrix(${camera.scale}, 0, 0, ${camera.scale}, ${camera.x}, ${camera.y})`;
    // Reveal labels at the exact zoom scale where they fit. The element list
    // is cached after layout, so animation frames do not query the DOM or
    // trigger a React render/re-layout.
    updateTileVisibility(camera.scale);
  };

  const setZoomedTarget = (target: number) => {
    zoomTargetRef.current = target;
  };

  const startZoomAnimation = () => {
    if (zoomFrameRef.current !== null) return;
    zoomingRef.current = true;
    const finishZoom = () => {
      zoomAnchorRef.current = null;
      zoomFrameRef.current = null;
      lastZoomFrameTimeRef.current = null;
      zoomingRef.current = false;
      updateTileVisibility(zoomTargetRef.current);

      // A stationary pointer can move over new cells as the map is transformed.
      // Defer hover work until the camera settles, then resolve the final cell once.
      const pointer = lastPointerRef.current;
      const scroller = scrollRef.current;
      if (pointer && scroller) {
        const targetElement = document.elementFromPoint(pointer.x, pointer.y);
        if (
          targetElement
          && scroller.contains(targetElement)
          && targetElement.closest('[data-hm-leaf-id],[data-hm-category-id]')
        ) {
          activeLeafRef.current = null;
          activeCategoryRef.current = null;
          targetElement.dispatchEvent(new PointerEvent('pointerover', {
            bubbles: true,
            clientX: pointer.x,
            clientY: pointer.y,
            pointerId: 1,
            pointerType: 'mouse',
            isPrimary: true,
          }));
          return;
        }
      }
      setCategoryHoverVisual(null);
      activeLeafRef.current = null;
      activeCategoryRef.current = null;
      onLeafHover?.(null);
      onCategoryHover(null);
    };
    const applyTarget = () => {
      const target = zoomTargetRef.current;
      const current = cameraRef.current;
      const anchor = zoomAnchorRef.current;
      applyCamera({
        scale: target,
        x: anchor ? anchor.x - anchor.contentX * target : current.x,
        y: anchor ? anchor.y - anchor.contentY * target : current.y,
      });
      finishZoom();
    };

    if (typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
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
        finishZoom();
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
    const surface = surfaceRef.current;
    if (!surface) return;
    categoryElementsRef.current = new Map(
      Array.from(surface.querySelectorAll<HTMLElement>('[data-hm-category-id]'))
        .map((element) => [element.dataset.hmCategoryId || '', element] as const)
        .filter(([id]) => !!id),
    );
    hoveredCategoryElementRef.current = activeCategoryRef.current
      ? categoryElementsRef.current.get(activeCategoryRef.current) || null
      : null;
    // Newly rendered tiles carry visibility for scale 1; a null previous scale
    // forces a full pass so a zoomed camera is reconciled immediately.
    lodRef.current = buildLodIndex(surface);
    lodScaleRef.current = null;
    applyCamera(cameraRef.current);
  }, [height, leafEntries, parents, width]);

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
    zoomingRef.current = false;
  }, []);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      lastPointerRef.current = { x: event.clientX, y: event.clientY };
      wheelPointerRef.current = localPoint(element, event.clientX, event.clientY);
      wheelDeltaRef.current += event.deltaY;
      if (wheelFrameRef.current !== null) return;
      wheelFrameRef.current = requestAnimationFrame(() => {
        wheelFrameRef.current = null;
        const { x, y } = wheelPointerRef.current;
        const zoomExponent = Math.max(-0.34, Math.min(0.34, -wheelDeltaRef.current * 0.0021));
        wheelDeltaRef.current = 0;
        if (Math.abs(zoomExponent) < 0.01) return;

        const currentTarget = zoomTargetRef.current;
        const nextZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, +(currentTarget * Math.exp(zoomExponent)).toFixed(2)));
        if (nextZoom === currentTarget) {
          if (zoomExponent < 0 && currentTarget <= MIN_ZOOM) onNavigateBack?.();
          return;
        }
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
  }, [onNavigateBack, onZoomDelta]);

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
    setCategoryHoverVisual(id);
    if (x != null && y != null) onCategoryPointerMove?.(x, y);
    if (activeCategoryRef.current === id && !forceAnchorUpdate) return;
    activeCategoryRef.current = id;
    const anchor = anchorFor(id, x != null && y != null ? { x, y } : undefined, rectOverride);
    if (anchor) onCategoryHover(anchor);
  };

  const onPointerOver = (event: React.PointerEvent<HTMLDivElement>) => {
    lastPointerRef.current = { x: event.clientX, y: event.clientY };
    if (zoomingRef.current) return;
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
    lastPointerRef.current = { x: event.clientX, y: event.clientY };
    const drag = dragRef.current;
    const element = scrollRef.current;
    if (!drag && zoomingRef.current) return;
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
        const bounds = element.getBoundingClientRect();
        const scaleX = element.clientWidth > 0 ? element.clientWidth / Math.max(bounds.width, 1) : 1;
        const scaleY = element.clientHeight > 0 ? element.clientHeight / Math.max(bounds.height, 1) : 1;
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
    if (zoomingRef.current) return;
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
    delete event.currentTarget.dataset.panning;
  };

  return (
      <div
        ref={scrollRef}
      onPointerOver={onPointerOver}
      onPointerMove={onPointerMove}
      onPointerOut={onPointerOut}
      onPointerDown={(event) => {
        if (zoomTargetRef.current <= MIN_ZOOM || event.button !== 0 || !scrollRef.current) return;
        if (zoomFrameRef.current !== null) {
          window.cancelAnimationFrame(zoomFrameRef.current);
          zoomFrameRef.current = null;
          lastZoomFrameTimeRef.current = null;
          zoomAnchorRef.current = null;
          zoomTargetRef.current = cameraRef.current.scale;
          zoomingRef.current = false;
          updateTileVisibility(cameraRef.current.scale);
        }
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
          setCategoryHoverVisual(null);
          onLeafHover?.(null);
          onCategoryHover(null);
        }
      }}
      aria-label="Interactive market heatmap. Use Tab to inspect categories and instruments."
      onDragStart={(event) => event.preventDefault()}
      className="cm-heatmap-map custom-scrollbar relative min-w-0 max-w-full select-none bg-slate-950 outline-none"
      style={{ width: '100%', height, overflow: 'hidden', userSelect: 'none', WebkitUserSelect: 'none' }}
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
          <CategoryTiles parents={parents} hoveredCategoryId={hoveredCategoryId} />
          <LeafTiles leafEntries={leafEntries} />
        </div>
      </div>
    </div>
  );
});

export default HeatmapTreemap;
