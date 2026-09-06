import React, { Profiler, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import HeatmapFilters from './HeatmapFilters';
import HeatmapTreemap, { type CategoryAnchor } from './HeatmapTreemap';
import HeatmapCategoryPanel, { type HeatmapCategoryPanelHandle } from './HeatmapCategoryPanel';
import {
  aggregateTinyLeaves,
  compressLeafWeights,
  leavesForCategory,
  type HeatmapData,
  type HeatmapMeta,
  type HeatmapNode,
} from './heatmap-layout';
import { recordCommit, recordLongTask } from './performance';
import { useMarketHeatmap } from '../../hooks/useQueries';
import { ViewState } from '../../components/ui/ViewState';
import { RefreshButton } from '../../components/ui/RefreshButton';

const OPEN_DELAY_MS = 90;
const CLOSE_DELAY_MS = 180;
const MIN_ZOOM = 1;
const MAX_ZOOM = 4;

function transformTree(
  raw: HeatmapNode,
  groupFilter: string,
  sortFilter: 'Weight' | 'Performance',
): HeatmapNode {
  const transform = (node: HeatmapNode | HeatmapData): HeatmapNode | HeatmapData => {
    if ('children' in node && node.children) {
      return { ...node, children: node.children.map(transform) } as HeatmapNode;
    }
    const leaf = node as HeatmapData;
    return sortFilter === 'Performance'
      ? { ...leaf, weight: Math.max(0.01, Math.abs(leaf.changePercent || 0.01)) * 1_000, weightLabel: 'Performance' }
      : { ...leaf };
  };
  const transformed = transform(raw) as HeatmapNode;
  if (groupFilter !== 'ALL') {
    transformed.children = (transformed.children || []).filter((group) => group.name === groupFilter);
  }
  return sortFilter === 'Weight' ? compressLeafWeights(transformed, 0.1) : transformed;
}

export const HeatmapPanel: React.FC = () => {
  const [view, setView] = useState<'market' | 'themes'>('market');
  const { data: rawData, isError, isLoading, refetch, isFetching } = useMarketHeatmap(view);
  const [groupFilter, setGroupFilter] = useState('ALL');
  const [sortFilter, setSortFilter] = useState<'Weight' | 'Performance'>('Weight');
  const [zoom, setZoom] = useState(1);
  const [highlightedCategoryId, setHighlightedCategoryId] = useState<string | null>(null);
  const [hoveredAnchor, setHoveredAnchor] = useState<CategoryAnchor | null>(null);
  const [hoveredLeaf, setHoveredLeaf] = useState<HeatmapData | null>(null);
  const [pinnedAnchor, setPinnedAnchor] = useState<CategoryAnchor | null>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 560 });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLElement>(null);
  const fullscreenButtonRef = useRef<HTMLButtonElement>(null);
  const resizeFrame = useRef<number | null>(null);
  const openTimer = useRef<number | null>(null);
  const closeTimer = useRef<number | null>(null);
  const categoryPanelRef = useRef<HeatmapCategoryPanelHandle>(null);
  const latestPointer = useRef<{ x: number; y: number } | null>(null);
  // Portal exit mounts a new inline button; resolve the current node at cleanup.
  const restoreFullscreenFocus = useCallback(() => fullscreenButtonRef.current?.focus({ preventScroll: true }), []);

  useEffect(() => {
    if (!isFullscreen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    fullscreenButtonRef.current?.focus();
    const containFocus = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return;
      const controls = [...(panelRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), select, a[href], [tabindex="0"]') ?? [])].filter(node => node.getClientRects().length);
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (!first) { event.preventDefault(); return; }
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    window.addEventListener('keydown', containFocus);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', containFocus);
      restoreFullscreenFocus();
    };
  }, [isFullscreen, restoreFullscreenFocus]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const update = (width: number, height: number) => {
      if (resizeFrame.current !== null) cancelAnimationFrame(resizeFrame.current);
      resizeFrame.current = requestAnimationFrame(() => {
        resizeFrame.current = null;
        const next = { width: Math.max(0, Math.floor(width)), height: Math.max(0, Math.floor(height)) };
        setDimensions((previous) => previous.width === next.width && previous.height === next.height ? previous : next);
      });
    };
    const bounds = element.getBoundingClientRect();
    update(bounds.width, bounds.height);
    if (typeof ResizeObserver === 'undefined') {
      const onResize = () => {
        const next = element.getBoundingClientRect();
        update(next.width, next.height);
      };
      window.addEventListener('resize', onResize);
      return () => window.removeEventListener('resize', onResize);
    }
    const observer = new ResizeObserver(([entry]) => update(entry.contentRect.width, entry.contentRect.height));
    observer.observe(element);
    return () => {
      observer.disconnect();
      if (resizeFrame.current !== null) cancelAnimationFrame(resizeFrame.current);
    };
  }, [isFullscreen]);

  useEffect(() => {
    if (typeof PerformanceObserver === 'undefined') return;
    try {
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => recordLongTask(entry.duration));
      });
      observer.observe({ entryTypes: ['longtask'] });
      return () => observer.disconnect();
    } catch {
      return undefined;
    }
  }, []);

  const clearTimer = (ref: React.MutableRefObject<number | null>) => {
    if (ref.current !== null) window.clearTimeout(ref.current);
    ref.current = null;
  };
  const cancelClose = useCallback(() => clearTimer(closeTimer), []);
  const scheduleClose = useCallback(() => {
    clearTimer(openTimer);
    clearTimer(closeTimer);
    closeTimer.current = window.setTimeout(() => {
      setHoveredAnchor(null);
      setHoveredLeaf(null);
    }, CLOSE_DELAY_MS);
  }, []);
  const handleCategoryHover = useCallback((anchor: CategoryAnchor | null) => {
    if (pinnedAnchor) return;
    if (!anchor) {
      // Finviz paints selection on a dedicated hover canvas. Mirror that
      // immediacy for the visual boundary while retaining the panel's close
      // grace period.
      setHighlightedCategoryId(null);
      scheduleClose();
      return;
    }
    setHighlightedCategoryId(anchor.id);
    if (anchor.pointer) latestPointer.current = anchor.pointer;
    const pointerDriven = !!anchor.pointer;
    clearTimer(closeTimer);
    clearTimer(openTimer);
    if (hoveredAnchor?.id === anchor.id) {
      setHoveredAnchor(pointerDriven && latestPointer.current ? { ...anchor, pointer: latestPointer.current } : anchor);
      return;
    }
    openTimer.current = window.setTimeout(() => {
      setHoveredAnchor(pointerDriven && latestPointer.current ? { ...anchor, pointer: latestPointer.current } : anchor);
    }, OPEN_DELAY_MS);
  }, [hoveredAnchor, pinnedAnchor, scheduleClose]);

  useEffect(() => () => {
    clearTimer(openTimer);
    clearTimer(closeTimer);
  }, []);

  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (pinnedAnchor || hoveredAnchor) {
        setPinnedAnchor(null);
        setHoveredAnchor(null);
        setHighlightedCategoryId(null);
      } else if (isFullscreen) {
        setIsFullscreen(false);
      }
    };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [hoveredAnchor, isFullscreen, pinnedAnchor]);

  useEffect(() => {
    setGroupFilter('ALL');
    setHighlightedCategoryId(null);
    setHoveredAnchor(null);
    setHoveredLeaf(null);
    setPinnedAnchor(null);
    setZoom(1);
  }, [view]);

  const meta = ((rawData as HeatmapNode | undefined)?._meta || {}) as HeatmapMeta;
  const sourceTree = useMemo<HeatmapNode | null>(() => {
    if (!rawData) return null;
    const { _meta: _meta, ...tree } = rawData as HeatmapNode;
    return transformTree(tree as HeatmapNode, groupFilter, sortFilter);
  }, [groupFilter, rawData, sortFilter]);
  const renderTree = useMemo(
    () => sourceTree ? aggregateTinyLeaves(sourceTree, dimensions.width, dimensions.height) : null,
    [dimensions.height, dimensions.width, sourceTree],
  );
  const groups = useMemo<string[]>(() => {
    const names = (rawData?.children || []).map((group: HeatmapNode) => String(group.name));
    return Array.from(new Set<string>(names)).sort();
  }, [rawData]);
  const activeAnchor = pinnedAnchor || hoveredAnchor;
  const panelLeaves = useMemo(
    () => activeAnchor && sourceTree ? leavesForCategory(sourceTree, activeAnchor.id, activeAnchor.name) : [],
    [activeAnchor, sourceTree],
  );
  const hasContent = !!renderTree?.children?.length && dimensions.width > 0;
  const zoomBy = useCallback((delta: number) => setZoom((current) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, +(current + delta).toFixed(2)))), []);
  const moveCategoryPanel = useCallback((x: number, y: number) => {
    latestPointer.current = { x, y };
    if (!pinnedAnchor) categoryPanelRef.current?.move(x, y);
  }, [pinnedAnchor]);
  const handleLeafHover = useCallback((leaf: HeatmapData | null) => {
    if (!pinnedAnchor) setHoveredLeaf(leaf);
  }, [pinnedAnchor]);

  const content = (
    <section ref={panelRef} role={isFullscreen ? 'dialog' : undefined} aria-modal={isFullscreen || undefined} aria-label={isFullscreen ? 'Fullscreen market map' : 'Market map'} className={`flex min-w-0 max-w-full flex-col bg-slate-950 font-sans ${isFullscreen ? 'cm-map-fullscreen fixed inset-0 z-50' : 'relative w-full overflow-hidden rounded-xl border border-slate-700 shadow-xl'}`}>
      <header className="cm-heatmap-header">
        <div>
          <h2 className="text-base font-semibold tracking-wide text-white">Market Heatmap</h2>
          <p className="cm-chart-note">
            {groups.length} top-level groups · {meta.payload_count ?? 0} instruments · sector → industry → instrument
          </p>
        </div>
        <div className="cm-heatmap-actions" role="group" aria-label="Map controls">
        <button type="button" className="cm-icon-button" aria-label="Zoom out" disabled={zoom <= MIN_ZOOM} onClick={() => zoomBy(-.5)}>−</button>
        <button type="button" className="cm-filter-chip" aria-label="Reset map zoom" onClick={() => setZoom(1)}>{Math.round(zoom * 100)}%</button>
        <button type="button" className="cm-icon-button" aria-label="Zoom in" disabled={zoom >= MAX_ZOOM} onClick={() => zoomBy(.5)}>+</button>
        <button ref={fullscreenButtonRef} type="button" onClick={() => setIsFullscreen((current) => !current)} className="cm-filter-chip" aria-pressed={isFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
        </button>
        </div>
      </header>

      <HeatmapFilters
        groupFilter={groupFilter}
        setGroupFilter={setGroupFilter}
        sortFilter={sortFilter}
        setSortFilter={setSortFilter}
        view={view}
        setView={setView}
        availableGroups={groups}
        meta={meta}
      />
      {meta.refresh_error && <div className="border-b border-rose-800 bg-rose-950/60 px-4 py-2 text-xs text-rose-200">Last refresh failed; showing the last healthy snapshot. {meta.refresh_error}</div>}
      {isError && hasContent && <p className="cm-news-updating" role="status">Map refresh failed. The previous snapshot remains visible.</p>}

      <div
        ref={containerRef}
        className="relative min-w-0 flex-1"
        style={{ height: isFullscreen ? undefined : 'clamp(560px, 72vh, 820px)', minHeight: isFullscreen ? 280 : 560 }}
      >
        {isError && !hasContent ? (
          <ViewState kind="error" title="Market map could not be loaded" description="Check again to retrieve an available snapshot." action={<RefreshButton onClick={() => refetch()} busy={isFetching} label="Retry market map"/>} compact/>
        ) : !hasContent ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-sm text-slate-500">
            {(isLoading || meta.refresh_in_progress) && <span className="h-6 w-6 animate-spin rounded-full border-2 border-slate-700 border-t-copper-400" />}
            <span>{isLoading || meta.refresh_in_progress ? 'Preparing the market snapshot…' : 'No instruments match this filter.'}</span>
          </div>
        ) : (
          <Profiler
            id="MarketHeatmap"
            onRender={(_id, phase, actualDuration) => recordCommit(phase, actualDuration)}
          >
            <HeatmapTreemap
              data={renderTree!}
              width={dimensions.width}
              height={dimensions.height}
              zoom={zoom}
              hoveredCategoryId={pinnedAnchor?.id || highlightedCategoryId}
              onCategoryHover={handleCategoryHover}
              onCategoryPointerMove={moveCategoryPanel}
              onLeafHover={handleLeafHover}
              onCategoryClick={(anchor) => {
                clearTimer(openTimer);
                clearTimer(closeTimer);
                setPinnedAnchor((current) => current?.id === anchor.id ? null : anchor);
                setHighlightedCategoryId(anchor.id);
                setHoveredAnchor(anchor);
              }}
              onZoomDelta={zoomBy}
            />
          </Profiler>
        )}
        {activeAnchor && (
          <HeatmapCategoryPanel
            ref={categoryPanelRef}
            categoryId={activeAnchor.id}
            categoryName={activeAnchor.name}
            leaves={panelLeaves}
            activeLeaf={hoveredLeaf}
            anchor={activeAnchor}
            view={view}
            pinned={!!pinnedAnchor}
            onPointerEnter={cancelClose}
            onPointerLeave={() => { if (!pinnedAnchor) scheduleClose(); }}
            onClose={() => { setPinnedAnchor(null); setHighlightedCategoryId(null); setHoveredAnchor(null); setHoveredLeaf(null); }}
          />
        )}
      </div>
      <footer className="cm-heatmap-footer">
        <span>Mouse wheel zooms · Drag zoomed map to pan · Double-click a ticker for details · Enter pins · Esc closes</span>
        <a href="https://www.logo.dev" target="_blank" rel="noopener" className="text-slate-500 hover:text-copper-300">Logos provided by Logo.dev</a>
      </footer>
    </section>
  );
  return isFullscreen ? createPortal(content, document.body) : content;
};

export default HeatmapPanel;
