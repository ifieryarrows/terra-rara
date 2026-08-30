import React, { Profiler, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import HeatmapFilters from './HeatmapFilters';
import HeatmapTreemap, { type CategoryAnchor } from './HeatmapTreemap';
import HeatmapCategoryPanel, { type HeatmapCategoryPanelHandle } from './HeatmapCategoryPanel';
import {
  aggregateTinyLeaves,
  leavesForCategory,
  type HeatmapData,
  type HeatmapMeta,
  type HeatmapNode,
} from './heatmap-layout';
import { recordCommit, recordLongTask } from './performance';
import { useMarketHeatmap } from '../../hooks/useQueries';

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
  return transformed;
}

export const HeatmapPanel: React.FC = () => {
  const [view, setView] = useState<'market' | 'themes'>('market');
  const { data: rawData, isError, error, isLoading } = useMarketHeatmap(view);
  const [groupFilter, setGroupFilter] = useState('ALL');
  const [sortFilter, setSortFilter] = useState<'Weight' | 'Performance'>('Weight');
  const [zoom, setZoom] = useState(1);
  const [hoveredAnchor, setHoveredAnchor] = useState<CategoryAnchor | null>(null);
  const [hoveredLeaf, setHoveredLeaf] = useState<HeatmapData | null>(null);
  const [pinnedAnchor, setPinnedAnchor] = useState<CategoryAnchor | null>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 560 });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const resizeFrame = useRef<number | null>(null);
  const openTimer = useRef<number | null>(null);
  const closeTimer = useRef<number | null>(null);
  const categoryPanelRef = useRef<HeatmapCategoryPanelHandle>(null);

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
      scheduleClose();
      return;
    }
    clearTimer(closeTimer);
    clearTimer(openTimer);
    if (hoveredAnchor?.id === anchor.id) {
      setHoveredAnchor(anchor);
      return;
    }
    openTimer.current = window.setTimeout(() => setHoveredAnchor(anchor), OPEN_DELAY_MS);
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
      } else if (isFullscreen) {
        setIsFullscreen(false);
      }
    };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [hoveredAnchor, isFullscreen, pinnedAnchor]);

  useEffect(() => {
    setGroupFilter('ALL');
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
    if (!pinnedAnchor) categoryPanelRef.current?.move(x, y);
  }, [pinnedAnchor]);
  const handleLeafHover = useCallback((leaf: HeatmapData | null) => {
    if (!pinnedAnchor) setHoveredLeaf(leaf);
  }, [pinnedAnchor]);

  return (
    <section className={`flex min-w-0 max-w-full flex-col overflow-hidden bg-slate-950 font-sans ${isFullscreen ? 'fixed inset-0 z-50' : 'relative w-full rounded-xl border border-slate-700 shadow-xl'}`}>
      <header className="flex items-center justify-between gap-3 border-b border-slate-700 bg-slate-900 px-4 py-3">
        <div>
          <h2 className="text-base font-semibold tracking-wide text-white">Market Heatmap</h2>
          <p className="mt-0.5 text-[10px] text-slate-500">
            {groups.length} top-level groups · {meta.payload_count ?? 0} instruments · sector → industry → instrument
          </p>
        </div>
        <button type="button" onClick={() => setIsFullscreen((current) => !current)} className="rounded border border-slate-600 bg-slate-800 px-2 py-1 text-xs text-slate-300 hover:text-white">
          {isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
        </button>
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

      <div
        ref={containerRef}
        className="relative min-w-0 flex-1"
        style={{ height: isFullscreen ? 'calc(100vh - 112px)' : 'clamp(560px, 72vh, 820px)', minHeight: isFullscreen ? 400 : 560 }}
      >
        {isError ? (
          <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-sm text-rose-300">Heatmap data is temporarily unavailable: {(error as Error)?.message}</div>
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
              hoveredCategoryId={activeAnchor?.id || null}
              onCategoryHover={handleCategoryHover}
              onCategoryPointerMove={moveCategoryPanel}
              onLeafHover={handleLeafHover}
              onCategoryClick={(anchor) => {
                clearTimer(openTimer);
                clearTimer(closeTimer);
                setPinnedAnchor((current) => current?.id === anchor.id ? null : anchor);
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
            onClose={() => { setPinnedAnchor(null); setHoveredAnchor(null); setHoveredLeaf(null); }}
          />
        )}
      </div>
      <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-800 bg-slate-950 px-3 py-1.5 text-[9px] text-slate-600">
        <span>Mouse wheel zooms · Drag zoomed map to pan · Double-click a ticker for details · Enter pins · Esc closes</span>
        <a href="https://www.logo.dev" target="_blank" rel="noopener" className="text-slate-500 hover:text-copper-300">Logos provided by Logo.dev</a>
      </footer>
    </section>
  );
};

export default HeatmapPanel;
