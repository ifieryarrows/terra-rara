import React, { Profiler, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import HeatmapFilters from './HeatmapFilters';
import HeatmapTreemap, { type CategoryAnchor } from './HeatmapTreemap';
import HeatmapCategoryPanel, { type HeatmapCategoryPanelHandle } from './HeatmapCategoryPanel';
import {
  aggregateTinyLeaves,
  compressLeafWeights,
  getNextZoomLevel,
  getPreviousZoomLevel,
  type HeatmapData,
  type HeatmapMeta,
  type HeatmapNode,
} from './heatmap-layout';
import { recordCommit, recordLongTask } from './performance';
import { useMarketHeatmap } from '../../hooks/useQueries';
import { ViewState } from '../../components/ui/ViewState';
import { RefreshButton } from '../../components/ui/RefreshButton';
import { Maximize2, Minimize2 } from 'lucide-react';

const OPEN_DELAY_MS = 90;
const CLOSE_DELAY_MS = 180;
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

function findMapCategory(node: HeatmapNode, id: string): HeatmapNode | null {
  if (!node.children?.length) return null;
  if (String(node.id || node.name) === id) return node;
  for (const child of node.children) {
    if (!('children' in child) || !child.children?.length) continue;
    const match = findMapCategory(child as HeatmapNode, id);
    if (match) return match;
  }
  return null;
}

function indexCategoryLeaves(root: HeatmapNode): Map<string, HeatmapData[]> {
  const index = new Map<string, HeatmapData[]>();
  const visit = (node: HeatmapNode | HeatmapData, depth: number): HeatmapData[] => {
    const children = 'children' in node ? node.children : undefined;
    if (!children?.length) return [node as HeatmapData];
    const leaves = children.flatMap((child) => visit(child, depth + 1));
    const category = node as HeatmapNode;
    index.set(String(category.id || `${depth}-${category.name}`), leaves);
    if (category.id) index.set(String(category.id), leaves);
    if (!index.has(category.name)) index.set(category.name, leaves);
    return leaves;
  };
  visit(root, 0);
  return index;
}

interface HeatmapFocusEntry {
  id: string;
  name: string;
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
  const [focusedPath, setFocusedPath] = useState<HeatmapFocusEntry[]>([]);
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
    setHoveredAnchor(null);
    setHoveredLeaf(null);
    setPinnedAnchor(null);
    setHighlightedCategoryId(null);
    setFocusedPath([]);
    setZoom(1);
  }, [view]);

  useEffect(() => {
    setFocusedPath([]);
    setZoom(1);
  }, [groupFilter]);

  const meta = ((rawData as HeatmapNode | undefined)?._meta || {}) as HeatmapMeta;
  const sourceTree = useMemo<HeatmapNode | null>(() => {
    if (!rawData) return null;
    const { _meta: _meta, ...tree } = rawData as HeatmapNode;
    return transformTree(tree as HeatmapNode, groupFilter, sortFilter);
  }, [groupFilter, rawData, sortFilter]);
  const focusedTree = useMemo(() => {
    if (!sourceTree) return null;
    let current = sourceTree;
    for (const entry of focusedPath) {
      const next = findMapCategory(current, entry.id);
      if (!next) return null;
      current = next;
    }
    return current;
  }, [focusedPath, sourceTree]);
  const renderTree = useMemo(
    () => focusedTree ? aggregateTinyLeaves(focusedTree, dimensions.width, dimensions.height) : null,
    [dimensions.height, dimensions.width, focusedTree],
  );
  const groups = useMemo<string[]>(() => {
    const names = (rawData?.children || []).map((group: HeatmapNode) => String(group.name));
    return Array.from(new Set<string>(names)).sort();
  }, [rawData]);
  const categoryLeafIndex = useMemo(
    () => sourceTree ? indexCategoryLeaves(sourceTree) : new Map<string, HeatmapData[]>(),
    [sourceTree],
  );
  const activeAnchor = pinnedAnchor || hoveredAnchor;
  const panelLeaves = useMemo(
    () => activeAnchor
      ? categoryLeafIndex.get(activeAnchor.id) || categoryLeafIndex.get(activeAnchor.name) || []
      : [],
    [activeAnchor?.id, activeAnchor?.name, categoryLeafIndex],
  );
  const hasContent = !!renderTree?.children?.length && dimensions.width > 0;
  const zoomBy = useCallback((delta: number) => {
    setZoom((current) => delta > 0 ? getNextZoomLevel(current) : getPreviousZoomLevel(current));
  }, []);
  const moveCategoryPanel = useCallback((x: number, y: number) => {
    latestPointer.current = { x, y };
    if (!pinnedAnchor) categoryPanelRef.current?.move(x, y);
  }, [pinnedAnchor]);
  const handleLeafHover = useCallback((leaf: HeatmapData | null) => {
    if (!pinnedAnchor) setHoveredLeaf(leaf);
  }, [pinnedAnchor]);
  const handleCategoryDrillDown = useCallback((anchor: CategoryAnchor) => {
    setPinnedAnchor(null);
    setHoveredAnchor(null);
    setHoveredLeaf(null);
    setZoom(1);
    setFocusedPath((current) => {
      const existingIndex = current.findIndex((entry) => entry.id === anchor.id);
      if (existingIndex >= 0) return current.slice(0, existingIndex + 1);
      if (!sourceTree || !findMapCategory(renderTree || sourceTree, anchor.id)) return current;
      return [...current, { id: anchor.id, name: anchor.name }];
    });
  }, [renderTree, sourceTree]);
  const handleMapNavigateBack = useCallback(() => {
    setFocusedPath((current) => current.length ? current.slice(0, -1) : current);
    setPinnedAnchor(null);
    setHoveredAnchor(null);
    setHoveredLeaf(null);
    setZoom(1);
  }, []);
  const handleCategoryClick = useCallback((anchor: CategoryAnchor) => {
    clearTimer(openTimer);
    clearTimer(closeTimer);
    setPinnedAnchor((current) => current?.id === anchor.id ? null : anchor);
    setHighlightedCategoryId(anchor.id);
    setHoveredAnchor(anchor);
  }, []);

  const content = (
    <section ref={panelRef} role={isFullscreen ? 'dialog' : undefined} aria-modal={isFullscreen || undefined} aria-label={isFullscreen ? 'Fullscreen market map' : 'Market map'} className={`cm-heatmap cm-heatmap--terminal min-w-0 max-w-full bg-slate-950 font-sans ${isFullscreen ? 'cm-map-fullscreen fixed inset-0 z-50' : 'relative w-full overflow-hidden rounded-xl border border-slate-700 shadow-xl'}`} data-cm-route-reveal="surface">
      <aside className="cm-heatmap-sidebar" aria-label="Market map controls">
        <header className="cm-heatmap-sidebar-head">
          <div className="cm-heatmap-sidebar-title-row">
            <div><span>MARKET MAP</span><h2>Heatmap</h2></div>
            <button ref={fullscreenButtonRef} type="button" className="cm-heatmap-fullscreen-button" onClick={() => setIsFullscreen((current) => !current)} aria-pressed={isFullscreen} aria-label={isFullscreen ? 'Exit fullscreen' : 'Open fullscreen'} title={isFullscreen ? 'Exit fullscreen' : 'Open fullscreen'}>
              {isFullscreen ? <Minimize2 size={15} aria-hidden="true"/> : <Maximize2 size={15} aria-hidden="true"/>}
            </button>
          </div>
          <p>{rawData
            ? <>{groups.length} sectors <i aria-hidden="true">/</i> {meta.payload_count ?? 0} instruments</>
            : isLoading || meta.refresh_in_progress ? 'Loading snapshot…' : 'No available snapshot'}</p>
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
          hasSnapshot={!!rawData}
        />

        {meta.refresh_error && <p className="cm-heatmap-error">Snapshot refresh failed. Showing the last available data.</p>}
      </aside>

      <div className="cm-heatmap-main">
        {isError && hasContent && <p className="cm-news-updating" role="status">Map refresh failed. The previous snapshot remains visible.</p>}
        <div
          ref={containerRef}
          className="cm-heatmap-stage relative min-w-0 flex-1"
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
            <>
              {focusedPath.length > 0 && (
                <nav
                  aria-label="Heatmap drill-down path"
                  className="absolute left-3 top-3 z-20 flex max-w-[calc(100%-1.5rem)] items-center gap-1 overflow-x-auto rounded-md border border-slate-700/80 bg-slate-950/90 px-2 py-1 text-xs text-slate-300 shadow backdrop-blur"
                >
                  <button type="button" className="shrink-0 hover:text-white" onClick={() => { setFocusedPath([]); setZoom(1); }}>Market</button>
                  {focusedPath.map((entry, index) => (
                    <React.Fragment key={entry.id}>
                      <span aria-hidden="true" className="text-slate-600">/</span>
                      <button
                        type="button"
                        className="shrink-0 hover:text-white"
                        aria-current={index === focusedPath.length - 1 ? 'page' : undefined}
                        onClick={() => { setFocusedPath((current) => current.slice(0, index + 1)); setZoom(1); }}
                      >
                        {entry.name}
                      </button>
                    </React.Fragment>
                  ))}
                </nav>
              )}
              <Profiler
                id="MarketHeatmap"
                onRender={(_id, phase, actualDuration) => recordCommit(phase, actualDuration)}
              >
                <HeatmapTreemap
                  key={focusedPath.map((entry) => entry.id).join('/') || 'market-root'}
                  data={renderTree!}
                  width={dimensions.width}
                  height={dimensions.height}
                  zoom={zoom}
                  resetKey={view}
                  hoveredCategoryId={pinnedAnchor?.id || highlightedCategoryId}
                  onCategoryHover={handleCategoryHover}
                  onCategoryPointerMove={moveCategoryPanel}
                  onLeafHover={handleLeafHover}
                  onCategoryDrillDown={handleCategoryDrillDown}
                  onNavigateBack={focusedPath.length ? handleMapNavigateBack : undefined}
                  onCategoryClick={handleCategoryClick}
                  onZoomDelta={zoomBy}
                />
              </Profiler>
            </>
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
        <footer className="cm-heatmap-footer">
          <span>Wheel to zoom <i/> drag to pan <i/> double-click a category to drill in or a ticker to open its quote</span>
          <a href="https://www.logo.dev" target="_blank" rel="noopener">Logos by Logo.dev</a>
        </footer>
      </div>
    </section>
  );
  return isFullscreen ? createPortal(content, document.body) : content;
};

export default HeatmapPanel;
