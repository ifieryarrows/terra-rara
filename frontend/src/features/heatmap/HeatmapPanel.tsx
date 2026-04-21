import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import HeatmapFilters from './HeatmapFilters';
import HeatmapTreemap from './HeatmapTreemap';
import HeatmapCategoryPanel from './HeatmapCategoryPanel';
import { HeatmapNode, leavesForCategory } from './heatmap-layout';
import { useMarketHeatmap } from '../../hooks/useQueries';

const MIN_ZOOM = 1;
const MAX_ZOOM = 2.5;
const ZOOM_STEP = 0.25;

export const HeatmapPanel: React.FC = () => {
  const { data: rawData, isError, error, isLoading, refetch, isFetching } = useMarketHeatmap();

  const [groupFilter, setGroupFilter] = useState('ALL');
  const [sortFilter, setSortFilter] = useState('Weight');
  const [zoom, setZoom] = useState(1);
  const [hoveredCategory, setHoveredCategory] = useState<string | null>(null);
  const [pinnedCategory, setPinnedCategory] = useState<string | null>(null);

  // Container sizing via ResizeObserver (works at first mount, not only
  // when DevTools is opened). Falls back to window resize for older
  // browsers.
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 600 });
  const [isFullscreen, setIsFullscreen] = useState(false);

  const measure = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const nextWidth = Math.max(0, Math.floor(rect.width));
    const nextHeight = isFullscreen
      ? Math.max(0, Math.floor(window.innerHeight - 120))
      : 600;
    setDimensions((prev) =>
      prev.width === nextWidth && prev.height === nextHeight
        ? prev
        : { width: nextWidth, height: nextHeight },
    );
  }, [isFullscreen]);

  useEffect(() => {
    measure();
    if (typeof ResizeObserver === 'undefined') {
      const handle = () => measure();
      window.addEventListener('resize', handle);
      return () => window.removeEventListener('resize', handle);
    }
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    window.addEventListener('resize', measure);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [measure]);

  // Re-measure once data arrives (container becomes visible).
  useEffect(() => {
    const id = requestAnimationFrame(() => measure());
    return () => cancelAnimationFrame(id);
  }, [rawData, isFullscreen, measure]);

  const availableGroups = useMemo<string[]>(() => {
    if (!rawData?.children) return [];
    const groups = rawData.children.map((g: any) => g.name).filter(Boolean);
    return Array.from(new Set<string>(groups)).sort();
  }, [rawData]);

  const filteredData = useMemo<HeatmapNode | null>(() => {
    if (!rawData) return null;

    const { _meta: _stripped, ...rest } = rawData as any;
    const clone: HeatmapNode = JSON.parse(JSON.stringify(rest));

    if (groupFilter !== 'ALL' && clone.children) {
      clone.children = clone.children.filter((g: any) => g.name === groupFilter);
    }

    if (sortFilter === 'Performance') {
      const remap = (node: any) => {
        if (node.children) {
          node.children.forEach(remap);
        } else {
          node.weight = Math.abs(node.changePercent || 0.01) * 1000;
          node.weightLabel = 'Performance';
        }
      };
      remap(clone);
    }

    return clone;
  }, [rawData, groupFilter, sortFilter]);

  const meta = (rawData as any)?._meta || {};
  const payloadCount: number | undefined = meta?.payload_count;
  const refreshError: string | null | undefined = meta?.refresh_error;

  const hasContent =
    !!filteredData?.children && filteredData.children.length > 0 && dimensions.width > 0;

  const activeCategory = pinnedCategory ?? hoveredCategory;
  const inspectorLeaves = useMemo(() => {
    if (!activeCategory || !filteredData) return [];
    return leavesForCategory(filteredData, activeCategory);
  }, [activeCategory, filteredData]);

  const handleZoomIn = () => setZoom((z) => Math.min(MAX_ZOOM, +(z + ZOOM_STEP).toFixed(2)));
  const handleZoomOut = () => setZoom((z) => Math.max(MIN_ZOOM, +(z - ZOOM_STEP).toFixed(2)));
  const handleZoomReset = () => setZoom(1);

  const handleManualRefresh = () => {
    refetch();
  };

  return (
    <div
      className={`flex flex-col bg-[#0f172a] overflow-hidden font-sans ${
        isFullscreen
          ? 'fixed inset-0 z-50'
          : 'relative rounded-lg shadow-xl border border-slate-700'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-900 border-b border-slate-700">
        <div>
          <h2 className="text-base font-semibold text-white tracking-wide leading-tight">
            CopperMind Universe Map
          </h2>
          <p className="text-[10px] text-slate-500 leading-none mt-0.5">
            {availableGroups.length} groups
            {typeof payloadCount === 'number' ? ` · ${payloadCount} symbols` : ''} · project universe only
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Zoom controls */}
          <div className="flex items-center gap-1 bg-slate-800 border border-slate-600 rounded">
            <button
              onClick={handleZoomOut}
              disabled={zoom <= MIN_ZOOM}
              title="Zoom out"
              className="px-2 py-1 text-slate-300 hover:text-white disabled:opacity-30 text-sm"
            >
              −
            </button>
            <button
              onClick={handleZoomReset}
              title="Reset zoom"
              className="px-2 py-1 text-slate-400 hover:text-white text-[10px] font-mono"
            >
              {Math.round(zoom * 100)}%
            </button>
            <button
              onClick={handleZoomIn}
              disabled={zoom >= MAX_ZOOM}
              title="Zoom in"
              className="px-2 py-1 text-slate-300 hover:text-white disabled:opacity-30 text-sm"
            >
              +
            </button>
          </div>

          <button
            onClick={handleManualRefresh}
            disabled={isFetching}
            title="Refresh now"
            className="text-slate-400 hover:text-white px-2 py-1 bg-slate-800 rounded border border-slate-600 transition-colors text-xs disabled:opacity-50"
          >
            {isFetching ? 'Refreshing…' : 'Refresh'}
          </button>
          <button
            onClick={() => setIsFullscreen((f) => !f)}
            className="text-slate-400 hover:text-white px-2 py-1 bg-slate-800 rounded border border-slate-600 transition-colors text-xs"
          >
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </button>
        </div>
      </div>

      <HeatmapFilters
        groupFilter={groupFilter}
        setGroupFilter={setGroupFilter}
        sortFilter={sortFilter}
        setSortFilter={setSortFilter}
        availableGroups={availableGroups}
        meta={meta}
      />

      {refreshError && (
        <div className="px-4 py-2 bg-rose-900/40 border-b border-rose-800 text-xs text-rose-200">
          Last refresh failed: {refreshError}
        </div>
      )}

      {/* Body — always render the measured container so ResizeObserver can fire
          on first mount even before data arrives. */}
      <div
        ref={containerRef}
        className="relative flex-1"
        style={{
          height: isFullscreen ? 'calc(100vh - 120px)' : '600px',
          minHeight: 400,
        }}
      >
        {isError ? (
          <div className="absolute inset-0 flex items-center justify-center text-red-400 text-sm">
            Error loading universe data: {(error as Error)?.message}
          </div>
        ) : !hasContent ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-2 text-sm">
            {isLoading || meta?.refresh_in_progress ? (
              <>
                <div className="w-6 h-6 border-2 border-slate-600 border-t-amber-400 rounded-full animate-spin" />
                <span>Loading CopperMind universe…</span>
              </>
            ) : rawData ? (
              'No symbols match the selected group filter.'
            ) : (
              'Waiting for universe data…'
            )}
          </div>
        ) : (
          <>
            <HeatmapTreemap
              data={filteredData!}
              width={activeCategory ? Math.max(200, dimensions.width - 280) : dimensions.width}
              height={dimensions.height}
              zoom={zoom}
              hoveredCategory={activeCategory}
              onCategoryHover={(name) => {
                if (!pinnedCategory) setHoveredCategory(name);
              }}
              onCategoryClick={(name) => {
                setPinnedCategory((prev) => (prev === name ? null : name));
                setHoveredCategory(name);
              }}
            />
            {activeCategory && (
              <HeatmapCategoryPanel
                categoryName={activeCategory}
                leaves={inspectorLeaves}
                onClose={() => {
                  setPinnedCategory(null);
                  setHoveredCategory(null);
                }}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default HeatmapPanel;
