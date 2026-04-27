import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import HeatmapFilters from './HeatmapFilters';
import HeatmapTreemap, { CategoryAnchor } from './HeatmapTreemap';
import HeatmapCategoryPanel from './HeatmapCategoryPanel';
import { HeatmapNode, leavesForCategory } from './heatmap-layout';
import { useMarketHeatmap } from '../../hooks/useQueries';

const MIN_ZOOM = 1;
const MAX_ZOOM = 2.5;
// Minimum gap between two auto-refresh invocations. Protects the
// backend + yfinance quota even if the server's next_refresh_at cycles
// quickly (e.g., after a failed refresh).
const AUTO_REFRESH_COOLDOWN_MS = 30_000;
// Debounce for category hover — below this we assume the pointer is
// transiting between cells and should NOT flash the inspector panel.
const CATEGORY_HOVER_DEBOUNCE_MS = 150;

export const HeatmapPanel: React.FC = () => {
  const { data: rawData, isError, error, isLoading, refetch, isFetching } = useMarketHeatmap();

  const [groupFilter, setGroupFilter] = useState('ALL');
  const [sortFilter, setSortFilter] = useState('Weight');
  const [zoom, setZoom] = useState(1);
  const [hoveredAnchor, setHoveredAnchor] = useState<CategoryAnchor | null>(null);
  const [pinnedAnchor, setPinnedAnchor] = useState<CategoryAnchor | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 600 });
  const [containerRect, setContainerRect] = useState<DOMRect | null>(null);
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
    setContainerRect(rect);
  }, [isFullscreen]);

  useEffect(() => {
    measure();
    if (typeof ResizeObserver === 'undefined') {
      const handle = () => measure();
      window.addEventListener('resize', handle);
      window.addEventListener('scroll', handle, { passive: true });
      return () => {
        window.removeEventListener('resize', handle);
        window.removeEventListener('scroll', handle);
      };
    }
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    window.addEventListener('resize', measure);
    window.addEventListener('scroll', measure, { passive: true });
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', measure);
      window.removeEventListener('scroll', measure);
    };
  }, [measure]);

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

  // Debounced hover: transit between cells <150ms is ignored so we
  // don't flicker the inspector between subcategories. Pin (click)
  // behavior still applies synchronously.
  const hoverTimeout = useRef<number | null>(null);
  const scheduleHover = useCallback((next: CategoryAnchor | null) => {
    if (hoverTimeout.current !== null) {
      window.clearTimeout(hoverTimeout.current);
      hoverTimeout.current = null;
    }
    if (next === null) {
      hoverTimeout.current = window.setTimeout(() => {
        setHoveredAnchor(null);
      }, CATEGORY_HOVER_DEBOUNCE_MS);
    } else {
      hoverTimeout.current = window.setTimeout(() => {
        setHoveredAnchor(next);
      }, CATEGORY_HOVER_DEBOUNCE_MS);
    }
  }, []);

  useEffect(() => () => {
    if (hoverTimeout.current !== null) window.clearTimeout(hoverTimeout.current);
  }, []);

  const activeAnchor = pinnedAnchor ?? hoveredAnchor;
  const activeCategory = activeAnchor?.name ?? null;
  const inspectorLeaves = useMemo(() => {
    if (!activeCategory || !filteredData) return [];
    return leavesForCategory(filteredData, activeCategory);
  }, [activeCategory, filteredData]);

  // Scroll-to-zoom — inside the heatmap only. Clamp to [MIN_ZOOM, MAX_ZOOM]
  // and round to one decimal to keep layout recalculations stable.
  const handleZoomDelta = useCallback((delta: number) => {
    setZoom((z) => {
      const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, +(z + delta).toFixed(2)));
      return next;
    });
  }, []);

  // Auto-refresh when the countdown hits 00:00 — cooldown-guarded.
  const lastAutoRefreshAt = useRef<number>(0);
  const handleCountdownElapsed = useCallback(() => {
    if (isFetching) return;
    const now = Date.now();
    if (now - lastAutoRefreshAt.current < AUTO_REFRESH_COOLDOWN_MS) return;
    lastAutoRefreshAt.current = now;
    refetch();
  }, [isFetching, refetch]);

  return (
    <div
      className={`flex flex-col bg-[#0f172a] overflow-hidden font-sans ${
        isFullscreen
          ? 'fixed inset-0 z-50'
          : 'relative rounded-lg shadow-xl border border-slate-700'
      }`}
    >
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
          {/* Manual Refresh button intentionally removed: countdown-driven
              auto-refresh covers the whole loop and prevents users from
              spamming yfinance. Scroll wheel over the heatmap handles zoom. */}
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
        onCountdownElapsed={handleCountdownElapsed}
      />

      {refreshError && (
        <div className="px-4 py-2 bg-rose-900/40 border-b border-rose-800 text-xs text-rose-200">
          Last refresh failed: {refreshError}
        </div>
      )}

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
            {/* Treemap keeps its full width — the inspector floats on top
                instead of shrinking the layout. */}
            <HeatmapTreemap
              data={filteredData!}
              width={dimensions.width}
              height={dimensions.height}
              zoom={zoom}
              hoveredCategory={activeCategory}
              onCategoryHover={(anchor) => {
                if (pinnedAnchor) return;
                scheduleHover(anchor);
              }}
              onCategoryClick={(anchor) => {
                setPinnedAnchor((prev) =>
                  prev && prev.name === anchor.name ? null : anchor,
                );
                setHoveredAnchor(anchor);
              }}
              onZoomDelta={handleZoomDelta}
            />
            {activeAnchor && (
              <HeatmapCategoryPanel
                categoryName={activeAnchor.name}
                leaves={inspectorLeaves}
                anchor={activeAnchor.rect}
                containerRect={containerRect}
                onClose={() => {
                  setPinnedAnchor(null);
                  setHoveredAnchor(null);
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
