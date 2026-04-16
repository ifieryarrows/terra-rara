import React, { useState, useEffect, useRef, useMemo } from 'react';
import HeatmapFilters from './HeatmapFilters';
import HeatmapTreemap from './HeatmapTreemap';
import { HeatmapNode } from './heatmap-layout';
import { useMarketHeatmap } from '../../hooks/useQueries';

export const HeatmapPanel: React.FC = () => {
  // Use the shared React Query hook (15-min polling, shared cache)
  const { data: rawData, isError, error } = useMarketHeatmap();

  // Filter state — project-universe axes only
  const [groupFilter, setGroupFilter] = useState('ALL');
  const [sortFilter, setSortFilter] = useState('Weight');

  // Dimensions
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 600 });
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: Math.max(containerRef.current.clientWidth, 800),
          height: isFullscreen ? window.innerHeight - 80 : 600,
        });
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isFullscreen]);

  // Extract list of available groups from the live payload
  const availableGroups = useMemo<string[]>(() => {
    if (!rawData?.children) return [];
    const groups = rawData.children.map((g: any) => g.name).filter(Boolean);
    return Array.from(new Set<string>(groups)).sort();
  }, [rawData]);

  // Filtered + sorted treemap data
  const filteredData = useMemo<HeatmapNode | null>(() => {
    if (!rawData) return null;

    // Strip _meta from the hierarchy clone so it doesn't confuse d3
    const { _meta: _stripped, ...rest } = rawData as any;
    const clone: HeatmapNode = JSON.parse(JSON.stringify(rest));

    // Apply group filter — filter at the top (group) level
    if (groupFilter !== 'ALL' && clone.children) {
      clone.children = clone.children.filter((g: any) => g.name === groupFilter);
    }

    // Apply sort mode: remap leaf weight for performance-based sizing
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

  const hasContent =
    filteredData?.children && filteredData.children.length > 0;

  return (
    <div
      className={`flex flex-col bg-[#0f172a] overflow-hidden ${
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
            {availableGroups.length} groups · project universe only
          </p>
        </div>
        <button
          onClick={() => setIsFullscreen((f) => !f)}
          className="text-slate-400 hover:text-white px-2 py-1 bg-slate-800 rounded border border-slate-600 transition-colors text-xs"
        >
          {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
        </button>
      </div>

      <HeatmapFilters
        groupFilter={groupFilter}
        setGroupFilter={setGroupFilter}
        sortFilter={sortFilter}
        setSortFilter={setSortFilter}
        availableGroups={availableGroups}
        meta={meta}
      />

      {isError ? (
        <div className="flex-1 flex items-center justify-center text-red-400 bg-[#0f172a] h-[600px] text-sm">
          Error loading universe data: {(error as Error)?.message}
        </div>
      ) : !hasContent ? (
        <div className="flex-1 flex flex-col items-center justify-center text-slate-500 bg-[#0f172a] h-[600px] gap-2 text-sm">
          {rawData
            ? 'No symbols match the selected group filter.'
            : 'Loading CopperMind universe…'}
        </div>
      ) : (
        <div
          ref={containerRef}
          className="flex-1 overflow-x-auto overflow-y-hidden"
          style={{ height: isFullscreen ? 'calc(100vh - 100px)' : '600px' }}
        >
          <HeatmapTreemap
            data={filteredData!}
            width={dimensions.width}
            height={dimensions.height}
          />
        </div>
      )}
    </div>
  );
};

export default HeatmapPanel;
