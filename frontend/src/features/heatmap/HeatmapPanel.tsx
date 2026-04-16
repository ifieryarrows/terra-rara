import React, { useState, useEffect, useRef, useCallback } from 'react';
import HeatmapFilters from './HeatmapFilters';
import HeatmapTreemap from './HeatmapTreemap';
import { HeatmapNode } from './heatmap-layout';

// Mock hook or replace with real fetching logic (React Query planned in next step)
// For now, we will fetch directly with useEffect
const API_URL = import.meta.env.VITE_API_URL || '';

export const HeatmapPanel: React.FC = () => {
  const [data, setData] = useState<HeatmapNode | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Filter state
  const [universeFilter, setUniverseFilter] = useState('ALL');
  const [sectorFilter, setSectorFilter] = useState('ALL');
  const [sortFilter, setSortFilter] = useState('MarketCap');
  
  // Dimensions
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 600 });
  const [isFullscreen, setIsFullscreen] = useState(false);

  const fetchHeatmap = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/market-heatmap`);
      if (!res.ok) throw new Error('Failed to fetch heatmap');
      const json = await res.json();
      setData(json);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    fetchHeatmap();
    // Poll exactly every 15 minutes (900,000 ms) as specified by the plan
    const interval = setInterval(fetchHeatmap, 900000);
    return () => clearInterval(interval);
  }, [fetchHeatmap]);

  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: Math.max(containerRef.current.clientWidth, 800), // Enforce min-width
          height: isFullscreen ? window.innerHeight - 80 : 600
        });
      }
    };
    
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isFullscreen]);

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  const filteredData = React.useMemo(() => {
    if (!data) return null;
    
    // Deep clone to avoid mutating state
    const clone = JSON.parse(JSON.stringify(data));
    
    if (universeFilter === 'ALL' && sectorFilter === 'ALL' && sortFilter === 'MarketCap') {
      return clone;
    }

    // Filter logic
    if (clone.children) {
      // 1. Sector level
      clone.children = clone.children.filter((sec: any) => {
        if (sectorFilter !== 'ALL' && sec.name !== sectorFilter) return false;
        
        // 2. Industry level
        if (sec.children) {
          sec.children = sec.children.map((ind: any) => {
            // 3. Stock level
            if (ind.children) {
              ind.children = ind.children.filter((stock: any) => {
                if (universeFilter === 'SP500' && !stock.isSP500) return false;
                if (universeFilter === 'NASDAQ100' && !stock.isNasdaq100) return false;
                return true;
              });
            }
            return ind;
          }).filter((ind: any) => ind.children && ind.children.length > 0); // Remove empty industries
        }
        
        return sec.children && sec.children.length > 0; // Remove empty sectors
      });
    }

    // Sort Logic (if changing value instead of market cap, we can remap the values)
    // Though squarify primarily uses value derived from sum(). We can mutate marketCap locally.
    if (sortFilter === 'Performance') {
      const applyPerformanceWeight = (node: any) => {
        if (node.children) {
          node.children.forEach(applyPerformanceWeight);
        } else {
          // Weight by absolute performance instead of market cap
          node.marketCap = Math.abs(node.changePercent || 1) * 1000;
        }
      };
      applyPerformanceWeight(clone);
    }

    return clone;
  }, [data, universeFilter, sectorFilter, sortFilter]);

  const meta = data?._meta || {};

  return (
    <div className={`flex flex-col bg-[#0f172a] overflow-hidden ${isFullscreen ? 'fixed inset-0 z-50' : 'relative rounded-lg shadow-xl border border-slate-700'}`}>
      <div className="flex items-center justify-between px-4 py-3 bg-slate-900 border-b border-slate-700">
        <h2 className="text-lg font-semibold text-white tracking-wide">Market Heatmap</h2>
        <button 
          onClick={toggleFullscreen}
          className="text-slate-400 hover:text-white px-2 py-1 bg-slate-800 rounded border border-slate-600 transition-colors text-sm"
        >
          {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
        </button>
      </div>

      <HeatmapFilters 
        universeFilter={universeFilter}
        setUniverseFilter={setUniverseFilter}
        sectorFilter={sectorFilter}
        setSectorFilter={setSectorFilter}
        sortFilter={sortFilter}
        setSortFilter={setSortFilter}
        meta={meta}
      />

      {error ? (
        <div className="flex-1 flex items-center justify-center text-red-500 bg-[#0f172a] h-[600px]">
          Error loading heatmap: {error}
        </div>
      ) : !filteredData || !filteredData.children || filteredData.children.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-slate-400 bg-[#0f172a] h-[600px]">
          {data ? 'No symbols match the current filters' : 'Loading market data...'}
        </div>
      ) : (
        <div 
          ref={containerRef} 
          className="flex-1 overflow-x-auto overflow-y-hidden"
          style={{ height: isFullscreen ? 'calc(100vh - 100px)' : '600px' }}
        >
          <HeatmapTreemap 
            data={filteredData} 
            width={dimensions.width} 
            height={dimensions.height} 
          />
        </div>
      )}
    </div>
  );
};

export default HeatmapPanel;
