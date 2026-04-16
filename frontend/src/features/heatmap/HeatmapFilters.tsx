import React, { useEffect, useState } from 'react';

interface HeatmapFiltersProps {
  universeFilter: string;
  setUniverseFilter: (val: string) => void;
  sectorFilter: string;
  setSectorFilter: (val: string) => void;
  sortFilter: string;
  setSortFilter: (val: string) => void;
  meta: any;
}

const HeatmapFilters: React.FC<HeatmapFiltersProps> = ({
  universeFilter, setUniverseFilter,
  sectorFilter, setSectorFilter,
  sortFilter, setSortFilter,
  meta
}) => {
  const [countdown, setCountdown] = useState<number>(15 * 60);

  useEffect(() => {
    if (!meta || !meta.next_refresh_at) return;
    
    const updateCountdown = () => {
      const nextRefresh = new Date(meta.next_refresh_at).getTime();
      const now = new Date().getTime();
      const diff = Math.max(0, Math.floor((nextRefresh - now) / 1000));
      setCountdown(diff);
    };

    updateCountdown();
    const interval = setInterval(updateCountdown, 1000);
    return () => clearInterval(interval);
  }, [meta]);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const isStale = meta?.is_stale;
  const isRefreshing = meta?.refresh_in_progress;

  return (
    <div className="flex flex-wrap items-center justify-between bg-[#0f172a] border-b border-slate-700 p-2 text-sm text-slate-300">
      {/* Filters */}
      <div className="flex flex-wrap gap-4">
        {/* Universe Filter */}
        <select 
          className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 outline-none"
          value={universeFilter} 
          onChange={(e) => setUniverseFilter(e.target.value)}
        >
          <option value="ALL">All Universe</option>
          <option value="SP500">S&P 500</option>
          <option value="NASDAQ100">Nasdaq 100</option>
        </select>

        {/* Sector Filter */}
        <select 
          className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 outline-none"
          value={sectorFilter} 
          onChange={(e) => setSectorFilter(e.target.value)}
        >
          <option value="ALL">All Sectors</option>
          <option value="Technology">Technology</option>
          <option value="Financial">Financial</option>
          <option value="Basic Materials">Basic Materials</option>
          <option value="Macro">Macro</option>
          <option value="Other">Other</option>
        </select>

        {/* Sort Filter (actually not applied to treemap as squarify does size-based, but we keep the UI intent) */}
        <select 
          className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 outline-none"
          value={sortFilter} 
          onChange={(e) => setSortFilter(e.target.value)}
        >
          <option value="MarketCap">Size by Market Cap</option>
          <option value="Performance">Size by Performance</option>
        </select>
      </div>

      {/* Status Indicators */}
      <div className="flex items-center gap-4 text-xs font-mono">
        <span className="text-slate-500 hidden sm:inline">Data delayed by 15 mins</span>
        
        {isRefreshing ? (
          <span className="text-amber-400 animate-pulse flex items-center gap-1">
            <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
            </svg>
            Refreshing...
          </span>
        ) : (
          <span className={isStale ? "text-amber-500" : "text-green-500"}>
            Refresh in: {formatTime(countdown)}
          </span>
        )}
      </div>
    </div>
  );
};

export default HeatmapFilters;
