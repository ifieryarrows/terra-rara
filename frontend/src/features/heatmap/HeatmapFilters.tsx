import React, { useEffect, useState } from 'react';

interface HeatmapFiltersProps {
  groupFilter: string;
  setGroupFilter: (val: string) => void;
  sortFilter: string;
  setSortFilter: (val: string) => void;
  availableGroups: string[];
  meta: any;
}

const HeatmapFilters: React.FC<HeatmapFiltersProps> = ({
  groupFilter,
  setGroupFilter,
  sortFilter,
  setSortFilter,
  availableGroups,
  meta,
}) => {
  const [countdown, setCountdown] = useState<number>(15 * 60);

  useEffect(() => {
    if (!meta?.next_refresh_at) return;
    const tick = () => {
      const next = new Date(meta.next_refresh_at).getTime();
      setCountdown(Math.max(0, Math.floor((next - Date.now()) / 1000)));
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [meta]);

  const fmt = (s: number) =>
    `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;

  const isStale = meta?.is_stale;
  const isRefreshing = meta?.refresh_in_progress;

  return (
    <div className="flex flex-wrap items-center justify-between bg-[#0f172a] border-b border-slate-700 px-3 py-2 text-sm text-slate-300 gap-2">
      <div className="flex flex-wrap gap-3 items-center">
        {/* Group filter — populated from the live API response */}
        <select
          className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs outline-none"
          value={groupFilter}
          onChange={(e) => setGroupFilter(e.target.value)}
        >
          <option value="ALL">All Groups</option>
          {availableGroups.map((g) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>

        {/* Sort mode */}
        <select
          className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs outline-none"
          value={sortFilter}
          onChange={(e) => setSortFilter(e.target.value)}
        >
          <option value="Weight">Size by Weight</option>
          <option value="Performance">Size by Performance</option>
        </select>
      </div>

      {/* Status indicator */}
      <div className="flex items-center gap-3 text-xs font-mono">
        <span className="text-slate-600 hidden sm:inline">15 min delay</span>
        {isRefreshing ? (
          <span className="text-amber-400 animate-pulse flex items-center gap-1">
            <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
            Refreshing
          </span>
        ) : (
          <span className={isStale ? 'text-amber-500' : 'text-green-500'}>
            {fmt(countdown)}
          </span>
        )}
      </div>
    </div>
  );
};

export default HeatmapFilters;
