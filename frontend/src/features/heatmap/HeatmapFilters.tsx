import { memo, useEffect, useState } from 'react';
import type { HeatmapMeta } from './heatmap-layout';

interface Props {
  groupFilter: string;
  setGroupFilter: (value: string) => void;
  sortFilter: 'Weight' | 'Performance';
  setSortFilter: (value: 'Weight' | 'Performance') => void;
  view: 'market' | 'themes';
  setView: (value: 'market' | 'themes') => void;
  availableGroups: string[];
  meta: HeatmapMeta;
}

const HeatmapFilters = memo(function HeatmapFilters({
  groupFilter,
  setGroupFilter,
  sortFilter,
  setSortFilter,
  view,
  setView,
  availableGroups,
  meta,
}: Props) {
  const [countdown, setCountdown] = useState(0);
  useEffect(() => {
    const tick = () => setCountdown(meta.next_refresh_at
      ? Math.max(0, Math.floor((new Date(meta.next_refresh_at).getTime() - Date.now()) / 1000))
      : 0);
    tick();
    const timer = window.setInterval(tick, 1_000);
    return () => window.clearInterval(timer);
  }, [meta.next_refresh_at]);
  const format = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-300">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex rounded border border-slate-700 bg-slate-900 p-0.5" aria-label="Heatmap hierarchy">
          {(['market', 'themes'] as const).map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => setView(option)}
              className={`rounded px-2 py-1 capitalize ${view === option ? 'bg-copper-500/20 text-copper-300' : 'text-slate-500 hover:text-slate-200'}`}
              aria-pressed={view === option}
            >
              {option === 'market' ? 'Market' : 'Themes'}
            </button>
          ))}
        </div>
        <select className="rounded border border-slate-700 bg-slate-900 px-2 py-1.5" value={groupFilter} onChange={(event) => setGroupFilter(event.target.value)} aria-label="Filter top-level category">
          <option value="ALL">All categories</option>
          {availableGroups.map((group) => <option key={group} value={group}>{group}</option>)}
        </select>
        <select className="rounded border border-slate-700 bg-slate-900 px-2 py-1.5" value={sortFilter} onChange={(event) => setSortFilter(event.target.value as 'Weight' | 'Performance')} aria-label="Cell sizing">
          <option value="Weight">Size by weight</option>
          <option value="Performance">Size by performance</option>
        </select>
      </div>
      <div className="flex items-center gap-2 font-mono">
        <span className={meta.refresh_in_progress ? 'text-amber-400' : meta.is_stale ? 'text-amber-500' : 'text-emerald-400'}>
          {meta.refresh_in_progress ? 'Refreshing' : `${format(countdown)} · ${meta.cache_state || 'cache'}`}
        </span>
      </div>
    </div>
  );
});

export default HeatmapFilters;
