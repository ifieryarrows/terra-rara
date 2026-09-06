import { memo, useEffect, useState } from 'react';
import type { HeatmapMeta } from './heatmap-layout';
import { FilterChip } from '../../components/ui/FilterChip';

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
    <div className="cm-heatmap-toolbar">
      <div className="cm-heatmap-fields">
        <div className="cm-news-filter-group" role="group" aria-label="Heatmap hierarchy">
          {(['market', 'themes'] as const).map((option) => (
            <FilterChip
              key={option}
              onClick={() => setView(option)}
              active={view === option}
            >
              {option === 'market' ? 'Market' : 'Themes'}
            </FilterChip>
          ))}
        </div>
        <label className="cm-field"><span>Category</span><select className="cm-input" value={groupFilter} onChange={(event) => setGroupFilter(event.target.value)} aria-label="Filter top-level category">
          <option value="ALL">All categories</option>
          {availableGroups.map((group) => <option key={group} value={group}>{group}</option>)}
        </select></label>
        <label className="cm-field"><span>Cell size</span><select className="cm-input" value={sortFilter} onChange={(event) => setSortFilter(event.target.value as 'Weight' | 'Performance')} aria-label="Cell sizing">
          <option value="Weight">Size by weight</option>
          <option value="Performance">Size by performance</option>
        </select></label>
        {(groupFilter !== 'ALL' || sortFilter !== 'Weight') && <button type="button" className="cm-filter-chip" onClick={() => { setGroupFilter('ALL'); setSortFilter('Weight'); }}>Reset map filters</button>}
      </div>
      <div className="flex items-center gap-2 font-mono">
        <span className="cm-chart-note">
          {meta.refresh_in_progress ? 'Refreshing snapshot' : meta.is_stale ? 'Older snapshot' : 'Available snapshot'}
          {meta.next_refresh_at && ` · Next check ${format(countdown)}`}
          {meta.source_delay_minutes > 0 && ` · Quotes delayed ${meta.source_delay_minutes} min`}
        </span>
      </div>
    </div>
  );
});

export default HeatmapFilters;
