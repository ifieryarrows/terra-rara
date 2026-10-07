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
  hasSnapshot: boolean;
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
  hasSnapshot,
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
  const status = !hasSnapshot
    ? meta.refresh_in_progress ? 'Preparing snapshot' : 'Snapshot unavailable'
    : meta.refresh_in_progress ? 'Refreshing snapshot' : meta.is_stale ? 'Older snapshot' : 'Available snapshot';
  const snapshotState = !hasSnapshot ? 'unavailable' : meta.is_stale ? 'stale' : meta.refresh_in_progress ? 'refreshing' : 'available';

  return <div className="cm-heatmap-filters">
    <fieldset className="cm-heatmap-filter-group">
      <legend>View</legend>
      <div className="cm-heatmap-view-options" role="group" aria-label="Map view">
        <FilterChip onClick={() => setView('market')} active={view === 'market'}>Market</FilterChip>
        <FilterChip onClick={() => setView('themes')} active={view === 'themes'}>Themes</FilterChip>
      </div>
    </fieldset>

    <fieldset className="cm-heatmap-filter-group cm-heatmap-universe">
      <legend>Universe</legend>
      <div className="cm-heatmap-group-list" role="group" aria-label="Filter top-level category">
        <button type="button" className="cm-heatmap-filter-row" aria-pressed={groupFilter === 'ALL'} onClick={() => setGroupFilter('ALL')}>All categories</button>
        {availableGroups.map((group) => <button key={group} type="button" className="cm-heatmap-filter-row" aria-pressed={groupFilter === group} onClick={() => setGroupFilter(group)}>{group}</button>)}
        {!availableGroups.length && <span className="cm-heatmap-filter-empty">Categories appear with the next available snapshot.</span>}
      </div>
    </fieldset>

    <fieldset className="cm-heatmap-filter-group">
      <legend>Tile size</legend>
      <div className="cm-heatmap-size-options" role="group" aria-label="Cell sizing">
        <FilterChip onClick={() => setSortFilter('Weight')} active={sortFilter === 'Weight'}>Weight</FilterChip>
        <FilterChip onClick={() => setSortFilter('Performance')} active={sortFilter === 'Performance'}>Change</FilterChip>
      </div>
    </fieldset>

    <div className="cm-heatmap-snapshot" aria-label="Snapshot status">
      <span className={`cm-heatmap-snapshot-dot is-${snapshotState}`} aria-hidden="true"/>
      <div><strong>{status}</strong><span>{meta.next_refresh_at ? `Next check ${format(countdown)}` : 'Refresh timing unavailable'}</span></div>
      {meta.source_delay_minutes > 0 && <small>Quotes delayed {meta.source_delay_minutes} min</small>}
    </div>
  </div>;
});

export default HeatmapFilters;
