import React, { useId, useMemo, useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import {
  Newspaper,
  Filter,
  RefreshCw,
  Search,
} from 'lucide-react';
import { useNewsFeed, useNewsStats, flattenNewsPages } from '../../hooks/useNews';
import type { NewsFeedFilters, NewsItem, NewsLabel } from '../../types';
import NewsCard from './NewsCard';
import NewsDetailDrawer from './NewsDetailDrawer';
import { FilterChip } from '../../components/ui/FilterChip';
import { ViewState } from '../../components/ui/ViewState';
import { RefreshButton } from '../../components/ui/RefreshButton';

const LABEL_OPTIONS: Array<{ id: 'all' | NewsLabel; label: string; tone: string }> = [
  { id: 'all', label: 'All', tone: 'bg-white/5 text-gray-300' },
  { id: 'BULLISH', label: 'Bullish', tone: 'bg-emerald-500/15 text-emerald-300' },
  { id: 'BEARISH', label: 'Bearish', tone: 'bg-rose-500/15 text-rose-300' },
  { id: 'NEUTRAL', label: 'Neutral', tone: 'bg-amber-500/10 text-amber-200' },
];

const SINCE_OPTIONS = [
  { id: 24, label: '24h' },
  { id: 48, label: '48h' },
  { id: 96, label: '4d' },
  { id: 168, label: '7d' },
];

const DEFAULT_FILTERS: NewsFeedFilters = {
  limit: 20,
  since_hours: 168,
  label: 'all',
  min_relevance: 0.2,
  channel: 'all',
};

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}

export const NewsIntelligencePanel: React.FC = () => {
  const filterId = useId();
  const [filters, setFilters] = useState<NewsFeedFilters>(DEFAULT_FILTERS);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [searchDraft, setSearchDraft] = useState('');
  const [selectedItem, setSelectedItem] = useState<NewsItem | null>(null);
  const hasActiveFilters = !!searchDraft || filters.label !== DEFAULT_FILTERS.label || filters.since_hours !== DEFAULT_FILTERS.since_hours || filters.min_relevance !== DEFAULT_FILTERS.min_relevance || filters.channel !== DEFAULT_FILTERS.channel || !!filters.publisher;
  const resetFilters = () => { setFilters(DEFAULT_FILTERS); setSearchDraft(''); };
  const loadMoreRef = useRef<HTMLDivElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const debouncedSearch = useDebouncedValue(searchDraft, 300);
  const effectiveFilters = useMemo<NewsFeedFilters>(
    () => ({ ...filters, search: debouncedSearch || undefined }),
    [filters, debouncedSearch],
  );
  const activeWindowHours = effectiveFilters.since_hours ?? 168;
  const activeWindowLabel = SINCE_OPTIONS.find((opt) => opt.id === activeWindowHours)?.label ?? `${activeWindowHours}h`;

  const feed = useNewsFeed(effectiveFilters);
  const stats = useNewsStats(effectiveFilters);

  const items = useMemo(() => flattenNewsPages(feed.data?.pages), [feed.data]);
  const totalMatching = feed.data?.pages?.[0]?.total ?? items.length;

  const availableChannels = useMemo(() => {
    const dist = stats.data?.channel_distribution ?? {};
    return Object.keys(dist).filter((k) => dist[k] > 0);
  }, [stats.data]);

  const topPublishers = stats.data?.top_publishers?.slice(0, 3) ?? [];

  const updateFilter = useCallback(<K extends keyof NewsFeedFilters>(key: K, value: NewsFeedFilters[K]) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  }, []);

  // Infinite scroll — fire the next page request when the sentinel scrolls
  // into view. Guarded on hasNextPage/isFetchingNextPage to avoid duplicate
  // fetches under rapid scroll.
  useEffect(() => {
    const el = loadMoreRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting && feed.hasNextPage && !feed.isFetchingNextPage) {
            feed.fetchNextPage();
          }
        }
      },
      { root: scrollRef.current, rootMargin: '200px', threshold: 0 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [feed]);

  const isLoading = feed.isLoading && items.length === 0;
  const isRefreshing = feed.isFetching && !feed.isFetchingNextPage;

  const labelDist = stats.data?.label_distribution ?? {};
  const bullishCount = labelDist.BULLISH ?? 0;
  const bearishCount = labelDist.BEARISH ?? 0;
  const neutralCount = labelDist.NEUTRAL ?? 0;

  return (
    <motion.aside
      className="cm-news-panel glass-panel"
      aria-label="News intelligence"
      initial={false}
    >
      {/* Header */}
      <div className="cm-news-header flex items-center justify-between px-3 sm:px-4 pt-4 pb-2.5 border-b border-white/5">
        <div className="flex items-center gap-2 text-gray-400">
          <Newspaper size={16} className="text-copper-400" />
          <h2>News Intelligence</h2>
        </div>
        <button
          type="button"
          onClick={() => feed.refetch()}
          className="cm-icon-button"
          disabled={isRefreshing}
          title="Refresh"
          aria-label="Refresh news feed"
        >
          <RefreshCw size={14} className={clsx(isRefreshing && 'animate-spin')} />
        </button>
      </div>

      <div ref={scrollRef} className="cm-news-scroll" tabIndex={0} role="region" aria-label="News filters and headlines">
      {/* Stats summary */}
      <div className="px-3 sm:px-4 pt-2.5 pb-3 border-b border-white/5">
        <div className="flex items-center gap-1.5 text-xs font-mono mb-2">
          <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300" title={`Bullish (${activeWindowLabel})`}>
            ↑ {stats.data ? bullishCount : '—'}
          </span>
          <span className="px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-300" title={`Bearish (${activeWindowLabel})`}>
            ↓ {stats.data ? bearishCount : '—'}
          </span>
          <span className="px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-200" title={`Neutral (${activeWindowLabel})`}>
            · {stats.data ? neutralCount : '—'}
          </span>
          <span className="ml-auto text-slate-400">
            {feed.data ? `${totalMatching} hit${totalMatching === 1 ? '' : 's'}` : 'Awaiting headlines'}
          </span>
        </div>

        {topPublishers.length > 0 && (
          <div className="flex items-center gap-1.5 flex-wrap">
            {topPublishers.map((p) => (
              <FilterChip
                key={p.publisher}
                active={filters.publisher === p.publisher}
                onClick={() => updateFilter('publisher', filters.publisher === p.publisher ? undefined : p.publisher)}
                title={`${p.publisher} (${p.count} articles)`}
              >
                {p.publisher}
              </FilterChip>
            ))}
          </div>
        )}

        <div className="cm-news-controls">
          <label className="cm-field">
            <span>Search headlines</span>
            <span><Search size={15} aria-hidden="true"/><input type="search" value={searchDraft} onChange={event => setSearchDraft(event.target.value)} placeholder="Company, topic or keyword" className="cm-input cm-input--search"/></span>
          </label>
          <button type="button" onClick={() => setFiltersOpen(value => !value)} className="cm-icon-button" aria-label="News filters" aria-expanded={filtersOpen} aria-controls={filterId}><Filter size={16} aria-hidden="true"/></button>
        </div>
        {hasActiveFilters && <button type="button" className="cm-filter-chip mt-2" onClick={resetFilters}>Reset filters{filters.publisher ? ` · ${filters.publisher}` : ''}</button>}
        <div id={filterId} hidden={!filtersOpen} className="mt-3 space-y-4">
          <fieldset className="cm-news-filter-group"><legend>Sentiment</legend>{LABEL_OPTIONS.map(option => <FilterChip key={option.id} active={filters.label === option.id} onClick={() => updateFilter('label', option.id)}>{option.label}</FilterChip>)}</fieldset>
          <fieldset className="cm-news-filter-group"><legend>Time window</legend>{SINCE_OPTIONS.map(option => <FilterChip key={option.id} active={filters.since_hours === option.id} onClick={() => updateFilter('since_hours', option.id)}>{option.label}</FilterChip>)}</fieldset>
          <label className="cm-field"><span>Minimum relevance · {Math.round((filters.min_relevance ?? 0) * 100)}%</span><input type="range" min={0} max={0.9} step={0.05} value={filters.min_relevance ?? 0} onChange={event => updateFilter('min_relevance', Number(event.target.value))} className="w-full"/></label>
          {(availableChannels.length > 1 || (filters.channel && filters.channel !== 'all')) && <fieldset className="cm-news-filter-group"><legend>Channel</legend><FilterChip active={!filters.channel || filters.channel === 'all'} onClick={() => updateFilter('channel', 'all')}>All channels</FilterChip>{Array.from(new Set([...availableChannels, ...(filters.channel && filters.channel !== 'all' ? [filters.channel] : [])])).map(channel => <FilterChip key={channel} active={filters.channel === channel} onClick={() => updateFilter('channel', channel)}>{channel === 'google_news' ? 'Google News' : channel === 'newsapi' ? 'NewsAPI' : channel}</FilterChip>)}</fieldset>}
        </div>
      </div>
      {/* Feed list */}
      {isRefreshing && items.length > 0 && <p className="cm-news-updating" role="status">Updating headlines… Previous results remain visible.</p>}
      <div className="px-2 sm:px-2.5 py-2.5 space-y-1.5">
        {isLoading && <ViewState kind="loading" title="Loading headlines" compact/>}

        {!isLoading && feed.isError && (
          <ViewState kind="error" title="News could not be loaded" description="Check again to retrieve the latest available headlines." action={<RefreshButton onClick={() => feed.refetch()} busy={isRefreshing} label="Retry"/>} compact/>
        )}

        {!isLoading && !feed.isError && items.length === 0 && (
          <ViewState kind="empty" title="No matching headlines" description="Try a broader search or reset your filters." action={hasActiveFilters && <button type="button" className="cm-button cm-button--secondary" onClick={resetFilters}>Show all headlines</button>} compact/>
        )}

        {items.map((item) => (
          <NewsCard
            key={item.id}
            item={item}
            selected={selectedItem?.id === item.id}
            onSelect={setSelectedItem}
          />
        ))}

        {/* Infinite scroll sentinel */}
        <div ref={loadMoreRef} />

        {feed.isFetchingNextPage && (
          <div className="flex justify-center py-3">
            <RefreshCw size={14} className="text-copper-400/80 animate-spin" />
          </div>
        )}

        {!feed.hasNextPage && items.length > 0 && (
          <div className="text-center py-2 text-xs font-mono text-slate-400 tracking-wider uppercase">
            — end of feed —
          </div>
        )}
      </div>
      </div>

      <NewsDetailDrawer item={selectedItem} onClose={() => setSelectedItem(null)} />
    </motion.aside>
  );
};

export default NewsIntelligencePanel;
