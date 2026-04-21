import React, { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import {
  Newspaper,
  Filter,
  RefreshCw,
  AlertCircle,
  Search,
  ChevronDown,
} from 'lucide-react';
import { useNewsFeed, useNewsStats, flattenNewsPages } from '../../hooks/useNews';
import type { NewsFeedFilters, NewsItem, NewsLabel } from '../../types';
import NewsCard from './NewsCard';
import NewsDetailDrawer from './NewsDetailDrawer';

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
  since_hours: 48,
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
  const [filters, setFilters] = useState<NewsFeedFilters>(DEFAULT_FILTERS);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [searchDraft, setSearchDraft] = useState('');
  const [selectedItem, setSelectedItem] = useState<NewsItem | null>(null);
  const loadMoreRef = useRef<HTMLDivElement | null>(null);

  const debouncedSearch = useDebouncedValue(searchDraft, 300);
  const effectiveFilters = useMemo<NewsFeedFilters>(
    () => ({ ...filters, search: debouncedSearch || undefined }),
    [filters, debouncedSearch],
  );

  const feed = useNewsFeed(effectiveFilters);
  const stats = useNewsStats(24);

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
      { root: null, rootMargin: '200px', threshold: 0 },
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
      className="glass-panel flex flex-col h-full max-h-[calc(100vh-120px)] overflow-hidden"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 pt-5 pb-3 border-b border-white/5">
        <div className="flex items-center gap-2 text-gray-400">
          <Newspaper size={16} className="text-copper-400" />
          <span className="text-xs font-bold tracking-widest uppercase">News Intelligence</span>
        </div>
        <button
          type="button"
          onClick={() => feed.refetch()}
          className={clsx(
            'p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors',
            isRefreshing && 'text-copper-300',
          )}
          title="Refresh"
          aria-label="Refresh news feed"
        >
          <RefreshCw size={14} className={clsx(isRefreshing && 'animate-spin')} />
        </button>
      </div>

      {/* Stats summary */}
      <div className="px-5 pt-3 pb-3 border-b border-white/5">
        <div className="flex items-center gap-1.5 text-[10px] font-mono mb-2">
          <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300" title="Bullish (24h)">
            ↑ {bullishCount}
          </span>
          <span className="px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-300" title="Bearish (24h)">
            ↓ {bearishCount}
          </span>
          <span className="px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-200" title="Neutral (24h)">
            · {neutralCount}
          </span>
          <span className="ml-auto text-gray-500">
            {totalMatching} hit{totalMatching === 1 ? '' : 's'}
          </span>
        </div>

        {topPublishers.length > 0 && (
          <div className="flex items-center gap-1.5 flex-wrap">
            {topPublishers.map((p) => (
              <button
                key={p.publisher}
                type="button"
                onClick={() => updateFilter('publisher', p.publisher)}
                className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white/5 text-gray-300 hover:bg-copper-500/20 hover:text-copper-200 transition-colors truncate max-w-[110px]"
                title={`${p.publisher} (${p.count} articles)`}
              >
                {p.publisher}
              </button>
            ))}
          </div>
        )}

        {/* Search + filter toggle */}
        <div className="flex items-center gap-2 mt-3">
          <div className="flex-1 relative">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-500" />
            <input
              type="text"
              value={searchDraft}
              onChange={(e) => setSearchDraft(e.target.value)}
              placeholder="Search headlines..."
              className="w-full bg-white/5 border border-white/5 rounded-lg pl-8 pr-2 py-1.5 text-xs text-gray-200 placeholder-gray-500 focus:outline-none focus:border-copper-400/40"
            />
          </div>
          <button
            type="button"
            onClick={() => setFiltersOpen((v) => !v)}
            className={clsx(
              'flex items-center gap-1 px-2 py-1.5 rounded-lg text-xs font-medium transition-colors',
              filtersOpen
                ? 'bg-copper-500/20 text-copper-200 border border-copper-400/40'
                : 'bg-white/5 text-gray-400 border border-white/5 hover:border-copper-400/30',
            )}
            title="Toggle filters"
          >
            <Filter size={12} />
            <ChevronDown size={12} className={clsx('transition-transform', filtersOpen && 'rotate-180')} />
          </button>
        </div>

        {filtersOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-3 space-y-3 overflow-hidden"
          >
            {/* Label chips */}
            <div className="flex items-center gap-1.5 flex-wrap">
              {LABEL_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => updateFilter('label', opt.id)}
                  className={clsx(
                    'text-[10px] font-mono px-2 py-0.5 rounded-full border transition-colors',
                    filters.label === opt.id
                      ? 'border-copper-400/60 ' + opt.tone
                      : 'border-white/5 bg-white/[0.02] text-gray-400 hover:border-copper-400/30',
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            {/* Time window */}
            <div>
              <div className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">Window</div>
              <div className="flex items-center gap-1.5">
                {SINCE_OPTIONS.map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => updateFilter('since_hours', opt.id)}
                    className={clsx(
                      'text-[10px] font-mono px-2 py-0.5 rounded-full border transition-colors',
                      filters.since_hours === opt.id
                        ? 'bg-copper-500/20 text-copper-200 border-copper-400/50'
                        : 'bg-white/[0.02] text-gray-400 border-white/5 hover:border-copper-400/30',
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Min relevance */}
            <div>
              <div className="flex items-center justify-between text-[10px] font-mono text-gray-400 mb-1">
                <span className="uppercase tracking-widest text-gray-500">Min relevance</span>
                <span>{Math.round((filters.min_relevance ?? 0) * 100)}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={0.9}
                step={0.05}
                value={filters.min_relevance ?? 0}
                onChange={(e) => updateFilter('min_relevance', Number(e.target.value))}
                className="w-full accent-copper-400"
              />
            </div>

            {/* Channel toggle — only render when both channels have data */}
            {availableChannels.length > 1 && (
              <div>
                <div className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">Channel</div>
                <div className="flex items-center gap-1.5 flex-wrap">
                  <button
                    type="button"
                    onClick={() => updateFilter('channel', 'all')}
                    className={clsx(
                      'text-[10px] font-mono px-2 py-0.5 rounded-full border transition-colors',
                      filters.channel === 'all' || !filters.channel
                        ? 'bg-copper-500/20 text-copper-200 border-copper-400/50'
                        : 'bg-white/[0.02] text-gray-400 border-white/5 hover:border-copper-400/30',
                    )}
                  >
                    All
                  </button>
                  {availableChannels.map((ch) => (
                    <button
                      key={ch}
                      type="button"
                      onClick={() => updateFilter('channel', ch)}
                      className={clsx(
                        'text-[10px] font-mono px-2 py-0.5 rounded-full border transition-colors',
                        filters.channel === ch
                          ? 'bg-copper-500/20 text-copper-200 border-copper-400/50'
                          : 'bg-white/[0.02] text-gray-400 border-white/5 hover:border-copper-400/30',
                      )}
                    >
                      {ch}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Publisher filter (shown when one is picked) */}
            {filters.publisher && (
              <div className="flex items-center justify-between text-[10px] font-mono">
                <span className="text-gray-500 uppercase tracking-widest">Publisher</span>
                <button
                  type="button"
                  onClick={() => updateFilter('publisher', undefined)}
                  className="text-copper-300 hover:text-copper-200"
                >
                  Clear "{filters.publisher}"
                </button>
              </div>
            )}
          </motion.div>
        )}
      </div>

      {/* Feed list */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
        {isLoading && (
          <div className="space-y-2">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="h-[96px] rounded-xl bg-white/[0.03] animate-pulse" />
            ))}
          </div>
        )}

        {!isLoading && feed.isError && (
          <div className="flex flex-col items-center gap-2 p-6 text-center">
            <AlertCircle size={20} className="text-rose-400" />
            <p className="text-xs text-gray-400">
              {feed.error?.message || 'Unable to load news feed.'}
            </p>
            <button
              type="button"
              onClick={() => feed.refetch()}
              className="text-xs text-copper-300 hover:text-copper-200"
            >
              Retry
            </button>
          </div>
        )}

        {!isLoading && !feed.isError && items.length === 0 && (
          <div className="flex flex-col items-center gap-2 p-6 text-center">
            <Newspaper size={20} className="text-gray-600" />
            <p className="text-xs text-gray-500">
              No articles match the current filters.
            </p>
          </div>
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
          <div className="text-center py-2 text-[10px] font-mono text-gray-600 tracking-wider uppercase">
            — end of feed —
          </div>
        )}
      </div>

      <NewsDetailDrawer item={selectedItem} onClose={() => setSelectedItem(null)} />
    </motion.aside>
  );
};

export default NewsIntelligencePanel;
