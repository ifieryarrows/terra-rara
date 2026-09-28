import React, { useId, useMemo, useState, useEffect, useLayoutEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import {
  ArrowLeft,
  ArrowRight,
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
  const [isDraggingHeadlines, setIsDraggingHeadlines] = useState(false);
  const hasActiveFilters = !!searchDraft || filters.label !== DEFAULT_FILTERS.label || filters.since_hours !== DEFAULT_FILTERS.since_hours || filters.min_relevance !== DEFAULT_FILTERS.min_relevance || filters.channel !== DEFAULT_FILTERS.channel || !!filters.publisher;
  const resetFilters = () => { setFilters(DEFAULT_FILTERS); setSearchDraft(''); };
  const newsRailRef = useRef<HTMLDivElement | null>(null);
  const originalNewsSetRef = useRef<HTMLDivElement | null>(null);
  const flowPausedRef = useRef(false);
  const flowHoveredRef = useRef(false);
  const flowResumeTimer = useRef<number | undefined>(undefined);
  const flowFrameRef = useRef<number | null>(null);
  const lastFlowFrameTimeRef = useRef<number | null>(null);
  const flowPositionRef = useRef(0);
  const dragState = useRef<{ pointerId: number; startX: number; startScroll: number; moved: boolean; captureTarget: HTMLElement } | null>(null);
  const suppressCardClick = useRef(false);

  const debouncedSearch = useDebouncedValue(searchDraft, 300);
  const effectiveFilters = useMemo<NewsFeedFilters>(
    () => ({ ...filters, search: debouncedSearch || undefined }),
    [filters, debouncedSearch],
  );
  const activeWindowHours = effectiveFilters.since_hours ?? 168;
  const activeWindowLabel = SINCE_OPTIONS.find((opt) => opt.id === activeWindowHours)?.label ?? `${activeWindowHours}h`;

  const pauseNewsFlow = useCallback(() => {
    flowPausedRef.current = true;
    if (flowResumeTimer.current !== undefined) window.clearTimeout(flowResumeTimer.current);
    flowResumeTimer.current = window.setTimeout(() => {
      flowPausedRef.current = false;
      flowResumeTimer.current = undefined;
    }, 3_000);
  }, []);

  useEffect(() => () => {
    if (flowResumeTimer.current !== undefined) window.clearTimeout(flowResumeTimer.current);
    if (flowFrameRef.current !== null) window.cancelAnimationFrame(flowFrameRef.current);
  }, []);

  const feed = useNewsFeed(effectiveFilters);
  const stats = useNewsStats(effectiveFilters);

  const items = useMemo(() => flattenNewsPages(feed.data?.pages), [feed.data]);
  const totalMatching = feed.data?.pages?.[0]?.total ?? items.length;

  useLayoutEffect(() => {
    const rail = newsRailRef.current;
    const set = originalNewsSetRef.current;
    if (rail && set) {
      rail.scrollLeft = set.getBoundingClientRect().width;
      flowPositionRef.current = rail.scrollLeft;
    }
  }, [items]);

  useEffect(() => {
    const rail = newsRailRef.current;
    const set = originalNewsSetRef.current;
    if (!rail || !set || items.length < 2 || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    let width = set.getBoundingClientRect().width;
    const recenter = () => {
      const nextWidth = set.getBoundingClientRect().width;
      if (nextWidth <= 0 || nextWidth === width) return;
      width = nextWidth;
      rail.scrollLeft = width;
      flowPositionRef.current = rail.scrollLeft;
    };
    const resizeObserver = new ResizeObserver(recenter);
    resizeObserver.observe(rail);
    resizeObserver.observe(set);

    const animate = (timestamp: number) => {
      const previous = lastFlowFrameTimeRef.current ?? timestamp;
      const elapsed = Math.min(48, Math.max(0, timestamp - previous));
      lastFlowFrameTimeRef.current = timestamp;
      width = set.getBoundingClientRect().width;
      if (width > rail.clientWidth && width > 0 && !flowPausedRef.current) {
        const speed = flowHoveredRef.current ? 4.5 : 21;
        const next = flowPositionRef.current - speed * elapsed / 1_000;
        flowPositionRef.current = next <= 0 ? width + next : next;
        rail.scrollLeft = flowPositionRef.current;
      } else {
        flowPositionRef.current = rail.scrollLeft;
      }
      flowFrameRef.current = window.requestAnimationFrame(animate);
    };
    flowFrameRef.current = window.requestAnimationFrame(animate);
    return () => {
      resizeObserver.disconnect();
      if (flowFrameRef.current !== null) window.cancelAnimationFrame(flowFrameRef.current);
      flowFrameRef.current = null;
      lastFlowFrameTimeRef.current = null;
    };
  }, [items.length]);

  const availableChannels = useMemo(() => {
    const dist = stats.data?.channel_distribution ?? {};
    return Object.keys(dist).filter((k) => dist[k] > 0);
  }, [stats.data]);

  const topPublishers = stats.data?.top_publishers?.slice(0, 3) ?? [];

  const updateFilter = useCallback(<K extends keyof NewsFeedFilters>(key: K, value: NewsFeedFilters[K]) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  }, []);

  const scrollHeadlines = (direction: -1 | 1) => {
    const rail = newsRailRef.current;
    if (!rail) return;
    pauseNewsFlow();
    rail.scrollBy({
      left: direction * Math.max(240, rail.clientWidth * 0.82),
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
    });
  };

  const startHeadlineDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    pauseNewsFlow();
    if (event.pointerType === 'touch' || event.button !== 0) return;
    const rail = event.currentTarget;
    const card = event.target instanceof HTMLElement ? event.target.closest<HTMLElement>('.cm-news-card') : null;
    const captureTarget = card ?? rail;
    dragState.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startScroll: rail.scrollLeft,
      moved: false,
      captureTarget,
    };
    captureTarget.setPointerCapture(event.pointerId);
    setIsDraggingHeadlines(true);
  };

  const moveHeadlineDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragState.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const delta = event.clientX - drag.startX;
    if (!drag.moved && Math.abs(delta) < 4) return;
    drag.moved = true;
    pauseNewsFlow();
    const rail = event.currentTarget;
    const width = originalNewsSetRef.current?.getBoundingClientRect().width ?? 0;
    const max = rail.scrollWidth - rail.clientWidth;
    let next = drag.startScroll - delta;
    if (width > rail.clientWidth) {
      while (next < 0) next += width;
      while (next > max) next -= width;
    }
    rail.scrollLeft = Math.max(0, Math.min(max, next));
    flowPositionRef.current = rail.scrollLeft;
  };

  const finishHeadlineDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragState.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    suppressCardClick.current = drag.moved;
    dragState.current = null;
    setIsDraggingHeadlines(false);
    pauseNewsFlow();
    if (drag.captureTarget.hasPointerCapture(event.pointerId)) {
      drag.captureTarget.releasePointerCapture(event.pointerId);
    }
    if (suppressCardClick.current) window.setTimeout(() => { suppressCardClick.current = false; }, 0);
  };

  const pages = feed.data?.pages ?? [];
  const priorPageIds = new Set(pages.slice(0, -1).flatMap((page) => page.items.map((item) => item.id)));
  const latestPageHasNewItems = pages.length < 2 || (pages[pages.length - 1]?.items.some((item) => !priorPageIds.has(item.id)) ?? false);
  const canLoadMore = !!feed.hasNextPage && latestPageHasNewItems;

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
      data-cm-route-reveal="surface"
      initial={false}
    >
      {/* Header */}
      <div className="cm-news-header flex items-center justify-between px-3 sm:px-4 pt-4 pb-2.5 border-b border-white/5">
        <div className="cm-news-title-group">
          <span className="cm-news-icon"><Newspaper size={16} aria-hidden="true" /></span>
          <div><h2>Copper news flow</h2><p>Headlines, sentiment &amp; sources</p></div>
        </div>
        <span className="cm-news-window-tag">{activeWindowLabel}</span>
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

      <div className="cm-news-scroll" role="region" aria-label="News filters and headlines">
      {/* Stats summary */}
      <div className="cm-news-summary px-3 sm:px-4 pt-2.5 pb-3 border-b border-white/5">
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
      {items.length > 0 && <div className="cm-news-rail-tools">
        <p><span>{items.length}{feed.data ? ` / ${totalMatching}` : ''}</span> headlines <small>· scroll sideways to browse</small></p>
        <div className="cm-news-rail-controls" aria-label="Browse headlines">
          <button type="button" className="cm-icon-button" onClick={() => scrollHeadlines(-1)} aria-label="Previous headlines" aria-controls={`${filterId}-headlines`}><ArrowLeft size={15} aria-hidden="true"/></button>
          <button type="button" className="cm-icon-button" onClick={() => scrollHeadlines(1)} aria-label="Next headlines" aria-controls={`${filterId}-headlines`}><ArrowRight size={15} aria-hidden="true"/></button>
        </div>
      </div>}
      <div
        id={`${filterId}-headlines`}
        ref={newsRailRef}
        className={clsx('cm-news-feed', isDraggingHeadlines && 'is-dragging')}
        role="region"
        aria-label="Headlines, scroll horizontally"
        tabIndex={0}
        onPointerDown={startHeadlineDrag}
        onPointerMove={moveHeadlineDrag}
        onPointerUp={finishHeadlineDrag}
        onPointerCancel={finishHeadlineDrag}
        onMouseEnter={() => { flowHoveredRef.current = true; }}
        onMouseLeave={() => { flowHoveredRef.current = false; }}
        onWheel={pauseNewsFlow}
        onKeyDown={event => {
          if (['ArrowLeft', 'ArrowRight', 'Home', 'End', ' '].includes(event.key)) pauseNewsFlow();
        }}
        onClickCapture={event => {
          if (!suppressCardClick.current) return;
          event.preventDefault();
          event.stopPropagation();
          suppressCardClick.current = false;
        }}
      >
        {isLoading && <ViewState kind="loading" title="Loading headlines" compact/>}

        {!isLoading && feed.isError && (
          <ViewState kind="error" title="News could not be loaded" description="Check again to retrieve the latest available headlines." action={<RefreshButton onClick={() => feed.refetch()} busy={isRefreshing} label="Retry"/>} compact/>
        )}

        {!isLoading && !feed.isError && items.length === 0 && (
          <ViewState kind="empty" title="No matching headlines" description="Try a broader search or reset your filters." action={hasActiveFilters && <button type="button" className="cm-button cm-button--secondary" onClick={resetFilters}>Show all headlines</button>} compact/>
        )}

        {items.length > 0 && <>
          <div className="cm-news-feed-set cm-news-feed-set--duplicate" aria-hidden="true">
            {items.map((item) => (
              <NewsCard key={`copy-${item.id}`} item={item} duplicate selected={selectedItem?.id === item.id} onSelect={(newsItem) => { pauseNewsFlow(); setSelectedItem(newsItem); }}/>
            ))}
          </div>
          <div ref={originalNewsSetRef} className="cm-news-feed-set">
            {items.map((item) => (
              <NewsCard key={item.id} item={item} selected={selectedItem?.id === item.id} onSelect={(newsItem) => { pauseNewsFlow(); setSelectedItem(newsItem); }}/>
            ))}
          </div>
        </>}

      </div>
      {canLoadMore && items.length > 0 && <div className="cm-news-more">
        <button type="button" className="cm-news-more-button" onClick={() => { void feed.fetchNextPage(); }} disabled={feed.isFetchingNextPage}>
          {feed.isFetchingNextPage ? <><RefreshCw size={13} className="animate-spin" aria-hidden="true"/> Loading</> : <>Load more <ArrowRight size={13} aria-hidden="true"/></>}
        </button>
      </div>}
      {feed.hasNextPage && items.length > 0 && !latestPageHasNewItems && <p className="cm-news-pagination-note" role="status">No additional unique headlines are available.</p>}
      </div>

      <NewsDetailDrawer item={selectedItem} onClose={() => setSelectedItem(null)} />
    </motion.aside>
  );
};

export default NewsIntelligencePanel;
