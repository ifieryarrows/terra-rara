import {
  useInfiniteQuery,
  useQuery,
  keepPreviousData,
} from '@tanstack/react-query';
import { fetchNews, fetchNewsById, fetchNewsStats } from '../api';
import type {
  NewsFeedFilters,
  NewsItem,
  NewsListResponse,
  NewsStatsResponse,
} from '../types';

const NEWS_PAGE_SIZE = 20;
const NEWS_POLL_MS = 90_000;
const NEWS_STATS_POLL_MS = 120_000;

/**
 * Stable queryKey dependency for the news feed. Purposefully excludes `limit`
 * and `offset` so pagination lives inside `useInfiniteQuery`'s `pageParam`.
 */
function buildNewsKey(filters: NewsFeedFilters) {
  return [
    'news-feed',
    {
      since_hours: filters.since_hours ?? 168,
      label: filters.label ?? 'all',
      event_type: filters.event_type ?? 'all',
      min_relevance: filters.min_relevance ?? 0,
      channel: filters.channel ?? 'all',
      publisher: (filters.publisher ?? '').trim().toLowerCase(),
      search: (filters.search ?? '').trim().toLowerCase(),
    },
  ] as const;
}

/**
 * Infinite-scrolling news feed used by NewsIntelligencePanel.
 *
 * Cache strategy:
 *   - staleTime 30s — debounce bursty UI refetches (filter toggles etc.)
 *   - refetchInterval 90s — match pipeline polling cadence without spamming.
 *   - keepPreviousData — when filters change the previous list stays visible
 *     until the new page resolves (no flash of emptiness).
 */
export function useNewsFeed(filters: NewsFeedFilters = {}) {
  return useInfiniteQuery<NewsListResponse, Error>({
    queryKey: buildNewsKey(filters),
    initialPageParam: 0,
    queryFn: async ({ pageParam }) => {
      const offset = typeof pageParam === 'number' ? pageParam : 0;
      return fetchNews({ ...filters, offset, limit: filters.limit ?? NEWS_PAGE_SIZE });
    },
    getNextPageParam: (lastPage) => {
      if (!lastPage.has_more) return undefined;
      return lastPage.offset + lastPage.limit;
    },
    staleTime: 30_000,
    refetchInterval: NEWS_POLL_MS,
    refetchOnWindowFocus: false,
    placeholderData: keepPreviousData,
  });
}

/**
 * Flatten paginated news feed results into a single item list. Callers can use
 * this in virtualized lists without re-computing `.flatMap` on every render.
 */
export function flattenNewsPages(pages: NewsListResponse[] | undefined): NewsItem[] {
  if (!pages || pages.length === 0) return [];
  return pages.flatMap((page) => page.items);
}

export function useNewsStats(sinceHours = 168) {
  return useQuery<NewsStatsResponse, Error>({
    queryKey: ['news-stats', sinceHours],
    queryFn: () => fetchNewsStats(sinceHours),
    refetchInterval: NEWS_STATS_POLL_MS,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });
}

export function useNewsDetail(processedId: number | null) {
  return useQuery<NewsItem, Error>({
    queryKey: ['news-detail', processedId],
    queryFn: () => fetchNewsById(processedId as number),
    enabled: processedId !== null && processedId !== undefined,
    staleTime: 5 * 60_000,
    refetchOnWindowFocus: false,
  });
}
