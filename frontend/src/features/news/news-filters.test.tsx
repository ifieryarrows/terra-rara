// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { NewsIntelligencePanel } from './NewsIntelligencePanel';

const state = vi.hoisted(() => ({ feed: vi.fn(), stats: vi.fn() }));
vi.mock('../../hooks/useNews', () => ({ useNewsFeed: state.feed, useNewsStats: state.stats, flattenNewsPages: (pages: any[]) => pages?.flatMap(page => page.items) ?? [] }));
vi.mock('./NewsDetailDrawer', () => ({ default: () => null }));
beforeEach(() => {
  vi.stubGlobal('IntersectionObserver', class { observe() {} disconnect() {} });
  state.feed.mockReturnValue({ data: { pages: [{ items: [], total: 0 }] }, isLoading: false, isFetching: false, isFetchingNextPage: false, hasNextPage: false, refetch: vi.fn() });
  state.stats.mockImplementation((filters: any) => ({ data: {
    top_publishers: [{ publisher: 'Reuters', count: 2 }],
    channel_distribution: filters.channel === 'newsapi' ? { newsapi: 2 } : { newsapi: 2, google_news: 3 },
  } }));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.clearAllMocks(); });

it('toggles a publisher off and restores all channels after the distribution narrows', async () => {
  const user = userEvent.setup();
  render(<NewsIntelligencePanel/>);
  const publisher = screen.getByRole('button', { name: 'Reuters' });
  await user.click(publisher);
  expect(publisher).toHaveAttribute('aria-pressed', 'true');
  await user.click(publisher);
  expect(publisher).toHaveAttribute('aria-pressed', 'false');
  await user.click(screen.getByRole('button', { name: 'News filters' }));
  await user.click(screen.getByRole('button', { name: 'NewsAPI' }));
  expect(screen.getByRole('button', { name: 'All channels' })).toBeVisible();
  await user.click(screen.getByRole('button', { name: 'All channels' }));
  expect(state.feed.mock.lastCall?.[0].channel).toBe('all');
});

it('debounces search and resets the effective query along with the visible input', async () => {
  const user = userEvent.setup();
  render(<NewsIntelligencePanel/>);
  await user.type(screen.getByRole('searchbox', { name: 'Search headlines' }), 'copper');
  await waitFor(() => expect(state.feed.mock.lastCall?.[0].search).toBe('copper'));
  await user.click(screen.getByRole('button', { name: 'Reset filters' }));
  expect(screen.getByRole('searchbox')).toHaveValue('');
  await waitFor(() => expect(state.feed.mock.lastCall?.[0].search).toBeUndefined());
});
