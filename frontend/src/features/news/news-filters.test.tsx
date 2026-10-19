// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
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

const cardItems = (count: number) => Array.from({ length: count }, (_, i) => ({
  id: i + 1,
  raw_id: i + 1,
  title: `QA headline ${i + 1}`,
  description: 'Synthetic article for marquee regression coverage.',
  url: 'https://example.com/qa',
  channel: 'newsapi',
  publisher: 'QA Publisher A',
  published_at: '2026-05-01T12:00:00Z',
  language: 'en',
  sentiment: { label: 'NEUTRAL', relevance: 0.8, confidence: 0.7, final_score: 0, reasoning: 'Synthetic.', scoring_mode: 'llm' },
}));
const feedWith = (items: ReturnType<typeof cardItems>) => ({ data: { pages: [{ items, total: items.length }] }, isLoading: false, isFetching: false, isFetchingNextPage: false, hasNextPage: false, refetch: vi.fn() });
const leftoverMarqueeTransform = 'translate3d(-420px, 0px, 0px)';
const stubFlowEnvironment = () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: false,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
  }));
  // jsdom has no PointerEvent: mirror the heatmap suite's stub so React handlers see pointer data.
  class TestPointerEvent extends MouseEvent {
    pointerId: number;
    pointerType: string;
    constructor(type: string, init: PointerEventInit = {}) {
      super(type, init);
      this.pointerId = init.pointerId ?? 1;
      this.pointerType = init.pointerType ?? 'mouse';
    }
  }
  vi.stubGlobal('PointerEvent', TestPointerEvent);
};
const feedTrack = () => {
  const track = document.querySelector<HTMLElement>('.cm-news-feed-track');
  if (!track) throw new Error('news feed track not rendered');
  return track;
};

// The marquee stops by parking the track at its current offset (translate3d(-Npx, ...)).
it('clears the stale marquee offset when the filtered feed empties out', async () => {
  stubFlowEnvironment();
  state.feed.mockReturnValue(feedWith(cardItems(4)));
  const { rerender } = render(<NewsIntelligencePanel/>);
  const track = feedTrack();
  track.style.transform = leftoverMarqueeTransform;
  state.feed.mockReturnValue(feedWith([]));
  rerender(<NewsIntelligencePanel/>);
  await waitFor(() => expect(track.style.transform).toBe('translate3d(0px, 0px, 0px)'));
});

it('clears the stale marquee offset when the remaining headlines no longer overflow the rail', async () => {
  stubFlowEnvironment();
  state.feed.mockReturnValue(feedWith(cardItems(4)));
  const { rerender } = render(<NewsIntelligencePanel/>);
  const track = feedTrack();
  track.style.transform = leftoverMarqueeTransform;
  state.feed.mockReturnValue(feedWith(cardItems(2)));
  rerender(<NewsIntelligencePanel/>);
  await waitFor(() => expect(track.style.transform).toBe('translate3d(0px, 0px, 0px)'));
});

// Pointer capture on the rail retargets the click away from controls rendered inside it,
// which silently killed the transient-state buttons (reset/retry).
it('keeps transient-state buttons inside the rail clickable', async () => {
  stubFlowEnvironment();
  state.feed.mockReturnValue(feedWith([]));
  render(<NewsIntelligencePanel/>);
  fireEvent.change(screen.getByRole('searchbox', { name: 'Search headlines' }), { target: { value: 'no-match' } });
  const reset = screen.getByRole('button', { name: 'Show all headlines' });
  const rail = document.querySelector<HTMLElement>('.cm-news-feed');
  if (!rail) throw new Error('news rail not rendered');
  const capture = vi.fn();
  rail.setPointerCapture = capture;
  fireEvent.pointerDown(reset, { pointerId: 9, button: 0, pointerType: 'mouse' });
  expect(capture).not.toHaveBeenCalled();
  expect(rail.classList.contains('is-dragging')).toBe(false);
});

it('still captures the pointer when a drag starts on a headline card', async () => {
  stubFlowEnvironment();
  state.feed.mockReturnValue(feedWith(cardItems(4)));
  render(<NewsIntelligencePanel/>);
  const card = screen.getByRole('button', { name: /QA headline 1/ });
  const capture = vi.fn();
  card.setPointerCapture = capture;
  fireEvent.pointerDown(card, { pointerId: 9, button: 0, pointerType: 'mouse' });
  expect(capture).toHaveBeenCalledWith(9);
});

// The refresh status must sit inside the headline count row: as a separate block above it,
// appearing and disappearing shoved the marquee (and every card below it) down mid-refresh.
it('renders the refresh status inside the headline count row', () => {
  stubFlowEnvironment();
  state.feed.mockReturnValue({ ...feedWith(cardItems(4)), isFetching: true });
  render(<NewsIntelligencePanel/>);
  const status = screen.getByRole('status');
  expect(status).toHaveTextContent('Updating headlines');
  const row = document.querySelector('.cm-news-rail-tools');
  if (!row) throw new Error('headline count row not rendered');
  expect(row).toContainElement(status);
  const count = row.querySelector('p');
  if (!count) throw new Error('headline count line not rendered');
  expect(count.compareDocumentPosition(status) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0);
});
