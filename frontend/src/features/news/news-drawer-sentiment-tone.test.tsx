// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, expect, it, vi } from 'vitest';
import { MotionConfig } from 'framer-motion';
import { NewsDetailDrawer } from './NewsDetailDrawer';
import type { NewsItem } from '../../types';

vi.mock('../../hooks/useNews', () => ({ useNewsDetail: () => ({ data: null }) }));
afterEach(cleanup);

function renderArticle(sentiment: Record<string, unknown>) {
  const item = {
    id: 1,
    title: 'Sample source story',
    channel: 'market',
    url: 'https://example.com/story',
    sentiment,
  } as unknown as NewsItem;

  render(
    <MotionConfig reducedMotion="always" transition={{ duration: 0 }}>
      <NewsDetailDrawer item={item} onClose={() => {}} />
    </MotionConfig>,
  );
}

it('does not colour a balanced score by its sign', () => {
  renderArticle({
    label: 'NEUTRAL',
    final_score: -0.376,
    impact_score_llm: -0.12,
    confidence: 0.5,
    relevance: 0.4,
    event_type: 'mixed_unclear',
  });

  const score = screen.getByText('-0.376');
  expect(score).toBeInTheDocument();
  expect(score.className).not.toMatch(/rose|emerald/);
  expect(screen.getByText('NEUTRAL')).toBeInTheDocument();
});

it('colours a bearish score with the bearish tone', () => {
  renderArticle({
    label: 'BEARISH',
    final_score: -0.376,
    impact_score_llm: -0.12,
    confidence: 0.5,
    relevance: 0.4,
    event_type: 'mixed_unclear',
  });

  expect(screen.getByText('-0.376').className).toMatch(/rose/);
});

it('colours a bullish score with the bullish tone', () => {
  renderArticle({
    label: 'BULLISH',
    final_score: 0.412,
    impact_score_llm: 0.22,
    confidence: 0.7,
    relevance: 0.6,
    event_type: 'supply_tightening',
  });

  expect(screen.getByText('+0.412').className).toMatch(/emerald/);
});
