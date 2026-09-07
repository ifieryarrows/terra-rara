// @vitest-environment jsdom
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import type { MotionValue } from 'framer-motion';
import { chapterAt, sceneProgress, sceneVisibility } from './story-timeline';
import { ResearchStory } from './ResearchStory';
const state = vi.hoisted(() => ({ scroll: null as MotionValue<number> | null }));
vi.mock('framer-motion', async importOriginal => {
  const actual = await importOriginal<typeof import('framer-motion')>();
  state.scroll = actual.motionValue(0);
  return { ...actual, useScroll: () => ({ scrollYProgress: state.scroll }) };
});
beforeEach(() => { state.scroll!.set(0); });
afterEach(cleanup);

describe('readable, reversible research story', () => {
  it('never blends two readings at any sampled forward or reverse position', () => {
    const samples = Array.from({ length: 1001 }, (_, i) => i / 1000);
    for (const p of [...samples, ...samples.reverse()]) {
      const visible = [0, 1, 2].filter(i => sceneVisibility(p, i) > 0);
      expect(visible.length).toBeLessThanOrEqual(1);
      if (visible.length) expect(visible[0]).toBe(chapterAt(p));
    }
  });
  it('allows a completed scene to remain readable over a meaningful scroll interval', () => {
    for (const [i, start, end] of [[0, .12, .24], [1, .43, .64], [2, .83, 1]]) {
      expect(sceneProgress(start, i)).toBeCloseTo(1);
      expect(sceneVisibility(start, i)).toBeCloseTo(1);
      expect(sceneVisibility(end, i)).toBeCloseTo(1);
      expect(end - start).toBeGreaterThanOrEqual(.12);
    }
  });
  it('updates the current chapter and removes inactive interactive readings', async () => {
    render(<MemoryRouter><ResearchStory enhanced/></MemoryRouter>);
    expect(screen.getByRole('link', { name: '01 Market' })).toHaveAttribute('aria-current', 'step');
    act(() => state.scroll!.set(.5));
    expect(screen.getByRole('link', { name: '02 Context' })).toHaveAttribute('aria-current', 'step');
    expect(screen.queryByText('Why these connections?')).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Read the reasoning')).toBeVisible());
    act(() => state.scroll!.set(0));
    await waitFor(() => expect(screen.getByText('Why these connections?')).toBeVisible());
    expect(screen.queryByText('Read the reasoning')).not.toBeInTheDocument();
  });
  it('preserves a focused disclosure until keyboard focus leaves the reading', async () => {
    render(<MemoryRouter><ResearchStory enhanced/><button>Outside the story</button></MemoryRouter>);
    act(() => state.scroll!.set(.5));
    await userEvent.click(screen.getByText('Read the reasoning'));
    act(() => state.scroll!.set(.9));
    expect(screen.getByText('Read the reasoning')).toHaveFocus();
    expect(screen.getByRole('link', { name: '02 Context' })).toHaveAttribute('aria-current', 'step');
    await userEvent.tab();
    expect(screen.getByRole('button', { name: 'Outside the story' })).toHaveFocus();
    expect(screen.queryByText('Read the reasoning')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: '03 Possibilities' })).toHaveAttribute('aria-current', 'step');
  });
  it('keeps every scene and explanation available in the static experience', async () => {
    render(<MemoryRouter><ResearchStory enhanced={false}/></MemoryRouter>);
    expect(screen.getAllByRole('img')).toHaveLength(3);
    await userEvent.click(screen.getByText('Why these connections?'));
    expect(screen.getByText(/These links do not represent measured correlations/)).toBeVisible();
    expect(screen.getByText('Beyond the last observation.')).toBeVisible();
  });
});
