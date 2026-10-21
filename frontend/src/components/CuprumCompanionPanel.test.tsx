// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CuprumCompanionPanel } from './CuprumCompanionPanel';
import type { CommentaryResponse } from '../types';

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

const COMMENTARY_FIXTURE: CommentaryResponse = {
  symbol: 'HG=F',
  commentary: [
    'RISKS (2 factors):',
    '• Dollar rebound could cap upside momentum.',
    '• Smelter restocking remains uneven.',
    'OPPORTUNITIES (1 factor):',
    '• Warehouse draws tighten prompt COMEX spreads.',
    'SYNTHESIS:',
    'Copper futures maintain a constructive 5-day posture supported by tight spreads.',
    'This is NOT financial advice.',
  ].join('\n'),
  error: null,
  generated_at: '2026-10-16T21:05:00Z',
  ai_stance: 'BULLISH',
  generation_mode: 'llm',
  model_name: 'nvidia/nemotron-3-super-120b-a12b:free',
  sections: {
    synthesis:
      'Copper futures maintain a constructive 5-day posture supported by tight spreads and Chinese equity momentum.',
    risks: [
      'Dollar rebound could cap upside momentum.',
      'Smelter restocking remains uneven.',
    ],
    opportunities: ['Warehouse draws tighten prompt COMEX spreads.'],
    xai_brief: null,
    bias_warning: 'Subject to macro volatility around upcoming PMI releases.',
  },
  companion_state: {
    character: 'Cuprum',
    mood: 'molten_bull',
    stance: 'BULLISH',
    energy: 'charged',
    confidence_band: 'high',
    headline_quote: 'Copper looks constructive near 5.10.',
    xai_highlights: {
      top_signals: [
        {
          feature: 'DXY_level',
          weight: 0.31,
          label: 'Dollar index',
          description: 'A stronger dollar has historically capped copper upside.',
          category: 'Macro',
          time_horizon: '5d',
        },
      ],
      memory_anchors: [{ date: '2026-09-16', weight: 0.033 }],
    },
  },
};

describe('CuprumCompanionPanel', () => {
  it('renders the living pet and opens a three-choice dialogue tree on click', () => {
    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);

    expect(screen.getByTestId('cuprum-companion-stage')).toBeInTheDocument();
    expect(screen.getByText('Molten Resonance')).toBeInTheDocument();
    expect(screen.getByText('humming')).toBeInTheDocument();
    expect(screen.queryByRole('tab')).not.toBeInTheDocument();
    expect(screen.queryByText(/CUPRUM EXECUTIVE THESIS/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));

    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent(
      /You caught me mid-roar/,
    );
    const choices = screen.getByRole('group', { name: 'Your reply' });
    expect(choices.querySelectorAll('button')).toHaveLength(3);

    fireEvent.click(screen.getByRole('button', { name: "How's your spark today?" }));
    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent(
      'Loud. The whole board is warm',
    );

    fireEvent.click(screen.getByRole('button', { name: "What's weighing on you?" }));
    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent(
      'The dollar keeps tapping on the window',
    );

    fireEvent.click(screen.getByRole('button', { name: 'Is that a warning?' }));
    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent(
      'Warnings have footnotes.',
    );
  });

  it('answers the market-weather branch from the real risk balance without numbers', () => {
    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    fireEvent.click(screen.getByRole('button', { name: 'What do you sense out there?' }));

    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent('the paper weather');
    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent(
      'More knots than doors today.',
    );
    expect(screen.getByTestId('cuprum-spoken-text')).not.toHaveTextContent('31');
    expect(screen.getByTestId('cuprum-spoken-text')).not.toHaveTextContent('%');
  });

  it('reveals ambient bubbles on their own, but never shows choices until the pet is clicked', async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, 'random').mockReturnValue(0);

    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);

    expect(screen.queryByTestId('cuprum-idle-bubble')).not.toBeInTheDocument();
    expect(screen.queryByRole('group', { name: 'Your reply' })).not.toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(6700);
    });

    expect(screen.getByTestId('cuprum-idle-bubble')).toBeInTheDocument();
    expect(screen.queryByRole('group', { name: 'Your reply' })).not.toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2200);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(8000);
    });

    expect(screen.queryByTestId('cuprum-idle-bubble')).not.toBeInTheDocument();
  });

  it('keeps ambient chatter quiet while a conversation is open', async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, 'random').mockReturnValue(0);

    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);
    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60000);
    });

    expect(screen.getByTestId('cuprum-dialogue-bubble')).toBeInTheDocument();
    expect(screen.queryByTestId('cuprum-idle-bubble')).not.toBeInTheDocument();
  });

  it('closes the conversation with Escape or the pet again', () => {
    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    expect(screen.getByTestId('cuprum-dialogue-bubble')).toBeInTheDocument();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByTestId('cuprum-dialogue-bubble')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    fireEvent.click(screen.getByRole('button', { name: 'Close Cuprum conversation' }));
    expect(screen.queryByTestId('cuprum-dialogue-bubble')).not.toBeInTheDocument();
  });

  it('keeps the conversation open when the dashboard is clicked outside the stage', () => {
    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    expect(screen.getByTestId('cuprum-dialogue-bubble')).toBeInTheDocument();

    fireEvent.pointerDown(document.body);
    expect(screen.getByTestId('cuprum-dialogue-bubble')).toBeInTheDocument();
  });

  it('does not start ambient chatter when dialogue opens while an idle bubble is visible', async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, 'random').mockReturnValue(0);

    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(6700);
    });
    expect(screen.getByTestId('cuprum-idle-bubble')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    expect(screen.queryByTestId('cuprum-idle-bubble')).not.toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60000);
    });
    expect(screen.getByTestId('cuprum-dialogue-bubble')).toBeInTheDocument();
    expect(screen.queryByTestId('cuprum-idle-bubble')).not.toBeInTheDocument();
  });

  it('moves focus without asking the browser to scroll', async () => {
    const focusSpy = vi.spyOn(HTMLElement.prototype, 'focus');

    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);
    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));

    await waitFor(() => {
      expect(focusSpy).toHaveBeenCalledWith({ preventScroll: true });
    });

    fireEvent.keyDown(document, { key: 'Escape' });
    await waitFor(() => {
      expect(focusSpy.mock.calls.filter(([options]) => options?.preventScroll === true)).toHaveLength(2);
    });
  });

  it('still reports a fresh snapshot after the reader already picked a reply', () => {
    const { rerender } = render(
      <CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />,
    );

    // A new pipeline snapshot lands while the card is on screen.
    rerender(
      <CuprumCompanionPanel
        commentary={{ ...COMMENTARY_FIXTURE, generated_at: '2026-10-17T21:05:00Z' }}
        explain={null}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent('wind changed');

    fireEvent.click(screen.getByRole('button', { name: 'What do you sense out there?' }));
    fireEvent.click(screen.getByRole('button', { name: "What's the loose thread?" }));
    fireEvent.click(screen.getByRole('button', { name: 'What worries you?' }));
    fireEvent.click(screen.getByRole('button', { name: 'What changed?' }));

    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent('wind changed');
  });

  it('keeps the same bubble frame while the reader moves through the tree', () => {
    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);
    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));

    const frame = screen.getByTestId('cuprum-dialogue-bubble');
    fireEvent.click(screen.getByRole('button', { name: 'What do you sense out there?' }));
    expect(screen.getByTestId('cuprum-dialogue-bubble')).toBe(frame);

    fireEvent.click(screen.getByRole('button', { name: 'Tell me more.' }));
    expect(screen.getByTestId('cuprum-dialogue-bubble')).toBe(frame);
  });

  it('keeps the reply choices outside the scrolling copy so all three stay reachable', () => {
    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);
    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));

    const bubble = screen.getByTestId('cuprum-dialogue-bubble');
    const copy = bubble.querySelector('.cm-cuprum-copy');
    const choices = screen.getByRole('group', { name: 'Your reply' });

    expect(copy).not.toBeNull();
    expect(copy).toContainElement(bubble.querySelector('.cm-cuprum-speech'));
    expect(copy).not.toContainElement(choices);
    expect(bubble).toContainElement(choices);
  });

  it('shows the talk frame while the pet is speaking and returns to the idle sprite', async () => {
    const { container } = render(
      <CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />,
    );
    const image = container.querySelector('[data-testid="cuprum-companion-stage"] img');

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));

    await waitFor(() => {
      expect(image?.getAttribute('src')).toContain('-talk.');
    });
  });

  it('keeps the pet interactive when every data source is empty', () => {
    render(<CuprumCompanionPanel commentary={null} explain={null} />);

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent(
      /The board is quiet/,
    );
    expect(screen.getByRole('group', { name: 'Your reply' })).toBeInTheDocument();
  });

  it('uses the story branch as a loop and advances to the next memory', () => {
    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    fireEvent.click(screen.getByRole('button', { name: "Tell me something I don't know." }));
    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent('first smiths');

    fireEvent.click(screen.getByRole('button', { name: 'Tell me another.' }));
    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent('Statue of Liberty');
  });
});
