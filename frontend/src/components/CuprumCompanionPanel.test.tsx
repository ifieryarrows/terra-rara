// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CuprumCompanionPanel } from './CuprumCompanionPanel';
import { computeBubblePlacement } from './CuprumSpeechBubble';
import {
  CuprumProvider,
  CUPRUM_KEY_STEP_PX,
  CUPRUM_SAFE_MARGIN,
  CUPRUM_STORAGE_KEYS,
  clampCuprumPosition,
} from '../contexts/CuprumContext';
import * as cosmicVision from '../utils/cosmicVision';
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
  it('renders the living pet without patina badge and opens a three-choice dialogue tree on click', () => {
    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);

    expect(screen.getByTestId('cuprum-companion-stage')).toBeInTheDocument();
    expect(document.querySelector('.cm-cuprum-identity')).not.toBeInTheDocument();
    expect(screen.queryByTestId('cuprum-patina-status')).not.toBeInTheDocument();
    expect(screen.queryByTestId('cuprum-patina-toggle')).not.toBeInTheDocument();
    expect(document.querySelector('.cm-cuprum-action-bar .cm-cuprum-controls')).toBeInTheDocument();
    expect(screen.queryByRole('tab')).not.toBeInTheDocument();
    expect(screen.queryByText(/CUPRUM EXECUTIVE THESIS/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));

    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent(
      /You caught me mid-roar/,
    );
    const choices = screen.getByRole('group', { name: 'Your reply' });
    expect(choices.querySelectorAll('button')).toHaveLength(4);

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

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    fireEvent.click(screen.getByRole('button', { name: 'Close conversation' }));
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
    fireEvent.click(screen.getByRole('button', { name: "What's pulling at the wires?" }));
    fireEvent.click(screen.getByRole('button', { name: 'The loudest one?' }));
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

  it('does not fast-forward typing when clicking the speech bubble in dialogue mode', () => {
    const { container } = render(
      <CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));

    const bubble = screen.getByTestId('cuprum-dialogue-bubble');
    fireEvent.click(bubble);

    // In dialogue mode, typing is not cancelled by clicking the bubble background
    const caret = container.querySelector('.cm-cuprum-caret');
    expect(caret).toBeInTheDocument();
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

  it('navigates via number keys including key 4 and allows returning to root seamlessly', () => {
    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);

    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    const bubble = screen.getByTestId('cuprum-dialogue-bubble');

    // Press key '4' (Choice 4: Tell me something I don't know.)
    fireEvent.keyDown(bubble, { key: '4' });
    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent('first smiths');

    // Select 'Why that one?'
    fireEvent.click(screen.getByRole('button', { name: 'Why that one?' }));
    expect(screen.getByRole('button', { name: 'Let me ask something else.' })).toBeInTheDocument();

    // Return to root via 'Let me ask something else.'
    fireEvent.click(screen.getByRole('button', { name: 'Let me ask something else.' }));
    expect(screen.getByTestId('cuprum-spoken-text')).not.toHaveTextContent('You came back');
    expect(screen.getByTestId('cuprum-spoken-text')).toHaveTextContent('The fire is still hot. What else do you want to ask?');
  });
});

describe('CuprumCompanionPanel draggable dock, clamping, keyboard control & edge-aware bubble', () => {
  const originalPointerEvent = window.PointerEvent;

  beforeEach(() => {
    window.localStorage.clear();
    Object.defineProperty(window, 'innerWidth', { configurable: true, writable: true, value: 1280 });
    Object.defineProperty(window, 'innerHeight', { configurable: true, writable: true, value: 800 });
    if (typeof window.PointerEvent === 'undefined') {
      class JsdomPointerEvent extends MouseEvent {
        readonly pointerId: number;
        constructor(type: string, params: PointerEventInit = {}) {
          super(type, params);
          this.pointerId = params.pointerId ?? 1;
        }
      }
      window.PointerEvent = JsdomPointerEvent as unknown as typeof PointerEvent;
    }
  });

  afterEach(() => {
    window.localStorage.clear();
    if (originalPointerEvent === undefined) {
      delete (window as { PointerEvent?: typeof PointerEvent }).PointerEvent;
    } else {
      window.PointerEvent = originalPointerEvent;
    }
  });

  it('clamps positions within safe viewport bounds and falls back cleanly', () => {
    const size = { width: 112, height: 122 };
    const viewport = { width: 1000, height: 700 };

    expect(clampCuprumPosition({ x: -50, y: -30 }, size, viewport)).toEqual({
      x: CUPRUM_SAFE_MARGIN,
      y: CUPRUM_SAFE_MARGIN,
    });
    expect(clampCuprumPosition({ x: 5000, y: 5000 }, size, viewport)).toEqual({
      x: 1000 - 112 - CUPRUM_SAFE_MARGIN,
      y: 700 - 122 - CUPRUM_SAFE_MARGIN,
    });
    expect(clampCuprumPosition({ x: 240, y: 310 }, size, viewport)).toEqual({ x: 240, y: 310 });
  });

  it('flips speech bubble below and right when docked near the top-left corner', () => {
    const topLeft = computeBubblePlacement(
      { x: 16, y: 16, width: 112, height: 122 },
      { width: 1280, height: 800 },
      'dialogue',
    );
    expect(topLeft.vertical).toBe('below');
    expect(topLeft.horizontal).toBe('right');
    expect(topLeft.offsetX).toBeGreaterThanOrEqual(0);

    const bottomRight = computeBubblePlacement(
      { x: 1140, y: 650, width: 112, height: 122 },
      { width: 1280, height: 800 },
      'dialogue',
    );
    expect(bottomRight.vertical).toBe('above');
    expect(bottomRight.horizontal).toBe('left');
    expect(bottomRight.offsetX).toBeLessThan(0);
  });

  it('supports >= 4px pointer dragging with clamping & persistence while preserving < 4px click', async () => {
    render(
      <CuprumProvider isolated>
        <CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />
      </CuprumProvider>,
    );

    const stage = screen.getByTestId('cuprum-companion-stage');
    const startX = Number(stage.getAttribute('data-pos-x'));
    const startY = Number(stage.getAttribute('data-pos-y'));

    // Tiny micro-jitter (< 4px) on pet button must NOT drag or suppress click
    const petBtn = screen.getByRole('button', { name: 'Talk to Cuprum' });
    fireEvent.pointerDown(petBtn, { button: 0, pointerId: 1, clientX: startX + 20, clientY: startY + 20 });
    fireEvent.pointerMove(petBtn, { pointerId: 1, clientX: startX + 22, clientY: startY + 21 });
    fireEvent.pointerUp(petBtn, { pointerId: 1, clientX: startX + 22, clientY: startY + 21 });
    expect(stage).toHaveAttribute('data-dragging', 'false');
    expect(Number(stage.getAttribute('data-pos-x'))).toBe(startX);

    // Real pointer click opens dialogue cleanly
    fireEvent.click(petBtn);
    expect(screen.getByTestId('cuprum-dialogue-bubble')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close Cuprum conversation' }));

    // Drag via grip button beyond viewport top-left should clamp to safe margin (16, 16)
    const grip = screen.getByRole('button', { name: 'Reposition Cuprum pet. Use arrow keys to move.' });
    fireEvent.pointerDown(grip, { button: 0, pointerId: 2, clientX: startX + 10, clientY: startY + 10 });
    fireEvent.pointerMove(grip, { pointerId: 2, clientX: -400, clientY: -400 });
    expect(stage).toHaveAttribute('data-dragging', 'true');
    fireEvent.pointerUp(grip, { pointerId: 2, clientX: -400, clientY: -400 });

    expect(stage).toHaveAttribute('data-dragging', 'false');
    expect(stage).toHaveAttribute('data-pos-x', String(CUPRUM_SAFE_MARGIN));
    expect(stage).toHaveAttribute('data-pos-y', String(CUPRUM_SAFE_MARGIN));
    expect(screen.getByTestId('cuprum-position-announcement')).toHaveTextContent(
      `Cuprum moved to left ${CUPRUM_SAFE_MARGIN}, top ${CUPRUM_SAFE_MARGIN}.`,
    );

    // Persisted position in localStorage includes docked: false flag
    const saved = JSON.parse(window.localStorage.getItem(CUPRUM_STORAGE_KEYS.position) || '{}');
    expect(saved).toEqual({ x: CUPRUM_SAFE_MARGIN, y: CUPRUM_SAFE_MARGIN, docked: false });

    // Canceling an active drag with Escape preserves previous position
    fireEvent.pointerDown(grip, { button: 0, pointerId: 3, clientX: 40, clientY: 40 });
    fireEvent.pointerMove(grip, { pointerId: 3, clientX: 280, clientY: 280 });
    expect(stage).toHaveAttribute('data-dragging', 'true');
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(stage).toHaveAttribute('data-dragging', 'false');
    expect(stage).toHaveAttribute('data-pos-x', String(CUPRUM_SAFE_MARGIN));
    expect(screen.getByTestId('cuprum-position-announcement')).toHaveTextContent('Cuprum move canceled.');

    // Next macrotask clears drag click-suppression lock so pet click opens dialogue normally
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 15));
    });

    // Opening dialogue while clamped top-left renders bubble below & open-right
    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    const bubble = screen.getByTestId('cuprum-dialogue-bubble');
    expect(bubble).toHaveAttribute('data-placement-vertical', 'below');
    expect(bubble).toHaveAttribute('data-placement-horizontal', 'right');
  });

  it('moves in 15px arrow increments on the grip handle, cancels drag on Escape, resets on Esc, and toggles pill', () => {
    render(
      <CuprumProvider isolated>
        <CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />
      </CuprumProvider>,
    );

    const stage = screen.getByTestId('cuprum-companion-stage');
    const initialX = Number(stage.getAttribute('data-pos-x'));
    const initialY = Number(stage.getAttribute('data-pos-y'));
    const grip = screen.getByRole('button', { name: 'Reposition Cuprum pet. Use arrow keys to move.' });

    fireEvent.keyDown(grip, { key: 'ArrowLeft' });
    fireEvent.keyDown(grip, { key: 'ArrowUp' });
    expect(stage).toHaveAttribute('data-pos-x', String(initialX - CUPRUM_KEY_STEP_PX));
    expect(stage).toHaveAttribute('data-pos-y', String(initialY - CUPRUM_KEY_STEP_PX));
    expect(screen.getByTestId('cuprum-position-announcement')).toHaveTextContent(
      `Cuprum moved to left ${initialX - CUPRUM_KEY_STEP_PX}, top ${initialY - CUPRUM_KEY_STEP_PX}.`,
    );

    // Escape while idle resets back to docked corner
    fireEvent.keyDown(grip, { key: 'Escape' });
    expect(stage).toHaveAttribute('data-pos-x', String(initialX));
    expect(stage).toHaveAttribute('data-pos-y', String(initialY));
    expect(screen.getByTestId('cuprum-position-announcement')).toHaveTextContent(
      'Cuprum reset to bottom-right corner.',
    );

    // Shrink viewport & dispatch resize to assert live re-clamping
    act(() => {
      Object.defineProperty(window, 'innerWidth', { configurable: true, writable: true, value: 480 });
      Object.defineProperty(window, 'innerHeight', { configurable: true, writable: true, value: 500 });
      window.dispatchEvent(new Event('resize'));
    });
    expect(Number(stage.getAttribute('data-pos-x'))).toBeLessThanOrEqual(480 - 112 - CUPRUM_SAFE_MARGIN);
    expect(Number(stage.getAttribute('data-pos-y'))).toBeLessThanOrEqual(500 - 122 - CUPRUM_SAFE_MARGIN);

    // Minimize into pill and expand back
    fireEvent.click(screen.getByRole('button', { name: 'Minimize Cuprum' }));
    expect(stage).toHaveAttribute('data-minimized', 'true');
    expect(screen.getByTestId('cuprum-minimized-pill')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Expand Cuprum' }));
    expect(stage).toHaveAttribute('data-minimized', 'false');
    expect(screen.queryByTestId('cuprum-minimized-pill')).not.toBeInTheDocument();
  });

  it('omits halo rings & patina bar, aligns inline controls, plays mimimimi speak blips while typing, and splits pet vs button audio', () => {
    vi.useFakeTimers();
    const chirpSpy = vi.spyOn(cosmicVision, 'playCuprumChirp').mockResolvedValue(true);
    const tapSpy = vi.spyOn(cosmicVision, 'playCuprumTap').mockResolvedValue(true);
    const speakSpy = vi.spyOn(cosmicVision, 'playCuprumSpeakBlip').mockResolvedValue(true);
    const chimeSpy = vi.spyOn(cosmicVision, 'playObservatoryChime').mockResolvedValue(true);

    render(<CuprumCompanionPanel commentary={COMMENTARY_FIXTURE} explain={null} />);

    // No halo SVG or patina section
    expect(document.querySelector('.cm-cuprum-halo')).not.toBeInTheDocument();
    expect(document.querySelector('.cm-cuprum-identity')).not.toBeInTheDocument();
    expect(screen.queryByTestId('cuprum-patina-status')).not.toBeInTheDocument();
    expect(document.querySelector('.cm-cuprum-action-bar .cm-cuprum-controls')).toBeInTheDocument();

    // Clicking pet plays playCuprumChirp (never playObservatoryChime or playCuprumTap)
    fireEvent.click(screen.getByRole('button', { name: 'Talk to Cuprum' }));
    expect(chirpSpy).toHaveBeenCalledTimes(1);
    expect(tapSpy).not.toHaveBeenCalled();
    expect(chimeSpy).not.toHaveBeenCalled();

    // Advancing typewriter ticks emits Cuprum's mimimimi speech blips
    act(() => {
      vi.advanceTimersByTime(160);
    });
    expect(speakSpy).toHaveBeenCalled();

    // Clicking a dialogue reply button, dismiss X, or minimize/expand plays playCuprumTap
    fireEvent.click(screen.getByRole('button', { name: "How's your spark today?" }));
    expect(tapSpy).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole('button', { name: 'Close conversation' }));
    expect(tapSpy).toHaveBeenCalledTimes(2);

    fireEvent.click(screen.getByRole('button', { name: 'Minimize Cuprum' }));
    expect(tapSpy).toHaveBeenCalledTimes(3);
    expect(chirpSpy).toHaveBeenCalledTimes(1);
  });
});
