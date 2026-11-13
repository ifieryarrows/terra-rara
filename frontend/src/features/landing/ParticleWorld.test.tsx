// @vitest-environment jsdom
import { act, cleanup, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { motionValue } from 'framer-motion';
import { MemoryRouter } from 'react-router-dom';
import { ParticleWorld } from './ParticleWorld';
import {
  CINEMATIC_ENTRY_FADE_FROM,
  CINEMATIC_ENTRY_RESET,
  CINEMATIC_ENTRY_TRIGGER,
  getCinematicEntryOpacity,
} from './cinematic-entry';
import { LandingPage } from './LandingPage';

function installViewportStubs(reducedMotion = false) {
  Object.defineProperty(window, 'innerWidth', { configurable: true, writable: true, value: 1440 });
  Object.defineProperty(window, 'innerHeight', { configurable: true, writable: true, value: 900 });
  HTMLCanvasElement.prototype.getBoundingClientRect = vi.fn().mockReturnValue({
    left: 0,
    top: 0,
    right: 1440,
    bottom: 900,
    width: 1440,
    height: 900,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  });
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: reducedMotion && query.includes('prefers-reduced-motion'),
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
  }));

  class MockIntersectionObserver {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
    constructor(cb?: IntersectionObserverCallback) {
      if (cb) {
        cb(
          [{ isIntersecting: true } as IntersectionObserverEntry],
          this as unknown as IntersectionObserver,
        );
      }
    }
  }
  vi.stubGlobal('IntersectionObserver', MockIntersectionObserver);

  class MockResizeObserver {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
  }
  vi.stubGlobal('ResizeObserver', MockResizeObserver);
}

describe('ParticleWorld Milky Way 3D perspective mouse tilt & landing reverse scroll CTA fade', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    installViewportStubs(false);
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('activates Canvas2D fallback at finale progress and updates Milky Way 3D projection when pointer moves across quadrants', () => {
    const arcCalls: Array<{ x: number; y: number; r: number }> = [];
    const mockCtx2D = {
      clearRect: vi.fn(),
      fillRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      arc: vi.fn((x: number, y: number, r: number) => {
        arcCalls.push({ x, y, r });
      }),
      fill: vi.fn(),
      stroke: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      setTransform: vi.fn(),
      translate: vi.fn(),
      scale: vi.fn(),
      rotate: vi.fn(),
      fillStyle: '',
      strokeStyle: '',
      globalCompositeOperation: 'source-over',
      globalAlpha: 1,
      createRadialGradient: vi.fn().mockReturnValue({ addColorStop: vi.fn() }),
    };

    const origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = vi.fn(function (
      this: HTMLCanvasElement,
      type: string,
    ) {
      if (type === 'webgl2' || type === 'webgl') return null;
      if (type === '2d') return mockCtx2D as unknown as CanvasRenderingContext2D;
      return origGetContext.call(this, type as '2d');
    }) as unknown as typeof HTMLCanvasElement.prototype.getContext;

    const progress = motionValue(0.96);
    const { container } = render(<ParticleWorld progress={progress} quality="balanced" />);

    const canvases = container.querySelectorAll('canvas');
    expect(canvases.length).toBe(2);
    const fallbackCanvas = canvases[1];
    expect(fallbackCanvas).toHaveAttribute('data-renderer', 'canvas2d');
    expect(fallbackCanvas).toHaveAttribute('data-quality', 'balanced');

    // Step initial centered animation frames
    act(() => {
      vi.advanceTimersByTime(120);
    });
    expect(arcCalls.length).toBeGreaterThan(20);

    // Move pointer to upper-left quadrant, settle pointer smoothing, and snapshot one draw pass
    act(() => {
      window.dispatchEvent(new MouseEvent('pointermove', { clientX: 60, clientY: 60 }));
      vi.advanceTimersByTime(450);
    });
    arcCalls.length = 0;
    act(() => {
      vi.advanceTimersByTime(24);
    });
    const topLeftPoints = arcCalls.slice(0, 60);
    expect(topLeftPoints.length).toBeGreaterThan(20);

    // Move pointer to lower-right quadrant so pitch/yaw/roll invert cleanly
    act(() => {
      window.dispatchEvent(new MouseEvent('pointermove', { clientX: 1380, clientY: 840 }));
      vi.advanceTimersByTime(450);
    });
    arcCalls.length = 0;
    act(() => {
      vi.advanceTimersByTime(24);
    });
    const bottomRightPoints = arcCalls.slice(0, 60);
    expect(bottomRightPoints.length).toBe(topLeftPoints.length);

    let meanDelta = 0;
    for (let i = 0; i < topLeftPoints.length; i += 1) {
      meanDelta += Math.hypot(
        bottomRightPoints[i].x - topLeftPoints[i].x,
        bottomRightPoints[i].y - topLeftPoints[i].y,
      );
    }
    meanDelta /= topLeftPoints.length;
    expect(meanDelta).toBeGreaterThan(8);
  });

  it('fades entry CTA smoothly on upward reverse scroll across [0.85, 0.93] instead of snapping', () => {
    expect(CINEMATIC_ENTRY_FADE_FROM).toBeLessThan(CINEMATIC_ENTRY_RESET);
    expect(CINEMATIC_ENTRY_RESET).toBeLessThan(CINEMATIC_ENTRY_TRIGGER);

    const downwardMax = getCinematicEntryOpacity(0.97);
    const midReveal = getCinematicEntryOpacity(0.91);
    const reverseAboveReset = getCinematicEntryOpacity(0.885);
    const reverseAtFadeStart = getCinematicEntryOpacity(0.865);
    const hiddenFloor = getCinematicEntryOpacity(0.85);

    expect(downwardMax).toBe(1);
    expect(midReveal).toBeGreaterThan(0.5);
    expect(midReveal).toBeLessThan(1);
    expect(reverseAboveReset).toBeGreaterThan(0.15);
    expect(reverseAboveReset).toBeLessThan(midReveal);
    expect(reverseAtFadeStart).toBeGreaterThan(0);
    expect(reverseAtFadeStart).toBeLessThan(reverseAboveReset);
    expect(hiddenFloor).toBe(0);
  });

  it('renders LandingPage hero Venus copy & Enter CopperMind CTA cleanly under reduced motion', () => {
    installViewportStubs(true);
    HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(null);

    render(
      <MemoryRouter>
        <LandingPage />
      </MemoryRouter>,
    );

    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Read the market.');
    const links = screen.getAllByRole('link', { name: /Enter CopperMind/i });
    expect(links.length).toBeGreaterThanOrEqual(1);
    expect(links[0]).toHaveAttribute('href', '/dashboard');
  });
});
