// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';
import { useMotionValue } from 'framer-motion';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CinematicLanding } from './CinematicLanding';
import { CosmicHud } from './CosmicHud';

function HudHarness({ initialProgress = 0, onStep = vi.fn() }: { initialProgress?: number; onStep?: (target: number) => void }) {
  const progress = useMotionValue(initialProgress);
  return <CosmicHud progress={progress} onStepToProgress={onStep} />;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('CosmicHud observatory scale ruler and sensory controls', () => {
  it('renders SFX toggle without ambient audio banner, logarithmic scale ruler notch readout, and clean target reticle without Red Light or CosmicLens branding', () => {
    render(<HudHarness initialProgress={0.74} />);
    expect(screen.getByRole('toolbar', { name: 'Observatory sensory controls' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Observatory sound effects on/i })).toBeInTheDocument();
    expect(screen.queryByText(/COSMIC LENS/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /red light/i })).not.toBeInTheDocument();
    expect(screen.getByText('0.72')).toBeInTheDocument();
    expect(screen.getByText('AU')).toBeInTheDocument();
    expect(screen.getByText('0.72 AU')).toBeInTheDocument();
  });

  it('tracks orbital transfer from 0.72 AU through Venus flyby into the Milky Way finale scale', () => {
    render(<HudHarness initialProgress={0.74} />);
    expect(screen.getByText('0.72')).toBeInTheDocument();
    expect(screen.getByText('AU')).toBeInTheDocument();
    expect(screen.getByText('VENUS · 2D')).toBeInTheDocument();
    expect(screen.getByText('MILKY WAY')).toBeInTheDocument();

    cleanup();
    render(<HudHarness initialProgress={0.80} />);
    expect(screen.getByText('149,598')).toBeInTheDocument();
    expect(screen.getByText('KM')).toBeInTheDocument();
    expect(screen.getByText('VENUS FLYBY')).toBeInTheDocument();

    cleanup();
    render(<HudHarness initialProgress={0.835} />);
    expect(screen.getByText('12,104')).toBeInTheDocument();
    expect(screen.getByText('KM')).toBeInTheDocument();

    cleanup();
    render(<HudHarness initialProgress={0.86} />);
    expect(screen.getByText('26')).toBeInTheDocument();
    expect(screen.getAllByText('THOUSAND LY').length).toBeGreaterThanOrEqual(1);

    cleanup();
    render(<HudHarness initialProgress={1.0} />);
    expect(screen.getByText('105')).toBeInTheDocument();
    expect(screen.getAllByText('THOUSAND LY').length).toBeGreaterThanOrEqual(1);
  });

  it('toggles landing UI sound effects without fetching background ambient song', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    const user = userEvent.setup();
    render(<HudHarness initialProgress={0} />);
    const soundBtn = screen.getByRole('button', { name: /Observatory sound effects on/i });
    expect(soundBtn).toHaveAttribute('aria-pressed', 'true');
    expect(soundBtn).toHaveTextContent('SOUND: ON');

    await user.click(soundBtn);
    expect(soundBtn).toHaveAttribute('aria-pressed', 'false');
    expect(soundBtn).toHaveTextContent('SOUND: OFF');

    await user.click(soundBtn);
    expect(soundBtn).toHaveAttribute('aria-pressed', 'true');
    expect(soundBtn).toHaveTextContent('SOUND: ON');

    expect(
      fetchSpy.mock.calls.some(([url]) => String(url).includes('milky-way-observatory')),
    ).toBe(false);
  });

  it('advances to the next storyboard milestone when clicking the target reticle (+)', async () => {
    const user = userEvent.setup();
    const onStep = vi.fn();
    render(<HudHarness initialProgress={0} onStep={onStep} />);
    await user.click(screen.getByRole('button', { name: 'Advance to next observatory milestone' }));
    expect(onStep).toHaveBeenCalledWith(0.24);

    cleanup();
    const onFlybyStep = vi.fn();
    render(<HudHarness initialProgress={0.69} onStep={onFlybyStep} />);
    await user.click(screen.getByRole('button', { name: 'Advance to next observatory milestone' }));
    expect(onFlybyStep).toHaveBeenCalledWith(0.79);
  });

  it('keeps the sound toggle above the finale entry overlay rather than inside the sticky world stacking context', async () => {
    vi.stubGlobal('matchMedia', (query: string) => ({
      matches: false,
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
    }));
    vi.stubGlobal('IntersectionObserver', class { observe = vi.fn(); unobserve = vi.fn(); disconnect = vi.fn(); });
    vi.stubGlobal('ResizeObserver', class { observe = vi.fn(); unobserve = vi.fn(); disconnect = vi.fn(); });
    HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(null);

    const { container } = render(
      <MemoryRouter>
        <CinematicLanding
          quality="balanced"
          entry={() => (
            <section className="cm-enter cm-enter--cinematic cm-cinematic-entry" style={{ pointerEvents: 'auto' }}>
              <button type="button">Enter CopperMind</button>
            </section>
          )}
        />
      </MemoryRouter>,
    );

    const stickyWorld = container.querySelector('.cm-cinematic-sticky');
    const entryLayer = container.querySelector('.cm-cinematic-entry-layer');
    const hudContainer = container.querySelector('.cm-cosmic-hud-container');
    expect(stickyWorld).not.toBeNull();
    expect(entryLayer).not.toBeNull();
    expect(hudContainer).not.toBeNull();
    expect(stickyWorld!.contains(hudContainer)).toBe(false);
    expect(entryLayer!.compareDocumentPosition(hudContainer!) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0);

    const user = userEvent.setup();
    const soundBtn = screen.getByRole('button', { name: /Observatory sound effects on/i });
    await user.click(soundBtn);
    expect(soundBtn).toHaveTextContent('SOUND: OFF');
    await user.click(soundBtn);
    expect(soundBtn).toHaveTextContent('SOUND: ON');
  });

  it('leaves wheel scrolling native and tags macOS cinematic scenes while keeping Windows wheel stepping active', () => {
    vi.stubGlobal('matchMedia', (query: string) => ({
      matches: false,
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
    }));
    vi.stubGlobal('IntersectionObserver', class { observe = vi.fn(); unobserve = vi.fn(); disconnect = vi.fn(); });
    vi.stubGlobal('ResizeObserver', class { observe = vi.fn(); unobserve = vi.fn(); disconnect = vi.fn(); });
    HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(null);
    Object.defineProperty(window, 'scrollTo', { configurable: true, value: vi.fn() });
    Object.defineProperty(document.documentElement, 'scrollHeight', { configurable: true, value: 5_000 });
    Object.defineProperty(window, 'scrollY', { configurable: true, value: 0, writable: true });
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      top: 0, bottom: 4000, left: 0, right: 1280, width: 1280, height: 4000, x: 0, y: 0, toJSON: () => ({}),
    });

    const navSpy = vi.spyOn(window, 'navigator', 'get').mockReturnValue({
      ...window.navigator,
      platform: 'MacIntel',
      userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    });

    const { container: macContainer, unmount: unmountMac } = render(
      <MemoryRouter>
        <CinematicLanding quality="balanced" />
      </MemoryRouter>,
    );
    const macScene = macContainer.querySelector('#research')!;
    expect(macScene).toHaveClass('cm-cinematic--mac');
    const macWheel = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 120 });
    macScene.dispatchEvent(macWheel);
    expect(macWheel.defaultPrevented).toBe(false);
    unmountMac();

    navSpy.mockReturnValue({
      ...window.navigator,
      platform: 'Win32',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    });
    const { container: winContainer } = render(
      <MemoryRouter>
        <CinematicLanding quality="balanced" />
      </MemoryRouter>,
    );
    const winScene = winContainer.querySelector('#research')!;
    expect(winScene).not.toHaveClass('cm-cinematic--mac');
    const winWheel = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 120 });
    winScene.dispatchEvent(winWheel);
    expect(winWheel.defaultPrevented).toBe(true);
  });
});
