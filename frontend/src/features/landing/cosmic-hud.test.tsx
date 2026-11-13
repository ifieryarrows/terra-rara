// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';
import { useMotionValue } from 'framer-motion';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CosmicHud } from './CosmicHud';

function HudHarness({ initialProgress = 0, onStep = vi.fn() }: { initialProgress?: number; onStep?: (target: number) => void }) {
  const progress = useMotionValue(initialProgress);
  return <CosmicHud progress={progress} onStepToProgress={onStep} />;
}

afterEach(() => {
  cleanup();
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
  });
});
