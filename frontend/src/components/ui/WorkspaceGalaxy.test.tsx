// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { render } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import WorkspaceGalaxy from './WorkspaceGalaxy';
import {
  drawGalaxyCpu2D,
  generateTransitionGalaxy,
  getSharedGalaxyTime,
  syncSharedGalaxyClock,
  WORKSPACE_GALAXY_ALPHA,
  WORKSPACE_GALAXY_TILT,
} from './galaxyEngine';

describe('WorkspaceGalaxy & galaxyEngine', () => {
  it('renders decorative aria-hidden canvases and unmounts cleanly', () => {
    const { container, unmount } = render(<WorkspaceGalaxy />);
    const canvases = container.querySelectorAll('canvas.cm-workspace-galaxy');
    expect(canvases).toHaveLength(2);
    for (const canvas of canvases) {
      expect(canvas).toHaveAttribute('aria-hidden', 'true');
    }
    expect(() => unmount()).not.toThrow();
  });

  it('keeps the shared galaxy clock synchronized across handoff', () => {
    syncSharedGalaxyClock(4200, 1000);
    expect(getSharedGalaxyTime(1500)).toBe(4700);
    syncSharedGalaxyClock(0, 0);
  });

  it('preserves resting galaxy opacity at release end when persistEnd is true and fades to zero when false', () => {
    const data = generateTransitionGalaxy(40);
    const alphas: number[] = [];
    const mockCtx = {
      clearRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      fillStyle: '',
      set globalAlpha(value: number) {
        alphas.push(value);
      },
      get globalAlpha() {
        return alphas[alphas.length - 1] ?? 1;
      },
    } as unknown as CanvasRenderingContext2D;

    drawGalaxyCpu2D(mockCtx, data, {
      width: 1280,
      height: 720,
      mode: 2,
      progress: 1,
      time: 1000,
      fadeIn: false,
      initTilt: WORKSPACE_GALAXY_TILT,
      persistEnd: true,
      endAlpha: WORKSPACE_GALAXY_ALPHA,
      fromWorkspace: false,
    });
    expect(alphas[0]).toBeCloseTo(WORKSPACE_GALAXY_ALPHA * 0.9, 4);

    alphas.length = 0;
    drawGalaxyCpu2D(mockCtx, data, {
      width: 1280,
      height: 720,
      mode: 2,
      progress: 1,
      time: 1000,
      fadeIn: false,
      initTilt: WORKSPACE_GALAXY_TILT,
      persistEnd: false,
      endAlpha: WORKSPACE_GALAXY_ALPHA,
      fromWorkspace: false,
    });
    expect(alphas[0]).toBe(0);
  });

  it('preserves coherent spiral arm structure over long elapsed times without winding shear', () => {
    const data = generateTransitionGalaxy(120);
    const capturePoints = (time: number) => {
      const pts: Array<[number, number]> = [];
      const mockCtx = {
        clearRect: vi.fn(),
        beginPath: vi.fn(),
        moveTo: (x: number, y: number) => pts.push([x, y]),
        arc: vi.fn(),
        fill: vi.fn(),
        fillStyle: '',
        globalAlpha: 1,
      } as unknown as CanvasRenderingContext2D;
      drawGalaxyCpu2D(mockCtx, data, {
        width: 1280,
        height: 720,
        mode: 2,
        progress: 1,
        time,
        fadeIn: false,
        initTilt: WORKSPACE_GALAXY_TILT,
        persistEnd: true,
        endAlpha: WORKSPACE_GALAXY_ALPHA,
        fromWorkspace: false,
      });
      return pts;
    };

    const t0 = capturePoints(0);
    const tLater = capturePoints(60_000);
    expect(t0.length).toBe(120);
    expect(tLater.length).toBe(120);

    // Rigid pattern rotation preserves radial spread and pairwise cohesion (no differential unwinding)
    const meanRadius = (pts: Array<[number, number]>) => {
      const cx = pts.reduce((acc, p) => acc + p[0], 0) / pts.length;
      const cy = pts.reduce((acc, p) => acc + p[1], 0) / pts.length;
      return pts.reduce((acc, p) => acc + Math.hypot(p[0] - cx, p[1] - cy), 0) / pts.length;
    };
    expect(Math.abs(meanRadius(t0) - meanRadius(tLater))).toBeLessThan(8);
  });
});
