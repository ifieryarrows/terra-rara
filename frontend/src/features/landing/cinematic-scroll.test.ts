// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CINEMATIC_SCROLL_SENSITIVITY, createCinematicWheelDampener } from './cinematic-scroll';

describe('cinematic scroll input', () => {
  beforeEach(() => {
    Object.defineProperty(window, 'scrollTo', { configurable: true, value: vi.fn() });
    Object.defineProperty(document.documentElement, 'scrollHeight', { configurable: true, value: 5_000 });
  });

  afterEach(() => { vi.restoreAllMocks(); });

  it('applies half of each vertical wheel delta', () => {
    const event = new WheelEvent('wheel', { cancelable: true, deltaY: 120 });
    const dampener = createCinematicWheelDampener();

    expect(dampener.handleWheel(event)).toBe(true);
    expect(event.defaultPrevented).toBe(true);
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 120 * CINEMATIC_SCROLL_SENSITIVITY, behavior: 'smooth' });
    dampener.dispose();
  });

  it('does not intercept browser zoom or horizontal gestures', () => {
    const zoom = new WheelEvent('wheel', { cancelable: true, ctrlKey: true, deltaY: 120 });
    const horizontal = new WheelEvent('wheel', { cancelable: true, deltaX: 160, deltaY: 40 });
    const dampener = createCinematicWheelDampener();

    expect(dampener.handleWheel(zoom)).toBe(false);
    expect(dampener.handleWheel(horizontal)).toBe(false);
    expect(window.scrollTo).not.toHaveBeenCalled();
    expect(zoom.defaultPrevented).toBe(false);
    expect(horizontal.defaultPrevented).toBe(false);
    dampener.dispose();
  });

  it('accumulates rapid wheel input against the smooth-scroll target', () => {
    const dampener = createCinematicWheelDampener();

    dampener.handleWheel(new WheelEvent('wheel', { cancelable: true, deltaY: 200 }));
    dampener.handleWheel(new WheelEvent('wheel', { cancelable: true, deltaY: 200 }));

    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: 400 * CINEMATIC_SCROLL_SENSITIVITY, behavior: 'smooth' });
    dampener.dispose();
  });
});
