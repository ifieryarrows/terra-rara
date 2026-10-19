// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CINEMATIC_RESTS, createCinematicSceneStepper, type CinematicSceneStepper } from './cinematic-scroll';

const GEOMETRY = { top: 0, length: 3_000 };
const restY = (index: number) => CINEMATIC_RESTS[index] * GEOMETRY.length;

function stepper(overrides: Partial<Parameters<typeof createCinematicSceneStepper>[0]> = {}): CinematicSceneStepper {
  return createCinematicSceneStepper({ getGeometry: () => GEOMETRY, ...overrides });
}

function atScrollY(value: number) {
  Object.defineProperty(window, 'scrollY', { configurable: true, value, writable: true });
}

function wheel(deltaY: number, init: WheelEventInit = {}): WheelEvent {
  return new WheelEvent('wheel', { cancelable: true, deltaY, ...init });
}

function key(name: string, init: KeyboardEventInit = {}): KeyboardEvent {
  return new KeyboardEvent('keydown', { cancelable: true, key: name, ...init });
}

describe('cinematic scene stepper', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    Object.defineProperty(window, 'scrollTo', { configurable: true, value: vi.fn() });
    atScrollY(0);
    Object.defineProperty(document.documentElement, 'scrollHeight', { configurable: true, value: 5_000 });
  });

  afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

  it('advances exactly one rest point per wheel gesture', () => {
    const scene = stepper();
    const event = wheel(240);

    expect(scene.handleWheel(event)).toBe(true);
    expect(event.defaultPrevented).toBe(true);
    expect(window.scrollTo).toHaveBeenCalledTimes(1);
    expect(window.scrollTo).toHaveBeenCalledWith({ top: restY(1), behavior: 'smooth' });
    scene.dispose();
  });

  it('never queues further input while a step is gliding', () => {
    const scene = stepper();
    scene.handleWheel(wheel(240));
    for (let index = 0; index < 6; index += 1) {
      const followUp = wheel(240);
      expect(scene.handleWheel(followUp)).toBe(true);
      expect(followUp.defaultPrevented).toBe(true);
    }
    expect(window.scrollTo).toHaveBeenCalledTimes(1);
    scene.dispose();
  });

  it('treats wheel events inside one momentum burst as a single gesture', () => {
    const scene = stepper({ lockTimeoutMs: 50 });
    scene.handleWheel(wheel(240)); // step to rest 1
    atScrollY(restY(1));
    vi.advanceTimersByTime(100); // glide finished, momentum stream continues
    scene.handleWheel(wheel(240)); // same gesture: swallowed, no second step
    expect(window.scrollTo).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(400); // deliberate new gesture after the burst
    scene.handleWheel(wheel(240));
    expect(window.scrollTo).toHaveBeenCalledTimes(2);
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: restY(2), behavior: 'smooth' });
    scene.dispose();
  });

  it('steps backwards to the nearest earlier rest point', () => {
    atScrollY(restY(2));
    const scene = stepper();

    expect(scene.handleWheel(wheel(-240))).toBe(true);
    expect(window.scrollTo).toHaveBeenCalledWith({ top: restY(1), behavior: 'smooth' });
    scene.dispose();
  });

  it('releases to native scrolling at the story bounds', () => {
    atScrollY(restY(CINEMATIC_RESTS.length - 1));
    const scene = stepper();
    const forward = wheel(240);
    expect(scene.handleWheel(forward)).toBe(false);
    expect(forward.defaultPrevented).toBe(false);

    atScrollY(0);
    const backward = wheel(-240);
    expect(scene.handleWheel(backward)).toBe(false);
    expect(backward.defaultPrevented).toBe(false);
    expect(window.scrollTo).not.toHaveBeenCalled();
    scene.dispose();
  });

  it('passes through zoom, horizontal and non-cancelable input', () => {
    const scene = stepper();
    expect(scene.handleWheel(wheel(240, { ctrlKey: true }))).toBe(false);
    expect(scene.handleWheel(wheel(40, { deltaX: 160 }))).toBe(false);
    expect(scene.handleWheel(wheel(240, { cancelable: false }))).toBe(false);
    expect(window.scrollTo).not.toHaveBeenCalled();
    scene.dispose();
  });

  it('releases the glide lock on scrollend before the timeout', () => {
    const scene = stepper();
    scene.handleWheel(wheel(240));
    atScrollY(restY(1));
    window.dispatchEvent(new Event('scrollend'));
    vi.advanceTimersByTime(300); // user pauses, then a new deliberate gesture
    scene.handleWheel(wheel(240));
    expect(window.scrollTo).toHaveBeenCalledTimes(2);
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: restY(2), behavior: 'smooth' });
    scene.dispose();
  });

  it('steps with keyboard navigation and swallows key repeat mid-story', () => {
    const scene = stepper();
    expect(scene.handleKeyDown(key('ArrowDown'))).toBe(true);
    expect(window.scrollTo).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(1_000);
    expect(scene.handleKeyDown(key('PageDown', { repeat: true }))).toBe(true);
    expect(window.scrollTo).toHaveBeenCalledTimes(1);
    scene.dispose();
  });

  it('leaves modified or interactive-focus key input to the browser', () => {
    const scene = stepper();
    expect(scene.handleKeyDown(key('ArrowDown', { ctrlKey: true }))).toBe(false);
    const button = document.createElement('button');
    document.body.append(button);
    button.focus();
    expect(scene.handleKeyDown(key('ArrowDown'))).toBe(false);
    button.remove();
    expect(window.scrollTo).not.toHaveBeenCalled();
    scene.dispose();
  });
});
