// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CINEMATIC_RESTS, createCinematicSceneStepper, isMacPlatform, type CinematicSceneStepper } from './cinematic-scroll';

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

  it('gives both the intact Venus flyby and the Milky Way stage their own rests between forecast and CTA', () => {
    const forecast = CINEMATIC_RESTS.indexOf(.69);
    expect(CINEMATIC_RESTS[forecast + 1]).toBe(.79);
    expect(CINEMATIC_RESTS[forecast + 2]).toBe(.86);
    expect(CINEMATIC_RESTS[forecast + 3]).toBe(.97);
  });

  it('steps from forecast to the intact Venus flyby first, then to Milky Way on the next scroll', () => {
    atScrollY(restY(CINEMATIC_RESTS.indexOf(.69)));
    const scene = stepper({ lockTimeoutMs: 50 });

    expect(scene.handleWheel(wheel(240))).toBe(true);
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: restY(CINEMATIC_RESTS.indexOf(.79)), behavior: 'smooth' });

    atScrollY(restY(CINEMATIC_RESTS.indexOf(.79)));
    vi.advanceTimersByTime(400);

    expect(scene.handleWheel(wheel(240))).toBe(true);
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: restY(CINEMATIC_RESTS.indexOf(.86)), behavior: 'smooth' });
    scene.dispose();
  });

  it('steps upward cleanly from Milky Way back to intact Venus flyby before forecast', () => {
    atScrollY(restY(CINEMATIC_RESTS.indexOf(.86)));
    const scene = stepper();

    expect(scene.handleWheel(wheel(-240))).toBe(true);
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: restY(CINEMATIC_RESTS.indexOf(.79)), behavior: 'smooth' });
    scene.dispose();
  });

  it('steps upward cleanly from the final CTA back to the Milky Way stage', () => {
    atScrollY(restY(CINEMATIC_RESTS.indexOf(.97)));
    const scene = stepper();

    expect(scene.handleWheel(wheel(-240))).toBe(true);
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: restY(CINEMATIC_RESTS.indexOf(.86)), behavior: 'smooth' });
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

  it('uses cached maxScrollTop when provided without reading document scrollHeight on wheel', () => {
    const scrollHeightSpy = vi.fn(() => 5_000);
    Object.defineProperty(document.documentElement, 'scrollHeight', { configurable: true, get: scrollHeightSpy });

    const scene = createCinematicSceneStepper({
      getGeometry: () => ({ top: 0, length: 3_000, maxScrollTop: 1_000 }),
    });

    atScrollY(920);
    expect(scene.handleWheel(wheel(240))).toBe(true);
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 1_000, behavior: 'smooth' });
    expect(scrollHeightSpy).not.toHaveBeenCalled();
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

  it('detects macOS platforms accurately without matching Windows, Linux, or iOS', () => {
    expect(isMacPlatform({ platform: 'MacIntel', userAgent: '', userAgentData: { platform: 'macOS' } })).toBe(true);
    expect(isMacPlatform({ platform: 'MacIntel', userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)' })).toBe(true);
    expect(isMacPlatform({ platform: '', userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0)' })).toBe(true);
    expect(isMacPlatform({ platform: 'Win32', userAgent: 'Windows NT 10.0', userAgentData: { platform: 'Windows' } })).toBe(false);
    expect(isMacPlatform({ platform: 'Linux x86_64', userAgent: 'X11; Linux x86_64' })).toBe(false);
    expect(isMacPlatform({ platform: 'iPhone', userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)' })).toBe(false);
    expect(isMacPlatform(undefined)).toBe(false);
  });
});
