/**
 * Cuprum pet sprite state machine.
 *
 * The companion renders single retro pixel sprite frames, and two independent
 * layers decide what is on screen:
 *
 * - `bob` is the idle metronome: `idle-2` is the raised pose and `idle-1` its
 *   landed counterpart. It flips every {@link BOB_FRAME_MS} and every phase
 *   runs in full — that is the single up-and-down movement of the pet.
 * - `overlay` is a transient expression frame. `blink` is aligned with the
 *   landed `idle-1` pose and may only start on a landed beat, so it can never
 *   cut a bob phase short or snap the sprite mid-air. `alert` is the click
 *   reaction: it starts immediately, always owns its full window, and when it
 *   ends the metronome restarts cleanly from the landed pose.
 *
 * This module is pure: no timers, no randomness, no DOM. Callers translate the
 * returned schedule descriptors into real timers.
 */

export const BOB_FRAME_MS = 600;
export const BLINK_FRAME_MS = 160;
export const ALERT_FRAME_MS = 280;
export const FIRST_BLINK_DELAY_MS = 2800;
export const BLINK_GAP_MIN_MS = 3400;
export const BLINK_GAP_RANGE_MS = 1200;

export type CuprumBobFrame = 'idle-1' | 'idle-2';
export type CuprumOverlayFrame = 'blink' | 'alert';
export type CuprumSpriteFrame = CuprumBobFrame | CuprumOverlayFrame;

export type CuprumSpriteEvent =
  | 'mounted'
  | 'bob-tick'
  | 'blink-request'
  | 'blink-end'
  | 'alert-start'
  | 'alert-end';

export interface CuprumSpriteState {
  /** Idle metronome phase; `idle-2` is the raised pose, `idle-1` the landed one. */
  bob: CuprumBobFrame;
  /** Transient expression riding on top of the bob, when one is playing. */
  overlay: CuprumOverlayFrame | null;
  /** A blink request that is waiting for the next landed beat. */
  blinkQueued: boolean;
}

export type CuprumSpriteSchedule =
  | { kind: 'tick'; delayMs: number }
  | { kind: 'blink-end'; delayMs: number }
  | { kind: 'alert-end'; delayMs: number }
  /** `delayMs` is only set for the first request; later gaps are randomized by the caller. */
  | { kind: 'blink-request'; delayMs?: number }
  | { kind: 'cancel-tick' };

export interface CuprumSpriteStep {
  state: CuprumSpriteState;
  schedule: CuprumSpriteSchedule[];
}

export const INITIAL_CUPRUM_SPRITE_STATE: CuprumSpriteState = {
  bob: 'idle-1',
  overlay: null,
  blinkQueued: false,
};

export function cuprumSpriteFrame(state: CuprumSpriteState): CuprumSpriteFrame {
  return state.overlay ?? state.bob;
}

export function cuprumBlinkGapMs(random: () => number = Math.random): number {
  return BLINK_GAP_MIN_MS + random() * BLINK_GAP_RANGE_MS;
}

/** Advances the sprite machine one event. Pure: callers own the timers. */
export function advanceCuprumSprite(
  state: CuprumSpriteState,
  event: CuprumSpriteEvent,
): CuprumSpriteStep {
  switch (event) {
    case 'mounted':
      return {
        state: { ...INITIAL_CUPRUM_SPRITE_STATE },
        schedule: [
          { kind: 'tick', delayMs: BOB_FRAME_MS },
          { kind: 'blink-request', delayMs: FIRST_BLINK_DELAY_MS },
        ],
      };
    case 'bob-tick': {
      // An alert owns the beat; its end event restarts the metronome.
      if (state.overlay === 'alert') return { state, schedule: [] };
      // Defensive: a blink cannot span a beat by construction, but if it ever
      // does, keep the metronome alive instead of stalling.
      if (state.overlay === 'blink') {
        return { state, schedule: [{ kind: 'tick', delayMs: BOB_FRAME_MS }] };
      }
      const bob: CuprumBobFrame = state.bob === 'idle-1' ? 'idle-2' : 'idle-1';
      if (bob === 'idle-1' && state.blinkQueued) {
        return {
          state: { bob, overlay: 'blink', blinkQueued: false },
          schedule: [
            { kind: 'tick', delayMs: BOB_FRAME_MS },
            { kind: 'blink-end', delayMs: BLINK_FRAME_MS },
          ],
        };
      }
      return { state: { ...state, bob }, schedule: [{ kind: 'tick', delayMs: BOB_FRAME_MS }] };
    }
    case 'blink-request':
      // Keep the request chain alive even when this particular request is
      // dropped because another transient frame is on screen.
      if (state.overlay !== null) return { state, schedule: [{ kind: 'blink-request' }] };
      return { state: { ...state, blinkQueued: true }, schedule: [{ kind: 'blink-request' }] };
    case 'blink-end':
      if (state.overlay !== 'blink') return { state, schedule: [] };
      return { state: { ...state, overlay: null }, schedule: [] };
    case 'alert-start':
      // Click feedback is immediate: it takes the beat over, always plays its
      // full window, and the metronome restarts on the landed pose afterwards.
      return {
        state: { ...state, overlay: 'alert', blinkQueued: false },
        schedule: [
          { kind: 'cancel-tick' },
          { kind: 'alert-end', delayMs: ALERT_FRAME_MS },
        ],
      };
    case 'alert-end':
      if (state.overlay !== 'alert') return { state, schedule: [] };
      return {
        state: { ...state, overlay: null, bob: 'idle-1' },
        schedule: [{ kind: 'tick', delayMs: BOB_FRAME_MS }],
      };
  }
}
