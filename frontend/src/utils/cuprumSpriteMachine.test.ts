import { describe, expect, it } from 'vitest';
import {
  ALERT_FRAME_MS,
  BLINK_FRAME_MS,
  BLINK_GAP_MIN_MS,
  BLINK_GAP_RANGE_MS,
  BOB_FRAME_MS,
  FIRST_BLINK_DELAY_MS,
  INITIAL_CUPRUM_SPRITE_STATE,
  advanceCuprumSprite,
  cuprumBlinkGapMs,
  cuprumSpriteFrame,
} from './cuprumSpriteMachine';
import type {
  CuprumSpriteEvent,
  CuprumSpriteSchedule,
  CuprumSpriteState,
} from './cuprumSpriteMachine';

function drive(events: CuprumSpriteEvent[]) {
  let state: CuprumSpriteState = INITIAL_CUPRUM_SPRITE_STATE;
  let schedule: CuprumSpriteSchedule[] = [];
  for (const event of events) {
    ({ state, schedule } = advanceCuprumSprite(state, event));
  }
  return { state, schedule };
}

describe('cuprumSpriteMachine', () => {
  it('starts on the landed pose and arms the metronome plus the first blink', () => {
    const { state, schedule } = drive(['mounted']);

    expect(cuprumSpriteFrame(state)).toBe('idle-1');
    expect(schedule).toEqual([
      { kind: 'tick', delayMs: BOB_FRAME_MS },
      { kind: 'blink-request', delayMs: FIRST_BLINK_DELAY_MS },
    ]);
  });

  it('flips the bob metronome on every tick without shortening any phase', () => {
    const up = drive(['mounted', 'bob-tick']);
    expect(cuprumSpriteFrame(up.state)).toBe('idle-2');
    expect(up.schedule).toEqual([{ kind: 'tick', delayMs: BOB_FRAME_MS }]);

    const down = drive(['mounted', 'bob-tick', 'bob-tick']);
    expect(cuprumSpriteFrame(down.state)).toBe('idle-1');
    expect(down.schedule).toEqual([{ kind: 'tick', delayMs: BOB_FRAME_MS }]);
  });

  it('queues a blink request and starts it on the next landed beat', () => {
    const queued = drive(['mounted', 'bob-tick', 'blink-request']);
    expect(cuprumSpriteFrame(queued.state)).toBe('idle-2');
    expect(queued.state.blinkQueued).toBe(true);
    expect(queued.schedule).toEqual([{ kind: 'blink-request' }]);

    const blinking = drive(['mounted', 'bob-tick', 'blink-request', 'bob-tick']);
    expect(cuprumSpriteFrame(blinking.state)).toBe('blink');
    expect(blinking.state.blinkQueued).toBe(false);
    // The blink rides inside the landed phase: the metronome beat is kept.
    expect(blinking.schedule).toEqual([
      { kind: 'tick', delayMs: BOB_FRAME_MS },
      { kind: 'blink-end', delayMs: BLINK_FRAME_MS },
    ]);
  });

  it('returns to the landed pose after a blink and ignores a stale blink-end', () => {
    const done = drive(['mounted', 'bob-tick', 'blink-request', 'bob-tick', 'blink-end']);
    expect(cuprumSpriteFrame(done.state)).toBe('idle-1');
    expect(done.schedule).toEqual([]);

    const stale = drive([
      'mounted',
      'bob-tick',
      'blink-request',
      'bob-tick',
      'alert-start',
      'blink-end',
    ]);
    expect(cuprumSpriteFrame(stale.state)).toBe('alert');
  });

  it('starts an alert immediately, owns its full window, and restarts the metronome', () => {
    const alerting = drive(['mounted', 'bob-tick', 'alert-start']);
    expect(cuprumSpriteFrame(alerting.state)).toBe('alert');
    expect(alerting.schedule).toEqual([
      { kind: 'cancel-tick' },
      { kind: 'alert-end', delayMs: ALERT_FRAME_MS },
    ]);

    // The tick that lands inside the alert window is a no-op: the alert is
    // never cut and re-arms the beat only when it ends.
    const midAlert = drive(['mounted', 'bob-tick', 'alert-start', 'bob-tick']);
    expect(cuprumSpriteFrame(midAlert.state)).toBe('alert');
    expect(midAlert.schedule).toEqual([]);

    const landed = drive(['mounted', 'bob-tick', 'alert-start', 'bob-tick', 'alert-end']);
    expect(cuprumSpriteFrame(landed.state)).toBe('idle-1');
    expect(landed.schedule).toEqual([{ kind: 'tick', delayMs: BOB_FRAME_MS }]);
  });

  it('drops a queued blink when an alert takes over', () => {
    const { state, schedule } = drive(['mounted', 'bob-tick', 'blink-request', 'alert-start']);
    expect(state.blinkQueued).toBe(false);
    expect(cuprumSpriteFrame(state)).toBe('alert');
    expect(schedule).toEqual([
      { kind: 'cancel-tick' },
      { kind: 'alert-end', delayMs: ALERT_FRAME_MS },
    ]);

    // The request chain keeps scheduling even while an alert is on screen.
    const duringAlert = drive(['mounted', 'blink-request', 'alert-start', 'blink-request']);
    expect(duringAlert.state.blinkQueued).toBe(false);
    expect(duringAlert.schedule).toEqual([{ kind: 'blink-request' }]);
  });

  it('restarts the alert window when the pet is clicked again mid-alert', () => {
    const { state, schedule } = drive(['mounted', 'alert-start', 'alert-start']);
    expect(cuprumSpriteFrame(state)).toBe('alert');
    expect(schedule).toEqual([
      { kind: 'cancel-tick' },
      { kind: 'alert-end', delayMs: ALERT_FRAME_MS },
    ]);
  });

  it('ignores alert-end when no alert is playing', () => {
    const { state, schedule } = drive(['mounted', 'alert-end']);
    expect(cuprumSpriteFrame(state)).toBe('idle-1');
    expect(schedule).toEqual([]);
  });

  it('randomizes the blink gap inside the configured window', () => {
    expect(cuprumBlinkGapMs(() => 0)).toBe(BLINK_GAP_MIN_MS);
    expect(cuprumBlinkGapMs(() => 1)).toBe(BLINK_GAP_MIN_MS + BLINK_GAP_RANGE_MS);
  });
});
