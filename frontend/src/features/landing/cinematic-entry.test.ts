// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { CINEMATIC_ENTRY_FADE_FROM, CINEMATIC_ENTRY_TRIGGER, getCinematicEntryOpacity, getCinematicParticleProgress } from './cinematic-entry';

describe('cinematic entry choreography', () => {
  it('starts particle release with the CTA reveal', () => {
    expect(getCinematicParticleProgress(CINEMATIC_ENTRY_TRIGGER, .5)).toBeCloseTo(.965);
    expect(getCinematicParticleProgress(CINEMATIC_ENTRY_TRIGGER, 0)).toBe(CINEMATIC_ENTRY_TRIGGER);
  });

  it('returns particle control to scroll when moving back before the reveal', () => {
    expect(getCinematicParticleProgress(.9, 1)).toBe(.9);
  });

  it('fades the CTA layer out before the returning copy can reach the viewport', () => {
    expect(getCinematicEntryOpacity(CINEMATIC_ENTRY_TRIGGER)).toBe(1);
    expect(getCinematicEntryOpacity(.97)).toBe(1);
    expect(getCinematicEntryOpacity((CINEMATIC_ENTRY_FADE_FROM + CINEMATIC_ENTRY_TRIGGER) / 2)).toBeCloseTo(.5);
    expect(getCinematicEntryOpacity(CINEMATIC_ENTRY_FADE_FROM)).toBe(0);
    expect(getCinematicEntryOpacity(.86)).toBe(0);
    expect(getCinematicEntryOpacity(.75)).toBe(0);
  });
});
