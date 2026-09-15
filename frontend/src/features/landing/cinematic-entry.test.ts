// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { CINEMATIC_ENTRY_TRIGGER, getCinematicParticleProgress } from './cinematic-entry';

describe('cinematic entry choreography', () => {
  it('starts particle release with the CTA reveal', () => {
    expect(getCinematicParticleProgress(CINEMATIC_ENTRY_TRIGGER, .5)).toBeCloseTo(.965);
    expect(getCinematicParticleProgress(CINEMATIC_ENTRY_TRIGGER, 0)).toBe(CINEMATIC_ENTRY_TRIGGER);
  });

  it('returns particle control to scroll when moving back before the reveal', () => {
    expect(getCinematicParticleProgress(.9, 1)).toBe(.9);
  });
});
