import { useEffect, useRef } from 'react';
import { animate, useMotionValue, type MotionValue } from 'framer-motion';

export const CINEMATIC_ENTRY_TRIGGER = .93;
export const CINEMATIC_ENTRY_RESET = .87;
/**
 * The CTA layer fades smoothly with scroll across [FADE_FROM, TRIGGER] and is
 * further damped by a Framer Motion spring in `CinematicEntryCTA`, so a single
 * upward or downward scroll step glides smoothly via opacity + transform and
 * never snaps invisible mid-motion.
 */
export const CINEMATIC_ENTRY_FADE_FROM = .85;

export function getCinematicEntryOpacity(progress: number) {
  const value = Math.max(0, Math.min(1, progress));
  const t = Math.max(0, Math.min(1, (value - CINEMATIC_ENTRY_FADE_FROM) / (CINEMATIC_ENTRY_TRIGGER - CINEMATIC_ENTRY_FADE_FROM)));
  return t * t * (3 - 2 * t);
}

export function getCinematicParticleProgress(scrollProgress: number, revealProgress: number) {
  const scroll = Math.max(0, Math.min(1, scrollProgress));
  const reveal = Math.max(0, Math.min(1, revealProgress));
  return scroll >= CINEMATIC_ENTRY_TRIGGER && reveal > 0
    ? Math.max(scroll, CINEMATIC_ENTRY_TRIGGER + reveal * (1 - CINEMATIC_ENTRY_TRIGGER))
    : scroll;
}

export type CinematicEntryPhase = 'idle' | 'revealed';

export function advanceCinematicEntry(
  phase: CinematicEntryPhase,
  progress: number,
): { phase: CinematicEntryPhase; changed: boolean; target: 0 | 1 } {
  if (phase === 'idle' && progress >= CINEMATIC_ENTRY_TRIGGER) {
    return { phase: 'revealed', changed: true, target: 1 };
  }
  if (phase === 'revealed' && progress < CINEMATIC_ENTRY_RESET) {
    return { phase: 'idle', changed: true, target: 0 };
  }
  return { phase, changed: false, target: phase === 'revealed' ? 1 : 0 };
}

export function useCinematicEntryReveal(progress: MotionValue<number>) {
  const reveal = useMotionValue(0);
  const animation = useRef<ReturnType<typeof animate> | null>(null);
  const phaseRef = useRef<CinematicEntryPhase>('idle');
  useEffect(() => {
    const update = (value: number) => {
      const step = advanceCinematicEntry(phaseRef.current, value);
      if (!step.changed) return;
      phaseRef.current = step.phase;
      animation.current?.stop();
      animation.current = step.target === 1
        ? animate(reveal, 1, { duration: 1.25, ease: [0.22, 0.61, 0.36, 1] })
        : animate(reveal, 0, { duration: .55, ease: [0.4, 0, 0.2, 1] });
    };
    update(progress.get());
    const unsubscribe = progress.on('change', update);
    return () => {
      unsubscribe();
      animation.current?.stop();
    };
  }, [progress, reveal]);
  return reveal;
}
