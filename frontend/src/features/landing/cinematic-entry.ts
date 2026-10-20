import { useEffect, useRef } from 'react';
import { animate, useMotionValue, type MotionValue } from 'framer-motion';

export const CINEMATIC_ENTRY_TRIGGER = .93;
export const CINEMATIC_ENTRY_RESET = .87;
/**
 * The CTA layer's opacity is scroll-linked from here to the trigger. The
 * content reveal keeps its own 2 s / .62 s clock, but the layer itself can
 * never linger over the returning story when a reverse gesture lands before
 * the reset animation has finished.
 */
export const CINEMATIC_ENTRY_FADE_FROM = .90;

export function getCinematicEntryOpacity(progress: number) {
  const value = Math.max(0, Math.min(1, progress));
  return Math.max(0, Math.min(1, (value - CINEMATIC_ENTRY_FADE_FROM) / (CINEMATIC_ENTRY_TRIGGER - CINEMATIC_ENTRY_FADE_FROM)));
}

export function getCinematicParticleProgress(scrollProgress: number, revealProgress: number) {
  const scroll = Math.max(0, Math.min(1, scrollProgress));
  const reveal = Math.max(0, Math.min(1, revealProgress));
  return scroll >= CINEMATIC_ENTRY_TRIGGER && reveal > 0
    ? Math.max(scroll, CINEMATIC_ENTRY_TRIGGER + reveal * (1 - CINEMATIC_ENTRY_TRIGGER))
    : scroll;
}

export function useCinematicEntryReveal(progress: MotionValue<number>) {
  const reveal = useMotionValue(0);
  const animation = useRef<ReturnType<typeof animate> | null>(null);
  const triggered = useRef(false);
  useEffect(() => {
    const update = (value: number) => {
      if (value >= CINEMATIC_ENTRY_TRIGGER && !triggered.current) {
        triggered.current = true;
        animation.current?.stop();
        animation.current = animate(reveal, 1, { duration: 2, ease: [0.22, 0.61, 0.36, 1] });
      } else if (value < CINEMATIC_ENTRY_RESET && triggered.current) {
        triggered.current = false;
        animation.current?.stop();
        animation.current = animate(reveal, 0, { duration: .62, ease: [0.4, 0, 0.2, 1] });
      }
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
