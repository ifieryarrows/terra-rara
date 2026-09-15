import { useEffect, useRef } from 'react';
import { animate, useMotionValue, type MotionValue } from 'framer-motion';

export const CINEMATIC_ENTRY_TRIGGER = .93;
export const CINEMATIC_ENTRY_RESET = .87;

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
