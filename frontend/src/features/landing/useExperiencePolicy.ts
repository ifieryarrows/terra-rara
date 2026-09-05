import { useEffect, useState } from 'react';
import { useReducedMotion } from 'framer-motion';

/** Enhanced animation is opt-in after hydration; base HTML is fully visible. */
export function useExperiencePolicy() {
  const reduce = useReducedMotion();
  const [desktop, setDesktop] = useState(false);
  useEffect(() => {
    const media = window.matchMedia('(min-width: 1024px) and (pointer: fine)');
    const hints = navigator as Navigator & { deviceMemory?: number; connection?: EventTarget & { saveData?: boolean } };
    const update = () => setDesktop(media.matches && !hints.connection?.saveData && (hints.deviceMemory === undefined || hints.deviceMemory > 4));
    update();
    media.addEventListener('change', update);
    hints.connection?.addEventListener('change', update);
    return () => {
      media.removeEventListener('change', update);
      hints.connection?.removeEventListener('change', update);
    };
  }, []);
  return { enhanced: desktop && reduce === false, reducedMotion: reduce !== false };
}
