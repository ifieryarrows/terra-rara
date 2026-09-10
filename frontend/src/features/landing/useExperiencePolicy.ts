import { useEffect, useState } from 'react';
import { useReducedMotion } from 'framer-motion';

export type ExperienceQuality = 'high' | 'balanced' | 'static';

type DeviceHints = Navigator & {
  deviceMemory?: number;
  connection?: EventTarget & { saveData?: boolean };
};

/** Cinematic animation is opt-in after hydration; base HTML stays fully visible. */
export function useExperiencePolicy() {
  const reduce = useReducedMotion();
  const [hardwareQuality, setHardwareQuality] = useState<ExperienceQuality>('static');
  useEffect(() => {
    const highQualityViewport = window.matchMedia('(min-width: 1024px) and (pointer: fine)');
    const hints = navigator as DeviceHints;
    const update = () => {
      const constrainedMemory = hints.deviceMemory !== undefined && hints.deviceMemory <= 2;
      const constrainedCpu = navigator.hardwareConcurrency !== undefined && navigator.hardwareConcurrency <= 2;
      if (hints.connection?.saveData || constrainedMemory || constrainedCpu) {
        setHardwareQuality('static');
      } else {
        setHardwareQuality(highQualityViewport.matches ? 'high' : 'balanced');
      }
    };
    update();
    highQualityViewport.addEventListener('change', update);
    hints.connection?.addEventListener('change', update);
    return () => {
      highQualityViewport.removeEventListener('change', update);
      hints.connection?.removeEventListener('change', update);
    };
  }, []);
  const quality = reduce === false ? hardwareQuality : 'static';
  return { enhanced: quality !== 'static', quality, reducedMotion: reduce !== false };
}
