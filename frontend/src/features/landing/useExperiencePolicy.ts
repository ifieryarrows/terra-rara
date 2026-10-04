import { useEffect, useState } from 'react';

export type ExperienceQuality = 'high' | 'balanced' | 'static';

type DeviceHints = Navigator & {
  deviceMemory?: number;
  connection?: EventTarget & { saveData?: boolean };
};

function readExperienceQuality(): ExperienceQuality {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') return 'static';

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const highQualityViewport = window.matchMedia('(min-width: 1024px) and (pointer: fine)');
  const hints = navigator as DeviceHints;
  const constrainedMemory = hints.deviceMemory !== undefined && hints.deviceMemory <= 2;
  const constrainedCpu = navigator.hardwareConcurrency !== undefined && navigator.hardwareConcurrency <= 2;

  if (reducedMotion.matches || hints.connection?.saveData || constrainedMemory || constrainedCpu) return 'static';
  return highQualityViewport.matches ? 'high' : 'balanced';
}

/** Choose the experience before the first client render to avoid a static-to-cinematic swap. */
export function useExperiencePolicy() {
  const [quality, setQuality] = useState<ExperienceQuality>(readExperienceQuality);
  useEffect(() => {
    const highQualityViewport = window.matchMedia('(min-width: 1024px) and (pointer: fine)');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const hints = navigator as DeviceHints;
    const update = () => setQuality(readExperienceQuality());
    update();
    highQualityViewport.addEventListener('change', update);
    reducedMotion.addEventListener('change', update);
    hints.connection?.addEventListener('change', update);
    return () => {
      highQualityViewport.removeEventListener('change', update);
      reducedMotion.removeEventListener('change', update);
      hints.connection?.removeEventListener('change', update);
    };
  }, []);
  return { enhanced: quality !== 'static', quality };
}
