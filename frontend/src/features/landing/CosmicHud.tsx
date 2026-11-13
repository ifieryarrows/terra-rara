import { useEffect, useRef } from 'react';
import { motion, useTransform, type MotionValue } from 'framer-motion';
import { Volume2, VolumeX } from 'lucide-react';
import {
  playObservatoryChime,
  stopObservatoryAmbient,
  useObservatorySfx,
} from '../../utils/cosmicVision';
import {
  startLandingVoyageAudio,
  stopLandingVoyageAudio,
  updateLandingVoyageScroll,
} from './landingVoyageAudio';

export type CosmicHudProps = {
  progress: MotionValue<number>;
  onStepToProgress: (targetProgress: number) => void;
};

const RETICLE_STOPS = [0, 0.24, 0.43, 0.66, 0.79, 0.86, 0.97] as const;

const MAJOR_STOPS = [
  { label: 'VENUS · 2D', topPct: 6 },
  { label: 'LIGHT-YEARS', topPct: 26 },
  { label: 'LIGHT-MIN', topPct: 48 },
  { label: '0.72 AU', topPct: 68 },
  { label: 'VENUS FLYBY', topPct: 79 },
  { label: 'MILKY WAY', topPct: 92 },
] as const;

const TICKS = Array.from({ length: 41 }, (_, i) => i);

/**
 * Logarithmic cosmic distance interpolation across the observatory journey:
 * p = 0.00 -> 26 THOUSAND LY (Observatory Opening)
 * p = 0.25 -> 18.0 -> 4.2 LIGHT-YEARS (Milky Way Dust Rift & Macro Constellation)
 * p = 0.50 -> 8.3 LIGHT-MIN (FinBERT Signal Stream)
 * p = 0.74 -> 0.72 AU (Venus Orbital Approach & end of 5D Forecast Horizon)
 * p = 0.80 -> 0.001 AU / 149,598 KM (Venus Hill Sphere flyby)
 * p = 0.835 -> 12,104 KM (Venus closest planetary flyby before dissolution)
 * p = 0.86 -> 26 THOUSAND LY (Milky Way Barred Spiral Core emergence)
 * p = 1.00 -> 105 THOUSAND LY (Full Milky Way Galactic Observatory scale)
 */
function formatCosmicScale(p: number): { num: string; unit: string } {
  if (p < 0.18) {
    const t = Math.max(0, p / 0.18);
    const val = 26 - t * 8;
    const str = val.toFixed(1);
    return {
      num: str.endsWith('.0') ? str.slice(0, -2) : str,
      unit: 'THOUSAND LY',
    };
  }
  if (p < 0.39) {
    const t = (p - 0.18) / (0.39 - 0.18);
    const val = 18 - t * 13.8;
    return {
      num: val.toFixed(1),
      unit: 'LIGHT-YEARS',
    };
  }
  if (p < 0.59) {
    const t = (p - 0.39) / (0.59 - 0.39);
    const val = 14.2 - t * 8.2;
    return {
      num: val.toFixed(1),
      unit: 'LIGHT-MIN',
    };
  }
  if (p < 0.74) {
    const t = (p - 0.59) / (0.74 - 0.59);
    const val = 5.73 * Math.pow(0.72 / 5.73, t);
    return {
      num: val.toFixed(2),
      unit: 'AU',
    };
  }
  if (p < 0.80) {
    const t = (p - 0.74) / (0.80 - 0.74);
    const val = 0.72 * Math.pow(0.001 / 0.72, t);
    return {
      num: val >= 0.05 ? val.toFixed(2) : val.toFixed(3),
      unit: 'AU',
    };
  }
  if (p <= 0.835) {
    const t = (p - 0.80) / (0.835 - 0.80);
    const km = Math.round(149598 * Math.pow(12104 / 149598, t));
    return {
      num: km.toLocaleString('en-US'),
      unit: 'KM',
    };
  }
  if (p < 0.86) {
    const t = (p - 0.835) / (0.86 - 0.835);
    const kly = 0.2 + (26 - 0.2) * (t * t * (3 - 2 * t));
    return {
      num: kly < 10 ? kly.toFixed(1) : String(Math.round(kly)),
      unit: 'THOUSAND LY',
    };
  }
  const t = Math.min(1, Math.max(0, (p - 0.86) / 0.14));
  const eased = 1 - Math.pow(1 - t, 2);
  const kly = Math.round(26 + (105 - 26) * eased);
  return {
    num: kly.toLocaleString('en-US'),
    unit: 'THOUSAND LY',
  };
}

export function CosmicHud({ progress, onStepToProgress }: CosmicHudProps) {
  const [soundActive, toggleSound] = useObservatorySfx();
  const voyageStartedRef = useRef(false);

  // Never carry Workspace ambient background music onto the Landing experience;
  // stop Landing's own scroll-synced soundscape when unmounting from `/`.
  useEffect(() => {
    stopObservatoryAmbient();
    return () => {
      voyageStartedRef.current = false;
      stopLandingVoyageAudio();
    };
  }, []);

  // Synchronize Landing's bespoke dual-stem soundscape continuously with scroll progress.
  useEffect(() => {
    let lastP = progress.get();
    updateLandingVoyageScroll(lastP, 0);

    const unsubscribe = progress.on('change', p => {
      const vel = Math.min(1, Math.abs(p - lastP) * 18);
      lastP = p;
      updateLandingVoyageScroll(p, vel);
    });

    if (!soundActive) {
      voyageStartedRef.current = false;
      stopLandingVoyageAudio();
      return () => unsubscribe();
    }

    const activateVoyage = (event?: Event) => {
      if (voyageStartedRef.current) return;
      const target = event?.target;
      if (target instanceof Element && target.closest('[data-cm-sound-toggle="true"]')) return;
      voyageStartedRef.current = true;
      updateLandingVoyageScroll(progress.get(), 0);
      void startLandingVoyageAudio();
    };

    window.addEventListener('pointerdown', activateVoyage, { passive: true });
    window.addEventListener('keydown', activateVoyage, { passive: true });
    window.addEventListener('wheel', activateVoyage, { passive: true });

    return () => {
      unsubscribe();
      window.removeEventListener('pointerdown', activateVoyage);
      window.removeEventListener('keydown', activateVoyage);
      window.removeEventListener('wheel', activateVoyage);
    };
  }, [progress, soundActive]);

  const rulerNumberRef = useRef<HTMLSpanElement>(null);
  const rulerUnitRef = useRef<HTMLSpanElement>(null);
  const stopRefs = useRef<(HTMLSpanElement | null)[]>([]);

  const notchTop = useTransform(progress, [0, 1], ['6%', '92%']);
  // Hidden at the initial Hero view (p = 0) so the landing hero remains unobstructed,
  // fades in cleanly as the cosmic journey begins, and stays legible through the Milky Way finale.
  const rulerOpacity = useTransform(progress, [0, 0.035, 0.09, 0.94, 1], [0, 0, 1, 1, 0.86]);
  const rulerVisibility = useTransform(rulerOpacity, value => (value <= 0.004 ? 'hidden' : 'visible'));
  const reticleOpacity = useTransform(progress, [0, 0.89, 0.96], [1, 1, 0]);
  const reticleVisibility = useTransform(reticleOpacity, value => (value <= 0.004 ? 'hidden' : 'visible'));
  const reticlePointerEvents = useTransform(reticleOpacity, value => (value >= 0.15 ? 'auto' : 'none'));

  useEffect(() => {
    const updateScale = (p: number) => {
      const { num, unit } = formatCosmicScale(p);
      if (rulerNumberRef.current) rulerNumberRef.current.textContent = num;
      if (rulerUnitRef.current) rulerUnitRef.current.textContent = unit;

      // Smoothly fade static stop labels to 0 when the moving notch is within ±10%
      // to eliminate any collision or visual crowding with the moving readout.
      const notchPct = 6 + p * 86;
      MAJOR_STOPS.forEach((stop, i) => {
        const el = stopRefs.current[i];
        if (el) {
          const factor = Math.min(1, Math.max(0, (Math.abs(notchPct - stop.topPct) - 4) / 6));
          el.style.opacity = factor.toFixed(3);
        }
      });
    };

    updateScale(progress.get());
    const unsubscribe = progress.on('change', updateScale);
    return () => {
      unsubscribe();
    };
  }, [progress]);

  const handleReticleClick = () => {
    void playObservatoryChime();
    const current = progress.get();
    const next = RETICLE_STOPS.find(stop => stop > current + 0.015);
    onStepToProgress(next !== undefined ? next : 0);
  };

  return (
    <div className="cm-cosmic-hud-container" aria-label="Observatory Cosmic HUD">
      {/* 1. Minimal Top-Right Ambient Audio Control */}
      <div
        className="cm-cosmic-hud-top"
        role="toolbar"
        aria-label="Observatory sensory controls"
      >
        <button
          type="button"
          className={`cm-hud-toggle ${soundActive ? 'cm-hud-toggle--active' : ''}`}
          data-cm-click-sound="off"
          data-cm-sound-toggle="true"
          onClick={() => {
            const next = !soundActive;
            toggleSound(next);
            if (next) {
              voyageStartedRef.current = true;
              updateLandingVoyageScroll(progress.get(), 0);
              void startLandingVoyageAudio();
              void playObservatoryChime({ force: true, volume: 0.34 });
            } else {
              voyageStartedRef.current = false;
              stopLandingVoyageAudio();
            }
          }}
          aria-pressed={soundActive}
          aria-label={`Observatory sound effects ${soundActive ? 'on' : 'off'}`}
        >
          {soundActive ? (
            <Volume2 size={13} aria-hidden="true" />
          ) : (
            <VolumeX size={13} aria-hidden="true" />
          )}
          <span>SOUND: {soundActive ? 'ON' : 'OFF'}</span>
        </button>
      </div>

      {/* 2. Right-Side Logarithmic Scale Ruler (fades in after Hero) */}
      <motion.div
        className="cm-cosmic-ruler"
        style={{ opacity: rulerOpacity, visibility: rulerVisibility }}
        aria-hidden="true"
      >
        <div className="cm-cosmic-ruler-rail-wrap">
          {MAJOR_STOPS.map((stop, i) => (
            <span
              key={stop.label}
              ref={el => {
                stopRefs.current[i] = el;
              }}
              className="cm-cosmic-ruler-stop"
              style={{ top: `${stop.topPct}%` }}
            >
              {stop.label}
            </span>
          ))}

          <div className="cm-cosmic-ruler-rail">
            {TICKS.map(tick => {
              const top = `${(tick / 40) * 100}%`;
              const isMajor = tick % 10 === 0;
              return (
                <span
                  key={tick}
                  className={`cm-cosmic-ruler-tick ${isMajor ? 'cm-cosmic-ruler-tick--major' : ''}`}
                  style={{ top }}
                />
              );
            })}

            <motion.div
              className="cm-cosmic-ruler-notch-group"
              style={{ top: notchTop }}
            >
              <div className="cm-cosmic-ruler-readout">
                <span ref={rulerNumberRef} className="cm-cosmic-ruler-num">26</span>
                <span ref={rulerUnitRef} className="cm-cosmic-ruler-unit">THOUSAND LY</span>
              </div>
              <span className="cm-cosmic-ruler-notch" />
            </motion.div>
          </div>
        </div>
      </motion.div>

      {/* 3. Clean Center-Bottom Interactive Target Reticle (+) */}
      <motion.div
        className="cm-cosmic-reticle-wrap"
        style={{ opacity: reticleOpacity, visibility: reticleVisibility, pointerEvents: reticlePointerEvents }}
      >
        <button
          type="button"
          className="cm-cosmic-reticle"
          onClick={handleReticleClick}
          aria-label="Advance to next observatory milestone"
        >
          <svg className="cm-cosmic-reticle-rings" viewBox="0 0 64 64" fill="none" aria-hidden="true">
            <circle cx="32" cy="32" r="28" stroke="currentColor" strokeWidth="1" strokeDasharray="3 3" opacity="0.38" />
            <circle cx="32" cy="32" r="20" stroke="currentColor" strokeWidth="1" opacity="0.58" />
          </svg>
          <span className="cm-cosmic-reticle-plus" aria-hidden="true">+</span>
        </button>
      </motion.div>
    </div>
  );
}
