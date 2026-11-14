import { useEffect, useId, useRef } from 'react';
import { motion, useMotionValue, useReducedMotion, useSpring, useTransform, type MotionValue } from 'framer-motion';
import { BrandMark } from '../../components/ui/BrandMark';

const cardinalTicks = [
  { x1: 210, y1: 24, x2: 210, y2: 36 },
  { x1: 396, y1: 210, x2: 384, y2: 210 },
  { x1: 210, y1: 396, x2: 210, y2: 384 },
  { x1: 24, y1: 210, x2: 36, y2: 210 },
] as const;

const TILT_SPRING = { stiffness: 120, damping: 20, mass: 0.65 };

export function CopperSignal({ progress }: { progress?: MotionValue<number> }) {
  const id = useId().replace(/:/g, '');
  const stageRef = useRef<HTMLDivElement>(null);
  const reducedMotion = useReducedMotion();

  const targetTiltX = useMotionValue(6);
  const targetTiltY = useMotionValue(-9);
  const rotateX = useSpring(targetTiltX, TILT_SPRING);
  const rotateY = useSpring(targetTiltY, TILT_SPRING);
  const glintX = useTransform(rotateY, [-18, 18], ['34%', '66%']);
  const glintY = useTransform(rotateX, [-16, 16], ['64%', '30%']);

  useEffect(() => {
    if (reducedMotion || typeof window === 'undefined') return;

    const stage = stageRef.current;
    let cx = window.innerWidth * 0.72;
    let cy = window.innerHeight * 0.48;
    let denomX = Math.max(window.innerWidth * 0.45, 220);
    let denomY = Math.max(window.innerHeight * 0.45, 220);

    const measureCenter = () => {
      denomX = Math.max(window.innerWidth * 0.45, 220);
      denomY = Math.max(window.innerHeight * 0.45, 220);
      const rect = stage?.getBoundingClientRect();
      if (rect && (rect.width > 0 || rect.height > 0)) {
        cx = rect.left + rect.width * 0.5;
        cy = rect.top + rect.height * 0.5;
      } else {
        cx = window.innerWidth * 0.72;
        cy = window.innerHeight * 0.48;
      }
    };

    measureCenter();

    const resizeObserver = typeof ResizeObserver !== 'undefined' && stage
      ? new ResizeObserver(() => { measureCenter(); })
      : null;
    if (resizeObserver && stage) {
      resizeObserver.observe(stage);
    }
    window.addEventListener('resize', measureCenter, { passive: true });

    let frameId = 0;
    let latestClientX = cx;
    let latestClientY = cy;

    const flushTilt = () => {
      frameId = 0;
      const normX = Math.max(-1, Math.min(1, (latestClientX - cx) / denomX));
      const normY = Math.max(-1, Math.min(1, (latestClientY - cy) / denomY));
      targetTiltY.set(normX * 16);
      targetTiltX.set(-normY * 14);
    };

    const handleMove = (event: PointerEvent) => {
      if (event.pointerType === 'touch') return;
      latestClientX = event.clientX;
      latestClientY = event.clientY;
      if (!frameId) {
        frameId = window.requestAnimationFrame(flushTilt);
      }
    };

    const handleLeave = () => {
      if (frameId) {
        window.cancelAnimationFrame(frameId);
        frameId = 0;
      }
      targetTiltX.set(6);
      targetTiltY.set(-9);
    };

    window.addEventListener('pointermove', handleMove, { passive: true });
    document.addEventListener('mouseleave', handleLeave, { passive: true });
    return () => {
      if (frameId) window.cancelAnimationFrame(frameId);
      resizeObserver?.disconnect();
      window.removeEventListener('resize', measureCenter);
      window.removeEventListener('pointermove', handleMove);
      document.removeEventListener('mouseleave', handleLeave);
    };
  }, [reducedMotion, targetTiltX, targetTiltY]);

  return (
    <figure
      className="cm-copper-signal"
      aria-label="Terra Rara Venus 3D emblem at the center of CopperMind market intelligence"
    >
      <div className="cm-signal-stage" ref={stageRef}>
        <motion.div
          className="cm-signal-rig"
          style={reducedMotion ? undefined : { rotateX, rotateY }}
        >
          <svg
            className="cm-signal-contours cm-signal-layer cm-signal-layer--contours"
            viewBox="0 0 420 420"
            aria-hidden="true"
          >
            <defs>
              <radialGradient id={`${id}-aura`} cx="210" cy="210" r="190" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="var(--cm-copper)" stopOpacity="0.11" />
                <stop offset="56%" stopColor="var(--cm-forecast)" stopOpacity="0.045" />
                <stop offset="100%" stopColor="var(--cm-bg)" stopOpacity="0" />
              </radialGradient>
            </defs>

            <circle cx="210" cy="210" r="186" fill={`url(#${id}-aura)`} />
            <circle
              cx="210"
              cy="210"
              r="168"
              fill="none"
              stroke="var(--cm-border)"
              strokeWidth="0.9"
              opacity="0.55"
            />
            <circle
              cx="210"
              cy="210"
              r="142"
              fill="none"
              stroke="var(--cm-border)"
              strokeWidth="0.7"
              strokeDasharray="2 7"
              opacity="0.45"
            />
            {progress ? (
              <motion.circle
                cx="210"
                cy="210"
                r="168"
                fill="none"
                stroke="var(--cm-copper)"
                strokeWidth="1.25"
                strokeLinecap="round"
                transform="rotate(-90 210 210)"
                opacity="0.65"
                style={{ pathLength: progress }}
              />
            ) : null}
            {cardinalTicks.map((tick, idx) => (
              <line
                key={idx}
                x1={tick.x1}
                y1={tick.y1}
                x2={tick.x2}
                y2={tick.y2}
                stroke="var(--cm-copper)"
                strokeWidth="0.9"
                opacity="0.5"
              />
            ))}
          </svg>

          {/* Sculpted Z-extruded depth stack (rear bevel -> bronze rim -> spherical core -> primary face -> foreground halo) */}
          <div className="cm-signal-emblem cm-signal-layer cm-signal-layer--depth-back" aria-hidden="true">
            <BrandMark size={172} variant="primary" className="cm-signal-brand-mark" />
          </div>
          <div className="cm-signal-emblem cm-signal-layer cm-signal-layer--depth-mid" aria-hidden="true">
            <BrandMark size={172} variant="primary" className="cm-signal-brand-mark" />
          </div>
          <div className="cm-signal-layer cm-signal-layer--core" aria-hidden="true">
            <motion.span
              className="cm-signal-core-glint"
              style={reducedMotion ? undefined : { left: glintX, top: glintY }}
            />
          </div>
          <div className="cm-signal-emblem cm-signal-layer cm-signal-layer--face" aria-hidden="true">
            <BrandMark size={172} variant="primary" className="cm-signal-brand-mark" />
          </div>
          <div className="cm-signal-layer cm-signal-layer--orbit" aria-hidden="true">
            <span className="cm-signal-orbit-ring" />
          </div>
        </motion.div>
      </div>
      <figcaption className="cm-signal-caption">
        VENUS CYPRIUM · HG=F OBSERVATORY
      </figcaption>
    </figure>
  );
}
