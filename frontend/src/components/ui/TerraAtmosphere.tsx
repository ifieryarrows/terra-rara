import { memo, useEffect, useRef } from 'react';
import { motion, useMotionValue, useTransform, type MotionValue } from 'framer-motion';
import './TerraAtmosphere.css';

let latestPointerPosition = { x: 0.5, y: 0.5 };

type TerraAtmosphereProps = {
  progress?: MotionValue<number>;
  interactive?: boolean;
  className?: string;
};

function StarFieldCanvas({ subtle = false }: { subtle?: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || (typeof navigator !== 'undefined' && /jsdom/i.test(navigator.userAgent))) return;

    let context: CanvasRenderingContext2D | null = null;
    try {
      context = canvas.getContext('2d', { alpha: true });
    } catch {
      return;
    }
    if (!context) return;

    let lastWidth = 0;
    let lastHeight = 0;

    const render = () => {
      try {
        const { width, height } = canvas.getBoundingClientRect();
        if (width === 0 || height === 0) return;

        const roundedWidth = Math.round(width);
        const roundedHeight = Math.round(height);
        const isMobileViewport = window.matchMedia('(max-width: 1023px)').matches;
        if (lastWidth > 0 && Math.abs(roundedWidth - lastWidth) < 2) {
          if (Math.abs(roundedHeight - lastHeight) < 2 || isMobileViewport) {
            return;
          }
        }
        lastWidth = roundedWidth;
        lastHeight = roundedHeight;

        const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
        canvas.width = Math.round(width * pixelRatio);
        canvas.height = Math.round(height * pixelRatio);
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        context.clearRect(0, 0, width, height);

        let seed = 0x74657272;
        const random = () => {
          seed = (seed * 1664525 + 1013904223) >>> 0;
          return seed / 0x100000000;
        };
        const gaussian = () => {
          const u = Math.max(random(), 0.0001);
          return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
        };

        // 1. Diagonal Milky Way Star-Cloud Glow Spine (omitted on Landing to keep deep space crisp like CosmicLens)
        if (!subtle && typeof context.createRadialGradient === 'function') {
          const veilScale = 0.85;
          const cloudNodes = [
            { x: width * 0.72, y: height * 0.12, r: width * 0.28, color: `rgba(142, 173, 222, ${(0.085 * veilScale).toFixed(3)})` },
            { x: width * 0.66, y: height * 0.28, r: width * 0.31, color: `rgba(221, 161, 116, ${(0.095 * veilScale).toFixed(3)})` },
            { x: width * 0.61, y: height * 0.44, r: width * 0.33, color: `rgba(221, 161, 116, ${(0.095 * veilScale).toFixed(3)})` },
            { x: width * 0.56, y: height * 0.62, r: width * 0.32, color: `rgba(191, 126, 79, ${(0.08 * veilScale).toFixed(3)})` },
            { x: width * 0.51, y: height * 0.82, r: width * 0.30, color: `rgba(219, 130, 150, ${(0.055 * veilScale).toFixed(3)})` },
            { x: width * 0.688, y: height * 0.466, r: width * 0.34, color: `rgba(255, 243, 226, ${(0.085 * veilScale).toFixed(3)})` },
          ];
          for (const node of cloudNodes) {
            const grad = context.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.r);
            grad.addColorStop(0, node.color);
            grad.addColorStop(0.45, `rgba(196, 152, 118, ${(0.045 * veilScale).toFixed(3)})`);
            grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
            context.fillStyle = grad;
            context.fillRect(0, 0, width, height);
          }
        }

        // 3. Starfield & Milky Way Band Stellar Concentration
        const coreX = width * 0.688;
        const coreY = height * 0.466;
        const tiltAngle = -0.48;
        const cosA = Math.cos(tiltAngle);
        const sinA = Math.sin(tiltAngle);
        const discAspect = 0.42;
        const maxDiscRadius = Math.sqrt(width * width + height * height) * 0.58;

        const count = Math.max(320, Math.min(900, Math.round((width * height) / 2100)));
        const colors = ['#fff3e2', '#dda174', '#8eadde', '#bf7e4f', '#db8296'];

        for (let index = 0; index < count; index += 1) {
          const mode = random();
          let x = 0;
          let y = 0;

          if (mode < 0.22) {
            // Deep space field stars
            x = random() * width;
            y = random() * height;
          } else if (mode < 0.68) {
            // Diagonal Milky Way spine stars with bimodal ridge density flanking the Great Dark Rift
            const t = random();
            const spineX = width * (0.71 - t * 0.22);
            const spineY = height * t;
            const ridgeSide = random() < 0.5 ? -1 : 1;
            const ridgeOffset = (0.018 + Math.abs(gaussian()) * 0.055) * width * ridgeSide;
            x = spineX + ridgeOffset;
            y = spineY + gaussian() * height * 0.035;
          } else {
            // Logarithmic spiral cluster distribution around galactic core
            const arm = (index % 4) * (Math.PI / 2);
            const t = Math.pow(random(), 1.55);
            const r = Math.max(10, t * maxDiscRadius);
            const theta = arm + 2.35 * Math.log(r / 18) + gaussian() * 0.21;

            const u = r * Math.cos(theta);
            const v = r * Math.sin(theta) * discAspect;

            x = coreX + u * cosA - v * sinA;
            y = coreY + u * sinA + v * cosA;
          }

          if (x < -4 || x > width + 4 || y < -4 || y > height + 4) continue;

          const isFocalStar = random() < 0.055;
          const radius = subtle
            ? (isFocalStar ? 0.72 + random() * 0.34 : 0.22 + random() * 0.34)
            : (isFocalStar ? 0.95 + random() * 0.45 : 0.30 + random() * 0.42);
          const baseAlpha = isFocalStar ? 0.74 + random() * 0.24 : 0.24 + random() * 0.52;

          context.globalAlpha = baseAlpha;
          context.fillStyle = colors[Math.floor(random() * colors.length)];
          context.beginPath();
          context.arc(x, y, radius, 0, Math.PI * 2);
          context.fill();
        }

        context.globalAlpha = 1;
      } catch {
        // Safe fallback in restricted environments
      }
    };

    render();
    let resizeTimer: number | undefined;
    const onResize = () => {
      if (resizeTimer !== undefined) window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        resizeTimer = undefined;
        render();
      }, 150);
    };
    window.addEventListener('resize', onResize, { passive: true });
    return () => {
      window.removeEventListener('resize', onResize);
      if (resizeTimer !== undefined) window.clearTimeout(resizeTimer);
    };
  }, [subtle]);

  return <canvas ref={canvasRef} className="cm-atmosphere-starfield" aria-hidden="true" />;
}

export const TerraAtmosphere = memo(function TerraAtmosphere({ progress, interactive = false, className = '' }: TerraAtmosphereProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const pointerRef = useRef<HTMLDivElement>(null);
  const staticProgress = useMotionValue(1);
  const timeline = progress ?? staticProgress;
  const backgroundColor = useTransform(
    timeline,
    progress ? [0, .26, .54, .78, .90, 1] : [0, .25, .5, .75, 1],
    progress
      ? ['#03050a', '#04060c', '#03050a', '#010205', '#000000', '#000000']
      : ['#060a11', '#070b13', '#060a11', '#060a12', '#060a11'],
  );
  const copperX = useTransform(timeline, [0, .5, 1], ['8%', '42%', '74%']);
  const blueX = useTransform(timeline, [0, .5, 1], ['96%', '72%', '38%']);
  const bandX = useTransform(timeline, [0, 1], ['-16%', '16%']);
  const glowOpacity = useTransform(timeline, [0, .48, .72, .88, 1], progress ? [.42, .34, .18, .03, 0] : [1, 1, 1, 1, 1]);
  const bandOpacity = useTransform(timeline, [0, .62, .84, 1], progress ? [.48, .30, .05, 0] : [1, 1, 1, 1]);
  const voidOpacity = useTransform(timeline, [0, .70, .87, 1], progress ? [0, .14, .94, 1] : [0, 0, 0, 0]);
  const starsOpacity = useTransform(timeline, [0, .24, .72, .90, 1], progress ? [.72, .86, .76, .62, .56] : [.82, .98, .88, .92, .96]);
  const starsX = useTransform(timeline, [0, 1], ['0%', '-4%']);
  const names = ['cm-global-atmosphere', progress ? 'cm-terra-atmosphere--landing-cinematic' : '', className].filter(Boolean).join(' ');

  useEffect(() => {
    const pointerLayer = pointerRef.current ?? rootRef.current;
    const canRenderPointerGlow = interactive && !!pointerLayer && !window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let frame = 0;
    let pointerX = latestPointerPosition.x;
    let pointerY = latestPointerPosition.y;
    const applyPointer = () => {
      frame = 0;
      if (!canRenderPointerGlow || !pointerLayer) return;
      const xPx = pointerX * window.innerWidth;
      const yPx = pointerY * window.innerHeight;
      pointerLayer.style.transform = `translate3d(${xPx.toFixed(1)}px, ${yPx.toFixed(1)}px, 0)`;
    };
    const onPointerMove = (event: PointerEvent) => {
      if (event.pointerType === 'touch') return;
      pointerX = Math.max(0, Math.min(1, event.clientX / Math.max(window.innerWidth, 1)));
      pointerY = Math.max(0, Math.min(1, event.clientY / Math.max(window.innerHeight, 1)));
      latestPointerPosition = { x: pointerX, y: pointerY };
      if (!canRenderPointerGlow) return;
      if (!frame) frame = window.requestAnimationFrame(applyPointer);
    };

    applyPointer();
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [interactive]);

  return (
    <div ref={rootRef} className={names} aria-hidden="true">
      <motion.div className="cm-atmosphere-base" style={{ backgroundColor }} />
      <motion.div className="cm-atmosphere-glow cm-atmosphere-glow--copper" style={{ x: copperX, opacity: glowOpacity }} />
      <motion.div className="cm-atmosphere-glow cm-atmosphere-glow--blue" style={{ x: blueX, opacity: glowOpacity }} />
      <motion.div className="cm-atmosphere-band" style={{ x: bandX, opacity: bandOpacity }} />
      {progress ? <motion.div className="cm-atmosphere-void" style={{ opacity: voidOpacity }} /> : null}
      <motion.div className="cm-atmosphere-stars" style={{ opacity: starsOpacity, ...(progress ? { x: starsX } : {}) }}>
        <StarFieldCanvas subtle={Boolean(progress)} />
      </motion.div>
      {progress || interactive ? <div ref={pointerRef} className="cm-atmosphere-pointer" /> : null}
      <div className="cm-atmosphere-grain" />
    </div>
  );
});
