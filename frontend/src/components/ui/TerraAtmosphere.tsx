import { useEffect, useRef } from 'react';
import { motion, useMotionValue, useTransform, type MotionValue } from 'framer-motion';
import './TerraAtmosphere.css';

type TerraAtmosphereProps = {
  progress?: MotionValue<number>;
  className?: string;
};

function StarFieldCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d', { alpha: true });
    if (!canvas || !context) return;

    const render = () => {
      const { width, height } = canvas.getBoundingClientRect();
      if (width === 0 || height === 0) return;

      const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, width, height);

      // A stable seed keeps the star map calm while a resize redraws it.
      let seed = 0x74657272;
      const random = () => {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        return seed / 0x100000000;
      };
      const gaussian = () => {
        const u = Math.max(random(), 0.0001);
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
      };
      const count = Math.max(64, Math.min(340, Math.round((width * height) / 5800)));
      const colors = ['#f4eee8', '#e8edf5', '#d8e3f5', '#d9aa87'];

      for (let index = 0; index < count; index += 1) {
        // Most stars gather in a soft, imperfect nebula band; a few drift wide.
        const isDrifter = random() < 0.24;
        const x = isDrifter ? random() * width : width * (0.5 + gaussian() * 0.29);
        const y = isDrifter ? random() * height : height * (0.51 + gaussian() * 0.25);
        if (x < -3 || x > width + 3 || y < -3 || y > height + 3) continue;

        const radius = random() < 0.08 ? 1.05 + random() * 0.48 : 0.38 + random() * 0.52;
        context.globalAlpha = 0.28 + random() * 0.56;
        context.fillStyle = colors[Math.floor(random() * colors.length)];
        context.beginPath();
        context.arc(x, y, radius, 0, Math.PI * 2);
        context.fill();
      }

      context.globalAlpha = 1;
    };

    render();
    window.addEventListener('resize', render, { passive: true });
    return () => window.removeEventListener('resize', render);
  }, []);

  return <canvas ref={canvasRef} className="cm-atmosphere-starfield" aria-hidden="true"/>;
}

export function TerraAtmosphere({ progress, className = '' }: TerraAtmosphereProps) {
  const staticProgress = useMotionValue(1);
  const timeline = progress ?? staticProgress;
  const backgroundColor = useTransform(timeline, [0, .25, .5, .75, 1], ['#080e17', '#0b1218', '#0a101a', '#090f1b', '#0b1119']);
  const copperX = useTransform(timeline, [0, .5, 1], ['8%', '42%', '74%']);
  const blueX = useTransform(timeline, [0, .5, 1], ['96%', '72%', '38%']);
  const bandX = useTransform(timeline, [0, 1], ['-16%', '16%']);
  const starsOpacity = useTransform(timeline, [0, .24, .72, 1], [.72, .96, .82, .94]);
  const starsX = useTransform(timeline, [0, 1], ['0%', '-4%']);
  const starsY = useTransform(timeline, [0, 1], ['0%', '2%']);
  const names = ['cm-global-atmosphere', className].filter(Boolean).join(' ');

  return <div className={names} aria-hidden="true">
    <motion.div className="cm-atmosphere-base" style={{ backgroundColor }}/>
    <motion.div className="cm-atmosphere-glow cm-atmosphere-glow--copper" style={{ x: copperX }}/>
    <motion.div className="cm-atmosphere-glow cm-atmosphere-glow--blue" style={{ x: blueX }}/>
    <motion.div className="cm-atmosphere-band" style={{ x: bandX }}/>
    <motion.div className="cm-atmosphere-stars" style={{ opacity: starsOpacity, x: starsX, y: starsY }}>
      <StarFieldCanvas/>
    </motion.div>
    {progress ? <div className="cm-atmosphere-pointer"/> : null}
    <div className="cm-atmosphere-grain"/>
  </div>;
}
