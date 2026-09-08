import { useEffect, useRef } from 'react';
import type { MotionValue } from 'framer-motion';

type Point = { x: number; y: number };

const PARTICLE_COUNT = 420;

function noise(index: number, salt: number) {
  const value = Math.sin((index + 1) * (12.9898 + salt * 17.17)) * 43758.5453;
  return value - Math.floor(value);
}

function copperForm(): Point[] {
  return Array.from({ length: PARTICLE_COUNT }, (_, index) => {
    const angle = Math.PI * .23 + (index / (PARTICLE_COUNT - 1)) * Math.PI * 1.54;
    const band = (noise(index, 1) - .5) * .055;
    const radius = .205 + band;
    return {
      x: .72 + Math.cos(angle) * radius,
      y: .49 + Math.sin(angle) * radius * 1.08,
    };
  });
}

function dispersedField(): Point[] {
  return Array.from({ length: PARTICLE_COUNT }, (_, index) => {
    const right = noise(index, 2) > .48;
    return {
      x: right ? .79 + noise(index, 3) * .19 : .015 + noise(index, 4) * .21,
      y: .08 + noise(index, 5) * .84,
    };
  });
}

const marketCells = [
  { x: .5, y: .21, w: .21, h: .43 },
  { x: .72, y: .21, w: .24, h: .2 },
  { x: .72, y: .43, w: .115, h: .21 },
  { x: .845, y: .43, w: .115, h: .21 },
  { x: .5, y: .66, w: .27, h: .14 },
  { x: .78, y: .66, w: .18, h: .14 },
];

function marketField(): Point[] {
  return Array.from({ length: PARTICLE_COUNT }, (_, index) => {
    const cell = marketCells[index % marketCells.length];
    const edge = noise(index, 6);
    if (edge < .5) {
      return { x: cell.x + noise(index, 7) * cell.w, y: cell.y + (edge < .25 ? 0 : cell.h) };
    }
    return { x: cell.x + (edge < .75 ? 0 : cell.w), y: cell.y + noise(index, 8) * cell.h };
  });
}

const networkNodes = [
  [.52, .28], [.5, .7], [.66, .18], [.64, .78], [.79, .34], [.8, .67], [.93, .49],
] as const;

function intelligenceNetwork(): Point[] {
  return Array.from({ length: PARTICLE_COUNT }, (_, index) => {
    const from = networkNodes[index % (networkNodes.length - 1)];
    const to = index % 3 === 0 ? networkNodes[networkNodes.length - 1] : networkNodes[(index + 2) % networkNodes.length];
    const t = noise(index, 9);
    const bend = Math.sin(t * Math.PI) * (noise(index, 10) - .5) * .08;
    return {
      x: from[0] + (to[0] - from[0]) * t,
      y: from[1] + (to[1] - from[1]) * t + bend,
    };
  });
}

function forecastPath(): Point[] {
  return Array.from({ length: PARTICLE_COUNT }, (_, index) => {
    const t = index / (PARTICLE_COUNT - 1);
    const uncertainty = Math.max(0, t - .62) * (noise(index, 11) - .5) * .34;
    return {
      x: .44 + t * .53,
      y: .66 - t * .32 + Math.sin(t * Math.PI * 4.2) * .04 + uncertainty,
    };
  });
}

function evidenceMark(): Point[] {
  return Array.from({ length: PARTICLE_COUNT }, (_, index) => {
    if (index < 260) {
      const angle = index / 260 * Math.PI * 2;
      const radius = .205 + (noise(index, 12) - .5) * .025;
      return { x: .75 + Math.cos(angle) * radius, y: .49 + Math.sin(angle) * radius };
    }
    const t = (index - 260) / 159;
    if (t < .43) {
      const local = t / .43;
      return { x: .65 + local * .075, y: .5 + local * .075 };
    }
    const local = (t - .43) / .57;
    return { x: .725 + local * .15, y: .575 - local * .21 };
  });
}

const keyframes = [
  { at: 0, points: copperForm() },
  { at: .16, points: copperForm() },
  { at: .27, points: dispersedField() },
  { at: .39, points: marketField() },
  { at: .56, points: intelligenceNetwork() },
  { at: .75, points: forecastPath() },
  { at: .92, points: evidenceMark() },
  { at: 1, points: evidenceMark() },
];

function ease(value: number) {
  return value * value * (3 - 2 * value);
}

export function ParticleWorld({ progress }: { progress: MotionValue<number> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;

    let width = 0;
    let height = 0;
    let frame = 0;
    let pointerX = -1000;
    let pointerY = -1000;

    const resize = () => {
      const bounds = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
      width = bounds.width;
      height = bounds.height;
      canvas.width = Math.max(1, Math.round(width * ratio));
      canvas.height = Math.max(1, Math.round(height * ratio));
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      schedule();
    };

    const draw = () => {
      frame = 0;
      const value = Math.max(0, Math.min(1, progress.get()));
      let frameIndex = keyframes.findIndex(keyframe => keyframe.at >= value);
      if (frameIndex <= 0) frameIndex = 1;
      const previous = keyframes[frameIndex - 1];
      const next = keyframes[frameIndex];
      const amount = ease((value - previous.at) / Math.max(.001, next.at - previous.at));

      context.clearRect(0, 0, width, height);
      context.save();
      context.shadowBlur = 13;
      context.shadowColor = 'rgba(230, 164, 122, .42)';
      context.fillStyle = 'rgba(230, 164, 122, .8)';
      context.beginPath();

      for (let index = 0; index < PARTICLE_COUNT; index += 1) {
        const from = previous.points[index];
        const to = next.points[index];
        let x = (from.x + (to.x - from.x) * amount) * width;
        let y = (from.y + (to.y - from.y) * amount) * height;
        const dx = x - pointerX;
        const dy = y - pointerY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const reach = Math.min(width, height) * .095;
        if (distance > 0 && distance < reach) {
          const force = (1 - distance / reach) * 18;
          x += dx / distance * force;
          y += dy / distance * force;
        }
        const drift = Math.sin(value * 22 + index * .71) * 1.2;
        const radius = index % 11 === 0 ? 1.9 : 1.05;
        context.moveTo(x + radius + drift, y);
        context.arc(x + drift, y, radius, 0, Math.PI * 2);
      }
      context.fill();
      context.restore();

      if (value > .6) {
        context.fillStyle = `rgba(155, 188, 251, ${Math.min(.48, (value - .6) * 1.2)})`;
        context.beginPath();
        for (let index = 6; index < PARTICLE_COUNT; index += 17) {
          const from = previous.points[index];
          const to = next.points[index];
          const x = (from.x + (to.x - from.x) * amount) * width;
          const y = (from.y + (to.y - from.y) * amount) * height;
          context.moveTo(x + 2.2, y);
          context.arc(x, y, 2.2, 0, Math.PI * 2);
        }
        context.fill();
      }
    };

    const schedule = () => {
      if (!frame) frame = window.requestAnimationFrame(draw);
    };
    const onPointerMove = (event: PointerEvent) => {
      pointerX = event.clientX;
      pointerY = event.clientY;
      schedule();
    };
    const unsubscribe = progress.on('change', schedule);
    window.addEventListener('resize', resize, { passive: true });
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    resize();

    return () => {
      unsubscribe();
      if (frame) window.cancelAnimationFrame(frame);
      window.removeEventListener('resize', resize);
      window.removeEventListener('pointermove', onPointerMove);
    };
  }, [progress]);

  return <canvas ref={canvasRef} className="cm-particle-world" aria-hidden="true"/>;
}
