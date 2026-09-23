import { useEffect, useRef } from 'react';
import { useReducedMotion } from 'framer-motion';
import './TerraCursor.css';

type DustMote = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  alpha: number;
  age: number;
  lifetime: number;
  rotation: number;
  spin: number;
  color: string;
  sparkle: boolean;
};

const DUST_COLORS = ['#946047', '#a66d4e', '#bb8060'];
const DUST_FRAME_INTERVAL_MS = 32;
const randomBetween = (min: number, max: number) => min + Math.random() * (max - min);

function makeCursor(fill: string) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><circle cx="16" cy="16" r="10.5" fill="none" stroke="#e6a47a" stroke-opacity=".42" stroke-width="1"/><path d="M16 7.5 17.7 14.3 24.5 16 17.7 17.7 16 24.5 14.3 17.7 7.5 16 14.3 14.3Z" fill="${fill}"/><circle cx="16" cy="16" r="1.2" fill="#fff"/></svg>`;
  return `url("data:image/svg+xml,${encodeURIComponent(svg)}") 16 16, auto`;
}

function drawSparkle(context: CanvasRenderingContext2D, mote: DustMote, life: number) {
  const size = mote.size * (.7 + life * .3);
  context.save();
  context.translate(mote.x, mote.y);
  context.rotate(mote.rotation + mote.spin * mote.age / mote.lifetime);
  context.beginPath();
  context.moveTo(0, -size * 1.7);
  context.lineTo(size * .3, -size * .3);
  context.lineTo(size * 1.7, 0);
  context.lineTo(size * .3, size * .3);
  context.lineTo(0, size * 1.7);
  context.lineTo(-size * .3, size * .3);
  context.lineTo(-size * 1.7, 0);
  context.lineTo(-size * .3, -size * .3);
  context.closePath();
  context.fill();
  context.restore();
}

export function TerraCursor() {
  const reducedMotion = useReducedMotion();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const root = document.getElementById('root');
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d', { alpha: true });
    if (!root || !canvas || !context || reducedMotion !== false || !window.matchMedia('(pointer: fine)').matches) return;

    const particles: DustMote[] = [];
    let width = 0;
    let height = 0;
    let frame = 0;
    let previousFrame = 0;
    let previousPoint: { x: number; y: number } | null = null;
    let distanceSinceMote = 0;

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      // Keep the dust backing store at 1x; the soft texture does not need a full DPR canvas.
      canvas.width = width;
      canvas.height = height;
      context.setTransform(1, 0, 0, 1, 0, 0);
    };

    const spawnDust = (x: number, y: number, directionX: number, directionY: number) => {
      const roll = Math.random();
      const count = roll < .72 ? 1 : roll < .98 ? 2 : 3;
      for (let index = 0; index < count; index += 1) {
        const sparkle = Math.random() < .07;
        const offset = randomBetween(2, 6);
        particles.push({
          x: x - directionX * offset + randomBetween(-3, 3),
          y: y - directionY * offset + randomBetween(-3, 3),
          vx: -directionX * randomBetween(.07, .24) + randomBetween(-.2, .2),
          vy: -directionY * randomBetween(.07, .24) + randomBetween(-.18, .24),
          size: sparkle ? randomBetween(1, 1.45) : randomBetween(.9, 2),
          alpha: sparkle ? randomBetween(.38, .52) : randomBetween(.22, .36),
          age: 0,
          lifetime: randomBetween(700, 1050),
          rotation: randomBetween(0, Math.PI * 2),
          spin: randomBetween(-.035, .035),
          color: sparkle ? '#f3eee6' : DUST_COLORS[Math.floor(Math.random() * DUST_COLORS.length)],
          sparkle,
        });
      }
      if (particles.length > 84) particles.splice(0, particles.length - 84);
    };

    const draw = (now: number) => {
      if (previousFrame && now - previousFrame < DUST_FRAME_INTERVAL_MS) {
        frame = window.requestAnimationFrame(draw);
        return;
      }
      frame = 0;
      const elapsed = previousFrame ? Math.min((now - previousFrame) / 16.67, 2.5) : 1;
      previousFrame = now;
      let minX = width;
      let minY = height;
      let maxX = 0;
      let maxY = 0;
      const include = (x: number, y: number, size: number) => {
        const padding = Math.max(3, size * 1.8);
        minX = Math.min(minX, x - padding);
        minY = Math.min(minY, y - padding);
        maxX = Math.max(maxX, x + padding);
        maxY = Math.max(maxY, y + padding);
      };

      for (let index = particles.length - 1; index >= 0; index -= 1) {
        const mote = particles[index];
        include(mote.x, mote.y, mote.size);
        mote.age += elapsed * 16.67;
        if (mote.age >= mote.lifetime) {
          particles.splice(index, 1);
          continue;
        }
        mote.x += mote.vx * elapsed;
        mote.y += mote.vy * elapsed;
        mote.vy += .005 * elapsed;
        include(mote.x, mote.y, mote.size);
      }

      if (minX < width && minY < height && maxX > 0 && maxY > 0) {
        const left = Math.max(0, minX);
        const top = Math.max(0, minY);
        const right = Math.min(width, maxX);
        const bottom = Math.min(height, maxY);
        context.clearRect(left, top, right - left, bottom - top);

        for (const mote of particles) {
          const life = 1 - mote.age / mote.lifetime;
          context.globalAlpha = mote.alpha * life ** 1.6;
          context.fillStyle = mote.color;
          if (mote.sparkle) {
            drawSparkle(context, mote, life);
          } else {
            context.beginPath();
            context.ellipse(mote.x, mote.y, mote.size, mote.size * .72, mote.rotation + mote.spin * mote.age / mote.lifetime, 0, Math.PI * 2);
            context.fill();
          }
        }
        context.globalAlpha = 1;
      }

      if (particles.length) frame = window.requestAnimationFrame(draw);
      else previousFrame = 0;
    };

    const schedule = () => { if (!frame) frame = window.requestAnimationFrame(draw); };
    const resetTrailOrigin = () => {
      previousPoint = null;
      distanceSinceMote = 0;
    };
    const onPointerMove = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node) || !root.contains(target) || event.pointerType === 'touch') {
        resetTrailOrigin();
        return;
      }
      const element = target instanceof Element ? target : null;
      if (element?.closest('input, textarea, select, [contenteditable="true"]')) {
        resetTrailOrigin();
        return;
      }

      const x = event.clientX;
      const y = event.clientY;
      if (!previousPoint) {
        previousPoint = { x, y };
        return;
      }
      const dx = x - previousPoint.x;
      const dy = y - previousPoint.y;
      const distance = Math.hypot(dx, dy);
      if (distance === 0) return;
      const spacing = 15;
      const moteCount = Math.floor((distanceSinceMote + distance) / spacing);
      const directionX = dx / distance;
      const directionY = dy / distance;
      for (let index = 0; index < moteCount; index += 1) {
        const along = spacing - distanceSinceMote + index * spacing;
        const ratio = along / distance;
        spawnDust(previousPoint.x + dx * ratio, previousPoint.y + dy * ratio, directionX, directionY);
      }
      distanceSinceMote = (distanceSinceMote + distance) % spacing;
      previousPoint = { x, y };
      if (moteCount) schedule();
    };
    const onPointerLeave = () => resetTrailOrigin();
    const onBlur = () => {
      resetTrailOrigin();
      particles.length = 0;
      if (frame) window.cancelAnimationFrame(frame);
      frame = 0;
      context.clearRect(0, 0, width, height);
    };

    resize();
    root.style.setProperty('--cm-terra-cursor', makeCursor('#f3eee6'));
    root.style.setProperty('--cm-terra-cursor-interactive', makeCursor('#f1bc97'));
    root.classList.add('cm-terra-cursor-enabled');
    window.addEventListener('resize', resize, { passive: true });
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('blur', onBlur);
    root.addEventListener('pointerleave', onPointerLeave, { passive: true });
    return () => {
      root.classList.remove('cm-terra-cursor-enabled');
      root.style.removeProperty('--cm-terra-cursor');
      root.style.removeProperty('--cm-terra-cursor-interactive');
      window.removeEventListener('resize', resize);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('blur', onBlur);
      root.removeEventListener('pointerleave', onPointerLeave);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [reducedMotion]);

  return <canvas ref={canvasRef} className="cm-terra-cursor-dust" aria-hidden="true"/>;
}
