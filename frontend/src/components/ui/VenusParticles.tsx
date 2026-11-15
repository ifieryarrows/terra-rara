import { memo, useEffect, useRef } from 'react';
import clsx from 'clsx';

interface VenusParticlesProps {
  className?: string;
  count?: number;
}

interface Particle {
  x: number;
  y: number;
  radius: number;
  baseAlpha: number;
  alpha: number;
  vx: number;
  vy: number;
  color: string;
  pulseSpeed: number;
  pulsePhase: number;
  wobbleSpeed: number;
  wobbleAmp: number;
  wobblePhase: number;
  age: number;
  life: number;
}

const PALETTE = [
  '#e5966b', // Core copper
  '#f5c28a', // Warm amber
  '#ffd7a8', // Golden stardust
  '#d97736', // Deep ember
  '#fff0db', // High-magnitude solar spec
];

export const VenusParticles = memo(({ className, count: targetCount }: VenusParticlesProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || typeof window === 'undefined') return;
    if (typeof navigator !== 'undefined' && /jsdom/i.test(navigator.userAgent)) return;

    let ctx: CanvasRenderingContext2D | null = null;
    try {
      ctx = canvas.getContext('2d', { alpha: true });
    } catch {
      return;
    }
    if (!ctx) return;

    let animId = 0;
    let isVisible = true;
    let isTabActive = !document.hidden;
    let width = 0;
    let height = 0;

    const prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches ?? false;

    // Detect mobile viewport for particle budgeting
    const isMobile = window.innerWidth < 768;
    const count = targetCount ?? (isMobile ? 22 : 38);

    let particles: Particle[] = [];

    const spawnParticle = (w: number, h: number, staggered: boolean): Particle => {
      const life = 240 + Math.random() * 260;
      const baseAlpha = 0.18 + Math.random() * 0.42;
      return {
        x: Math.random() * w,
        y: Math.random() * h,
        radius: 0.8 + Math.random() * 1.5,
        baseAlpha,
        alpha: staggered ? baseAlpha : 0,
        // Gentle eastward solar drift with subtle thermal updraft
        vx: 0.09 + Math.random() * 0.20,
        vy: -(0.03 + Math.random() * 0.12),
        color: PALETTE[Math.floor(Math.random() * PALETTE.length)],
        pulseSpeed: 0.012 + Math.random() * 0.022,
        pulsePhase: Math.random() * Math.PI * 2,
        wobbleSpeed: 0.007 + Math.random() * 0.015,
        wobbleAmp: 0.25 + Math.random() * 0.55,
        wobblePhase: Math.random() * Math.PI * 2,
        age: staggered ? Math.random() * life : 0,
        life,
      };
    };

    const resetParticle = (p: Particle, w: number, h: number) => {
      p.x = Math.random() * w;
      p.y = Math.random() * h;
      p.age = 0;
      p.life = 240 + Math.random() * 260;
      p.alpha = 0;
      p.pulsePhase = Math.random() * Math.PI * 2;
    };

    const initParticles = (w: number, h: number) => {
      particles = Array.from({ length: count }, () => spawnParticle(w, h, true));
    };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const nextW = Math.round(rect.width);
      const nextH = Math.round(rect.height);
      if (nextW === 0 || nextH === 0) return;

      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      if (width !== nextW || height !== nextH) {
        width = nextW;
        height = nextH;
        canvas.width = Math.round(nextW * dpr);
        canvas.height = Math.round(nextH * dpr);
        ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (particles.length === 0) {
          initParticles(width, height);
        }
      }
    };

    resize();

    let frame = 0;
    const render = () => {
      if (!ctx || width === 0 || height === 0) return;

      frame += 1;
      ctx.clearRect(0, 0, width, height);

      const marginX = Math.max(36, width * 0.14);
      const marginY = Math.max(28, height * 0.16);

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        if (!prefersReducedMotion) {
          p.x += p.vx + Math.cos(frame * p.wobbleSpeed + p.wobblePhase) * p.wobbleAmp * 0.1;
          p.y += p.vy;
          p.age += 1;

          if (p.age >= p.life || p.x > width + 10 || p.x < -10 || p.y < -10 || p.y > height + 10) {
            resetParticle(p, width, height);
          }

          // Smooth fade-in and fade-out lifecycle envelope
          const lifeRatio = Math.min(1, Math.max(0, p.age / p.life));
          const lifeFade = Math.pow(Math.sin(Math.PI * lifeRatio), 0.72);

          // Soft spatial edge feathering so particles dissolve before hitting bounds
          const edgeX = Math.min(1, Math.max(0, Math.min(p.x, width - p.x) / marginX));
          const edgeY = Math.min(1, Math.max(0, Math.min(p.y, height - p.y) / marginY));
          const edgeFade = edgeX * edgeY;

          const shimmer = Math.max(
            0.12,
            Math.min(0.68, p.baseAlpha + Math.sin(frame * p.pulseSpeed + p.pulsePhase) * 0.14)
          );
          p.alpha = shimmer * lifeFade * edgeFade;
        }

        if (p.alpha <= 0.005) continue;

        ctx.globalAlpha = p.alpha;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fill();
      }

      if (!prefersReducedMotion && isVisible && isTabActive) {
        animId = requestAnimationFrame(render);
      }
    };

    // Initial render
    render();

    // Visibility Observer to completely pause CPU when hero is offscreen
    let observer: IntersectionObserver | null = null;
    if (typeof IntersectionObserver !== 'undefined') {
      observer = new IntersectionObserver(
        ([entry]) => {
          const wasVisible = isVisible;
          isVisible = entry.isIntersecting;
          if (!wasVisible && isVisible && isTabActive && !prefersReducedMotion) {
            cancelAnimationFrame(animId);
            animId = requestAnimationFrame(render);
          }
        },
        { threshold: 0.05 }
      );
      observer.observe(canvas);
    }

    // Visibility change handler for inactive browser tabs
    const handleVisibilityChange = () => {
      isTabActive = !document.hidden;
      if (isTabActive && isVisible && !prefersReducedMotion) {
        cancelAnimationFrame(animId);
        animId = requestAnimationFrame(render);
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Resize observer
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => {
        resize();
        if (prefersReducedMotion) render();
      });
      ro.observe(canvas);
    }

    return () => {
      cancelAnimationFrame(animId);
      observer?.disconnect();
      ro?.disconnect();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [targetCount]);

  return (
    <div
      className={clsx(
        "cm-venus-atmosphere pointer-events-none absolute inset-0 z-0 overflow-hidden",
        className
      )}
      aria-hidden="true"
    >
      <canvas
        ref={canvasRef}
        className="block h-full w-full opacity-85"
      />
    </div>
  );
});

VenusParticles.displayName = 'VenusParticles';
