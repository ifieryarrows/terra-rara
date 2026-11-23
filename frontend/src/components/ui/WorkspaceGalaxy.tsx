import { useEffect, useRef } from 'react';
import {
  clamp,
  createWebglTransitionRenderer,
  drawGalaxyCpu2D,
  generateTransitionGalaxy,
  getSharedGalaxyTime,
  syncSharedGalaxyClock,
  WORKSPACE_GALAXY_ALPHA,
  WORKSPACE_GALAXY_TILT,
  WORKSPACE_SPIN_RATE,
  type GalaxyParticlesData,
  type LiveGalaxyState,
} from './galaxyEngine';

export default function WorkspaceGalaxy() {
  const webglCanvasRef = useRef<HTMLCanvasElement>(null);
  const fallbackCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (typeof navigator !== 'undefined' && /jsdom/i.test(navigator.userAgent)) return;

    const webglCanvas = webglCanvasRef.current;
    const fallbackCanvas = fallbackCanvasRef.current;
    if (!webglCanvas || !fallbackCanvas) return;

    const webglData = generateTransitionGalaxy(4800);
    let glRenderer = createWebglTransitionRenderer(webglCanvas, webglData);
    let fallbackData: GalaxyParticlesData | null = null;
    let ctx2d: CanvasRenderingContext2D | null = null;

    if (!glRenderer) {
      fallbackData = generateTransitionGalaxy(1400);
      try {
        ctx2d = fallbackCanvas.getContext('2d', { alpha: true });
      } catch {
        ctx2d = null;
      }
    }
    if (!glRenderer && !ctx2d) return;

    webglCanvas.style.display = glRenderer ? 'block' : 'none';
    fallbackCanvas.style.display = glRenderer ? 'none' : 'block';

    const motionQuery = typeof window.matchMedia === 'function'
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;
    let prefersReducedMotion = motionQuery?.matches ?? false;
    let isSkippedByContentVisibility = false;
    let width = window.innerWidth;
    let height = window.innerHeight;
    let pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    let frame = 0;
    let lastTickNow = performance.now();
    let elapsedTime = getSharedGalaxyTime(lastTickNow);

    const resize = () => {
      const target = glRenderer ? webglCanvas : fallbackCanvas;
      const rect = target.getBoundingClientRect();
      width = rect.width > 1 ? rect.width : window.innerWidth;
      height = rect.height > 1 ? rect.height : window.innerHeight;
      pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
      target.width = Math.max(1, Math.round(width * pixelRatio));
      target.height = Math.max(1, Math.round(height * pixelRatio));
      if (ctx2d) {
        ctx2d.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      }
    };

    const advanceClock = (now: number) => {
      const activePhase = document.documentElement.dataset.cmTransitionPhase;
      if (activePhase && activePhase !== 'reveal') {
        lastTickNow = now;
        elapsedTime = getSharedGalaxyTime(now);
        return elapsedTime;
      }
      const dt = clamp(now - lastTickNow, 0, 100);
      lastTickNow = now;
      elapsedTime += dt * WORKSPACE_SPIN_RATE;
      syncSharedGalaxyClock(elapsedTime, now);
      return elapsedTime;
    };

    const drawOnce = (now: number) => {
      const elapsed = advanceClock(now);
      if (glRenderer) {
        glRenderer.draw({
          width,
          height,
          pixelRatio,
          mode: 2,
          progress: 1,
          time: elapsed,
          fadeIn: false,
          initTilt: WORKSPACE_GALAXY_TILT,
          persistEnd: true,
          endAlpha: WORKSPACE_GALAXY_ALPHA,
          fromWorkspace: false,
        });
      } else if (ctx2d && fallbackData) {
        drawGalaxyCpu2D(ctx2d, fallbackData, {
          width,
          height,
          mode: 2,
          progress: 1,
          time: elapsed,
          fadeIn: false,
          initTilt: WORKSPACE_GALAXY_TILT,
          persistEnd: true,
          endAlpha: WORKSPACE_GALAXY_ALPHA,
          fromWorkspace: false,
        });
      }
    };

    const canAnimate = () => !prefersReducedMotion && !document.hidden && !isSkippedByContentVisibility;

    const tick = (now: number) => {
      frame = 0;
      drawOnce(now);
      if (canAnimate()) {
        frame = window.requestAnimationFrame(tick);
      }
    };

    const schedule = () => {
      if (!frame) {
        frame = window.requestAnimationFrame(tick);
      }
    };

    const onVisibilityChange = () => {
      if (canAnimate()) {
        lastTickNow = performance.now();
        schedule();
      } else if (frame) {
        window.cancelAnimationFrame(frame);
        frame = 0;
      }
    };

    const onMotionChange = (event: MediaQueryListEvent) => {
      prefersReducedMotion = event.matches;
      if (prefersReducedMotion) {
        if (frame) {
          window.cancelAnimationFrame(frame);
          frame = 0;
        }
        drawOnce(performance.now());
      } else if (canAnimate()) {
        lastTickNow = performance.now();
        schedule();
      }
    };

    const atmosphereHost = webglCanvas.parentElement;
    const onContentVisibilityChange = (event: Event) => {
      const skipped = Boolean((event as Event & { skipped?: boolean }).skipped);
      isSkippedByContentVisibility = skipped;
      if (skipped) {
        if (frame) {
          window.cancelAnimationFrame(frame);
          frame = 0;
        }
      } else if (canAnimate()) {
        lastTickNow = performance.now();
        schedule();
      }
    };

    const onResize = () => {
      resize();
      if (prefersReducedMotion) {
        drawOnce(performance.now());
      } else {
        schedule();
      }
    };

    const onGalaxyState = (event: Event) => {
      const detail = (event as CustomEvent<{ state?: LiveGalaxyState }>).detail;
      if (!detail || detail.state) return;
      detail.state = {
        progress: 1,
        pointerX: 0.5,
        pointerY: 0.5,
        elapsed: elapsedTime,
        fromWorkspace: true,
        tilt: WORKSPACE_GALAXY_TILT,
      };
    };

    resize();
    if (prefersReducedMotion) {
      drawOnce(performance.now());
    } else {
      schedule();
    }

    const resizeObserver = typeof ResizeObserver !== 'undefined' && atmosphereHost
      ? new ResizeObserver(onResize)
      : null;
    if (resizeObserver && atmosphereHost) {
      resizeObserver.observe(atmosphereHost);
    }

    window.addEventListener('resize', onResize, { passive: true });
    document.addEventListener('visibilitychange', onVisibilityChange);
    atmosphereHost?.addEventListener('contentvisibilityautostatechange', onContentVisibilityChange);
    motionQuery?.addEventListener('change', onMotionChange);
    window.addEventListener('cm:galaxy-state', onGalaxyState);

    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener('resize', onResize);
      document.removeEventListener('visibilitychange', onVisibilityChange);
      atmosphereHost?.removeEventListener('contentvisibilityautostatechange', onContentVisibilityChange);
      motionQuery?.removeEventListener('change', onMotionChange);
      window.removeEventListener('cm:galaxy-state', onGalaxyState);
      if (frame) window.cancelAnimationFrame(frame);
      if (glRenderer) {
        glRenderer.dispose();
        glRenderer = null;
      }
    };
  }, []);

  return (
    <>
      <canvas ref={webglCanvasRef} className="cm-workspace-galaxy" aria-hidden="true" />
      <canvas ref={fallbackCanvasRef} className="cm-workspace-galaxy" aria-hidden="true" />
    </>
  );
}
