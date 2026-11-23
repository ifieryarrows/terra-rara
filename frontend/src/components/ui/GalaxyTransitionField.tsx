import { useEffect, useRef } from 'react';
import {
  clamp,
  clamp01,
  createWebglTransitionRenderer,
  drawGalaxyCpu2D,
  generateTransitionGalaxy,
  getSharedGalaxyTime,
  syncSharedGalaxyClock,
  WORKSPACE_GALAXY_ALPHA,
  type GalaxyParticlesData,
  type LiveGalaxyState,
  type TransitionPhase,
} from './galaxyEngine';

function queryLiveGalaxyState(): LiveGalaxyState | null {
  const detail: { state?: LiveGalaxyState } = {};
  window.dispatchEvent(new CustomEvent('cm:galaxy-state', { detail }));
  return detail.state ?? null;
}

export default function GalaxyTransitionField({
  phase,
  reducedMotion,
  scatterMs,
  assembleMs,
  releaseMs,
  persistEndGalaxy = false,
}: {
  phase: TransitionPhase;
  reducedMotion: boolean;
  scatterMs: number;
  assembleMs: number;
  releaseMs: number;
  persistEndGalaxy?: boolean;
}) {
  const webglCanvasRef = useRef<HTMLCanvasElement>(null);
  const fallbackCanvasRef = useRef<HTMLCanvasElement>(null);
  const phaseRef = useRef<TransitionPhase>(phase);
  const persistEndRef = useRef<boolean>(persistEndGalaxy);
  const inboundStartedAtRef = useRef<number | null>(null);
  const releaseStartedAtRef = useRef<number | null>(null);
  const revealStartedAtRef = useRef<number | null>(null);
  const scheduleDrawRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    persistEndRef.current = persistEndGalaxy;
  }, [persistEndGalaxy]);

  useEffect(() => {
    phaseRef.current = phase;
    const now = performance.now();
    if (phase === 'scatter' && inboundStartedAtRef.current === null) {
      inboundStartedAtRef.current = now;
    } else if (phase === 'assemble' && inboundStartedAtRef.current === null) {
      inboundStartedAtRef.current = now - scatterMs;
    } else if (phase === 'release') {
      releaseStartedAtRef.current = now;
    } else if (phase === 'reveal') {
      revealStartedAtRef.current = now;
    }
    scheduleDrawRef.current?.();
  }, [phase, scatterMs]);

  useEffect(() => {
    if (reducedMotion || (typeof navigator !== 'undefined' && /jsdom/i.test(navigator.userAgent))) {
      return;
    }

    const webglCanvas = webglCanvasRef.current;
    const fallbackCanvas = fallbackCanvasRef.current;
    if (!webglCanvas || !fallbackCanvas) return;

    const live = queryLiveGalaxyState();
    const hasActiveMilkyWay = Boolean(live && live.progress >= 0.78);
    const fromWorkspace = Boolean(live?.fromWorkspace);
    const ptrDeltaX = live ? live.pointerX - 0.5 : 0;
    const ptrDeltaY = live ? live.pointerY - 0.5 : 0;
    const initPitch = clamp(0.76 - ptrDeltaY * 0.96, 0.28, 1.28);
    const initTilt: readonly [number, number, number] = live?.tilt ?? [
      initPitch - Math.PI / 2,
      hasActiveMilkyWay ? ptrDeltaX * 1.16 : -0.14,
      hasActiveMilkyWay ? -0.16 + ptrDeltaX * 0.24 : -0.16,
    ];
    const baseElapsed = live ? live.elapsed : 0;

    const webglData = generateTransitionGalaxy(4800);
    let glRenderer = createWebglTransitionRenderer(webglCanvas, webglData);
    let fallbackData: GalaxyParticlesData | null = null;
    let ctx2d: CanvasRenderingContext2D | null = null;

    if (!glRenderer) {
      fallbackData = generateTransitionGalaxy(1400);
      ctx2d = fallbackCanvas.getContext('2d', { alpha: true });
    }
    if (!glRenderer && !ctx2d) return;

    webglCanvas.style.display = glRenderer ? 'block' : 'none';
    fallbackCanvas.style.display = glRenderer ? 'none' : 'block';

    let width = window.innerWidth;
    let height = window.innerHeight;
    let pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    let frame = 0;

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
      const target = glRenderer ? webglCanvas : fallbackCanvas;
      target.width = Math.max(1, Math.round(width * pixelRatio));
      target.height = Math.max(1, Math.round(height * pixelRatio));
      if (ctx2d) {
        ctx2d.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      }
    };

    const mountTime = performance.now();
    if (inboundStartedAtRef.current === null && (phaseRef.current === 'scatter' || phaseRef.current === 'assemble')) {
      inboundStartedAtRef.current = phaseRef.current === 'assemble' ? mountTime - scatterMs : mountTime;
    }

    const totalInboundMs = scatterMs + assembleMs;
    let signaledReady = false;

    const drawFrame = (now: number) => {
      frame = 0;
      const currentPhase = phaseRef.current;
      const persistEnd = persistEndRef.current;
      const elapsedNow = baseElapsed + (now - mountTime);

      if (currentPhase === 'reveal') {
        if (!persistEnd) {
          if (glRenderer) {
            glRenderer.draw({
              width,
              height,
              pixelRatio,
              mode: 2,
              progress: 1,
              time: baseElapsed,
              fadeIn: false,
              initTilt,
              persistEnd: false,
              endAlpha: WORKSPACE_GALAXY_ALPHA,
              fromWorkspace: false,
            });
          } else {
            ctx2d?.clearRect(0, 0, width, height);
          }
          return;
        }

        const revealTime = getSharedGalaxyTime(now);
        if (glRenderer) {
          glRenderer.draw({
            width,
            height,
            pixelRatio,
            mode: 2,
            progress: 1,
            time: revealTime,
            fadeIn: false,
            initTilt,
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
            time: revealTime,
            fadeIn: false,
            initTilt,
            persistEnd: true,
            endAlpha: WORKSPACE_GALAXY_ALPHA,
            fromWorkspace: false,
          });
        }

        const revealAge = now - (revealStartedAtRef.current ?? now);
        if (revealAge < 600) {
          schedule();
        }
        return;
      }

      syncSharedGalaxyClock(elapsedNow, now);

      let mode = 1;
      let progress = 1;
      if (currentPhase === 'scatter' || currentPhase === 'assemble') {
        mode = 0;
        const originNow = inboundStartedAtRef.current ?? now;
        progress = clamp01((now - originNow) / totalInboundMs);
      } else if (currentPhase === 'release') {
        mode = 2;
        const originNow = releaseStartedAtRef.current ?? now;
        progress = clamp01((now - originNow) / releaseMs);
      }

      const activeFromWorkspace = mode === 0 && fromWorkspace;
      if (glRenderer) {
        glRenderer.draw({
          width,
          height,
          pixelRatio,
          mode,
          progress,
          time: elapsedNow,
          fadeIn: !hasActiveMilkyWay,
          initTilt,
          persistEnd,
          endAlpha: WORKSPACE_GALAXY_ALPHA,
          fromWorkspace: activeFromWorkspace,
        });
      } else if (ctx2d && fallbackData) {
        drawGalaxyCpu2D(ctx2d, fallbackData, {
          width,
          height,
          mode,
          progress,
          time: elapsedNow,
          fadeIn: !hasActiveMilkyWay,
          initTilt,
          persistEnd,
          endAlpha: WORKSPACE_GALAXY_ALPHA,
          fromWorkspace: activeFromWorkspace,
        });
      }

      if (!signaledReady) {
        signaledReady = true;
        document.documentElement.dataset.cmGalaxyActive = 'true';
      }

      if (currentPhase === 'scatter' || currentPhase === 'assemble' || currentPhase === 'release') {
        schedule();
      }
    };

    const schedule = () => {
      if (!frame) frame = window.requestAnimationFrame(drawFrame);
    };
    scheduleDrawRef.current = schedule;

    resize();
    window.addEventListener('resize', resize, { passive: true });
    schedule();

    return () => {
      scheduleDrawRef.current = null;
      window.removeEventListener('resize', resize);
      if (frame) window.cancelAnimationFrame(frame);
      if (glRenderer) {
        glRenderer.dispose();
        glRenderer = null;
      }
      delete document.documentElement.dataset.cmGalaxyActive;
    };
  }, [reducedMotion, scatterMs, assembleMs, releaseMs]);

  if (reducedMotion) return null;

  return (
    <>
      <canvas ref={webglCanvasRef} className="cm-logo-transition__particles" aria-hidden="true" />
      <canvas ref={fallbackCanvasRef} className="cm-logo-transition__particles" aria-hidden="true" />
    </>
  );
}
