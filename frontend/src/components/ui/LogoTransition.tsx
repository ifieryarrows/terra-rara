import { createContext, lazy, Suspense, useCallback, useContext, useEffect, useLayoutEffect, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode } from 'react';
import { Link, useLocation, useNavigate, type LinkProps } from 'react-router-dom';
import { useReducedMotion } from 'framer-motion';
import { BrandMark } from './BrandMark';
import { TERRA_RARA_MARK_STAR_PATH } from './brand-mark-geometry';

const TransitionAtmosphere = lazy(() => import('./TerraAtmosphere').then(module => ({ default: module.TerraAtmosphere })));
const transitionStylesReady = import('./LogoTransition.css').then(() => true, () => false);

type TransitionPhase = 'scatter' | 'assemble' | 'hold' | 'release' | 'reveal';
type TransitionOrigin = { left: number; top: number; width: number; height: number };
type Particle = {
  targetX: number;
  targetY: number;
  sourceX: number;
  sourceY: number;
  scatterX: number;
  scatterY: number;
  size: number;
  alpha: number;
  color: string;
  delay: number;
  twist: number;
  rotation: number;
};

const SCATTER_MS = 420;
const ASSEMBLE_MS = 1_520;
const SETTLE_MS = 520;
const RELEASE_MS = 1_520;
// Keep the transparent final frame in place while destination content enters.
const REVEAL_MS = 2_000;
const ABOUT_REVEAL_MS = 1_200;
const DASHBOARD_READY_TIMEOUT_MS = 15_000;
const LOGO_SIZE = 224;

const coordinates = (TERRA_RARA_MARK_STAR_PATH.match(/-?\d*\.?\d+/g) ?? []).map(Number);
const starVertices = Array.from({ length: Math.floor(coordinates.length / 2) }, (_, index) => ({
  x: coordinates[index * 2],
  y: coordinates[index * 2 + 1],
}));

function createRandom(seed: number) {
  let value = seed >>> 0;
  return () => {
    value = (value * 1_664_525 + 1_013_904_223) >>> 0;
    return value / 4_294_967_296;
  };
}

function isInsideStar(x: number, y: number) {
  let inside = false;
  for (let index = 0, previous = starVertices.length - 1; index < starVertices.length; previous = index, index += 1) {
    const currentPoint = starVertices[index];
    const previousPoint = starVertices[previous];
    const crosses = currentPoint.y > y !== previousPoint.y > y
      && x < ((previousPoint.x - currentPoint.x) * (y - currentPoint.y)) / (previousPoint.y - currentPoint.y) + currentPoint.x;
    if (crosses) inside = !inside;
  }
  return inside;
}

function createParticles(width: number, height: number, origin: TransitionOrigin | null): Particle[] {
  const random = createRandom(0x7e22a4);
  const size = Math.min(LOGO_SIZE, width * .7, height * .46);
  const scale = size / 48;
  const centerX = width / 2;
  const centerY = height / 2;
  const source = origin && origin.width > 0 && origin.height > 0
    ? origin
    : { left: centerX - 18, top: centerY - 18, width: 36, height: 36 };
  const particles: Particle[] = [];
  const starPoints: Array<{ x: number; y: number }> = [];

  for (let attempt = 0; attempt < 2_500 && starPoints.length < 104; attempt += 1) {
    const x = 11.25 + random() * 25.5;
    const y = 11.25 + random() * 25.5;
    if (isInsideStar(x, y)) starPoints.push({ x, y });
  }

  const addParticle = (x: number, y: number, color: string, ring = false) => {
    particles.push({
      targetX: centerX + (x - 24) * scale,
      targetY: centerY + (y - 24) * scale,
      sourceX: source.left + random() * source.width,
      sourceY: source.top + random() * source.height,
      scatterX: random() * width,
      scatterY: random() * height,
      size: ring ? .85 + random() * .65 : .95 + random() * .85,
      alpha: ring ? .62 + random() * .3 : .68 + random() * .3,
      color,
      delay: random() * .24,
      twist: (random() - .5) * 32,
      rotation: random() * Math.PI,
    });
  };

  for (const point of starPoints) {
    addParticle(point.x, point.y, random() < .22 ? '#d9956c' : '#f3eee6');
  }

  for (let index = 0; index < 72; index += 1) {
    const angle = (index / 72) * Math.PI * 2;
    const radius = 15.75 + (random() - .5) * .16;
    addParticle(24 + Math.cos(angle) * radius, 24 + Math.sin(angle) * radius, random() < .24 ? '#f3eee6' : '#c9825b', true);
  }

  return particles;
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function easeInOutCubic(value: number) {
  return value < .5 ? 4 * value ** 3 : 1 - (-2 * value + 2) ** 3 / 2;
}

function easeOutCubic(value: number) {
  return 1 - (1 - value) ** 3;
}

function LogoParticleField({ phase, reducedMotion, origin }: { phase: TransitionPhase; reducedMotion: boolean; origin: TransitionOrigin | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d', { alpha: true });
    if (!canvas || !context || reducedMotion) return;

    let width = 1;
    let height = 1;
    let pixelRatio = 1;
    let frame = 0;
    let phaseStartedAt = performance.now();
    let particles: Particle[] = [];

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
      canvas.width = Math.max(1, Math.round(width * pixelRatio));
      canvas.height = Math.max(1, Math.round(height * pixelRatio));
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      particles = createParticles(width, height, origin);
    };

    const draw = (now: number) => {
      context.clearRect(0, 0, width, height);
      const elapsed = now - phaseStartedAt;
      const releaseProgress = easeInOutCubic(clamp01(elapsed / RELEASE_MS));
      const centerX = width / 2;
      const centerY = height / 2;
      const formProgress = easeInOutCubic(clamp01(elapsed / ASSEMBLE_MS));
      const rotation = phase === 'assemble' ? (1 - formProgress) * .68 + Math.sin(formProgress * Math.PI) * .055 : 0;
      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);

      if (phase === 'hold' || phase === 'reveal') return;

      for (const particle of particles) {
        let x = particle.sourceX;
        let y = particle.sourceY;
        let alpha = particle.alpha;
        let size = particle.size;

        if (phase === 'scatter') {
          const progress = easeOutCubic(clamp01(elapsed / SCATTER_MS));
          x += (particle.scatterX - x) * progress;
          y += (particle.scatterY - y) * progress;
          x += Math.sin(progress * Math.PI + particle.rotation) * particle.twist * .18;
          y += Math.cos(progress * Math.PI + particle.rotation) * particle.twist * .18;
        } else if (phase === 'assemble') {
          const localProgress = clamp01((clamp01(elapsed / ASSEMBLE_MS) - particle.delay) / (1 - particle.delay));
          const eased = easeInOutCubic(localProgress);
          const offsetX = particle.targetX - centerX;
          const offsetY = particle.targetY - centerY;
          const targetX = centerX + offsetX * cos - offsetY * sin;
          const targetY = centerY + offsetX * sin + offsetY * cos;
          const arc = Math.sin(localProgress * Math.PI) * particle.twist * (1 - localProgress * .25);
          x = particle.scatterX + (targetX + arc - particle.scatterX) * eased;
          y = particle.scatterY + (targetY - arc * .55 - particle.scatterY) * eased;
          x += Math.sin(now * .0012 + particle.rotation) * (1 - eased) * 2.4;
          y += Math.cos(now * .0011 + particle.rotation * 1.3) * (1 - eased) * 2.4;
          alpha *= .22 + eased * .78;
        } else if (phase === 'release') {
          const localProgress = clamp01((releaseProgress - particle.delay * .32) / (1 - particle.delay * .32));
          const eased = easeInOutCubic(localProgress);
          const arc = Math.sin(localProgress * Math.PI) * particle.twist * 1.2;
          x = particle.targetX + (particle.scatterX - particle.targetX) * eased + arc;
          y = particle.targetY + (particle.scatterY - particle.targetY) * eased - arc * .6;
          alpha *= clamp01((1 - localProgress) / .3);
          size *= 1 + localProgress * .28;
        } else {
          continue;
        }

        context.globalAlpha = Math.max(0, alpha);
        context.fillStyle = particle.color;
        context.beginPath();
        context.ellipse(x, y, size, size * .72, particle.rotation, 0, Math.PI * 2);
        context.fill();
      }
      context.globalAlpha = 1;

      frame = window.requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener('resize', resize, { passive: true });
    frame = window.requestAnimationFrame(now => {
      phaseStartedAt = now;
      draw(now);
    });

    return () => {
      window.removeEventListener('resize', resize);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [phase, reducedMotion, origin]);

  return reducedMotion ? null : <canvas ref={canvasRef} className="cm-logo-transition__particles" aria-hidden="true"/>;
}

const TransitionContext = createContext<((to?: string, origin?: TransitionOrigin) => void) | null>(null);

function canonicalPath(pathname: string) {
  return pathname === '/overview' ? '/dashboard' : pathname;
}

export function LogoTransitionProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const { pathname, key } = useLocation();
  const reducedMotionPreference = useReducedMotion();
  const [active, setActive] = useState(false);
  const [phase, setPhase] = useState<TransitionPhase>('scatter');
  const [reducedMotion, setReducedMotion] = useState(true);
  const [origin, setOrigin] = useState<TransitionOrigin | null>(null);
  const started = useRef(false);
  const navigationStarted = useRef(false);
  const destination = useRef('/dashboard');
  const previousLocationKey = useRef(key);

  const startTransition = useCallback((to = '/dashboard', source?: TransitionOrigin) => {
    if (started.current) return;
    started.current = true;
    void transitionStylesReady.then(stylesReady => {
      if (!stylesReady) {
        started.current = false;
        navigate(to, { preventScrollReset: true });
        return;
      }
      navigationStarted.current = false;
      const useStaticMotion = reducedMotionPreference !== false;
      document.documentElement.dataset.cmRouteTransition = 'leaving';
      destination.current = to;
      setOrigin(source ?? null);
      setReducedMotion(useStaticMotion);
      setActive(true);
      if (useStaticMotion) {
        setPhase('hold');
      } else {
        setPhase('scatter');
      }
    });
  }, [navigate, reducedMotionPreference]);

  useLayoutEffect(() => {
    const root = document.documentElement;
    if (previousLocationKey.current !== key) {
      previousLocationKey.current = key;
      if (active) root.dataset.cmRouteTransition = 'arriving';
      else delete root.dataset.cmRouteTransition;
    }
    if (active) {
      root.dataset.cmTransitionPhase = reducedMotion ? 'reduced' : phase;
      return;
    }
    delete root.dataset.cmTransitionPhase;
    delete root.dataset.cmRouteTransition;
  }, [active, phase, reducedMotion, key]);

  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      const target = event.target instanceof Element ? event.target : null;
      const anchor = target?.closest<HTMLAnchorElement>('a[href]');
      if (!anchor || anchor.hasAttribute('download') || (anchor.target && anchor.target !== '_self')) return;
      if (anchor.closest('.cm-workspace-nav')) return;

      let url: URL;
      try { url = new URL(anchor.href, window.location.href); } catch { return; }
      if (url.origin !== window.location.origin) return;
      const current = new URL(window.location.href);
      if (url.pathname === current.pathname && url.search === current.search) {
        if (!url.hash || url.hash === current.hash) return;
        let targetId: string;
        try { targetId = decodeURIComponent(url.hash.slice(1)); } catch { return; }
        if (!document.getElementById(targetId)) return;
        event.preventDefault();
        navigate(`${url.pathname}${url.search}${url.hash}`, { preventScrollReset: true });
        return;
      }

      const rect = anchor.getBoundingClientRect();
      event.preventDefault();
      startTransition(`${url.pathname}${url.search}${url.hash}`, {
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
      });
    };
    document.addEventListener('click', onClick, true);
    return () => document.removeEventListener('click', onClick, true);
  }, [navigate, startTransition]);

  useEffect(() => {
    if (!active) return;
    const preventScroll = (event: Event) => {
      if (event.cancelable && (!(event instanceof WheelEvent) || !event.ctrlKey)) event.preventDefault();
    };
    const preventKeyboardScroll = (event: KeyboardEvent) => {
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) return;
      if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(event.key)) event.preventDefault();
    };
    window.addEventListener('wheel', preventScroll, { capture: true, passive: false });
    window.addEventListener('touchmove', preventScroll, { capture: true, passive: false });
    window.addEventListener('keydown', preventKeyboardScroll, true);
    return () => {
      window.removeEventListener('wheel', preventScroll, true);
      window.removeEventListener('touchmove', preventScroll, true);
      window.removeEventListener('keydown', preventKeyboardScroll, true);
    };
  }, [active]);

  useEffect(() => {
    if (!active || phase !== 'scatter') return;
    const timer = window.setTimeout(() => setPhase('assemble'), SCATTER_MS);
    return () => window.clearTimeout(timer);
  }, [active, phase]);

  useEffect(() => {
    if (!active || phase !== 'hold' || navigationStarted.current) return;
    navigationStarted.current = true;
    navigate(destination.current);
  }, [active, phase, navigate]);

  useEffect(() => {
    if (!active || phase !== 'assemble') return;
    const timer = window.setTimeout(() => setPhase('hold'), ASSEMBLE_MS);
    return () => window.clearTimeout(timer);
  }, [active, phase]);

  useEffect(() => {
    if (!active || phase !== 'hold') return;
    let completed = false;
    let finishTimer: number | undefined;
    const targetPath = canonicalPath(new URL(destination.current, window.location.origin).pathname);
    const finishWhenReady = () => {
      if (completed || canonicalPath(pathname) !== targetPath) return;
      if (targetPath === '/') {
        // The enhanced landing scene is code-split. Its small Hero fallback is
        // intentionally usable, but do not start the route reveal until the
        // actual cinematic scene (or the static story) has mounted.
        if (!document.querySelector('.cm-cinematic, .cm-research-intro')) return;
      } else if (!document.getElementById('main-content')) {
        return;
      }
      if (targetPath === '/dashboard') {
        const dashboard = document.querySelector<HTMLElement>('[data-cm-dashboard-ready]');
        if (dashboard?.dataset.cmDashboardReady !== 'true') return;
      }

      completed = true;
      observer.disconnect();
      finishTimer = window.setTimeout(
        () => setPhase(reducedMotion ? 'reveal' : 'release'),
        reducedMotion ? 120 : SETTLE_MS,
      );
    };
    const observer = new MutationObserver(finishWhenReady);
    observer.observe(document.getElementById('root') ?? document.body, {
      attributes: true,
      attributeFilter: ['data-cm-dashboard-ready'],
      childList: true,
      subtree: true,
    });
    finishWhenReady();
    const timeout = window.setTimeout(() => {
      if (completed) return;
      completed = true;
      observer.disconnect();
      setPhase(reducedMotion ? 'reveal' : 'release');
    }, DASHBOARD_READY_TIMEOUT_MS);
    return () => {
      observer.disconnect();
      window.clearTimeout(timeout);
      if (finishTimer !== undefined) window.clearTimeout(finishTimer);
    };
  }, [active, phase, pathname, reducedMotion]);

  useEffect(() => {
    if (!active || phase !== 'release') return;
    const timer = window.setTimeout(() => setPhase('reveal'), RELEASE_MS);
    return () => window.clearTimeout(timer);
  }, [active, phase]);

  useEffect(() => {
    if (!active || phase !== 'reveal') return;
    const targetPath = canonicalPath(new URL(destination.current, window.location.origin).pathname);
    const revealDuration = targetPath === '/' ? ABOUT_REVEAL_MS : REVEAL_MS;
    const timer = window.setTimeout(() => {
      setActive(false);
      setPhase('scatter');
      setOrigin(null);
      destination.current = '/dashboard';
      started.current = false;
      navigationStarted.current = false;
    }, reducedMotion ? 160 : revealDuration);
    return () => window.clearTimeout(timer);
  }, [active, phase, reducedMotion]);

  return <TransitionContext.Provider value={startTransition}>
    {children}
    {active && <div className={`cm-logo-transition cm-logo-transition--${phase}${reducedMotion ? ' cm-logo-transition--reduced' : ''}`} role="status" aria-live="polite" aria-label="Opening CopperMind workspace">
      <Suspense fallback={null}><TransitionAtmosphere className="cm-terra-atmosphere--transition"/></Suspense>
      <div className="cm-logo-transition__backdrop"/>
      <LogoParticleField phase={phase} reducedMotion={reducedMotion} origin={origin}/>
      <div className="cm-logo-transition__glow" aria-hidden="true"/>
      <div className="cm-logo-transition__mark" aria-hidden="true"><BrandMark size={LOGO_SIZE} variant="primary"/></div>
      <span className="sr-only">Opening CopperMind workspace</span>
    </div>}
  </TransitionContext.Provider>;
}

type EnterWorkspaceLinkProps = Omit<LinkProps, 'to'>;

export function EnterWorkspaceLink({ onClick, ...props }: EnterWorkspaceLinkProps) {
  const startTransition = useContext(TransitionContext);
  const handleClick = (event: ReactMouseEvent<HTMLAnchorElement>) => {
    onClick?.(event);
    const { currentTarget } = event;
    const modified = event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey;
    if (event.defaultPrevented || !startTransition || modified || (currentTarget.target && currentTarget.target !== '_self')) return;
    event.preventDefault();
    const rect = currentTarget.getBoundingClientRect();
    startTransition('/dashboard', { left: rect.left, top: rect.top, width: rect.width, height: rect.height });
  };

  return <Link {...props} to="/dashboard" onClick={handleClick}/>;
}
