import { createContext, lazy, Suspense, useCallback, useContext, useEffect, useLayoutEffect, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode } from 'react';
import { Link, useLocation, useNavigate, type LinkProps } from 'react-router-dom';
import { useReducedMotion } from 'framer-motion';
import { BrandMark } from './BrandMark';
import { playObservatoryDisperse, playObservatoryTransition } from '../../utils/cosmicVision';

const preloadAtmosphere = () => import('./TerraAtmosphere');
const TransitionAtmosphere = lazy(() => preloadAtmosphere().then(module => ({ default: module.TerraAtmosphere })));
const preloadGalaxyField = () => import('./GalaxyTransitionField');
const GalaxyTransitionField = lazy(preloadGalaxyField);
const transitionStylesReady = import('./LogoTransition.css').then(() => true, () => false);

type TransitionPhase = 'scatter' | 'assemble' | 'hold' | 'release' | 'reveal';
type TransitionOrigin = { left: number; top: number; width: number; height: number };

const SCATTER_MS = 420;
const ASSEMBLE_MS = 1_520;
const SETTLE_MS = 520;
const RELEASE_MS = 1_520;
// Keep the transparent final frame in place while destination content enters.
const REVEAL_MS = 2_000;
const TAB_REVEAL_MS = 160;
const ABOUT_REVEAL_MS = 1_200;
const DASHBOARD_READY_TIMEOUT_MS = 15_000;
const LOGO_SIZE = 224;
const ROUTE_TRANSITION_MIN_HEIGHT = '--cm-route-transition-min-height';

const TransitionContext = createContext<((to?: string, origin?: TransitionOrigin) => void) | null>(null);

function canonicalPath(pathname: string) {
  return pathname === '/overview' ? '/dashboard' : pathname;
}

function isAboutDashboardPair(from: string, to: string) {
  const source = canonicalPath(from);
  const destination = canonicalPath(to);
  return (source === '/' && destination === '/dashboard') || (source === '/dashboard' && destination === '/');
}

export interface LogoTransitionState {
  active: boolean;
  phase: TransitionPhase;
  reducedMotion: boolean;
  origin: TransitionOrigin | null;
}

export type LogoTransitionEvent =
  | { type: 'start'; reducedMotion: boolean; origin: TransitionOrigin | null }
  | { type: 'advance'; phase: TransitionPhase }
  | { type: 'reset' };

export const INITIAL_LOGO_TRANSITION_STATE: LogoTransitionState = {
  active: false,
  phase: 'scatter',
  reducedMotion: true,
  origin: null,
};

export function advanceLogoTransition(
  state: LogoTransitionState,
  event: LogoTransitionEvent,
): LogoTransitionState {
  switch (event.type) {
    case 'start':
      return {
        active: true,
        phase: event.reducedMotion ? 'hold' : 'scatter',
        reducedMotion: event.reducedMotion,
        origin: event.origin,
      };
    case 'advance':
      if (!state.active) return state;
      return { ...state, phase: event.phase };
    case 'reset':
      return INITIAL_LOGO_TRANSITION_STATE;
  }
}

const SNAPSHOT_ATMO_VARIANTS = [
  'cm-terra-atmosphere--landing-cinematic',
  'cm-terra-atmosphere--landing-static',
  'cm-terra-atmosphere--workspace',
] as const;

export function captureOutgoingBackdrop(host: HTMLDivElement): void {
  const sourceAtmo = document.querySelector<HTMLElement>(
    '.cm-landing .cm-global-atmosphere, .cm-workspace .cm-global-atmosphere',
  );
  const snapshotAtmo = host.querySelector<HTMLElement>('.cm-logo-transition__snapshot-atmo');
  const snapshotParticles = host.querySelector<HTMLCanvasElement>('.cm-logo-transition__snapshot-particles');

  const bodyBg = getComputedStyle(document.body).backgroundColor;
  let fallbackBg = bodyBg && bodyBg !== 'rgba(0, 0, 0, 0)' && bodyBg !== 'transparent' ? bodyBg : '#060a11';

  if (snapshotAtmo) {
    for (const variant of SNAPSHOT_ATMO_VARIANTS) {
      snapshotAtmo.classList.toggle(variant, Boolean(sourceAtmo?.classList.contains(variant)));
    }

    if (sourceAtmo) {
      const copyLayer = (
        selector: string,
        props: ReadonlyArray<'backgroundColor' | 'backgroundImage' | 'opacity' | 'transform'>,
      ): CSSStyleDeclaration | null => {
        const src = sourceAtmo.querySelector<HTMLElement>(selector);
        const dst = snapshotAtmo.querySelector<HTMLElement>(selector);
        if (!dst) return null;
        if (!src) {
          dst.style.opacity = '0';
          return null;
        }
        const computed = getComputedStyle(src);
        for (const prop of props) {
          const val = computed[prop];
          if (val) dst.style[prop] = val;
        }
        return computed;
      };

      const baseComputed = copyLayer('.cm-atmosphere-base', ['backgroundColor', 'backgroundImage']);
      copyLayer('.cm-atmosphere-glow--copper', ['opacity', 'transform']);
      copyLayer('.cm-atmosphere-glow--blue', ['opacity', 'transform']);
      copyLayer('.cm-atmosphere-band', ['opacity', 'transform']);
      const voidComputed = copyLayer('.cm-atmosphere-void', ['backgroundColor', 'opacity']);
      copyLayer('.cm-atmosphere-stars', ['opacity', 'transform']);

      if (voidComputed && Number.parseFloat(voidComputed.opacity) >= 0.98) {
        fallbackBg = voidComputed.backgroundColor || '#000000';
      } else if (baseComputed?.backgroundColor && baseComputed.backgroundColor !== 'rgba(0, 0, 0, 0)') {
        fallbackBg = baseComputed.backgroundColor;
      }

      const sourceStars = sourceAtmo.querySelector<HTMLCanvasElement>('.cm-atmosphere-starfield');
      const targetStars = snapshotAtmo.querySelector<HTMLCanvasElement>('.cm-atmosphere-starfield');
      if (sourceStars && targetStars && sourceStars.width > 0 && sourceStars.height > 0) {
        targetStars.width = sourceStars.width;
        targetStars.height = sourceStars.height;
        const starCtx = get2dContext(targetStars);
        if (starCtx) {
          starCtx.clearRect(0, 0, targetStars.width, targetStars.height);
          starCtx.drawImage(sourceStars, 0, 0);
        }
      }
    }
  }

  host.style.backgroundColor = fallbackBg;

  if (snapshotParticles) {
    snapshotParticles.style.opacity = '0';
    if (snapshotParticles.width > 0 && snapshotParticles.height > 0) {
      get2dContext(snapshotParticles)?.clearRect(0, 0, snapshotParticles.width, snapshotParticles.height);
    }
    if (!document.documentElement.dataset.cmGalaxyActive && document.querySelector('.cm-landing .cm-particle-world[data-renderer]')) {
      window.dispatchEvent(new CustomEvent('cm:particle-snapshot', {
        detail: {
          drawTo: (sourceCanvas: HTMLCanvasElement) => {
            if (sourceCanvas.width <= 0 || sourceCanvas.height <= 0) return;
            snapshotParticles.width = sourceCanvas.width;
            snapshotParticles.height = sourceCanvas.height;
            const ctx = get2dContext(snapshotParticles);
            if (ctx) {
              ctx.clearRect(0, 0, snapshotParticles.width, snapshotParticles.height);
              ctx.drawImage(sourceCanvas, 0, 0);
              snapshotParticles.style.opacity = '1';
            }
          },
        },
      }));
    }
  }

  host.dataset.captured = 'true';
}

function get2dContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  if (typeof navigator !== 'undefined' && /jsdom/i.test(navigator.userAgent)) return null;
  try {
    return canvas.getContext('2d');
  } catch {
    return null;
  }
}

export function LogoTransitionProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const { pathname, key } = useLocation();
  const reducedMotionPreference = useReducedMotion();
  const [machine, setMachine] = useState<LogoTransitionState>(INITIAL_LOGO_TRANSITION_STATE);
  const [targetKind, setTargetKind] = useState<'dashboard' | 'landing'>('dashboard');
  const { active, phase, reducedMotion } = machine;
  const started = useRef(false);
  const navigationStarted = useRef(false);
  const snapshotCaptured = useRef(false);
  const snapshotHostRef = useRef<HTMLDivElement>(null);
  const destination = useRef('/dashboard');
  const previousPathnameRef = useRef(pathname);
  const previousLocationKey = useRef(key);
  const genericRevealTimer = useRef<number | undefined>(undefined);

  const startTransition = useCallback((to = '/dashboard', source?: TransitionOrigin) => {
    if (started.current) return;
    started.current = true;
    snapshotCaptured.current = false;
    const nextTargetKind = canonicalPath(new URL(to, window.location.origin).pathname) === '/dashboard'
      ? 'dashboard'
      : 'landing';
    setTargetKind(nextTargetKind);
    if (nextTargetKind === 'dashboard') {
      void preloadAtmosphere();
    }
    void playObservatoryTransition({ volume: 0.44 });
    if (genericRevealTimer.current !== undefined) {
      window.clearTimeout(genericRevealTimer.current);
      genericRevealTimer.current = undefined;
    }
    delete document.documentElement.dataset.cmTabTransition;
    void Promise.all([
      transitionStylesReady,
      reducedMotionPreference ? Promise.resolve(true) : preloadGalaxyField().then(() => true, () => false),
    ]).then(([stylesReady]) => {
      if (!stylesReady) {
        started.current = false;
        navigate(to, { preventScrollReset: true });
        return;
      }
      navigationStarted.current = false;
      const useStaticMotion = reducedMotionPreference === true;
      const root = document.documentElement;
      const transitionHeight = Math.max(root.scrollHeight, document.body.scrollHeight, window.innerHeight);
      root.style.setProperty(ROUTE_TRANSITION_MIN_HEIGHT, `${transitionHeight}px`);
      root.dataset.cmLogoTransition = 'true';
      root.dataset.cmRouteTransition = 'leaving';
      destination.current = to;
      setMachine(prev => advanceLogoTransition(prev, {
        type: 'start',
        reducedMotion: useStaticMotion,
        origin: source ?? null,
      }));
    });
  }, [navigate, reducedMotionPreference]);

  useEffect(() => () => {
    if (genericRevealTimer.current !== undefined) {
      window.clearTimeout(genericRevealTimer.current);
      genericRevealTimer.current = undefined;
    }
    const root = document.documentElement;
    delete root.dataset.cmTabTransition;
    delete root.dataset.cmTransitionPhase;
    delete root.dataset.cmRouteTransition;
    delete root.dataset.cmLogoTransition;
    delete root.dataset.cmGalaxyActive;
  }, []);

  useLayoutEffect(() => {
    const root = document.documentElement;
    const isPathnameChange = previousPathnameRef.current !== pathname;
    previousPathnameRef.current = pathname;
    previousLocationKey.current = key;

    if (isPathnameChange) {
      if (active) {
        if (genericRevealTimer.current !== undefined) {
          window.clearTimeout(genericRevealTimer.current);
          genericRevealTimer.current = undefined;
        }
        delete root.dataset.cmTabTransition;
        root.dataset.cmRouteTransition = 'arriving';
      } else {
        delete root.dataset.cmTransitionPhase;
        delete root.dataset.cmRouteTransition;
        if (genericRevealTimer.current !== undefined) window.clearTimeout(genericRevealTimer.current);
        root.dataset.cmTabTransition = 'arriving';
        genericRevealTimer.current = window.setTimeout(() => {
          delete root.dataset.cmTabTransition;
          genericRevealTimer.current = undefined;
        }, TAB_REVEAL_MS);
      }
    }
    if (active) {
      if (phase === 'hold' && !snapshotCaptured.current) {
        snapshotCaptured.current = true;
        if (snapshotHostRef.current) {
          captureOutgoingBackdrop(snapshotHostRef.current);
        }
      }
      root.dataset.cmTransitionPhase = reducedMotion ? 'reduced' : phase;
      return;
    }
    snapshotCaptured.current = false;
    delete root.dataset.cmTransitionPhase;
    delete root.dataset.cmRouteTransition;
    delete root.dataset.cmLogoTransition;
    delete root.dataset.cmGalaxyActive;
    root.style.removeProperty(ROUTE_TRANSITION_MIN_HEIGHT);
  }, [active, phase, reducedMotion, key, pathname]);

  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      const target = event.target instanceof Element ? event.target : null;
      const anchor = target?.closest<HTMLAnchorElement>('a[href]');
      if (!anchor || anchor.hasAttribute('download') || (anchor.target && anchor.target !== '_self')) return;

      let url: URL;
      try { url = new URL(anchor.href, window.location.href); } catch { return; }
      if (url.origin !== window.location.origin) return;
      const current = new URL(window.location.href);
      if (url.pathname === current.pathname && url.search === current.search) {
        if (!url.hash || url.hash === current.hash) return;
        let targetId: string;
        try { targetId = decodeURIComponent(url.hash.slice(1)); } catch { return; }
        const targetElement = document.getElementById(targetId);
        if (!targetElement) return;
        event.preventDefault();
        window.history.pushState(null, '', url.hash);
        targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
      }

      if (!isAboutDashboardPair(current.pathname, url.pathname)) return;

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
    const timer = window.setTimeout(() => setMachine(prev => advanceLogoTransition(prev, { type: 'advance', phase: 'assemble' })), SCATTER_MS);
    return () => window.clearTimeout(timer);
  }, [active, phase]);

  useEffect(() => {
    if (!active || phase !== 'hold' || navigationStarted.current) return;
    navigationStarted.current = true;
    navigate(destination.current);
  }, [active, phase, navigate]);

  useEffect(() => {
    if (!active || phase !== 'assemble') return;
    const timer = window.setTimeout(() => setMachine(prev => advanceLogoTransition(prev, { type: 'advance', phase: 'hold' })), ASSEMBLE_MS);
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
        () => setMachine(prev => advanceLogoTransition(prev, { type: 'advance', phase: reducedMotion ? 'reveal' : 'release' })),
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
      setMachine(prev => advanceLogoTransition(prev, { type: 'advance', phase: reducedMotion ? 'reveal' : 'release' }));
    }, DASHBOARD_READY_TIMEOUT_MS);
    return () => {
      observer.disconnect();
      window.clearTimeout(timeout);
      if (finishTimer !== undefined) window.clearTimeout(finishTimer);
    };
  }, [active, phase, pathname, reducedMotion]);

  useEffect(() => {
    if (!active || phase !== 'release') return;
    void playObservatoryDisperse({ volume: 0.36 });
    const timer = window.setTimeout(() => setMachine(prev => advanceLogoTransition(prev, { type: 'advance', phase: 'reveal' })), RELEASE_MS);
    return () => window.clearTimeout(timer);
  }, [active, phase]);

  useEffect(() => {
    if (!active || phase !== 'reveal') return;
    const targetPath = canonicalPath(new URL(destination.current, window.location.origin).pathname);
    const revealDuration = targetPath === '/' ? ABOUT_REVEAL_MS : REVEAL_MS;
    const timer = window.setTimeout(() => {
      setMachine(prev => advanceLogoTransition(prev, { type: 'reset' }));
      destination.current = '/dashboard';
      started.current = false;
      navigationStarted.current = false;
      snapshotCaptured.current = false;
    }, reducedMotion ? 160 : revealDuration);
    return () => window.clearTimeout(timer);
  }, [active, phase, reducedMotion]);

  return <TransitionContext.Provider value={startTransition}>
    {children}
    {active && <div className={`cm-logo-transition cm-logo-transition--${phase}${reducedMotion ? ' cm-logo-transition--reduced' : ''}`} role="status" aria-live="polite" aria-label="Opening CopperMind workspace">
      {targetKind === 'dashboard' && phase !== 'scatter' && phase !== 'assemble'
        ? <Suspense fallback={null}><TransitionAtmosphere className="cm-terra-atmosphere--transition"/></Suspense>
        : null}
      <div ref={snapshotHostRef} className="cm-logo-transition__snapshot" aria-hidden="true">
        <div className="cm-global-atmosphere cm-logo-transition__snapshot-atmo">
          <div className="cm-atmosphere-base"/>
          <div className="cm-atmosphere-glow cm-atmosphere-glow--copper"/>
          <div className="cm-atmosphere-glow cm-atmosphere-glow--blue"/>
          <div className="cm-atmosphere-band"/>
          <div className="cm-atmosphere-void"/>
          <div className="cm-atmosphere-stars">
            <canvas className="cm-atmosphere-starfield" aria-hidden="true"/>
          </div>
          <div className="cm-atmosphere-grain"/>
        </div>
        <canvas className="cm-logo-transition__snapshot-particles" aria-hidden="true"/>
      </div>
      <div className="cm-logo-transition__backdrop"/>
      <Suspense fallback={null}>
        <GalaxyTransitionField
          phase={phase}
          reducedMotion={reducedMotion}
          scatterMs={SCATTER_MS}
          assembleMs={ASSEMBLE_MS}
          releaseMs={RELEASE_MS}
        />
      </Suspense>
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

  return <Link data-cm-click-sound="off" {...props} to="/dashboard" onClick={handleClick}/>;
}
