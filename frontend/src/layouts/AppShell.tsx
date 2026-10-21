import { useState, useEffect, useRef, Suspense } from 'react';
import { Outlet, NavLink, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Brain, CheckCircle, Server } from 'lucide-react';
import { clsx } from 'clsx';
import { Brand } from '../components/ui/Brand';
import { TerraAtmosphere } from '../components/ui/TerraAtmosphere';
import '../design/workspace.css';

export const SCROLL_THRESHOLD = 70;

const navigation = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Overview' },
  { to: '/models', icon: Brain, label: 'Models' },
  { to: '/validation', icon: CheckCircle, label: 'Validation' },
  { to: '/system', icon: Server, label: 'System' },
];

// Keep the route chunks warm while the current page is idle. Without this the
// first switch to each workspace page replaces the shell with the top-level
// route loader for a frame — the dark "Opening your workspace" flash.
const routePrefetch: Record<string, () => Promise<unknown>> = {
  '/dashboard': () => import('../pages/OverviewPage'),
  '/models': () => import('../pages/ModelsPage'),
  '/validation': () => import('../pages/ValidationPage'),
  '/system': () => import('../pages/SystemPage'),
};

function prefetchRoute(path: string) {
  const load = routePrefetch[path];
  if (load) void load().catch(() => undefined);
}

function prefetchWorkspaceRoutes() {
  for (const path of Object.keys(routePrefetch)) prefetchRoute(path);
}

function WorkspacePageSkeleton() {
  return (
    <div className="cm-workspace-page-skeleton" role="status" aria-busy="true" aria-label="Loading page">
      <span className="sr-only">Loading page.</span>
      <div className="cm-state-skeleton" aria-hidden="true"><span /><span /><span /></div>
    </div>
  );
}

function getScrollY(): number {
  if (typeof window === 'undefined') return 0;
  if (typeof window.scrollY === 'number') return window.scrollY;
  return (
    document.scrollingElement?.scrollTop ??
    document.documentElement?.scrollTop ??
    document.body?.scrollTop ??
    0
  );
}

function getMaxScrollY(): number {
  if (typeof window === 'undefined' || typeof document === 'undefined') return Infinity;
  const scrollHeight = Math.max(
    document.scrollingElement?.scrollHeight ?? 0,
    document.documentElement?.scrollHeight ?? 0,
    document.body?.scrollHeight ?? 0
  );
  const innerHeight = window.innerHeight || document.documentElement?.clientHeight || 0;
  return innerHeight > 0 && scrollHeight > innerHeight ? scrollHeight - innerHeight : Infinity;
}

export function getBoundedScrollY(): number {
  const rawScroll = Math.max(0, getScrollY());
  if (rawScroll <= SCROLL_THRESHOLD) return rawScroll;
  const maxScrollY = getMaxScrollY();
  return maxScrollY !== Infinity ? Math.min(rawScroll, maxScrollY) : rawScroll;
}

export function AppShell() {
  const [isVisible, setIsVisible] = useState(() => {
    return getBoundedScrollY() <= SCROLL_THRESHOLD;
  });
  const [isFocused, setIsFocused] = useState(false);
  const location = useLocation();
  const lastScrollYRef = useRef(getBoundedScrollY());
  const previousPathnameRef = useRef(location.pathname);

  // Synchronize scroll baseline and restore visibility upon route change
  useEffect(() => {
    if (previousPathnameRef.current !== location.pathname) {
      previousPathnameRef.current = location.pathname;
      setIsVisible(true);
      lastScrollYRef.current = 0;
    }
  }, [location.pathname]);

  useEffect(() => {
    let ticking = false;
    let frameId: number | null = null;
    lastScrollYRef.current = getBoundedScrollY();

    const updateHeader = () => {
      const currentScrollY = getBoundedScrollY();
      const lastScrollY = lastScrollYRef.current;

      if (currentScrollY <= SCROLL_THRESHOLD) {
        setIsVisible(true);
      } else if (currentScrollY > lastScrollY) {
        setIsVisible(false);
      } else if (currentScrollY < lastScrollY) {
        setIsVisible(true);
      }

      lastScrollYRef.current = currentScrollY;
      ticking = false;
      frameId = null;
    };

    const scheduleUpdate = () => {
      if (!ticking) {
        ticking = true;
        frameId = window.requestAnimationFrame(updateHeader);
      }
    };

    window.addEventListener('scroll', scheduleUpdate, { passive: true });
    window.addEventListener('resize', scheduleUpdate, { passive: true });

    return () => {
      window.removeEventListener('scroll', scheduleUpdate);
      window.removeEventListener('resize', scheduleUpdate);
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId);
      }
    };
  }, []);

  useEffect(() => {
    const idle = 'requestIdleCallback' in window
      ? window.requestIdleCallback(prefetchWorkspaceRoutes, { timeout: 2500 })
      : undefined;
    const timer = idle === undefined ? window.setTimeout(prefetchWorkspaceRoutes, 1600) : undefined;
    return () => {
      if (idle !== undefined) window.cancelIdleCallback(idle);
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, []);

  const isHidden = !isVisible && !isFocused;

  return (
    <div className="cm-workspace">
      <TerraAtmosphere className="cm-terra-atmosphere--workspace" interactive />
      <a className="cm-skip" href="#main-content">Skip to workspace</a>
      <header
        className={clsx('cm-workspace-header', isHidden && 'cm-workspace-header--hidden')}
        data-state={isHidden ? 'hidden' : 'visible'}
        data-hidden={isHidden ? 'true' : 'false'}
        onFocusCapture={() => setIsFocused(true)}
        onBlurCapture={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
            setIsFocused(false);
          }
        }}
      >
        <div className="cm-workspace-bar">
          <Brand />
          <nav className="cm-workspace-nav" aria-label="Workspace">
            {navigation.map(({ to, icon: Icon, label }) => (
              <NavLink
                to={to}
                key={to}
                end
                className="cm-nav-link"
                onMouseEnter={() => { prefetchRoute(to); }}
                onFocus={() => { prefetchRoute(to); }}
              >
                <Icon size={17} aria-hidden="true" />
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main id="main-content" tabIndex={-1} className="cm-workspace-main">
        <Suspense fallback={<WorkspacePageSkeleton />}>
          <Outlet />
        </Suspense>
      </main>
      <footer className="cm-workspace-footer">
        <span>CopperMind / Terra Rara · Research workspace</span>
        <Link to="/">About the platform</Link>
      </footer>
    </div>
  );
}
