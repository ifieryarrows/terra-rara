import { useState, useEffect, useRef, useContext, lazy, Suspense } from 'react';
import { Outlet, NavLink, Link, useLocation } from 'react-router-dom';
import { QueryClientContext, useIsFetching, useQuery, type QueryClient } from '@tanstack/react-query';
import {
  LayoutDashboard,
  Brain,
  CheckCircle,
  Server,
  RefreshCw,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { clsx } from 'clsx';
import { fetchLivePrice } from '../api';
import { Brand } from '../components/ui/Brand';
import { TerraAtmosphere } from '../components/ui/TerraAtmosphere';
import { CuprumProvider, useCuprumFeed } from '../contexts/CuprumContext';
import { useSystemStatus } from '../hooks/useQueries';
import { playObservatoryChime, useObservatorySound } from '../utils/cosmicVision';
import '../design/workspace.css';

const LazyCuprumCompanionPanel = lazy(() =>
  import('../components/CuprumCompanionPanel').then((m) => ({ default: m.CuprumCompanionPanel })),
);

export const SCROLL_THRESHOLD = 70;

const navigation = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Desk' },
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

function HeaderQuoteStatus({
  price,
  changePct,
  healthStatus,
  isLive,
}: {
  price: number | null | undefined;
  changePct: number | null | undefined;
  healthStatus: 'healthy' | 'degraded' | 'unhealthy' | 'ready';
  isLive: boolean;
}) {
  const tone =
    changePct == null || changePct === 0 ? 'neutral' : changePct > 0 ? 'good' : 'bad';
  const healthLabel =
    healthStatus === 'healthy'
      ? 'Models healthy'
      : healthStatus === 'degraded'
        ? 'Models degraded'
        : healthStatus === 'unhealthy'
          ? 'Attention needed'
          : 'Desk ready';

  return (
    <div className="cm-header-quote" aria-label="COMEX Copper live telemetry">
      <span className="cm-header-symbol">HG=F</span>
      {typeof price === 'number' && Number.isFinite(price) ? (
        <strong className="cm-header-price cm-tabular">${price.toFixed(4)}</strong>
      ) : (
        <span className="cm-header-price cm-header-price--muted">COMEX</span>
      )}
      {typeof changePct === 'number' && Number.isFinite(changePct) ? (
        <span className={clsx('cm-header-delta cm-tabular', `cm-tone-${tone}`)}>
          {changePct > 0 ? '+' : ''}
          {changePct.toFixed(2)}%
        </span>
      ) : null}
      <span
        className={clsx(
          'cm-header-health',
          `cm-header-health--${healthStatus}`,
          isLive && 'cm-header-health--live',
        )}
        title={healthLabel}
      >
        <span className="cm-header-health-dot" aria-hidden="true" />
        <span className="cm-header-health-label">{isLive ? 'LIVE' : healthLabel}</span>
      </span>
    </div>
  );
}

function HeaderActions({
  refreshing,
  onRefresh,
  soundOn,
  onToggleSound,
}: {
  refreshing: boolean;
  onRefresh: () => void;
  soundOn: boolean;
  onToggleSound: () => void;
}) {
  return (
    <div className="cm-header-actions">
      <button
        type="button"
        className={clsx('cm-header-action', soundOn && 'cm-header-action--active')}
        data-cm-click-sound="off"
        onClick={onToggleSound}
        aria-pressed={soundOn}
        aria-label={soundOn ? 'Mute observatory ambient sound' : 'Enable observatory ambient sound'}
        title={soundOn ? 'Sound on' : 'Sound muted'}
      >
        {soundOn ? <Volume2 size={15} aria-hidden="true" /> : <VolumeX size={15} aria-hidden="true" />}
        <span className="cm-header-action-label">{soundOn ? 'Audio' : 'Muted'}</span>
      </button>

      <button
        type="button"
        className={clsx('cm-header-action cm-header-action--refresh', refreshing && 'cm-header-action--busy')}
        onClick={onRefresh}
        disabled={refreshing}
        aria-label="Refresh workspace telemetry"
        title="Refresh workspace telemetry"
      >
        <RefreshCw
          size={15}
          className={refreshing ? 'cm-spin' : undefined}
          aria-hidden="true"
        />
        <span className="cm-header-action-label">{refreshing ? 'Syncing' : 'Refresh'}</span>
      </button>
    </div>
  );
}

function HeaderBarWithQuery({
  queryClient,
  soundOn,
  onToggleSound,
  navNode,
}: {
  queryClient: QueryClient;
  soundOn: boolean;
  onToggleSound: () => void;
  navNode: React.ReactNode;
}) {
  const { feed } = useCuprumFeed();
  const liveQuery = useQuery({
    queryKey: ['live-price'],
    queryFn: fetchLivePrice,
    staleTime: 60_000,
    refetchInterval: 60_000,
  });
  const statusQuery = useSystemStatus();
  const isFetching = useIsFetching();

  const price =
    liveQuery.data?.price ?? (typeof feed.quote?.price === 'number' ? feed.quote.price : null);
  const changePct =
    typeof feed.quote?.change_pct === 'number'
      ? feed.quote.change_pct
      : typeof feed.trendPct === 'number'
        ? feed.trendPct
        : null;
  const healthStatus = statusQuery.data?.status ?? 'ready';
  const isLive = Boolean(liveQuery.data?.price && !liveQuery.data?.error);
  const refreshing = isFetching > 0 || liveQuery.isFetching;

  const handleRefresh = () => {
    void playObservatoryChime();
    void queryClient.invalidateQueries();
  };

  return (
    <>
      <div className="cm-header-lead">
        <Brand />
        <HeaderQuoteStatus
          price={price}
          changePct={changePct}
          healthStatus={healthStatus}
          isLive={isLive}
        />
      </div>
      {navNode}
      <HeaderActions
        refreshing={refreshing}
        onRefresh={handleRefresh}
        soundOn={soundOn}
        onToggleSound={onToggleSound}
      />
    </>
  );
}

function HeaderBarFallback({
  soundOn,
  onToggleSound,
  navNode,
}: {
  soundOn: boolean;
  onToggleSound: () => void;
  navNode: React.ReactNode;
}) {
  const { feed } = useCuprumFeed();
  const price = typeof feed.quote?.price === 'number' ? feed.quote.price : null;
  const changePct = typeof feed.quote?.change_pct === 'number' ? feed.quote.change_pct : null;

  return (
    <>
      <div className="cm-header-lead">
        <Brand />
        <HeaderQuoteStatus
          price={price}
          changePct={changePct}
          healthStatus="ready"
          isLive={false}
        />
      </div>
      {navNode}
      <HeaderActions
        refreshing={false}
        onRefresh={() => {
          void playObservatoryChime();
        }}
        soundOn={soundOn}
        onToggleSound={onToggleSound}
      />
    </>
  );
}

function AppShellContent() {
  const [isVisible, setIsVisible] = useState(() => {
    return getBoundedScrollY() <= SCROLL_THRESHOLD;
  });
  const [isFocused, setIsFocused] = useState(false);
  const location = useLocation();
  const lastScrollYRef = useRef(getBoundedScrollY());
  const previousPathnameRef = useRef(location.pathname);
  const queryClient = useContext(QueryClientContext);
  const [soundOn, toggleSound] = useObservatorySound();

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

  const handleToggleSound = () => {
    const next = !soundOn;
    toggleSound(next);
    if (next) void playObservatoryChime({ force: true, volume: 0.4 });
  };

  const navNode = (
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
          <Icon size={16} aria-hidden="true" />
          <span>{label}</span>
        </NavLink>
      ))}
    </nav>
  );

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
          {queryClient ? (
            <HeaderBarWithQuery
              queryClient={queryClient}
              soundOn={soundOn}
              onToggleSound={handleToggleSound}
              navNode={navNode}
            />
          ) : (
            <HeaderBarFallback
              soundOn={soundOn}
              onToggleSound={handleToggleSound}
              navNode={navNode}
            />
          )}
        </div>
      </header>
      <main id="main-content" tabIndex={-1} className="cm-workspace-main">
        <Suspense fallback={<WorkspacePageSkeleton />}>
          <Outlet />
        </Suspense>
      </main>
      <Suspense fallback={null}>
        <LazyCuprumCompanionPanel />
      </Suspense>
      <footer className="cm-workspace-footer">
        <span>CopperMind / Terra Rara · COMEX HG=F Research Desk</span>
        <Link to="/">About the platform</Link>
      </footer>
    </div>
  );
}

export function AppShell() {
  return (
    <CuprumProvider>
      <AppShellContent />
    </CuprumProvider>
  );
}
