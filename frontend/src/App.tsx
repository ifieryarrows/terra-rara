import { lazy, Suspense, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, Link, useLocation, useNavigationType } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MotionConfig } from 'framer-motion';
import { LandingPage } from './features/landing/LandingPage';
import { RouteBoundary } from './components/ui/RouteBoundary';
import { motionTokens } from './design/motion';
import { SpeedInsights } from '@vercel/speed-insights/react';
import './App.css';

const AppShell = lazy(() => import('./layouts/AppShell').then(m => ({ default: m.AppShell })));
const OverviewPage = lazy(() => import('./pages/OverviewPage').then(m => ({ default: m.OverviewPage })));
const ModelsPage = lazy(() => import('./pages/ModelsPage').then(m => ({ default: m.ModelsPage })));
const ValidationPage = lazy(() => import('./pages/ValidationPage').then(m => ({ default: m.ValidationPage })));
const SystemPage = lazy(() => import('./pages/SystemPage').then(m => ({ default: m.SystemPage })));

const queryClient = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } } });
const titles: Record<string, string> = { '/': 'CopperMind Terra Rara | Copper Market Intelligence', '/dashboard': 'Market Overview | CopperMind', '/models': 'Model Intelligence | CopperMind', '/validation': 'Walk-Forward Validation | CopperMind', '/system': 'System Status | CopperMind' };

const scrollPositions = new Map<string, number>();

function RouteLifecycle() {
  const { pathname, key, hash } = useLocation();
  const navigationType = useNavigationType();
  useEffect(() => {
    document.title = titles[pathname] || 'Page not found | CopperMind';
    const focusMain = () => {
      const main = document.getElementById('main-content');
      if (!main) return false;
      main.focus({ preventScroll: true });
      const anchor = hash ? document.getElementById(hash.slice(1)) : null;
      if (anchor) anchor.scrollIntoView({ behavior: 'instant' });
      else window.scrollTo({ top: navigationType === 'POP' ? (scrollPositions.get(key) ?? 0) : 0, behavior: 'instant' });
      return true;
    };
    let observer: MutationObserver | undefined;
    if (!focusMain()) {
      observer = new MutationObserver(() => { if (focusMain()) observer?.disconnect(); });
      observer.observe(document.getElementById('root') ?? document.body, { childList: true, subtree: true });
    }
    return () => {
      scrollPositions.set(key, window.scrollY);
      if (scrollPositions.size > 50) scrollPositions.delete(scrollPositions.keys().next().value!);
      observer?.disconnect();
    };
  }, [pathname, key, hash, navigationType]);
  return null;
}

function LandingEntry() {
  const { search, hash } = useLocation();
  if (new URLSearchParams(search).has('symbol')) return <Navigate replace to={'/dashboard' + search + hash}/>;
  return <LandingPage/>;
}
function RouteLoading() { return <div className="cm-route-loading" role="status">Opening your workspace…</div>; }

export function AppRoutes() {
  const { pathname } = useLocation();
  return <MotionConfig reducedMotion="user" transition={{ duration: motionTokens.ui, ease: motionTokens.ease }}><RouteLifecycle/><RouteBoundary key={pathname}><Suspense fallback={<RouteLoading/>}><Routes>
    <Route path="/" element={<LandingEntry/>}/>
    <Route element={<AppShell/>}>
      <Route path="dashboard" element={<OverviewPage/>}/>
      <Route path="models" element={<ModelsPage/>}/>
      <Route path="validation" element={<ValidationPage/>}/>
      <Route path="system" element={<SystemPage/>}/>
    </Route>
    <Route path="overview" element={<Navigate to="/dashboard" replace/>}/>
    <Route path="*" element={<main id="main-content" tabIndex={-1} className="cm-route-loading"><h1>That page is not here.</h1><Link to="/dashboard">Open dashboard</Link><Link to="/">Explore CopperMind</Link></main>}/>
  </Routes></Suspense></RouteBoundary></MotionConfig>;
}
export default function App() {
  return <QueryClientProvider client={queryClient}><BrowserRouter><AppRoutes/><SpeedInsights/></BrowserRouter></QueryClientProvider>;
}
