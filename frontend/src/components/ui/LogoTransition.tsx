import { createContext, useCallback, useContext, useLayoutEffect, useRef, type MouseEvent as ReactMouseEvent, type ReactNode } from 'react';
import { Link, useLocation, useNavigate, type LinkProps } from 'react-router-dom';

const ROUTE_REVEAL_MS = 2_400;
const TransitionContext = createContext<((to?: string) => void) | null>(null);

/** Keep the light page-to-page reveal without inserting a full-screen logo transition. */
export function LogoTransitionProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const { pathname, search } = useLocation();
  const previousLocation = useRef(`${pathname}${search}`);
  const revealTimer = useRef<number | undefined>(undefined);

  useLayoutEffect(() => {
    const location = `${pathname}${search}`;
    if (previousLocation.current === location) return;
    previousLocation.current = location;

    const root = document.documentElement;
    root.dataset.cmRouteTransition = 'arriving';
    root.dataset.cmTransitionPhase = 'reveal';
    if (revealTimer.current !== undefined) window.clearTimeout(revealTimer.current);
    revealTimer.current = window.setTimeout(() => {
      delete root.dataset.cmTransitionPhase;
      delete root.dataset.cmRouteTransition;
      revealTimer.current = undefined;
    }, ROUTE_REVEAL_MS);

    return () => {
      if (revealTimer.current !== undefined) window.clearTimeout(revealTimer.current);
    };
  }, [pathname, search]);

  const startTransition = useCallback((to = '/dashboard') => navigate(to), [navigate]);
  return <TransitionContext.Provider value={startTransition}>{children}</TransitionContext.Provider>;
}

type EnterWorkspaceLinkProps = Omit<LinkProps, 'to'>;

export function EnterWorkspaceLink({ onClick, ...props }: EnterWorkspaceLinkProps) {
  const navigateToWorkspace = useContext(TransitionContext);
  const handleClick = (event: ReactMouseEvent<HTMLAnchorElement>) => {
    onClick?.(event);
    const { currentTarget } = event;
    const modified = event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey;
    if (event.defaultPrevented || !navigateToWorkspace || modified || (currentTarget.target && currentTarget.target !== '_self')) return;
    event.preventDefault();
    navigateToWorkspace('/dashboard');
  };

  return <Link {...props} to="/dashboard" onClick={handleClick}/>;
}
