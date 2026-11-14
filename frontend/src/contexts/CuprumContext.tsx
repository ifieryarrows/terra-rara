import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import type {
  CommentaryResponse,
  ExplainResponse,
  TFTAnalysisResponse,
} from '../types';

export interface CuprumPosition {
  x: number;
  y: number;
}

export interface CuprumSize {
  width: number;
  height: number;
}

export interface CuprumViewport {
  width: number;
  height: number;
}

export interface CuprumQuoteSnapshot {
  symbol?: string | null;
  price?: number | null;
  previous_close?: number | null;
  change?: number | null;
  change_pct?: number | null;
  currency?: string | null;
  as_of?: string | null;
  fetched_at?: string | null;
  market_status?: string | null;
  [key: string]: unknown;
}

export type CuprumSummaryPayload =
  | string
  | {
      headline?: string | null;
      text?: string | null;
      regime?: string | null;
      conviction?: number | string | null;
      stale?: boolean | null;
      [key: string]: unknown;
    }
  | null;

export interface CuprumSentimentSnapshot {
  sentiment_index?: number | null;
  sentiment_label?: string | null;
  final_score?: number | null;
  label?: string | null;
  [key: string]: unknown;
}

export interface CuprumTelemetry {
  activeSymbol?: string | null;
  trendPct?: number | null;
  conviction?: number | string | null;
  confidenceScore?: number | null;
  sentimentScore?: number | null;
  regime?: string | null;
  isStale?: boolean;
  hasError?: boolean;
  quote?: CuprumQuoteSnapshot | Record<string, unknown> | null;
  forecast?: TFTAnalysisResponse | Record<string, unknown> | null;
  sentiment?: CuprumSentimentSnapshot | Record<string, unknown> | object | null;
  summary?: CuprumSummaryPayload;
}

export interface CuprumFeedInput extends CuprumTelemetry {
  commentary?: CommentaryResponse | null;
  explain?: ExplainResponse | null;
  tftAnalysis?: TFTAnalysisResponse | null;
  loading?: boolean;
  error?: string | null;
}

export interface CuprumFeedState extends CuprumFeedInput {
  commentary: CommentaryResponse | null;
  explain: ExplainResponse | null;
  tftAnalysis: TFTAnalysisResponse | null;
  loading: boolean;
  error: string | null;
  activeSymbol: string | null;
  trendPct: number | null;
  conviction: number | string | null;
  confidenceScore: number | null;
  sentimentScore: number | null;
  regime: string | null;
  isStale: boolean;
  hasError: boolean;
  quote: CuprumQuoteSnapshot | Record<string, unknown> | null;
  forecast: TFTAnalysisResponse | Record<string, unknown> | null;
  sentiment: CuprumSentimentSnapshot | Record<string, unknown> | object | null;
  summary: CuprumSummaryPayload;
  updatedAt: number | null;
}

export interface CuprumFeedActions {
  publishFeed: (next: Partial<CuprumFeedInput> | null) => void;
  clearFeed: () => void;
}

export interface CuprumDockState {
  position: CuprumPosition;
  isDocked: boolean;
  minimized: boolean;
  visible: boolean;
  widgetSize: CuprumSize;
  viewport: CuprumViewport;
}

export interface CuprumDockActions {
  setPosition: (next: CuprumPosition) => void;
  resetPosition: () => void;
  setMinimized: (next: boolean) => void;
  toggleMinimized: () => void;
  setVisible: (next: boolean) => void;
  setWidgetSize: (size: CuprumSize) => void;
}

export interface CuprumContextValue extends CuprumFeedActions, CuprumDockState, CuprumDockActions {
  feed: CuprumFeedState;
}

export const CUPRUM_SAFE_MARGIN = 16;
export const CUPRUM_KEY_STEP_PX = 15;
export const CUPRUM_DRAG_THRESHOLD_PX = 4;

export const DEFAULT_CUPRUM_SIZE: CuprumSize = { width: 176, height: 172 };
export const DEFAULT_PILL_SIZE: CuprumSize = { width: 132, height: 42 };

export const CUPRUM_STORAGE_KEYS = {
  position: 'cm-cuprum-position-v1',
  minimized: 'cm-cuprum-minimized-v1',
  visible: 'cm-cuprum-visible-v1',
} as const;

export const INITIAL_CUPRUM_FEED: CuprumFeedState = {
  commentary: null,
  explain: null,
  tftAnalysis: null,
  loading: false,
  error: null,
  activeSymbol: null,
  trendPct: null,
  conviction: null,
  confidenceScore: null,
  sentimentScore: null,
  regime: null,
  isStale: false,
  hasError: false,
  quote: null,
  forecast: null,
  sentiment: null,
  summary: null,
  updatedAt: null,
};

export function getViewportSize(): CuprumViewport {
  if (typeof window === 'undefined') {
    return { width: 1280, height: 800 };
  }
  const width = window.innerWidth || document.documentElement?.clientWidth || 1280;
  const height = window.innerHeight || document.documentElement?.clientHeight || 800;
  return { width, height };
}

export function clampCuprumPosition(
  pos: CuprumPosition,
  size: CuprumSize = DEFAULT_CUPRUM_SIZE,
  viewport: CuprumViewport = getViewportSize(),
  margin: number = CUPRUM_SAFE_MARGIN,
): CuprumPosition {
  const safeW = Math.max(1, Number.isFinite(size.width) && size.width > 0 ? size.width : DEFAULT_CUPRUM_SIZE.width);
  const safeH = Math.max(1, Number.isFinite(size.height) && size.height > 0 ? size.height : DEFAULT_CUPRUM_SIZE.height);
  const vpW = Math.max(safeW, Number.isFinite(viewport.width) && viewport.width > 0 ? viewport.width : 1280);
  const vpH = Math.max(safeH, Number.isFinite(viewport.height) && viewport.height > 0 ? viewport.height : 800);

  const effMarginX = Math.max(4, Math.min(margin, Math.floor((vpW - safeW) / 2)));
  const effMarginY = Math.max(4, Math.min(margin, Math.floor((vpH - safeH) / 2)));
  const minX = effMarginX;
  const maxX = Math.max(minX, vpW - safeW - effMarginX);
  const minY = effMarginY;
  const maxY = Math.max(minY, vpH - safeH - effMarginY);

  const rawX = Number.isFinite(pos.x) ? pos.x : maxX;
  const rawY = Number.isFinite(pos.y) ? pos.y : maxY;

  return {
    x: Math.round(Math.min(Math.max(rawX, minX), maxX)),
    y: Math.round(Math.min(Math.max(rawY, minY), maxY)),
  };
}

export function getDefaultDockedPosition(
  size: CuprumSize = DEFAULT_CUPRUM_SIZE,
  viewport: CuprumViewport = getViewportSize(),
  margin: number = CUPRUM_SAFE_MARGIN,
): CuprumPosition {
  const safeW = size.width > 0 ? size.width : DEFAULT_CUPRUM_SIZE.width;
  const safeH = size.height > 0 ? size.height : DEFAULT_CUPRUM_SIZE.height;
  const vpW = viewport.width > 0 ? viewport.width : 1280;
  const vpH = viewport.height > 0 ? viewport.height : 800;

  return clampCuprumPosition(
    {
      x: vpW - safeW - margin,
      y: vpH - safeH - margin,
    },
    { width: safeW, height: safeH },
    { width: vpW, height: vpH },
    margin,
  );
}

interface StoredPositionPayload {
  x: number;
  y: number;
  docked?: boolean;
}

function readStoredPosition(): StoredPositionPayload | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(CUPRUM_STORAGE_KEYS.position);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredPositionPayload>;
    if (typeof parsed.x === 'number' && Number.isFinite(parsed.x) && typeof parsed.y === 'number' && Number.isFinite(parsed.y)) {
      return {
        x: parsed.x,
        y: parsed.y,
        docked: parsed.docked === true,
      };
    }
    return null;
  } catch {
    return null;
  }
}

function writeStoredPosition(payload: StoredPositionPayload | null): void {
  if (typeof window === 'undefined') return;
  try {
    if (!payload) {
      window.localStorage.removeItem(CUPRUM_STORAGE_KEYS.position);
    } else {
      window.localStorage.setItem(CUPRUM_STORAGE_KEYS.position, JSON.stringify(payload));
    }
  } catch {
    // Ignore storage quota or private-mode access errors.
  }
}

function readStoredBoolean(key: string, fallback: boolean): boolean {
  if (typeof window === 'undefined') return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    if (raw === 'true') return true;
    if (raw === 'false') return false;
    return fallback;
  } catch {
    return fallback;
  }
}

function writeStoredBoolean(key: string, value: boolean): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(key, value ? 'true' : 'false');
  } catch {
    // Ignore storage errors.
  }
}

function isTftAnalysisPayload(value: unknown): value is TFTAnalysisResponse {
  return (
    typeof value === 'object' &&
    value !== null &&
    'horizon_days' in value &&
    'prediction_log_return' in value
  );
}

function mergeFeedState(prev: CuprumFeedState, next: Partial<CuprumFeedInput>): CuprumFeedState {
  const merged: CuprumFeedState = {
    commentary: next.commentary !== undefined ? next.commentary : prev.commentary,
    explain: next.explain !== undefined ? next.explain : prev.explain,
    tftAnalysis:
      next.tftAnalysis !== undefined
        ? next.tftAnalysis
        : next.forecast !== undefined && isTftAnalysisPayload(next.forecast)
          ? next.forecast
          : prev.tftAnalysis,
    loading: next.loading !== undefined ? next.loading : prev.loading,
    error: next.error !== undefined ? next.error : prev.error,
    activeSymbol: next.activeSymbol !== undefined ? next.activeSymbol : prev.activeSymbol,
    trendPct: next.trendPct !== undefined ? next.trendPct : prev.trendPct,
    conviction: next.conviction !== undefined ? next.conviction : prev.conviction,
    confidenceScore: next.confidenceScore !== undefined ? next.confidenceScore : prev.confidenceScore,
    sentimentScore: next.sentimentScore !== undefined ? next.sentimentScore : prev.sentimentScore,
    regime: next.regime !== undefined ? next.regime : prev.regime,
    isStale: next.isStale !== undefined ? next.isStale : prev.isStale,
    hasError: next.hasError !== undefined ? next.hasError : prev.hasError,
    quote: next.quote !== undefined ? next.quote : prev.quote,
    forecast:
      next.forecast !== undefined
        ? next.forecast
        : next.tftAnalysis !== undefined
          ? next.tftAnalysis
          : prev.forecast,
    sentiment: next.sentiment !== undefined ? next.sentiment : prev.sentiment,
    summary: next.summary !== undefined ? next.summary : prev.summary,
    updatedAt: prev.updatedAt,
  };

  const keys = Object.keys(merged) as (keyof CuprumFeedState)[];
  let changed = false;
  for (const key of keys) {
    if (key === 'updatedAt') continue;
    if (!Object.is(merged[key], prev[key])) {
      changed = true;
      break;
    }
  }

  if (!changed) return prev;
  merged.updatedAt = Date.now();
  return merged;
}

const CuprumFeedStateContext = createContext<CuprumFeedState | null>(null);
const CuprumFeedActionsContext = createContext<CuprumFeedActions | null>(null);
const CuprumDockStateContext = createContext<CuprumDockState | null>(null);
const CuprumDockActionsContext = createContext<CuprumDockActions | null>(null);
const CuprumContext = createContext<CuprumContextValue | null>(null);

interface CuprumProviderProps {
  children: ReactNode;
  initialFeed?: Partial<CuprumFeedInput>;
  isolated?: boolean;
}

function CuprumFeedProvider({
  children,
  initialFeed,
}: {
  children: ReactNode;
  initialFeed?: Partial<CuprumFeedInput>;
}) {
  const [feed, setFeed] = useState<CuprumFeedState>(() =>
    initialFeed ? mergeFeedState(INITIAL_CUPRUM_FEED, initialFeed) : INITIAL_CUPRUM_FEED,
  );

  const publishFeed = useCallback((next: Partial<CuprumFeedInput> | null) => {
    if (!next) return;
    setFeed((prev) => mergeFeedState(prev, next));
  }, []);

  const clearFeed = useCallback(() => {
    setFeed(INITIAL_CUPRUM_FEED);
  }, []);

  const feedActions = useMemo<CuprumFeedActions>(
    () => ({ publishFeed, clearFeed }),
    [publishFeed, clearFeed],
  );

  return (
    <CuprumFeedActionsContext.Provider value={feedActions}>
      <CuprumFeedStateContext.Provider value={feed}>{children}</CuprumFeedStateContext.Provider>
    </CuprumFeedActionsContext.Provider>
  );
}

function CuprumDockProvider({ children }: { children: ReactNode }) {
  const [viewport, setViewport] = useState<CuprumViewport>(() => getViewportSize());
  const [minimized, setMinimizedState] = useState<boolean>(() =>
    readStoredBoolean(CUPRUM_STORAGE_KEYS.minimized, false),
  );
  const [visible, setVisibleState] = useState<boolean>(() =>
    readStoredBoolean(CUPRUM_STORAGE_KEYS.visible, true),
  );
  const [widgetSize, setWidgetSizeState] = useState<CuprumSize>(() =>
    minimized ? DEFAULT_PILL_SIZE : DEFAULT_CUPRUM_SIZE,
  );

  const storedRef = useRef<StoredPositionPayload | null>(readStoredPosition());
  const [isDocked, setIsDocked] = useState<boolean>(() => {
    const stored = storedRef.current;
    return stored ? stored.docked === true : true;
  });
  const [position, setPositionState] = useState<CuprumPosition>(() => {
    const vp = getViewportSize();
    const initialSize = minimized ? DEFAULT_PILL_SIZE : DEFAULT_CUPRUM_SIZE;
    const stored = storedRef.current;
    if (!stored || stored.docked) {
      return getDefaultDockedPosition(initialSize, vp);
    }
    return clampCuprumPosition({ x: stored.x, y: stored.y }, initialSize, vp);
  });

  // Keep latest size & viewport in refs so position actions remain referentially stable.
  const widgetSizeRef = useRef(widgetSize);
  const viewportRef = useRef(viewport);
  useEffect(() => {
    widgetSizeRef.current = widgetSize;
    viewportRef.current = viewport;
  }, [widgetSize, viewport]);

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const handleResize = () => {
      const nextVp = getViewportSize();
      setViewport((prev) =>
        prev.width === nextVp.width && prev.height === nextVp.height ? prev : nextVp,
      );
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    setPositionState((prev) => {
      const target = isDocked
        ? getDefaultDockedPosition(widgetSize, viewport)
        : clampCuprumPosition(prev, widgetSize, viewport);
      if (target.x === prev.x && target.y === prev.y) return prev;
      if (!isDocked) {
        writeStoredPosition({ x: target.x, y: target.y, docked: false });
      }
      return target;
    });
  }, [isDocked, widgetSize, viewport]);

  const setPosition = useCallback((next: CuprumPosition) => {
    const clamped = clampCuprumPosition(next, widgetSizeRef.current, viewportRef.current);
    setIsDocked(false);
    setPositionState(clamped);
    writeStoredPosition({ x: clamped.x, y: clamped.y, docked: false });
  }, []);

  const resetPosition = useCallback(() => {
    const dockedPos = getDefaultDockedPosition(widgetSizeRef.current, viewportRef.current);
    setIsDocked(true);
    setPositionState(dockedPos);
    writeStoredPosition(null);
  }, []);

  const setMinimized = useCallback((next: boolean) => {
    setMinimizedState(next);
    writeStoredBoolean(CUPRUM_STORAGE_KEYS.minimized, next);
  }, []);

  const toggleMinimized = useCallback(() => {
    setMinimizedState((prev) => {
      const next = !prev;
      writeStoredBoolean(CUPRUM_STORAGE_KEYS.minimized, next);
      return next;
    });
  }, []);

  const setVisible = useCallback((next: boolean) => {
    setVisibleState(next);
    writeStoredBoolean(CUPRUM_STORAGE_KEYS.visible, next);
  }, []);

  const setWidgetSize = useCallback((size: CuprumSize) => {
    if (!(size.width > 0 && size.height > 0)) return;
    const rounded = { width: Math.ceil(size.width), height: Math.ceil(size.height) };
    widgetSizeRef.current = rounded;
    setWidgetSizeState((prev) =>
      prev.width === rounded.width && prev.height === rounded.height ? prev : rounded,
    );
  }, []);

  const dockActions = useMemo<CuprumDockActions>(
    () => ({
      setPosition,
      resetPosition,
      setMinimized,
      toggleMinimized,
      setVisible,
      setWidgetSize,
    }),
    [setPosition, resetPosition, setMinimized, toggleMinimized, setVisible, setWidgetSize],
  );

  const dockState = useMemo<CuprumDockState>(
    () => ({
      position,
      isDocked,
      minimized,
      visible,
      widgetSize,
      viewport,
    }),
    [position, isDocked, minimized, visible, widgetSize, viewport],
  );

  return (
    <CuprumDockActionsContext.Provider value={dockActions}>
      <CuprumDockStateContext.Provider value={dockState}>{children}</CuprumDockStateContext.Provider>
    </CuprumDockActionsContext.Provider>
  );
}

function CuprumLegacyBridge({ children }: { children: ReactNode }) {
  const feed = useContext(CuprumFeedStateContext) ?? INITIAL_CUPRUM_FEED;
  const feedActions = useContext(CuprumFeedActionsContext);
  const dockState = useContext(CuprumDockStateContext);
  const dockActions = useContext(CuprumDockActionsContext);

  const value = useMemo<CuprumContextValue>(() => {
    if (!feedActions || !dockState || !dockActions) {
      throw new Error('CuprumLegacyBridge must be rendered inside CuprumProvider');
    }
    return {
      feed,
      publishFeed: feedActions.publishFeed,
      clearFeed: feedActions.clearFeed,
      position: dockState.position,
      setPosition: dockActions.setPosition,
      resetPosition: dockActions.resetPosition,
      isDocked: dockState.isDocked,
      minimized: dockState.minimized,
      setMinimized: dockActions.setMinimized,
      toggleMinimized: dockActions.toggleMinimized,
      visible: dockState.visible,
      setVisible: dockActions.setVisible,
      widgetSize: dockState.widgetSize,
      setWidgetSize: dockActions.setWidgetSize,
      viewport: dockState.viewport,
    };
  }, [feed, feedActions, dockState, dockActions]);

  return <CuprumContext.Provider value={value}>{children}</CuprumContext.Provider>;
}

function CuprumProviderInner({ children, initialFeed }: Omit<CuprumProviderProps, 'isolated'>) {
  return (
    <CuprumFeedProvider initialFeed={initialFeed}>
      <CuprumDockProvider>
        <CuprumLegacyBridge>{children}</CuprumLegacyBridge>
      </CuprumDockProvider>
    </CuprumFeedProvider>
  );
}

export function CuprumProvider({ children, initialFeed, isolated = false }: CuprumProviderProps) {
  const existing = useContext(CuprumContext);
  if (existing && !isolated) {
    return <>{children}</>;
  }
  return <CuprumProviderInner initialFeed={initialFeed}>{children}</CuprumProviderInner>;
}

export function useOptionalCuprumContext(): CuprumContextValue | null {
  return useContext(CuprumContext);
}

export function useOptionalCuprumFeedState(): CuprumFeedState | null {
  return useContext(CuprumFeedStateContext);
}

export function useOptionalCuprumDockState(): CuprumDockState | null {
  return useContext(CuprumDockStateContext);
}

export function useOptionalCuprumDockActions(): CuprumDockActions | null {
  return useContext(CuprumDockActionsContext);
}

export function useCuprum(): CuprumContextValue {
  const ctx = useContext(CuprumContext);
  if (!ctx) {
    throw new Error('useCuprum must be used within a CuprumProvider');
  }
  return ctx;
}

const NOOP_PUBLISH = () => {};
const NOOP_CLEAR = () => {};

/**
 * Subscribe or publish market telemetry into the global Cuprum pet context.
 * Calling `useCuprumFeed(telemetry)` syncs new telemetry without triggering re-renders
 * when values are unchanged; calling `useCuprumFeed()` returns `{ feed, publishFeed, clearFeed }`.
 */
export function useCuprumFeed(
  nextFeed?: Partial<CuprumFeedInput> | null,
  options?: { resetOnUnmount?: boolean },
) {
  const actions = useContext(CuprumFeedActionsContext);
  const feed = useContext(CuprumFeedStateContext);

  const publishFeed = actions?.publishFeed ?? NOOP_PUBLISH;
  const clearFeed = actions?.clearFeed ?? NOOP_CLEAR;

  useEffect(() => {
    if (nextFeed) {
      publishFeed(nextFeed);
    }
  }, [nextFeed, publishFeed]);

  const resetOnUnmount = options?.resetOnUnmount === true;
  useEffect(() => {
    if (!resetOnUnmount) return undefined;
    return () => clearFeed();
  }, [resetOnUnmount, clearFeed]);

  return useMemo(
    () => ({
      feed: feed ?? INITIAL_CUPRUM_FEED,
      publishFeed,
      clearFeed,
    }),
    [feed, publishFeed, clearFeed],
  );
}
