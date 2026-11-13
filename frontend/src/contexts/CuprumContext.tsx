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

export interface CuprumContextValue {
  feed: CuprumFeedState;
  publishFeed: (next: Partial<CuprumFeedInput> | null) => void;
  clearFeed: () => void;
  position: CuprumPosition;
  setPosition: (next: CuprumPosition) => void;
  resetPosition: () => void;
  isDocked: boolean;
  minimized: boolean;
  setMinimized: (next: boolean) => void;
  toggleMinimized: () => void;
  visible: boolean;
  setVisible: (next: boolean) => void;
  widgetSize: CuprumSize;
  setWidgetSize: (size: CuprumSize) => void;
  viewport: CuprumViewport;
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
const CuprumFeedActionsContext = createContext<{
  publishFeed: (next: Partial<CuprumFeedInput> | null) => void;
  clearFeed: () => void;
} | null>(null);
const CuprumContext = createContext<CuprumContextValue | null>(null);

interface CuprumProviderProps {
  children: ReactNode;
  initialFeed?: Partial<CuprumFeedInput>;
  isolated?: boolean;
}

function CuprumProviderInner({ children, initialFeed }: Omit<CuprumProviderProps, 'isolated'>) {
  const [feed, setFeed] = useState<CuprumFeedState>(() =>
    initialFeed ? mergeFeedState(INITIAL_CUPRUM_FEED, initialFeed) : INITIAL_CUPRUM_FEED,
  );
  const [viewport, setViewport] = useState<CuprumViewport>(() => getViewportSize());
  const [minimized, setMinimizedState] = useState<boolean>(() =>
    readStoredBoolean(CUPRUM_STORAGE_KEYS.minimized, false),
  );
  const [visible, setVisibleState] = useState<boolean>(() =>
    readStoredBoolean(CUPRUM_STORAGE_KEYS.visible, true),
  );
  const [widgetSize, setWidgetSizeState] = useState<CuprumSize>(() =>
    readStoredBoolean(CUPRUM_STORAGE_KEYS.minimized, false) ? DEFAULT_PILL_SIZE : DEFAULT_CUPRUM_SIZE,
  );

  const storedRef = useRef<StoredPositionPayload | null>(readStoredPosition());
  const [isDocked, setIsDocked] = useState<boolean>(() => {
    const stored = storedRef.current;
    return stored ? stored.docked === true : true;
  });
  const [position, setPositionState] = useState<CuprumPosition>(() => {
    const vp = getViewportSize();
    const initialSize = readStoredBoolean(CUPRUM_STORAGE_KEYS.minimized, false)
      ? DEFAULT_PILL_SIZE
      : DEFAULT_CUPRUM_SIZE;
    const stored = storedRef.current;
    if (!stored || stored.docked) {
      return getDefaultDockedPosition(initialSize, vp);
    }
    return clampCuprumPosition({ x: stored.x, y: stored.y }, initialSize, vp);
  });

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

  const publishFeed = useCallback((next: Partial<CuprumFeedInput> | null) => {
    if (!next) return;
    setFeed((prev) => mergeFeedState(prev, next));
  }, []);

  const clearFeed = useCallback(() => {
    setFeed(INITIAL_CUPRUM_FEED);
  }, []);

  const setPosition = useCallback(
    (next: CuprumPosition) => {
      const clamped = clampCuprumPosition(next, widgetSize, viewport);
      setIsDocked(false);
      setPositionState(clamped);
      writeStoredPosition({ x: clamped.x, y: clamped.y, docked: false });
    },
    [widgetSize, viewport],
  );

  const resetPosition = useCallback(() => {
    const dockedPos = getDefaultDockedPosition(widgetSize, viewport);
    setIsDocked(true);
    setPositionState(dockedPos);
    writeStoredPosition(null);
  }, [widgetSize, viewport]);

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
    setWidgetSizeState((prev) =>
      prev.width === rounded.width && prev.height === rounded.height ? prev : rounded,
    );
  }, []);

  const feedActions = useMemo(
    () => ({ publishFeed, clearFeed }),
    [publishFeed, clearFeed],
  );

  const value = useMemo<CuprumContextValue>(
    () => ({
      feed,
      publishFeed,
      clearFeed,
      position,
      setPosition,
      resetPosition,
      isDocked,
      minimized,
      setMinimized,
      toggleMinimized,
      visible,
      setVisible,
      widgetSize,
      setWidgetSize,
      viewport,
    }),
    [
      feed,
      publishFeed,
      clearFeed,
      position,
      setPosition,
      resetPosition,
      isDocked,
      minimized,
      setMinimized,
      toggleMinimized,
      visible,
      setVisible,
      widgetSize,
      setWidgetSize,
      viewport,
    ],
  );

  return (
    <CuprumFeedActionsContext.Provider value={feedActions}>
      <CuprumFeedStateContext.Provider value={feed}>
        <CuprumContext.Provider value={value}>{children}</CuprumContext.Provider>
      </CuprumFeedStateContext.Provider>
    </CuprumFeedActionsContext.Provider>
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

  return {
    feed: feed ?? INITIAL_CUPRUM_FEED,
    publishFeed,
    clearFeed,
  };
}
