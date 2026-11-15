import {
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react';

const useIsomorphicLayoutEffect =
  typeof window !== 'undefined' ? useLayoutEffect : useEffect;
import {
  CUPRUM_DRAG_THRESHOLD_PX,
  CUPRUM_KEY_STEP_PX,
  CUPRUM_STORAGE_KEYS,
  DEFAULT_CUPRUM_SIZE,
  DEFAULT_PILL_SIZE,
  clampCuprumPosition,
  getDefaultDockedPosition,
  getViewportSize,
  useOptionalCuprumDockActions,
  useOptionalCuprumDockState,
  useOptionalCuprumFeedState,
  type CuprumPosition,
  type CuprumSize,
  type CuprumViewport,
} from '../contexts/CuprumContext';
import type {
  CommentaryResponse,
  ExplainResponse,
  TFTAnalysisResponse,
} from '../types';
import {
  applyCuprumChoice,
  buildCuprumContext,
  createCuprumRuntime,
  pickIdleLine,
  resolveCuprumNode,
  ROOT_NODE_ID,
  type CuprumChoice,
  type CuprumIdleLine,
  type CuprumMood,
  type CuprumRuntime,
} from '../utils/cuprumDialogue';
import {
  INITIAL_CUPRUM_SPRITE_STATE,
  advanceCuprumSprite,
  cuprumBlinkGapMs,
  cuprumSpriteFrame,
  type CuprumSpriteEvent,
  type CuprumSpriteFrame,
  type CuprumSpriteState,
} from '../utils/cuprumSpriteMachine';
import {
  buildCuprumBriefing,
  synthesizeCommentaryFromTelemetry,
  type CuprumTelemetrySnapshot,
} from '../utils/cuprumVoice';
import { playCuprumChirp, playCuprumTap } from '../utils/cosmicVision';
import { CuprumSpeechBubble, computeBubblePlacement } from './CuprumSpeechBubble';
import './CuprumCompanionPanel.css';

export interface CuprumCompanionPanelProps {
  commentary?: CommentaryResponse | null;
  explain?: ExplainResponse | null;
  tftAnalysis?: TFTAnalysisResponse | null;
  loading?: boolean;
  error?: string;
}

interface MoodConfig {
  title: string;
  subtitle: string;
  prefix: string;
  ringColor: string;
  coreColor: string;
}

const MOOD_META: Record<CuprumMood, MoodConfig> = {
  molten_bull: {
    title: 'Molten Resonance',
    subtitle: 'A small forge with a big opinion.',
    prefix: 'cuprum-bull',
    ringColor: 'var(--cm-copper, #dda174)',
    coreColor: 'var(--cm-positive, #74c7a9)',
  },
  patina_bear: {
    title: 'Patina Resonance',
    subtitle: 'Green around the edges, soft in the middle.',
    prefix: 'cuprum-bear',
    ringColor: 'var(--cm-positive, #74c7a9)',
    coreColor: 'var(--cm-forecast, #8eadde)',
  },
  tempered_neutral: {
    title: 'Harmonic Core',
    subtitle: 'Listening to the quiet between prices.',
    prefix: 'cuprum-neutral',
    ringColor: 'var(--cm-forecast, #8eadde)',
    coreColor: 'var(--cm-copper, #dda174)',
  },
};

const IDLE_FIRST_DELAY_MS = 6500;
const IDLE_DELAY_MIN_MS = 22000;
const IDLE_DELAY_RANGE_MS = 16000;

function initialVisibility(): boolean {
  return typeof document === 'undefined' || document.visibilityState !== 'hidden';
}

export interface CuprumSessionState {
  mode: 'idle' | 'dialogue';
  activeNodeId: string | null;
  idleLine: CuprumIdleLine | null;
  isTyping: boolean;
}

export type CuprumSessionEvent =
  | { type: 'show-idle'; line: CuprumIdleLine }
  | { type: 'dismiss-idle' }
  | { type: 'open-dialogue'; nodeId: string }
  | { type: 'advance-dialogue'; nodeId: string }
  | { type: 'close-dialogue' }
  | { type: 'set-typing'; typing: boolean };

export const INITIAL_CUPRUM_SESSION_STATE: CuprumSessionState = {
  mode: 'idle',
  activeNodeId: null,
  idleLine: null,
  isTyping: false,
};

export function advanceCuprumSession(
  state: CuprumSessionState,
  event: CuprumSessionEvent,
): CuprumSessionState {
  switch (event.type) {
    case 'show-idle':
      if (state.mode !== 'idle') return state;
      return { ...state, idleLine: event.line };
    case 'dismiss-idle':
      if (!state.idleLine) return state;
      return { ...state, idleLine: null };
    case 'open-dialogue':
      return {
        mode: 'dialogue',
        activeNodeId: event.nodeId,
        idleLine: null,
        isTyping: false,
      };
    case 'advance-dialogue':
      if (state.mode !== 'dialogue') return state;
      return {
        ...state,
        activeNodeId: event.nodeId,
      };
    case 'close-dialogue':
      return {
        mode: 'idle',
        activeNodeId: null,
        idleLine: null,
        isTyping: false,
      };
    case 'set-typing':
      if (state.isTyping === event.typing) return state;
      return { ...state, isTyping: event.typing };
  }
}

interface DragSession {
  pointerId: number;
  target: HTMLElement;
  startClientX: number;
  startClientY: number;
  pointerOffsetX: number;
  pointerOffsetY: number;
  startPos: CuprumPosition;
  lastPreviewPos: CuprumPosition;
  moved: boolean;
  source: 'handle' | 'pet';
}

interface CuprumSpriteStageProps {
  moodPrefix: string;
  moodTitle: string;
  isTyping: boolean;
  alertEpoch: number;
  onClick: () => void;
  onPointerDown: (event: ReactPointerEvent<HTMLElement>) => void;
  onPointerMove: (event: ReactPointerEvent<HTMLElement>) => void;
  onPointerUp: (event: ReactPointerEvent<HTMLElement>) => void;
  onPointerCancel: () => void;
  onLostPointerCapture: (event: ReactPointerEvent<HTMLElement>) => void;
}

const SPRITE_FRAMES = ['idle-1', 'idle-2', 'blink', 'alert', 'talk'] as const;

const CuprumSpriteStage = memo(function CuprumSpriteStage({
  moodPrefix,
  moodTitle,
  isTyping,
  alertEpoch,
  onClick,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onPointerCancel,
  onLostPointerCapture,
}: CuprumSpriteStageProps) {
  const [frame, setFrame] = useState<CuprumSpriteFrame>(() =>
    cuprumSpriteFrame(INITIAL_CUPRUM_SPRITE_STATE),
  );
  const [talkFlap, setTalkFlap] = useState(true);
  const dispatchSpriteRef = useRef<(event: CuprumSpriteEvent) => void>(() => {});
  const preloadedPrefixesRef = useRef<Set<string>>(new Set());
  const preloadedImagesRef = useRef<HTMLImageElement[]>([]);

  // Preload small sprite frames once per mood when idle so swaps never flash or delay.
  useEffect(() => {
    if (typeof window === 'undefined' || preloadedPrefixesRef.current.has(moodPrefix)) return;

    const preload = () => {
      if (preloadedPrefixesRef.current.has(moodPrefix)) return;
      preloadedPrefixesRef.current.add(moodPrefix);
      const images: HTMLImageElement[] = [];
      for (const frameName of SPRITE_FRAMES) {
        const img = new Image();
        img.src = `/assets/cuprum/${moodPrefix}-${frameName}.webp?v=retro-v7`;
        if (typeof img.decode === 'function') {
          img.decode().catch(() => {});
        }
        images.push(img);
      }
      preloadedImagesRef.current = images;
    };

    if (typeof window.requestIdleCallback === 'function') {
      const idleId = window.requestIdleCallback(preload, { timeout: 1500 });
      return () => window.cancelIdleCallback(idleId);
    }
    const timer = window.setTimeout(preload, 250);
    return () => window.clearTimeout(timer);
  }, [moodPrefix]);

  // Retro pixel sprite machine: clean up-down bob + transient expressions.
  useEffect(() => {
    let machine: CuprumSpriteState = INITIAL_CUPRUM_SPRITE_STATE;
    const timers: Record<'tick' | 'blinkEnd' | 'alertEnd' | 'blinkRequest', number> = {
      tick: 0,
      blinkEnd: 0,
      alertEnd: 0,
      blinkRequest: 0,
    };
    const clearTimer = (key: keyof typeof timers) => {
      if (timers[key]) {
        window.clearTimeout(timers[key]);
        timers[key] = 0;
      }
    };
    const dispatch = (event: CuprumSpriteEvent) => {
      const { state, schedule } = advanceCuprumSprite(machine, event);
      machine = state;
      setFrame(cuprumSpriteFrame(state));
      for (const item of schedule) {
        switch (item.kind) {
          case 'tick':
            clearTimer('tick');
            timers.tick = window.setTimeout(() => dispatch('bob-tick'), item.delayMs);
            break;
          case 'blink-end':
            clearTimer('blinkEnd');
            timers.blinkEnd = window.setTimeout(() => dispatch('blink-end'), item.delayMs);
            break;
          case 'alert-end':
            clearTimer('alertEnd');
            timers.alertEnd = window.setTimeout(() => dispatch('alert-end'), item.delayMs);
            break;
          case 'blink-request':
            clearTimer('blinkRequest');
            timers.blinkRequest = window.setTimeout(
              () => dispatch('blink-request'),
              item.delayMs ?? cuprumBlinkGapMs(),
            );
            break;
          case 'cancel-tick':
            clearTimer('tick');
            break;
        }
      }
    };
    dispatchSpriteRef.current = dispatch;
    dispatch('mounted');
    return () => {
      dispatchSpriteRef.current = () => {};
      clearTimer('tick');
      clearTimer('blinkEnd');
      clearTimer('alertEnd');
      clearTimer('blinkRequest');
    };
  }, []);

  useEffect(() => {
    if (alertEpoch > 0) {
      dispatchSpriteRef.current('alert-start');
    }
  }, [alertEpoch]);

  // Talking mouth flap rhythm while text types out.
  useEffect(() => {
    if (!isTyping) {
      setTalkFlap(true);
      return;
    }
    const timer = window.setInterval(() => {
      setTalkFlap((prev) => !prev);
    }, 220);
    return () => {
      window.clearInterval(timer);
    };
  }, [isTyping]);

  const currentFrame: CuprumSpriteFrame | 'talk' = isTyping && talkFlap ? 'talk' : frame;
  const currentWebp = `/assets/cuprum/${moodPrefix}-${currentFrame}.webp?v=retro-v7`;
  const currentPng = `/assets/cuprum/${moodPrefix}-${currentFrame}.png?v=retro-v7`;

  return (
    <div
      className="cm-cuprum-sprite-stage"
      onClick={onClick}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onLostPointerCapture={onLostPointerCapture}
      aria-hidden="true"
    >
      <span className="cm-cuprum-sprite-frame">
        <img
          src={currentWebp}
          alt={`Cuprum — ${moodTitle}`}
          className="cm-cuprum-sprite-img"
          draggable={false}
          decoding="async"
          onError={(e) => {
            if (!e.currentTarget.src.endsWith('.png')) {
              e.currentTarget.src = currentPng;
            }
          }}
        />
      </span>
    </div>
  );
});

export function CuprumCompanionPanel({
  commentary,
  explain,
  tftAnalysis,
  loading,
  error,
}: CuprumCompanionPanelProps) {
  const feed = useOptionalCuprumFeedState();
  const dockState = useOptionalCuprumDockState();
  const dockActions = useOptionalCuprumDockActions();
  const hasProvider = dockState !== null && dockActions !== null;

  // Resolve effective market inputs: explicit props override context feed so standalone unit
  // tests and isolated previews work without requiring CuprumProvider.
  const propCommentary = commentary;
  const propExplain = explain;
  const propTftAnalysis = tftAnalysis;
  const propLoading = loading;
  const propError = error;

  const effectiveExplain =
    propExplain !== undefined ? propExplain : (feed?.explain ?? null);
  const effectiveTftAnalysis =
    propTftAnalysis !== undefined ? propTftAnalysis : (feed?.tftAnalysis ?? null);
  const effectiveLoading =
    propLoading !== undefined ? propLoading : (feed?.loading ?? false);
  const effectiveError =
    propError !== undefined
      ? propError
      : (feed?.error ??
        (feed?.hasError ? 'Market feed unavailable' : undefined)) ?? undefined;

  const telemetry = useMemo<CuprumTelemetrySnapshot | null>(() => {
    if (!feed) return null;
    const quotePct =
      typeof feed.quote?.change_pct === 'number' && Number.isFinite(feed.quote.change_pct)
        ? feed.quote.change_pct
        : null;
    const trendPct =
      typeof feed.trendPct === 'number' && Number.isFinite(feed.trendPct)
        ? feed.trendPct
        : quotePct;
    const hasSignal =
      feed.activeSymbol != null ||
      trendPct != null ||
      feed.conviction != null ||
      feed.confidenceScore != null ||
      feed.sentimentScore != null ||
      feed.regime != null ||
      feed.isStale ||
      feed.hasError;
    if (!hasSignal) return null;
    return {
      activeSymbol: feed.activeSymbol ?? null,
      trendPct,
      conviction: feed.conviction ?? null,
      confidenceScore: feed.confidenceScore ?? null,
      sentimentScore: feed.sentimentScore ?? null,
      regime: feed.regime ?? null,
      isStale: Boolean(feed.isStale),
      hasError: Boolean(feed.hasError),
    };
  }, [feed]);

  const rawCommentary =
    propCommentary !== undefined ? propCommentary : (feed?.commentary ?? null);

  const effectiveCommentary = useMemo(() => {
    if (rawCommentary !== null) return rawCommentary;
    if (effectiveLoading || effectiveError) return null;
    return synthesizeCommentaryFromTelemetry(telemetry, effectiveTftAnalysis);
  }, [rawCommentary, effectiveLoading, effectiveError, telemetry, effectiveTftAnalysis]);

  // Standalone dock state fallback when rendered outside CuprumProvider.
  const [localViewport, setLocalViewport] = useState<CuprumViewport>(() => getViewportSize());
  const [localMinimized, setLocalMinimized] = useState(false);
  const [localSize, setLocalSize] = useState<CuprumSize>(DEFAULT_CUPRUM_SIZE);
  const [localDocked, setLocalDocked] = useState(true);
  const [localPos, setLocalPos] = useState<CuprumPosition>(() =>
    getDefaultDockedPosition(DEFAULT_CUPRUM_SIZE, getViewportSize()),
  );

  useEffect(() => {
    if (hasProvider || typeof window === 'undefined') return undefined;
    const onResize = () => {
      const vp = getViewportSize();
      setLocalViewport((prev) => (prev.width === vp.width && prev.height === vp.height ? prev : vp));
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [hasProvider]);

  useEffect(() => {
    if (hasProvider) return;
    setLocalPos((prev) => {
      const target = localDocked
        ? getDefaultDockedPosition(localSize, localViewport)
        : clampCuprumPosition(prev, localSize, localViewport);
      return target.x === prev.x && target.y === prev.y ? prev : target;
    });
  }, [hasProvider, localDocked, localSize, localViewport]);

  const viewport = dockState ? dockState.viewport : localViewport;
  const minimized = dockState ? dockState.minimized : localMinimized;
  const isVisible = dockState ? dockState.visible : true;
  const currentSize = dockState ? dockState.widgetSize : localSize;
  const persistedPos = dockState ? dockState.position : localPos;

  const commitPosition = useCallback(
    (next: CuprumPosition) => {
      if (dockActions) {
        dockActions.setPosition(next);
      } else {
        setLocalDocked(false);
        setLocalPos(clampCuprumPosition(next, localSize, localViewport));
      }
    },
    [dockActions, localSize, localViewport],
  );

  const commitResetPosition = useCallback(() => {
    if (dockActions) {
      dockActions.resetPosition();
    } else {
      setLocalDocked(true);
      setLocalPos(getDefaultDockedPosition(localSize, localViewport));
      try {
        window.localStorage.removeItem(CUPRUM_STORAGE_KEYS.position);
      } catch {
        // Ignore storage errors in restricted environments.
      }
    }
  }, [dockActions, localSize, localViewport]);

  const handleToggleMinimized = useCallback(() => {
    void playCuprumTap({ volume: 0.4 }).catch(() => {});
    if (dockActions) {
      dockActions.toggleMinimized();
    } else {
      setLocalMinimized((prev) => !prev);
    }
  }, [dockActions]);

  const updateMeasuredSize = useCallback(
    (size: CuprumSize) => {
      if (!(size.width > 0 && size.height > 0)) return;
      if (dockActions) {
        dockActions.setWidgetSize(size);
      } else {
        const rounded = { width: Math.ceil(size.width), height: Math.ceil(size.height) };
        setLocalSize((prev) =>
          prev.width === rounded.width && prev.height === rounded.height ? prev : rounded,
        );
      }
    },
    [dockActions],
  );

  // Measure dock size for responsive clamping & bubble auto-flipping.
  const dockDocked = dockState ? dockState.isDocked : localDocked;
  const dockSize = dockState ? dockState.widgetSize : localSize;
  const dockPos = dockState ? dockState.position : localPos;
  const dockVp = dockState ? dockState.viewport : localViewport;
  const setDockPos = dockActions ? dockActions.setPosition : setLocalPos;

  const latestDockRef = useRef({ docked: dockDocked, pos: dockPos });
  latestDockRef.current = { docked: dockDocked, pos: dockPos };

  const anchorMetaRef = useRef<{
    init: boolean;
    size: CuprumSize;
    r: number | null;
    b: number | null;
    lastPos: CuprumPosition | null;
  }>({ init: false, size: dockSize, r: null, b: null, lastPos: null });

  const pendingAnchorRef = useRef<{
    targetSize: CuprumSize;
    prevSize: CuprumSize;
    prevPos: CuprumPosition;
    r: number | null;
    b: number | null;
  } | null>(null);

  const stageRef = useRef<HTMLDivElement>(null);
  useIsomorphicLayoutEffect(() => {
    const fallback = minimized ? DEFAULT_PILL_SIZE : DEFAULT_CUPRUM_SIZE;
    const node = stageRef.current;
    if (!node) {
      anchorMetaRef.current.init = false;
      pendingAnchorRef.current = null;
      updateMeasuredSize(fallback);
      return undefined;
    }

    const applyMeasure = (raw: CuprumSize) => {
      const nextSize = { width: Math.ceil(raw.width), height: Math.ceil(raw.height) };
      const meta = anchorMetaRef.current;
      const latest = latestDockRef.current;

      if (!meta.init) {
        meta.init = true;
        meta.size = nextSize;
        meta.lastPos = latest.pos;
        updateMeasuredSize(nextSize);
        return;
      }

      if (latest.docked) {
        meta.size = nextSize;
        meta.r = null;
        meta.b = null;
        meta.lastPos = null;
        pendingAnchorRef.current = null;
        updateMeasuredSize(nextSize);
        return;
      }

      if (meta.size.width !== nextSize.width || meta.size.height !== nextSize.height) {
        const movedExternally =
          meta.lastPos !== null &&
          (latest.pos.x !== meta.lastPos.x || latest.pos.y !== meta.lastPos.y);
        pendingAnchorRef.current = {
          targetSize: nextSize,
          prevSize: meta.size,
          prevPos: latest.pos,
          r: movedExternally ? null : meta.r,
          b: movedExternally ? null : meta.b,
        };
        meta.size = nextSize;
      }
      updateMeasuredSize(nextSize);
    };

    const measure = () => {
      const rect = node.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        applyMeasure({ width: rect.width, height: rect.height });
      } else {
        applyMeasure(fallback);
      }
    };

    measure();

    if (typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [minimized, updateMeasuredSize]);

  useIsomorphicLayoutEffect(() => {
    const pending = pendingAnchorRef.current;
    if (!pending || dockDocked) return;
    if (
      dockSize.width !== pending.targetSize.width ||
      dockSize.height !== pending.targetSize.height
    ) {
      return;
    }
    pendingAnchorRef.current = null;
    const { prevSize, prevPos, r: effR, b: effB } = pending;
    const intendedR =
      effR ??
      (prevPos.x + prevSize.width / 2 > dockVp.width / 2
        ? prevPos.x + prevSize.width
        : null);
    const intendedB = effB ?? prevPos.y + prevSize.height;
    const rawX = intendedR !== null ? intendedR - dockSize.width : prevPos.x;
    const rawY = intendedB - dockSize.height;
    const clamped = clampCuprumPosition({ x: rawX, y: rawY }, dockSize, dockVp);
    anchorMetaRef.current.r =
      intendedR !== null && clamped.x !== Math.round(rawX) ? intendedR : null;
    anchorMetaRef.current.b = clamped.y !== Math.round(rawY) ? intendedB : null;
    anchorMetaRef.current.lastPos = clamped;
    if (clamped.x !== prevPos.x || clamped.y !== prevPos.y) {
      setDockPos(clamped);
    }
  }, [dockSize, dockDocked, dockVp, setDockPos]);

  // Dragging & keyboard repositioning state (drag moves update stageRef transform directly)
  const [isDragging, setIsDragging] = useState(false);
  const [liveAnnouncement, setLiveAnnouncement] = useState('');
  const dragSessionRef = useRef<DragSession | null>(null);
  const suppressClickRef = useRef(false);
  const dragHandleRef = useRef<HTMLButtonElement>(null);

  const [session, setSession] = useState<CuprumSessionState>(INITIAL_CUPRUM_SESSION_STATE);
  const { mode, activeNodeId, idleLine, isTyping } = session;
  const [runtime, setRuntime] = useState<CuprumRuntime>(() => createCuprumRuntime());
  const [inView, setInView] = useState(true);
  const [docVisible, setDocVisible] = useState(initialVisibility);
  const [interactionEpoch, setInteractionEpoch] = useState(0);
  const [alertEpoch, setAlertEpoch] = useState(0);

  const modeRef = useRef<'idle' | 'dialogue'>('idle');
  const ambientTimerRef = useRef<number | null>(null);
  const petButtonRef = useRef<HTMLButtonElement>(null);
  const bubbleRef = useRef<HTMLDivElement>(null);
  const recentIdleIdsRef = useRef<string[]>([]);
  const hasShownIdleRef = useRef(false);
  const freshDataRef = useRef(false);
  const hasMountedRef = useRef(false);
  const generatedAt = effectiveCommentary?.generated_at ?? null;
  const lastGeneratedAtRef = useRef<string | null>(generatedAt);

  useIsomorphicLayoutEffect(() => {
    const node = stageRef.current;
    if (!node || !isVisible) return;
    const target =
      isDragging && dragSessionRef.current ? dragSessionRef.current.lastPreviewPos : persistedPos;
    node.style.transform = `translate3d(${target.x}px, ${target.y}px, 0)`;
  }, [persistedPos.x, persistedPos.y, isDragging, isVisible]);

  const ctx = useMemo(() => {
    const built = buildCuprumContext(
      effectiveCommentary,
      effectiveExplain,
      effectiveTftAnalysis,
      effectiveLoading,
      effectiveError,
    );
    const briefing = buildCuprumBriefing(
      effectiveCommentary,
      effectiveExplain,
      effectiveTftAnalysis,
      telemetry,
    );
    return {
      ...built,
      briefing,
      telemetry,
    };
  }, [
    effectiveCommentary,
    effectiveExplain,
    effectiveTftAnalysis,
    effectiveLoading,
    effectiveError,
    telemetry,
  ]);
  const moodMeta = MOOD_META[ctx.mood];

  const noteInteraction = useCallback(() => {
    setInteractionEpoch((value) => value + 1);
  }, []);

  // Notice a fresh pipeline snapshot so the next conversation can acknowledge it.
  useEffect(() => {
    if (!hasMountedRef.current) {
      hasMountedRef.current = true;
      lastGeneratedAtRef.current = generatedAt;
      return;
    }
    if (generatedAt && lastGeneratedAtRef.current && generatedAt !== lastGeneratedAtRef.current) {
      freshDataRef.current = true;
    }
    lastGeneratedAtRef.current = generatedAt;
  }, [generatedAt]);

  // Pause ambient chatter when the pet is off-screen, minimized, or the tab is hidden.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || typeof IntersectionObserver === 'undefined') return;
    const observer = new IntersectionObserver(
      ([entry]) => setInView(entry?.isIntersecting ?? true),
      { threshold: 0.2 },
    );
    observer.observe(stage);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    const onChange = () => setDocVisible(document.visibilityState !== 'hidden');
    document.addEventListener('visibilitychange', onChange);
    return () => document.removeEventListener('visibilitychange', onChange);
  }, []);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  // Minimizing closes any active conversation or ambient line so expanding starts clean.
  useEffect(() => {
    if (!minimized) return;
    if (ambientTimerRef.current !== null) {
      window.clearTimeout(ambientTimerRef.current);
      ambientTimerRef.current = null;
    }
    modeRef.current = 'idle';
    setSession((prev) =>
      prev.mode === 'idle' && !prev.idleLine && !prev.isTyping
        ? prev
        : advanceCuprumSession(prev, { type: 'close-dialogue' }),
    );
  }, [minimized]);

  // Ambient speech bubbles: first after a short welcome, then a long, irregular cadence.
  useEffect(() => {
    if (minimized || !isVisible || isDragging || mode !== 'idle' || idleLine || !inView || !docVisible) {
      return;
    }
    const delay = hasShownIdleRef.current
      ? IDLE_DELAY_MIN_MS + Math.random() * IDLE_DELAY_RANGE_MS
      : IDLE_FIRST_DELAY_MS;
    const timer = window.setTimeout(() => {
      ambientTimerRef.current = null;
      if (modeRef.current !== 'idle') return;
      const line = pickIdleLine(ctx, runtime, recentIdleIdsRef.current);
      recentIdleIdsRef.current = [...recentIdleIdsRef.current, line.id].slice(-4);
      hasShownIdleRef.current = true;
      setSession((prev) => advanceCuprumSession(prev, { type: 'show-idle', line }));
    }, delay);
    ambientTimerRef.current = timer;
    return () => {
      window.clearTimeout(timer);
      if (ambientTimerRef.current === timer) ambientTimerRef.current = null;
    };
  }, [minimized, isVisible, isDragging, mode, idleLine, inView, docVisible, interactionEpoch, ctx, runtime]);

  const dismissIdle = useCallback(() => {
    setSession((prev) => advanceCuprumSession(prev, { type: 'dismiss-idle' }));
  }, []);

  const setIsTyping = useCallback((typing: boolean) => {
    setSession((prev) => advanceCuprumSession(prev, { type: 'set-typing', typing }));
  }, []);

  const closeConversation = useCallback(
    (restoreFocus = false) => {
      noteInteraction();
      modeRef.current = 'idle';
      setSession((prev) => advanceCuprumSession(prev, { type: 'close-dialogue' }));
      if (restoreFocus) {
        window.setTimeout(() => petButtonRef.current?.focus({ preventScroll: true }), 0);
      }
    },
    [noteInteraction],
  );

  const closeDialogue = useCallback(() => {
    closeConversation(true);
  }, [closeConversation]);

  const openConversation = useCallback(() => {
    if (ambientTimerRef.current !== null) {
      window.clearTimeout(ambientTimerRef.current);
      ambientTimerRef.current = null;
    }
    modeRef.current = 'dialogue';
    const fresh = freshDataRef.current;
    freshDataRef.current = false;
    setRuntime((previous) => ({
      ...previous,
      turn: 0,
      freshData: fresh,
      sameChoiceStreak: 0,
      lastChoiceId: null,
      storyStreak: 0,
    }));
    setSession((prev) => advanceCuprumSession(prev, { type: 'open-dialogue', nodeId: ROOT_NODE_ID }));
    setAlertEpoch((value) => value + 1);
  }, []);

  const handlePetClick = useCallback(() => {
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    noteInteraction();
    void playCuprumChirp({ volume: mode === 'dialogue' ? 0.34 : 0.48 }).catch(() => {});
    if (mode === 'dialogue') {
      closeConversation(false);
      return;
    }
    openConversation();
  }, [mode, noteInteraction, closeConversation, openConversation]);

  const handleChoose = useCallback(
    (choice: CuprumChoice) => {
      noteInteraction();
      void playCuprumTap({ volume: 0.42 }).catch(() => {});
      if (choice.close) {
        closeConversation(false);
        return;
      }
      setRuntime((previous) => applyCuprumChoice(previous, choice));
      if (choice.next) {
        const nextId = choice.next;
        setSession((prev) => advanceCuprumSession(prev, { type: 'advance-dialogue', nodeId: nextId }));
      } else {
        setSession((prev) => advanceCuprumSession(prev, { type: 'set-typing', typing: false }));
      }
    },
    [noteInteraction, closeConversation],
  );

  const resolvedNode = useMemo(
    () => (activeNodeId ? resolveCuprumNode(activeNodeId, ctx, runtime) : null),
    [activeNodeId, ctx, runtime],
  );

  const windowFallbackCleanupRef = useRef<(() => void) | null>(null);

  // Pointer drag lifecycle (shared by dedicated drag handle AND pet sprite with >=4px threshold).
  const finishDrag = useCallback(
    (commit: boolean, finalClient?: { x: number; y: number }) => {
      windowFallbackCleanupRef.current?.();
      windowFallbackCleanupRef.current = null;

      const drag = dragSessionRef.current;
      if (!drag) return;
      dragSessionRef.current = null;

      try {
        if (drag.target.hasPointerCapture?.(drag.pointerId)) {
          drag.target.releasePointerCapture(drag.pointerId);
        }
      } catch {
        // Ignore release errors when pointer already left capture.
      }

      if (drag.moved) {
        suppressClickRef.current = true;
        if (typeof document !== 'undefined') {
          document.documentElement.dataset.cmSuppressClickSound = '1';
          window.setTimeout(() => {
            delete document.documentElement.dataset.cmSuppressClickSound;
          }, 25);
        }
        window.setTimeout(() => {
          suppressClickRef.current = false;
        }, 0);

        if (commit) {
          const targetPos = finalClient
            ? clampCuprumPosition(
                {
                  x: finalClient.x - drag.pointerOffsetX,
                  y: finalClient.y - drag.pointerOffsetY,
                },
                currentSize,
                viewport,
              )
            : drag.lastPreviewPos;
          if (stageRef.current) {
            stageRef.current.style.transform = `translate3d(${targetPos.x}px, ${targetPos.y}px, 0)`;
          }
          commitPosition(targetPos);
          setLiveAnnouncement(`Cuprum moved to left ${targetPos.x}, top ${targetPos.y}.`);
        } else {
          if (stageRef.current) {
            stageRef.current.style.transform = `translate3d(${drag.startPos.x}px, ${drag.startPos.y}px, 0)`;
          }
          setLiveAnnouncement('Cuprum move canceled.');
        }
      }

      setIsDragging(false);
    },
    [commitPosition, currentSize, viewport],
  );

  const startDrag = useCallback(
    (event: ReactPointerEvent<HTMLElement>, source: 'handle' | 'pet') => {
      if (event.button !== 0) return;
      noteInteraction();
      suppressClickRef.current = false;

      const basePos = latestDockRef.current.pos;
      const dockRect = stageRef.current?.getBoundingClientRect();
      const anchorX = dockRect && dockRect.width > 0 ? dockRect.left : basePos.x;
      const anchorY = dockRect && dockRect.height > 0 ? dockRect.top : basePos.y;

      let captured = false;
      try {
        event.currentTarget.setPointerCapture?.(event.pointerId);
        captured = event.currentTarget.hasPointerCapture?.(event.pointerId) ?? false;
      } catch {
        // Ignore capture errors when pointer capture isn't supported in test environments.
      }

      dragSessionRef.current = {
        pointerId: event.pointerId,
        target: event.currentTarget,
        startClientX: event.clientX,
        startClientY: event.clientY,
        pointerOffsetX: event.clientX - anchorX,
        pointerOffsetY: event.clientY - anchorY,
        startPos: basePos,
        lastPreviewPos: basePos,
        moved: false,
        source,
      };

      if (!captured && typeof window !== 'undefined') {
        const onWindowMove = (moveEvt: PointerEvent) => {
          const active = dragSessionRef.current;
          if (!active || active.pointerId !== moveEvt.pointerId) return;
          const dist = Math.hypot(moveEvt.clientX - active.startClientX, moveEvt.clientY - active.startClientY);
          if (!active.moved) {
            if (dist < CUPRUM_DRAG_THRESHOLD_PX) return;
            active.moved = true;
            suppressClickRef.current = true;
            setIsDragging(true);
          }
          const nextPos = clampCuprumPosition(
            {
              x: moveEvt.clientX - active.pointerOffsetX,
              y: moveEvt.clientY - active.pointerOffsetY,
            },
            currentSize,
            viewport,
          );
          active.lastPreviewPos = nextPos;
          if (stageRef.current) {
            stageRef.current.style.transform = `translate3d(${nextPos.x}px, ${nextPos.y}px, 0)`;
          }
        };
        const onWindowUp = (upEvt: PointerEvent) => {
          const active = dragSessionRef.current;
          if (!active || active.pointerId !== upEvt.pointerId) return;
          finishDrag(true, { x: upEvt.clientX, y: upEvt.clientY });
        };
        const onWindowCancel = (cancelEvt: PointerEvent) => {
          const active = dragSessionRef.current;
          if (!active || active.pointerId !== cancelEvt.pointerId) return;
          finishDrag(false);
        };
        window.addEventListener('pointermove', onWindowMove);
        window.addEventListener('pointerup', onWindowUp);
        window.addEventListener('pointercancel', onWindowCancel);
        windowFallbackCleanupRef.current = () => {
          window.removeEventListener('pointermove', onWindowMove);
          window.removeEventListener('pointerup', onWindowUp);
          window.removeEventListener('pointercancel', onWindowCancel);
        };
      }
    },
    [noteInteraction, currentSize, viewport, finishDrag],
  );

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      const drag = dragSessionRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;

      const distance = Math.hypot(
        event.clientX - drag.startClientX,
        event.clientY - drag.startClientY,
      );

      if (!drag.moved) {
        if (distance < CUPRUM_DRAG_THRESHOLD_PX) return;
        drag.moved = true;
        suppressClickRef.current = true;
        setIsDragging(true);
      }

      const nextPos = clampCuprumPosition(
        {
          x: event.clientX - drag.pointerOffsetX,
          y: event.clientY - drag.pointerOffsetY,
        },
        currentSize,
        viewport,
      );
      drag.lastPreviewPos = nextPos;
      if (stageRef.current) {
        stageRef.current.style.transform = `translate3d(${nextPos.x}px, ${nextPos.y}px, 0)`;
      }
    },
    [currentSize, viewport],
  );

  const handlePointerUp = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      const drag = dragSessionRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;
      finishDrag(true, { x: event.clientX, y: event.clientY });
    },
    [finishDrag],
  );

  const handlePointerCancel = useCallback(() => {
    finishDrag(false);
  }, [finishDrag]);

  const handlePetPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => startDrag(event, 'pet'),
    [startDrag],
  );

  const handleHandlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => startDrag(event, 'handle'),
    [startDrag],
  );

  const handleLostPointerCapture = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      const drag = dragSessionRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;
      finishDrag(true);
    },
    [finishDrag],
  );

  // Cancel active drag or close open conversation on Escape.
  useEffect(() => {
    if (!isDragging && mode !== 'dialogue') return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (dragSessionRef.current) {
        event.preventDefault();
        finishDrag(false);
        return;
      }
      if (mode === 'dialogue') {
        event.preventDefault();
        closeConversation(true);
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [isDragging, mode, finishDrag, closeConversation]);

  // Keyboard repositioning on dedicated drag handle button (15px arrow steps, Escape reset).
  const handleDragHandleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLButtonElement>) => {
      let nextPos: CuprumPosition;
      switch (event.key) {
        case 'ArrowLeft':
          nextPos = { x: persistedPos.x - CUPRUM_KEY_STEP_PX, y: persistedPos.y };
          break;
        case 'ArrowRight':
          nextPos = { x: persistedPos.x + CUPRUM_KEY_STEP_PX, y: persistedPos.y };
          break;
        case 'ArrowUp':
          nextPos = { x: persistedPos.x, y: persistedPos.y - CUPRUM_KEY_STEP_PX };
          break;
        case 'ArrowDown':
          nextPos = { x: persistedPos.x, y: persistedPos.y + CUPRUM_KEY_STEP_PX };
          break;
        case 'Escape':
          if (mode === 'dialogue' && !dragSessionRef.current) {
            return;
          }
          event.preventDefault();
          event.stopPropagation();
          if (dragSessionRef.current) {
            finishDrag(false);
            return;
          }
          commitResetPosition();
          setLiveAnnouncement('Cuprum reset to bottom-right corner.');
          return;
        default:
          return;
      }

      event.preventDefault();
      noteInteraction();
      const clamped = clampCuprumPosition(nextPos, currentSize, viewport);
      commitPosition(clamped);
      setLiveAnnouncement(`Cuprum moved to left ${clamped.x}, top ${clamped.y}.`);
    },
    [
      persistedPos.x,
      persistedPos.y,
      currentSize,
      viewport,
      noteInteraction,
      commitPosition,
      commitResetPosition,
      finishDrag,
      mode,
    ],
  );

  useEffect(() => () => {
    windowFallbackCleanupRef.current?.();
    windowFallbackCleanupRef.current = null;
  }, []);

  // Move focus into dialogue bubble on open/turn without scrolling the dashboard.
  useEffect(() => {
    if (mode !== 'dialogue' || minimized) return;
    const timer = window.setTimeout(
      () => bubbleRef.current?.focus({ preventScroll: true }),
      0,
    );
    return () => {
      window.clearTimeout(timer);
    };
  }, [mode, activeNodeId, runtime.turn, minimized]);

  const anchorRect = useMemo(
    () => ({
      x: persistedPos.x,
      y: persistedPos.y,
      width: currentSize.width,
      height: currentSize.height,
    }),
    [persistedPos.x, persistedPos.y, currentSize.width, currentSize.height],
  );

  const bubblePlacement = useMemo(
    () =>
      computeBubblePlacement(
        anchorRect,
        viewport,
        mode === 'dialogue' ? 'dialogue' : 'idle',
      ),
    [anchorRect, viewport, mode],
  );

  if (!isVisible) return null;

  const dockStyle = {
    '--cuprum-ring': moodMeta.ringColor,
    '--cuprum-core': moodMeta.coreColor,
  } as CSSProperties;

  return (
    <aside
      ref={stageRef}
      className={`cm-cuprum-dock ${minimized ? 'cm-cuprum-dock--minimized' : 'cm-cuprum-dock--expanded'}${isDragging ? ' cm-cuprum-dock--dragging' : ''}`}
      style={dockStyle}
      aria-label="Cuprum companion"
      data-testid="cuprum-companion-stage"
      data-minimized={minimized ? 'true' : 'false'}
      data-dragging={isDragging ? 'true' : 'false'}
      data-pos-x={persistedPos.x}
      data-pos-y={persistedPos.y}
    >
      <p className="sr-only" aria-live="polite" data-testid="cuprum-position-announcement">
        {liveAnnouncement}
      </p>

      {!minimized && (
        <div className="cm-cuprum-bubble-anchor">
          {mode === 'dialogue' ? (
            resolvedNode ? (
              <CuprumSpeechBubble
                ref={bubbleRef}
                key="dialogue"
                mode="dialogue"
                speech={resolvedNode.speech}
                action={resolvedNode.action}
                choices={resolvedNode.choices}
                accent={moodMeta.ringColor}
                coreColor={moodMeta.coreColor}
                placement={bubblePlacement}
                onChoose={handleChoose}
                onClose={closeDialogue}
                onTypingChange={setIsTyping}
              />
            ) : null
          ) : idleLine ? (
            <CuprumSpeechBubble
              key={`idle-${idleLine.id}`}
              mode="idle"
              speech={idleLine.text}
              accent={moodMeta.ringColor}
              coreColor={moodMeta.coreColor}
              placement={bubblePlacement}
              onClose={dismissIdle}
              onTypingChange={setIsTyping}
            />
          ) : null}
        </div>
      )}

      {minimized ? (
        <div className="cm-cuprum-pill" data-testid="cuprum-minimized-pill">
          <button
            ref={dragHandleRef}
            type="button"
            className="cm-cuprum-drag-handle cm-cuprum-drag-handle--pill"
            data-cm-click-sound="off"
            aria-label="Reposition Cuprum pet. Use arrow keys to move."
            aria-roledescription="movable widget"
            onPointerDown={handleHandlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerCancel}
            onLostPointerCapture={handleLostPointerCapture}
            onKeyDown={handleDragHandleKeyDown}
            onDoubleClick={commitResetPosition}
            title="Drag or use arrow keys to move Cuprum (Esc resets)"
          >
            <GripIcon />
          </button>

          <button
            type="button"
            className="cm-cuprum-pill-expand"
            data-cm-click-sound="off"
            onClick={() => {
              if (suppressClickRef.current) {
                suppressClickRef.current = false;
                return;
              }
              handleToggleMinimized();
            }}
            aria-label="Expand Cuprum"
          >
            <span
              className="cm-cuprum-pet-dot"
              style={{ backgroundColor: moodMeta.ringColor }}
              aria-hidden="true"
            />
            <span>Cuprum</span>
          </button>

          <div className="sr-only">
            <span>{moodMeta.title}</span>
          </div>
        </div>
      ) : (
        <div className="cm-cuprum-pet-shell">
          <CuprumSpriteStage
            moodPrefix={moodMeta.prefix}
            moodTitle={moodMeta.title}
            isTyping={isTyping}
            alertEpoch={alertEpoch}
            onClick={handlePetClick}
            onPointerDown={handlePetPointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerCancel}
            onLostPointerCapture={handleLostPointerCapture}
          />

          <div className="cm-cuprum-action-bar">
            <button
              ref={petButtonRef}
              type="button"
              data-cm-click-sound="off"
              onClick={handlePetClick}
              onPointerDown={handlePetPointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerCancel}
              onLostPointerCapture={handleLostPointerCapture}
              className="cm-cuprum-pet-btn"
              aria-expanded={mode === 'dialogue'}
              aria-controls="cuprum-bubble"
              aria-label={mode === 'dialogue' ? 'Close Cuprum conversation' : 'Talk to Cuprum'}
              title={mode === 'dialogue' ? 'Close the conversation' : 'Talk to Cuprum'}
            >
              <span className="cm-cuprum-talk">
                <span
                  className="cm-cuprum-pet-dot"
                  style={{ backgroundColor: moodMeta.ringColor }}
                  aria-hidden="true"
                />
                <span>{mode === 'dialogue' ? 'Close' : 'Talk to Cuprum'}</span>
              </span>
            </button>

            <div className="cm-cuprum-controls">
              <button
                ref={dragHandleRef}
                type="button"
                className="cm-cuprum-drag-handle"
                data-cm-click-sound="off"
                aria-label="Reposition Cuprum pet. Use arrow keys to move."
                aria-roledescription="movable widget"
                onPointerDown={handleHandlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerCancel={handlePointerCancel}
                onLostPointerCapture={handleLostPointerCapture}
                onKeyDown={handleDragHandleKeyDown}
                onDoubleClick={commitResetPosition}
                title="Drag or use arrow keys to move Cuprum (Esc resets)"
              >
                <GripIcon />
              </button>

              <button
                type="button"
                className="cm-cuprum-min-btn"
                data-cm-click-sound="off"
                onClick={handleToggleMinimized}
                aria-label="Minimize Cuprum"
                title="Minimize Cuprum"
              >
                –
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}

function GripIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
      <circle cx="3.5" cy="2.5" r="1" />
      <circle cx="8.5" cy="2.5" r="1" />
      <circle cx="3.5" cy="6" r="1" />
      <circle cx="8.5" cy="6" r="1" />
      <circle cx="3.5" cy="9.5" r="1" />
      <circle cx="8.5" cy="9.5" r="1" />
    </svg>
  );
}
