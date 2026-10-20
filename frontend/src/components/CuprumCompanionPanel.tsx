import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
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
  type CuprumEnergy,
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
import { CuprumSpeechBubble } from './CuprumSpeechBubble';
import './CuprumCompanionPanel.css';

interface Props {
  commentary: CommentaryResponse | null;
  explain: ExplainResponse | null;
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
    ringColor: '#f43f5e',
    coreColor: '#fb923c',
  },
  patina_bear: {
    title: 'Patina Resonance',
    subtitle: 'Green around the edges, soft in the middle.',
    prefix: 'cuprum-bear',
    ringColor: '#10b981',
    coreColor: '#2dd4bf',
  },
  tempered_neutral: {
    title: 'Harmonic Core',
    subtitle: 'Listening to the quiet between prices.',
    prefix: 'cuprum-neutral',
    ringColor: '#e08b56',
    coreColor: '#fbbf24',
  },
};

const ENERGY_META: Record<CuprumEnergy, string> = {
  calm: 'quiet embers',
  charged: 'humming',
  volatile: 'sparking',
};

const IDLE_FIRST_DELAY_MS = 6500;
const IDLE_DELAY_MIN_MS = 22000;
const IDLE_DELAY_RANGE_MS = 16000;

function initialVisibility(): boolean {
  return typeof document === 'undefined' || document.visibilityState !== 'hidden';
}

export function CuprumCompanionPanel({
  commentary,
  explain,
  tftAnalysis = null,
  loading = false,
  error,
}: Props) {
  const [mode, setMode] = useState<'idle' | 'dialogue'>('idle');
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<CuprumRuntime>(() => createCuprumRuntime());
  const [idleLine, setIdleLine] = useState<CuprumIdleLine | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [inView, setInView] = useState(true);
  const [visible, setVisible] = useState(initialVisibility);
  const [interactionEpoch, setInteractionEpoch] = useState(0);
  const [frame, setFrame] = useState<CuprumSpriteFrame>(() =>
    cuprumSpriteFrame(INITIAL_CUPRUM_SPRITE_STATE),
  );

  const stageRef = useRef<HTMLDivElement>(null);
  const petButtonRef = useRef<HTMLButtonElement>(null);
  const bubbleRef = useRef<HTMLDivElement>(null);
  const dispatchSpriteRef = useRef<(event: CuprumSpriteEvent) => void>(() => {});
  const recentIdleIdsRef = useRef<string[]>([]);
  const hasShownIdleRef = useRef(false);
  const freshDataRef = useRef(false);
  const hasMountedRef = useRef(false);
  const generatedAt = commentary?.generated_at ?? null;
  const lastGeneratedAtRef = useRef<string | null>(generatedAt);

  const ctx = useMemo(
    () => buildCuprumContext(commentary, explain, tftAnalysis, loading, error),
    [commentary, explain, tftAnalysis, loading, error],
  );
  const moodMeta = MOOD_META[ctx.mood];
  const energyLabel = ENERGY_META[ctx.energy];

  const noteInteraction = useCallback(() => {
    setInteractionEpoch((value) => value + 1);
  }, []);

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

  // Pause ambient chatter when the card is off-screen or the tab is hidden.
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
    const onChange = () => setVisible(document.visibilityState !== 'hidden');
    document.addEventListener('visibilitychange', onChange);
    return () => document.removeEventListener('visibilitychange', onChange);
  }, []);

  // Ambient speech bubbles: first after a short welcome, then a long, irregular
  // cadence. Any interaction resets the clock, and dialogue always wins.
  useEffect(() => {
    if (mode !== 'idle' || idleLine || !inView || !visible) return;
    const delay = hasShownIdleRef.current
      ? IDLE_DELAY_MIN_MS + Math.random() * IDLE_DELAY_RANGE_MS
      : IDLE_FIRST_DELAY_MS;
    const timer = window.setTimeout(() => {
      const line = pickIdleLine(ctx, runtime, recentIdleIdsRef.current);
      recentIdleIdsRef.current = [...recentIdleIdsRef.current, line.id].slice(-4);
      hasShownIdleRef.current = true;
      setIdleLine(line);
    }, delay);
    return () => {
      window.clearTimeout(timer);
    };
  }, [mode, idleLine, inView, visible, interactionEpoch, ctx, runtime]);

  const dismissIdle = useCallback(() => {
    setIdleLine(null);
    setIsTyping(false);
  }, []);

  const closeConversation = useCallback(
    (restoreFocus = false) => {
      noteInteraction();
      setMode('idle');
      setActiveNodeId(null);
      setIsTyping(false);
      if (restoreFocus) {
        window.setTimeout(() => petButtonRef.current?.focus(), 0);
      }
    },
    [noteInteraction],
  );

  const closeDialogue = useCallback(() => {
    closeConversation(true);
  }, [closeConversation]);

  const openConversation = useCallback(() => {
    setIdleLine(null);
    setActiveNodeId(ROOT_NODE_ID);
    setRuntime((previous) => ({ ...previous, freshData: freshDataRef.current }));
    freshDataRef.current = false;
    setMode('dialogue');
    dispatchSpriteRef.current('alert-start');
  }, []);

  const handlePetClick = useCallback(() => {
    noteInteraction();
    if (mode === 'dialogue') {
      closeConversation(false);
      return;
    }
    openConversation();
  }, [mode, noteInteraction, closeConversation, openConversation]);

  const handleChoose = useCallback(
    (choice: CuprumChoice) => {
      noteInteraction();
      if (choice.close) {
        closeConversation(false);
        return;
      }
      setRuntime((previous) => applyCuprumChoice(previous, choice));
      if (choice.next) setActiveNodeId(choice.next);
      setIsTyping(false);
    },
    [noteInteraction, closeConversation],
  );

  const resolvedNode = useMemo(
    () => (activeNodeId ? resolveCuprumNode(activeNodeId, ctx, runtime) : null),
    [activeNodeId, ctx, runtime],
  );

  // Esc closes the conversation; a click outside the stage dismisses it too.
  useEffect(() => {
    if (mode !== 'dialogue') return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      closeConversation(true);
    };
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (stageRef.current?.contains(target)) return;
      closeConversation(false);
    };
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('pointerdown', onPointerDown, true);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('pointerdown', onPointerDown, true);
    };
  }, [mode, closeConversation]);

  // Move focus into the new bubble so screen readers get the whole turn and
  // the 1/2/3 shortcuts work without hunting for the menu.
  useEffect(() => {
    if (mode !== 'dialogue') return;
    const timer = window.setTimeout(() => bubbleRef.current?.focus(), 0);
    return () => {
      window.clearTimeout(timer);
    };
  }, [mode, activeNodeId, runtime.turn]);

  const currentFrame: CuprumSpriteFrame | 'talk' = isTyping ? 'talk' : frame;
  const currentWebp = `/assets/cuprum/${moodMeta.prefix}-${currentFrame}.webp?v=retro-outline-v4`;
  const currentPng = `/assets/cuprum/${moodMeta.prefix}-${currentFrame}.png?v=retro-outline-v4`;

  return (
    <div
      ref={stageRef}
      className="w-full relative transition-colors"
      style={{ '--cuprum-ring': moodMeta.ringColor } as CSSProperties}
      data-testid="cuprum-companion-stage"
      onPointerDown={noteInteraction}
    >
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          opacity: 0.14,
          filter: 'blur(26px)',
          transition: 'all 700ms ease',
          background: `radial-gradient(circle at 20% 48%, ${moodMeta.ringColor} 0%, transparent 62%)`,
        }}
        aria-hidden="true"
      />

      <div className="relative z-10 cm-cuprum-stage">
        <div className="cm-cuprum-pet" style={{ minWidth: 170 }}>
          <button
            ref={petButtonRef}
            type="button"
            onClick={handlePetClick}
            className="relative flex flex-col items-center justify-center rounded-xl p-3 focus:outline-none"
            style={{ cursor: 'pointer', transition: 'transform 200ms ease' }}
            aria-expanded={mode === 'dialogue'}
            aria-controls="cuprum-bubble"
            aria-label={mode === 'dialogue' ? 'Close Cuprum conversation' : 'Talk to Cuprum'}
            title={mode === 'dialogue' ? 'Close the conversation' : 'Talk to Cuprum'}
          >
            <svg
              viewBox="0 0 200 200"
              className="pointer-events-none absolute inset-0 h-full w-full"
              style={{ animation: 'spin 40s linear infinite', opacity: 0.55 }}
              aria-hidden="true"
            >
              <circle
                cx="100"
                cy="100"
                r="90"
                fill="none"
                stroke={moodMeta.ringColor}
                strokeOpacity="0.25"
                strokeWidth="1.2"
                strokeDasharray="6 8"
              />
              <circle
                cx="100"
                cy="100"
                r="78"
                fill="none"
                stroke={moodMeta.coreColor}
                strokeOpacity="0.3"
                strokeWidth="1.5"
                strokeDasharray="16 12"
              />
            </svg>

            <div className="relative flex items-center justify-center" style={{ width: 140, height: 140 }}>
              <picture>
                <source srcSet={currentWebp} type="image/webp" />
                <img
                  src={currentPng}
                  alt={`Cuprum — ${moodMeta.title}`}
                  className="h-full w-full select-none"
                  style={{
                    imageRendering: 'pixelated',
                    objectFit: 'contain',
                    transition: 'opacity 150ms ease',
                    filter: `drop-shadow(0 0 16px ${moodMeta.ringColor}55)`,
                  }}
                  loading="lazy"
                />
              </picture>
            </div>

            <span className="cm-cuprum-talk mt-1 inline-flex items-center gap-1.5 rounded-full bg-white/5 px-2.5 py-1 text-[11px] text-slate-300">
              <span
                className="cm-cuprum-pet-dot"
                style={{
                  backgroundColor: moodMeta.ringColor,
                  animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                }}
                aria-hidden="true"
              />
              {mode === 'dialogue' ? 'Close conversation' : 'Talk to Cuprum'}
            </span>
          </button>

          <div className="cm-cuprum-identity" title={moodMeta.subtitle}>
            <p className="text-sm font-medium text-slate-100">{moodMeta.title}</p>
            <p className="text-xs text-slate-400">{energyLabel}</p>
          </div>
        </div>

        <div className="min-w-0 flex-1">
          <div className="cm-cuprum-bubble-slot">
            {mode === 'dialogue' && resolvedNode ? (
              <CuprumSpeechBubble
                ref={bubbleRef}
                key={`dialogue-${runtime.turn}-${resolvedNode.id}`}
                mode="dialogue"
                speech={resolvedNode.speech}
                action={resolvedNode.action}
                choices={resolvedNode.choices}
                accent={moodMeta.ringColor}
                coreColor={moodMeta.coreColor}
                onChoose={handleChoose}
                onClose={closeDialogue}
                onTypingChange={setIsTyping}
              />
            ) : idleLine ? (
              <CuprumSpeechBubble
                key={`idle-${idleLine.id}`}
                mode="idle"
                speech={idleLine.text}
                accent={moodMeta.ringColor}
                coreColor={moodMeta.coreColor}
                onClose={dismissIdle}
                onTypingChange={setIsTyping}
              />
            ) : (
              <div className="cm-cuprum-idle-wire" aria-hidden="true">
                <span className="cm-cuprum-idle-wire-dot" />
              </div>
            )}
          </div>

          <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
            Cuprum is a companion, not a financial advisor.
          </p>
        </div>
      </div>
    </div>
  );
}
