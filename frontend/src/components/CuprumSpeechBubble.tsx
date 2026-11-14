import {
  forwardRef,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
} from 'react';
import clsx from 'clsx';
import type { CuprumChoice } from '../utils/cuprumDialogue';
import { playCuprumSpeakBlip, playCuprumTap } from '../utils/cosmicVision';

const useIsomorphicLayoutEffect =
  typeof window !== 'undefined' ? useLayoutEffect : useEffect;

const TYPE_INTERVAL_MS = 17;
const IDLE_BASE_MS = 4600;
const IDLE_PER_CHAR_MS = 34;
const IDLE_MAX_MS = 9000;

export type BubbleVerticalPlacement = 'above' | 'below';
export type BubbleHorizontalPlacement = 'left' | 'right';

export interface BubbleAnchorRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface BubblePlacement {
  vertical: BubbleVerticalPlacement;
  horizontal: BubbleHorizontalPlacement;
  offsetX: number;
  width: number;
  maxHeight: number;
  tailX: number;
}

const DEFAULT_BUBBLE_MARGIN = 12;
const DEFAULT_BUBBLE_GAP = 12;

export function computeBubblePlacement(
  anchorRect: BubbleAnchorRect,
  viewport: { width: number; height: number },
  mode: 'idle' | 'dialogue',
  bubbleSize?: { width?: number; height?: number },
  margin: number = DEFAULT_BUBBLE_MARGIN,
  gap: number = DEFAULT_BUBBLE_GAP,
): BubblePlacement {
  const vpW = viewport.width > 0 ? viewport.width : 1280;
  const vpH = viewport.height > 0 ? viewport.height : 800;

  const estWidth = bubbleSize?.width && bubbleSize.width > 0 ? bubbleSize.width : mode === 'idle' ? 280 : 336;
  const estHeight = bubbleSize?.height && bubbleSize.height > 0 ? bubbleSize.height : mode === 'idle' ? 96 : 290;

  const effWidth = Math.round(Math.min(estWidth, Math.max(200, vpW - margin * 2)));
  const spaceAbove = anchorRect.y - margin - gap;
  const spaceBelow = vpH - (anchorRect.y + anchorRect.height) - margin - gap;

  let vertical: BubbleVerticalPlacement = 'above';
  if (spaceAbove < estHeight && spaceBelow > spaceAbove) {
    vertical = 'below';
  } else if (spaceAbove < 140 && spaceBelow >= spaceAbove) {
    vertical = 'below';
  }

  const availableVertical = vertical === 'above' ? spaceAbove : spaceBelow;
  const maxHeight = Math.round(Math.max(120, Math.min(vpH - margin * 2, availableVertical)));

  const anchorCenterX = anchorRect.x + anchorRect.width / 2;
  const horizontal: BubbleHorizontalPlacement = anchorCenterX >= vpW * 0.5 ? 'left' : 'right';

  const idealLeft =
    horizontal === 'left'
      ? anchorRect.x + anchorRect.width - effWidth
      : anchorRect.x;

  const maxLeft = Math.max(margin, vpW - effWidth - margin);
  const clampedLeft = Math.round(Math.min(Math.max(idealLeft, margin), maxLeft));
  const offsetX = Math.round(clampedLeft - anchorRect.x);

  const rawTailX = anchorCenterX - clampedLeft;
  const tailX = Math.round(Math.min(Math.max(rawTailX, 24), effWidth - 24));

  return {
    vertical,
    horizontal,
    offsetX,
    width: effWidth,
    maxHeight,
    tailX,
  };
}

export interface CuprumSpeechBubbleProps {
  mode: 'idle' | 'dialogue';
  speech: string;
  action?: string | null;
  choices?: CuprumChoice[];
  accent: string;
  coreColor: string;
  placement?: BubblePlacement;
  onChoose?: (choice: CuprumChoice) => void;
  onClose: () => void;
  onTypingChange?: (typing: boolean) => void;
}

export type SpeechBubblePhase = 'typing' | 'revealed' | 'paused';

export interface SpeechBubbleMachineState {
  phase: SpeechBubblePhase;
  visibleCount: number;
  paused: boolean;
}

export type SpeechBubbleEvent =
  | { type: 'reset'; length: number; reducedMotion: boolean }
  | { type: 'tick'; length: number }
  | { type: 'reveal-all'; length: number }
  | { type: 'set-paused'; paused: boolean };

export function createInitialSpeechBubbleState(
  length: number,
  reducedMotion: boolean,
): SpeechBubbleMachineState {
  const skip = reducedMotion || length === 0;
  return {
    phase: skip ? 'revealed' : 'typing',
    visibleCount: skip ? length : 0,
    paused: false,
  };
}

export function advanceSpeechBubble(
  state: SpeechBubbleMachineState,
  event: SpeechBubbleEvent,
): SpeechBubbleMachineState {
  switch (event.type) {
    case 'reset': {
      const skip = event.reducedMotion || event.length === 0;
      return {
        phase: skip ? (state.paused ? 'paused' : 'revealed') : 'typing',
        visibleCount: skip ? event.length : 0,
        paused: state.paused,
      };
    }
    case 'tick': {
      const nextCount = Math.min(event.length, state.visibleCount + 1);
      const finished = nextCount >= event.length;
      return {
        ...state,
        visibleCount: nextCount,
        phase: finished ? (state.paused ? 'paused' : 'revealed') : 'typing',
      };
    }
    case 'reveal-all':
      return {
        ...state,
        visibleCount: event.length,
        phase: state.paused ? 'paused' : 'revealed',
      };
    case 'set-paused':
      return {
        ...state,
        paused: event.paused,
        phase: state.phase === 'typing' ? 'typing' : event.paused ? 'paused' : 'revealed',
      };
  }
}

export const CuprumSpeechBubble = forwardRef<HTMLDivElement, CuprumSpeechBubbleProps>(
  function CuprumSpeechBubble(
    {
      mode,
      speech,
      action,
      choices,
      accent,
      coreColor,
      placement,
      onChoose,
      onClose,
      onTypingChange,
    },
    ref,
  ) {
    const rawSpeech = speech.trim();
    const stageMatch = rawSpeech.match(/^\s*\*\((.*?)\)\*\s*|^\s*\*(.*?)\*\s*/);
    const text = stageMatch ? rawSpeech.slice(stageMatch[0].length).trim() : rawSpeech;
    const stageAction = stageMatch ? (stageMatch[1] || stageMatch[2]).trim() : null;
    const resolvedAction = action
      ? stageAction
        ? `${action.replace(/\.?$/, ';')} ${stageAction}.`
        : action
      : stageAction
        ? `${stageAction}.`
        : null;

    const reducedMotion = usePrefersReducedMotion();
    const [bubbleState, setBubbleState] = useState<{
      phase: SpeechBubblePhase;
      paused: boolean;
    }>(() => ({
      phase: getPrefersReducedMotion() || text.length === 0 ? 'revealed' : 'typing',
      paused: false,
    }));
    const { paused } = bubbleState;
    const typing = bubbleState.phase === 'typing';
    const typingCallbackRef = useRef(onTypingChange);
    const speechTextRef = useRef<HTMLSpanElement | null>(null);
    const typingRafRef = useRef<number>(0);

    useEffect(() => {
      typingCallbackRef.current = onTypingChange;
    }, [onTypingChange]);

    useEffect(() => {
      typingCallbackRef.current?.(typing);
    }, [typing]);

    useEffect(
      () => () => {
        typingCallbackRef.current?.(false);
      },
      [],
    );

    useIsomorphicLayoutEffect(() => {
      if (typingRafRef.current) {
        window.cancelAnimationFrame(typingRafRef.current);
        typingRafRef.current = 0;
      }

      if (reducedMotion || text.length === 0) {
        if (speechTextRef.current) {
          speechTextRef.current.textContent = text;
        }
        setBubbleState((prev) => {
          const nextPhase: SpeechBubblePhase = prev.paused ? 'paused' : 'revealed';
          return prev.phase === nextPhase ? prev : { ...prev, phase: nextPhase };
        });
        return;
      }

      if (speechTextRef.current) {
        speechTextRef.current.textContent = '';
      }
      setBubbleState((prev) =>
        prev.phase === 'typing' ? prev : { ...prev, phase: 'typing' },
      );

      let startTime: number | null = null;
      let revealedCount = 0;
      let spokenCount = 0;
      let syllableStep = 0;
      let cancelled = false;

      const step = (now: number) => {
        if (cancelled) return;
        if (startTime === null) {
          startTime = now;
        }
        const elapsed = Math.max(0, now - startTime);
        const targetCount = Math.min(text.length, Math.floor(elapsed / TYPE_INTERVAL_MS) + 1);

        if (targetCount > revealedCount) {
          if (mode === 'dialogue') {
            for (let i = revealedCount; i < targetCount; i += 1) {
              const ch = text.charAt(i);
              if (/\S/.test(ch)) {
                if (spokenCount % 3 === 0) {
                  void playCuprumSpeakBlip({ pitchStep: syllableStep });
                  syllableStep += 1;
                }
                spokenCount += 1;
              }
            }
          }
          revealedCount = targetCount;
          if (speechTextRef.current) {
            speechTextRef.current.textContent = text.slice(0, revealedCount);
          }
        }

        if (revealedCount >= text.length) {
          typingRafRef.current = 0;
          setBubbleState((prev) => {
            const nextPhase: SpeechBubblePhase = prev.paused ? 'paused' : 'revealed';
            return prev.phase === nextPhase ? prev : { ...prev, phase: nextPhase };
          });
          return;
        }

        typingRafRef.current = window.requestAnimationFrame(step);
      };

      typingRafRef.current = window.requestAnimationFrame(step);

      return () => {
        cancelled = true;
        if (typingRafRef.current) {
          window.cancelAnimationFrame(typingRafRef.current);
          typingRafRef.current = 0;
        }
      };
    }, [text, reducedMotion, mode]);

    useEffect(() => {
      if (mode !== 'idle' || typing || paused) return;
      const duration = Math.min(IDLE_MAX_MS, IDLE_BASE_MS + text.length * IDLE_PER_CHAR_MS);
      const timer = window.setTimeout(onClose, duration);
      return () => {
        window.clearTimeout(timer);
      };
    }, [mode, typing, paused, text.length, onClose]);

    const revealAll = useCallback(() => {
      if (typingRafRef.current) {
        window.cancelAnimationFrame(typingRafRef.current);
        typingRafRef.current = 0;
      }
      if (speechTextRef.current) {
        speechTextRef.current.textContent = text;
      }
      setBubbleState((prev) => {
        const nextPhase: SpeechBubblePhase = prev.paused ? 'paused' : 'revealed';
        return prev.phase === nextPhase ? prev : { ...prev, phase: nextPhase };
      });
    }, [text]);

    const setPaused = useCallback((nextPaused: boolean) => {
      setBubbleState((prev) => {
        const nextPhase: SpeechBubblePhase =
          prev.phase === 'typing' ? 'typing' : nextPaused ? 'paused' : 'revealed';
        if (prev.paused === nextPaused && prev.phase === nextPhase) return prev;
        return { paused: nextPaused, phase: nextPhase };
      });
    }, []);

    const handleClick = useCallback(() => {
      if (mode === 'dialogue') {
        // In dialogue mode, clicking background/speech area does NOT skip typewriter reveal
        return;
      }
      if (typing) {
        revealAll();
        return;
      }
      if (mode === 'idle') onClose();
    }, [mode, typing, revealAll, onClose]);

    const handleKeyDown = useCallback(
      (event: KeyboardEvent<HTMLDivElement>) => {
        if (event.key === 'Escape') {
          event.preventDefault();
          onClose();
          return;
        }
        if (mode !== 'dialogue') return;
        const num = Number(event.key);
        if (!Number.isInteger(num) || num < 1 || num > (choices?.length ?? 0)) return;
        event.preventDefault();
        const choice = choices?.[num - 1];
        if (choice) onChoose?.(choice);
      },
      [mode, choices, onChoose, onClose],
    );

    const spokenText = resolvedAction ? `${resolvedAction} ${text}` : text;
    const vertical = placement?.vertical ?? 'above';
    const horizontal = placement?.horizontal ?? 'left';

    const bubbleStyle: CSSProperties = {
      '--cuprum-accent': accent,
      '--cuprum-core': coreColor,
      ...(placement
        ? {
            '--cuprum-bubble-offset-x': `${placement.offsetX}px`,
            '--cuprum-bubble-width': `${placement.width}px`,
            '--cuprum-bubble-max-h': `${placement.maxHeight}px`,
            '--cuprum-tail-x': `${placement.tailX}px`,
          }
        : {}),
    } as CSSProperties;

    return (
      <div
        ref={ref}
        id="cuprum-bubble"
        role="region"
        aria-label={mode === 'dialogue' ? 'Cuprum conversation' : 'Cuprum is speaking'}
        tabIndex={-1}
        className={clsx(
          'cm-cuprum-bubble',
          mode === 'dialogue' ? 'cm-cuprum-bubble--dialogue' : 'cm-cuprum-bubble--idle',
          vertical === 'below' ? 'cm-cuprum-bubble--below' : 'cm-cuprum-bubble--above',
          horizontal === 'right' ? 'cm-cuprum-bubble--open-right' : 'cm-cuprum-bubble--open-left',
        )}
        style={bubbleStyle}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
        onFocus={() => setPaused(true)}
        onBlur={() => setPaused(false)}
        data-placement-vertical={vertical}
        data-placement-horizontal={horizontal}
        data-testid={mode === 'dialogue' ? 'cuprum-dialogue-bubble' : 'cuprum-idle-bubble'}
      >
        <span className="cm-cuprum-tail" aria-hidden="true">
          <svg viewBox="0 0 24 12" width="24" height="12" focusable="false">
            <path
              d="M1 0 V5 H7 V10 H12 V12 H14 V10 H17 V5 H23 V0"
              fill="var(--cuprum-bubble-bg)"
              stroke="var(--cuprum-accent)"
              strokeOpacity="0.42"
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
            />
          </svg>
        </span>

        <div className="cm-cuprum-bubble-inner">
          {mode === 'dialogue' && (
            <button
              type="button"
              className="cm-cuprum-dismiss"
              data-cm-click-sound="off"
              aria-label="Close conversation"
              onClick={(event) => {
                event.stopPropagation();
                void playCuprumTap({ volume: 0.36 }).catch(() => {});
                onClose();
              }}
            >
              ×
            </button>
          )}

          <div className="cm-cuprum-copy" key={text}>
            {resolvedAction && <p className="cm-cuprum-action">{resolvedAction}</p>}

            <p className="cm-cuprum-speech" aria-hidden="true">
              <span ref={speechTextRef} />
              {typing && <span className="cm-cuprum-caret" aria-hidden="true" />}
            </p>
          </div>
          <p className="sr-only" aria-live="polite" data-testid="cuprum-spoken-text">
            {spokenText}
          </p>

          {mode === 'dialogue' && choices && choices.length > 0 && (
            <div className="cm-cuprum-choices" role="group" aria-label="Your reply">
              {choices.map((choice, index) => (
                <button
                  key={choice.id}
                  type="button"
                  className="cm-cuprum-choice"
                  data-cm-click-sound="off"
                  style={
                    reducedMotion ? undefined : { animationDelay: `${60 + index * 55}ms` }
                  }
                  onClick={(event) => {
                    event.stopPropagation();
                    onChoose?.(choice);
                  }}
                  aria-keyshortcuts={`${index + 1}`}
                >
                  <span className="cm-cuprum-choice-index" aria-hidden="true">
                    {index + 1}
                  </span>
                  <span>{choice.label}</span>
                </button>
              ))}
            </div>
          )}

          {mode === 'idle' && (
            <span className="cm-cuprum-whisper" aria-hidden="true">
              ···
            </span>
          )}
        </div>
      </div>
    );
  },
);

function getPrefersReducedMotion(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(getPrefersReducedMotion);

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = () => setReduced(media.matches);
    media.addEventListener?.('change', onChange);
    return () => {
      media.removeEventListener?.('change', onChange);
    };
  }, []);

  return reduced;
}
