import {
  forwardRef,
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
} from 'react';
import clsx from 'clsx';
import type { CuprumChoice } from '../utils/cuprumDialogue';

const TYPE_INTERVAL_MS = 17;
const IDLE_BASE_MS = 4600;
const IDLE_PER_CHAR_MS = 34;
const IDLE_MAX_MS = 9000;

export interface CuprumSpeechBubbleProps {
  mode: 'idle' | 'dialogue';
  speech: string;
  action?: string | null;
  choices?: CuprumChoice[];
  accent: string;
  coreColor: string;
  onChoose?: (choice: CuprumChoice) => void;
  onClose: () => void;
  onTypingChange?: (typing: boolean) => void;
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
      onChoose,
      onClose,
      onTypingChange,
    },
    ref,
  ) {
    const text = speech.trim();
    const reducedMotion = usePrefersReducedMotion();
    const [visibleCount, setVisibleCount] = useState(() =>
      getPrefersReducedMotion() ? text.length : 0,
    );
    const [typing, setTyping] = useState(() => !getPrefersReducedMotion() && text.length > 0);
    const [paused, setPaused] = useState(false);
    const typingCallbackRef = useRef(onTypingChange);

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

    useEffect(() => {
      if (reducedMotion || text.length === 0) {
        setVisibleCount(text.length);
        setTyping(false);
        return;
      }

      setVisibleCount(0);
      setTyping(true);
      let index = 0;
      const interval = window.setInterval(() => {
        index += 1;
        setVisibleCount(index);
        if (index >= text.length) {
          window.clearInterval(interval);
          setTyping(false);
        }
      }, TYPE_INTERVAL_MS);

      return () => {
        window.clearInterval(interval);
      };
    }, [text, reducedMotion]);

    useEffect(() => {
      if (mode !== 'idle' || typing || paused) return;
      const duration = Math.min(IDLE_MAX_MS, IDLE_BASE_MS + text.length * IDLE_PER_CHAR_MS);
      const timer = window.setTimeout(onClose, duration);
      return () => {
        window.clearTimeout(timer);
      };
    }, [mode, typing, paused, text.length, onClose]);

    const revealAll = useCallback(() => {
      setVisibleCount(text.length);
      setTyping(false);
    }, [text.length]);

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

    const visibleText = text.slice(0, visibleCount);
    const spokenText = action ? `${action} ${text}` : text;

    return (
      <div
        ref={ref}
        id="cuprum-bubble"
        role="region"
        aria-label={mode === 'dialogue' ? 'Cuprum conversation' : 'Cuprum is speaking'}
        tabIndex={-1}
        className={clsx(
          'cm-cuprum-bubble relative w-full',
          mode === 'dialogue' ? 'cm-cuprum-bubble--dialogue' : 'cm-cuprum-bubble--idle',
        )}
        style={
          {
            '--cuprum-accent': accent,
            '--cuprum-core': coreColor,
          } as CSSProperties
        }
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
        onFocus={() => setPaused(true)}
        onBlur={() => setPaused(false)}
        data-testid={mode === 'dialogue' ? 'cuprum-dialogue-bubble' : 'cuprum-idle-bubble'}
      >
        <span className="cm-cuprum-tail" aria-hidden="true">
          <svg viewBox="0 0 16 24" width="16" height="24" focusable="false">
            <path
              d="M16 1 H8 V7 H2 V13 H8 V23 H16"
              fill="var(--cuprum-bubble-bg)"
              stroke="var(--cuprum-accent)"
              strokeOpacity="0.38"
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
              aria-label="Close conversation"
              onClick={(event) => {
                event.stopPropagation();
                onClose();
              }}
            >
              ×
            </button>
          )}

          <div className="cm-cuprum-copy" key={text}>
            {action && <p className="cm-cuprum-action">{action}</p>}

            <p className="cm-cuprum-speech" aria-hidden="true">
              {visibleText}
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
