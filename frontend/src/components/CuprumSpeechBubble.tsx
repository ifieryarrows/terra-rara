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
  turnKey?: string;
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
      turnKey,
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
    const animationKey = turnKey ?? text;
    const reducedMotion = usePrefersReducedMotion();
    const [visibleCount, setVisibleCount] = useState(() =>
      getPrefersReducedMotion() ? text.length : 0,
    );
    const [typing, setTyping] = useState(() => !getPrefersReducedMotion() && text.length > 0);
    const [animatedKey, setAnimatedKey] = useState(animationKey);
    const [paused, setPaused] = useState(false);
    const typingCallbackRef = useRef(onTypingChange);
    const typingIntervalRef = useRef<number | null>(null);
    const choiceHandledRef = useRef(false);
    const revealedKeyRef = useRef<string | null>(null);
    const typingForText = typing || (animatedKey !== animationKey && text.length > 0);
    const visibleCountForText = animatedKey === animationKey ? visibleCount : 0;

    useEffect(() => {
      typingCallbackRef.current = onTypingChange;
    }, [onTypingChange]);

    useEffect(() => {
      typingCallbackRef.current?.(typingForText);
    }, [typingForText]);

    useEffect(
      () => () => {
        typingCallbackRef.current?.(false);
      },
      [],
    );

    useEffect(() => {
      setAnimatedKey(animationKey);
      choiceHandledRef.current = false;
      if (reducedMotion || text.length === 0 || revealedKeyRef.current === animationKey) {
        setVisibleCount(text.length);
        setTyping(false);
        return;
      }

      setVisibleCount(0);
      setTyping(true);
      let index = 0;
      typingIntervalRef.current = window.setInterval(() => {
        index += 1;
        setVisibleCount(index);
        if (index >= text.length) {
          if (typingIntervalRef.current !== null) {
            window.clearInterval(typingIntervalRef.current);
            typingIntervalRef.current = null;
          }
          setTyping(false);
        }
      }, TYPE_INTERVAL_MS);

      return () => {
        if (typingIntervalRef.current !== null) {
          window.clearInterval(typingIntervalRef.current);
          typingIntervalRef.current = null;
        }
      };
    }, [text, reducedMotion, animationKey]);

    useEffect(() => {
      if (mode !== 'idle' || typing || paused) return;
      const duration = Math.min(IDLE_MAX_MS, IDLE_BASE_MS + text.length * IDLE_PER_CHAR_MS);
      const timer = window.setTimeout(onClose, duration);
      return () => {
        window.clearTimeout(timer);
      };
    }, [mode, typingForText, paused, text.length, onClose]);

    const revealAll = useCallback(() => {
      if (typingIntervalRef.current !== null) {
        window.clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }
      revealedKeyRef.current = animationKey;
      setAnimatedKey(animationKey);
      setVisibleCount(text.length);
      setTyping(false);
    }, [text, animationKey]);

    const handleClick = useCallback(() => {
      if (typingForText) {
        revealAll();
        return;
      }
      if (mode === 'idle') onClose();
    }, [typingForText, revealAll, mode, onClose]);

    const choose = useCallback(
      (choice: CuprumChoice) => {
        if (typingForText || choiceHandledRef.current) return;
        choiceHandledRef.current = true;
        onChoose?.(choice);
      },
      [typingForText, onChoose],
    );

    const handleKeyDown = useCallback(
      (event: KeyboardEvent<HTMLDivElement>) => {
        if (event.key === 'Escape') {
          event.preventDefault();
          onClose();
          return;
        }
        if (mode !== 'dialogue' || !/^[1-3]$/.test(event.key)) return;
        event.preventDefault();
        if (typingForText) {
          revealAll();
          return;
        }
        const choice = choices?.[Number(event.key) - 1];
        if (choice) choose(choice);
      },
      [mode, typingForText, revealAll, choices, choose, onClose],
    );

    const visibleText = text.slice(0, visibleCountForText);
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
              {typingForText && <span className="cm-cuprum-caret" aria-hidden="true" />}
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
                  disabled={typingForText || choiceHandledRef.current}
                  title={typingForText ? 'Wait until Cuprum finishes speaking' : undefined}
                  style={
                    reducedMotion ? undefined : { animationDelay: `${60 + index * 55}ms` }
                  }
                  onClick={(event) => {
                    event.stopPropagation();
                    choose(choice);
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
