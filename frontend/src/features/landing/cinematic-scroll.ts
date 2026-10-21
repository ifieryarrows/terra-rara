/**
 * Gesture scene stepper for the pinned cinematic story.
 *
 * One wheel gesture or key press advances at most one narrative rest point, so
 * fast input can no longer skip a scene. Each step uses the browser's native
 * smooth scroll; input that arrives while a step is gliding — or that is part
 * of the same momentum burst — is swallowed, never queued. At the story bounds
 * the stepper releases input back to native scrolling, and zoom gestures,
 * horizontal scrolls, modified keys and focused controls always pass through.
 * There is no rAF loop; the lock ends on `scrollend` with a timeout fallback.
 */

export const CINEMATIC_STEP_LOCK_MS = 900;
const WHEEL_GESTURE_GAP_MS = 260;

/**
 * Reading rest points as story progress (0 = hero rest, 1 = story end).
 *
 * Every scroll-linked stage meant to be read gets its own stop, so one gesture
 * lands on a settled state instead of a mid-transition frame: the market frame
 * assembling (.24 → .30, where the last tile lands), the four news sequence
 * layers (.43 headline, .48 entities, .52 signal read, .57 two reads), the
 * forecast draw and its range band (.66 → .69), the ingot's quiet
 * negative-space stage (.86, formed from .70 and held to .93), and the outro
 * (.97).
 */
export const CINEMATIC_RESTS = [0, .24, .30, .43, .48, .52, .57, .66, .69, .86, .97] as const;

export type CinematicSceneGeometry = {
  /** Story section top in document coordinates. */
  top: number;
  /** Scrollable length: section height minus viewport height. */
  length: number;
};

export type CinematicSceneStepper = {
  /** Returns true when the event was consumed by the story pacing. */
  handleWheel(event: WheelEvent): boolean;
  handleKeyDown(event: KeyboardEvent): boolean;
  dispose(): void;
};

function maxScrollTop(): number {
  const scrollable = document.scrollingElement ?? document.documentElement;
  return Math.max(0, scrollable.scrollHeight - window.innerHeight);
}

const STEP_KEY_DIRECTION: Record<string, 1 | -1 | undefined> = {
  ArrowDown: 1, PageDown: 1, ' ': 1,
  ArrowUp: -1, PageUp: -1,
};

export function createCinematicSceneStepper(options: {
  getGeometry: () => CinematicSceneGeometry | null;
  scrollTo?: (top: number) => void;
  lockTimeoutMs?: number;
}): CinematicSceneStepper {
  const scrollTo = options.scrollTo ?? ((top: number) => window.scrollTo({ top, behavior: 'smooth' }));
  const lockTimeoutMs = options.lockTimeoutMs ?? CINEMATIC_STEP_LOCK_MS;
  let locked = false;
  let releaseTimer = 0;
  let burstOwned = false;
  let lastWheelAt = 0;

  const release = () => {
    locked = false;
    if (releaseTimer) { window.clearTimeout(releaseTimer); releaseTimer = 0; }
  };
  window.addEventListener('scrollend', release);

  const findTarget = (direction: 1 | -1): number | null => {
    const geometry = options.getGeometry();
    if (!geometry) return null;
    const bound = maxScrollTop();
    const current = window.scrollY;
    const positions = CINEMATIC_RESTS.map(progress => Math.min(bound, Math.max(0, geometry.top + progress * geometry.length)));
    if (direction > 0) return positions.find(position => position > current + 1) ?? null;
    for (let index = positions.length - 1; index >= 0; index -= 1) {
      if (positions[index] < current - 1) return positions[index];
    }
    return null;
  };

  const step = (direction: 1 | -1): boolean => {
    const target = findTarget(direction);
    if (target === null) return false;
    locked = true;
    if (releaseTimer) window.clearTimeout(releaseTimer);
    releaseTimer = window.setTimeout(release, lockTimeoutMs);
    scrollTo(target);
    return true;
  };

  const handleWheel = (event: WheelEvent): boolean => {
    if (!event.cancelable || event.defaultPrevented || event.ctrlKey) return false;
    if (event.deltaY === 0 || Math.abs(event.deltaX) > Math.abs(event.deltaY)) return false;
    if (locked) { event.preventDefault(); return true; }
    const direction: 1 | -1 = event.deltaY > 0 ? 1 : -1;
    const now = Date.now();
    if (burstOwned && now - lastWheelAt < WHEEL_GESTURE_GAP_MS && findTarget(direction) !== null) {
      lastWheelAt = now;
      event.preventDefault();
      return true;
    }
    if (!step(direction)) { burstOwned = false; return false; }
    burstOwned = true;
    lastWheelAt = now;
    event.preventDefault();
    return true;
  };

  const handleKeyDown = (event: KeyboardEvent): boolean => {
    if (!event.cancelable || event.defaultPrevented || event.altKey || event.metaKey || event.ctrlKey) return false;
    if (event.shiftKey && event.key.startsWith('Arrow')) return false; // keep native selection shortcuts
    let direction = STEP_KEY_DIRECTION[event.key];
    if (event.key === ' ' && event.shiftKey) direction = -1;
    if (!direction) return false;
    const focused = document.activeElement;
    if (focused instanceof HTMLElement && (focused.isContentEditable || /^(INPUT|TEXTAREA|SELECT|BUTTON|A|SUMMARY)$/.test(focused.tagName))) return false;
    if ((locked || event.repeat) && findTarget(direction) !== null) {
      event.preventDefault();
      return true;
    }
    if (!step(direction)) return false;
    event.preventDefault();
    return true;
  };

  return {
    handleWheel,
    handleKeyDown,
    dispose() {
      window.removeEventListener('scrollend', release);
      if (releaseTimer) window.clearTimeout(releaseTimer);
      releaseTimer = 0;
      locked = false;
    },
  };
}
