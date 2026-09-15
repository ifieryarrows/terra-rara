export const CINEMATIC_SCROLL_SENSITIVITY = .5;
const CINEMATIC_SCROLL_TARGET_RESET_MS = 450;

/**
 * Splits vertical wheel input into two native-scroll movements while keeping
 * the browser's smooth scroll interpolation for the storytelling timeline.
 */
export function createCinematicWheelDampener() {
  let targetScrollY: number | null = null;
  let resetTimer: number | null = null;
  const resetTarget = () => { targetScrollY = null; resetTimer = null; };
  const scheduleTargetReset = () => {
    if (resetTimer !== null) window.clearTimeout(resetTimer);
    resetTimer = window.setTimeout(resetTarget, CINEMATIC_SCROLL_TARGET_RESET_MS);
  };
  const handleWheel = (event: WheelEvent) => {
    if (!event.cancelable || event.defaultPrevented || event.ctrlKey || event.deltaY === 0 || Math.abs(event.deltaY) < Math.abs(event.deltaX)) return false;
    event.preventDefault();
    const currentScrollY = window.scrollY;
    if (targetScrollY === null || Math.abs(currentScrollY - targetScrollY) < 1) targetScrollY = currentScrollY;
    const maxScrollY = Math.max(0, (document.scrollingElement ?? document.documentElement).scrollHeight - window.innerHeight);
    targetScrollY = Math.max(0, Math.min(maxScrollY, targetScrollY + event.deltaY * CINEMATIC_SCROLL_SENSITIVITY));
    window.scrollTo({ top: targetScrollY, behavior: 'smooth' });
    scheduleTargetReset();
    return true;
  };
  return {
    handleWheel,
    dispose() {
      if (resetTimer !== null) window.clearTimeout(resetTimer);
      targetScrollY = null;
      resetTimer = null;
    },
  };
}
