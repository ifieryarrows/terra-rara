function easeInOutCubic(value: number) {
  return value < .5 ? 4 * value ** 3 : 1 - (-2 * value + 2) ** 3 / 2;
}

/** Scrolls the document through visible intermediate positions and cancels when the user takes control. */
export function scrollToElement(element: HTMLElement, instant = false) {
  const start = window.scrollY;
  const scrollPadding = Number.parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
  const maxScroll = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
  const destination = Math.max(0, Math.min(maxScroll, element.getBoundingClientRect().top + start - scrollPadding));
  const distance = destination - start;

  if (instant || window.matchMedia('(prefers-reduced-motion: reduce)').matches || Math.abs(distance) < 1) {
    window.scrollTo({ top: destination, behavior: 'instant' });
    return () => undefined;
  }

  const duration = Math.max(760, Math.min(1_700, 620 + Math.abs(distance) * .28));
  let frame = 0;
  let startedAt = 0;
  let cancelled = false;
  const cleanup = () => {
    if (frame) window.cancelAnimationFrame(frame);
    window.removeEventListener('wheel', interrupt, true);
    window.removeEventListener('touchstart', interrupt, true);
    window.removeEventListener('pointerdown', interrupt, true);
    window.removeEventListener('keydown', interrupt, true);
  };
  const interrupt = () => {
    cancelled = true;
    cleanup();
  };
  const tick = (now: number) => {
    if (cancelled) return;
    if (!startedAt) startedAt = now;
    const progress = Math.min(1, (now - startedAt) / duration);
    window.scrollTo({ top: start + distance * easeInOutCubic(progress), behavior: 'instant' });
    if (progress < 1) frame = window.requestAnimationFrame(tick);
    else cleanup();
  };

  window.addEventListener('wheel', interrupt, { capture: true, passive: true });
  window.addEventListener('touchstart', interrupt, { capture: true, passive: true });
  window.addEventListener('pointerdown', interrupt, { capture: true, passive: true });
  window.addEventListener('keydown', interrupt, true);
  frame = window.requestAnimationFrame(tick);
  return cleanup;
}
