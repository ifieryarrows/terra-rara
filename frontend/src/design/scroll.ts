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

  try {
    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return () => undefined;
  } catch {
    window.scrollTo({ top: destination, behavior: 'smooth' });
    return () => undefined;
  }
}
