import { memo, useEffect, useRef, useState } from 'react';
import { logoUrl, normalizeLogoTicker } from './heatmap-utils';
import { hasFailedLogo, hasLoadedLogo, markLogoFailed, markLogoLoaded } from './logo-cache';

interface Props {
  ticker: string;
  label?: string;
  size?: number;
  className?: string;
  defer?: boolean;
}

let cachedObserverCtor: typeof IntersectionObserver | null = null;
let sharedLogoObserver: IntersectionObserver | null = null;
const logoObserverCallbacks = new WeakMap<Element, () => void>();

function observeLogoVisibility(element: Element, onVisible: () => void): () => void {
  if (typeof IntersectionObserver === 'undefined') {
    onVisible();
    return () => {};
  }
  if (cachedObserverCtor !== IntersectionObserver || !sharedLogoObserver) {
    cachedObserverCtor = IntersectionObserver;
    sharedLogoObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const callback = logoObserverCallbacks.get(entry.target);
          if (callback) {
            logoObserverCallbacks.delete(entry.target);
            sharedLogoObserver?.unobserve(entry.target);
            callback();
          }
        }
      },
      { rootMargin: '48px' },
    );
  }
  const observer = sharedLogoObserver;
  logoObserverCallbacks.set(element, onVisible);
  observer.observe(element);
  return () => {
    logoObserverCallbacks.delete(element);
    observer.unobserve(element);
  };
}

export const CompanyLogo = memo(function CompanyLogo({
  ticker,
  label,
  size = 32,
  className = '',
  defer = true,
}: Props) {
  const holderRef = useRef<HTMLSpanElement>(null);
  // All display sizes intentionally share one normalized 128px asset URL.
  const src = logoUrl(ticker);
  const [visible, setVisible] = useState(!defer);
  const [loadedSrc, setLoadedSrc] = useState<string | null>(() => src && hasLoadedLogo(src) ? src : null);
  const [failedSrc, setFailedSrc] = useState<string | null>(() => src && hasFailedLogo(src) ? src : null);
  const loaded = !!src && (loadedSrc === src || hasLoadedLogo(src));
  const failed = !src || failedSrc === src || hasFailedLogo(src);

  useEffect(() => {
    if (!defer || visible || !holderRef.current || typeof IntersectionObserver === 'undefined') {
      setVisible(true);
      return;
    }
    return observeLogoVisibility(holderRef.current, () => setVisible(true));
  }, [defer, visible]);

  const initials = normalizeLogoTicker(ticker).replace(/[^A-Z0-9]/g, '').slice(0, 2) || '?';
  return (
    <span
      ref={holderRef}
      className={`pointer-events-none relative inline-flex shrink-0 select-none items-center justify-center overflow-hidden rounded-full text-[9px] font-bold text-slate-200 ${loaded && !failed ? 'bg-slate-950/85 ring-1 ring-white/15' : ''} ${className}`}
      style={{ width: size, height: size, userSelect: 'none', WebkitUserSelect: 'none' }}
      aria-hidden="true"
    >
      {visible && src && !failed && (
        <img
          src={src}
          alt=""
          width={size}
          height={size}
          loading="eager"
          decoding="async"
          draggable={false}
          className={`h-full w-full object-contain ${loaded ? '' : 'opacity-0'}`}
          onLoad={() => {
            markLogoLoaded(src);
            setLoadedSrc(src);
          }}
          onError={() => {
            markLogoFailed(src);
            setFailedSrc(src);
          }}
        />
      )}
      {(!loaded || failed) && <span className="absolute inset-0 flex items-center justify-center" title={failed ? `${label || ticker} logo unavailable` : undefined}>{initials}</span>}
    </span>
  );
});
