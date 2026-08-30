import { memo, useEffect, useRef, useState } from 'react';
import { logoUrl, normalizeLogoTicker } from './heatmap-utils';
import { hasFailedLogo, markLogoFailed } from './logo-cache';

interface Props {
  ticker: string;
  label?: string;
  size?: number;
  className?: string;
  defer?: boolean;
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
  const [failed, setFailed] = useState(() => !src || hasFailedLogo(src));

  useEffect(() => {
    setFailed(!src || hasFailedLogo(src));
  }, [src]);

  useEffect(() => {
    if (!defer || visible || !holderRef.current || typeof IntersectionObserver === 'undefined') {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => entry.isIntersecting && setVisible(true),
      { rootMargin: '48px' },
    );
    observer.observe(holderRef.current);
    return () => observer.disconnect();
  }, [defer, visible]);

  const initials = normalizeLogoTicker(ticker).replace(/[^A-Z0-9]/g, '').slice(0, 2) || '?';
  return (
    <span
      ref={holderRef}
      className={`pointer-events-none inline-flex shrink-0 select-none items-center justify-center overflow-hidden rounded-full bg-slate-950/85 text-[9px] font-bold text-slate-200 ring-1 ring-white/15 ${className}`}
      style={{ width: size, height: size, userSelect: 'none', WebkitUserSelect: 'none' }}
      aria-hidden="true"
    >
      {visible && src && !failed ? (
        <img
          src={src}
          alt=""
          width={size}
          height={size}
          loading="lazy"
          decoding="async"
          draggable={false}
          className="h-full w-full object-contain"
          onError={() => {
            markLogoFailed(src);
            setFailed(true);
          }}
        />
      ) : (
        <span title={`${label || ticker} logo unavailable`}>{initials}</span>
      )}
    </span>
  );
});
