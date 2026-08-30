import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { HeatmapData } from './heatmap-layout';
import { CompanyLogo } from './CompanyLogo';
import { Sparkline } from './Sparkline';
import { clampTooltipPosition } from './heatmap-utils';

export interface HeatmapTooltipHandle {
  show: (data: HeatmapData, x: number, y: number) => void;
  move: (x: number, y: number) => void;
  hide: () => void;
}

const HeatmapTooltip = forwardRef<HeatmapTooltipHandle>(function HeatmapTooltip(_, ref) {
  const [data, setData] = useState<HeatmapData | null>(null);
  const dataRef = useRef<HeatmapData | null>(null);
  const elementRef = useRef<HTMLDivElement>(null);
  const pointerRef = useRef({ x: 0, y: 0 });
  const frameRef = useRef<number | null>(null);

  const applyPosition = useCallback(() => {
    frameRef.current = null;
    if (!elementRef.current) return;
    const position = clampTooltipPosition(
      pointerRef.current.x,
      pointerRef.current.y,
      window.innerWidth,
      window.innerHeight,
    );
    elementRef.current.style.transform = `translate3d(${position.left}px, ${position.top}px, 0)`;
  }, []);

  const move = useCallback((x: number, y: number) => {
    pointerRef.current = { x, y };
    if (frameRef.current === null) frameRef.current = requestAnimationFrame(applyPosition);
  }, [applyPosition]);

  useImperativeHandle(ref, () => ({
    show(next, x, y) {
      if (dataRef.current?.id !== next.id || dataRef.current?.name !== next.name) {
        dataRef.current = next;
        setData(next);
      }
      move(x, y);
    },
    move,
    hide() {
      dataRef.current = null;
      setData(null);
    },
  }), [move]);

  useEffect(() => {
    if (data) move(pointerRef.current.x, pointerRef.current.y);
  }, [data, move]);

  useEffect(() => () => {
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
  }, []);

  if (!data || typeof document === 'undefined') return null;
  const change = data.changePercent || 0;
  const body = (
    <div
      ref={elementRef}
      role="tooltip"
      className="pointer-events-none fixed left-0 top-0 z-[80] w-[292px] rounded-lg border border-slate-600 bg-slate-950/95 p-3 text-slate-100 shadow-2xl backdrop-blur"
      style={{ willChange: 'transform' }}
    >
      <div className="flex items-center gap-3">
        <CompanyLogo ticker={data.logoTicker || data.name} label={data.shortName} size={38} defer={false} />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-2">
            <strong className="text-base tracking-wide">{data.name}</strong>
            <span className={change > 0 ? 'text-emerald-400' : change < 0 ? 'text-rose-400' : 'text-slate-400'}>
              {change > 0 ? '+' : ''}{change.toFixed(2)}%
            </span>
          </div>
          <p className="truncate text-xs text-slate-400">{data.shortName || data.name}</p>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-[1fr_auto] items-center gap-3 border-t border-white/10 pt-2">
        <div>
          <div className="font-mono text-lg tabular-nums">
            {data.price == null ? '—' : `$${data.price.toFixed(data.price < 10 ? 4 : 2)}`}
          </div>
          <p className="mt-1 truncate text-[10px] text-slate-500">
            {[data.sector || data.group, data.industry || data.subgroup].filter(Boolean).join(' · ')}
          </p>
        </div>
        <Sparkline values={data.sparkline} positive={change >= 0} width={80} height={28} />
      </div>
    </div>
  );
  return createPortal(body, document.body);
});

export default HeatmapTooltip;
