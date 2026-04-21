import React, { useLayoutEffect, useRef, useState } from 'react';
import { HeatmapData } from './heatmap-layout';

interface HeatmapTooltipProps {
  data: HeatmapData;
  x: number;
  y: number;
}

/**
 * Viewport-aware tooltip. Measured with `useLayoutEffect` and flipped
 * to the opposite side of the cursor when it would otherwise overflow
 * the viewport, so tooltips never clip at the edges of the dashboard.
 */
const HeatmapTooltip: React.FC<HeatmapTooltipProps> = ({ data, x, y }) => {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number }>({
    left: x + 15,
    top: y + 15,
  });

  useLayoutEffect(() => {
    if (!data) return;
    const el = ref.current;
    if (!el) return;

    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const vpW = window.innerWidth;
    const vpH = window.innerHeight;

    const MARGIN = 8;
    const OFFSET = 15;

    let left = x + OFFSET;
    let top = y + OFFSET;

    if (left + w + MARGIN > vpW) {
      left = Math.max(MARGIN, x - OFFSET - w);
    }
    if (top + h + MARGIN > vpH) {
      top = Math.max(MARGIN, y - OFFSET - h);
    }
    left = Math.max(MARGIN, Math.min(left, vpW - w - MARGIN));
    top = Math.max(MARGIN, Math.min(top, vpH - h - MARGIN));

    setPos({ left, top });
  }, [data, x, y]);

  if (!data) return null;

  const formatWeight = () => {
    if (data.weight === undefined) return 'N/A';
    const label = data.weightLabel || '';
    if (label === 'Market Cap') return `$${(data.weight / 1e9).toFixed(2)}B`;
    if (label === 'Dollar Volume') return `$${(data.weight / 1e6).toFixed(1)}M/day`;
    return 'Equal Weight';
  };

  return (
    <div
      ref={ref}
      style={{
        position: 'fixed',
        left: pos.left,
        top: pos.top,
        zIndex: 50,
        pointerEvents: 'none',
      }}
      className="bg-[#0f172a] border border-slate-700 shadow-xl rounded px-3 py-2 min-w-[220px] font-sans text-white text-sm"
    >
      <div className="font-bold text-base mb-1 truncate">{data.shortName || data.name}</div>
      <div className="text-slate-500 text-xs mb-2 uppercase tracking-wider">
        {data.group}{data.subgroup ? ` · ${data.subgroup}` : ''}
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
        <div className="text-slate-400">Ticker</div>
        <div className="text-right font-mono text-slate-200">{data.name}</div>

        <div className="text-slate-400">Price</div>
        <div className="text-right font-mono text-slate-200">
          {data.price !== undefined ? `$${data.price.toFixed(data.price < 10 ? 4 : 2)}` : 'N/A'}
        </div>

        <div className="text-slate-400">Change</div>
        <div className={`text-right font-mono font-semibold ${(data.changePercent ?? 0) > 0 ? 'text-green-400' : (data.changePercent ?? 0) < 0 ? 'text-red-400' : 'text-slate-400'}`}>
          {data.changePercent !== undefined
            ? `${data.changePercent > 0 ? '+' : ''}${data.changePercent.toFixed(2)}%`
            : 'N/A'}
        </div>

        <div className="text-slate-400">{data.weightLabel || 'Weight'}</div>
        <div className="text-right font-mono text-slate-200">{formatWeight()}</div>
      </div>
    </div>
  );
};

export default HeatmapTooltip;
