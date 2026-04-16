import React from 'react';
import { HeatmapData } from './heatmap-layout';

interface HeatmapTooltipProps {
  data: HeatmapData;
  x: number;
  y: number;
}

const HeatmapTooltip: React.FC<HeatmapTooltipProps> = ({ data, x, y }) => {
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
      style={{
        position: 'fixed',
        left: x + 15,
        top: y + 15,
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
