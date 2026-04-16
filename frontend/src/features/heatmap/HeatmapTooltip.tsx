import React from 'react';
import { HeatmapData } from './heatmap-layout';

interface HeatmapTooltipProps {
  data: HeatmapData;
  x: number;
  y: number;
}

const HeatmapTooltip: React.FC<HeatmapTooltipProps> = ({ data, x, y }) => {
  if (!data) return null;

  return (
    <div
      style={{
        position: 'fixed',
        left: x + 15,
        top: y + 15,
        zIndex: 50,
        pointerEvents: 'none',
      }}
      className="bg-[#0f172a] border border-slate-700 shadow-xl rounded px-3 py-2 min-w-[200px] font-sans text-white text-sm"
    >
      <div className="font-bold text-base mb-1">{data.shortName || data.name}</div>
      <div className="text-slate-400 text-xs mb-2">{data.sector} - {data.industry}</div>
      
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="text-slate-400">Ticker</div>
        <div className="text-right font-mono">{data.name}</div>
        
        <div className="text-slate-400">Price</div>
        <div className="text-right font-mono">${data.price?.toFixed(2) || 'N/A'}</div>
        
        <div className="text-slate-400">Change</div>
        <div className={`text-right font-mono ${data.changePercent! > 0 ? 'text-green-500' : 'text-red-500'}`}>
          {data.changePercent! > 0 ? '+' : ''}{data.changePercent?.toFixed(2)}%
        </div>
        
        <div className="text-slate-400">Market Cap</div>
        <div className="text-right font-mono">
          {data.marketCap ? `$${(data.marketCap / 1e9).toFixed(2)}B` : 'N/A'}
        </div>
      </div>
    </div>
  );
};

export default HeatmapTooltip;
