import React from 'react';
import { HeatmapData } from './heatmap-layout';

interface HeatmapCategoryPanelProps {
  categoryName: string | null;
  leaves: HeatmapData[];
  onClose: () => void;
}

const HeatmapCategoryPanel: React.FC<HeatmapCategoryPanelProps> = ({
  categoryName,
  leaves,
  onClose,
}) => {
  if (!categoryName) return null;

  const sorted = [...leaves].sort(
    (a, b) => (b.changePercent ?? 0) - (a.changePercent ?? 0),
  );

  const up = sorted.filter((l) => (l.changePercent ?? 0) > 0).length;
  const down = sorted.filter((l) => (l.changePercent ?? 0) < 0).length;

  return (
    <aside
      className="absolute top-0 right-0 h-full w-[280px] max-w-[85vw] bg-[#0b1220] border-l border-slate-700 shadow-2xl flex flex-col z-30"
      onMouseEnter={(e) => e.stopPropagation()}
    >
      <header className="flex items-center justify-between px-3 py-2.5 border-b border-slate-700">
        <div className="min-w-0">
          <p className="text-[10px] uppercase tracking-widest text-slate-500 leading-none">
            Category Inspector
          </p>
          <h3 className="text-sm font-semibold text-white truncate">{categoryName}</h3>
          <p className="text-[10px] text-slate-500 mt-0.5">
            {sorted.length} symbols · <span className="text-emerald-400">{up}↑</span>{' '}
            <span className="text-rose-400">{down}↓</span>
          </p>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-white text-xs px-2 py-1 border border-slate-700 rounded"
        >
          Close
        </button>
      </header>

      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {sorted.length === 0 ? (
          <div className="p-4 text-slate-500 text-xs">
            No symbols in this category have live data.
          </div>
        ) : (
          <ul className="divide-y divide-slate-800">
            {sorted.map((item) => {
              const ch = item.changePercent ?? 0;
              return (
                <li
                  key={item.name}
                  className="px-3 py-2 hover:bg-slate-800/50 transition-colors"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <div className="min-w-0">
                      <div className="text-sm font-semibold text-white truncate">
                        {item.name}
                      </div>
                      <div className="text-[10px] text-slate-500 truncate">
                        {item.shortName || item.subgroup || ''}
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="text-sm font-mono text-slate-200 tabular-nums">
                        {item.price !== undefined
                          ? `$${item.price.toFixed(item.price < 10 ? 4 : 2)}`
                          : '—'}
                      </div>
                      <div
                        className={`text-[11px] font-mono font-semibold tabular-nums ${
                          ch > 0
                            ? 'text-emerald-400'
                            : ch < 0
                            ? 'text-rose-400'
                            : 'text-slate-500'
                        }`}
                      >
                        {item.changePercent !== undefined
                          ? `${ch > 0 ? '+' : ''}${ch.toFixed(2)}%`
                          : '—'}
                      </div>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
};

export default HeatmapCategoryPanel;
