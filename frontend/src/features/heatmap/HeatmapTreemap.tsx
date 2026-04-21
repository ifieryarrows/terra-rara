import React, { useMemo, useState, useRef, useEffect } from 'react';
import { buildTreemapLayout, HeatmapNode, HeatmapData } from './heatmap-layout';
import HeatmapTooltip from './HeatmapTooltip';

interface HeatmapTreemapProps {
  data: HeatmapNode;
  width: number;
  height: number;
  zoom: number;
  hoveredCategory: string | null;
  onCategoryHover: (name: string | null) => void;
  onCategoryClick?: (name: string) => void;
}

export function getColorForChange(change: number | undefined): string {
  if (change === undefined) return '#1e293b';
  if (change >= 2.0) return '#15803d';
  if (change > 0) return '#22c55e';
  if (change <= -2.0) return '#b91c1c';
  if (change < 0) return '#ef4444';
  return '#334155';
}

const HeatmapTreemap: React.FC<HeatmapTreemapProps> = ({
  data,
  width,
  height,
  zoom,
  hoveredCategory,
  onCategoryHover,
  onCategoryClick,
}) => {
  const [hoveredNode, setHoveredNode] = useState<{ node: any; x: number; y: number } | null>(
    null,
  );
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep scroll position stable when zoom changes
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // noop — just ensures re-render; could be extended to re-center on hover
  }, [zoom]);

  const scaledWidth = Math.max(1, Math.round(width * zoom));
  const scaledHeight = Math.max(1, Math.round(height * zoom));

  const layout = useMemo(() => {
    return buildTreemapLayout(data, scaledWidth, scaledHeight);
  }, [data, scaledWidth, scaledHeight]);

  const leaves = layout.leaves();
  const parentNodes = layout.descendants().filter((d) => d.depth > 0 && d.children);

  const handleLeafMove = (e: React.MouseEvent, leaf: any) => {
    setHoveredNode({ node: leaf, x: e.clientX, y: e.clientY });
  };

  const handleLeafLeave = () => setHoveredNode(null);

  return (
    <div
      ref={scrollRef}
      style={{
        width: '100%',
        height,
        position: 'relative',
        overflow: zoom > 1 ? 'auto' : 'hidden',
        backgroundColor: '#0f172a',
      }}
      className="custom-scrollbar"
    >
      <div
        style={{
          width: scaledWidth,
          height: scaledHeight,
          position: 'relative',
        }}
      >
        {/* Parent cells — used for hover/click detection (category selection) */}
        {parentNodes.map((p, i) => {
          const nodeWidth = p.x1 - p.x0;
          const nodeHeight = p.y1 - p.y0;
          if (nodeWidth < 20 || nodeHeight < 20) return null;

          const name = (p.data as any).name as string;
          const isHighlighted = hoveredCategory === name;

          return (
            <div
              key={`parent-${i}`}
              onMouseEnter={() => onCategoryHover(name)}
              onMouseLeave={() => onCategoryHover(null)}
              onClick={() => onCategoryClick && onCategoryClick(name)}
              style={{
                position: 'absolute',
                left: p.x0,
                top: p.y0,
                width: nodeWidth,
                height: nodeHeight,
                border: isHighlighted
                  ? '2px solid #facc15'
                  : '1px solid #1e293b',
                boxShadow: isHighlighted
                  ? '0 0 0 2px rgba(250, 204, 21, 0.25), inset 0 0 0 1px rgba(250, 204, 21, 0.4)'
                  : undefined,
                cursor: onCategoryClick ? 'pointer' : 'default',
                zIndex: isHighlighted ? 5 : 1,
                transition: 'box-shadow 120ms ease',
              }}
            >
              {p.depth <= 2 && nodeWidth > 50 && (
                <div
                  className={`px-1 pt-1 truncate bg-[#0f172a] bg-opacity-90 ${
                    p.depth === 1
                      ? 'text-[11px] font-semibold text-slate-300 uppercase tracking-widest'
                      : 'text-[9px] font-medium text-slate-500 uppercase tracking-wider'
                  }`}
                  style={{ pointerEvents: 'none' }}
                >
                  {name}
                </div>
              )}
            </div>
          );
        })}

        {/* Leaves (symbols) */}
        {leaves.map((leaf, i) => {
          const cellData = leaf.data as HeatmapData;
          const cellWidth = leaf.x1 - leaf.x0;
          const cellHeight = leaf.y1 - leaf.y0;
          if (cellWidth < 8 || cellHeight < 8) return null;

          const bgColor = getColorForChange(cellData.changePercent);
          const isLarge = cellWidth > 60 && cellHeight > 40;
          const isMedium = cellWidth > 40 && cellHeight > 25;

          return (
            <div
              key={`leaf-${i}`}
              onMouseMove={(e) => handleLeafMove(e, cellData)}
              onMouseLeave={handleLeafLeave}
              style={{
                position: 'absolute',
                left: leaf.x0,
                top: leaf.y0,
                width: cellWidth,
                height: cellHeight,
                backgroundColor: bgColor,
                border: '1px solid #0f172a',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                transition: 'filter 0.1s ease',
                overflow: 'hidden',
                zIndex: 2,
              }}
              className="hover:brightness-125"
            >
              <span
                className={`font-bold text-white ${
                  isLarge ? 'text-sm' : 'text-[10px]'
                } leading-tight truncate px-1`}
              >
                {cellData.name}
              </span>
              {isMedium && cellData.changePercent !== undefined && (
                <span
                  className={`font-thin text-white ${
                    isLarge ? 'text-xs' : 'text-[9px]'
                  } leading-tight`}
                >
                  {cellData.changePercent > 0 ? '+' : ''}
                  {cellData.changePercent.toFixed(2)}%
                </span>
              )}
              {isLarge && cellData.price !== undefined && (
                <span className="font-thin text-white text-[10px] opacity-80 leading-tight">
                  ${cellData.price.toFixed(2)}
                </span>
              )}
            </div>
          );
        })}

        {hoveredNode && (
          <HeatmapTooltip data={hoveredNode.node} x={hoveredNode.x} y={hoveredNode.y} />
        )}
      </div>
    </div>
  );
};

export default HeatmapTreemap;
