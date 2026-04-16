import React, { useMemo, useState } from 'react';
import { buildTreemapLayout, HeatmapNode, HeatmapData } from './heatmap-layout';
import HeatmapTooltip from './HeatmapTooltip';

interface HeatmapTreemapProps {
  data: HeatmapNode;
  width: number;
  height: number;
}

export function getColorForChange(change: number | undefined): string {
  if (change === undefined) return '#1e293b'; // neutral dark gray

  if (change >= 2.0) return '#15803d'; // strong green (tailwind green-700)
  if (change > 0) return '#22c55e'; // medium green (tailwind green-500)
  if (change <= -2.0) return '#b91c1c'; // sharp red (tailwind red-700)
  if (change < 0) return '#ef4444'; // slight red (tailwind red-500)
  return '#334155'; // neutral gray (tailwind slate-700)
}

const HeatmapTreemap: React.FC<HeatmapTreemapProps> = ({ data, width, height }) => {
  const [hoveredNode, setHoveredNode] = useState<{ node: any; x: number; y: number } | null>(null);

  const layout = useMemo(() => {
    return buildTreemapLayout(data, width, height);
  }, [data, width, height]);

  const leaves = layout.leaves();
  const parentNodes = layout.descendants().filter(d => d.depth > 0 && d.children);

  const handleMouseMove = (e: React.MouseEvent, leaf: any) => {
    setHoveredNode({
      node: leaf,
      x: e.clientX,
      y: e.clientY
    });
  };

  const handleMouseLeave = () => {
    setHoveredNode(null);
  };

  return (
    <div style={{ width, height, position: 'relative', overflow: 'hidden', backgroundColor: '#0f172a' }}>
      {/* Background container */}
      
      {/* Parent Labels (group / subgroup levels) */}
      {parentNodes.map((p, i) => {
        const nodeWidth = p.x1 - p.x0;
        if (nodeWidth < 40 || p.y1 - p.y0 < 20) return null;

        return (
          <div
            key={`parent-${i}`}
            style={{
              position: 'absolute',
              left: p.x0,
              top: p.y0,
              width: nodeWidth,
              height: p.y1 - p.y0,
              border: '1px solid #1e293b',
              pointerEvents: 'none',
            }}
          >
            {/* depth 1 = group, depth 2 = subgroup — show label if enough room */}
            {p.depth <= 2 && nodeWidth > 50 && (
              <div
                className={`px-1 pt-1 truncate bg-[#0f172a] bg-opacity-90 ${
                  p.depth === 1
                    ? 'text-[11px] font-semibold text-slate-300 uppercase tracking-widest'
                    : 'text-[9px] font-medium text-slate-500 uppercase tracking-wider'
                }`}
              >
                {(p.data as any).name}
              </div>
            )}
          </div>
        );
      })}

      {/* Leaves (Stocks) */}
      {leaves.map((leaf, i) => {
        const cellData = leaf.data as HeatmapData;
        const cellWidth = leaf.x1 - leaf.x0;
        const cellHeight = leaf.y1 - leaf.y0;
        
        // Hide extremely tiny cells
        if (cellWidth < 10 || cellHeight < 10) return null;

        const bgColor = getColorForChange(cellData.changePercent);
        const isLarge = cellWidth > 60 && cellHeight > 40;
        const isMedium = cellWidth > 40 && cellHeight > 25;

        return (
          <div
            key={`leaf-${i}`}
            onMouseMove={(e) => handleMouseMove(e, cellData)}
            onMouseLeave={handleMouseLeave}
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
              overflow: 'hidden'
            }}
            className="hover:brightness-125"
          >
            <span className={`font-bold font-sans text-white ${isLarge ? 'text-sm' : 'text-[10px]'} leading-tight truncate px-1`}>
              {cellData.name}
            </span>
            {isMedium && cellData.changePercent !== undefined && (
              <span className={`font-thin text-white ${isLarge ? 'text-xs' : 'text-[9px]'} leading-tight`}>
                {cellData.changePercent > 0 ? '+' : ''}{cellData.changePercent.toFixed(2)}%
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

      {/* Tooltip Overlay */}
      {hoveredNode && (
        <HeatmapTooltip 
          data={hoveredNode.node} 
          x={hoveredNode.x} 
          y={hoveredNode.y} 
        />
      )}
    </div>
  );
};

export default HeatmapTreemap;
