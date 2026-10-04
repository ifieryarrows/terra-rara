import { memo } from 'react';
import { categoryHeaderPadding, detailLevel, stockTextSizes, type HeatmapData, type HeatmapNode, type LayoutNode } from './heatmap-layout';
import { CompanyLogo } from './CompanyLogo';
import { getColorForChange } from './heatmap-utils';

const LOGO_INSTRUMENT_TYPES = new Set(['equity', 'etf', 'mutualfund']);
const MIN_TILE_SIZE = 4;

function minimumScaleForSize(width: number, height: number, minWidth: number, minHeight: number, minArea: number) {
  const safeWidth = Math.max(width, 0.001);
  const safeHeight = Math.max(height, 0.001);
  return Math.max(1, minWidth / safeWidth, minHeight / safeHeight, Math.sqrt(minArea / (safeWidth * safeHeight)));
}

export interface LeafEntry {
  leaf: LayoutNode;
  parentId: string;
  renderId: string;
}

export const CategoryTiles = memo(function CategoryTiles({
  parents,
  hoveredCategoryId,
  zoom = 1,
}: {
  parents: LayoutNode[];
  hoveredCategoryId: string | null;
  zoom?: number;
}) {
  return parents.map((node) => {
    const nodeData = node.data as HeatmapNode;
    const id = String(nodeData.id || `${node.depth}-${nodeData.name}`);
    const nodeWidth = node.x1 - node.x0;
    const nodeHeight = node.y1 - node.y0;
    if (nodeWidth < 24 || nodeHeight < 20) return null;
    const headerPadding = categoryHeaderPadding(node.depth, nodeWidth, nodeHeight);
    const active = hoveredCategoryId === id;
    return (
      <div
        key={id}
        data-hm-category-id={id}
        role="button"
        tabIndex={0}
        aria-label={`${node.depth === 1 ? 'Sector or asset class' : 'Industry or theme'}: ${nodeData.name}`}
        aria-pressed={active}
        className={`absolute overflow-hidden outline-none focus-visible:ring-2 focus-visible:ring-copper-400 ${zoom > 1 ? 'cursor-grab' : 'cursor-pointer'}`}
        style={{
          left: node.x0, top: node.y0, width: nodeWidth, height: nodeHeight,
          border: active ? '2px solid #d99a5b' : node.depth === 1 ? '1px solid #334155' : '1px solid #1e293b',
          backgroundColor: active ? '#d99a5b' : '#020617',
          boxShadow: active ? '0 0 0 2px rgba(217,154,91,.22), inset 0 0 18px rgba(217,154,91,.08)' : undefined,
          // Category geometry stays below stock cells so the copper
          // highlight never intercepts stock hover/focus events.
          zIndex: 1,
        }}
      >
        {headerPadding > 1 && (
          <div
            className={node.depth === 1
              ? 'pointer-events-none truncate bg-slate-900/95 px-1.5 pt-0.5 text-[10px] font-bold uppercase tracking-wide text-slate-200'
              : 'pointer-events-none truncate bg-slate-800/95 px-1 text-[8px] font-semibold uppercase tracking-wide text-slate-400'}
            style={{ height: headerPadding - 1 }}
          >
            {nodeData.name}
          </div>
        )}
      </div>
    );
  });
});

export const LeafTiles = memo(function LeafTiles({ leafEntries, zoom = 1 }: { leafEntries: LeafEntry[]; zoom?: number }) {
  return leafEntries.map(({ leaf, parentId, renderId }) => {
    const item = leaf.data as HeatmapData;
    const cellWidth = leaf.x1 - leaf.x0;
    const cellHeight = leaf.y1 - leaf.y0;
    if (cellWidth <= 0 || cellHeight <= 0) return null;
    const level = detailLevel(cellWidth, cellHeight);
    const textSizes = stockTextSizes(cellWidth, cellHeight, level);
    const tileScale = minimumScaleForSize(cellWidth, cellHeight, MIN_TILE_SIZE, MIN_TILE_SIZE, MIN_TILE_SIZE ** 2);
    const tickerScale = minimumScaleForSize(cellWidth, cellHeight, 24, 18, 520);
    const changeScale = minimumScaleForSize(cellWidth, cellHeight, 44, 25, 1_250);
    const logoScale = minimumScaleForSize(cellWidth, cellHeight, 66, 42, 2_800);
    const tickerSizesAtReveal = stockTextSizes(cellWidth * tickerScale, cellHeight * tickerScale, 'ticker');
    const changeSizesAtReveal = stockTextSizes(cellWidth * changeScale, cellHeight * changeScale, 'change');
    const change = item.changePercent || 0;
    const showTicker = level !== 'color';
    const showChange = ['change', 'logo', 'price'].includes(level);
    const fallbackLogoTicker = LOGO_INSTRUMENT_TYPES.has((item.instrumentType || '').toLowerCase())
      ? item.name
      : null;
    const logoTicker = item.logoTicker || fallbackLogoTicker;
    const showLogo = ['logo', 'price'].includes(level) && !!logoTicker && !item.aggregateCount;
    const logoSize = level === 'price'
      ? Math.min(42, cellHeight * 0.34)
      : level === 'logo'
        ? Math.min(28, cellHeight * 0.3)
        : Math.min(28, cellHeight * logoScale * 0.3) / logoScale;
    return (
      <div
        key={renderId}
        data-hm-leaf-id={renderId}
        data-hm-parent-id={parentId}
        data-hm-tile-min-scale={tileScale}
        role="button"
        tabIndex={0}
        aria-label={`${item.aggregateCount ? item.shortName : `${item.name}, ${item.shortName || ''}`}. Price ${item.price ?? 'unavailable'}. Daily change ${change >= 0 ? 'plus ' : 'minus '}${Math.abs(change).toFixed(2)} percent.`}
        className={`absolute z-[2] flex flex-col items-center justify-center overflow-hidden text-center text-white outline-none transition-[filter] duration-75 hover:brightness-125 focus-visible:z-10 focus-visible:ring-2 focus-visible:ring-white ${zoom > 1 ? 'cursor-grab' : 'cursor-crosshair'}`}
        style={{
          left: leaf.x0,
          top: leaf.y0,
          width: cellWidth,
          height: cellHeight,
          backgroundColor: getColorForChange(item.changePercent),
          visibility: tileScale <= 1 ? 'visible' : 'hidden',
        }}
      >
        {logoTicker && !item.aggregateCount && (
          <div
            data-hm-detail-min-scale={logoScale}
            style={{ visibility: showLogo ? 'visible' : 'hidden' }}
          >
            <CompanyLogo
              ticker={logoTicker}
              label={item.shortName}
              size={logoSize}
              className="mb-1"
            />
          </div>
        )}
        <strong
          data-hm-detail-min-scale={tickerScale}
          className="max-w-full truncate px-1 font-bold tracking-[-0.02em]"
          style={{
            visibility: showTicker ? 'visible' : 'hidden',
            fontSize: showTicker ? textSizes.ticker : tickerSizesAtReveal.ticker / tickerScale,
            lineHeight: 1.04,
            textShadow: '0 1px 2px rgba(0,0,0,.45)',
          }}
        >
          {item.name}
        </strong>
        <span
          data-hm-detail-min-scale={changeScale}
          className="font-semibold tabular-nums tracking-[-0.015em]"
          style={{
            visibility: showChange ? 'visible' : 'hidden',
            fontSize: showChange ? textSizes.change : changeSizesAtReveal.change / changeScale,
            lineHeight: 1.08,
            textShadow: '0 1px 2px rgba(0,0,0,.42)',
          }}
        >
          {change > 0 ? '+' : ''}{change.toFixed(2)}%
        </span>
      </div>
    );
  });
});
