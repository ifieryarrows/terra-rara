import { memo } from 'react';
import { categoryHeaderPadding, detailLevel, stockTextSizes, type HeatmapData, type HeatmapNode, type LayoutNode } from './heatmap-layout';
import { CompanyLogo } from './CompanyLogo';
import { getColorForChange } from './heatmap-utils';

const LOGO_INSTRUMENT_TYPES = new Set(['equity', 'etf', 'mutualfund']);
const MIN_TILE_SIZE = 4;
const textWidthCache = new Map<string, number>();
let textMeasureContext: CanvasRenderingContext2D | null | undefined;

function textWidthAt100px(text: string, weight: 600 | 700) {
  const key = `${weight}:${text}`;
  const cached = textWidthCache.get(key);
  if (cached != null) return cached;
  if (typeof document !== 'undefined') {
    textMeasureContext ??= document.createElement('canvas').getContext('2d');
  }
  const context = textMeasureContext;
  let width = text.length * 66;
  if (context) {
    context.font = `${weight} 100px "Geist Sans", system-ui, sans-serif`;
    width = context.measureText(text).width;
  }
  if (textWidthCache.size >= 2_048) textWidthCache.clear();
  textWidthCache.set(key, width);
  return width;
}

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
}: {
  parents: LayoutNode[];
  hoveredCategoryId: string | null;
}) {
  return parents.map((node) => {
    const nodeData = node.data as HeatmapNode;
    const id = String(nodeData.id || `${node.depth}-${nodeData.name}`);
    return <CategoryTile key={id} node={node} active={hoveredCategoryId === id} />;
  });
});

const CategoryTile = memo(function CategoryTile({ node, active }: { node: LayoutNode; active: boolean }) {
  const nodeData = node.data as HeatmapNode;
  const nodeWidth = node.x1 - node.x0;
  const nodeHeight = node.y1 - node.y0;
  if (nodeWidth < 24 || nodeHeight < 20) return null;
  const headerPadding = categoryHeaderPadding(node.depth, nodeWidth, nodeHeight);
  const id = String(nodeData.id || `${node.depth}-${nodeData.name}`);
  const headerWeight = node.depth === 1 ? 700 : 600;
  const headerTextWidth = (textWidthAt100px(nodeData.name.toLocaleUpperCase(), headerWeight) / 100) * 1.06;
  const headerFontSize = Math.max(5, Math.min(
    node.depth === 1 ? 10 : 8,
    (nodeWidth - 10) / Math.max(0.1, headerTextWidth),
    (headerPadding - 2) * 0.72,
  ));
  return (
    <div
      data-hm-category-id={id}
      role="button"
      tabIndex={0}
      aria-label={`${node.depth === 1 ? 'Sector or asset class' : 'Industry or theme'}: ${nodeData.name}. Double-click or press Shift+Enter to drill into this category.`}
      aria-pressed={active}
      className="absolute overflow-hidden outline-none focus-visible:ring-2 focus-visible:ring-copper-400"
      style={{
        left: node.x0, top: node.y0, width: nodeWidth, height: nodeHeight,
        border: active ? '1px solid #d99a5b' : '0.5px solid #253244',
        backgroundColor: '#020617',
        boxShadow: undefined,
        // Category geometry stays below stock cells so the copper
        // highlight never intercepts stock hover/focus events.
        zIndex: 1,
      }}
    >
      {headerPadding > 1 && (
        <div
          className={node.depth === 1
            ? 'pointer-events-none overflow-hidden whitespace-nowrap bg-slate-900/95 px-1.5 pt-0.5 font-bold uppercase tracking-wide text-slate-200'
            : 'pointer-events-none overflow-hidden whitespace-nowrap bg-slate-800/95 px-1 font-semibold uppercase tracking-wide text-slate-400'}
          style={{ height: headerPadding - 1, fontSize: headerFontSize, lineHeight: 1.15 }}
        >
          {nodeData.name}
        </div>
      )}
    </div>
  );
});

export const LeafTiles = memo(function LeafTiles({ leafEntries }: { leafEntries: LeafEntry[] }) {
  return leafEntries.map(({ leaf, parentId, renderId }) => {
    const item = leaf.data as HeatmapData;
    const cellWidth = leaf.x1 - leaf.x0;
    const cellHeight = leaf.y1 - leaf.y0;
    if (cellWidth <= 0 || cellHeight <= 0) return null;
    const level = detailLevel(cellWidth, cellHeight);
    const tickerScale = minimumScaleForSize(cellWidth, cellHeight, 24, 18, 520);
    const changeScale = minimumScaleForSize(cellWidth, cellHeight, 44, 25, 1_250);
    const tileScale = minimumScaleForSize(cellWidth, cellHeight, MIN_TILE_SIZE, MIN_TILE_SIZE, MIN_TILE_SIZE ** 2);
    const change = item.changePercent || 0;
    const changeLabel = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
    const tickerSizingScale = level === 'color' ? tickerScale : 1;
    const tickerWidth = cellWidth * tickerSizingScale;
    const tickerHeight = cellHeight * tickerSizingScale;
    const tickerLevel = level === 'color' ? 'ticker' : level;
    const tickerSizes = stockTextSizes(tickerWidth, tickerHeight, tickerLevel);
    const tickerTextWidth = (textWidthAt100px(item.name, 700) / 100) * 1.04;
    const tickerFontSize = Math.max(0.5, Math.min(
      tickerSizes.ticker,
      Math.max(0, tickerWidth - 8) / Math.max(0.1, tickerTextWidth),
      Math.max(0, tickerHeight * 0.19) / 1.04,
    ) / tickerSizingScale);
    const changeSizingScale = ['change', 'logo', 'price'].includes(level) ? 1 : changeScale;
    const changeWidth = cellWidth * changeSizingScale;
    const changeHeight = cellHeight * changeSizingScale;
    const changeLevel = ['change', 'logo', 'price'].includes(level) ? level : 'change';
    const changeSizes = stockTextSizes(changeWidth, changeHeight, changeLevel);
    const changeTextWidth = (textWidthAt100px(changeLabel, 600) / 100) * 1.08;
    const changeFontSize = Math.max(0.5, Math.min(
      changeSizes.change,
      Math.max(0, changeWidth - 2) / Math.max(0.1, changeTextWidth),
      Math.max(0, changeHeight * 0.13) / 1.08,
    ) / changeSizingScale);
    const showTicker = level !== 'color';
    const showChange = ['change', 'logo', 'price'].includes(level);
    const fallbackLogoTicker = LOGO_INSTRUMENT_TYPES.has((item.instrumentType || '').toLowerCase())
      ? item.name
      : null;
    const logoTicker = item.logoTicker || fallbackLogoTicker;
    const targetLogoSize = level === 'price'
      ? Math.min(42, cellHeight * 0.34)
      : Math.min(28, cellHeight * 0.3);
    const logoSize = Math.min(targetLogoSize, cellWidth - 8, cellHeight * 0.22);
    const showLogo = ['logo', 'price'].includes(level)
      && showChange
      && logoSize >= 8
      && !!logoTicker
      && !item.aggregateCount;
    return (
      <div
        key={renderId}
        data-hm-leaf-id={renderId}
        data-hm-parent-id={parentId}
        data-hm-tile-min-scale={tileScale}
        role="button"
        tabIndex={0}
        aria-label={`${item.aggregateCount ? item.shortName : `${item.name}, ${item.shortName || ''}`}. Price ${item.price ?? 'unavailable'}. Daily change ${change >= 0 ? 'plus ' : 'minus '}${Math.abs(change).toFixed(2)} percent.`}
        className="absolute z-[2] flex flex-col items-center justify-center overflow-hidden text-center text-white outline-none transition-[filter] duration-75 hover:brightness-125 focus-visible:z-10 focus-visible:ring-2 focus-visible:ring-white"
        style={{
          left: leaf.x0,
          top: leaf.y0,
          width: cellWidth,
          height: cellHeight,
          backgroundColor: getColorForChange(item.changePercent),
          visibility: tileScale <= 1 ? 'visible' : 'hidden',
        }}
      >
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          {showLogo && (
            <CompanyLogo
              ticker={logoTicker || item.name}
              label={item.shortName}
              size={logoSize}
              className="mb-1"
            />
          )}
          <strong
            data-hm-detail-min-scale={tickerScale}
            className="max-w-full whitespace-nowrap px-1 font-bold tracking-[-0.02em]"
            style={{ visibility: showTicker ? 'visible' : 'hidden', fontSize: tickerFontSize, lineHeight: 1.04, textShadow: '0 1px 2px rgba(0,0,0,.45)' }}
          >
            {item.name}
          </strong>
          <span
            data-hm-detail-min-scale={changeScale}
            className="max-w-full whitespace-nowrap font-semibold tabular-nums tracking-[-0.015em]"
            style={{ visibility: showChange ? 'visible' : 'hidden', fontSize: changeFontSize, lineHeight: 1.08, textShadow: '0 1px 2px rgba(0,0,0,.42)' }}
          >
            {changeLabel}
          </span>
        </div>
      </div>
    );
  });
});
