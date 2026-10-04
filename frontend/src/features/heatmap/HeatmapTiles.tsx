import { memo } from 'react';
import {
  categoryHeaderPadding,
  formatTileChange,
  formatTilePrice,
  planTileTiers,
  tierAtScale,
  TICKER_TRACKING_EM,
  TILE_LINE_HEIGHT as LINE_HEIGHT,
  TILE_LOGO_GAP_EM,
  type HeatmapData,
  type HeatmapNode,
  type LayoutNode,
  type TileTierPlan,
} from './heatmap-layout';
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

function categoryWeightedChange(node: LayoutNode): number {
  const leaves = node.leaves();
  if (!leaves.length) return 0;
  let totalWeight = 0;
  let weightedChange = 0;
  for (const leaf of leaves) {
    const data = leaf.data as HeatmapData;
    const w = Math.max(0.0001, data.weight || 1);
    const chg = data.changePercent || 0;
    weightedChange += chg * w;
    totalWeight += w;
  }
  return totalWeight > 0 ? weightedChange / totalWeight : 0;
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
    const isSector = node.depth === 1;
    const avgChange = isSector ? 0 : categoryWeightedChange(node);
    const industryHeaderBg = getColorForChange(avgChange);

    return (
      <div
        key={id}
        data-hm-category-id={id}
        data-hm-depth={node.depth}
        role="button"
        tabIndex={0}
        aria-label={`${isSector ? 'Sector or asset class' : 'Industry or theme'}: ${nodeData.name}`}
        aria-pressed={active}
        className={`absolute overflow-hidden outline-none focus-visible:ring-2 focus-visible:ring-copper-400 ${zoom > 1 ? 'cursor-grab' : 'cursor-pointer'}`}
        style={{
          left: node.x0,
          top: node.y0,
          width: nodeWidth,
          height: nodeHeight,
          border: active
            ? '2px solid #d99a5b'
            : isSector
              ? '1.5px solid #202432'
              : '1px solid #12151c',
          backgroundColor: active ? '#d99a5b' : '#020617',
          boxShadow: active
            ? '0 0 0 2px rgba(217,154,91,.22), inset 0 0 18px rgba(217,154,91,.08)'
            : undefined,
          // Category geometry stays below stock cells so the copper
          // highlight never intercepts stock hover/focus events.
          zIndex: 1,
        }}
      >
        {headerPadding > 1 && (
          <div
            className={`pointer-events-none relative z-[3] flex items-center ${
              isSector
                ? 'border-b border-[#10131a] bg-[#161922] px-2 font-bold uppercase tracking-wider text-white'
                : 'border-b border-[#12151c] px-1.5 font-bold uppercase tracking-wide text-white'
            }`}
            style={{
              height: headerPadding - 1,
              backgroundColor: isSector ? '#161922' : industryHeaderBg,
            }}
          >
            <span
              className="truncate"
              style={{
                fontSize: isSector ? '10.5px' : '8.5px',
                lineHeight: 1,
                textShadow: '0 1px 2px rgba(0,0,0,0.85)',
              }}
            >
              {nodeData.name}
            </span>
            {!isSector && nodeWidth >= 28 && (
              <svg
                className="pointer-events-none absolute -bottom-[4px] left-1.5 z-[3]"
                width="10"
                height="5"
                viewBox="0 0 10 5"
                aria-hidden="true"
              >
                <polygon points="0,0 5,5 10,0" fill={industryHeaderBg} />
                <polyline
                  points="0,0 5,5 10,0"
                  fill="none"
                  stroke="#12151c"
                  strokeWidth="1"
                  strokeLinejoin="miter"
                />
              </svg>
            )}
          </div>
        )}
      </div>
    );
  });
});

const TEXT_SHADOW = '0 1px 2px rgba(0,0,0,0.85), 0 0 1px rgba(0,0,0,0.9)';

function lodVisibility(minScale: number, maxScale: number, scale: number) {
  return minScale <= scale && scale < maxScale ? 'visible' : 'hidden';
}

/**
 * One self-contained layer per disclosure tier. Layers are absolutely
 * positioned over the tile and toggled with `visibility`, so a hidden tier
 * (e.g. the logo row of `large`) never reserves height in the visible one.
 */
function TileTierLayer({
  plan,
  item,
  logoTicker,
  changeText,
  priceText,
}: {
  plan: TileTierPlan;
  item: HeatmapData;
  logoTicker: string | null;
  changeText: string;
  priceText: string | null;
}) {
  const { tier, minScale, maxScale, typography } = plan;
  const showChange = tier !== 'small' && typography.change > 0;
  const showPrice = tier === 'large' && !!priceText && typography.price > 0;
  const showLogo = tier === 'large' && !!logoTicker && typography.logo > 0;
  return (
    <div
      data-hm-lod-min={minScale}
      data-hm-lod-max={Number.isFinite(maxScale) ? maxScale : undefined}
      data-hm-tier={tier}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center overflow-hidden text-center"
      style={{ padding: typography.padding, visibility: lodVisibility(minScale, maxScale, 1) }}
    >
      {showLogo && (
        <CompanyLogo
          ticker={logoTicker as string}
          label={item.shortName}
          size={typography.logo}
          className="block"
        />
      )}
      <strong
        className="block whitespace-nowrap font-bold text-white"
        style={{
          fontSize: typography.ticker,
          lineHeight: LINE_HEIGHT,
          letterSpacing: `${TICKER_TRACKING_EM}em`,
          marginTop: showLogo ? typography.ticker * TILE_LOGO_GAP_EM : 0,
          textShadow: TEXT_SHADOW,
        }}
      >
        {item.name}
      </strong>
      {showChange && (
        <span
          className="block whitespace-nowrap font-bold tabular-nums text-white"
          style={{
            fontSize: typography.change,
            lineHeight: LINE_HEIGHT,
            marginTop: typography.gap,
            textShadow: TEXT_SHADOW,
          }}
        >
          {changeText}
        </span>
      )}
      {showPrice && (
        <span
          className="block whitespace-nowrap font-medium tabular-nums text-white/85"
          style={{
            fontSize: typography.price,
            lineHeight: LINE_HEIGHT,
            marginTop: typography.gap,
            textShadow: TEXT_SHADOW,
          }}
        >
          {priceText}
        </span>
      )}
    </div>
  );
}

export const LeafTiles = memo(function LeafTiles({ leafEntries, zoom = 1 }: { leafEntries: LeafEntry[]; zoom?: number }) {
  return leafEntries.map(({ leaf, parentId, renderId }) => {
    const item = leaf.data as HeatmapData;
    const cellWidth = leaf.x1 - leaf.x0;
    const cellHeight = leaf.y1 - leaf.y0;
    if (cellWidth <= 0 || cellHeight <= 0) return null;
    const tileScale = minimumScaleForSize(cellWidth, cellHeight, MIN_TILE_SIZE, MIN_TILE_SIZE, MIN_TILE_SIZE ** 2);
    const change = item.changePercent || 0;
    const changeText = formatTileChange(item.changePercent);
    const priceText = item.aggregateCount ? null : formatTilePrice(item.price);
    const fallbackLogoTicker = LOGO_INSTRUMENT_TYPES.has((item.instrumentType || '').toLowerCase())
      ? item.name
      : null;
    const logoTicker = item.aggregateCount ? null : item.logoTicker || fallbackLogoTicker;
    // Tiles below the readability floor at every zoom level render colour only;
    // their details stay reachable through hover/focus (stock details panel).
    const plans = planTileTiers(cellWidth, cellHeight, {
      ticker: item.name,
      change: changeText,
      price: priceText,
      hasLogo: !!logoTicker,
    });
    const tierAtRest = tierAtScale(plans, 1);
    return (
      <div
        key={renderId}
        data-hm-leaf-id={renderId}
        data-hm-parent-id={parentId}
        data-hm-lod-min={tileScale}
        data-hm-tier-at-rest={tierAtRest}
        role="button"
        tabIndex={0}
        aria-label={`${item.aggregateCount ? item.shortName : `${item.name}, ${item.shortName || ''}`}. Price ${item.price ?? 'unavailable'}. Daily change ${change >= 0 ? 'plus ' : 'minus '}${Math.abs(change).toFixed(2)} percent.`}
        className={`absolute z-[2] overflow-hidden text-white outline-none transition-[filter,box-shadow] duration-75 hover:brightness-125 hover:shadow-[inset_0_0_0_1px_rgba(255,255,255,0.4)] focus-visible:z-10 focus-visible:ring-2 focus-visible:ring-white ${zoom > 1 ? 'cursor-grab' : 'cursor-crosshair'}`}
        style={{
          left: leaf.x0,
          top: leaf.y0,
          width: cellWidth,
          height: cellHeight,
          backgroundColor: getColorForChange(item.changePercent),
          border: '1px solid #12151c',
          boxSizing: 'border-box',
          visibility: tileScale <= 1 ? 'visible' : 'hidden',
        }}
      >
        {plans.map((plan) => (
          <TileTierLayer
            key={plan.tier}
            plan={plan}
            item={item}
            logoTicker={logoTicker}
            changeText={changeText}
            priceText={priceText}
          />
        ))}
      </div>
    );
  });
});
