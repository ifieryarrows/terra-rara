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

const TEXT_SHADOW = '0 1px 2px rgba(0,0,0,.45)';

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
        className="block whitespace-nowrap font-bold"
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
          className="block whitespace-nowrap font-semibold tabular-nums"
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
          className="block whitespace-nowrap font-medium tabular-nums text-white/80"
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
        className={`absolute z-[2] overflow-hidden text-white outline-none transition-[filter] duration-75 hover:brightness-125 focus-visible:z-10 focus-visible:ring-2 focus-visible:ring-white ${zoom > 1 ? 'cursor-grab' : 'cursor-crosshair'}`}
        style={{
          left: leaf.x0,
          top: leaf.y0,
          width: cellWidth,
          height: cellHeight,
          backgroundColor: getColorForChange(item.changePercent),
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
