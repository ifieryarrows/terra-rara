import {
  hierarchy,
  treemap,
  treemapResquarify,
  type HierarchyRectangularNode,
} from 'd3-hierarchy';

export interface HeatmapData {
  id?: string;
  name: string;
  shortName?: string;
  price?: number;
  changePercent?: number;
  weight?: number;
  weightLabel?: string;
  group?: string;
  subgroup?: string;
  category?: string;
  sourceTag?: string;
  instrumentType?: string;
  sector?: string | null;
  industry?: string | null;
  exchange?: string | null;
  logoTicker?: string | null;
  sparkline?: number[] | null;
  asOf?: string | null;
  aggregateCount?: number;
  aggregateMembers?: HeatmapData[];
}

export interface HeatmapMeta {
  view?: 'market' | 'themes';
  is_stale: boolean;
  refresh_in_progress: boolean;
  last_updated_at: string | null;
  next_refresh_at: string | null;
  source_delay_minutes: number;
  payload_count?: number;
  refresh_error?: string | null;
  cache_state?: 'fresh' | 'stale' | 'refreshing' | 'empty';
  cache_age_seconds?: number;
}

export interface HeatmapNode {
  id?: string;
  name: string;
  children?: (HeatmapNode | HeatmapData)[];
  _meta?: HeatmapMeta;
}

export type LayoutNode = HierarchyRectangularNode<HeatmapNode | HeatmapData>;

const safeLeafWeight = (leaf: HeatmapData) => Math.max(0.0001, leaf.weight || 1);

/**
 * Pull every leaf weight toward the visible-universe mean. A 10% compression
 * preserves the total weight while reducing every pairwise weight gap by
 * exactly 10%, so dominant names remain dominant without overwhelming the map.
 */
export function compressLeafWeights(root: HeatmapNode, compression = 0.1): HeatmapNode {
  const leaves: HeatmapData[] = [];
  const collect = (node: HeatmapNode | HeatmapData) => {
    const children = 'children' in node ? node.children : undefined;
    if (children?.length) children.forEach(collect);
    else leaves.push(node as HeatmapData);
  };
  collect(root);
  if (!leaves.length) return root;

  const ratio = Math.max(0, Math.min(1, compression));
  const mean = leaves.reduce((sum, leaf) => sum + safeLeafWeight(leaf), 0) / leaves.length;
  const transform = (node: HeatmapNode | HeatmapData): HeatmapNode | HeatmapData => {
    const children = 'children' in node ? node.children : undefined;
    if (children?.length) return { ...node, children: children.map(transform) } as HeatmapNode;
    const leaf = node as HeatmapData;
    return { ...leaf, weight: safeLeafWeight(leaf) * (1 - ratio) + mean * ratio };
  };
  return transform(root) as HeatmapNode;
}

export function createTreemapHierarchy(data: HeatmapNode): LayoutNode {
  return hierarchy<HeatmapNode | HeatmapData>(data)
    .sum((node) => ('children' in node && node.children ? 0 : Math.max(0.0001, (node as HeatmapData).weight || 1)))
    .sort((a, b) => (b.value || 0) - (a.value || 0)) as LayoutNode;
}

const HEADER_WIDTH_THRESHOLD = 46;
const MINIMUM_HEADER_CONTENT_HEIGHT = 4;

/**
 * Reserve a full category header only when the category can also contain a
 * visible child row. D3 applies padding before laying out descendants; a fixed
 * 16/22px top padding on a shorter node moves its descendants beyond the
 * parent's bottom edge.
 */
export function categoryHeaderPadding(depth: number, width: number, height: number): number {
  const target = depth === 1 ? 22 : depth === 2 ? 16 : 1;
  return (
    depth > 0
    && depth < 3
    && width > HEADER_WIDTH_THRESHOLD
    && height >= target + MINIMUM_HEADER_CONTENT_HEIGHT
  ) ? target : 1;
}

/** Mutates and reuses the hierarchy so resquarify preserves row topology on resize. */
export function layoutTreemap(root: LayoutNode, width: number, height: number): LayoutNode {
  treemap<HeatmapNode | HeatmapData>()
    .size([Math.max(1, width), Math.max(1, height)])
    // A single layout pixel is the shared boundary between adjacent cells.
    // Cells do not draw their own borders, avoiding the old 1 + 1 + 1px
    // stacked gutter while allowing the category backing layer to recolour
    // every internal boundary on hover.
    .paddingInner(1)
    .paddingOuter(1)
    .paddingTop((node) => categoryHeaderPadding(
      node.depth,
      Math.max(0, node.x1 - node.x0),
      Math.max(0, node.y1 - node.y0),
    ))
    .round(true)
    .tile(treemapResquarify)(root);
  return root;
}

export function buildTreemapLayout(data: HeatmapNode, width: number, height: number): LayoutNode {
  return layoutTreemap(createTreemapHierarchy(data), width, height);
}

export function leavesForCategory(root: HeatmapNode, categoryId: string, categoryName?: string): HeatmapData[] {
  const find = (node: HeatmapNode | HeatmapData, byName = false): HeatmapNode | HeatmapData | null => {
    const children = 'children' in node ? node.children : undefined;
    if (node.id === categoryId || (byName && children?.length && node.name === categoryName)) return node;
    for (const child of children || []) {
      const match = find(child, byName);
      if (match) return match;
    }
    return null;
  };
  const category = find(root) || (categoryName ? find(root, true) : null);
  if (!category) return [];
  const output: HeatmapData[] = [];
  const walk = (node: HeatmapNode | HeatmapData) => {
    const children = 'children' in node ? node.children : undefined;
    if (children?.length) children.forEach(walk);
    else output.push(node as HeatmapData);
  };
  walk(category);
  return output;
}

/* ------------------------------------------------------------------------- *
 * Progressive disclosure (tile level-of-detail)
 *
 * Every leaf is planned once, in layout (base) pixels. The camera only applies
 * a uniform transform, so a tier that fits geometrically at scale 1 fits at
 * every scale; the only scale-dependent question is whether the transformed
 * glyphs are readable. Each tier therefore gets the exact camera scale at
 * which its smallest line reaches the readability floor:
 *
 *   micro  -> colour only (hover/focus still opens the stock details)
 *   small  -> ticker only, compact type
 *   medium -> ticker + daily change (logo dropped to save vertical space)
 *   large  -> logo + ticker + daily change + price
 *
 * Exactly one tier is visible per tile at any scale, and every tier is laid
 * out in its own layer, so hidden content never reserves space or pushes
 * visible text out of the cell.
 * ------------------------------------------------------------------------- */

export const HEATMAP_MAX_ZOOM = 4;

export type TileTier = 'micro' | 'small' | 'medium' | 'large';
export type TextTier = Exclude<TileTier, 'micro'>;

export interface TileContent {
  ticker: string;
  change: string;
  price?: string | null;
  hasLogo: boolean;
}

export interface TileTypography {
  /** All values are layout pixels; the camera transform scales them uniformly. */
  ticker: number;
  change: number;
  price: number;
  logo: number;
  gap: number;
  padding: number;
}

export interface TileTierPlan {
  tier: TextTier;
  /** Camera scale from which the tier is readable (inclusive). */
  minScale: number;
  /** Camera scale from which the next tier takes over (exclusive). */
  maxScale: number;
  typography: TileTypography;
}

/** Smallest on-screen size (CSS px) at which secondary lines remain legible. */
export const READABLE_SECONDARY_PX = 8.5;

interface TierSpec {
  minTicker: number;
  maxTicker: number;
  changeRatio: number;
  priceRatio: number;
  logoRatio: number;
  maxLogo: number;
}

const TIER_SPECS: Record<TextTier, TierSpec> = {
  small: { minTicker: 9, maxTicker: 13, changeRatio: 0, priceRatio: 0, logoRatio: 0, maxLogo: 0 },
  medium: { minTicker: 11, maxTicker: 22, changeRatio: 0.76, priceRatio: 0, logoRatio: 0, maxLogo: 0 },
  large: { minTicker: 14, maxTicker: 44, changeRatio: 0.66, priceRatio: 0.56, logoRatio: 1.45, maxLogo: 44 },
};

export const TILE_LINE_HEIGHT = 1.1;
const LINE_HEIGHT = TILE_LINE_HEIGHT;
const LINE_GAP_EM = 0.1;
/** Gap between logo and ticker (em of ticker size). */
export const TILE_LOGO_GAP_EM = 0.22;
const LOGO_GAP_EM = TILE_LOGO_GAP_EM;
/** Guards against font fallback / rendering differences in the width estimate. */
const WIDTH_SAFETY = 1.08;
/** Letter spacing applied to the ticker line (em). */
export const TICKER_TRACKING_EM = -0.02;

/** Conservative advance widths (em) for a bold geometric sans such as Geist. */
function charAdvance(char: string): number {
  if (char >= '0' && char <= '9') return 0.62; // tabular figures
  if ('MW'.includes(char)) return 0.92;
  if ('IJ1'.includes(char)) return 0.4;
  if (char === '%') return 0.88;
  if (char === '$') return 0.64;
  if (char === '+') return 0.62;
  if (char === '-' || char === '\u2212') return 0.44;
  if (char === '.' || char === ',' || char === ':') return 0.3;
  if (char === ' ') return 0.28;
  if (char === '&') return 0.76;
  if (char >= 'A' && char <= 'Z') return 0.72;
  if (char >= 'a' && char <= 'z') return 0.58;
  return 0.72;
}

/** Estimated rendered width of `text` in em units (font-size = 1). */
export function estimateTextWidthEm(text: string, trackingEm = 0): number {
  let width = 0;
  for (const char of text) width += charAdvance(char) + trackingEm;
  return Math.max(0, width);
}

export function formatTileChange(change: number | undefined): string {
  const value = Number.isFinite(change) ? (change as number) : 0;
  return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
}

export function formatTilePrice(price: number | null | undefined): string | null {
  if (price == null || !Number.isFinite(price)) return null;
  const digits = Math.abs(price) < 1 ? 4 : 2;
  return `$${price.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
}

function cellPadding(width: number, height: number): number {
  return Math.max(1.5, Math.min(8, Math.min(width, height) * 0.07));
}

function planTier(tier: TextTier, width: number, height: number, content: TileContent): Omit<TileTierPlan, 'maxScale'> {
  const spec = TIER_SPECS[tier];
  const padding = cellPadding(width, height);
  const innerWidth = Math.max(0, width - padding * 2);
  const innerHeight = Math.max(0, height - padding * 2);
  const changeRatio = content.change ? spec.changeRatio : 0;
  const priceRatio = content.price ? spec.priceRatio : 0;
  const logoRatio = content.hasLogo ? spec.logoRatio : 0;

  const widthEm = Math.max(
    estimateTextWidthEm(content.ticker, TICKER_TRACKING_EM),
    changeRatio ? estimateTextWidthEm(content.change) * changeRatio : 0,
    priceRatio ? estimateTextWidthEm(content.price || '') * priceRatio : 0,
    logoRatio,
  ) * WIDTH_SAFETY;
  const textHeightEm = LINE_HEIGHT
    + (changeRatio ? LINE_GAP_EM + changeRatio * LINE_HEIGHT : 0)
    + (priceRatio ? LINE_GAP_EM + priceRatio * LINE_HEIGHT : 0);
  const logoHeightEm = logoRatio ? logoRatio + LOGO_GAP_EM : 0;

  const byWidth = widthEm > 0 ? innerWidth / widthEm : 0;
  let ticker = Math.min(spec.maxTicker, byWidth, innerHeight / (textHeightEm + logoHeightEm));
  let logo = logoRatio ? ticker * logoRatio : 0;
  if (logoRatio && logo > spec.maxLogo) {
    // The logo stops growing at its cap; give the remaining height to text.
    ticker = Math.min(spec.maxTicker, byWidth, (innerHeight - spec.maxLogo) / (textHeightEm + LOGO_GAP_EM));
    logo = Math.min(spec.maxLogo, ticker * logoRatio);
  }
  ticker = Math.max(0, ticker);

  const change = ticker * changeRatio;
  const price = ticker * priceRatio;
  const minScale = ticker > 0
    ? Math.max(
      spec.minTicker / ticker,
      changeRatio ? READABLE_SECONDARY_PX / change : 0,
      priceRatio ? READABLE_SECONDARY_PX / price : 0,
    )
    : Number.POSITIVE_INFINITY;

  return {
    tier,
    minScale,
    typography: { ticker, change, price, logo, gap: ticker * LINE_GAP_EM, padding },
  };
}

/**
 * Plan every readable tier of a tile. Tiers are returned in ascending order,
 * have monotonic, non-overlapping [minScale, maxScale) ranges, and tiers that
 * can never become readable within `maxZoom` are omitted.
 */
export function planTileTiers(
  width: number,
  height: number,
  content: TileContent,
  maxZoom = HEATMAP_MAX_ZOOM,
): TileTierPlan[] {
  if (!(width > 0) || !(height > 0) || !content.ticker) return [];
  const tiers: TextTier[] = ['small', 'medium'];
  // `large` only exists when it adds content beyond `medium`.
  if (content.hasLogo || content.price) tiers.push('large');

  let floor = 0;
  const planned = tiers.map((tier) => {
    const plan = planTier(tier, width, height, content);
    floor = Math.max(floor, plan.minScale);
    return { ...plan, minScale: floor };
  });

  return planned
    .map((plan, index) => ({
      ...plan,
      maxScale: planned[index + 1]?.minScale ?? Number.POSITIVE_INFINITY,
    }))
    .filter((plan) => plan.minScale <= maxZoom && plan.minScale < plan.maxScale);
}

/** Tier shown at a given camera scale; `micro` when no text is readable. */
export function tierAtScale(plans: TileTierPlan[], scale: number): TileTier {
  return plans.find((plan) => plan.minScale <= scale && scale < plan.maxScale)?.tier ?? 'micro';
}

/**
 * Collapse projected sub-pixel leaves per industry into a single +N cell.
 * The aggregate retains the exact summed weight; the original tree is kept by
 * the panel so every instrument remains discoverable.
 */
export function aggregateTinyLeaves(
  root: HeatmapNode,
  width: number,
  height: number,
  minimumArea = 34,
): HeatmapNode {
  const totalWeight = (root.children || []).reduce((total, group) => {
    const groupNode = group as HeatmapNode;
    return total + (groupNode.children || []).reduce((subtotal, subgroup) => {
      const subgroupNode = subgroup as HeatmapNode;
      return subtotal + (subgroupNode.children || []).reduce(
        (sum, leaf) => sum + Math.max(0.0001, (leaf as HeatmapData).weight || 1), 0,
      );
    }, 0);
  }, 0);
  if (!totalWeight || width <= 0 || height <= 0) return root;
  const availableArea = width * height;
  const children = (root.children || []).map((group) => {
    const groupNode = group as HeatmapNode;
    return {
      ...groupNode,
      children: (groupNode.children || []).map((subgroup) => {
        const subgroupNode = subgroup as HeatmapNode;
        const kept: HeatmapData[] = [];
        const tiny: HeatmapData[] = [];
        (subgroupNode.children || []).forEach((candidate) => {
          const leaf = candidate as HeatmapData;
          const projectedArea = ((leaf.weight || 1) / totalWeight) * availableArea;
          (projectedArea < minimumArea ? tiny : kept).push(leaf);
        });
        if (tiny.length < 2) return { ...subgroupNode, children: [...kept, ...tiny] };
        const weight = tiny.reduce((sum, leaf) => sum + (leaf.weight || 1), 0);
        const weightedChange = tiny.reduce(
          (sum, leaf) => sum + (leaf.changePercent || 0) * (leaf.weight || 1), 0,
        ) / Math.max(weight, 1);
        const aggregate: HeatmapData = {
          id: `${subgroupNode.id || subgroupNode.name}-aggregate`,
          name: `+${tiny.length}`,
          shortName: `${tiny.length} smaller instruments`,
          weight,
          changePercent: weightedChange,
          aggregateCount: tiny.length,
          aggregateMembers: tiny,
          group: tiny[0]?.group,
          subgroup: subgroupNode.name,
        };
        return { ...subgroupNode, children: [...kept, aggregate] };
      }),
    };
  });
  return { ...root, children };
}

export function categoryStats(leaves: HeatmapData[]) {
  if (!leaves.length) return { averageChange: 0, advancing: 0, declining: 0, unchanged: 0 };
  const totalWeight = leaves.reduce((sum, leaf) => sum + Math.max(1, leaf.weight || 1), 0);
  const averageChange = leaves.reduce(
    (sum, leaf) => sum + (leaf.changePercent || 0) * Math.max(1, leaf.weight || 1), 0,
  ) / totalWeight;
  return {
    averageChange,
    advancing: leaves.filter((leaf) => (leaf.changePercent || 0) > 0).length,
    declining: leaves.filter((leaf) => (leaf.changePercent || 0) < 0).length,
    unchanged: leaves.filter((leaf) => (leaf.changePercent || 0) === 0).length,
  };
}
