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

export type DetailLevel = 'color' | 'ticker' | 'change' | 'logo' | 'price';

export interface StockTextSizes {
  ticker: number;
  change: number;
}

const clamp = (value: number, minimum: number, maximum: number) => (
  Math.max(minimum, Math.min(maximum, value))
);

/** Finviz-inspired type scaling that remains bounded at every LOD level. */
export function stockTextSizes(width: number, height: number, level: DetailLevel): StockTextSizes {
  if (level === 'color') return { ticker: 0, change: 0 };
  if (level === 'price') {
    const ticker = clamp(Math.min(width / 6.5, height / 6), 16, 44);
    return { ticker, change: clamp(ticker * 0.64, 11, 28) };
  }
  if (level === 'logo') {
    const ticker = clamp(Math.min(width / 5.4, height / 4.8), 10.5, 20);
    return { ticker, change: clamp(ticker * 0.68, 8.5, 14) };
  }
  if (level === 'change') {
    const ticker = clamp(Math.min(width / 4.8, height / 3.2), 9.5, 16);
    return { ticker, change: clamp(ticker * 0.7, 8, 12) };
  }
  return { ticker: clamp(Math.min(width / 4.6, height / 2.8), 8.5, 14), change: 0 };
}

export function detailLevel(width: number, height: number): DetailLevel {
  const area = width * height;
  if (width < 24 || height < 18 || area < 520) return 'color';
  if (width < 44 || height < 25 || area < 1_250) return 'ticker';
  if (width < 66 || height < 42 || area < 2_800) return 'change';
  if (width < 100 || height < 72 || area < 6_800) return 'logo';
  return 'price';
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
