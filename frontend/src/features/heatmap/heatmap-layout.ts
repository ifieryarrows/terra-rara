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

export function createTreemapHierarchy(data: HeatmapNode): LayoutNode {
  return hierarchy<HeatmapNode | HeatmapData>(data)
    .sum((node) => ('children' in node && node.children ? 0 : Math.max(0.0001, (node as HeatmapData).weight || 1)))
    .sort((a, b) => (b.value || 0) - (a.value || 0)) as LayoutNode;
}

/** Mutates and reuses the hierarchy so resquarify preserves row topology on resize. */
export function layoutTreemap(root: LayoutNode, width: number, height: number): LayoutNode {
  treemap<HeatmapNode | HeatmapData>()
    .size([Math.max(1, width), Math.max(1, height)])
    .paddingInner(1)
    .paddingOuter(1)
    .paddingTop((node) => (node.depth === 1 ? 22 : node.depth === 2 ? 16 : 0))
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
