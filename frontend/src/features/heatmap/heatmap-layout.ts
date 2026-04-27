import { treemap, treemapSquarify, hierarchy, HierarchyRectangularNode } from 'd3-hierarchy';

export interface HeatmapData {
  name: string;
  shortName?: string;
  price?: number;
  changePercent?: number;
  /** Sizing weight: Market Cap, Dollar Volume, or Equal Weight (1.0) */
  weight?: number;
  /** Human-readable label for the weight metric used */
  weightLabel?: string;
  /** Top-level project group (e.g. "Copper Miners", "Battery Metals") */
  group?: string;
  /** Second-level project subgroup (e.g. "Major Producers", "Lithium") */
  subgroup?: string;
  /** Raw CSV category (e.g. "miner_major") */
  category?: string;
  /** Raw CSV source_tag (e.g. "copper_core") */
  sourceTag?: string;
}

export interface HeatmapMeta {
  is_stale: boolean;
  refresh_in_progress: boolean;
  last_updated_at: string | null;
  next_refresh_at: string | null;
  source_delay_minutes: number;
  payload_count?: number;
  refresh_error?: string | null;
  cache_state?: 'fresh' | 'stale' | 'refreshing' | 'empty';
}

export interface HeatmapNode {
  name: string;
  children?: (HeatmapNode | HeatmapData)[];
  _meta?: HeatmapMeta;
}

export interface TreemapLayoutNode extends HierarchyRectangularNode<HeatmapNode | HeatmapData> {}

export function buildTreemapLayout(
  data: HeatmapNode,
  width: number,
  height: number,
  paddingTop = 24,
): HierarchyRectangularNode<HeatmapNode | HeatmapData> {
  const root = hierarchy<HeatmapNode | HeatmapData>(data)
    .sum((d: any) => {
      return d.children ? 0 : (d.weight || 1);
    })
    .sort((a, b) => (b.value || 0) - (a.value || 0));

  const layout = treemap<HeatmapNode | HeatmapData>()
    .size([Math.max(1, width), Math.max(1, height)])
    .paddingInner(1)
    .paddingOuter(1)
    .paddingTop(paddingTop)
    .round(true)
    .tile(treemapSquarify.ratio(1));

  layout(root);

  return root as HierarchyRectangularNode<HeatmapNode | HeatmapData>;
}

/**
 * Extract the list of leaf HeatmapData entries belonging to a given group or
 * subgroup. Used by the category inspector side panel.
 */
export function leavesForCategory(
  root: HeatmapNode,
  categoryName: string,
): HeatmapData[] {
  const out: HeatmapData[] = [];

  const walk = (node: any, inside: boolean) => {
    if (!node) return;
    const isMatch = inside || node.name === categoryName;
    if (node.children && node.children.length) {
      node.children.forEach((c: any) => walk(c, isMatch));
    } else if (isMatch) {
      out.push(node as HeatmapData);
    }
  };

  walk(root, false);
  return out;
}
