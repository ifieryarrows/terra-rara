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

export interface HeatmapNode {
  name: string;
  children?: (HeatmapNode | HeatmapData)[];
  _meta?: {
    is_stale: boolean;
    refresh_in_progress: boolean;
    last_updated_at: string | null;
    next_refresh_at: string | null;
    source_delay_minutes: number;
  };
}

export interface TreemapLayoutNode extends HierarchyRectangularNode<HeatmapNode | HeatmapData> {}

export function buildTreemapLayout(
  data: HeatmapNode,
  width: number,
  height: number
): HierarchyRectangularNode<HeatmapNode | HeatmapData> {
  const root = hierarchy<HeatmapNode | HeatmapData>(data)
    .sum((d: any) => {
      // Only leaf nodes carry a weight value for squarification
      return d.children ? 0 : (d.weight || 1);
    })
    .sort((a, b) => (b.value || 0) - (a.value || 0));

  const layout = treemap<HeatmapNode | HeatmapData>()
    .size([width, height])
    .paddingInner(1)
    .paddingOuter(1)
    .paddingTop(24)
    .round(true)
    .tile(treemapSquarify.ratio(1));

  layout(root);

  return root as HierarchyRectangularNode<HeatmapNode | HeatmapData>;
}
