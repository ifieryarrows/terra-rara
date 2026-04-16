import { treemap, treemapSquarify, hierarchy, HierarchyRectangularNode } from 'd3-hierarchy';

export interface HeatmapData {
  name: string;
  shortName?: string;
  price?: number;
  changePercent?: number;
  marketCap?: number;
  sector?: string;
  industry?: string;
  isSP500?: boolean;
  isNasdaq100?: boolean;
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

export interface TreemapLayoutNode extends HierarchyRectangularNode<HeatmapNode | HeatmapData> {
  // Add any custom layout properties if needed
}

export function buildTreemapLayout(
  data: HeatmapNode,
  width: number,
  height: number
): HierarchyRectangularNode<HeatmapNode | HeatmapData> {
  // Create D3 hierarchy
  const root = hierarchy<HeatmapNode | HeatmapData>(data)
    .sum((d: any) => {
      // Only leaf nodes should have marketCap to sum
      return d.children ? 0 : (d.marketCap || 0);
    })
    .sort((a, b) => (b.value || 0) - (a.value || 0));

  // Configure treemap layout
  const layout = treemap<HeatmapNode | HeatmapData>()
    .size([width, height])
    .paddingInner(1) // Thin cell spacing
    .paddingOuter(1)
    .paddingTop(24)  // Space for sector/industry headers
    .round(true)
    .tile(treemapSquarify.ratio(1)); // Squarified with standard ratio

  // Generate layout
  layout(root);

  return root as HierarchyRectangularNode<HeatmapNode | HeatmapData>;
}
