import { describe, expect, it } from 'vitest';
import {
  aggregateTinyLeaves,
  buildTreemapLayout,
  categoryStats,
  compressLeafWeights,
  detailLevel,
  leavesForCategory,
  stockTextSizes,
  type HeatmapData,
  type HeatmapNode,
} from './heatmap-layout';

const leaf = (name: string, weight: number): HeatmapData => ({
  id: `leaf-${name}`, name, weight, price: 10, changePercent: weight % 3 - 1,
});

const tree = (leaves: HeatmapData[]): HeatmapNode => ({
  id: 'root', name: 'Root', children: [{
    id: 'sector', name: 'Sector', children: [{ id: 'industry', name: 'Industry', children: leaves }],
  }],
});

describe('heatmap layout', () => {
  it('is deterministic and keyed by stable IDs', () => {
    const data = tree([leaf('A', 50), leaf('B', 30), leaf('C', 20)]);
    const first = buildTreemapLayout(data, 800, 500).leaves().map((node) => [node.data.id, node.x0, node.y0, node.x1, node.y1]);
    const second = buildTreemapLayout(data, 800, 500).leaves().map((node) => [node.data.id, node.x0, node.y0, node.x1, node.y1]);
    expect(second).toEqual(first);
  });

  it('aggregates projected tiny leaves while preserving total weight and panel access', () => {
    const original = tree([leaf('BIG', 1_000), ...Array.from({ length: 12 }, (_, index) => leaf(`T${index}`, 1))]);
    const before = leavesForCategory(original, 'industry');
    const aggregated = aggregateTinyLeaves(original, 500, 300, 200);
    const after = (aggregated.children?.[0] as HeatmapNode).children?.[0] as HeatmapNode;
    const aggregate = after.children?.find((item) => (item as HeatmapData).aggregateCount) as HeatmapData;
    expect(before).toHaveLength(13);
    expect(aggregate.aggregateCount).toBe(12);
    expect(aggregate.weight).toBe(12);
    expect(leavesForCategory(original, 'stale-layout-id', 'Industry')).toHaveLength(13);
  });

  it('uses progressive detail thresholds and computes breadth', () => {
    expect(detailLevel(12, 12)).toBe('color');
    expect(detailLevel(36, 22)).toBe('ticker');
    expect(detailLevel(60, 36)).toBe('change');
    expect(detailLevel(90, 55)).toBe('logo');
    expect(detailLevel(140, 90)).toBe('price');
    expect(categoryStats([{ name: 'A', weight: 1, changePercent: 2 }, { name: 'B', weight: 1, changePercent: -1 }])).toMatchObject({
      averageChange: 0.5, advancing: 1, declining: 1,
    });
  });

  it('reduces leaf weight differences by exactly ten percent without changing the total', () => {
    const compressed = compressLeafWeights(tree([leaf('NVDA', 100), leaf('SMALL', 10)]), 0.1);
    const leaves = leavesForCategory(compressed, 'industry');
    const weights = leaves.map((item) => item.weight || 0);
    expect(weights).toEqual([95.5, 14.5]);
    expect(weights[0] + weights[1]).toBe(110);
    expect(weights[0] - weights[1]).toBe(81);
  });

  it('scales ticker and change type with cell size while keeping ticker dominant', () => {
    const medium = stockTextSizes(100, 72, 'price');
    const large = stockTextSizes(382, 389, 'price');
    expect(large.ticker).toBeGreaterThan(medium.ticker);
    expect(large.change).toBeGreaterThan(medium.change);
    expect(large.ticker).toBeGreaterThan(large.change);
    expect(large).toEqual({ ticker: 44, change: 28 });
    expect(stockTextSizes(20, 16, 'color')).toEqual({ ticker: 0, change: 0 });
  });

  it('keeps real-size and 1,000-instrument layout p95 within the performance budget', () => {
    const realUniverse = tree(Array.from({ length: 194 }, (_, index) => leaf(`R${index}`, 1 + (index % 40))));
    const data = tree(Array.from({ length: 1_000 }, (_, index) => leaf(`S${index}`, 1 + (index % 40))));
    const measure = (input: HeatmapNode) => Array.from({ length: 30 }, () => {
      const started = performance.now();
      const layout = buildTreemapLayout(input, 1536, 820);
      expect(layout.leaves()).toHaveLength(leavesForCategory(input, 'root').length);
      return performance.now() - started;
    }).sort((a, b) => a - b)[Math.ceil(30 * 0.95) - 1];
    const realP95 = measure(realUniverse);
    const largeP95 = measure(data);
    console.info(`[heatmap-benchmark] layout p95 real=${realP95.toFixed(2)}ms large=${largeP95.toFixed(2)}ms`);
    expect(realP95).toBeLessThanOrEqual(8);
    expect(largeP95).toBeLessThanOrEqual(12);
  });
});
