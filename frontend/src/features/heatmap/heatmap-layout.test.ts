import { describe, expect, it } from 'vitest';
import {
  aggregateTinyLeaves,
  buildTreemapLayout,
  categoryHeaderPadding,
  categoryStats,
  compressLeafWeights,
  estimateTextWidthEm,
  formatTileChange,
  formatTilePrice,
  HEATMAP_MAX_ZOOM,
  leavesForCategory,
  planTileTiers,
  READABLE_SECONDARY_PX,
  TICKER_TRACKING_EM,
  tierAtScale,
  type HeatmapData,
  type HeatmapNode,
  type TileContent,
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

  it('keeps descendants inside tiny category bounds at every supported viewport', () => {
    const data: HeatmapNode = {
      id: 'root',
      name: 'Root',
      children: Array.from({ length: 18 }, (_, groupIndex) => ({
        id: `group-${groupIndex}`,
        name: `Group ${groupIndex}`,
        children: Array.from({ length: 5 }, (_, subgroupIndex) => ({
          id: `group-${groupIndex}-subgroup-${subgroupIndex}`,
          name: `Subgroup ${subgroupIndex}`,
          children: Array.from({ length: 3 }, (_, leafIndex) => leaf(
            `${groupIndex}-${subgroupIndex}-${leafIndex}`,
            groupIndex === 0 ? 10_000 : 1 + leafIndex,
          )),
        })),
      })),
    };

    for (const [width, height] of [[390, 560], [768, 560], [1_536, 820], [6_144, 3_280]]) {
      const layout = buildTreemapLayout(data, width, height);
      for (const node of layout.descendants()) {
        if (!node.parent) continue;
        expect(node.x0).toBeGreaterThanOrEqual(node.parent.x0);
        expect(node.y0).toBeGreaterThanOrEqual(node.parent.y0);
        expect(node.x1).toBeLessThanOrEqual(node.parent.x1);
        expect(node.y1).toBeLessThanOrEqual(node.parent.y1);
        expect(node.x1).toBeGreaterThanOrEqual(node.x0);
        expect(node.y1).toBeGreaterThanOrEqual(node.y0);
      }
    }

    expect(categoryHeaderPadding(1, 100, 20)).toBe(1);
    expect(categoryHeaderPadding(2, 100, 16)).toBe(1);
    expect(categoryHeaderPadding(1, 100, 40)).toBe(22);
    expect(categoryHeaderPadding(2, 100, 40)).toBe(16);
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

  it('computes category breadth', () => {
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
});

describe('tile progressive disclosure', () => {
  const content = (ticker = 'NVDA', hasLogo = true): TileContent => ({
    ticker,
    change: formatTileChange(-12.34),
    price: formatTilePrice(1_234.5),
    hasLogo,
  });

  it('assigns large / medium / small / micro tiers by the computed pixel size at rest', () => {
    expect(tierAtScale(planTileTiers(240, 200, content()), 1)).toBe('large');
    expect(tierAtScale(planTileTiers(90, 60, content()), 1)).toBe('medium');
    expect(tierAtScale(planTileTiers(48, 18, content()), 1)).toBe('small');
    expect(tierAtScale(planTileTiers(14, 10, content()), 1)).toBe('micro');
  });

  it('upgrades a tile to richer tiers as the camera zooms in', () => {
    const plans = planTileTiers(48, 30, content());
    const tiers = [1, 1.5, 2, 3, 4].map((scale) => tierAtScale(plans, scale));
    const order = ['micro', 'small', 'medium', 'large'];
    for (let index = 1; index < tiers.length; index += 1) {
      expect(order.indexOf(tiers[index])).toBeGreaterThanOrEqual(order.indexOf(tiers[index - 1]));
    }
    expect(tiers[tiers.length - 1]).toBe('large');
  });

  it('keeps tier ranges contiguous, non-overlapping and within the zoom range', () => {
    for (const [width, height] of [[6, 6], [20, 14], [40, 24], [70, 45], [120, 80], [400, 300], [600, 40], [30, 300]]) {
      const plans = planTileTiers(width, height, content('GOOGL'));
      plans.forEach((plan, index) => {
        expect(plan.minScale).toBeLessThan(plan.maxScale);
        expect(plan.minScale).toBeLessThanOrEqual(HEATMAP_MAX_ZOOM);
        if (index > 0) expect(plan.minScale).toBe(plans[index - 1].maxScale);
      });
      for (const scale of [1, 1.7, 2.6, 4]) {
        expect(plans.filter((plan) => plan.minScale <= scale && scale < plan.maxScale).length).toBeLessThanOrEqual(1);
      }
    }
  });

  it('never lets visible content overflow its cell or render below the readability floor', () => {
    for (const [width, height] of [[24, 18], [44, 25], [66, 42], [100, 72], [160, 110], [382, 389], [900, 120], [50, 400]]) {
      for (const ticker of ['A', 'NVDA', 'GOOGL', 'BRK-B', 'WWWWW']) {
        for (const plan of planTileTiers(width, height, content(ticker))) {
          const { typography: t, tier, minScale } = plan;
          const inner = { width: width - t.padding * 2, height: height - t.padding * 2 };
          const lines = [estimateTextWidthEm(ticker, TICKER_TRACKING_EM) * t.ticker];
          let stack = t.ticker * 1.1;
          if (tier !== 'small') {
            lines.push(estimateTextWidthEm(formatTileChange(-12.34)) * t.change);
            stack += t.gap + t.change * 1.1;
          }
          if (tier === 'large') {
            lines.push(estimateTextWidthEm(formatTilePrice(1_234.5) || '') * t.price, t.logo);
            stack += t.gap + t.price * 1.1 + t.logo + t.ticker * 0.22;
          }
          expect(Math.max(...lines)).toBeLessThanOrEqual(inner.width + 1e-6);
          expect(stack).toBeLessThanOrEqual(inner.height + 1e-6);
          const visibleFrom = Math.max(1, minScale);
          expect(t.ticker * visibleFrom).toBeGreaterThanOrEqual(tier === 'small' ? 9 - 1e-6 : 11 - 1e-6);
          if (tier !== 'small') expect(t.change * visibleFrom).toBeGreaterThanOrEqual(READABLE_SECONDARY_PX - 1e-6);
          if (tier === 'large') expect(t.price * visibleFrom).toBeGreaterThanOrEqual(READABLE_SECONDARY_PX - 1e-6);
        }
      }
    }
  });

  it('drops the logo outside the large tier and skips large when it adds nothing', () => {
    const withLogo = planTileTiers(400, 300, content());
    expect(withLogo.find((plan) => plan.tier === 'medium')?.typography.logo ?? 0).toBe(0);
    expect(withLogo.find((plan) => plan.tier === 'large')?.typography.logo).toBeGreaterThan(0);
    const aggregate = planTileTiers(400, 300, { ticker: '+12', change: '+0.50%', price: null, hasLogo: false });
    expect(aggregate.map((plan) => plan.tier)).not.toContain('large');
    expect(tierAtScale(aggregate, 1)).toBe('medium');
  });

  it('scales type with the cell while keeping the ticker dominant', () => {
    const medium = planTileTiers(130, 100, content()).find((plan) => plan.tier === 'large');
    const large = planTileTiers(382, 389, content()).find((plan) => plan.tier === 'large');
    expect(large && medium).toBeTruthy();
    expect(large!.typography.ticker).toBeGreaterThan(medium!.typography.ticker);
    expect(large!.typography.ticker).toBeGreaterThan(large!.typography.change);
    expect(large!.typography.change).toBeGreaterThan(large!.typography.price);
    expect(large!.typography.ticker).toBeLessThanOrEqual(44);
    expect(large!.typography.logo).toBeLessThanOrEqual(44);
  });

  it('formats tile values compactly', () => {
    expect(formatTileChange(2)).toBe('+2.00%');
    expect(formatTileChange(-0.5)).toBe('-0.50%');
    expect(formatTileChange(undefined)).toBe('0.00%');
    expect(formatTilePrice(1234.5)).toBe('$1,234.50');
    expect(formatTilePrice(0.1234)).toBe('$0.1234');
    expect(formatTilePrice(undefined)).toBeNull();
  });
});

describe('heatmap layout performance', () => {
  it('keeps real-size and 1,000-instrument layout p95 within the performance budget', () => {
    const realUniverse = tree(Array.from({ length: 194 }, (_, index) => leaf(`R${index}`, 1 + (index % 40))));
    const data = tree(Array.from({ length: 1_000 }, (_, index) => leaf(`S${index}`, 1 + (index % 40))));
    const measure = (input: HeatmapNode) => {
      const expectedLeaves = leavesForCategory(input, 'root').length;
      const batchSize = 3;
      const sampleCount = 20;
      // Warm up D3's hierarchy/tile path before sampling so JIT compilation and
      // first-use allocations do not become a false layout regression.
      for (let index = 0; index < 4; index += 1) buildTreemapLayout(input, 1536, 820);
      return Array.from({ length: sampleCount }, () => {
        let layout: ReturnType<typeof buildTreemapLayout> | undefined;
        const started = performance.now();
        for (let index = 0; index < batchSize; index += 1) layout = buildTreemapLayout(input, 1536, 820);
        const elapsed = (performance.now() - started) / batchSize;
        expect(layout?.leaves()).toHaveLength(expectedLeaves);
        return elapsed;
      }).sort((a, b) => a - b)[Math.ceil(sampleCount * 0.95) - 1];
    };
    const realP95 = measure(realUniverse);
    const largeP95 = measure(data);
    console.info(`[heatmap-benchmark] layout p95 real=${realP95.toFixed(2)}ms large=${largeP95.toFixed(2)}ms`);
    expect(realP95).toBeLessThanOrEqual(8);
    expect(largeP95).toBeLessThanOrEqual(12);
  });
});
