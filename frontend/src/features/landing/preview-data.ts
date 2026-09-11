/** Illustrative fixtures, not market quotes, model output, or backtest results. */
export const previewSeries = [100, 101, 100.4, 102.2, 101.5, 103, 102.6, 101.8, 104, 103.1, 105.4, 104.2, 106, 105.2, 104.8, 107, 106.3, 107.6, 106.8, 108];

export const previewMarkets = [
  { symbol: 'HG=F', name: 'Copper futures', change: '+1.24%', tone: 'up', size: 'large' },
  { symbol: 'FCX', name: 'Freeport-McMoRan', change: '+1.80%', tone: 'up', size: 'wide' },
  { symbol: 'BHP', name: 'BHP Group', change: '−0.62%', tone: 'down', size: '' },
  { symbol: 'RIO', name: 'Rio Tinto', change: '+0.91%', tone: 'up', size: '' },
  { symbol: 'SCCO', name: 'Southern Copper', change: '+0.45%', tone: 'up', size: '' },
  { symbol: 'GLD', name: 'Gold ETF', change: '−0.28%', tone: 'down', size: '' },
];

/**
 * Deterministic landing view-model adapted from the production /api/news payload.
 *
 * This is deliberately not a live article, a stored model result, or a claim
 * about FCX. These presentation fields are adapted from the real V2 contract so
 * the product story can explain what the dashboard actually exposes without
 * calling the API or inventing a new scoring system for marketing.
 */
export const newsIntelligencePreview = {
  symbol: 'FCX',
  company: 'Freeport-McMoRan',
  headline: 'Copper supply pressure comes into focus',
  description: 'An illustrative source line enters the workspace as readable editorial context.',
  publisher: 'DETERMINISTIC DEMO INPUT',
  horizon: '1–5D HG=F IMPACT',
  label: 'BULLISH',
  impactScoreLlm: 0.46,
  finalScore: 0.41,
  confidence: 0.68,
  relevance: 0.91,
  eventType: 'supply_disruption',
  finbert: { pos: 0.68, neu: 0.22, neg: 0.10 },
  reasoning: 'Supply tightening can support copper over the short horizon; demand and inventories remain the context to watch.',
} as const;

export function linePath(values: number[], width = 640, height = 250) {
  const min = Math.min(...values) - 2;
  const max = Math.max(...values) + 2;
  return values.map((value, index) => `${index ? 'L' : 'M'}${(index / (values.length - 1) * width).toFixed(2)},${(height - (value - min) / (max - min) * height).toFixed(2)}`).join(' ');
}
