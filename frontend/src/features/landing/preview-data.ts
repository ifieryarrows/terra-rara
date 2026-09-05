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

export function linePath(values: number[], width = 640, height = 250) {
  const min = Math.min(...values) - 2;
  const max = Math.max(...values) + 2;
  return values.map((value, index) => `${index ? 'L' : 'M'}${(index / (values.length - 1) * width).toFixed(2)},${(height - (value - min) / (max - min) * height).toFixed(2)}`).join(' ');
}
