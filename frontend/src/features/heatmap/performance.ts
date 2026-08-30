export interface HeatmapMetricStore {
  layoutMs: number[];
  commits: { phase: string; duration: number; at: number }[];
  longTasks: number[];
  resizeLayouts: number;
}

declare global {
  interface Window {
    __COPPERMIND_HEATMAP_METRICS__?: HeatmapMetricStore;
  }
}

export function heatmapMetrics(): HeatmapMetricStore {
  const fallback: HeatmapMetricStore = { layoutMs: [], commits: [], longTasks: [], resizeLayouts: 0 };
  if (typeof window === 'undefined') return fallback;
  return (window.__COPPERMIND_HEATMAP_METRICS__ ||= fallback);
}

export function recordLayout(duration: number) {
  const store = heatmapMetrics();
  store.layoutMs.push(duration);
  if (store.layoutMs.length > 200) store.layoutMs.shift();
  if (typeof document !== 'undefined') {
    const sorted = [...store.layoutMs].sort((a, b) => a - b);
    const p95 = sorted[Math.max(0, Math.ceil(sorted.length * 0.95) - 1)] || 0;
    document.documentElement.dataset.heatmapLayoutP95 = p95.toFixed(3);
    document.documentElement.dataset.heatmapLayoutSamples = String(store.layoutMs.length);
  }
}

export function recordCommit(phase: string, duration: number) {
  const store = heatmapMetrics();
  store.commits.push({ phase, duration, at: performance.now() });
  if (store.commits.length > 200) store.commits.shift();
  if (typeof document !== 'undefined') {
    document.documentElement.dataset.heatmapCommitCount = String(store.commits.length);
  }
}

export function recordLongTask(duration: number) {
  const store = heatmapMetrics();
  store.longTasks.push(duration);
  if (store.longTasks.length > 200) store.longTasks.shift();
  if (typeof document !== 'undefined') {
    document.documentElement.dataset.heatmapLongTaskMax = String(Math.max(0, ...store.longTasks).toFixed(3));
  }
}
