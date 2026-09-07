/** One scroll authority. Reading windows never overlap, even when scroll stops. */
export const storyPhases = [
  { id: 'market', label: 'Market', start: 0, end: .3 },
  { id: 'news', label: 'Context', start: .3, end: .7 },
  { id: 'forecast', label: 'Possibilities', start: .7, end: 1 },
] as const;
export const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
export const chapterAt = (p: number) => p < .3 ? 0 : p < .7 ? 1 : 2;
export function sceneVisibility(p: number, index: number) {
  if (index === 0) return 1 - clamp01((p - .24) / .05);
  if (index === 1) return Math.min(clamp01((p - .31) / .05), 1 - clamp01((p - .64) / .05));
  return clamp01((p - .71) / .05);
}
export function sceneProgress(p: number, index: number) {
  const starts = [0, .31, .71];
  const ends = [.12, .43, .83];
  return clamp01((p - starts[index]) / (ends[index] - starts[index]));
}
