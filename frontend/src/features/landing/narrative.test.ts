import { describe, expect, it } from 'vitest';
import { sceneIndex, sceneOpacity, tracePath } from './narrative';
import { evidenceFrom } from './evidence-adapter';
describe('continuous narrative contracts', () => {
    it('keeps trace geometry finite through every handoff and reverse scroll', () => { for (let i = 0; i <= 120; i++) {
        const p = i / 120;
        expect(tracePath(p)).not.toMatch(/NaN|Infinity/);
        expect(tracePath(p).match(/L/g)).toHaveLength(64);
        const forward = tracePath(p).match(/-?\d+\.\d+/g)!.map(Number), reverse = tracePath(1 - (1 - p)).match(/-?\d+\.\d+/g)!.map(Number);
        forward.forEach((v, j) => expect(Math.abs(v - reverse[j])).toBeLessThanOrEqual(.011));
    } });
    it('holds one scene at anchors and does not crossfade scenario labels', () => { for (let i = 0; i < 7; i++) {
        expect(sceneIndex(i / 6)).toBe(i);
        expect(sceneOpacity(i / 6, i)).toBe(1);
    } for (let n = 0; n <= 120; n++)
        expect(Array.from({ length: 7 }, (_, i) => sceneOpacity(n / 120, i)).filter(x => x > 0).length).toBeLessThanOrEqual(1); });
    it('clamps overscroll at both ends', () => { expect(tracePath(-1)).toBe(tracePath(0)); expect(tracePath(2)).toBe(tracePath(1)); });
});
describe('published evidence', () => {
    it('does not turn an absent report into performance claims', () => { expect(evidenceFrom({ available: false, summary_metrics: { mae: 0 } })).toEqual({ kind: 'unavailable' }); expect(evidenceFrom(null)).toEqual({ kind: 'unavailable' }); });
    it('retains legitimate zero values and rejects impossible or nonfinite metrics', () => { expect(evidenceFrom({ summary_metrics: { mae: 0, directional_accuracy: 1.3, rmse: Infinity } })).toMatchObject({ kind: 'summary', metrics: [{ label: 'MAE · report units', value: '0.0000' }] }); });
    it('normalizes legacy summary fields without inventing series or report dates', () => { expect(evidenceFrom({ summary_metrics: { mean_da: .6 }, report_date: 'bad', window_metrics: [{}] })).toEqual({ kind: 'summary', date: null, windows: 1, metrics: [{ label: 'Direction accuracy', value: '60.0%' }] }); });
});
