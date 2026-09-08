type Metric = {
    label: string;
    value: string;
};
export type Evidence = {
    kind: 'unavailable';
} | {
    kind: 'summary';
    date: string | null;
    metrics: Metric[];
    windows: number;
};
const record = (v: unknown): Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v) ? v as Record<string, unknown> : {};
export function evidenceFrom(raw: unknown): Evidence {
    const data = record(raw), summary = record(data.summary_metrics);
    if (data.available === false)
        return { kind: 'unavailable' };
    const metrics: Metric[] = [];
    const add = (label: string, value: unknown, percent = false) => {
        if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || (percent && value > 1))
            return;
        metrics.push({ label, value: percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(4) });
    };
    add('Direction accuracy', summary.directional_accuracy ?? summary.mean_da, true);
    add('MAE · report units', summary.mae ?? summary.mean_mae);
    add('RMSE · report units', summary.rmse);
    const windows = Array.isArray(data.window_metrics) ? data.window_metrics.length : 0;
    if (!metrics.length && !windows)
        return { kind: 'unavailable' };
    const date = typeof data.report_date === 'string' && Number.isFinite(Date.parse(data.report_date)) ? new Date(data.report_date).toISOString().slice(0, 10) : null;
    return { kind: 'summary', date, metrics, windows };
}
