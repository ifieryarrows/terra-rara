import type { ReactNode } from 'react';

export function MetricCard({ label, value, hint, tone = 'neutral' }: {
  label: string; value: ReactNode; hint?: string; tone?: 'good' | 'bad' | 'neutral';
}) {
  return <div className="cm-panel"><p className="cm-metric-label">{label}</p><p className={`cm-metric-value cm-tone-${tone}`}>{value}</p>{hint && <p className="cm-metric-hint">{hint}</p>}</div>;
}
