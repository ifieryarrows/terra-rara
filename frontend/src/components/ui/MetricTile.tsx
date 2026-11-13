import type { ReactNode } from 'react';
import clsx from 'clsx';

export type MetricTileTone = 'good' | 'bad' | 'neutral' | 'copper' | 'forecast';

export interface MetricTileProps {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  delta?: ReactNode;
  badge?: ReactNode;
  tone?: MetricTileTone;
  emphasis?: 'default' | 'hero' | 'compact';
  align?: 'start' | 'end';
  className?: string;
}

/** Open-canvas telemetry readout framed solely by typography and a hairline anchor. */
export function MetricTile({
  label,
  value,
  hint,
  delta,
  badge,
  tone = 'neutral',
  emphasis = 'default',
  align = 'start',
  className,
}: MetricTileProps) {
  return (
    <div
      className={clsx(
        'cm-metric-tile',
        `cm-metric-tile--${tone}`,
        emphasis !== 'default' && `cm-metric-tile--${emphasis}`,
        align === 'end' && 'cm-metric-tile--end',
        className,
      )}
    >
      <div className="cm-metric-tile__head">
        <span className="cm-metric-tile__label">{label}</span>
        {badge ? <span className="cm-metric-tile__badge">{badge}</span> : null}
      </div>
      <div className="cm-metric-tile__body">
        <strong className="cm-metric-tile__value cm-tabular">{value}</strong>
        {delta ? <span className="cm-metric-tile__delta cm-tabular">{delta}</span> : null}
      </div>
      {hint ? <span className="cm-metric-tile__hint">{hint}</span> : null}
    </div>
  );
}
