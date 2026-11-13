import type { ReactNode } from 'react';
import clsx from 'clsx';
import { AlertCircle, Compass, Radio } from 'lucide-react';

export interface EmptyStateProps {
  kind?: 'empty' | 'error' | 'unavailable';
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  compact?: boolean;
  className?: string;
}

const ICONS = {
  empty: Compass,
  error: AlertCircle,
  unavailable: Radio,
} as const;

/** Unboxed fail-open status indicator that keeps TerraAtmosphere visible behind fallback copy. */
export function EmptyState({
  kind = 'empty',
  title,
  description,
  action,
  compact = false,
  className,
}: EmptyStateProps) {
  const Icon = ICONS[kind];
  const isError = kind === 'error';

  return (
    <div
      className={clsx(
        'cm-empty-state',
        `cm-empty-state--${kind}`,
        compact && 'cm-empty-state--compact',
        className,
      )}
      role={isError ? 'alert' : 'status'}
      aria-live="polite"
    >
      <div className="cm-empty-state__icon" aria-hidden="true">
        <Icon size={compact ? 18 : 22} />
      </div>
      <div className="cm-empty-state__copy">
        <strong className="cm-empty-state__title">{title}</strong>
        {description ? <p className="cm-empty-state__description">{description}</p> : null}
        {action ? <div className="cm-empty-state__action">{action}</div> : null}
      </div>
    </div>
  );
}
