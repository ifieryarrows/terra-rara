import { memo, type CSSProperties, type ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import clsx from 'clsx';

interface PanelProps {
  id?: string;
  title: string;
  eyebrow?: string;
  headerExtra?: ReactNode;
  icon?: LucideIcon;
  children: ReactNode;
  className?: string;
  colSpan?: number;
}

export const FinancialPanel = memo(function FinancialPanel({
  id,
  title,
  eyebrow,
  headerExtra,
  icon: Icon,
  children,
  className,
  colSpan = 12,
}: PanelProps) {
  return (
    <section
      id={id}
      className={clsx('cm-panel cm-financial-panel', className)}
      style={{ '--cm-panel-span': colSpan } as CSSProperties}
      aria-label={title}
      data-cm-route-reveal="surface"
    >
      <div className="cm-panel-head">
        <div className="cm-panel-title-wrap">
          {eyebrow && (
            <span className="cm-panel-eyebrow" data-cm-route-reveal="copy">
              {eyebrow}
            </span>
          )}
          <h2 className="cm-panel-title" data-cm-route-reveal="copy">
            {Icon && <Icon size={18} aria-hidden="true" />}
            <span>{title}</span>
          </h2>
        </div>
        {headerExtra && (
          <div className="cm-panel-extra" data-cm-route-reveal="copy">
            {headerExtra}
          </div>
        )}
      </div>
      <div className="cm-panel-body">{children}</div>
    </section>
  );
});
