import { memo, type CSSProperties, type ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import clsx from 'clsx';

interface PanelProps {
  title: string;
  icon?: LucideIcon;
  children: ReactNode;
  className?: string;
  colSpan?: number;
}

export const FinancialPanel = memo(function FinancialPanel({ title, icon: Icon, children, className, colSpan = 12 }: PanelProps) {
  return (
    <section className={clsx('cm-panel cm-financial-panel', className)} style={{ '--cm-panel-span': colSpan } as CSSProperties} aria-label={title}>
      <h2 className="cm-panel-title">{Icon && <Icon size={18} aria-hidden="true" />}{title}</h2>
      <div className="cm-panel-body">{children}</div>
    </section>
  );
});
