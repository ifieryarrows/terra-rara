import type { ReactNode } from 'react';
import { AlertCircle, Inbox, LoaderCircle } from 'lucide-react';

export function ViewState({ kind, title, description, action, compact = false }: {
  kind: 'loading' | 'error' | 'empty'; title: string; description?: ReactNode; action?: ReactNode; compact?: boolean;
}) {
  const Icon = kind === 'loading' ? LoaderCircle : kind === 'error' ? AlertCircle : Inbox;
  return <div className={`cm-view-state cm-view-state--${kind}${compact ? ' cm-view-state--compact' : ''}`} role={kind === 'error' ? 'alert' : 'status'}>
    <Icon className="cm-state-icon" size={24} aria-hidden="true"/>
    <h2>{title}</h2>{description && <p>{description}</p>}{action}
    {kind === 'loading' && <div className="cm-state-skeleton" aria-hidden="true"><span/><span/><span/></div>}
  </div>;
}
