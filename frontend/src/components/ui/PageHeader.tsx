import type { ReactNode } from 'react';

export function PageHeader({ eyebrow, title, description, actions }: {
  eyebrow: string; title: string; description: ReactNode; actions?: ReactNode;
}) {
  return <header className="cm-page-header"><div><p className="cm-eyebrow">{eyebrow}</p><h1>{title}</h1><div className="cm-page-description">{description}</div></div>{actions && <div className="cm-page-actions">{actions}</div>}</header>;
}
