import type { ReactNode } from 'react';

export function PageHeader({ eyebrow, title, description, actions }: {
  eyebrow: string; title: string; description: ReactNode; actions?: ReactNode;
}) {
  return <header className="cm-page-header"><div><p className="cm-eyebrow" data-cm-route-reveal="copy">{eyebrow}</p><h1 data-cm-route-reveal="copy">{title}</h1><div className="cm-page-description" data-cm-route-reveal="copy">{description}</div></div>{actions && <div className="cm-page-actions" data-cm-route-reveal="surface">{actions}</div>}</header>;
}
