import type { ReactNode } from 'react';

export function SectionHeader({ title, description, eyebrow }: {
  title: string; description?: ReactNode; eyebrow?: string;
}) {
  return <header className="cm-section-header">
    {eyebrow && <p className="cm-eyebrow" data-cm-route-reveal="copy">{eyebrow}</p>}
    <h2 data-cm-route-reveal="copy">{title}</h2>
    {description && <p className="cm-section-description" data-cm-route-reveal="copy">{description}</p>}
  </header>;
}
