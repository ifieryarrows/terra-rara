import type { ReactNode } from 'react';

export function SectionHeader({ title, description, eyebrow }: {
  title: string; description?: ReactNode; eyebrow?: string;
}) {
  return <header className="cm-section-header">
    {eyebrow && <p className="cm-eyebrow">{eyebrow}</p>}
    <h2>{title}</h2>
    {description && <p className="cm-section-description">{description}</p>}
  </header>;
}
