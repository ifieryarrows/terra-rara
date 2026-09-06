import type { ReactNode } from 'react';

/** A named, keyboard-scrollable viewport preserves table semantics on small screens. */
export function DataTable({ caption, children }: { caption: string; children: ReactNode }) {
  return <div className="cm-table-scroll" tabIndex={0} role="region" aria-label={caption}><table className="cm-table"><caption>{caption}</caption>{children}</table></div>;
}
