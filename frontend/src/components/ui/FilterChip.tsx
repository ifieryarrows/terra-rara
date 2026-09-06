import type { ButtonHTMLAttributes } from 'react';

export function FilterChip({ active, className = '', children, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { active: boolean }) {
  return <button {...props} type="button" aria-pressed={active} className={`cm-filter-chip ${className}`}>{children}</button>;
}
