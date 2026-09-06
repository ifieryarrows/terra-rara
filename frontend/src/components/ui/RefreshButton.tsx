import { RefreshCw } from 'lucide-react';

export function RefreshButton({ onClick, busy, label = 'Refresh' }: { onClick: () => void; busy: boolean; label?: string }) {
  return <button type="button" className="cm-button cm-button--secondary cm-refresh" onClick={onClick} disabled={busy}><RefreshCw size={15} aria-hidden="true"/>{busy ? 'Refreshing…' : label}</button>;
}
