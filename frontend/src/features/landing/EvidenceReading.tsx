import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { fetchLatestBacktest } from '../../api';
import { evidenceFrom } from './evidence-adapter';
export function EvidenceReading() {
    const query = useQuery({ queryKey: ['landing-evidence'], queryFn: fetchLatestBacktest, staleTime: 600000, refetchOnWindowFocus: false, retry: 1 });
    const data = evidenceFrom(query.data);
    return <div className="cw-evidence-reading" aria-live="polite">
    {query.isPending ? <p>Checking for published validation…</p> : query.isError ? <><p>Published validation could not be loaded.</p><button onClick={() => query.refetch()}>Try again</button></> : data.kind === 'unavailable' ? <><strong>Published validation is not available yet.</strong><p>Inspect model metadata and data freshness while a report is unavailable.</p></> : <><p>Published report · {data.date ?? 'Date not supplied'} · {data.windows} evaluation windows</p><dl>{data.metrics.map(m => <div key={m.label}><dt>{m.label}</dt><dd>{m.value}</dd></div>)}</dl><p>Aggregate results, not a historical prediction replay. Inspect the report for horizon, sample and baseline details.</p></>}
    <div className="cw-evidence-links"><Link to="/models">Model metadata</Link><Link to="/system">Data freshness</Link></div>
  </div>;
}
