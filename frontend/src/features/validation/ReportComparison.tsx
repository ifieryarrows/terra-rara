import { DataTable } from '../../components/ui/DataTable';
import { SectionHeader } from '../../components/ui/SectionHeader';

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const format = (value: unknown, percent = false) => finite(value) ? percent ? `${(value * 100).toFixed(2)}%` : value.toFixed(4) : '—';

/** Renders only comparison fields actually emitted by the backend. No inferred winner. */
export function ReportComparison({ data }: { data: Record<string, unknown> }) {
  const metrics = [
    { label: 'Direction accuracy', tft: data.tft_da, theta: data.theta_da, percent: true, reading: 'Higher is better' },
    { label: 'Sharpe ratio', tft: data.tft_sharpe, theta: data.theta_sharpe, reading: 'Higher is better' },
    { label: 'MAE', tft: data.tft_mae, theta: data.theta_mae, reading: 'Lower is better' },
  ];
  const available = metrics.some(metric => finite(metric.tft) || finite(metric.theta));
  if (!Object.keys(data).length) return null;
  return <section>
    <SectionHeader title={available ? 'Against the Theta baseline' : 'Baseline comparison'} description={available ? 'Compare the reported model and baseline values on the same metric. A missing value is shown as a dash.' : 'The published comparison is available in the report details below.'}/>
    {available && <DataTable caption="Reported TFT-ASRO and Theta baseline metrics. Values are from the same comparison report; no new winner or horizon is inferred.">
      <thead><tr><th scope="col">Metric</th><th scope="col">TFT-ASRO</th><th scope="col">Theta</th><th scope="col">Reading guide</th></tr></thead>
      <tbody>{metrics.map(metric => <tr key={metric.label}><th scope="row">{metric.label}</th><td>{format(metric.tft, metric.percent)}</td><td>{format(metric.theta, metric.percent)}</td><td>{metric.reading}</td></tr>)}</tbody>
    </DataTable>}
    <details className="cm-data-disclosure mt-4"><summary>Comparison report details</summary><pre>{JSON.stringify(data, null, 2)}</pre></details>
  </section>;
}
