import { MetricCard as Stat } from '../components/ui/MetricCard';
import { PageHeader } from '../components/ui/PageHeader';
import { ViewState } from '../components/ui/ViewState';
import { RefreshButton } from '../components/ui/RefreshButton';
import { DataTable } from '../components/ui/DataTable';
import { SectionHeader } from '../components/ui/SectionHeader';
import { ReportComparison } from '../features/validation/ReportComparison';
import { useBacktestReport } from '../hooks/useQueries';

const fmtPct = (v?: any) =>
  typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(2)}%` : '—';
const fmtNum = (v?: any, digits = 3) =>
  typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';

export const ValidationPage = () => {
  const { data, isLoading, isError, refetch, isFetching } = useBacktestReport();
  const header = <PageHeader eyebrow="03 / THE EVIDENCE" title="Walk-Forward Validation" description={<>Out-of-sample backtest results and baseline comparisons.{data?.report_date && <span className="block">Report generated {new Date(data.report_date).toLocaleString()}</span>}</>} actions={<>{data?.verdict && <span className="cm-filter-chip">Verdict: {data.verdict}</span>}<RefreshButton onClick={() => refetch()} busy={isFetching}/></>}/>;

  if (isLoading) {
    return <div className="space-y-6">{header}<ViewState kind="loading" title="Loading validation evidence" description="Retrieving the available out-of-sample report."/></div>;
  }

  // Empty-state (204-like) or real error
  if (isError || !data || (data as any).available === false) {
    const isRealError = isError && !data;
    return <div className="space-y-6">{header}<ViewState kind={isRealError ? 'error' : 'empty'} title={isRealError ? 'The validation report could not be loaded' : 'No backtest report available yet'} description={isRealError ? 'Check again to retrieve the report. Other research views remain available.' : 'Published out-of-sample results and baseline comparisons will appear here when a report is available.'}/></div>;
  }

  const summary = data.summary_metrics || {};
  const theta = data.theta_comparison || {};
  const windows = data.window_metrics || [];

  // Both API-backed reports and the file-backed rolling-window report are valid.
  const da = summary.directional_accuracy ?? summary.mean_da;
  const sharpe = summary.sharpe_ratio ?? summary.mean_sharpe;
  const mae = summary.mae ?? summary.mean_mae;
  const rmse = summary.rmse;
  const vr = summary.variance_ratio ?? summary.mean_vr;

  return (
    <div className="space-y-6">
      {header}

      <section>
        <SectionHeader eyebrow="OUT-OF-SAMPLE EVIDENCE" title="Validation at a glance" description="Read aggregate results first, then compare the baseline and individual windows. Metrics retain the horizon and aggregation of the published report."/>
        <div className="cm-metric-grid">
          <Stat
            label="Directional Accuracy"
            value={fmtPct(da)}
            tone={typeof da === 'number' ? (da >= 0.52 ? 'good' : 'bad') : 'neutral'}
          />
          <Stat
            label="Sharpe Ratio"
            value={fmtNum(sharpe)}
            tone={typeof sharpe === 'number' ? (sharpe >= 0 ? 'good' : 'bad') : 'neutral'}
          />
          <Stat label="Variance Ratio" value={fmtNum(vr)} />
          <Stat label="MAE" value={fmtNum(mae, 4)} hint="Average absolute prediction error; lower is better" />
          <Stat label="RMSE" value={fmtNum(rmse, 4)} hint="Error measure that weighs larger misses more heavily" />
        </div>
      </section>

      <ReportComparison data={theta}/>

      {windows.length > 0 && (
        <section>
          <SectionHeader title={`Window-by-window results (${windows.length})`} description="Look for consistency across evaluation windows. The table scrolls horizontally on smaller screens."/>
          <DataTable caption="Out-of-sample results by validation window. MAE is average absolute error; RMSE weighs large errors more heavily. Variance ratio compares predicted and observed variability.">
              <thead>
                <tr>
                  <th scope="col">Window</th>
                  <th scope="col">Direction accuracy</th>
                  <th scope="col">Sharpe</th>
                  <th scope="col">MAE</th>
                  <th scope="col">RMSE</th>
                  <th scope="col">Variance ratio</th>
                </tr>
              </thead>
              <tbody className="font-mono text-slate-200">
                {windows.map((w: any, i: number) => (
                  <tr key={i} className="border-t border-slate-800">
                    <th scope="row">{w.window_id ?? i + 1}</th>
                    <td className="px-3 py-1.5">{fmtPct(w.directional_accuracy ?? w.da)}</td>
                    <td className="px-3 py-1.5">{fmtNum(w.sharpe_ratio ?? w.sharpe)}</td>
                    <td className="px-3 py-1.5">{fmtNum(w.mae, 4)}</td>
                    <td className="px-3 py-1.5">{fmtNum(w.rmse, 4)}</td>
                    <td className="px-3 py-1.5">{fmtNum(w.variance_ratio)}</td>
                  </tr>
                ))}
              </tbody>
          </DataTable>
        </section>
      )}
    </div>
  );
};
