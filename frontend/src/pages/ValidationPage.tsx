import { useBacktestReport } from '../hooks/useQueries';

const Stat = ({
  label,
  value,
  tone = 'neutral',
  hint,
}: {
  label: string;
  value: React.ReactNode;
  tone?: 'good' | 'bad' | 'neutral';
  hint?: string;
}) => (
  <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
    <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">{label}</p>
    <p
      className={`font-mono text-xl tabular-nums ${
        tone === 'good' ? 'text-emerald-400' : tone === 'bad' ? 'text-rose-400' : 'text-slate-100'
      }`}
    >
      {value}
    </p>
    {hint && <p className="text-[10px] text-slate-500 mt-1">{hint}</p>}
  </div>
);

const fmtPct = (v?: any) =>
  typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : '—';
const fmtNum = (v?: any, digits = 3) =>
  typeof v === 'number' ? v.toFixed(digits) : '—';

export const ValidationPage = () => {
  const { data, isLoading, isError, error, refetch, isFetching } = useBacktestReport();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-6 h-6 border-2 border-slate-600 border-t-amber-400 rounded-full animate-spin" />
      </div>
    );
  }

  // Empty-state (204-like) or real error
  if (isError || !data || (data as any).available === false) {
    const isRealError = isError && !data;
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white">Walk-Forward Validation</h2>
          <p className="text-sm text-slate-400 mt-1">
            Out-of-sample backtest results and baseline comparisons.
          </p>
        </div>
        <div
          className={`p-10 border rounded-lg text-sm ${
            isRealError
              ? 'border-rose-800/40 bg-rose-950/30 text-rose-200'
              : 'border-slate-700 bg-slate-900 text-slate-300'
          }`}
        >
          {isRealError ? (
            <>
              Failed to load the backtest report.
              {error instanceof Error ? ` (${error.message})` : ''}
            </>
          ) : (
            <>
              No backtest report has been generated yet.
              <br />
              Run{' '}
              <code className="text-amber-300 font-mono">
                python -m backend.backtest.runner --include-tft
              </code>{' '}
              and results will appear here automatically.
            </>
          )}
        </div>
      </div>
    );
  }

  const summary = data.summary_metrics || {};
  const theta = data.theta_comparison || {};
  const windows = data.window_metrics || [];

  const da = summary.directional_accuracy;
  const sharpe = summary.sharpe_ratio;
  const mae = summary.mae;
  const rmse = summary.rmse;
  const vr = summary.variance_ratio;

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white">Walk-Forward Validation</h2>
          <p className="text-sm text-slate-400 mt-1">
            Report{' '}
            <span className="font-mono text-slate-200">
              {data.report_date ? new Date(data.report_date).toLocaleString() : '—'}
            </span>
          </p>
        </div>
        <div className="flex items-center gap-3">
          {data.verdict && (
            <span className="px-3 py-1 rounded text-xs font-bold tracking-wider uppercase bg-violet-400/10 text-violet-300 border border-violet-400/30">
              Verdict: {data.verdict}
            </span>
          )}
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="text-slate-400 hover:text-white px-3 py-1 bg-slate-800 rounded border border-slate-700 text-xs disabled:opacity-50"
          >
            {isFetching ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      <section>
        <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-3">Summary Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
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
          <Stat label="MAE" value={fmtNum(mae, 4)} />
          <Stat label="RMSE" value={fmtNum(rmse, 4)} />
        </div>
      </section>

      {theta && Object.keys(theta).length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-3">
            Theta Baseline Comparison
          </h3>
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
            <pre className="text-xs text-slate-300 font-mono overflow-x-auto">
              {JSON.stringify(theta, null, 2)}
            </pre>
          </div>
        </section>
      )}

      {windows.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-3">
            Per-Window Metrics ({windows.length} windows)
          </h3>
          <div className="overflow-x-auto bg-slate-900 border border-slate-800 rounded-lg">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-800/50 text-slate-400 uppercase tracking-wider">
                <tr>
                  <th className="px-3 py-2">#</th>
                  <th className="px-3 py-2">DA</th>
                  <th className="px-3 py-2">Sharpe</th>
                  <th className="px-3 py-2">MAE</th>
                  <th className="px-3 py-2">RMSE</th>
                  <th className="px-3 py-2">VR</th>
                </tr>
              </thead>
              <tbody className="font-mono text-slate-200">
                {windows.map((w: any, i: number) => (
                  <tr key={i} className="border-t border-slate-800">
                    <td className="px-3 py-1.5 text-slate-500">{w.window_id ?? i + 1}</td>
                    <td className="px-3 py-1.5">{fmtPct(w.directional_accuracy)}</td>
                    <td className="px-3 py-1.5">{fmtNum(w.sharpe_ratio)}</td>
                    <td className="px-3 py-1.5">{fmtNum(w.mae, 4)}</td>
                    <td className="px-3 py-1.5">{fmtNum(w.rmse, 4)}</td>
                    <td className="px-3 py-1.5">{fmtNum(w.variance_ratio)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
};
