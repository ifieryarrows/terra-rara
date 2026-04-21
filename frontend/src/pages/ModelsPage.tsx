import { useTftModelSummary } from '../hooks/useQueries';

const Metric = ({
  label,
  value,
  hint,
  tone = 'neutral',
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  tone?: 'good' | 'bad' | 'neutral';
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

const fmtPct = (v?: number) => (v == null ? '—' : `${(v * 100).toFixed(2)}%`);
const fmtNum = (v?: number, digits = 4) => (v == null ? '—' : v.toFixed(digits));

export const ModelsPage = () => {
  const { data, isLoading, isError, error } = useTftModelSummary('HG=F');

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-6 h-6 border-2 border-slate-600 border-t-amber-400 rounded-full animate-spin" />
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="p-10 border border-rose-800/40 bg-rose-950/30 rounded-lg text-rose-200 text-sm">
        No TFT model metadata available. {error instanceof Error ? `(${error.message})` : ''}
        <br />
        Run the training workflow to populate this page.
      </div>
    );
  }

  const m = data.metrics ?? {};
  const da = m.directional_accuracy;
  const sharpe = m.sharpe_ratio;
  const vr = m.variance_ratio;
  const mae = m.mae;
  const rmse = m.rmse;
  const sortino = m.sortino_ratio;
  const tail = m.tail_capture_rate;

  const gate = data.quality_gate;

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white">TFT-ASRO Model</h2>
          <p className="text-sm text-slate-400 mt-1">
            Symbol <span className="font-mono text-slate-200">{data.symbol}</span>
            {data.trained_at && (
              <>
                {' '}· Checkpoint trained at{' '}
                <span
                  className="font-mono text-slate-200"
                  title="TFT checkpoint training completion time. This is NOT the last prediction timestamp — see the System page for prediction freshness."
                >
                  {new Date(data.trained_at).toLocaleString()}
                </span>
              </>
            )}
          </p>
        </div>
        {gate && (
          <span
            className={`px-3 py-1 rounded text-xs font-bold tracking-wider uppercase ${
              gate.passed
                ? 'bg-emerald-400/10 text-emerald-400 border border-emerald-400/30'
                : 'bg-rose-400/10 text-rose-400 border border-rose-400/30'
            }`}
          >
            Quality Gate: {gate.passed ? 'Passed' : 'Failed'}
          </span>
        )}
      </div>

      {/* Quality gate reasons */}
      {gate && gate.reasons?.length > 0 && (
        <div
          className={`rounded-lg border p-4 text-sm ${
            gate.passed
              ? 'border-emerald-800/40 bg-emerald-950/20 text-emerald-200'
              : 'border-rose-800/40 bg-rose-950/20 text-rose-200'
          }`}
        >
          <p className="text-xs uppercase tracking-widest mb-2">Gate Notes</p>
          <ul className="list-disc list-inside space-y-1">
            {gate.reasons.map((r: string, i: number) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Core metrics */}
      <section>
        <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-3">Core Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Metric
            label="Directional Accuracy"
            value={fmtPct(da)}
            tone={da != null ? (da >= 0.52 ? 'good' : 'bad') : 'neutral'}
            hint="Share of correct direction calls"
          />
          <Metric
            label="Sharpe Ratio"
            value={fmtNum(sharpe, 3)}
            tone={sharpe != null ? (sharpe >= 0 ? 'good' : 'bad') : 'neutral'}
            hint="Risk-adjusted strategy return"
          />
          <Metric
            label="Sortino Ratio"
            value={fmtNum(sortino, 3)}
            tone={sortino != null ? (sortino >= 0 ? 'good' : 'bad') : 'neutral'}
          />
          <Metric
            label="Variance Ratio"
            value={fmtNum(vr, 3)}
            tone={vr != null ? (vr >= 0.5 && vr <= 1.5 ? 'good' : 'bad') : 'neutral'}
            hint="pred σ / actual σ (target ≈ 1.0)"
          />
          <Metric label="MAE" value={fmtNum(mae)} />
          <Metric label="RMSE" value={fmtNum(rmse)} />
          <Metric label="Tail Capture" value={fmtPct(tail)} hint="Extreme-move coverage" />
          <Metric
            label="Pred σ / Actual σ"
            value={`${fmtNum(m.pred_std, 4)} / ${fmtNum(m.actual_std, 4)}`}
          />
        </div>
      </section>

      {/* Variable importance */}
      {data.variable_importance && data.variable_importance.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-3">
            Variable Importance (Top 20)
          </h3>
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-2">
            {data.variable_importance.map((vi: any, i: number) => {
              const max = data.variable_importance[0]?.importance || 1;
              const pct = (vi.importance / max) * 100;
              const label = vi.label || vi.description || vi.feature;
              return (
                <div key={i} title={vi.feature}>
                  <div className="flex justify-between items-center text-xs mb-1 gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      {vi.category && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 uppercase tracking-wider shrink-0">
                          {vi.category}
                        </span>
                      )}
                      <span className="text-slate-200 truncate">{label}</span>
                      {vi.time_horizon && (
                        <span className="text-[9px] font-mono text-slate-500 shrink-0">{vi.time_horizon}</span>
                      )}
                    </div>
                    <span className="text-slate-500 font-mono shrink-0">{vi.importance.toFixed(4)}</span>
                  </div>
                  <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-amber-500 to-rose-500"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Config */}
      {data.config && Object.keys(data.config).length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-3">Training Config</h3>
          <pre className="bg-slate-900 border border-slate-800 rounded-lg p-4 text-xs text-slate-300 font-mono overflow-x-auto">
            {JSON.stringify(data.config, null, 2)}
          </pre>
        </section>
      )}
    </div>
  );
};
