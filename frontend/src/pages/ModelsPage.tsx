import { MetricCard as Metric } from '../components/ui/MetricCard';
import { PageHeader } from '../components/ui/PageHeader';
import { ViewState } from '../components/ui/ViewState';
import { RefreshButton } from '../components/ui/RefreshButton';
import { SectionHeader } from '../components/ui/SectionHeader';
import { useTftModelSummary } from '../hooks/useQueries';
import { DEFAULT_COPPER_SYMBOL } from '../config/instruments';

const fmtPct = (v?: number) => (v == null ? '—' : `${(v * 100).toFixed(2)}%`);
const fmtNum = (v?: number, digits = 4) => (v == null ? '—' : v.toFixed(digits));

export const ModelsPage = () => {
  const { data, isLoading, isError, refetch, isFetching } = useTftModelSummary(DEFAULT_COPPER_SYMBOL);
  const header = <PageHeader eyebrow="02 / MODEL INTELLIGENCE" title="TFT-ASRO Model" description={<>Weekly strategy, daily diagnostics and the evidence behind each forecast.{data?.trained_at && <span className="block">{data.symbol} · Checkpoint trained {new Date(data.trained_at).toLocaleString()}</span>}</>} actions={data?.quality_gate && <span className={`cm-filter-chip cm-tone-${data.quality_gate.passed ? 'good' : 'bad'}`}>Quality gate: {data.quality_gate.passed ? 'Passed' : 'Failed'}</span>}/>;

  if (isLoading) {
    return <div className="space-y-6">{header}<ViewState kind="loading" title="Loading model intelligence" description="Retrieving the available checkpoint and validation metrics."/></div>;
  }

  if (isError || !data) {
    return <div className="space-y-6">{header}<ViewState kind={isError ? 'error' : 'empty'} title={isError ? 'Model intelligence could not be loaded' : 'No model metadata available'} description="Checkpoint details and validation metrics will appear when they are available. You can check again or inspect the System page for availability." action={<RefreshButton onClick={() => refetch()} busy={isFetching} label="Try again"/>}/></div>;
  }

  const m = data.metrics ?? {};
  const weeklyDa = m.weekly_directional_accuracy;
  const weeklySampleCount = m.weekly_sample_count;
  const weeklyDaThreshold = weeklySampleCount != null && weeklySampleCount < 80 ? 0.51 : 0.53;
  const sharpe = m.sharpe_ratio;
  const vr = m.variance_ratio;
  const mae = m.mae;
  const rmse = m.rmse;
  const sortino = m.sortino_ratio;
  const tail = m.tail_capture_rate;
  const tailCaptureThreshold = 0.35;

  const gate = data.quality_gate;

  return (
    <div className="space-y-6">
      {header}

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

      {gate && gate.warnings?.length > 0 && (
        <div className="rounded-lg border border-amber-700/40 bg-amber-950/20 p-4 text-sm text-amber-100">
          <p className="text-xs uppercase tracking-widest mb-2">Stability Warnings</p>
          <ul className="list-disc list-inside space-y-1">
            {gate.warnings.map((warning: string, i: number) => (
              <li key={i}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Primary Horizon: Weekly Forecast (5D) */}
      <section>
        <SectionHeader eyebrow="PRIMARY / 5 TRADING DAYS" title="Weekly strategy" description="Read the five-day cumulative forecast first. Weekly risk ratios use √52 annualization."/>
        <div className="cm-metric-grid">
          <Metric
            label="Weekly Directional Accuracy"
            value={fmtPct(weeklyDa)}
            tone={weeklyDa != null ? (weeklyDa >= weeklyDaThreshold ? 'good' : 'bad') : 'neutral'}
            hint={`Weekly sign accuracy · gate ≥ ${(weeklyDaThreshold * 100).toFixed(0)}%`}
          />
          <Metric
            label="Weekly Sharpe Ratio"
            value={fmtNum(m.weekly_sharpe_ratio, 2)}
            tone={m.weekly_sharpe_ratio != null ? (m.weekly_sharpe_ratio >= -0.20 ? 'good' : 'bad') : 'neutral'}
            hint="52-week annualized return · gate ≥ -0.20"
          />
          <Metric
            label="Weekly Sortino Ratio"
            value={fmtNum(m.weekly_sortino_ratio, 2)}
            tone={m.weekly_sortino_ratio != null ? (m.weekly_sortino_ratio >= 0 ? 'good' : 'bad') : 'neutral'}
            hint="Penalizes downside volatility only (√52 factor)"
          />
          <Metric
            label="Weekly Magnitude Ratio"
            value={fmtNum(m.weekly_magnitude_ratio, 2)}
            tone={
              m.weekly_magnitude_ratio != null
                ? m.weekly_magnitude_ratio >= 0.65 && m.weekly_magnitude_ratio <= 1.35
                  ? 'good'
                  : 'bad'
                : 'neutral'
            }
            hint="Bounded pred |abs| / actual |abs| · gate [0.65, 1.35]"
          />
          <Metric
            label="Weekly Tail Capture"
            value={fmtPct(m.weekly_tail_capture_rate)}
            tone={
              m.weekly_tail_capture_rate != null
                ? m.weekly_tail_capture_rate >= 0.45
                  ? 'good'
                  : 'bad'
                : 'neutral'
            }
            hint="Correct direction on extreme weekly moves · gate ≥ 45%"
          />
          <Metric
            label="Raw Magnitude Ratio"
            value={fmtNum(m.weekly_raw_magnitude_ratio, 2)}
            tone={
              m.weekly_raw_magnitude_ratio != null
                ? m.weekly_raw_magnitude_ratio <= 1.8
                  ? 'good'
                  : m.weekly_raw_magnitude_ratio <= 3.0
                  ? 'neutral'
                  : 'bad'
                : 'neutral'
            }
            hint="Pre-cap neural network scale · warns if > 1.8"
          />
          <Metric
            label="Cap Clipping Rate"
            value={fmtPct(m.weekly_median_bound_applied_rate)}
            tone={
              m.weekly_median_bound_applied_rate != null
                ? m.weekly_median_bound_applied_rate <= 0.3
                  ? 'good'
                  : m.weekly_median_bound_applied_rate <= 0.5
                  ? 'neutral'
                  : 'bad'
                : 'neutral'
            }
            hint="Fraction of predictions capped by weekly median ceiling"
          />
          <Metric
            label="Weekly PI80 Coverage"
            value={fmtPct(m.weekly_pi80_coverage)}
            tone={
              m.weekly_pi80_coverage != null
                ? m.weekly_pi80_coverage >= 0.74 && m.weekly_pi80_coverage <= 0.86
                  ? 'good'
                  : 'bad'
                : 'neutral'
            }
            hint="80% prediction interval empirical coverage (target ≈ 80%)"
          />
        </div>
      </section>

      {/* Single-Step Diagnostics: Daily Path (T+1) */}
      <details className="cm-data-disclosure cm-diagnostics">
        <summary><span>Daily diagnostics <small>T+1 · single-step evidence</small></span><span aria-hidden="true">+</span></summary>
        <p className="cm-section-description">These metrics describe the daily path. Daily risk ratios use √252 annualization; evaluate them separately from the weekly strategy above.</p>
        <div className="cm-metric-grid">
          <Metric
            label="Daily Directional Accuracy"
            value={fmtPct(m.directional_accuracy)}
            tone={m.directional_accuracy != null ? (m.directional_accuracy >= 0.50 ? 'good' : 'neutral') : 'neutral'}
            hint="Single-day (T+1) direction accuracy"
          />
          <Metric
            label="Daily Sharpe Ratio"
            value={fmtNum(sharpe, 3)}
            tone={sharpe != null ? (sharpe >= -0.30 ? 'good' : 'bad') : 'neutral'}
            hint="252-day annualized sanity check · gate ≥ -0.30"
          />
          <Metric
            label="Daily Sortino Ratio"
            value={fmtNum(sortino, 3)}
            tone={sortino != null ? (sortino >= 0 ? 'good' : 'bad') : 'neutral'}
            hint="Single-day downside risk-adjusted return"
          />
          <Metric
            label="Variance Ratio"
            value={fmtNum(vr, 3)}
            tone={vr != null ? (vr >= 0.5 && vr <= 1.5 ? 'good' : 'bad') : 'neutral'}
            hint="pred σ / actual σ (target ≈ 1.0)"
          />
          <Metric label="MAE" value={fmtNum(mae)} hint="Average absolute daily prediction error; lower is better" />
          <Metric label="RMSE" value={fmtNum(rmse)} hint="Daily error measure that weighs larger misses more heavily" />
          <Metric
            label="Daily Tail Capture"
            value={fmtPct(tail)}
            tone={tail != null ? (tail >= tailCaptureThreshold ? 'good' : 'bad') : 'neutral'}
            hint="Correct direction on extreme daily moves · ≥ 35%"
          />
          <Metric
            label="Pred σ / Actual σ"
            value={`${fmtNum(m.pred_std, 4)} / ${fmtNum(m.actual_std, 4)}`}
            hint="Daily standard deviation ratio"
          />
        </div>
      </details>

      {/* Variable importance */}
      {data.variable_importance && data.variable_importance.length > 0 && (
        <section>
          <SectionHeader title="Model inputs" description="Relative feature importance from the published model. Importance describes model use, not a causal effect on price."/>
          <div className="cm-panel space-y-4">
            {data.variable_importance.map((vi: any, i: number) => {
              const max = data.variable_importance[0]?.importance || 1;
              const pct = (vi.importance / max) * 100;
              const label = vi.label || vi.description || vi.feature;
              return (
                <div key={i} title={vi.feature}>
                  <div className="flex justify-between items-start text-sm mb-2 gap-3">
                    <div className="flex flex-wrap items-center gap-2 min-w-0">
                      {vi.category && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 uppercase tracking-wider shrink-0">
                          {vi.category}
                        </span>
                      )}
                      <span className="text-slate-200 whitespace-normal break-words">{label}</span>
                    </div>
                    <span className="text-slate-400 font-mono shrink-0">{vi.importance.toFixed(4)}</span>
                  </div>
                  <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-copper-400 origin-left"
                      style={{ transform: `scaleX(${Math.max(0, Math.min(1, pct / 100))})` }}
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
        <details className="cm-data-disclosure">
          <summary>Training configuration</summary>
          <pre>
            {JSON.stringify(data.config, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
};
