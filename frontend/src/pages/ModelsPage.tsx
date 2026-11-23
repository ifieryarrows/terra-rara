import { useMemo, useState } from 'react';
import { MetricCard as Metric } from '../components/ui/MetricCard';
import { PageHeader } from '../components/ui/PageHeader';
import { ViewState } from '../components/ui/ViewState';
import { ModelsSkeleton } from '../components/skeletons';
import { RefreshButton } from '../components/ui/RefreshButton';
import { SectionHeader } from '../components/ui/SectionHeader';
import { useTftModelSummary } from '../hooks/useQueries';
import { DEFAULT_COPPER_SYMBOL } from '../config/instruments';
import {
  computeQualityGateThresholds,
  hasWeeklySignCollapse,
} from '../features/forecast/ModelReliability';

const fmtPct = (v?: number) => (v == null ? '—' : `${(v * 100).toFixed(2)}%`);
const fmtNum = (v?: number, digits = 4) => (v == null ? '—' : v.toFixed(digits));

export const ModelsPage = () => {
  const { data, isLoading, isError, refetch, isFetching } = useTftModelSummary(DEFAULT_COPPER_SYMBOL);
  const [configOpen, setConfigOpen] = useState(false);
  const configJson = useMemo(() => (configOpen && data?.config ? JSON.stringify(data.config, null, 2) : ''), [configOpen, data?.config]);
  const header = <PageHeader eyebrow="02 / MODEL INTELLIGENCE" title="TFT-ASRO Model" description={<>Weekly strategy, daily diagnostics and the evidence behind each forecast.{data?.trained_at && <span className="block">{data.symbol} · Checkpoint trained {new Date(data.trained_at).toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })}</span>}</>} actions={data?.quality_gate && <span className={`cm-filter-chip cm-tone-${data.quality_gate.passed ? 'good' : 'bad'}`}>Quality gate: {data.quality_gate.passed ? 'Passed' : 'Failed'}</span>}/>;

  if (isLoading) {
    return <ModelsSkeleton />;
  }

  if (isError || !data) {
    return <div className="space-y-6">{header}<ViewState kind={isError ? 'error' : 'empty'} title={isError ? 'Model intelligence could not be loaded' : 'No model metadata available'} description="Checkpoint details and validation metrics will appear when they are available. You can check again or inspect the System page for availability." action={<RefreshButton onClick={() => refetch()} busy={isFetching} label="Try again"/>}/></div>;
  }

  const m = data.metrics ?? {};
  const weeklyDa = m.weekly_directional_accuracy;
  const thresholds = computeQualityGateThresholds(m.weekly_sample_count);
  const weeklyDaThreshold = thresholds.minWeeklyDa;
  const hasHacSharpe = m.weekly_sharpe_ratio_hac != null;
  const weeklySharpe = m.weekly_sharpe_ratio_hac ?? m.weekly_sharpe_ratio;
  const pi80Low = thresholds.pi80Low;
  const pi80High = thresholds.pi80High;
  const weeklyMaeRatio = m.weekly_mae_vs_naive_zero;
  const predPosRate = m.weekly_pred_positive_rate;
  const actualPosRate = m.weekly_actual_positive_rate;
  const signCollapsed = hasWeeklySignCollapse(predPosRate, actualPosRate);
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
          className={`border-l-2 pl-4 py-1 text-sm ${
            gate.passed
              ? 'border-[#74c7a9]/70 text-slate-200'
              : 'border-[#db8296]/70 text-slate-200'
          }`}
        >
          <p className={`text-xs uppercase tracking-widest mb-2 ${gate.passed ? 'text-[#74c7a9]' : 'text-[#db8296]'}`}>
            Gate Notes
          </p>
          <ul className="list-disc list-inside space-y-1 text-slate-300">
            {gate.reasons.map((r: string, i: number) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {gate && gate.warnings?.length > 0 && (
        <div className="border-l-2 border-[#dda174]/70 pl-4 py-1 text-sm text-slate-200">
          <p className="text-xs uppercase tracking-widest mb-2 text-[#dda174]">Stability Warnings</p>
          <ul className="list-disc list-inside space-y-1 text-slate-300">
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
            value={fmtNum(weeklySharpe, 2)}
            tone={weeklySharpe != null ? (weeklySharpe >= thresholds.minWeeklySharpe ? 'good' : 'bad') : 'neutral'}
            hint={hasHacSharpe ? 'Newey-West HAC √52 · gate ≥ -0.35' : 'Sample √52 · gate ≥ -0.35'}
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
                ? m.weekly_magnitude_ratio >= 0.02 && m.weekly_magnitude_ratio <= 2.5
                  ? 'good'
                  : 'bad'
                : 'neutral'
            }
            hint="Bounded pred |abs| / actual |abs| · gate [0.02, 2.50]"
          />
          <Metric
            label="Weekly Tail Capture"
            value={fmtPct(m.weekly_tail_capture_rate)}
            tone={
              m.weekly_tail_capture_rate != null
                ? m.weekly_tail_capture_rate >= 0.30
                  ? 'good'
                  : 'neutral'
                : 'neutral'
            }
            hint="Correct direction on extreme weekly moves · diagnostic ≥ 30%"
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
                ? m.weekly_pi80_coverage >= pi80Low && m.weekly_pi80_coverage <= pi80High
                  ? 'good'
                  : 'bad'
                : 'neutral'
            }
            hint={`80% interval empirical coverage · gate [${(pi80Low * 100).toFixed(0)}%, ${(pi80High * 100).toFixed(0)}%]`}
          />
          {weeklyMaeRatio != null && (
            <Metric
              label="Weekly MAE vs Naive"
              value={fmtNum(weeklyMaeRatio, 3)}
              tone={weeklyMaeRatio <= 1.0 ? 'good' : weeklyMaeRatio <= thresholds.maxWeeklyMaeVsNaiveZero ? 'neutral' : 'bad'}
              hint="Zero-baseline error ratio · gate ≤ 1.12"
            />
          )}
          {(predPosRate != null || actualPosRate != null) && (
            <Metric
              label="Weekly Sign Balance"
              value={`${predPosRate != null ? `${(predPosRate * 100).toFixed(0)}%` : '—'} / ${actualPosRate != null ? `${(actualPosRate * 100).toFixed(0)}%` : '—'}`}
              tone={signCollapsed ? 'bad' : 'good'}
              hint="Predicted vs actual positive weeks · anti-collapse"
            />
          )}
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
            tone={sharpe != null ? (sharpe >= -0.30 ? 'good' : 'neutral') : 'neutral'}
            hint="252-day annualized sanity check · diagnostic ≥ -0.30"
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
            tone={vr != null ? (vr >= 0.25 && vr <= 2.5 ? 'good' : 'neutral') : 'neutral'}
            hint="pred σ / actual σ · diagnostic [0.25, 2.50]"
          />
          <Metric label="MAE" value={fmtNum(mae)} hint="Average absolute daily prediction error; lower is better" />
          <Metric label="RMSE" value={fmtNum(rmse)} hint="Daily error measure that weighs larger misses more heavily" />
          <Metric
            label="Daily Tail Capture"
            value={fmtPct(tail)}
            tone={tail != null ? (tail >= tailCaptureThreshold ? 'good' : 'neutral') : 'neutral'}
            hint="Correct direction on extreme daily moves · diagnostic ≥ 35%"
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
                        <span className="text-xs px-1.5 py-0.5 rounded-sm bg-white/[0.05] text-[#8eadde] uppercase tracking-wider shrink-0">
                          {vi.category}
                        </span>
                      )}
                      <span className="text-slate-200 whitespace-normal break-words">{label}</span>
                    </div>
                    <span className="text-slate-400 font-mono cm-tabular shrink-0">{vi.importance.toFixed(4)}</span>
                  </div>
                  <div className="h-[2px] bg-white/[0.08] overflow-hidden">
                    <div
                      className="h-full bg-[#dda174] origin-left"
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
        <details className="cm-data-disclosure" onToggle={(e) => setConfigOpen(e.currentTarget.open)}>
          <summary>Training configuration</summary>
          {configOpen && (
            <pre>
              {configJson}
            </pre>
          )}
        </details>
      )}
    </div>
  );
};
