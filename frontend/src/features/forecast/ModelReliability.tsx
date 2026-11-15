import { useId } from 'react';
import { Link } from 'react-router-dom';
import type { TFTModelMetadata } from '../../types';
import { finite } from './chart-data';

type ExtendedReliabilityMetrics = TFTModelMetadata['metrics'] & {
  weekly_sharpe_ratio?: number;
  weekly_sortino_ratio?: number;
  weekly_magnitude_ratio?: number;
  weekly_raw_magnitude_ratio?: number;
  weekly_tail_capture_rate?: number;
  weekly_pi80_coverage?: number;
};

function polarToXY(cx: number, cy: number, radius: number, angleDeg: number) {
  const rad = (angleDeg * Math.PI) / 180;
  return {
    x: cx + radius * Math.cos(rad),
    y: cy + radius * Math.sin(rad),
  };
}

function describeArc(cx: number, cy: number, radius: number, startDeg: number, endDeg: number): string {
  const start = polarToXY(cx, cy, radius, startDeg);
  const end = polarToXY(cx, cy, radius, endDeg);
  const largeArc = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${start.x.toFixed(2)} ${start.y.toFixed(2)} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x.toFixed(2)} ${end.y.toFixed(2)}`;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function ModelReliability({ metrics, unavailable }: { metrics?: TFTModelMetadata['metrics']; unavailable: boolean }) {
  const uid = useId().replace(/:/g, '');
  const m = (metrics ?? {}) as ExtendedReliabilityMetrics;

  const weekly = m.weekly_directional_accuracy;
  const dailySharpe = m.sharpe_ratio;
  const weeklySharpe = m.weekly_sharpe_ratio;
  const pi80 = m.weekly_pi80_coverage;
  const varianceRatio = m.variance_ratio;
  const predStd = m.pred_std;
  const actualStd = m.actual_std;
  const threshold = m.weekly_sample_count != null && m.weekly_sample_count < 80 ? 0.51 : 0.53;

  const weeklyKnown = finite(weekly);
  const weeklyPassed = weeklyKnown && weekly >= threshold;

  const gateChecks: boolean[] = [];
  if (weeklyKnown) gateChecks.push(weekly >= threshold);
  if (finite(dailySharpe)) gateChecks.push(dailySharpe >= -0.30);
  if (finite(weeklySharpe)) gateChecks.push(weeklySharpe >= -0.20);
  if (finite(m.weekly_magnitude_ratio)) gateChecks.push(m.weekly_magnitude_ratio >= 0.65 && m.weekly_magnitude_ratio <= 1.35);
  if (finite(m.weekly_raw_magnitude_ratio)) gateChecks.push(m.weekly_raw_magnitude_ratio <= 3.0);
  if (finite(m.weekly_tail_capture_rate)) gateChecks.push(m.weekly_tail_capture_rate >= 0.45);
  if (finite(pi80)) gateChecks.push(pi80 >= 0.74 && pi80 <= 0.86);

  const anyGateBreach = gateChecks.some(ok => !ok);
  const hasFullSummaryTelemetry = weeklyKnown && finite(dailySharpe);

  let gateLabel = 'Unavailable';
  let gateToneClass = 'text-slate-400';
  let gateDotClass = 'bg-slate-500';
  if (!unavailable && metrics) {
    if (anyGateBreach) {
      gateLabel = 'Gate Breach';
      gateToneClass = 'cm-tone-bad';
      gateDotClass = 'bg-rose-400';
    } else if (hasFullSummaryTelemetry) {
      gateLabel = 'Quality Gate Passed';
      gateToneClass = 'cm-tone-good';
      gateDotClass = 'bg-emerald-400 shadow-[0_0_8px_rgba(106,217,174,0.75)]';
    } else if (weeklyPassed) {
      gateLabel = '5D Gate Met · Partial';
      gateToneClass = 'text-[#e6a47a]';
      gateDotClass = 'bg-[#e6a47a]';
    } else {
      gateLabel = 'Telemetry Pending';
    }
  }

  // Orbital gauge geometry: 250deg arc from 145deg to 395deg
  const cx = 95;
  const cy = 78;
  const radius = 60;
  const startDeg = 145;
  const sweepDeg = 250;
  const endDeg = startDeg + sweepDeg;

  const trackPath = describeArc(cx, cy, radius, startDeg, endDeg);
  const outerHaloPath = describeArc(cx, cy, radius + 9, startDeg + 8, endDeg - 8);
  const progressRatio = weeklyKnown ? clamp01(weekly) : 0;
  const activeEndDeg = startDeg + progressRatio * sweepDeg;
  const activePath = progressRatio > 0.01 ? describeArc(cx, cy, radius, startDeg, activeEndDeg) : '';
  const tipPos = polarToXY(cx, cy, radius, activeEndDeg);

  const threshDeg = startDeg + threshold * sweepDeg;
  const threshInner = polarToXY(cx, cy, radius - 6, threshDeg);
  const threshOuter = polarToXY(cx, cy, radius + 6, threshDeg);

  const fiftyDeg = startDeg + 0.5 * sweepDeg;
  const fiftyInner = polarToXY(cx, cy, radius - 4, fiftyDeg);
  const fiftyOuter = polarToXY(cx, cy, radius + 4, fiftyDeg);

  const arcStroke = !weeklyKnown
    ? 'rgba(163, 175, 191, 0.35)'
    : weeklyPassed
      ? `url(#${uid}-arc-good)`
      : `url(#${uid}-arc-bad)`;

  const dispersionRatio = finite(varianceRatio)
    ? varianceRatio
    : (finite(predStd) && finite(actualStd) && actualStd > 0 ? predStd / actualStd : null);

  const dailySharpePct = finite(dailySharpe) ? clamp01((dailySharpe + 1) / 3) * 100 : null;
  const weeklySharpePct = finite(weeklySharpe) ? clamp01((weeklySharpe + 1) / 3) * 100 : null;

  return (
    <div className="cm-reliability space-y-3">
      {/* Top status line: separates horizons and reports honest Quality Gate telemetry */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[0.07] pb-2.5 text-xs">
        <span className="inline-flex items-center gap-2 font-medium tracking-wide">
          <span className={`h-2 w-2 rounded-full ${gateDotClass}`} aria-hidden="true" />
          <span className={gateToneClass}>{gateLabel}</span>
        </span>
        <span className="text-[11px] text-slate-400">5D Primary vs T+1 Diagnostic</span>
      </div>

      {/* Cosmic Orbital Arc Gauge for Primary 5D Directional Accuracy */}
      <div className="flex flex-col items-center pt-1 pb-1.5">
        <div className="relative w-full max-w-[13.5rem]">
          <svg viewBox="0 0 190 124" className="block h-auto w-full overflow-visible" aria-hidden="true">
            <defs>
              <linearGradient id={`${uid}-arc-good`} x1="0%" y1="100%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#dda174" />
                <stop offset="100%" stopColor="#74c7a9" />
              </linearGradient>
              <linearGradient id={`${uid}-arc-bad`} x1="0%" y1="100%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#dda174" />
                <stop offset="100%" stopColor="#db8296" />
              </linearGradient>
            </defs>

            {/* Subtle outer orbital hairline */}
            <path d={outerHaloPath} fill="none" stroke="rgba(221, 161, 116, 0.14)" strokeWidth="1" strokeDasharray="2 5" />

            {/* Background gauge arc */}
            <path d={trackPath} fill="none" stroke="rgba(163, 175, 191, 0.14)" strokeWidth="6.5" strokeLinecap="round" />

            {/* 50% coin-flip baseline tick */}
            <line
              x1={fiftyInner.x}
              y1={fiftyInner.y}
              x2={fiftyOuter.x}
              y2={fiftyOuter.y}
              stroke="rgba(163, 175, 191, 0.35)"
              strokeWidth="1.25"
            />

            {/* Active 5D accuracy arc */}
            {activePath && (
              <path d={activePath} fill="none" stroke={arcStroke} strokeWidth="6.5" strokeLinecap="round" />
            )}

            {/* Gate threshold marker */}
            <line
              x1={threshInner.x}
              y1={threshInner.y}
              x2={threshOuter.x}
              y2={threshOuter.y}
              stroke="#dda174"
              strokeWidth="2"
              strokeLinecap="round"
            />

            {/* Active tip beacon */}
            {weeklyKnown && progressRatio > 0.01 && (
              <g>
                <circle cx={tipPos.x} cy={tipPos.y} r="5.5" fill={weeklyPassed ? 'rgba(116, 199, 169, 0.25)' : 'rgba(219, 130, 150, 0.25)'} />
                <circle cx={tipPos.x} cy={tipPos.y} r="3" fill={weeklyPassed ? '#74c7a9' : '#db8296'} stroke="#060a11" strokeWidth="1.2" />
              </g>
            )}
          </svg>

          {/* Compact core figure centered safely inside the orbital ring */}
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center pt-3 text-center">
            <span className="text-[10px] font-semibold tracking-[0.2em] text-slate-400 uppercase">
              5D DA
            </span>
            <strong
              className={`mt-0.5 text-[1.65rem] leading-none font-medium tracking-tight tabular-nums ${
                !weeklyKnown ? 'text-slate-300' : weeklyPassed ? 'cm-tone-good' : 'cm-tone-bad'
              }`}
            >
              {weeklyKnown ? `${(weekly * 100).toFixed(1)}%` : '—'}
            </strong>
          </div>
        </div>

        {/* Descriptive label & threshold caption placed cleanly below the arc */}
        <div className="mt-1 flex flex-col items-center text-center">
          <span className="text-xs font-medium text-slate-200">Weekly Direction Accuracy</span>
          <span className="mt-0.5 text-[11px] text-slate-400">
            5 trading days · gate ≥ {(threshold * 100).toFixed(0)}%
          </span>
        </div>
        {!weeklyKnown && <p className="cm-chart-note">Weekly metric unavailable</p>}
      </div>

      {/* Refined telemetry tracks (open hairline rows, zero box clutter) */}
      <div className="divide-y divide-white/[0.06]">
        {finite(weeklySharpe) && (
          <div className="cm-reliability-row">
            <div className="min-w-0 flex-1">
              <span>Weekly Sharpe</span>
              <small>5D cumulative · annualized · gate ≥ -0.20</small>
              {weeklySharpePct !== null && (
                <div className="mt-1.5 h-1 w-full max-w-[11rem] overflow-hidden rounded-full bg-white/[0.06]" aria-hidden="true">
                  <div
                    className={`h-full rounded-full ${weeklySharpe >= 0 ? 'bg-[#74c7a9]/75' : weeklySharpe >= -0.2 ? 'bg-[#dda174]/80' : 'bg-[#db8296]/75'}`}
                    style={{ width: `${Math.max(8, weeklySharpePct)}%` }}
                  />
                </div>
              )}
            </div>
            <strong className={weeklySharpe >= 0 ? 'cm-tone-good' : weeklySharpe >= -0.2 ? 'text-slate-100' : 'cm-tone-bad'}>
              {weeklySharpe.toFixed(2)}
            </strong>
          </div>
        )}

        <div className="cm-reliability-row">
          <div className="min-w-0 flex-1">
            <span>Daily Sharpe</span>
            <small>T+1 diagnostic · annualized daily returns</small>
            {dailySharpePct !== null && (
              <div className="mt-1.5 h-1 w-full max-w-[11rem] overflow-hidden rounded-full bg-white/[0.06]" aria-hidden="true">
                <div
                  className={`h-full rounded-full ${dailySharpe! >= 0 ? 'bg-[#8eadde]/85' : dailySharpe! >= -0.3 ? 'bg-[#dda174]/80' : 'bg-[#db8296]/75'}`}
                  style={{ width: `${Math.max(8, dailySharpePct)}%` }}
                />
              </div>
            )}
          </div>
          <strong className={!finite(dailySharpe) ? 'text-slate-400' : dailySharpe >= 0 ? 'cm-tone-good' : dailySharpe < -0.3 ? 'cm-tone-bad' : 'text-slate-100'}>
            {finite(dailySharpe) ? dailySharpe.toFixed(2) : '—'}
          </strong>
        </div>

        <div className="cm-reliability-row">
          <div className="min-w-0 flex-1">
            <span>Risk Dispersion &amp; PI80</span>
            <small>
              {finite(pi80)
                ? '80% interval empirical coverage (target 74–86%)'
                : 'Dispersion variance ratio & interval calibration'}
            </small>
            {finite(pi80) && (
              <div className="relative mt-1.5 h-1 w-full max-w-[11rem] overflow-hidden rounded-full bg-white/[0.06]" aria-hidden="true">
                {/* Target [74%, 86%] highlight band */}
                <div className="absolute inset-y-0 left-[74%] w-[12%] bg-[#74c7a9]/25" />
                <div
                  className={`h-full rounded-full ${pi80 >= 0.74 && pi80 <= 0.86 ? 'bg-[#74c7a9]/80' : 'bg-[#dda174]/80'}`}
                  style={{ width: `${clamp01(pi80) * 100}%` }}
                />
              </div>
            )}
          </div>
          <strong className={finite(pi80) ? (pi80 >= 0.74 && pi80 <= 0.86 ? 'cm-tone-good' : 'text-[#dda174]') : 'text-slate-300'}>
            {finite(pi80)
              ? `${(pi80 * 100).toFixed(0)}%${finite(dispersionRatio) ? ` · VR ${dispersionRatio.toFixed(2)}` : ''}`
              : finite(dispersionRatio)
                ? `VR ${dispersionRatio.toFixed(2)}`
                : '—'}
          </strong>
        </div>
      </div>

      {unavailable && <p className="cm-chart-note">Model metrics could not be refreshed.</p>}
      <Link className="cm-text-link" to="/models">Review model metrics →</Link>
    </div>
  );
}
