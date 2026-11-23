import { useId } from 'react';
import { Link } from 'react-router-dom';
import type { QualityGateResponse, TFTMetrics } from '../../types';
import { finite } from './chart-data';

export interface QualityGateThresholds {
  sampleCount: number;
  nEff: number;
  minWeeklyDa: number;
  minWeeklySharpe: number;
  minWeeklyMagnitudeRatio: number;
  maxWeeklyMagnitudeRatio: number;
  maxWeeklyRawMagnitudeRatio: number;
  maxWeeklyMaeVsNaiveZero: number;
  pi80Low: number;
  pi80High: number;
  maxPi80WidthRatio: number;
  minPi96Cov: number;
  maxPi96WidthRatio: number;
  maxCrossingRate: number;
}

/**
 * Mirror `backend/app/quality_gate.py` threshold derivations so UI badges and
 * gauges never contradict backend promotion decisions.
 */
export function computeQualityGateThresholds(sampleCount?: number | null): QualityGateThresholds {
  const n = typeof sampleCount === 'number' && Number.isFinite(sampleCount) && sampleCount > 0
    ? Math.trunc(sampleCount)
    : 0;
  const minWeeklyDa = n === 0 || n >= 300 ? 0.49 : n >= 150 ? 0.47 : 0.45;
  const nEff = Math.max((n > 0 ? n : 100) / 5, 1);
  const pi80Tol = Math.max(0.08, Math.min(0.18, 1.65 * Math.sqrt((0.8 * 0.2) / Math.max(nEff, 10))));
  const pi80Low = Math.round((0.8 - pi80Tol) * 10000) / 10000;
  const pi80High = Math.round(Math.min(0.98, 0.8 + pi80Tol) * 10000) / 10000;
  const minPi96Cov = nEff < 25 ? 0.85 : 0.88;

  return {
    sampleCount: n,
    nEff,
    minWeeklyDa,
    minWeeklySharpe: -0.35,
    minWeeklyMagnitudeRatio: 0.02,
    maxWeeklyMagnitudeRatio: 2.5,
    maxWeeklyRawMagnitudeRatio: 3.0,
    maxWeeklyMaeVsNaiveZero: 1.12,
    pi80Low,
    pi80High,
    maxPi80WidthRatio: 2.2,
    minPi96Cov,
    maxPi96WidthRatio: 3.2,
    maxCrossingRate: 0.001,
  };
}

export function hasWeeklySignCollapse(predPosRate?: number, actualPosRate?: number): boolean {
  if (!finite(predPosRate) || !finite(actualPosRate)) return false;
  return (predPosRate > 0.92 && actualPosRate < 0.67) || (predPosRate < 0.08 && actualPosRate > 0.33);
}

export function evaluateClientQualityGate(
  metrics?: TFTMetrics | null,
  qualityGate?: QualityGateResponse | null,
): {
  thresholds: QualityGateThresholds;
  passed: boolean | null;
  partial: boolean;
  reasons: string[];
  warnings: string[];
} {
  const m = metrics ?? {};
  const thresholds = computeQualityGateThresholds(m.weekly_sample_count);

  if (qualityGate) {
    return {
      thresholds,
      passed: qualityGate.passed,
      partial: false,
      reasons: qualityGate.reasons ?? [],
      warnings: qualityGate.warnings ?? [],
    };
  }

  const weeklyDa = m.weekly_directional_accuracy;
  const effectiveWeeklySharpe = finite(m.weekly_sharpe_ratio_hac)
    ? m.weekly_sharpe_ratio_hac
    : m.weekly_sharpe_ratio;

  const breachChecks: boolean[] = [
    finite(weeklyDa) ? weeklyDa >= thresholds.minWeeklyDa : true,
    finite(effectiveWeeklySharpe) ? effectiveWeeklySharpe >= thresholds.minWeeklySharpe : true,
    finite(m.weekly_magnitude_ratio)
      ? m.weekly_magnitude_ratio >= thresholds.minWeeklyMagnitudeRatio &&
        m.weekly_magnitude_ratio <= thresholds.maxWeeklyMagnitudeRatio
      : true,
    finite(m.weekly_raw_magnitude_ratio)
      ? m.weekly_raw_magnitude_ratio <= thresholds.maxWeeklyRawMagnitudeRatio
      : true,
    finite(m.weekly_mae_vs_naive_zero)
      ? m.weekly_mae_vs_naive_zero <= thresholds.maxWeeklyMaeVsNaiveZero
      : true,
    !hasWeeklySignCollapse(m.weekly_pred_positive_rate, m.weekly_actual_positive_rate),
    finite(m.weekly_pi80_coverage)
      ? m.weekly_pi80_coverage >= thresholds.pi80Low && m.weekly_pi80_coverage <= thresholds.pi80High
      : true,
    finite(m.weekly_pi80_width_ratio)
      ? m.weekly_pi80_width_ratio <= thresholds.maxPi80WidthRatio
      : true,
    finite(m.weekly_pi96_coverage) ? m.weekly_pi96_coverage >= thresholds.minPi96Cov : true,
    finite(m.weekly_pi96_width_ratio)
      ? m.weekly_pi96_width_ratio <= thresholds.maxPi96WidthRatio
      : true,
    ...[
      m.weekly_quantile_crossing_rate,
      m.weekly_sorted_quantile_crossing_rate,
      m.weekly_ordered_quantile_crossing_rate,
      m.weekly_median_sort_gap_max,
      m.quantile_crossing_rate,
      m.median_sort_gap_max,
    ].map((v) => (finite(v) ? v <= thresholds.maxCrossingRate : true)),
  ];

  const anyBreach = breachChecks.some((ok) => !ok);
  const weeklyPassed = finite(weeklyDa) && weeklyDa >= thresholds.minWeeklyDa;
  const hasSummaryTelemetry =
    finite(weeklyDa) &&
    (finite(effectiveWeeklySharpe) || finite(m.weekly_pi80_coverage) || finite(m.sharpe_ratio));

  const warnings: string[] = [];
  if (finite(m.sharpe_ratio) && m.sharpe_ratio < -0.3) {
    warnings.push(`T+1 diagnostic Sharpe ${m.sharpe_ratio.toFixed(2)} < -0.30`);
  }
  if (finite(m.weekly_tail_capture_rate) && m.weekly_tail_capture_rate < 0.3) {
    warnings.push(`Weekly tail capture ${(m.weekly_tail_capture_rate * 100).toFixed(0)}% < 30%`);
  }

  return {
    thresholds,
    passed: anyBreach ? false : hasSummaryTelemetry || weeklyPassed ? true : null,
    partial: !anyBreach && !hasSummaryTelemetry && weeklyPassed,
    reasons: [],
    warnings,
  };
}

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

export function ModelReliability({
  metrics,
  qualityGate,
  unavailable,
}: {
  metrics?: TFTMetrics | null;
  qualityGate?: QualityGateResponse | null;
  unavailable: boolean;
}) {
  const uid = useId().replace(/:/g, '');
  const m: TFTMetrics = metrics ?? {};
  const evaluation = evaluateClientQualityGate(m, qualityGate);
  const { thresholds } = evaluation;

  const weekly = m.weekly_directional_accuracy;
  const dailySharpe = m.sharpe_ratio;
  const hasHacSharpe = finite(m.weekly_sharpe_ratio_hac);
  const weeklySharpe = hasHacSharpe ? m.weekly_sharpe_ratio_hac : m.weekly_sharpe_ratio;
  const weeklyMaeRatio = m.weekly_mae_vs_naive_zero;
  const predPosRate = m.weekly_pred_positive_rate;
  const actualPosRate = m.weekly_actual_positive_rate;
  const signCollapsed = hasWeeklySignCollapse(predPosRate, actualPosRate);
  const pi80 = m.weekly_pi80_coverage;
  const varianceRatio = m.variance_ratio;
  const predStd = m.pred_std;
  const actualStd = m.actual_std;
  const threshold = thresholds.minWeeklyDa;

  const weeklyKnown = finite(weekly);
  const weeklyPassed = weeklyKnown && weekly >= threshold;

  let gateLabel = 'Unavailable';
  let gateToneClass = 'text-slate-400';
  let gateDotClass = 'bg-slate-500';
  if (!unavailable && (metrics || qualityGate)) {
    if (evaluation.passed === false) {
      gateLabel = 'Gate Breach';
      gateToneClass = 'cm-tone-bad';
      gateDotClass = 'bg-rose-400';
    } else if (evaluation.passed === true && !evaluation.partial) {
      gateLabel = 'Quality Gate Passed';
      gateToneClass = 'cm-tone-good';
      gateDotClass = 'bg-emerald-400 shadow-[0_0_8px_rgba(106,217,174,0.75)]';
    } else if (evaluation.passed === true && evaluation.partial) {
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

  const dailySharpeRatio01 = finite(dailySharpe) ? clamp01((dailySharpe + 1) / 3) : null;
  const weeklySharpeRatio01 = finite(weeklySharpe) ? clamp01((weeklySharpe + 1) / 3) : null;
  const pi80InGate = finite(pi80) && pi80 >= thresholds.pi80Low && pi80 <= thresholds.pi80High;
  const pi80LowPct = (thresholds.pi80Low * 100).toFixed(0);
  const pi80HighPct = (thresholds.pi80High * 100).toFixed(0);

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

            <path d={outerHaloPath} fill="none" stroke="rgba(221, 161, 116, 0.14)" strokeWidth="1" strokeDasharray="2 5" />
            <path d={trackPath} fill="none" stroke="rgba(163, 175, 191, 0.14)" strokeWidth="6.5" strokeLinecap="round" />

            <line
              x1={fiftyInner.x}
              y1={fiftyInner.y}
              x2={fiftyOuter.x}
              y2={fiftyOuter.y}
              stroke="rgba(163, 175, 191, 0.35)"
              strokeWidth="1.25"
            />

            {activePath && (
              <path d={activePath} fill="none" stroke={arcStroke} strokeWidth="6.5" strokeLinecap="round" />
            )}

            <line
              x1={threshInner.x}
              y1={threshInner.y}
              x2={threshOuter.x}
              y2={threshOuter.y}
              stroke="#dda174"
              strokeWidth="2"
              strokeLinecap="round"
            />

            {weeklyKnown && progressRatio > 0.01 && (
              <g>
                <circle cx={tipPos.x} cy={tipPos.y} r="5.5" fill={weeklyPassed ? 'rgba(116, 199, 169, 0.25)' : 'rgba(219, 130, 150, 0.25)'} />
                <circle cx={tipPos.x} cy={tipPos.y} r="3" fill={weeklyPassed ? '#74c7a9' : '#db8296'} stroke="#060a11" strokeWidth="1.2" />
              </g>
            )}
          </svg>

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
              <span>{hasHacSharpe ? 'Weekly Sharpe (HAC)' : 'Weekly Sharpe'}</span>
              <small>
                {hasHacSharpe ? '5D cumulative · HAC √52' : '5D cumulative · √52'} · gate ≥ {thresholds.minWeeklySharpe.toFixed(2)}
              </small>
              {weeklySharpeRatio01 !== null && (
                <div className="mt-1.5 h-1 w-full max-w-[11rem] overflow-hidden rounded-full bg-white/[0.06]" aria-hidden="true">
                  <div
                    className={`h-full w-full origin-left rounded-full ${
                      weeklySharpe >= 0
                        ? 'bg-[#74c7a9]/75'
                        : weeklySharpe >= thresholds.minWeeklySharpe
                          ? 'bg-[#dda174]/80'
                          : 'bg-[#db8296]/75'
                    }`}
                    style={{ transform: `scaleX(${Math.max(0.08, weeklySharpeRatio01)})` }}
                  />
                </div>
              )}
            </div>
            <strong
              className={
                weeklySharpe >= 0
                  ? 'cm-tone-good'
                  : weeklySharpe >= thresholds.minWeeklySharpe
                    ? 'text-slate-100'
                    : 'cm-tone-bad'
              }
            >
              {weeklySharpe.toFixed(2)}
            </strong>
          </div>
        )}

        {finite(weeklyMaeRatio) && (
          <div className="cm-reliability-row">
            <div className="min-w-0 flex-1">
              <span>Weekly MAE vs Naive</span>
              <small>Zero-baseline ratio · gate ≤ {thresholds.maxWeeklyMaeVsNaiveZero.toFixed(2)}</small>
            </div>
            <strong
              className={
                weeklyMaeRatio <= 1.0
                  ? 'cm-tone-good'
                  : weeklyMaeRatio <= thresholds.maxWeeklyMaeVsNaiveZero
                    ? 'text-slate-100'
                    : 'cm-tone-bad'
              }
            >
              {weeklyMaeRatio.toFixed(3)}
            </strong>
          </div>
        )}

        {(finite(predPosRate) || finite(actualPosRate)) && (
          <div className="cm-reliability-row">
            <div className="min-w-0 flex-1">
              <span>Weekly Sign Balance</span>
              <small>Predicted vs actual positive rate · anti-collapse</small>
            </div>
            <strong className={signCollapsed ? 'cm-tone-bad' : 'text-slate-100'}>
              {finite(predPosRate) ? `${(predPosRate * 100).toFixed(0)}%` : '—'} /{' '}
              {finite(actualPosRate) ? `${(actualPosRate * 100).toFixed(0)}%` : '—'}
            </strong>
          </div>
        )}

        <div className="cm-reliability-row">
          <div className="min-w-0 flex-1">
            <span>Risk Dispersion &amp; PI80</span>
            <small>
              {finite(pi80)
                ? `80% interval empirical coverage (gate ${pi80LowPct}–${pi80HighPct}%)`
                : 'Dispersion variance ratio & interval calibration'}
            </small>
            {finite(pi80) && (
              <div className="relative mt-1.5 h-1 w-full max-w-[11rem] overflow-hidden rounded-full bg-white/[0.06]" aria-hidden="true">
                <div
                  className="absolute inset-y-0 bg-[#74c7a9]/25"
                  style={{
                    left: `${(thresholds.pi80Low * 100).toFixed(1)}%`,
                    width: `${((thresholds.pi80High - thresholds.pi80Low) * 100).toFixed(1)}%`,
                  }}
                />
                <div
                  className={`h-full w-full origin-left rounded-full ${pi80InGate ? 'bg-[#74c7a9]/80' : 'bg-[#db8296]/80'}`}
                  style={{ transform: `scaleX(${clamp01(pi80)})` }}
                />
              </div>
            )}
          </div>
          <strong
            className={
              finite(pi80)
                ? pi80InGate
                  ? 'cm-tone-good'
                  : 'cm-tone-bad'
                : 'text-slate-300'
            }
          >
            {finite(pi80)
              ? `${(pi80 * 100).toFixed(0)}%${finite(dispersionRatio) ? ` · VR ${dispersionRatio.toFixed(2)}` : ''}`
              : finite(dispersionRatio)
                ? `VR ${dispersionRatio.toFixed(2)}`
                : '—'}
          </strong>
        </div>

        <div className="cm-reliability-row">
          <div className="min-w-0 flex-1">
            <span>Daily Sharpe</span>
            <small>T+1 diagnostic · √252 · non-blocking warn &lt; -0.30</small>
            {dailySharpeRatio01 !== null && (
              <div className="mt-1.5 h-1 w-full max-w-[11rem] overflow-hidden rounded-full bg-white/[0.06]" aria-hidden="true">
                <div
                  className={`h-full w-full origin-left rounded-full ${
                    dailySharpe! >= 0 ? 'bg-[#8eadde]/85' : dailySharpe! >= -0.3 ? 'bg-[#dda174]/80' : 'bg-[#db8296]/75'
                  }`}
                  style={{ transform: `scaleX(${Math.max(0.08, dailySharpeRatio01)})` }}
                />
              </div>
            )}
          </div>
          <strong
            className={
              !finite(dailySharpe)
                ? 'text-slate-400'
                : dailySharpe >= 0
                  ? 'cm-tone-good'
                  : dailySharpe < -0.3
                    ? 'text-[#dda174]'
                    : 'text-slate-100'
            }
          >
            {finite(dailySharpe) ? dailySharpe.toFixed(2) : '—'}
          </strong>
        </div>
      </div>

      {evaluation.passed === false && evaluation.reasons.length > 0 && (
        <p className="cm-chart-note cm-tone-bad">
          {evaluation.reasons[0]}
          {evaluation.reasons.length > 1 ? ` (+${evaluation.reasons.length - 1} more)` : ''}
        </p>
      )}
      {evaluation.passed === true && evaluation.warnings.length > 0 && (
        <p className="cm-chart-note text-[#dda174]">
          {evaluation.warnings.length} diagnostic {evaluation.warnings.length === 1 ? 'warning' : 'warnings'} (non-blocking)
        </p>
      )}
      {unavailable && <p className="cm-chart-note">Model metrics could not be refreshed.</p>}
      <Link className="cm-text-link" to="/models">Review model metrics →</Link>
    </div>
  );
}
