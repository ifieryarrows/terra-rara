import { memo, useId, useMemo, useState } from 'react';
import { Brain } from 'lucide-react';
import type { ExplainAttentionPoint, ExplainFeatureEntry, ExplainRecent, ExplainResponse } from '../types';
import { driverCategoryLabel, driverCategoryTone } from '../utils/driverCategories';

/** Keep the newest encoder-window days visible when the window is longer. */
const ATTENTION_BAR_LIMIT = 60;
/** The Market drivers card shows five rows as well. */
const ENCODER_ENTRY_LIMIT = 5;
const PEAK_CHIP_LIMIT = 3;
const EMPTY_STATE_TEXT = 'Model focus will appear after the next model refresh.';

const STAGE_WIDTH = 360;
const STAGE_HEIGHT = 132;
const WAVE_LEFT = 10;
const WAVE_RIGHT = 308;
const CORE_X = 338;
const CORE_Y = 66;

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function signed(value: number, precision: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(precision)}`;
}

/** Five-session delta line; returns null (omitted) when the payload cannot back it. */
function recentLine(recent: ExplainRecent | undefined): string | null {
  if (!recent) return null;
  const basis = recent.basis || '5 sessions';
  if (recent.kind === 'pct') {
    const pct = finiteNumber(recent.change_pct);
    return pct === null ? null : `${signed(pct, 2)}% over last ${basis}`;
  }
  const change = finiteNumber(recent.change);
  return change === null ? null : `${signed(change, 3)} over last ${basis}`;
}

function slotX(index: number, total: number): number {
  if (total <= 1) return (WAVE_LEFT + WAVE_RIGHT) / 2;
  return WAVE_LEFT + (index / (total - 1)) * (WAVE_RIGHT - WAVE_LEFT);
}

function buildSmoothPath(points: Array<{ x: number; y: number }>): string {
  if (points.length === 0) return '';
  if (points.length === 1) return `M ${points[0].x.toFixed(1)},${points[0].y.toFixed(1)}`;
  let d = `M ${points[0].x.toFixed(1)},${points[0].y.toFixed(1)}`;
  for (let i = 0; i < points.length - 1; i += 1) {
    const curr = points[i];
    const next = points[i + 1];
    const midX = ((curr.x + next.x) / 2).toFixed(1);
    d += ` C ${midX},${curr.y.toFixed(1)} ${midX},${next.y.toFixed(1)} ${next.x.toFixed(1)},${next.y.toFixed(1)}`;
  }
  return d;
}

function FeatureTemplateCard({ entry, rank, total }: { entry: ExplainFeatureEntry; rank: number; total: number }) {
  const weight = finiteNumber(entry.weight);
  const recent = recentLine(entry.recent);
  const category = entry.category ? (driverCategoryLabel[entry.category] || driverCategoryLabel.Other) : 'Signal';
  const horizonText = entry.time_horizon ? `${entry.time_horizon} window` : 'Encoder input';
  return (
    <div className="cm-focus-template-card" role="region" aria-label="Selected encoder signal details" data-testid="model-focus-feature-card">
      <div className="cm-focus-template-head">
        <span>VSN ENCODER FOCUS · #{rank} OF {total}</span>
        <span>{category.toUpperCase()} · {horizonText.toUpperCase()}</span>
      </div>
      <div className="cm-focus-template-title">
        <strong>{entry.label || entry.feature}</strong>
        {entry.feature && <code className="cm-focus-template-code">{entry.feature}</code>}
      </div>
      {entry.description && <p className="text-xs text-slate-300 leading-relaxed">{entry.description}</p>}
      <div className="cm-focus-template-metrics">
        <div>
          <small>Selection Share</small>
          <span>{weight === null ? '—' : `${(weight * 100).toFixed(1)}%`}</span>
        </div>
        <div>
          <small>Recent Shift</small>
          <span>{recent ?? '—'}</span>
        </div>
        <div>
          <small>Observed As Of</small>
          <span>{entry.recent?.as_of || 'Latest window'}</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Raw TFT-ASRO model focus for the latest forecast: top VSN encoder features
 * with their recent five-session delta, the encoder-window attention profile
 * and the raw payload. `data === null` is the normal "not built yet" state;
 * `error` carries the load failure when one occurred.
 */
export const ModelFocusPanel = memo(function ModelFocusPanel({ data, error }: { data: ExplainResponse | null; error?: string | null }) {
  const uid = useId().replace(/:/g, '');
  const [hoveredFeatureIdx, setHoveredFeatureIdx] = useState<number | null>(null);
  const [selectedSessionIdx, setSelectedSessionIdx] = useState<number | null>(null);

  const encoderEntries = Array.isArray(data?.variable_selection?.encoder)
    ? data.variable_selection.encoder.filter((entry) => entry && typeof entry === 'object')
    : [];
  const topEncoder = encoderEntries.slice(0, ENCODER_ENTRY_LIMIT);
  const attentionBars = useMemo(
    () => (Array.isArray(data?.attention?.series) ? data.attention.series : []).slice(-ATTENTION_BAR_LIMIT),
    [data?.attention?.series],
  );
  const maxEncoderWeight = topEncoder.reduce((max, entry) => Math.max(max, finiteNumber(entry.weight) ?? 0), 0);
  const maxAttentionWeight = attentionBars.reduce((max, point) => Math.max(max, finiteNumber(point?.weight) ?? 0), 0);
  const totalAttentionWeight = attentionBars.reduce((sum, point) => sum + (finiteNumber(point?.weight) ?? 0), 0);
  const meanAttentionWeight = attentionBars.length > 0 ? (totalAttentionWeight > 0 ? totalAttentionWeight / attentionBars.length : 1 / attentionBars.length) : 0;
  const rawJson = useMemo(() => (data ? JSON.stringify(data, null, 2) : ''), [data]);

  const rankedSessions = useMemo(() => {
    return attentionBars
      .map((point, index) => ({
        index,
        date: point?.date ?? null,
        weight: finiteNumber(point?.weight) ?? 0,
      }))
      .sort((a, b) => b.weight - a.weight || b.index - a.index);
  }, [attentionBars]);

  const peakSessions = useMemo(() => rankedSessions.slice(0, PEAK_CHIP_LIMIT), [rankedSessions]);
  const rankByIndex = useMemo(() => {
    const map = new Map<number, number>();
    rankedSessions.forEach((item, rankIdx) => map.set(item.index, rankIdx + 1));
    return map;
  }, [rankedSessions]);

  const waveGeometry = useMemo(() => {
    if (attentionBars.length === 0) return null;
    const upperPts: Array<{ x: number; y: number; ratio: number }> = [];
    const lowerPts: Array<{ x: number; y: number }> = [];
    for (let i = 0; i < attentionBars.length; i += 1) {
      const weight = finiteNumber(attentionBars[i]?.weight) ?? 0;
      const ratio = maxAttentionWeight > 0 ? Math.max(0, Math.min(1, weight / maxAttentionWeight)) : 0;
      const x = slotX(i, attentionBars.length);
      const upperAmp = 6 + ratio * 42;
      const lowerAmp = 4 + ratio * 26;
      upperPts.push({ x, y: CORE_Y - upperAmp, ratio });
      lowerPts.push({ x, y: CORE_Y + lowerAmp });
    }
    const upperLine = buildSmoothPath(upperPts);
    const lowerReversed = [...lowerPts].reverse();
    const lowerLine = buildSmoothPath(lowerReversed);
    const envelopePath = `${upperLine} L ${lowerReversed[0].x.toFixed(1)},${lowerReversed[0].y.toFixed(1)} ${lowerLine.replace(/^M [^C]+/, '')} Z`;
    return { upperPts, lowerPts, upperLine, envelopePath };
  }, [attentionBars, maxAttentionWeight]);

  if (!data || (topEncoder.length === 0 && attentionBars.length === 0)) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center gap-2" role={error ? 'alert' : 'status'}>
        <Brain size={24} className="text-slate-400" aria-hidden="true" />
        <p className="text-xs text-slate-400">{error ? 'Model focus could not be loaded.' : EMPTY_STATE_TEXT}</p>
        {error && <p className="text-xs text-slate-500 break-words max-w-full">{error}</p>}
      </div>
    );
  }

  const notes = typeof data.notes === 'string' ? data.notes : '';
  const referenceDate = typeof data.reference_price_date === 'string' ? data.reference_price_date : '';
  const horizonLabel = typeof data.horizon?.label === 'string' ? data.horizon.label : '';
  // The explain payload states which forecast day the attention series belongs
  // to; older payloads without `step` fall back to a horizon-less heading.
  const horizonIndex = finiteNumber(data.attention?.step?.horizon_index);
  const stepDayNumber = Number.isInteger(horizonIndex) && (horizonIndex as number) >= 0 ? (horizonIndex as number) + 1 : null;
  const attentionHeading =
    stepDayNumber !== null
      ? `Attention · forecast day ${stepDayNumber} (t+${stepDayNumber})`
      : 'Attention';

  const defaultSessionIdx = peakSessions[0]?.index ?? Math.max(0, attentionBars.length - 1);
  const activeSessionIdx =
    selectedSessionIdx !== null && selectedSessionIdx >= 0 && selectedSessionIdx < attentionBars.length
      ? selectedSessionIdx
      : defaultSessionIdx;
  const activeSession: ExplainAttentionPoint | undefined = attentionBars[activeSessionIdx];
  const activeSessionWeight = finiteNumber(activeSession?.weight) ?? 0;
  const activeSessionRank = rankByIndex.get(activeSessionIdx) ?? 1;
  const activeSessionOffset = attentionBars.length - 1 - activeSessionIdx;
  const activeRelativeMultiple = meanAttentionWeight > 0 ? activeSessionWeight / meanAttentionWeight : 1;
  const windowStart = data.attention?.window?.start || attentionBars[0]?.date || null;
  const windowEnd = data.attention?.window?.end || attentionBars[attentionBars.length - 1]?.date || null;
  const activeFeature = hoveredFeatureIdx !== null ? topEncoder[hoveredFeatureIdx] : null;

  return (
    <div className="space-y-4">
      {topEncoder.length > 0 && (
        <div className="space-y-2.5" onMouseLeave={() => setHoveredFeatureIdx(null)}>
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-mono uppercase tracking-widest text-slate-400">
              VSN Encoder Selection
            </span>
            <span className="text-xs font-mono text-slate-400">
              Top {topEncoder.length} inputs
            </span>
          </div>
          <div className="space-y-1.5">
            {topEncoder.map((entry, index) => {
              const weight = finiteNumber(entry.weight);
              const recent = recentLine(entry.recent);
              const isActive = hoveredFeatureIdx === index;
              return (
                <div
                  key={entry.feature || index}
                  tabIndex={0}
                  data-active={isActive ? 'true' : undefined}
                  className="cm-focus-vsn-row group"
                  onMouseEnter={() => setHoveredFeatureIdx(index)}
                  onFocus={() => setHoveredFeatureIdx(index)}
                  onBlur={() => setHoveredFeatureIdx(null)}
                >
                  <div className="flex justify-between items-start mb-1 gap-3">
                    <div className="flex items-start gap-2 min-w-0 flex-1">
                      {entry.category && (
                        <span className={`text-xs px-1.5 py-0.5 rounded ${driverCategoryTone[entry.category] || driverCategoryTone.Other} font-medium tracking-wide shrink-0`}>
                          {driverCategoryLabel[entry.category] || driverCategoryLabel.Other}
                        </span>
                      )}
                      <span className="text-xs text-gray-300 group-hover:text-copper-400 transition-colors min-w-0 flex-1 whitespace-normal break-words leading-relaxed">
                        {entry.label || entry.feature}
                      </span>
                    </div>
                    <span className="text-xs font-mono text-slate-400 shrink-0">
                      {weight === null ? '—' : `${(weight * 100).toFixed(1)}%`}
                    </span>
                  </div>
                  <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-copper-500 to-copper-400"
                      style={{ transformOrigin: 'left', transform: `scaleX(${maxEncoderWeight > 0 && weight !== null ? Math.max(0, Math.min(1, weight / maxEncoderWeight)) : 0})` }}
                    />
                  </div>
                  {recent && <p className="mt-1 text-xs text-slate-400 font-mono">{recent}</p>}
                </div>
              );
            })}
          </div>
          {activeFeature && hoveredFeatureIdx !== null && (
            <FeatureTemplateCard entry={activeFeature} rank={hoveredFeatureIdx + 1} total={topEncoder.length} />
          )}
        </div>
      )}

      {attentionBars.length > 0 && (
        <div className="cm-focus-attention-shell">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs text-slate-300 uppercase tracking-widest font-medium">{attentionHeading}</p>
              <span className="text-xs font-mono text-copper-400">
                {attentionBars.length}D memory → {stepDayNumber ? `T+${stepDayNumber} anchor` : '5D horizon'}
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Temporal resonance across the {attentionBars.length}-session encoder lookback
              {windowStart && windowEnd ? ` (${windowStart} to ${windowEnd})` : ''}. Curved copper filaments trace the historical sessions the network weighted most heavily to anchor its forecast step.
            </p>
          </div>

          <div
            className="cm-focus-attention-stage"
            onMouseLeave={() => setSelectedSessionIdx(null)}
          >
            {waveGeometry && (
              <svg
                viewBox={`0 0 ${STAGE_WIDTH} ${STAGE_HEIGHT}`}
                preserveAspectRatio="none"
                className="cm-focus-attention-svg"
                aria-hidden="true"
              >
                <defs>
                  <linearGradient id={`${uid}-wave`} x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="var(--cm-forecast)" stopOpacity="0.16" />
                    <stop offset="65%" stopColor="var(--cm-forecast)" stopOpacity="0.28" />
                    <stop offset="100%" stopColor="var(--cm-copper)" stopOpacity="0.38" />
                  </linearGradient>
                  <linearGradient id={`${uid}-stroke`} x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="var(--cm-forecast)" stopOpacity="0.72" />
                    <stop offset="70%" stopColor="var(--cm-forecast)" stopOpacity="0.92" />
                    <stop offset="100%" stopColor="var(--cm-copper)" stopOpacity="1" />
                  </linearGradient>
                </defs>

                {/* Topographic reference rules */}
                <line x1={WAVE_LEFT} x2={WAVE_RIGHT} y1="26" y2="26" stroke="var(--cm-border)" strokeDasharray="2 6" strokeOpacity="0.55" />
                <line x1={WAVE_LEFT} x2={CORE_X} y1={CORE_Y} y2={CORE_Y} stroke="var(--cm-copper)" strokeOpacity="0.28" strokeWidth="1" />
                <line x1={WAVE_LEFT} x2={WAVE_RIGHT} y1="106" y2="106" stroke="var(--cm-border)" strokeDasharray="2 6" strokeOpacity="0.55" />

                {/* Mirrored interferometric memory envelope */}
                <path d={waveGeometry.envelopePath} fill={`url(#${uid}-wave)`} />
                <path d={waveGeometry.upperLine} fill="none" stroke={`url(#${uid}-stroke)`} strokeWidth="1.6" strokeLinejoin="round" />

                {/* Session nodal spines */}
                {waveGeometry.upperPts.map((pt, idx) => {
                  const lowPt = waveGeometry.lowerPts[idx];
                  const isPeak = peakSessions.some((p) => p.index === idx);
                  const isSelected = idx === activeSessionIdx;
                  return (
                    <line
                      key={idx}
                      x1={pt.x}
                      x2={pt.x}
                      y1={pt.y}
                      y2={lowPt.y}
                      stroke={isSelected || isPeak ? 'var(--cm-copper)' : 'var(--cm-forecast)'}
                      strokeOpacity={isSelected ? 0.95 : isPeak ? 0.68 : 0.18 + pt.ratio * 0.35}
                      strokeWidth={isSelected ? 1.8 : isPeak ? 1.25 : 0.8}
                    />
                  );
                })}

                {/* Curved signal arcs from top memory peaks + active session into the T+1 Forecast Core */}
                {peakSessions.map((peak, rankIdx) => {
                  const pt = waveGeometry.upperPts[peak.index];
                  if (!pt) return null;
                  const midX = (pt.x + CORE_X) / 2;
                  const archY = Math.max(8, Math.min(pt.y, CORE_Y) - (16 + (PEAK_CHIP_LIMIT - rankIdx) * 6));
                  const arcPath = `M ${pt.x.toFixed(1)},${pt.y.toFixed(1)} Q ${midX.toFixed(1)},${archY.toFixed(1)} ${CORE_X},${CORE_Y}`;
                  const isSelected = peak.index === activeSessionIdx;
                  return (
                    <path
                      key={`arc-${peak.index}`}
                      d={arcPath}
                      fill="none"
                      stroke="var(--cm-copper)"
                      strokeWidth={isSelected ? 1.75 : 1.05}
                      strokeOpacity={isSelected ? 0.92 : 0.38}
                      strokeDasharray={isSelected ? undefined : '3 4'}
                    />
                  );
                })}

                {!peakSessions.some((p) => p.index === activeSessionIdx) && waveGeometry.upperPts[activeSessionIdx] && (() => {
                  const pt = waveGeometry.upperPts[activeSessionIdx];
                  const midX = (pt.x + CORE_X) / 2;
                  const archY = Math.max(10, pt.y - 18);
                  return (
                    <path
                      d={`M ${pt.x.toFixed(1)},${pt.y.toFixed(1)} Q ${midX.toFixed(1)},${archY.toFixed(1)} ${CORE_X},${CORE_Y}`}
                      fill="none"
                      stroke="var(--cm-copper)"
                      strokeWidth="1.65"
                      strokeOpacity="0.9"
                    />
                  );
                })()}

                {/* Active session crosshair and node */}
                {waveGeometry.upperPts[activeSessionIdx] && (() => {
                  const pt = waveGeometry.upperPts[activeSessionIdx];
                  return (
                    <g>
                      <line x1={pt.x} x2={pt.x} y1="8" y2={STAGE_HEIGHT - 8} stroke="var(--cm-copper)" strokeOpacity="0.45" strokeDasharray="2 3" />
                      <circle cx={pt.x} cy={pt.y} r="4.2" fill="var(--cm-bg)" stroke="var(--cm-copper)" strokeWidth="1.8" />
                      <circle cx={pt.x} cy={pt.y} r="1.8" fill="var(--cm-copper)" />
                    </g>
                  );
                })()}

                {/* Right-side T+1 / 5D Forecast Anchor Core */}
                <g transform={`translate(${CORE_X} ${CORE_Y})`}>
                  <circle r="16" fill="none" stroke="var(--cm-copper)" strokeOpacity="0.22" strokeDasharray="2 3" />
                  <circle r="10" fill="none" stroke="var(--cm-copper)" strokeOpacity="0.48" />
                  <circle r="4.2" fill="var(--cm-copper)" />
                  <text x="0" y="-21" textAnchor="middle" fill="var(--cm-copper)" fontSize="8.5" fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace" fontWeight="600">
                    {stepDayNumber ? `T+${stepDayNumber}` : '5D'}
                  </text>
                  <text x="0" y="27" textAnchor="middle" fill="var(--cm-muted)" fontSize="7.5" fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace">
                    ANCHOR
                  </text>
                </g>
              </svg>
            )}

            {/* Interactive 60-session hit-strip (no native browser title attribute) */}
            <div className="cm-model-focus-attention" role="group" aria-label="Encoder attention sessions">
              {attentionBars.map((point, index) => {
                const weight = finiteNumber(point?.weight) ?? 0;
                const ratio = maxAttentionWeight > 0 ? weight / maxAttentionWeight : 0;
                const label = `${point?.date ?? '—'}: ${(weight * 100).toFixed(1)}%`;
                const isSelected = index === activeSessionIdx;
                return (
                  <span
                    key={`${point?.date ?? 'day'}-${index}`}
                    role="button"
                    tabIndex={0}
                    aria-label={label}
                    aria-pressed={isSelected}
                    data-active={isSelected ? 'true' : undefined}
                    data-date={point?.date ?? ''}
                    style={{ height: `${Math.max(2, ratio * 100)}%` }}
                    data-testid="model-focus-attention"
                    onMouseEnter={() => setSelectedSessionIdx(index)}
                    onFocus={() => setSelectedSessionIdx(index)}
                    onClick={() => setSelectedSessionIdx(index)}
                    onKeyDown={(e) => {
                      if (e.key === 'ArrowLeft') {
                        e.preventDefault();
                        setSelectedSessionIdx(Math.max(0, index - 1));
                      } else if (e.key === 'ArrowRight') {
                        e.preventDefault();
                        setSelectedSessionIdx(Math.min(attentionBars.length - 1, index + 1));
                      }
                    }}
                  />
                );
              })}
            </div>
          </div>

          {/* Interactive Peak Memory Chips */}
          {peakSessions.length > 0 && (
            <div className="cm-focus-peak-strip" role="group" aria-label="Peak attention sessions">
              <span className="text-xs font-mono uppercase tracking-wider text-slate-400 mr-1">
                Peak sessions:
              </span>
              {peakSessions.map((peak, idx) => {
                const isSelected = peak.index === activeSessionIdx;
                return (
                  <button
                    key={`${peak.date ?? 'peak'}-${peak.index}`}
                    type="button"
                    className="cm-focus-peak-chip"
                    aria-pressed={isSelected}
                    onMouseEnter={() => setSelectedSessionIdx(peak.index)}
                    onFocus={() => setSelectedSessionIdx(peak.index)}
                    onClick={() => setSelectedSessionIdx(peak.index)}
                  >
                    <b>#{idx + 1}</b>
                    <span>{peak.date ?? `Day ${peak.index + 1}`}</span>
                    <span>{(peak.weight * 100).toFixed(1)}%</span>
                  </button>
                );
              })}
            </div>
          )}

          {/* Custom CopperMind Session Template Card */}
          <div
            className="cm-focus-template-card cm-focus-template-card--forecast"
            role="status"
            aria-live="polite"
            data-testid="model-focus-attention-card"
          >
            <div className="cm-focus-template-head">
              <span>
                {activeSessionRank <= PEAK_CHIP_LIMIT ? `PEAK MEMORY ANCHOR · #${activeSessionRank}` : `SESSION MEMORY · RANK #${activeSessionRank} OF ${attentionBars.length}`}
              </span>
              <span>{activeSessionOffset === 0 ? 'REFERENCE CLOSE (T-0)' : `LOOKBACK T-${activeSessionOffset}D`}</span>
            </div>
            <div className="cm-focus-template-title">
              <strong>{activeSession?.date ?? `Session ${activeSessionIdx + 1}`}</strong>
              <code className="cm-focus-template-code">
                {stepDayNumber ? `Feeds T+${stepDayNumber} → 5D outlook` : 'Feeds 5D outlook'}
              </code>
            </div>
            <div className="cm-focus-template-metrics">
              <div>
                <small>Attention Share</small>
                <span>{(activeSessionWeight * 100).toFixed(1)}%</span>
              </div>
              <div>
                <small>Relative Density</small>
                <span>{`${activeRelativeMultiple.toFixed(2)}× avg`}</span>
              </div>
              <div>
                <small>Window Span</small>
                <span>{attentionBars.length} sessions</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {referenceDate && <p className="cm-chart-note">{horizonLabel ? `Reference ${referenceDate} · predicts ${horizonLabel}` : `Reference ${referenceDate}`}</p>}
      {notes && <p className="cm-chart-note">{notes}</p>}

      <details className="cm-data-disclosure">
        <summary>Raw JSON</summary>
        <pre>{rawJson}</pre>
      </details>
    </div>
  );
});


