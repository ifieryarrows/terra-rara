import { memo, useCallback, useId, useMemo, useState } from 'react';
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
const WAVE_LEFT = 12;
const WAVE_RIGHT = 348;
const PEAK_Y = 18;
const BASELINE_Y = 104;

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

interface WavePoint {
  x: number;
  y: number;
  ratio: number;
}

interface WaveGeometry {
  pts: WavePoint[];
  curveLine: string;
  areaPath: string;
  uniformY: number;
}

interface PeakSession {
  index: number;
  date: string | null;
  weight: number;
}

function FeatureTemplateCard({
  entry,
  rank,
  total,
}: {
  entry: ExplainFeatureEntry;
  rank: number;
  total: number;
}) {
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

function FeatureSlotPlaceholder({ count }: { count: number }) {
  return (
    <div className="cm-focus-template-card" role="region" aria-label="Encoder signal inspector">
      <div className="cm-focus-template-head">
        <span>VSN ENCODER INSPECTOR · {count} SIGNALS</span>
        <span>HOVER OR FOCUS ROW</span>
      </div>
      <div className="cm-focus-template-title">
        <strong>Select an encoder input above</strong>
      </div>
      <p className="text-xs text-slate-400 leading-relaxed">
        Inspect variable selection share, five-session shift, and encoder window metadata.
      </p>
    </div>
  );
}

/** Isolates VSN row hover/focus state so it never re-renders the attention profile below. */
const VsnEncoderSection = memo(function VsnEncoderSection({
  topEncoder,
  maxEncoderWeight,
}: {
  topEncoder: ExplainFeatureEntry[];
  maxEncoderWeight: number;
}) {
  const [hoveredFeatureIdx, setHoveredFeatureIdx] = useState<number | null>(null);
  const activeFeature = hoveredFeatureIdx !== null ? topEncoder[hoveredFeatureIdx] : null;

  return (
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
      <div className="cm-focus-feature-slot">
        {activeFeature && hoveredFeatureIdx !== null ? (
          <FeatureTemplateCard
            entry={activeFeature}
            rank={hoveredFeatureIdx + 1}
            total={topEncoder.length}
          />
        ) : (
          <FeatureSlotPlaceholder count={topEncoder.length} />
        )}
      </div>
    </div>
  );
});

/** Static SVG layer (grid, density wave, 60 pillars); never re-renders on bar hover. */
const AttentionWaveBackdrop = memo(function AttentionWaveBackdrop({
  uid,
  waveGeometry,
  peakIndexSet,
  attentionLength,
  windowStart,
  stepDayNumber,
}: {
  uid: string;
  waveGeometry: WaveGeometry;
  peakIndexSet: Set<number>;
  attentionLength: number;
  windowStart: string | null;
  stepDayNumber: number | null;
}) {
  return (
    <>
      <defs>
        <linearGradient id={`${uid}-wave`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--cm-copper)" stopOpacity="0.30" />
          <stop offset="55%" stopColor="var(--cm-forecast)" stopOpacity="0.16" />
          <stop offset="100%" stopColor="var(--cm-forecast)" stopOpacity="0.01" />
        </linearGradient>
        <linearGradient id={`${uid}-stroke`} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="var(--cm-forecast)" stopOpacity="0.68" />
          <stop offset="65%" stopColor="var(--cm-forecast)" stopOpacity="0.90" />
          <stop offset="100%" stopColor="var(--cm-copper)" stopOpacity="1" />
        </linearGradient>
      </defs>

      {[0.25, 0.5, 0.75].map((q) => {
        const qx = WAVE_LEFT + (WAVE_RIGHT - WAVE_LEFT) * q;
        return (
          <line
            key={q}
            x1={qx}
            x2={qx}
            y1={PEAK_Y - 4}
            y2={BASELINE_Y}
            stroke="var(--cm-border)"
            strokeOpacity="0.35"
            strokeWidth="0.75"
          />
        );
      })}

      <line
        x1={WAVE_LEFT}
        x2={WAVE_RIGHT}
        y1={waveGeometry.uniformY}
        y2={waveGeometry.uniformY}
        stroke="var(--cm-muted)"
        strokeDasharray="3 4"
        strokeOpacity="0.45"
        strokeWidth="0.9"
      />
      <line
        x1={WAVE_LEFT}
        x2={WAVE_RIGHT}
        y1={BASELINE_Y}
        y2={BASELINE_Y}
        stroke="var(--cm-border)"
        strokeOpacity="0.85"
        strokeWidth="1"
      />

      <path d={waveGeometry.areaPath} fill={`url(#${uid}-wave)`} />
      <path
        d={waveGeometry.curveLine}
        fill="none"
        stroke={`url(#${uid}-stroke)`}
        strokeWidth="1.5"
        strokeLinejoin="round"
      />

      {waveGeometry.pts.map((pt, idx) => {
        const isPeak = peakIndexSet.has(idx);
        return (
          <g key={idx}>
            <line
              x1={pt.x}
              x2={pt.x}
              y1={pt.y}
              y2={BASELINE_Y}
              stroke={isPeak ? 'var(--cm-copper)' : 'var(--cm-forecast)'}
              strokeOpacity={isPeak ? 0.82 : 0.24 + pt.ratio * 0.34}
              strokeWidth={isPeak ? 1.85 : 1.15}
              strokeLinecap="round"
            />
            {isPeak && (
              <circle
                cx={pt.x}
                cy={pt.y}
                r="2.2"
                fill="var(--cm-copper)"
                fillOpacity="0.9"
              />
            )}
          </g>
        );
      })}

      <text x={WAVE_LEFT} y="121" fill="var(--cm-muted)" fontSize="8" fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace">
        {windowStart || `-${attentionLength}D`}
      </text>
      <text x={(WAVE_LEFT + WAVE_RIGHT) / 2} y="121" textAnchor="middle" fill="var(--cm-muted)" fontSize="7.5" fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace" opacity="0.82">
        DASHED FLOOR = UNIFORM WEIGHT
      </text>
      <text x={WAVE_RIGHT} y="121" textAnchor="end" fill="var(--cm-copper)" fontSize="8" fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace" fontWeight="600">
        {stepDayNumber ? `TARGET T+${stepDayNumber}` : 'TARGET 5D'}
      </text>
    </>
  );
});

const AttentionHitBar = memo(function AttentionHitBar({
  index,
  date,
  label,
  heightPct,
  isSelected,
  onSelect,
  onStep,
}: {
  index: number;
  date: string;
  label: string;
  heightPct: number;
  isSelected: boolean;
  onSelect: (index: number) => void;
  onStep: (index: number, delta: number) => void;
}) {
  return (
    <span
      role="button"
      tabIndex={0}
      aria-label={label}
      aria-pressed={isSelected}
      data-active={isSelected ? 'true' : undefined}
      data-date={date}
      style={{ height: `${heightPct}%` }}
      data-testid="model-focus-attention"
      onMouseEnter={() => onSelect(index)}
      onFocus={() => onSelect(index)}
      onClick={() => onSelect(index)}
      onKeyDown={(e) => {
        if (e.key === 'ArrowLeft') {
          e.preventDefault();
          onStep(index, -1);
        } else if (e.key === 'ArrowRight') {
          e.preventDefault();
          onStep(index, 1);
        }
      }}
    />
  );
});

const AttentionProfileSection = memo(function AttentionProfileSection({
  uid,
  attentionBars,
  maxAttentionWeight,
  meanAttentionWeight,
  peakSessions,
  rankByIndex,
  waveGeometry,
  attentionHeading,
  stepDayNumber,
  windowStart,
  windowEnd,
}: {
  uid: string;
  attentionBars: ExplainAttentionPoint[];
  maxAttentionWeight: number;
  meanAttentionWeight: number;
  peakSessions: PeakSession[];
  rankByIndex: Map<number, number>;
  waveGeometry: WaveGeometry | null;
  attentionHeading: string;
  stepDayNumber: number | null;
  windowStart: string | null;
  windowEnd: string | null;
}) {
  const [selectedSessionIdx, setSelectedSessionIdx] = useState<number | null>(null);
  const handleSelect = useCallback((index: number) => setSelectedSessionIdx(index), []);
  const handleStep = useCallback(
    (index: number, delta: number) =>
      setSelectedSessionIdx(Math.max(0, Math.min(attentionBars.length - 1, index + delta))),
    [attentionBars.length],
  );

  const peakIndexSet = useMemo(() => new Set(peakSessions.map((p) => p.index)), [peakSessions]);

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
  const activePt = waveGeometry?.pts[activeSessionIdx];

  return (
    <div className="cm-focus-attention-shell">
      <div className="space-y-1">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs text-slate-300 uppercase tracking-widest font-medium">{attentionHeading}</p>
          <span className="text-xs font-mono text-copper-400">
            {attentionBars.length}D memory → {stepDayNumber ? `T+${stepDayNumber} anchor` : '5D horizon'}
          </span>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          Distribution of encoder lookback attention across {attentionBars.length} sessions
          {windowStart && windowEnd ? ` (${windowStart} to ${windowEnd})` : ''}. Bars above the dashed floor mark dates the model weighted most heavily.
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
            <AttentionWaveBackdrop
              uid={uid}
              waveGeometry={waveGeometry}
              peakIndexSet={peakIndexSet}
              attentionLength={attentionBars.length}
              windowStart={windowStart}
              stepDayNumber={stepDayNumber}
            />
            {activePt && (
              <g>
                <line
                  x1={activePt.x}
                  x2={activePt.x}
                  y1={activePt.y}
                  y2={BASELINE_Y}
                  stroke="#f7dcb8"
                  strokeOpacity="0.98"
                  strokeWidth="2.4"
                  strokeLinecap="round"
                />
                <line
                  x1={activePt.x}
                  x2={activePt.x}
                  y1="8"
                  y2={BASELINE_Y + 4}
                  stroke="#f7dcb8"
                  strokeOpacity="0.45"
                  strokeDasharray="2 3"
                />
                <circle cx={activePt.x} cy={activePt.y} r="4.2" fill="var(--cm-bg)" stroke="#f7dcb8" strokeWidth="1.75" />
                <circle cx={activePt.x} cy={activePt.y} r="1.8" fill="var(--cm-copper)" />
              </g>
            )}
          </svg>
        )}

        <div className="cm-model-focus-attention" role="group" aria-label="Encoder attention sessions">
          {attentionBars.map((point, index) => {
            const weight = finiteNumber(point?.weight) ?? 0;
            const ratio = maxAttentionWeight > 0 ? weight / maxAttentionWeight : 0;
            return (
              <AttentionHitBar
                key={`${point?.date ?? 'day'}-${index}`}
                index={index}
                date={point?.date ?? ''}
                label={`${point?.date ?? '—'}: ${(weight * 100).toFixed(1)}%`}
                heightPct={Math.max(2, ratio * 100)}
                isSelected={index === activeSessionIdx}
                onSelect={handleSelect}
                onStep={handleStep}
              />
            );
          })}
        </div>
      </div>

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
  );
});

/**
 * Raw TFT-ASRO model focus for the latest forecast: top VSN encoder features
 * with their recent five-session delta, the encoder-window attention profile
 * and the raw payload. `data === null` is the normal "not built yet" state;
 * `error` carries the load failure when one occurred.
 */
export const ModelFocusPanel = memo(function ModelFocusPanel({ data, error }: { data: ExplainResponse | null; error?: string | null }) {
  const uid = useId().replace(/:/g, '');
  const [jsonOpen, setJsonOpen] = useState(false);

  const topEncoder = useMemo(() => {
    const entries = Array.isArray(data?.variable_selection?.encoder)
      ? data.variable_selection.encoder.filter((entry) => entry && typeof entry === 'object')
      : [];
    return entries.slice(0, ENCODER_ENTRY_LIMIT);
  }, [data?.variable_selection?.encoder]);

  const attentionBars = useMemo(
    () => (Array.isArray(data?.attention?.series) ? data.attention.series : []).slice(-ATTENTION_BAR_LIMIT),
    [data?.attention?.series],
  );
  const maxEncoderWeight = topEncoder.reduce((max, entry) => Math.max(max, finiteNumber(entry.weight) ?? 0), 0);
  const maxAttentionWeight = attentionBars.reduce((max, point) => Math.max(max, finiteNumber(point?.weight) ?? 0), 0);
  const totalAttentionWeight = attentionBars.reduce((sum, point) => sum + (finiteNumber(point?.weight) ?? 0), 0);
  const meanAttentionWeight = attentionBars.length > 0 ? (totalAttentionWeight > 0 ? totalAttentionWeight / attentionBars.length : 1 / attentionBars.length) : 0;
  const rawJson = useMemo(() => (jsonOpen && data ? JSON.stringify(data, null, 2) : ''), [data, jsonOpen]);

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

  const waveGeometry = useMemo<WaveGeometry | null>(() => {
    if (attentionBars.length === 0) return null;
    const pts: WavePoint[] = [];
    const spanY = BASELINE_Y - PEAK_Y;
    for (let i = 0; i < attentionBars.length; i += 1) {
      const weight = finiteNumber(attentionBars[i]?.weight) ?? 0;
      const ratio = maxAttentionWeight > 0 ? Math.max(0, Math.min(1, weight / maxAttentionWeight)) : 0;
      const x = slotX(i, attentionBars.length);
      const barH = 3 + ratio * (spanY - 3);
      pts.push({ x, y: BASELINE_Y - barH, ratio });
    }
    const curveLine = buildSmoothPath(pts);
    const first = pts[0];
    const last = pts[pts.length - 1];
    const areaPath = `${curveLine} L ${last.x.toFixed(1)},${BASELINE_Y} L ${first.x.toFixed(1)},${BASELINE_Y} Z`;
    const meanRatio = maxAttentionWeight > 0 ? Math.max(0, Math.min(1, meanAttentionWeight / maxAttentionWeight)) : 0;
    const uniformY = BASELINE_Y - (3 + meanRatio * (spanY - 3));
    return { pts, curveLine, areaPath, uniformY };
  }, [attentionBars, maxAttentionWeight, meanAttentionWeight]);

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

  const windowStart = data.attention?.window?.start || attentionBars[0]?.date || null;
  const windowEnd = data.attention?.window?.end || attentionBars[attentionBars.length - 1]?.date || null;

  return (
    <div className="space-y-4">
      {topEncoder.length > 0 && (
        <VsnEncoderSection topEncoder={topEncoder} maxEncoderWeight={maxEncoderWeight} />
      )}

      {attentionBars.length > 0 && (
        <AttentionProfileSection
          uid={uid}
          attentionBars={attentionBars}
          maxAttentionWeight={maxAttentionWeight}
          meanAttentionWeight={meanAttentionWeight}
          peakSessions={peakSessions}
          rankByIndex={rankByIndex}
          waveGeometry={waveGeometry}
          attentionHeading={attentionHeading}
          stepDayNumber={stepDayNumber}
          windowStart={windowStart}
          windowEnd={windowEnd}
        />
      )}

      {referenceDate && <p className="cm-chart-note">{horizonLabel ? `Reference ${referenceDate} · predicts ${horizonLabel}` : `Reference ${referenceDate}`}</p>}
      {notes && <p className="cm-chart-note">{notes}</p>}

      <details className="cm-data-disclosure" onToggle={(e) => setJsonOpen(e.currentTarget.open)}>
        <summary>Raw JSON</summary>
        {jsonOpen && <pre>{rawJson}</pre>}
      </details>
    </div>
  );
});


