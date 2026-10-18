import { memo, useMemo } from 'react';
import { Brain } from 'lucide-react';
import type { ExplainRecent, ExplainResponse } from '../types';
import { driverCategoryLabel, driverCategoryTone } from '../utils/driverCategories';

/** Keep the newest encoder-window days visible when the window is longer. */
const ATTENTION_BAR_LIMIT = 60;
/** The Market drivers card shows five rows as well. */
const ENCODER_ENTRY_LIMIT = 5;
const EMPTY_STATE_TEXT = 'Model focus will appear after the next model refresh.';

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

/**
 * Raw TFT-ASRO model focus for the latest forecast: top VSN encoder features
 * with their recent five-session delta, the encoder-window attention profile
 * and the raw payload. `data === null` is the normal "not built yet" state;
 * `error` carries the load failure when one occurred.
 */
export const ModelFocusPanel = memo(function ModelFocusPanel({ data, error }: { data: ExplainResponse | null; error?: string | null }) {
  const encoderEntries = Array.isArray(data?.variable_selection?.encoder)
    ? data.variable_selection.encoder.filter((entry) => entry && typeof entry === 'object')
    : [];
  const topEncoder = encoderEntries.slice(0, ENCODER_ENTRY_LIMIT);
  const attentionBars = (Array.isArray(data?.attention?.series) ? data.attention.series : []).slice(-ATTENTION_BAR_LIMIT);
  const maxEncoderWeight = topEncoder.reduce((max, entry) => Math.max(max, finiteNumber(entry.weight) ?? 0), 0);
  const maxAttentionWeight = attentionBars.reduce((max, point) => Math.max(max, finiteNumber(point?.weight) ?? 0), 0);
  const rawJson = useMemo(() => (data ? JSON.stringify(data, null, 2) : ''), [data]);

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
  const attentionHeading =
    Number.isInteger(horizonIndex) && (horizonIndex as number) >= 0
      ? `Attention · forecast day ${(horizonIndex as number) + 1} (t+${(horizonIndex as number) + 1})`
      : 'Attention';

  return (
    <div className="space-y-4">
      {topEncoder.length > 0 && (
        <div className="space-y-4">
          {topEncoder.map((entry, index) => {
            const weight = finiteNumber(entry.weight);
            const recent = recentLine(entry.recent);
            return (
              <div key={entry.feature || index} className="group" title={entry.feature || undefined}>
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
      )}

      {attentionBars.length > 0 && (
        <div>
          <p className="text-xs text-slate-400 uppercase tracking-widest mb-2">{attentionHeading}</p>
          <div className="cm-model-focus-attention">
            {attentionBars.map((point, index) => {
              const weight = finiteNumber(point?.weight) ?? 0;
              const ratio = maxAttentionWeight > 0 ? weight / maxAttentionWeight : 0;
              return (
                <span
                  key={`${point?.date ?? 'day'}-${index}`}
                  style={{ height: `${Math.max(2, ratio * 100)}%` }}
                  title={`${point?.date ?? '—'}: ${(weight * 100).toFixed(1)}%`}
                  data-testid="model-focus-attention"
                />
              );
            })}
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

