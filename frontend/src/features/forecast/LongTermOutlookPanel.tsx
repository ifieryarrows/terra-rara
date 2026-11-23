import { useMemo, useState } from 'react';
import { Layers, ShieldCheck } from 'lucide-react';
import clsx from 'clsx';
import type {
  LongTermHorizonForecast,
  LongTermHorizonKey,
  LongTermMacroPillar,
  LongTermOutlookResponse,
  LongTermStance,
} from '../../types';
import { DEFAULT_COPPER_SYMBOL } from '../../config/instruments';
import { useLongTermOutlook } from '../../hooks/useQueries';
import { SegmentedTabs, type SegmentedTabOption } from '../../components/ui/SegmentedTabs';
import { ViewState } from '../../components/ui/ViewState';
import { RefreshButton } from '../../components/ui/RefreshButton';

const CANONICAL_PILLAR_ORDER = [
  'trend_structure',
  'usd_macro',
  'china_industrial_demand',
  'cross_asset_confirmation',
] as const;

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function formatPrice(value: number | null | undefined): string {
  if (!finite(value)) return '—';
  return `$${value.toFixed(2)}`;
}

function formatSignedPct(pctValue: number | null | undefined, digits = 2): string {
  if (!finite(pctValue)) return '—';
  if (Math.abs(pctValue) < 0.005) return `0.${'0'.repeat(digits)}%`;
  return `${pctValue > 0 ? '+' : ''}${pctValue.toFixed(digits)}%`;
}

function formatSignedDecimalReturn(decimalValue: number | null | undefined, digits = 2): string {
  if (!finite(decimalValue)) return '—';
  return formatSignedPct(decimalValue * 100, digits);
}

function stanceMeta(stance?: LongTermStance | string | null): {
  glyph: string;
  textClass: string;
  barColor: string;
  label: string;
} {
  const norm = stance?.toUpperCase();
  if (norm === 'BULLISH') {
    return {
      glyph: '▲',
      textClass: 'text-[#74c7a9]',
      barColor: 'bg-[#74c7a9]/80',
      label: 'BULLISH',
    };
  }
  if (norm === 'BEARISH') {
    return {
      glyph: '▼',
      textClass: 'text-[#db8296]',
      barColor: 'bg-[#db8296]/80',
      label: 'BEARISH',
    };
  }
  return {
    glyph: '■',
    textClass: 'text-[#dda174]',
    barColor: 'bg-[#dda174]/80',
    label: 'NEUTRAL',
  };
}

function positionOnRange(value: number, low: number, high: number): number {
  if (!finite(value) || !finite(low) || !finite(high) || high <= low) return 50;
  return Math.max(2, Math.min(98, ((value - low) / (high - low)) * 100));
}

function horizonTabLabel(key: string, forecast?: LongTermHorizonForecast): string {
  if (forecast?.horizon_label) return forecast.horizon_label;
  if (key === '20D') return '1 Month (20D)';
  if (key === '60D') return '1 Quarter (60D)';
  return key;
}

export interface LongTermOutlookPanelProps {
  data?: LongTermOutlookResponse | null;
  isLoading?: boolean;
  isError?: boolean;
  isFetching?: boolean;
  onRefresh?: () => void;
}

export function LongTermOutlookPanel({
  data,
  isLoading = false,
  isError = false,
  isFetching = false,
  onRefresh,
}: LongTermOutlookPanelProps) {
  const availableHorizons = useMemo(() => {
    const keys = (data?.available_horizons ?? []).filter(
      (k): k is LongTermHorizonKey => k === '20D' || k === '60D',
    );
    if (keys.length > 0) return keys;
    const dictKeys = Object.keys(data?.horizons ?? {}).filter(
      (k): k is LongTermHorizonKey => k === '20D' || k === '60D',
    );
    return dictKeys.length > 0 ? dictKeys : (['20D', '60D'] as LongTermHorizonKey[]);
  }, [data?.available_horizons, data?.horizons]);

  const preferredDefault: LongTermHorizonKey =
    data?.default_horizon === '60D' && availableHorizons.includes('60D')
      ? '60D'
      : availableHorizons[0] ?? '20D';

  const [selectedHorizon, setSelectedHorizon] = useState<LongTermHorizonKey>(preferredDefault);

  const activeHorizonKey: LongTermHorizonKey = availableHorizons.includes(selectedHorizon)
    ? selectedHorizon
    : preferredDefault;

  const horizonOptions: readonly SegmentedTabOption<LongTermHorizonKey>[] = useMemo(
    () =>
      availableHorizons.map((key) => {
        const forecast = data?.horizons?.[key];
        return {
          value: key,
          label: horizonTabLabel(key, forecast),
          badge: forecast ? `${forecast.horizon_weeks}W` : key === '20D' ? '4W' : '12W',
        };
      }),
    [availableHorizons, data?.horizons],
  );

  const activeForecast = data?.horizons?.[activeHorizonKey] ?? null;
  const activeValidation = data?.validation?.[activeHorizonKey] ?? null;

  const orderedPillars = useMemo(() => {
    const pillarsMap = data?.macro_scorecard?.pillars;
    if (!pillarsMap) return [];
    const seen = new Set<string>();
    const list: LongTermMacroPillar[] = [];
    for (const key of CANONICAL_PILLAR_ORDER) {
      const p = pillarsMap[key];
      if (p) {
        list.push(p);
        seen.add(key);
      }
    }
    for (const [key, p] of Object.entries(pillarsMap)) {
      if (!seen.has(key) && p) list.push(p);
    }
    return list;
  }, [data?.macro_scorecard?.pillars]);

  if (isLoading && !data) {
    return (
      <ViewState
        kind="loading"
        title="Loading 1M / 1Q long-term outlook"
        description="Retrieving latest persisted structural forecast and macro scorecard."
        compact
      />
    );
  }

  if (isError && !data) {
    return (
      <ViewState
        kind="error"
        title="Long-term outlook could not be loaded"
        description="Check again to retrieve the 1-Month (20D) and 1-Quarter (60D) structural outlook."
        action={onRefresh ? <RefreshButton onClick={onRefresh} busy={isFetching} label="Try again" /> : undefined}
        compact
      />
    );
  }

  if (!data || data.status === 'unavailable' || !activeForecast) {
    return (
      <ViewState
        kind="empty"
        title="Long-term outlook snapshot pending"
        description={
          data?.message ||
          '1-Month (20D) and 1-Quarter (60D) structural outlook snapshots will appear once the pipeline worker runs.'
        }
        action={onRefresh ? <RefreshButton onClick={onRefresh} busy={isFetching} label="Check snapshot" /> : undefined}
        compact
      />
    );
  }

  const stance = stanceMeta(activeForecast.stance);
  const moveToneClass =
    Math.abs(activeForecast.expected_return_pct) < 0.005
      ? 'text-slate-200'
      : activeForecast.expected_return_pct > 0
        ? 'text-[#74c7a9]'
        : 'text-[#db8296]';

  const probSum = activeForecast.prob_bullish + activeForecast.prob_neutral + activeForecast.prob_bearish;
  const normScale = probSum > 0 ? 100 / probSum : 0;
  const bullPct = probSum > 0 ? activeForecast.prob_bullish * normScale : 33.4;
  const neuPct = probSum > 0 ? activeForecast.prob_neutral * normScale : 33.3;
  const bearPct = probSum > 0 ? activeForecast.prob_bearish * normScale : 33.3;

  const q50Pct = positionOnRange(activeForecast.target_price_q50, activeForecast.price_q10, activeForecast.price_q90);
  const q25Pct = positionOnRange(activeForecast.price_q25, activeForecast.price_q10, activeForecast.price_q90);
  const q75Pct = positionOnRange(activeForecast.price_q75, activeForecast.price_q10, activeForecast.price_q90);
  const refTickPct =
    activeForecast.reference_price >= activeForecast.price_q10 &&
    activeForecast.reference_price <= activeForecast.price_q90
      ? positionOnRange(activeForecast.reference_price, activeForecast.price_q10, activeForecast.price_q90)
      : null;

  const scorecard = data.macro_scorecard;
  const compositeStance = stanceMeta(scorecard?.composite_stance);

  return (
    <div className="space-y-6" data-testid="long-term-outlook-panel">
      {/* Top Bar: Horizon Switcher + Purged Walk-Forward Validation Badge */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.07] pb-3">
        <SegmentedTabs
          value={activeHorizonKey}
          onChange={setSelectedHorizon}
          options={horizonOptions}
          ariaLabel="Long-term forecast horizon"
          size="sm"
          variant="underline"
        />

        {activeValidation ? (
          <div
            className="flex flex-wrap items-center gap-3 text-xs text-slate-300 cm-tabular"
            data-testid="long-term-validation-badge"
          >
            <span className="inline-flex items-center gap-1.5 text-[#74c7a9] font-medium">
              <ShieldCheck size={13} aria-hidden="true" />
              <span>
                Purged WF · {activeValidation.n_folds}F · gap {activeValidation.purge_gap_weeks}w · √
                {Math.round(activeValidation.annualisation)}
              </span>
            </span>
            <span className="text-slate-400">
              OOS DA <strong className="text-slate-100">{(activeValidation.directional_accuracy * 100).toFixed(1)}%</strong>
            </span>
            <span className="text-slate-400">
              MAE/Naive{' '}
              <strong className={activeValidation.mae_vs_naive_zero <= 1.0 ? 'cm-tone-good' : 'text-slate-100'}>
                {activeValidation.mae_vs_naive_zero.toFixed(3)}
              </strong>
            </span>
            <span className="text-slate-400">
              Sharpe{' '}
              <strong className={activeValidation.sharpe_ratio >= 0 ? 'cm-tone-good' : 'cm-tone-bad'}>
                {activeValidation.sharpe_ratio.toFixed(2)}
              </strong>
            </span>
            <span className="text-slate-400">
              PI80 <strong className="text-slate-100">{(activeValidation.pi80_coverage * 100).toFixed(0)}%</strong>
            </span>
          </div>
        ) : (
          <span className="text-xs text-slate-400">Purged walk-forward telemetry pending</span>
        )}
      </div>

      {/* Main Forecast Grid: Target / Range (Left) vs Regime Probabilities & Components (Right) */}
      <div
        className="grid gap-6 items-start"
        style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 19rem), 1fr))' }}
      >
        <div className="space-y-5 min-w-0">
          {/* Target Price & Expected Move Headline */}
          <div className="cm-weekly-headline-row">
            <div>
              <div className="mb-1.5 flex flex-wrap items-center gap-2 text-xs text-slate-400 uppercase tracking-widest">
                <span>Expected {activeForecast.horizon_label} Target</span>
                {activeForecast.target_date && (
                  <span className="text-xs text-slate-400 normal-case tracking-normal cm-tabular">
                    · Target {activeForecast.target_date}
                  </span>
                )}
              </div>
              <div className="flex flex-wrap items-baseline gap-3">
                <span className={clsx('text-4xl font-light tracking-tight cm-tabular', moveToneClass)} data-testid="lt-target-price">
                  {formatPrice(activeForecast.target_price_q50)}
                </span>
                <span className={clsx('text-lg font-medium cm-tabular', moveToneClass)} data-testid="lt-expected-move">
                  {formatSignedPct(activeForecast.expected_return_pct)}
                </span>
                <span className="text-xs text-slate-400 cm-tabular">
                  (from {formatPrice(activeForecast.reference_price)})
                </span>
              </div>
            </div>

            <div className={clsx('cm-consensus-readout', stance.textClass)} data-testid="lt-stance-readout">
              <span aria-hidden="true">{stance.glyph}</span>
              <span>{stance.label}</span>
            </div>
          </div>

          {/* Calibrated Price Range (Q10 – Q90 with Q25–Q75 Core) */}
          <div className="cm-weekly-range">
            <div className="cm-weekly-range-head">
              <span>Calibrated Price Range (Q10 – Q90)</span>
              <span className="cm-tabular">
                Core IQR: {formatPrice(activeForecast.price_q25)} – {formatPrice(activeForecast.price_q75)}
              </span>
            </div>

            <div className="cm-quantile-spectrum">
              <div className="cm-quantile-bound">
                <small>Q10 LOW</small>
                <span className="cm-tabular text-[#db8296]">{formatPrice(activeForecast.price_q10)}</span>
                <small className="cm-tabular">{formatSignedDecimalReturn(activeForecast.return_q10)}</small>
              </div>

              <div className="cm-quantile-bar-wrap" aria-hidden="true">
                <div className="cm-quantile-bar-track">
                  <div className="cm-quantile-bar-fill opacity-60" />
                  <div
                    className="absolute inset-y-0 rounded-full bg-[#8eadde]/65"
                    style={{ left: `${q25Pct}%`, width: `${Math.max(2, q75Pct - q25Pct)}%` }}
                  />
                  {refTickPct !== null && (
                    <span
                      className="cm-quantile-zero-tick"
                      style={{ left: `${refTickPct}%` }}
                      title={`Reference ${formatPrice(activeForecast.reference_price)}`}
                    />
                  )}
                  <span className="cm-quantile-median-dot" style={{ left: `${q50Pct}%` }} />
                </div>
              </div>

              <div className="cm-quantile-bound cm-quantile-bound--high">
                <small>Q90 HIGH</small>
                <span className="cm-tabular text-[#74c7a9]">{formatPrice(activeForecast.price_q90)}</span>
                <small className="cm-tabular">{formatSignedDecimalReturn(activeForecast.return_q90)}</small>
              </div>
            </div>
          </div>

          {/* Model Component Decomposition */}
          <div className="cm-weekly-meta-grid">
            <div>
              <span>Macro Ridge ({(activeForecast.components.macro_ridge_weight * 100).toFixed(0)}%)</span>
              <strong className="cm-tabular">{formatSignedDecimalReturn(activeForecast.components.macro_ridge_return)}</strong>
            </div>
            <div>
              <span>Damped Theta ({(activeForecast.components.theta_weight * 100).toFixed(0)}%)</span>
              <strong className="cm-tabular">{formatSignedDecimalReturn(activeForecast.components.theta_damped_return)}</strong>
            </div>
            <div>
              <span>Horizon Volatility (σ)</span>
              <strong className="cm-tabular">
                {(activeForecast.components.horizon_vol * 100).toFixed(1)}% · Ann. {(activeForecast.components.realized_vol_annualized * 100).toFixed(1)}%
              </strong>
            </div>
          </div>
        </div>

        {/* Right Column: Regime Stance Probability Bar + Key Drivers */}
        <div className="space-y-4 min-w-0">
          <div className="space-y-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium tracking-wider text-slate-300 uppercase">
                Regime Probability Breakdown
              </span>
              <span className="text-xs text-slate-300 cm-tabular" data-testid="lt-conviction-readout">
                {activeForecast.conviction_label} conviction · {Math.round(activeForecast.conviction * 100)}%
              </span>
            </div>

            <div
              className="flex h-2 w-full overflow-hidden rounded-full bg-white/[0.06]"
              role="img"
              aria-label={`Regime probabilities: Bullish ${bullPct.toFixed(0)}%, Neutral ${neuPct.toFixed(0)}%, Bearish ${bearPct.toFixed(0)}%`}
            >
              <div className="h-full bg-[#74c7a9]/80" style={{ width: `${bullPct}%` }} />
              <div className="h-full bg-[#dda174]/80" style={{ width: `${neuPct}%` }} />
              <div className="h-full bg-[#db8296]/80" style={{ width: `${bearPct}%` }} />
            </div>

            <div className="grid grid-cols-3 gap-2 text-xs cm-tabular">
              <div>
                <span className="text-slate-400 block text-xs">Bullish</span>
                <strong className="text-[#74c7a9] font-medium">{bullPct.toFixed(0)}%</strong>
              </div>
              <div className="text-center">
                <span className="text-slate-400 block text-xs">Neutral</span>
                <strong className="text-[#dda174] font-medium">{neuPct.toFixed(0)}%</strong>
              </div>
              <div className="text-right">
                <span className="text-slate-400 block text-xs">Bearish</span>
                <strong className="text-[#db8296] font-medium">{bearPct.toFixed(0)}%</strong>
              </div>
            </div>

            <p className="text-xs text-slate-400">
              Neutral regime band: ±{activeForecast.regime_threshold_pct.toFixed(1)}% around spot.
            </p>
          </div>

          {activeForecast.key_drivers.length > 0 && (
            <div className="border-t border-white/[0.06] pt-3 space-y-2">
              <span className="text-xs font-medium tracking-wider text-slate-400 uppercase block">
                Top {activeForecast.horizon_key} Macro Ridge Contributors
              </span>
              <div className="space-y-1.5">
                {activeForecast.key_drivers.slice(0, 4).map((drv) => {
                  const drvMeta = stanceMeta(drv.direction);
                  return (
                    <div key={drv.feature} className="flex items-center justify-between gap-2 text-xs">
                      <span className="text-slate-200 truncate" title={drv.feature}>
                        {drv.label}
                      </span>
                      <span className={clsx('shrink-0 font-medium cm-tabular', drvMeta.textClass)}>
                        {drvMeta.glyph} {formatSignedDecimalReturn(drv.contribution_return)} (z={drv.z_score >= 0 ? '+' : ''}
                        {drv.z_score.toFixed(1)})
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 4-Pillar Macro Factor Scorecard */}
      {orderedPillars.length > 0 && (
        <div className="border-t border-white/[0.07] pt-5 space-y-3" data-testid="long-term-macro-scorecard">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Layers size={14} className="text-[#dda174]" aria-hidden="true" />
              <h3 className="text-xs font-semibold tracking-wider text-slate-300 uppercase">
                4-Pillar Macro Factor Scorecard
              </h3>
            </div>
            {scorecard && (
              <span className={clsx('text-xs font-medium cm-tabular', compositeStance.textClass)}>
                Composite {compositeStance.glyph} {compositeStance.label} ({scorecard.composite_score >= 0 ? '+' : ''}
                {scorecard.composite_score.toFixed(2)})
              </span>
            )}
          </div>

          <div className="cm-metric-grid">
            {orderedPillars.map((pillar) => {
              const pMeta = stanceMeta(pillar.stance);
              return (
                <div
                  key={pillar.key}
                  className="border-t border-white/[0.07] pt-3 space-y-2.5 min-w-0"
                  data-testid={`macro-pillar-${pillar.key}`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-xs font-medium text-slate-100">{pillar.label}</span>
                    <span className={clsx('text-xs font-semibold shrink-0 cm-tabular', pMeta.textClass)}>
                      {pMeta.glyph} {pillar.score_pct}/100
                    </span>
                  </div>

                  <div className="h-1 w-full overflow-hidden rounded-full bg-white/[0.06]" aria-hidden="true">
                    <div
                      className={clsx('h-full w-full origin-left rounded-full', pMeta.barColor)}
                      style={{ transform: `scaleX(${clamp01(pillar.score_pct / 100)})` }}
                    />
                  </div>

                  <p className="text-xs leading-relaxed text-slate-400">{pillar.summary}</p>

                  {pillar.metrics.length > 0 && (
                    <div className="space-y-1 pt-1 border-t border-white/[0.06]">
                      {pillar.metrics.slice(0, 3).map((m) => {
                        const mTone = stanceMeta(m.signal);
                        return (
                          <div key={m.name} className="flex items-center justify-between gap-2 text-xs">
                            <span className="text-slate-400 truncate">{m.name}</span>
                            <span className={clsx('font-medium shrink-0 cm-tabular', mTone.textClass)}>
                              {m.formatted_value}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Provenance Footnote */}
      {(data.data_window || data.as_of_date) && (
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.06] pt-3 text-xs text-slate-400 cm-tabular">
          <span>
            {data.data_window
              ? `Weekly history ${data.data_window.start_date ?? '—'} to ${data.data_window.end_date ?? '—'} · ${data.data_window.weekly_bars} bars (${data.data_window.years_covered.toFixed(1)}y)${data.data_window.fred_macro_enabled ? ' · FRED macro' : ''}`
              : 'Structural weekly macro-trend ensemble'}
          </span>
          <span>
            {data.model_version ?? 'lt_structural_v1'}
            {data.as_of_date ? ` · As of ${data.as_of_date}` : ''}
          </span>
        </div>
      )}
    </div>
  );
}

export function ConnectedLongTermOutlookPanel({
  symbol = DEFAULT_COPPER_SYMBOL,
}: {
  symbol?: string;
}) {
  const { data, isLoading, isError, isFetching, refetch } = useLongTermOutlook(symbol);
  return (
    <LongTermOutlookPanel
      data={data}
      isLoading={isLoading}
      isError={isError}
      isFetching={isFetching}
      onRefresh={() => {
        void refetch();
      }}
    />
  );
}
