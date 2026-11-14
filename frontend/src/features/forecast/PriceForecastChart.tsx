import { memo, useCallback, useEffect, useId, useMemo, useState, useSyncExternalStore } from 'react';
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { HistoryDataPoint, TFTAnalysisResponse } from '../../types';
import { chartTokens as theme } from '../../design/chart-tokens';
import { DataTable } from '../../components/ui/DataTable';
import { ViewState } from '../../components/ui/ViewState';
import { finite, formatChartDate, formatPrice, formatReturnPct, preparePriceChart, type PriceChartRow } from './chart-data';

const HISTORY_WINDOWS = [30, 90, 180] as const;
type PreparedPriceChart = ReturnType<typeof preparePriceChart>;

interface DotRenderProps {
  cx?: number;
  cy?: number;
  index?: number;
  payload?: PriceChartRow;
}

interface ChartHoverStore {
  getSnapshot: () => PriceChartRow | null;
  set: (row: PriceChartRow | null) => void;
  queue: (row: PriceChartRow | null) => void;
  cancel: () => void;
  subscribe: (listener: () => void) => () => void;
}

function createHoverStore(): ChartHoverStore {
  let current: PriceChartRow | null = null;
  let pending: PriceChartRow | null = null;
  let raf = 0;
  const listeners = new Set<() => void>();
  const set = (next: PriceChartRow | null) => {
    if (current === next || (current && next && current.date === next.date && current.isForecast === next.isForecast)) return;
    current = next;
    listeners.forEach(fn => fn());
  };
  const cancel = () => {
    if (raf) {
      window.cancelAnimationFrame(raf);
      raf = 0;
    }
    pending = null;
  };
  return {
    getSnapshot: () => current,
    set,
    queue: (row) => {
      pending = row;
      if (raf) return;
      raf = window.requestAnimationFrame(() => {
        raf = 0;
        set(pending);
      });
    },
    cancel,
    subscribe: (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}

function impulseToneClass(impulse: string | null | undefined): string {
  const normalized = impulse?.toUpperCase();
  if (normalized === 'BULLISH') return 'text-emerald-300';
  if (normalized === 'BEARISH') return 'text-rose-300';
  return 'text-slate-300';
}

function returnToneClass(value: number | null | undefined): string {
  if (!finite(value) || Math.abs(value * 100) < 0.005) return 'text-slate-300';
  return value > 0 ? 'text-emerald-300' : 'text-rose-300';
}

export function PriceTooltip({ active, payload, onActiveRowChange }: {
  active?: boolean;
  payload?: Array<{ payload: PriceChartRow }>;
  onActiveRowChange?: (row: PriceChartRow | null) => void;
}) {
  const row = payload?.[0]?.payload ?? null;
  useEffect(() => {
    if (!onActiveRowChange) return;
    onActiveRowChange(active && row ? row : null);
  }, [active, row, onActiveRowChange]);

  if (!active || !row) return null;
  const hasConviction = Array.isArray(row.priceConvictionRange) && row.priceConvictionRange.length === 2;
  const hasReturnRange = finite(row.returnQ10) && finite(row.returnQ90);
  const horizonTag = row.isForecast
    ? (finite(row.day) ? `T+${row.day} · 5D Path` : 'Daily forecast · USD')
    : 'Observed close · USD';

  return (
    <div className="cm-chart-tooltip rounded-lg border border-white/10 bg-[#060a11]/94 px-3.5 py-2.5 text-xs text-slate-100 backdrop-blur-md">
      <div className="mb-1.5 flex items-center justify-between gap-4 border-b border-white/10 pb-1.5">
        <p className="font-medium text-slate-100">{formatChartDate(row.date)}</p>
        <strong className={row.isForecast ? 'font-semibold tracking-wider text-[#8eadde] uppercase' : 'font-semibold tracking-wider text-[#dda174] uppercase'}>
          {horizonTag}
        </strong>
      </div>
      <div className="space-y-1 tabular-nums">
        {finite(row.price) && <p className="text-slate-200">Close: {formatPrice(row.price)}</p>}
        {row.isForecast && (
          <>
            <div className="flex items-baseline justify-between gap-4">
              <p className="font-medium text-white">Median: {formatPrice(row.priceMedian)}</p>
              {finite(row.cumulativeReturn) && (
                <span className={returnToneClass(row.cumulativeReturn)}>{formatReturnPct(row.cumulativeReturn)}</span>
              )}
            </div>
            {hasConviction && (
              <p className="text-slate-300">
                Core Q25–Q75: {formatPrice(row.priceConvictionRange![0])} – {formatPrice(row.priceConvictionRange![1])}
              </p>
            )}
            <div className="flex items-baseline justify-between gap-3 text-slate-300">
              <p>Q10: {formatPrice(row.priceQ10)}</p>
              <p>Q90: {formatPrice(row.priceQ90)}</p>
            </div>
            {hasReturnRange && (
              <p className="border-t border-white/10 pt-1 text-[11px] text-slate-400">
                Return envelope: {formatReturnPct(row.returnQ10)} to {formatReturnPct(row.returnQ90)}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/** Isolated readout subscriber so pointer/keyboard hover updates only this strip. */
const ChartHoverReadout = memo(function ChartHoverReadout({
  store,
  chart,
}: {
  store: ChartHoverStore;
  chart: PreparedPriceChart;
}) {
  const activeRow = useSyncExternalStore(store.subscribe, store.getSnapshot, () => null);
  const forecastRow = activeRow
    ? (activeRow.isForecast ? activeRow : null)
    : (chart.hasForecast ? chart.horizonRow : null);
  const medianRet = activeRow?.isForecast
    ? activeRow.cumulativeReturn
    : (chart.weeklyReturn ?? chart.horizonRow?.cumulativeReturn ?? null);
  const q10Ret = activeRow?.isForecast
    ? activeRow.returnQ10
    : (chart.weeklyQ10Return ?? chart.horizonRow?.returnQ10 ?? null);
  const q90Ret = activeRow?.isForecast
    ? activeRow.returnQ90
    : (chart.weeklyQ90Return ?? chart.horizonRow?.returnQ90 ?? null);
  const horizonTitle = activeRow
    ? (finite(activeRow.day) ? `T+${activeRow.day} · 5D Path` : 'Forecast')
    : `5D Horizon · ${forecastRow ? formatChartDate(forecastRow.date, true) : ''}`;

  return (
    <div className="mb-2 flex shrink-0 flex-wrap items-center justify-between gap-x-6 gap-y-2 border-b border-white/[0.07] pb-2.5 text-xs tabular-nums">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        {forecastRow ? (
          <>
            {activeRow && <span className="font-medium text-slate-200">{formatChartDate(activeRow.date)}</span>}
            <span className="text-[11px] font-semibold tracking-wider text-[#8eadde] uppercase">{horizonTitle}</span>
            <span className="text-slate-100">
              Q50 <strong className="font-semibold text-white">{formatPrice(forecastRow.priceMedian)}</strong>{' '}
              <span className={returnToneClass(medianRet)}>{formatReturnPct(medianRet)}</span>
            </span>
            <span className="text-slate-400">
              Q10–Q90 <span className="text-slate-200">{formatPrice(forecastRow.priceQ10)} – {formatPrice(forecastRow.priceQ90)}</span>
              {finite(q10Ret) && finite(q90Ret) && (
                <span className="ml-1.5 text-slate-400">({formatReturnPct(q10Ret)} to {formatReturnPct(q90Ret)})</span>
              )}
            </span>
          </>
        ) : activeRow ? (
          <>
            <span className="font-medium text-slate-200">{formatChartDate(activeRow.date)}</span>
            <span className="text-[11px] font-semibold tracking-wider text-[#dda174] uppercase">Observed Close</span>
            <span className="text-slate-100">
              Close <strong className="font-semibold text-white">{formatPrice(activeRow.price)}</strong>
            </span>
          </>
        ) : (
          <>
            <span className="text-[11px] font-semibold tracking-wider text-[#dda174] uppercase">
              Last Close · {chart.lastDate ? formatChartDate(chart.lastDate, true) : '—'}
            </span>
            <span className="font-semibold text-white">{formatPrice(chart.lastPrice)}</span>
            <span className="text-slate-400">
              {chart.degraded ? '5D forecast unavailable (degraded)' : 'Historical closes only'}
            </span>
          </>
        )}
      </div>

      {/* T+1 Diagnostic Impulse kept cleanly separate from the 5D Horizon */}
      {chart.hasForecast && (chart.t1Impulse || finite(chart.t1Return)) && (
        <div className="flex items-center gap-2 border-l border-white/10 pl-3.5 text-[11px]">
          <span className="tracking-wider text-slate-400 uppercase">T+1 Diagnostic</span>
          <span className={`font-semibold tracking-wide ${impulseToneClass(chart.t1Impulse)}`}>
            {chart.t1Impulse ?? 'NEUTRAL'}
            {finite(chart.t1Return) ? ` ${formatReturnPct(chart.t1Return)}` : ''}
          </span>
        </div>
      )}
    </div>
  );
});

const ChartDataDisclosure = memo(function ChartDataDisclosure({ rows }: { rows: PriceChartRow[] }) {
  const [tableOpen, setTableOpen] = useState(false);
  return (
    <details className="cm-chart-table" onToggle={event => setTableOpen(event.currentTarget.open)}>
      <summary>View chart data</summary>
      {tableOpen && (
        <DataTable caption="Values in USD for the selected historical window and available daily forecast. Hidden chart series remain in this table.">
          <thead>
            <tr>
              {['Date', 'Observed', 'Forecast median', 'Q10', 'Q90', 'Cum. return'].map(label => (
                <th key={label} scope="col">{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={`${row.date}-${index}`}>
                <th scope="row">{row.date.slice(0, 10)}</th>
                {[
                  formatPrice(row.price),
                  formatPrice(row.isForecast ? row.priceMedian : null),
                  formatPrice(row.isForecast ? row.priceQ10 : null),
                  formatPrice(row.isForecast ? row.priceQ90 : null),
                  formatReturnPct(row.isForecast ? row.cumulativeReturn : null),
                ].map((value, column) => (
                  <td key={column}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </DataTable>
      )}
    </details>
  );
});

export const PriceForecastChart = memo(function PriceForecastChart({ history, forecast, historyError }: {
  history: HistoryDataPoint[]; forecast: TFTAnalysisResponse | null; historyError?: boolean; forecastError?: boolean;
}) {
  const id = useId();
  const uid = id.replace(/:/g, '');
  const reducedMotion = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  const [count, setCount] = useState(30);
  const [medianVisible, setMedianVisible] = useState(true);
  const [rangeVisible, setRangeVisible] = useState(true);
  const chart = useMemo(() => preparePriceChart(history, forecast, count), [history, forecast, count]);
  const [animateSeries, setAnimateSeries] = useState(true);
  const hoverStore = useMemo(createHoverStore, []);
  const yDecimals = chart.domain[1] - chart.domain[0] < 0.08 ? 3 : 2;
  const horizonDate = chart.horizonRow?.date;
  const forecastCount = chart.rows.filter(r => r.isForecast).length;
  const animateActive = !reducedMotion && animateSeries;

  // Draw the entry animation once — including the moment the forecast lands —
  // then let refreshes redraw in place without synchronous useLayoutEffect thrashing.
  useEffect(() => {
    if (!animateSeries || chart.rows.length === 0) return;
    const delay = chart.hasForecast ? 1600 : 2500;
    const timer = window.setTimeout(() => setAnimateSeries(false), delay);
    return () => window.clearTimeout(timer);
  }, [animateSeries, chart.rows.length, chart.hasForecast]);

  useEffect(() => {
    hoverStore.set(null);
    return hoverStore.cancel;
  }, [chart, hoverStore]);

  const renderBridgeDot = useCallback((props: DotRenderProps) => {
    const { cx, cy, payload, index } = props;
    if (!chart.hasForecast || !payload || payload.date !== chart.lastDate || !finite(cx) || !finite(cy)) {
      return <g key={`b-${index ?? 0}`} />;
    }
    return (
      <g key={`bridge-${payload.date}`}>
        <circle cx={cx} cy={cy} r={6.5} fill="rgba(221, 161, 116, 0.22)" />
        <circle cx={cx} cy={cy} r={3.5} fill="#dda174" stroke="#060a11" strokeWidth={1.5} />
      </g>
    );
  }, [chart.hasForecast, chart.lastDate]);

  const renderHorizonDot = useCallback((props: DotRenderProps) => {
    const { cx, cy, payload, index } = props;
    if (!payload?.isForecast || payload.date !== horizonDate || !finite(cx) || !finite(cy)) {
      return <g key={`h-${index ?? 0}`} />;
    }
    return (
      <g key={`horizon-${payload.date}`}>
        <circle cx={cx} cy={cy} r={8} fill="rgba(142, 173, 222, 0.2)" />
        <circle cx={cx} cy={cy} r={4} fill="#8eadde" stroke="#060a11" strokeWidth={1.75} />
      </g>
    );
  }, [horizonDate]);

  const handleChartMouseMove = useCallback((state: unknown) => {
    const next = state as { activeTooltipIndex?: number; activePayload?: Array<{ payload?: PriceChartRow }> } | null;
    const nextRow = next?.activePayload?.[0]?.payload ?? (typeof next?.activeTooltipIndex === 'number' ? chart.rows[next.activeTooltipIndex] ?? null : null);
    hoverStore.queue(nextRow);
  }, [chart.rows, hoverStore]);

  const handleChartMouseLeave = useCallback(() => {
    hoverStore.cancel();
    hoverStore.set(null);
  }, [hoverStore]);

  if (!chart.rows.length) {
    return (
      <ViewState
        kind={historyError ? 'error' : 'empty'}
        title={historyError ? 'Price history could not be loaded' : 'No chart data available'}
        description="Historical closes are needed to anchor the daily forecast. Use Refresh overview to check again."
        compact
      />
    );
  }

  return (
    <div className="cm-price-chart">
      {/* Corner reticle accents for precision observatory coordinate framing */}
      <span className="cm-reticle cm-reticle--tl" aria-hidden="true">+</span>
      <span className="cm-reticle cm-reticle--tr" aria-hidden="true">+</span>
      <span className="cm-reticle cm-reticle--bl" aria-hidden="true">+</span>
      <span className="cm-reticle cm-reticle--br" aria-hidden="true">+</span>

      {/* Micro-segmented telemetry bar with window buttons & telemetry indicators */}
      <div className="cm-chart-toolbar cm-chart-telemetry-bar">
        <div className="cm-telemetry-segment cm-telemetry-segment--window">
          <fieldset className="cm-chart-window">
            <legend>WINDOW</legend>
            <div className="cm-chart-window-options">
              {HISTORY_WINDOWS.map(size => (
                <button
                  key={size}
                  type="button"
                  aria-pressed={count === size}
                  aria-label={`Last ${size} daily closes`}
                  onClick={() => {
                    setAnimateSeries(false);
                    setCount(size);
                  }}
                >
                  {size}<span>D</span>
                </button>
              ))}
            </div>
          </fieldset>
        </div>
        <div className="cm-telemetry-segment cm-telemetry-segment--telemetry" aria-hidden="true">
          <span className="cm-telemetry-pill">
            <span className="cm-telemetry-dot cm-telemetry-dot--live" />
            <span>{chart.rows.length} PTS</span>
          </span>
          <span className="cm-telemetry-pill">
            <span>{chart.hasForecast ? '5D TFT-ASRO' : 'HISTORICAL'}</span>
          </span>
        </div>
      </div>

      {/* Laser series indicators */}
      <div className="cm-chart-legend" role="group" aria-label="Chart series">
        <span>
          <i className="cm-chart-key cm-chart-key--observed cm-laser-indicator cm-laser-indicator--copper" aria-hidden="true" />
          Observed close
        </span>
        <button
          type="button"
          aria-pressed={medianVisible}
          disabled={!chart.hasForecast}
          onClick={() => setMedianVisible(value => !value)}
        >
          <i className="cm-chart-key cm-chart-key--median cm-laser-indicator cm-laser-indicator--median" aria-hidden="true" />
          Forecast median
        </button>
        <button
          type="button"
          aria-pressed={rangeVisible}
          disabled={!chart.hasForecast}
          title="Outer Q10–Q90 (80% interval) and inner Q25–Q75 conviction envelope"
          onClick={() => setRangeVisible(value => !value)}
        >
          <i className="cm-chart-key cm-chart-key--range cm-laser-indicator cm-laser-indicator--range" aria-hidden="true" />
          Q10–Q90 range
        </button>
      </div>

      <div
        className="cm-price-plot flex flex-col"
        role="group"
        aria-label="Copper historical prices and daily forecast"
        aria-describedby={`${id}-help`}
      >
        <ChartHoverReadout store={hoverStore} chart={chart} />

        <div className="min-h-0 w-full flex-1">
          <ResponsiveContainer width="100%" height="100%" debounce={60}>
            <ComposedChart
              accessibilityLayer
              data={chart.rows}
              margin={{ top: 22, right: 8, left: 0, bottom: 4 }}
              onMouseMove={handleChartMouseMove}
              onMouseLeave={handleChartMouseLeave}
            >
              <defs>
                <linearGradient id={`${uid}-copper-area`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#dda174" stopOpacity={0.2} />
                  <stop offset="65%" stopColor="#dda174" stopOpacity={0.04} />
                  <stop offset="100%" stopColor="#dda174" stopOpacity={0} />
                </linearGradient>
                <linearGradient id={`${uid}-fan-outer`} x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#8eadde" stopOpacity={0.05} />
                  <stop offset="100%" stopColor="#8eadde" stopOpacity={0.13} />
                </linearGradient>
                <linearGradient id={`${uid}-fan-inner`} x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#8eadde" stopOpacity={0.11} />
                  <stop offset="100%" stopColor="#8eadde" stopOpacity={0.23} />
                </linearGradient>
              </defs>

              <CartesianGrid stroke="rgba(142, 173, 222, 0.06)" vertical={false} strokeDasharray="3 6" />
              <XAxis
                dataKey="date"
                tick={{ fill: theme.text, fontSize: 11 }}
                tickFormatter={value => formatChartDate(value, true)}
                axisLine={false}
                tickLine={false}
                minTickGap={30}
                interval="preserveStartEnd"
              />
              <YAxis
                orientation="right"
                domain={chart.domain}
                tick={{ fill: theme.text, fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={value => `$${value.toFixed(yDecimals)}`}
                width={62}
              />
              <Tooltip
                cursor={{ stroke: 'rgba(221, 161, 116, 0.4)', strokeWidth: 1, strokeDasharray: '3 3' }}
                content={<PriceTooltip onActiveRowChange={hoverStore.set} />}
                isAnimationActive={false}
                wrapperStyle={{ zIndex: 20, pointerEvents: 'none' }}
              />

              {/* Historical close curve & subtle gradient fill */}
              <Area
                isAnimationActive={animateActive}
                type="linear"
                dataKey="price"
                stroke={theme.copper}
                fill={`url(#${uid}-copper-area)`}
                strokeWidth={2.2}
                dot={renderBridgeDot}
                activeDot={{ r: 4.5, fill: '#dda174', stroke: '#060a11', strokeWidth: 2 }}
                connectNulls={false}
              />

              {/* Outer Q10–Q90 & inner Q25–Q75 quantile envelopes */}
              {rangeVisible && (
                <>
                  <Area
                    isAnimationActive={false}
                    type="linear"
                    dataKey="priceRange"
                    stroke="none"
                    fill={`url(#${uid}-fan-outer)`}
                    connectNulls={false}
                    tooltipType="none"
                  />
                  {(['priceQ10', 'priceQ90'] as const).map(key => (
                    <Line
                      key={key}
                      isAnimationActive={false}
                      type="linear"
                      dataKey={key}
                      stroke={theme.forecast}
                      strokeWidth={1}
                      strokeDasharray="2 4"
                      strokeOpacity={0.55}
                      dot={false}
                      activeDot={false}
                      connectNulls={false}
                    />
                  ))}
                  <Area
                    isAnimationActive={false}
                    type="linear"
                    dataKey="priceConvictionRange"
                    stroke="none"
                    fill={`url(#${uid}-fan-inner)`}
                    connectNulls={false}
                    tooltipType="none"
                  />
                </>
              )}

              {/* Crisp Q50 median trajectory */}
              {medianVisible && (
                <Line
                  isAnimationActive={animateActive}
                  type="linear"
                  dataKey="priceMedian"
                  stroke="#8eadde"
                  strokeWidth={2.4}
                  strokeDasharray="5 3"
                  dot={renderHorizonDot}
                  activeDot={{ r: 5, fill: '#8eadde', stroke: '#060a11', strokeWidth: 2 }}
                  connectNulls={false}
                />
              )}

              {chart.hasForecast && (
                <ReferenceLine
                  x={chart.lastDate}
                  stroke="rgba(221, 161, 116, 0.38)"
                  strokeDasharray="3 4"
                  label={{ value: 'Last close', position: 'insideTopLeft', fill: '#dda174', fontSize: 11 }}
                />
              )}
              {chart.hasForecast && forecastCount >= 3 && horizonDate && (
                <ReferenceLine
                  x={horizonDate}
                  stroke="rgba(142, 173, 222, 0.28)"
                  strokeDasharray="2 4"
                  label={{ value: '5D Horizon', position: 'insideTopRight', fill: '#8eadde', fontSize: 11 }}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Explanatory footnote styled as observatory telemetry note with info reticle */}
      <p id={`${id}-help`} className="cm-chart-note cm-telemetry-note">
        <span className="cm-reticle cm-reticle--info" aria-hidden="true">⌖</span>
        <span>
          Hover or use the chart’s arrow keys to inspect a date. On touch screens, the data table provides every value.
          Outer Q10–Q90 is the 80% interval with an inner Q25–Q75 conviction core; outcomes can fall outside both bands.
        </span>
      </p>

      <ChartDataDisclosure rows={chart.rows} />
    </div>
  );
});
