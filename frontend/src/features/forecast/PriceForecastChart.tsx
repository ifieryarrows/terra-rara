import { memo, useId, useMemo, useState } from 'react';
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useReducedMotion } from 'framer-motion';
import type { HistoryDataPoint, TFTAnalysisResponse } from '../../types';
import { chartTokens as theme } from '../../design/chart-tokens';
import { DataTable } from '../../components/ui/DataTable';
import { FilterChip } from '../../components/ui/FilterChip';
import { ViewState } from '../../components/ui/ViewState';
import { finite, formatChartDate, formatPrice, preparePriceChart, type PriceChartRow } from './chart-data';

export function PriceTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: PriceChartRow }> }) {
  const row = payload?.[0]?.payload;
  if (!active || !row) return null;
  return <div className="cm-chart-tooltip">
    <p>{formatChartDate(row.date)}</p>
    <strong>{row.isForecast ? 'Daily forecast · USD' : 'Observed close · USD'}</strong>
    {finite(row.price) && <p>Close: {formatPrice(row.price)}</p>}
    {row.isForecast && <><p>Median: {formatPrice(row.priceMedian)}</p><p>Q10: {formatPrice(row.priceQ10)}</p><p>Q90: {formatPrice(row.priceQ90)}</p></>}
  </div>;
}

export const PriceForecastChart = memo(function PriceForecastChart({ history, forecast, historyError, forecastError }: {
  history: HistoryDataPoint[]; forecast: TFTAnalysisResponse | null; historyError?: boolean; forecastError?: boolean;
}) {
  const id = useId();
  const reducedMotion = useReducedMotion();
  const [count, setCount] = useState(30);
  const [medianVisible, setMedianVisible] = useState(true);
  const [rangeVisible, setRangeVisible] = useState(true);
  const [tableOpen, setTableOpen] = useState(false);
  const chart = useMemo(() => preparePriceChart(history, forecast, count), [history, forecast, count]);
  if (!chart.rows.length) return <ViewState kind={historyError ? 'error' : 'empty'} title={historyError ? 'Price history could not be loaded' : 'No chart data available'} description="Historical closes are needed to anchor the daily forecast. Use Refresh overview to check again." compact/>;
  const status = chart.hasForecast
    ? 'Daily path · T+1 to T+5. The primary 5-day cumulative forecast is shown separately.'
    : chart.degraded ? 'Forecast is degraded. Historical closes remain available.'
    : forecastError ? 'Forecast could not be refreshed. Showing historical closes only.'
    : forecast?.prediction?.daily_forecasts?.length && !chart.aligned ? 'Waiting for a forecast based on the latest close. An older path is not attached to newer prices.'
    : 'No dated daily forecast is available. Showing historical closes only.';
  return <div className="cm-price-chart">
    <div className="cm-chart-toolbar">
      <fieldset className="cm-chart-period"><legend>Historical window</legend>{[30, 90, 180].map(size => <FilterChip key={size} active={count === size} onClick={() => setCount(size)}>{size} closes</FilterChip>)}</fieldset>
      <p className="cm-chart-meta">USD · {chart.historyCount} closes<br/>Last close: {formatChartDate(chart.lastDate!)}</p>
    </div>
    <p id={`${id}-description`} className="cm-chart-note" role="status">{status}</p>
    <div className="cm-chart-legend" role="group" aria-label="Chart series">
      <span><i className="cm-chart-key cm-chart-key--observed" aria-hidden="true"/>Observed close</span>
      <button type="button" aria-pressed={medianVisible} disabled={!chart.hasForecast} onClick={() => setMedianVisible(value => !value)}><i className="cm-chart-key cm-chart-key--median" aria-hidden="true"/>Forecast median</button>
      <button type="button" aria-pressed={rangeVisible} disabled={!chart.hasForecast} onClick={() => setRangeVisible(value => !value)}><i className="cm-chart-key cm-chart-key--range" aria-hidden="true"/>Q10–Q90 range</button>
    </div>
    <div className="cm-price-plot" role="group" aria-label="Copper historical prices and daily forecast" aria-describedby={`${id}-description ${id}-help`}>
      <ResponsiveContainer width="100%" height="100%" debounce={60}>
        <ComposedChart accessibilityLayer data={chart.rows} margin={{ top: 22, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid stroke={theme.grid} vertical={false} strokeDasharray="4 4"/>
          <XAxis dataKey="date" tick={{ fill: theme.text, fontSize: 12 }} tickFormatter={value => formatChartDate(value, true)} axisLine={false} tickLine={false} minTickGap={28} interval="preserveStartEnd"/>
          <YAxis orientation="right" domain={chart.domain} tick={{ fill: theme.text, fontSize: 12 }} axisLine={false} tickLine={false} tickFormatter={value => `$${value.toFixed(2)}`} width={58}/>
          <Tooltip content={<PriceTooltip/>} isAnimationActive={false} wrapperStyle={{ zIndex: 2, pointerEvents: 'none' }}/>
          <Area isAnimationActive={!reducedMotion} type="linear" dataKey="price" stroke={theme.copper} fill={theme.copper} fillOpacity={.08} strokeWidth={2} connectNulls={false}/>
          {rangeVisible && <Area isAnimationActive={false} type="linear" dataKey="priceRange" stroke="none" fill={theme.forecast} fillOpacity={.12} connectNulls={false} tooltipType="none"/>}
          {rangeVisible && <Line isAnimationActive={false} type="linear" dataKey="priceQ10" stroke={theme.forecast} strokeWidth={1} strokeDasharray="2 4" strokeOpacity={.7} dot={false} connectNulls={false}/>}
          {rangeVisible && <Line isAnimationActive={false} type="linear" dataKey="priceQ90" stroke={theme.forecast} strokeWidth={1} strokeDasharray="2 4" strokeOpacity={.7} dot={false} connectNulls={false}/>}
          {medianVisible && <Line isAnimationActive={!reducedMotion} type="linear" dataKey="priceMedian" stroke={theme.forecast} strokeWidth={2} strokeDasharray="6 3" dot={false} connectNulls={false}/>}
          {chart.hasForecast && <ReferenceLine x={chart.lastDate} stroke={theme.text} strokeDasharray="3 3" label={{ value: 'Last close', position: 'insideTopLeft', fill: theme.text, fontSize: 12 }}/ >}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
    <p id={`${id}-help`} className="cm-chart-note">Hover or use the chart’s arrow keys to inspect a date. On touch screens, the data table provides every value. Q10–Q90 is the model’s 80% interval; outcomes can fall outside it.</p>
    <details className="cm-chart-table" onToggle={event => setTableOpen(event.currentTarget.open)}>
      <summary>View chart data</summary>
      {tableOpen && <DataTable caption="Values in USD for the selected historical window and available daily forecast. Hidden chart series remain in this table.">
        <thead><tr>{['Date', 'Observed', 'Forecast median', 'Q10', 'Q90'].map(label => <th key={label} scope="col">{label}</th>)}</tr></thead>
        <tbody>{chart.rows.map((row, index) => <tr key={`${row.date}-${index}`}><th scope="row">{row.date.slice(0, 10)}</th>{[row.price, row.isForecast ? row.priceMedian : null, row.isForecast ? row.priceQ10 : null, row.isForecast ? row.priceQ90 : null].map((value, column) => <td key={column}>{formatPrice(value)}</td>)}</tr>)}</tbody>
      </DataTable>}
    </details>
  </div>;
});
