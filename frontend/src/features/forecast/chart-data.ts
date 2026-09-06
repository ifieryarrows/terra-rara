import type { HistoryDataPoint, TFTAnalysisResponse } from '../../types';
import { isForecastAligned, mapTftForecastRows } from '../../utils/forecast';

export interface PriceChartRow {
  date: string;
  price?: number | null;
  priceMedian?: number | null;
  priceQ10?: number | null;
  priceQ90?: number | null;
  priceRange?: [number, number];
  isForecast?: boolean;
}

export const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
export const formatPrice = (value: unknown) => finite(value) ? `$${value.toFixed(4)}` : '—';
export function formatChartDate(value: string, compact = false) {
  const parsed = new Date(value.slice(0, 10) + 'T00:00:00Z');
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleDateString(undefined, { timeZone: 'UTC', month: 'short', day: 'numeric', ...(compact ? {} : { year: 'numeric' }) });
}

/** Display-only windowing. Forecast dates/prices and the exact baseline guard stay authoritative. */
export function preparePriceChart(history: HistoryDataPoint[], forecast: TFTAnalysisResponse | null, count: number) {
  const valid = history.filter(point => finite(point.price));
  const recent = valid.slice(-count);
  const last = recent[recent.length - 1];
  const prediction = forecast?.prediction;
  const degraded = forecast?.quality_state === 'degraded' || forecast?.model_state === 'retrain_required' || forecast?.is_forecast_healthy === false;
  const aligned = !!last && isForecastAligned(prediction?.reference_price_date, last.date);
  const forecastRows = !degraded && aligned ? mapTftForecastRows(prediction?.daily_forecasts ?? [], prediction?.reference_price_date) : [];
  const hasForecast = forecastRows.some(row => [row.priceMedian, row.priceQ10, row.priceQ90].some(finite));
  const rows: PriceChartRow[] = recent.map(point => ({ date: point.date, price: point.price }));
  if (hasForecast && last) {
    Object.assign(rows[rows.length - 1], { priceMedian: last.price, priceQ10: last.price, priceQ90: last.price });
    rows.push(...forecastRows);
  }
  const values: number[] = [];
  for (const row of rows) {
    // Invalid observations create gaps; never invent zeroes or repair crossed quantiles.
    for (const key of ['price', 'priceMedian', 'priceQ10', 'priceQ90'] as const) {
      if (finite(row[key])) values.push(row[key]);
      else if (key in row) row[key] = null;
    }
    if (finite(row.priceQ10) && finite(row.priceQ90) && row.priceQ10 <= row.priceQ90) row.priceRange = [row.priceQ10, row.priceQ90];
  }
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const padding = Math.max((max - min) * .05, Math.abs(max) * .001, .001);
  return { rows, domain: [min - padding, max + padding] as [number, number], lastDate: last?.date, historyCount: recent.length, hasForecast, aligned, degraded };
}
