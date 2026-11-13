import type { HistoryDataPoint, TFTAnalysisResponse, TFTDailyForecast } from '../../types';
import { isForecastAligned, mapTftForecastRows } from '../../utils/forecast';

type ExtendedForecastSource = TFTDailyForecast & {
  price_q25?: number | null;
  price_q75?: number | null;
};

export interface PriceChartRow {
  date: string;
  price?: number | null;
  priceMedian?: number | null;
  priceQ10?: number | null;
  priceQ90?: number | null;
  priceQ25?: number | null;
  priceQ75?: number | null;
  priceQ02?: number | null;
  priceQ98?: number | null;
  priceRange?: [number, number];
  priceConvictionRange?: [number, number];
  isForecast?: boolean;
  day?: number;
  dailyReturn?: number | null;
  cumulativeReturn?: number | null;
  returnQ10?: number | null;
  returnQ90?: number | null;
}

export const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
export const formatPrice = (value: unknown) => finite(value) ? `$${value.toFixed(4)}` : '—';

export function formatReturnPct(value: unknown, signed = true): string {
  if (!finite(value)) return '—';
  const pct = value * 100;
  if (Math.abs(pct) < 0.005) return '0.00%';
  return `${signed && pct > 0 ? '+' : ''}${pct.toFixed(2)}%`;
}

export function formatChartDate(value: string, compact = false) {
  const parsed = new Date(value.slice(0, 10) + 'T00:00:00Z');
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleDateString('en-US', { timeZone: 'UTC', month: 'short', day: 'numeric', ...(compact ? {} : { year: 'numeric' }) });
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
  const refPrice = finite(prediction?.reference_price) && prediction.reference_price > 0
    ? prediction.reference_price
    : (last?.price ?? null);

  const sourceByDate = new Map<string, ExtendedForecastSource>();
  for (const item of (prediction?.daily_forecasts ?? []) as ExtendedForecastSource[]) {
    if (item.forecast_date) sourceByDate.set(item.forecast_date, item);
  }

  const rows: PriceChartRow[] = recent.map(point => ({ date: point.date, price: point.price }));
  if (hasForecast && last) {
    Object.assign(rows[rows.length - 1], {
      priceMedian: last.price,
      priceQ10: last.price,
      priceQ90: last.price,
      priceQ25: last.price,
      priceQ75: last.price,
      priceConvictionRange: [last.price, last.price] as [number, number],
    });
    rows.push(...forecastRows.map((row, index) => {
      const src = sourceByDate.get(row.date);
      const day = finite(src?.day) ? src.day : index + 1;
      const dailyReturn = finite(src?.daily_return) ? src.daily_return : null;
      const cumulativeReturn = finite(src?.cumulative_return)
        ? src.cumulative_return
        : (finite(row.priceMedian) && finite(refPrice) && refPrice > 0 ? row.priceMedian / refPrice - 1 : null);
      const returnQ10 = finite(row.priceQ10) && finite(refPrice) && refPrice > 0 ? row.priceQ10 / refPrice - 1 : null;
      const returnQ90 = finite(row.priceQ90) && finite(refPrice) && refPrice > 0 ? row.priceQ90 / refPrice - 1 : null;
      const priceQ02 = finite(src?.price_q02) ? src.price_q02 : null;
      const priceQ98 = finite(src?.price_q98) ? src.price_q98 : null;

      let priceQ25: number | null | undefined;
      let priceQ75: number | null | undefined;
      let priceConvictionRange: [number, number] | undefined;

      if (finite(src?.price_q25) && finite(src?.price_q75) && src.price_q25 <= src.price_q75) {
        priceQ25 = src.price_q25;
        priceQ75 = src.price_q75;
        priceConvictionRange = [priceQ25, priceQ75];
      } else if (
        finite(row.priceQ10)
        && finite(row.priceMedian)
        && finite(row.priceQ90)
        && row.priceQ10 <= row.priceMedian
        && row.priceMedian <= row.priceQ90
      ) {
        // 50% IQR conviction core derived symmetrically inside valid non-crossed Q10–Q50–Q90 bounds.
        priceQ25 = row.priceMedian - 0.526 * (row.priceMedian - row.priceQ10);
        priceQ75 = row.priceMedian + 0.526 * (row.priceQ90 - row.priceMedian);
        priceConvictionRange = [priceQ25, priceQ75];
      }

      return {
        ...row,
        day,
        dailyReturn,
        cumulativeReturn,
        returnQ10,
        returnQ90,
        priceQ02,
        priceQ98,
        ...(priceQ25 !== undefined ? { priceQ25, priceQ75, priceConvictionRange } : {}),
      };
    }));
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
  const padding = Math.max((max - min) * .06, Math.abs(max) * .001, .001);

  const horizonRow = hasForecast ? rows[rows.length - 1] ?? null : null;
  const t1Impulse = !degraded ? (forecast?.t1_impulse ?? forecast?.weekly_forecast?.t1_impulse ?? null) : null;
  const rawT1Return = !degraded ? (forecast?.t1_return ?? forecast?.weekly_forecast?.t1_return ?? prediction?.predicted_return_median) : null;
  const t1Return = finite(rawT1Return) ? rawT1Return : null;

  const rawWeeklyReturn = !degraded ? (forecast?.primary_forecast_return ?? forecast?.weekly_forecast?.expected_return ?? prediction?.weekly_return) : null;
  const rawWeeklyQ10 = !degraded ? (forecast?.primary_forecast_q10 ?? forecast?.weekly_forecast?.q10_return ?? prediction?.weekly_return_q10_calibrated) : null;
  const rawWeeklyQ90 = !degraded ? (forecast?.primary_forecast_q90 ?? forecast?.weekly_forecast?.q90_return ?? prediction?.weekly_return_q90_calibrated) : null;

  return {
    rows,
    domain: [min - padding, max + padding] as [number, number],
    lastDate: last?.date,
    lastPrice: last?.price ?? null,
    refPrice,
    historyCount: recent.length,
    hasForecast,
    aligned,
    degraded,
    horizonRow,
    t1Impulse,
    t1Return,
    weeklyReturn: finite(rawWeeklyReturn) ? rawWeeklyReturn : null,
    weeklyQ10Return: finite(rawWeeklyQ10) ? rawWeeklyQ10 : null,
    weeklyQ90Return: finite(rawWeeklyQ90) ? rawWeeklyQ90 : null,
    direction: !degraded ? (forecast?.direction ?? null) : null,
  };
}
