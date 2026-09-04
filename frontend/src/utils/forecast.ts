import type { TFTDailyForecast } from '../types';

export interface ForecastChartRow {
  date: string;
  priceMedian: number | null;
  priceQ10: number | null;
  priceQ90: number | null;
  isForecast: true;
}

function dateKey(value: string | null | undefined): string | null {
  if (!value) return null;
  const match = value.match(/^\d{4}-\d{2}-\d{2}/);
  return match?.[0] ?? null;
}

/**
 * A forecast path may only be attached to the history bar it was generated
 * from. The API regenerates a rolling five-day path when a newer bar arrives;
 * this guard prevents a transient polling race from shifting an older path to
 * a newer close on the chart.
 */
export function isForecastAligned(
  referenceDate: string | null | undefined,
  latestHistoryDate: string | null | undefined,
): boolean {
  const reference = dateKey(referenceDate);
  const latest = dateKey(latestHistoryDate);
  return reference !== null && reference === latest;
}

/** Map the backend contract without recalculating dates, prices, or returns. */
export function mapTftForecastRows(
  rows: TFTDailyForecast[],
  referenceDate?: string | null,
): ForecastChartRow[] {
  const reference = dateKey(referenceDate);
  return rows.flatMap((forecast) => {
    if (!forecast.forecast_date) return [];
    const forecastDate = dateKey(forecast.forecast_date);
    if (!forecastDate || (reference && forecastDate <= reference)) return [];
    return [{
      date: forecast.forecast_date,
      priceMedian: forecast.price_median,
      priceQ10: forecast.price_q10,
      priceQ90: forecast.price_q90,
      isForecast: true as const,
    }];
  }).slice(0, 5);
}
