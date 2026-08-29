import type { TFTDailyForecast } from '../types';

export interface ForecastChartRow {
  date: string;
  priceMedian: number | null;
  priceQ10: number | null;
  priceQ90: number | null;
  isForecast: true;
}

/** Map the backend contract without recalculating dates, prices, or returns. */
export function mapTftForecastRows(rows: TFTDailyForecast[]): ForecastChartRow[] {
  return rows.slice(0, 5).flatMap((forecast) => {
    if (!forecast.forecast_date) return [];
    return [{
      date: forecast.forecast_date,
      priceMedian: forecast.price_median,
      priceQ10: forecast.price_q10,
      priceQ90: forecast.price_q90,
      isForecast: true as const,
    }];
  });
}
