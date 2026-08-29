import { describe, expect, it } from 'vitest';
import { mapTftForecastRows } from './forecast';
import type { TFTDailyForecast } from '../types';

function row(overrides: Partial<TFTDailyForecast> = {}): TFTDailyForecast {
  return {
    day: 1,
    forecast_date: '2026-08-31',
    daily_return: 0.01,
    cumulative_return: 0.01,
    price_median: 6.65487,
    price_q10: 6.5,
    price_q90: 6.8,
    price_q02: 6.4,
    price_q98: 6.9,
    ...overrides,
  };
}

describe('mapTftForecastRows', () => {
  it('preserves backend forecast date and price values exactly', () => {
    expect(mapTftForecastRows([row()])).toEqual([{
      date: '2026-08-31',
      priceMedian: 6.65487,
      priceQ10: 6.5,
      priceQ90: 6.8,
      isForecast: true,
    }]);
  });

  it('does not invent a date when the backend date is unavailable', () => {
    expect(mapTftForecastRows([row({ forecast_date: null })])).toEqual([]);
  });
});
