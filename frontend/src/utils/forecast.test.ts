import { describe, expect, it } from 'vitest';
import { isForecastAligned, mapTftForecastRows } from './forecast';
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

  it('keeps only the next five dates after the forecast reference bar', () => {
    const rows = [
      row({ forecast_date: '2026-08-31' }),
      row({ day: 2, forecast_date: '2026-09-01' }),
      row({ day: 3, forecast_date: '2026-09-02' }),
    ];

    expect(mapTftForecastRows(rows, '2026-08-31')).toHaveLength(2);
    expect(mapTftForecastRows(rows, '2026-08-31')[0].date).toBe('2026-09-01');
  });
});

describe('isForecastAligned', () => {
  it('aligns timestamps that represent the same market date', () => {
    expect(isForecastAligned('2026-09-01', '2026-09-01T04:00:00Z')).toBe(true);
  });

  it('rejects a previous forecast vintage after a new market bar arrives', () => {
    expect(isForecastAligned('2026-09-01', '2026-09-02')).toBe(false);
  });
});
