// @vitest-environment jsdom
import { cloneElement, type ReactElement } from 'react';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';
import { afterEach, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import type { HistoryDataPoint, TFTAnalysisResponse } from '../../types';
import { preparePriceChart } from './chart-data';
import { PriceForecastChart, PriceTooltip } from './PriceForecastChart';
import { ModelReliability } from './ModelReliability';

vi.mock('framer-motion', () => ({ useReducedMotion: () => true }));
vi.mock('recharts', async importOriginal => ({ ...await importOriginal<typeof import('recharts')>(), ResponsiveContainer: ({ children }: { children: ReactElement }) => cloneElement(children, { width: 600, height: 300 } as object) }));
afterEach(cleanup);
const history: HistoryDataPoint[] = Array.from({ length: 100 }, (_, i) => ({ date: new Date(Date.UTC(2026, 0, i + 1)).toISOString().slice(0, 10), price: 4 + i / 1000, sentiment_index: null, sentiment_news_count: null }));
const forecast = { prediction: { reference_price_date: history[99].date, daily_forecasts: [{ forecast_date: '2026-04-13', price_median: 4.2, price_q10: null, price_q90: 4.4 }] } } as TFTAnalysisResponse;

it('changes the history window without moving or recomputing forecast dates/prices', () => {
  const thirty = preparePriceChart(history, forecast, 30);
  const ninety = preparePriceChart(history, forecast, 90);
  expect(thirty.rows).toHaveLength(31);
  expect(ninety.rows).toHaveLength(91);
  expect(thirty.rows[30]).toEqual(ninety.rows[90]);
  expect(thirty.rows[30]).toMatchObject({ date: '2026-04-13', priceMedian: 4.2, priceQ10: null });
  expect(thirty.rows[29]).toMatchObject({ price: 4.099, priceMedian: 4.099 });
});

it('keeps historical prices but never bridges a stale, undated or degraded forecast', () => {
  const variants = [
    { ...forecast, prediction: { ...forecast.prediction!, reference_price_date: '2026-01-01' } },
    { ...forecast, prediction: { ...forecast.prediction!, reference_price_date: null } },
    { ...forecast, quality_state: 'degraded' },
  ];
  for (const variant of variants) {
    const chart = preparePriceChart(history, variant, 30);
    expect(chart.rows).toHaveLength(30);
    expect(chart.hasForecast).toBe(false);
    expect(chart.rows[29].priceMedian).toBeUndefined();
  }
});

it('keeps a nonzero axis span for flat prices and leaves crossed/missing intervals unfilled', () => {
  const flat = preparePriceChart(history.map(row => ({ ...row, price: 4 })), null, 30);
  expect(flat.domain[0]).toBeLessThan(4);
  expect(flat.domain[1]).toBeGreaterThan(4);
  const crossed = { ...forecast, prediction: { ...forecast.prediction!, daily_forecasts: [{ ...forecast.prediction!.daily_forecasts[0], price_q10: 5, price_q90: 4 }] } };
  const row = preparePriceChart(history, crossed, 30).rows[30];
  expect(row.priceRange).toBeUndefined();
  expect(row.priceConvictionRange).toBeUndefined();
  expect(row.priceQ10).toBe(5);
  expect(preparePriceChart([{ ...history[0], price: NaN }], null, 30).rows).toHaveLength(0);
});

it('computes the Q25-Q75 inner conviction envelope and horizon readouts for valid 5D paths', () => {
  const fullForecast = {
    direction: 'BULLISH',
    primary_forecast_return: 0.025,
    primary_forecast_q10: -0.012,
    primary_forecast_q90: 0.064,
    t1_impulse: 'BEARISH',
    t1_return: -0.004,
    prediction: {
      reference_price: 4.099,
      reference_price_date: history[99].date,
      weekly_return: 0.025,
      daily_forecasts: [
        { day: 1, forecast_date: '2026-04-13', daily_return: -0.004, cumulative_return: -0.004, price_median: 4.083, price_q10: 4.01, price_q90: 4.16, price_q02: 3.95, price_q98: 4.22 },
        { day: 5, forecast_date: '2026-04-17', daily_return: 0.008, cumulative_return: 0.025, price_median: 4.2015, price_q10: 4.05, price_q90: 4.36, price_q02: 3.96, price_q98: 4.45 },
      ],
    },
  } as TFTAnalysisResponse;
  const chart = preparePriceChart(history, fullForecast, 30);
  expect(chart.rows).toHaveLength(32);
  expect(chart.rows[31].priceConvictionRange).toBeDefined();
  expect(chart.rows[31].priceConvictionRange![0]).toBeGreaterThan(4.05);
  expect(chart.rows[31].priceConvictionRange![1]).toBeLessThan(4.36);

  render(<PriceForecastChart history={history} forecast={fullForecast}/>);
  expect(screen.getByText(/5D Horizon/)).toBeVisible();
  expect(screen.getByText(/T\+1 Diagnostic/)).toBeVisible();
  expect(screen.getByText(/BEARISH -0\.40%/)).toBeVisible();
});

it('lets readers change periods and series and inspect all available table values on demand', async () => {
  const user = userEvent.setup();
  const { container } = render(<PriceForecastChart history={history} forecast={forecast}/>);
  expect(screen.queryByRole('table')).not.toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'Last 90 daily closes' }));
  expect(screen.getByRole('button', { name: 'Last 90 daily closes' })).toHaveAttribute('aria-pressed', 'true');
  await user.click(screen.getByRole('button', { name: 'Forecast median' }));
  expect(screen.getByRole('button', { name: 'Forecast median' })).toHaveAttribute('aria-pressed', 'false');
  const details = container.querySelector('details')!;
  details.open = true;
  fireEvent(details, new Event('toggle'));
  const table = await screen.findByRole('table');
  expect(within(table).getAllByRole('row')).toHaveLength(92);
  expect(within(table).getByText('$4.2000')).toBeVisible();
  expect(within(table).getByText('2026-04-13').closest('tr')).toHaveTextContent('—');
  details.open = false;
  fireEvent(details, new Event('toggle'));
  expect(screen.queryByRole('table')).not.toBeInTheDocument();
}, 15000);

it('labels historical-only, unavailable and loading failures without inventing forecast values', () => {
  const { rerender } = render(<PriceForecastChart history={history} forecast={null}/>);
  expect(screen.getByRole('button', { name: 'Forecast median' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Q10–Q90 range' })).toBeDisabled();
  rerender(<PriceForecastChart history={[]} forecast={null} historyError/>);
  expect(screen.getByRole('alert')).toHaveTextContent('Price history could not be loaded');
  rerender(<PriceForecastChart history={[]} forecast={null}/>);
  expect(screen.getByRole('status')).toHaveTextContent('No chart data available');
});

it('renders a partially available tooltip without crashing or substituting zero', () => {
  render(<PriceTooltip active payload={[{ payload: { date: '2026-04-13', isForecast: true, priceMedian: 0, priceQ10: null, priceQ90: 4.4 } }]}/>);
  expect(screen.getByText('Median: $0.0000')).toBeVisible();
  expect(screen.getByText('Q10: —')).toBeVisible();
  expect(screen.getByText('Q90: $4.4000')).toBeVisible();
});

it('does not turn missing daily Sharpe into a healthy status alongside weekly accuracy', () => {
  render(<MemoryRouter><ModelReliability metrics={{ weekly_directional_accuracy: .54, weekly_sample_count: 100 }} unavailable={false}/></MemoryRouter>);
  expect(screen.getByText('54.0%')).toBeVisible();
  expect(screen.getByText('Daily Sharpe').closest('.cm-reliability-row')).toHaveTextContent('—');
  expect(screen.queryByText('HEALTHY')).not.toBeInTheDocument();
  expect(screen.getByRole('link', { name: /Review model metrics/ })).toHaveAttribute('href', '/models');
});

it('renders complete reliability telemetry including RMSE/MAE, Sharpe, PI80, and gate state', () => {
  render(
    <MemoryRouter>
      <ModelReliability
        metrics={{
          weekly_directional_accuracy: 0.562,
          weekly_sample_count: 120,
          sharpe_ratio: 0.84,
          rmse: 0.0184,
          mae: 0.0132,
          variance_ratio: 0.92,
        }}
        unavailable={false}
      />
    </MemoryRouter>
  );
  expect(screen.getByText('Quality Gate Passed')).toBeVisible();
  expect(screen.getByText('56.2%')).toBeVisible();
  expect(screen.getByText('0.84')).toBeVisible();
  expect(screen.getByText('0.0184 / 0.0132')).toBeVisible();
  expect(screen.getByText('VR 0.92')).toBeVisible();
});
