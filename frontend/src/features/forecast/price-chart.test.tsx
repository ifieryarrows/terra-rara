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
  expect(row.priceQ10).toBe(5);
  expect(preparePriceChart([{ ...history[0], price: NaN }], null, 30).rows).toHaveLength(0);
});

it('lets readers change periods and series and inspect all available table values on demand', async () => {
  const user = userEvent.setup();
  const { container } = render(<PriceForecastChart history={history} forecast={forecast}/>);
  expect(screen.queryByRole('table')).not.toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: '90 closes' }));
  expect(screen.getByRole('button', { name: '90 closes' })).toHaveAttribute('aria-pressed', 'true');
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
});

it('labels historical-only, unavailable and loading failures without inventing forecast values', () => {
  const { rerender } = render(<PriceForecastChart history={history} forecast={null}/>);
  expect(screen.getByRole('button', { name: 'Forecast median' })).toBeDisabled();
  expect(screen.getByText(/Showing historical closes only/)).toBeVisible();
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
