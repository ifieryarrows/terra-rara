// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, describe, expect, it } from 'vitest';
import type { ExplainAttentionPoint, ExplainResponse } from '../types';
import { ModelFocusPanel } from './ModelFocusPanel';

const attentionSeries: ExplainAttentionPoint[] = Array.from({ length: 60 }, (_, index) => ({
  date: new Date(Date.UTC(2026, 6, 1) + index * 86_400_000).toISOString().slice(0, 10),
  weight: 1 / 60,
}));

const payload: ExplainResponse = {
  schema_version: 1,
  symbol: 'HG=F',
  reference_price_date: '2026-10-14',
  generated_at: '2026-10-15T06:00:00+00:00',
  run_id: 'run-2026-10-15',
  model: { checkpoint: 'tft-asro.ckpt', trained_at: '2026-10-01T04:00:00+00:00' },
  horizon: { label: '5-day cumulative log return (primary)', days: 5 },
  layout_hash: { encoder: 'a1b2c3d4e5f6', decoder: 'a1b2c3d4e5f7', static: 'a1b2c3d4e5f8' },
  variable_selection: {
    method: 'vsn',
    encoder: [
      {
        feature: 'HG_F_close', weight: 0.42, label: 'Copper close price', description: '', category: 'Trend', time_horizon: 'short',
        recent: { basis: '5 sessions', kind: 'pct', last_value: 4.62, change: 0.095, change_pct: 2.1, as_of: '2026-10-14' },
      },
      {
        feature: 'HG_F_rsi14', weight: 0.21, label: 'RSI (14)', description: '', category: 'Momentum', time_horizon: 'short',
        recent: { basis: '5 sessions', kind: 'abs', last_value: 61.2, change: 0.12, change_pct: null, as_of: '2026-10-14' },
      },
      { feature: 'HG_F_sma20', weight: 0.14, label: 'SMA (20)', description: '', category: 'Trend', time_horizon: 'short' },
      { feature: 'news_sentiment_5d', weight: 0.09, label: 'News sentiment (5d)', description: '', category: 'Sentiment', time_horizon: 'short' },
      { feature: 'HG_F_atr14', weight: 0.06, label: 'ATR (14)', description: '', category: 'Volatility', time_horizon: 'short' },
      { feature: 'sector_energy_ret', weight: 0.02, label: 'Energy sector return', description: '', category: 'Sector', time_horizon: 'long' },
    ],
    decoder: [],
    static: [],
  },
  attention: {
    method: 'tft_attention',
    step: { kind: 'first_forecast_day', horizon_index: 0 },
    window: { start: attentionSeries[0].date, end: attentionSeries[59].date, days: 60 },
    top_days: [attentionSeries[10], attentionSeries[20]],
    series: attentionSeries,
  },
  notes:
    'Model focus only: variable-selection (VSN) weights and attention weights show where the TFT-ASRO model looked inside the latest encoder window. They are raw, pre-calibration signals and are not causal: do not read them as market drivers or investment advice.',
};

afterEach(cleanup);

describe('ModelFocusPanel', () => {
  it('renders encoder drivers, attention bars, reference meta, notes and raw JSON from the explain payload', () => {
    render(<ModelFocusPanel data={payload}/>);

    expect(screen.getByText('Copper close price')).toBeVisible();
    expect(screen.getByText('Momentum signal')).toBeVisible();
    expect(screen.getByText('News sentiment')).toBeVisible();
    expect(screen.getByText('+2.10% over last 5 sessions')).toBeVisible();
    expect(screen.getByText('+0.120 over last 5 sessions')).toBeVisible();
    expect(screen.queryByText('Energy sector return')).not.toBeInTheDocument();
    expect(screen.getByText('Attention · forecast day 1 (t+1)')).toBeVisible();

    const footnote = screen.getByText(payload.notes);
    expect(footnote).toBeVisible();
    expect(footnote).toHaveTextContent('not causal');
    expect(screen.getByText('Reference 2026-10-14')).toBeVisible();

    const bars = screen.getAllByTestId('model-focus-attention');
    expect(bars).toHaveLength(60);
    expect(bars[0]).toHaveAttribute('title', `${attentionSeries[0].date}: 1.7%`);

    const details = screen.getByText('Raw JSON').closest('details');
    expect(details?.querySelector('pre')?.textContent).toBe(JSON.stringify(payload, null, 2));
  });

  it('shows the empty state when the explanation is not available yet', () => {
    render(<ModelFocusPanel data={null}/>);

    expect(screen.getByText('Model focus will appear after the next model refresh.')).toBeVisible();
    expect(screen.queryByTestId('model-focus-attention')).not.toBeInTheDocument();
  });

  it('degrades to the empty state for an empty payload object without throwing', () => {
    render(<ModelFocusPanel data={{} as unknown as ExplainResponse}/>);

    expect(screen.getByText('Model focus will appear after the next model refresh.')).toBeVisible();
  });

  it('omits missing sections instead of throwing on a partial payload', () => {
    const partial = {
      reference_price_date: null,
      variable_selection: { encoder: [payload.variable_selection.encoder[2]] },
    } as unknown as ExplainResponse;

    render(<ModelFocusPanel data={partial}/>);

    expect(screen.getByText('SMA (20)')).toBeVisible();
    expect(screen.queryByText(/^Reference /)).not.toBeInTheDocument();
    expect(screen.queryByText(/forecast day 1/)).not.toBeInTheDocument();
    expect(screen.queryByTestId('model-focus-attention')).not.toBeInTheDocument();
  });
});
