// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, describe, expect, it } from 'vitest';
import { CuprumCompanionPanel } from './CuprumCompanionPanel';
import type { CommentaryResponse, ExplainResponse } from '../types';

afterEach(cleanup);

const EXPLAIN_FIXTURE: ExplainResponse = {
  schema_version: 1,
  symbol: 'HG=F',
  reference_price_date: '2026-10-16',
  generated_at: '2026-10-16T21:00:00Z',
  run_id: 'run-42',
  model: { checkpoint: 'best.ckpt', trained_at: '2026-10-15T00:00:00Z' },
  horizon: { label: '5-day cumulative log return (primary)', days: 5 },
  layout_hash: { encoder: 'abc', decoder: 'def', static: 'ghi' },
  variable_selection: {
    method: 'vsn_softmax_mean',
    encoder: [
      {
        feature: 'HG=F_close',
        weight: 0.42,
        label: 'Copper · close price',
        description: 'COMEX daily settlement price',
        category: 'Price Action',
        time_horizon: 'Daily',
        recent: {
          basis: '5 sessions',
          kind: 'pct',
          last_value: 4.56,
          change: 0.09,
          change_pct: 2.1,
          as_of: '2026-10-16',
        },
      },
      {
        feature: 'GXC_ret5',
        weight: 0.25,
        label: 'GXC · return lag 5 days',
        description: 'China ETF 5-day return',
        category: 'Momentum',
        time_horizon: 'Short-term (1-5d)',
      },
    ],
    decoder: [],
    static: [],
  },
  attention: {
    method: 'multihead_mean',
    step: { kind: 'first_forecast_day', horizon_index: 0 },
    window: { start: '2026-08-19', end: '2026-10-16', days: 3 },
    top_days: [
      { date: '2026-09-16', weight: 0.033 },
      { date: '2026-09-28', weight: 0.029 },
    ],
    series: [
      { date: '2026-09-16', weight: 0.033 },
      { date: '2026-09-28', weight: 0.029 },
      { date: '2026-10-16', weight: 0.021 },
    ],
  },
  notes:
    'Model focus only: variable-selection (VSN) weights and attention weights show where the TFT-ASRO model looked inside the latest encoder window. They are raw, pre-calibration signals and are not causal.',
};

const COMMENTARY_FIXTURE: CommentaryResponse = {
  symbol: 'HG=F',
  commentary: [
    'RISKS (3 factors):',
    '• Dollar rebound could cap upside momentum.',
    '• Smelter restocking remains uneven.',
    'OPPORTUNITIES (3 factors):',
    '• Warehouse draws tighten prompt COMEX spreads.',
    '• Chinese grid stimulus supports medium-term demand.',
    'SYNTHESIS:',
    'Copper futures maintain a constructive 5-day posture supported by tight spreads and Chinese equity momentum.',
    'BIAS ACKNOWLEDGMENT:',
    'Subject to macro volatility around upcoming PMI releases.',
    'This is NOT financial advice.',
  ].join('\n'),
  error: null,
  generated_at: '2026-10-16T21:05:00Z',
  ai_stance: 'BULLISH',
  generation_mode: 'llm',
  model_name: 'nvidia/nemotron-3-super-120b-a12b:free',
};

describe('CuprumCompanionPanel', () => {
  it('renders Cuprum in Molten Resonance mode with executive synthesis and XAI summary pills', () => {
    render(
      <CuprumCompanionPanel
        commentary={COMMENTARY_FIXTURE}
        explain={EXPLAIN_FIXTURE}
      />,
    );

    expect(screen.getByTestId('cuprum-companion-stage')).toBeInTheDocument();
    expect(screen.getByText('Molten Resonance')).toBeInTheDocument();
    expect(screen.getByText('BULLISH')).toBeInTheDocument();
    expect(
      screen.getByText(/Copper futures maintain a constructive 5-day posture/i),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/Copper · close price/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/2026-09-16/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/not causal/i)).toBeInTheDocument();
  });

  it('switches between Synthesis, Risks, Opportunities, and XAI Focus & Memory tabs', () => {
    render(
      <CuprumCompanionPanel
        commentary={COMMENTARY_FIXTURE}
        explain={EXPLAIN_FIXTURE}
      />,
    );

    fireEvent.click(screen.getByRole('tab', { name: /Risks/i }));
    expect(screen.getByText(/Dollar rebound could cap upside momentum/i)).toBeInTheDocument();
    expect(screen.getByText(/Subject to macro volatility/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: /Opportunities/i }));
    expect(
      screen.getByText(/Warehouse draws tighten prompt COMEX spreads/i),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: /XAI Focus & Memory/i }));
    expect(screen.getByTestId('cuprum-xai-tab')).toBeInTheDocument();
    expect(screen.getByTestId('model-focus-attention')).toBeInTheDocument();
    expect(screen.getByText('+2.10% over last 5 sessions')).toBeInTheDocument();
    expect(screen.getByText('2026-09-28')).toBeInTheDocument();
  });

  it('renders empty state gracefully when both commentary and explain are null', () => {
    render(<CuprumCompanionPanel commentary={null} explain={null} />);
    expect(
      screen.getByText('Model focus will appear after the next model refresh.'),
    ).toBeInTheDocument();
  });

  it('toggles tactical console open and closed when clicking the pet mascot', () => {
    render(
      <CuprumCompanionPanel
        commentary={COMMENTARY_FIXTURE}
        explain={EXPLAIN_FIXTURE}
      />,
    );

    const petButton = screen.getByRole('button', { name: /Cuprum companion/i });
    expect(screen.getByText('Close console')).toBeInTheDocument();

    // Click to close
    fireEvent.click(petButton);
    expect(screen.getByText('Open console')).toBeInTheDocument();
    expect(screen.getByText(/CUPRUM DIALOGUE \/\/ MINIMIZED/i)).toBeInTheDocument();

    // Click to reopen
    fireEvent.click(petButton);
    expect(screen.getByText('Close console')).toBeInTheDocument();
    expect(screen.getByText(/CUPRUM EXECUTIVE THESIS/i)).toBeInTheDocument();
  });
});
