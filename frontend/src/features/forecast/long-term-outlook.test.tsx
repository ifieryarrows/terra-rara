// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { LongTermOutlookResponse, TFTMetrics } from '../../types';
import { LongTermOutlookPanel } from './LongTermOutlookPanel';
import { computeQualityGateThresholds, ModelReliability } from './ModelReliability';

afterEach(cleanup);

const OUTLOOK_FIXTURE: LongTermOutlookResponse = {
  status: 'ok',
  symbol: 'HG=F',
  as_of_date: '2026-11-20',
  generated_at: '2026-11-21T18:00:00Z',
  model_version: 'lt_structural_v1',
  source: 'snapshot',
  current_price: 4.92,
  available_horizons: ['20D', '60D'],
  default_horizon: '20D',
  data_window: {
    start_date: '2011-11-25',
    end_date: '2026-11-20',
    weekly_bars: 780,
    years_covered: 15.0,
    fred_macro_enabled: true,
    symbols_used: ['HG=F', 'DX-Y.NYB', 'GXC', 'COPX', 'FCX'],
  },
  horizons: {
    '20D': {
      horizon_key: '20D',
      horizon_label: '1 Month (20D)',
      horizon_days: 20,
      horizon_weeks: 4,
      target_date: '2026-12-18',
      reference_price: 4.92,
      target_price_q50: 5.04,
      expected_return: 0.0244,
      expected_return_pct: 2.44,
      expected_log_return: 0.0241,
      price_q10: 4.61,
      price_q25: 4.82,
      price_q75: 5.26,
      price_q90: 5.48,
      return_q10: -0.063,
      return_q25: -0.02,
      return_q75: 0.069,
      return_q90: 0.114,
      stance: 'BULLISH',
      prob_bullish: 0.54,
      prob_neutral: 0.28,
      prob_bearish: 0.18,
      conviction: 0.62,
      conviction_label: 'HIGH',
      regime_threshold_pct: 2.6,
      components: {
        macro_ridge_return: 0.028,
        theta_damped_return: 0.017,
        macro_ridge_weight: 0.65,
        theta_weight: 0.35,
        realized_vol_annualized: 0.21,
        horizon_vol: 0.058,
        cqr_q10_width: 0.089,
        cqr_q90_width: 0.084,
      },
      key_drivers: [
        {
          feature: 'dxy_ret_12w',
          label: 'USD 12W Retreat',
          pillar: 'usd_macro',
          z_score: -1.1,
          contribution_return: 0.011,
          direction: 'BULLISH',
        },
      ],
    },
    '60D': {
      horizon_key: '60D',
      horizon_label: '1 Quarter (60D)',
      horizon_days: 60,
      horizon_weeks: 12,
      target_date: '2027-02-12',
      reference_price: 4.92,
      target_price_q50: 4.78,
      expected_return: -0.0285,
      expected_return_pct: -2.85,
      expected_log_return: -0.0289,
      price_q10: 4.18,
      price_q25: 4.46,
      price_q75: 5.12,
      price_q90: 5.42,
      return_q10: -0.1504,
      return_q25: -0.0935,
      return_q75: 0.0407,
      return_q90: 0.1016,
      stance: 'BEARISH',
      prob_bullish: 0.21,
      prob_neutral: 0.31,
      prob_bearish: 0.48,
      conviction: 0.44,
      conviction_label: 'MEDIUM',
      regime_threshold_pct: 4.5,
      components: {
        macro_ridge_return: -0.036,
        theta_damped_return: -0.015,
        macro_ridge_weight: 0.65,
        theta_weight: 0.35,
        realized_vol_annualized: 0.21,
        horizon_vol: 0.101,
        cqr_q10_width: 0.125,
        cqr_q90_width: 0.132,
      },
      key_drivers: [
        {
          feature: 'gxc_ret_26w',
          label: 'China Industrial Drag',
          pillar: 'china_industrial_demand',
          z_score: -0.9,
          contribution_return: -0.014,
          direction: 'BEARISH',
        },
      ],
    },
  },
  macro_scorecard: {
    composite_score: 0.34,
    composite_stance: 'BULLISH',
    pillars: {
      trend_structure: {
        key: 'trend_structure',
        label: 'Trend Structure',
        score: 0.42,
        score_pct: 71,
        stance: 'BULLISH',
        summary: 'Spot holds cleanly above 40W structural baseline.',
        metrics: [{ name: 'Price vs 40W SMA', value: 0.064, formatted_value: '+6.4%', signal: 'BULLISH' }],
      },
      usd_macro: {
        key: 'usd_macro',
        label: 'USD & Macro',
        score: 0.18,
        score_pct: 59,
        stance: 'NEUTRAL',
        summary: 'Dollar impulse muted alongside stable real yields.',
        metrics: [{ name: 'DXY 12W Trend', value: -0.012, formatted_value: '-1.2%', signal: 'BULLISH' }],
      },
      china_industrial_demand: {
        key: 'china_industrial_demand',
        label: 'China & Industrial Demand',
        score: 0.31,
        score_pct: 66,
        stance: 'BULLISH',
        summary: 'Broad China large-cap and metals basket constructive.',
        metrics: [{ name: 'GXC 12W Momentum', value: 0.048, formatted_value: '+4.8%', signal: 'BULLISH' }],
      },
      cross_asset_confirmation: {
        key: 'cross_asset_confirmation',
        label: 'Cross-Asset Confirmation',
        score: 0.45,
        score_pct: 73,
        stance: 'BULLISH',
        summary: 'Copper miners lead front-month futures higher.',
        metrics: [{ name: 'COPX/HG=F Spread', value: 0.032, formatted_value: '+3.2%', signal: 'BULLISH' }],
      },
    },
  },
  validation: {
    '20D': {
      horizon_key: '20D',
      horizon_weeks: 4,
      purge_gap_weeks: 4,
      annualisation: 12.0,
      n_folds: 5,
      n_purged_oos_samples: 300,
      n_non_overlapping_samples: 75,
      directional_accuracy: 0.56,
      non_overlapping_da: 0.57,
      mae_vs_naive_zero: 0.98,
      mae_vs_theta: 0.96,
      theta_directional_accuracy: 0.52,
      sharpe_ratio: 0.68,
      non_overlapping_sharpe: 0.71,
      pi80_coverage: 0.81,
      pi50_coverage: 0.51,
      pinball_loss_mean: 0.015,
      pinball_skill_vs_empirical: 0.04,
      brier_score: 0.59,
      regime_accuracy: 0.48,
      quantile_crossing_rate: 0.0,
      folds: [],
    },
    '60D': {
      horizon_key: '60D',
      horizon_weeks: 12,
      purge_gap_weeks: 12,
      annualisation: 4.0,
      n_folds: 4,
      n_purged_oos_samples: 240,
      n_non_overlapping_samples: 20,
      directional_accuracy: 0.59,
      non_overlapping_da: 0.6,
      mae_vs_naive_zero: 0.94,
      mae_vs_theta: 0.92,
      theta_directional_accuracy: 0.51,
      sharpe_ratio: 0.74,
      non_overlapping_sharpe: 0.77,
      pi80_coverage: 0.79,
      pi50_coverage: 0.49,
      pinball_loss_mean: 0.022,
      pinball_skill_vs_empirical: 0.06,
      brier_score: 0.56,
      regime_accuracy: 0.52,
      quantile_crossing_rate: 0.0,
      folds: [],
    },
  },
  message: null,
};

describe('LongTermOutlookPanel', () => {
  it('renders 20D default target, Q10-Q90 range, regime probabilities, 4-pillar scorecard, and purged WF badge, then switches to 60D', () => {
    render(<LongTermOutlookPanel data={OUTLOOK_FIXTURE} />);

    expect(screen.getByTestId('lt-target-price')).toHaveTextContent('$5.04');
    expect(screen.getByTestId('lt-expected-move')).toHaveTextContent('+2.44%');
    expect(screen.getByTestId('lt-stance-readout')).toHaveTextContent('BULLISH');
    expect(screen.getByTestId('lt-conviction-readout')).toHaveTextContent('HIGH conviction · 62%');
    expect(screen.getByText('$4.61')).toBeVisible();
    expect(screen.getByText('$5.48')).toBeVisible();
    expect(screen.getByText(/Core IQR: \$4\.82 – \$5\.26/)).toBeVisible();

    const valBadge = screen.getByTestId('long-term-validation-badge');
    expect(valBadge).toHaveTextContent('Purged WF · 5F · gap 4w · √12');
    expect(valBadge).toHaveTextContent('56.0%');
    expect(valBadge).toHaveTextContent('0.980');
    expect(valBadge).toHaveTextContent('0.68');
    expect(valBadge).toHaveTextContent('81%');

    // All 4 macro scorecard pillars
    const scorecard = screen.getByTestId('long-term-macro-scorecard');
    expect(within(scorecard).getByTestId('macro-pillar-trend_structure')).toHaveTextContent('Trend Structure');
    expect(within(scorecard).getByTestId('macro-pillar-usd_macro')).toHaveTextContent('USD & Macro');
    expect(within(scorecard).getByTestId('macro-pillar-china_industrial_demand')).toHaveTextContent('China & Industrial Demand');
    expect(within(scorecard).getByTestId('macro-pillar-cross_asset_confirmation')).toHaveTextContent('Cross-Asset Confirmation');

    // Switch to 1 Quarter (60D)
    const tablist = screen.getByRole('tablist', { name: 'Long-term forecast horizon' });
    fireEvent.click(within(tablist).getByRole('tab', { name: /1 Quarter \(60D\)/i }));

    expect(screen.getByTestId('lt-target-price')).toHaveTextContent('$4.78');
    expect(screen.getByTestId('lt-expected-move')).toHaveTextContent('-2.85%');
    expect(screen.getByTestId('lt-stance-readout')).toHaveTextContent('BEARISH');
    expect(screen.getByTestId('lt-conviction-readout')).toHaveTextContent('MEDIUM conviction · 44%');
    expect(screen.getByTestId('long-term-validation-badge')).toHaveTextContent('Purged WF · 4F · gap 12w · √4');
  });

  it('renders graceful unavailable, loading, and error states with retry support', () => {
    const onRefresh = vi.fn();
    const { rerender } = render(
      <LongTermOutlookPanel
        data={{ ...OUTLOOK_FIXTURE, status: 'unavailable', source: 'unavailable', horizons: {}, message: 'Awaiting worker run.' }}
        onRefresh={onRefresh}
      />,
    );

    expect(screen.getByRole('status')).toHaveTextContent('Long-term outlook snapshot pending');
    expect(screen.getByText('Awaiting worker run.')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Check snapshot' }));
    expect(onRefresh).toHaveBeenCalledOnce();

    rerender(<LongTermOutlookPanel data={null} isLoading />);
    expect(screen.getByRole('status')).toHaveAttribute('aria-busy', 'true');

    rerender(<LongTermOutlookPanel data={null} isError onRefresh={onRefresh} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Long-term outlook could not be loaded');
  });

  it('suppresses signed zero on flat expected move', () => {
    const zeroMove: LongTermOutlookResponse = {
      ...OUTLOOK_FIXTURE,
      horizons: {
        ...OUTLOOK_FIXTURE.horizons,
        '20D': {
          ...OUTLOOK_FIXTURE.horizons['20D'],
          expected_return: 0,
          expected_return_pct: 0.001,
          stance: 'NEUTRAL',
        },
      },
    };
    render(<LongTermOutlookPanel data={zeroMove} />);
    expect(screen.getByTestId('lt-expected-move')).toHaveTextContent(/^0\.00%$/);
  });
});

describe('ModelReliability quality_gate.py alignment', () => {
  it('passes a genuinely valid 5D model that failed obsolete legacy thresholds', () => {
    const passingMetrics: TFTMetrics = {
      weekly_sample_count: 186, // min_weekly_da = 0.47, n_eff = 37.2 -> PI80 [0.6915, 0.9085]
      weekly_directional_accuracy: 0.5,
      weekly_sharpe_ratio_hac: -0.32, // valid >= -0.35 (failed old -0.20)
      weekly_magnitude_ratio: 0.08, // valid [0.02, 2.50] (failed old [0.65, 1.35])
      weekly_raw_magnitude_ratio: 1.4,
      weekly_mae_vs_naive_zero: 0.99,
      weekly_pred_positive_rate: 0.58,
      weekly_actual_positive_rate: 0.55,
      weekly_pi80_coverage: 0.71, // valid inside [0.69, 0.91] (failed old [0.74, 0.86])
      weekly_tail_capture_rate: 0.2, // non-blocking diagnostic (failed old >= 0.45)
      sharpe_ratio: -0.8, // T+1 non-blocking diagnostic (failed old >= -0.30)
    };

    const thresholds = computeQualityGateThresholds(186);
    expect(thresholds.minWeeklyDa).toBe(0.47);
    expect(thresholds.pi80Low).toBeCloseTo(0.6918, 4);
    expect(thresholds.pi80High).toBeCloseTo(0.9082, 4);

    render(
      <MemoryRouter>
        <ModelReliability metrics={passingMetrics} unavailable={false} />
      </MemoryRouter>,
    );

    expect(screen.getByText('Quality Gate Passed')).toBeVisible();
    expect(screen.queryByText('Gate Breach')).not.toBeInTheDocument();
    expect(screen.getByText('Weekly Sharpe (HAC)')).toBeVisible();
    expect(screen.getByText('-0.32')).toBeVisible();
    expect(screen.getByText('Weekly MAE vs Naive')).toBeVisible();
    expect(screen.getByText('0.990')).toBeVisible();
    expect(screen.getByText('Weekly Sign Balance')).toBeVisible();
    expect(screen.getByText('58% / 55%')).toBeVisible();
    expect(screen.getByText(/gate 69–91%/)).toBeVisible();
    expect(screen.getByText(/gate ≥ 47%/)).toBeVisible();
  });

  it('honors authoritative backend qualityGate.passed=true and flags real gate breaches', () => {
    const { rerender } = render(
      <MemoryRouter>
        <ModelReliability
          metrics={{ weekly_directional_accuracy: 0.52, weekly_sample_count: 100 }}
          qualityGate={{ passed: true, reasons: [], warnings: ['T1Sharpe=-0.50'], metrics: {} }}
          unavailable={false}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('Quality Gate Passed')).toBeVisible();
    expect(screen.getByText(/1 diagnostic warning/)).toBeVisible();

    rerender(
      <MemoryRouter>
        <ModelReliability
          metrics={{
            weekly_directional_accuracy: 0.55,
            weekly_sample_count: 200,
            weekly_mae_vs_naive_zero: 1.21,
            weekly_pred_positive_rate: 0.97,
            weekly_actual_positive_rate: 0.5,
          }}
          qualityGate={{ passed: false, reasons: ['WeeklyMAEvsNaiveZero=1.2100 > 1.12'], warnings: [], metrics: {} }}
          unavailable={false}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('Gate Breach')).toBeVisible();
    expect(screen.getByText(/WeeklyMAEvsNaiveZero=1\.2100 > 1\.12/)).toBeVisible();
  });
});
