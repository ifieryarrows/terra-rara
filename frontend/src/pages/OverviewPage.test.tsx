// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { AppShell } from '../layouts/AppShell';
import { OverviewPage } from './OverviewPage';
import * as api from '../api';
import type {
  AnalysisReport,
  BacktestReportResponse,
  CommentaryResponse,
  ConsensusSignal,
  ExplainResponse,
  HealthResponse,
  HistoryResponse,
  LongTermOutlookResponse,
  NewsListResponse,
  NewsStatsResponse,
  TFTAnalysisResponse,
  TFTModelSummaryResponse,
} from '../types';

vi.mock('../api', async () => {
  const actual = await vi.importActual<typeof import('../api')>('../api');
  return {
    ...actual,
    fetchAnalysis: vi.fn(),
    fetchHistory: vi.fn(),
    fetchHealth: vi.fn(),
    fetchCommentary: vi.fn(),
    fetchMarketPrices: vi.fn(),
    fetchTFTAnalysis: vi.fn(),
    fetchExplain: vi.fn(),
    fetchLivePrice: vi.fn(),
    fetchConsensusSignal: vi.fn(),
    fetchTftModelSummary: vi.fn(),
    fetchLatestBacktest: vi.fn(),
    fetchLongTermOutlook: vi.fn(),
    fetchMarketHeatmap: vi.fn(),
    fetchHeatmapCategoryContext: vi.fn(),
    fetchSentimentSummary: vi.fn(),
    fetchNews: vi.fn(),
    fetchNewsById: vi.fn(),
    fetchNewsStats: vi.fn(),
  };
});

const ANALYSIS_FIXTURE: AnalysisReport = {
  symbol: 'HG=F',
  current_price: 4.525,
  predicted_return: 0.0085,
  predicted_price: 4.5635,
  baseline_price: 4.525,
  baseline_price_date: '2026-11-11',
  target_type: 'next_day_simple_return',
  confidence_lower: -0.006,
  confidence_upper: 0.022,
  sentiment_index: 0.34,
  sentiment_label: 'Bullish',
  top_influencers: [
    { feature: 'dxy_ret_5d', importance: 0.31, label: 'USD Dollar 5D Drift', category: 'Macro' },
    { feature: 'lme_inv_delta', importance: 0.24, label: 'Exchange Inventory Draw', category: 'Supply' },
    { feature: 'china_pmi_mom', importance: 0.19, label: 'China Industrial Momentum', category: 'Demand' },
  ],
  data_quality: {
    news_count_7d: 42,
    missing_days: 0,
    coverage_pct: 100,
  },
  generated_at: '2026-11-12T10:00:00Z',
};

const HISTORY_FIXTURE: HistoryResponse = {
  symbol: 'HG=F',
  data: Array.from({ length: 12 }, (_, idx) => ({
    date: `2026-11-${String(idx + 1).padStart(2, '0')}`,
    price: 4.41 + idx * 0.011,
    sentiment_index: 0.18 + idx * 0.01,
    sentiment_news_count: 5,
  })),
};

const TFT_FIXTURE: TFTAnalysisResponse = {
  symbol: 'HG=F',
  model_type: 'TFT-ASRO',
  model_state: 'ready',
  quality_state: 'verified',
  is_forecast_healthy: true,
  direction: 'BULLISH',
  weekly_trend: 'BULLISH',
  primary_horizon: '5D',
  primary_forecast_return: 0.0245,
  primary_forecast_q10: -0.008,
  primary_forecast_q90: 0.051,
  reference_price_date: '2026-11-11',
  t1_impulse: 'BULLISH',
  t1_return: 0.0042,
  weekly_forecast: {
    horizon: '5D',
    expected_return: 0.0245,
    q10_return: -0.008,
    q90_return: 0.051,
    calibrated: true,
    t1_impulse: 'BULLISH',
    t1_return: 0.0042,
    regime: 'trending_bull',
  },
  risk_level: 'MEDIUM',
  prediction: {
    predicted_return_median: 0.0042,
    predicted_return_q10: -0.004,
    predicted_return_q90: 0.012,
    predicted_price_median: 4.544,
    predicted_price_q10: 4.507,
    predicted_price_q90: 4.579,
    confidence_band_96: [4.46, 4.65],
    volatility_estimate: 0.014,
    quantiles: { '0.1': -0.004, '0.25': -0.001, '0.5': 0.0042, '0.75': 0.008, '0.9': 0.012 },
    weekly_return: 0.0245,
    weekly_price: 4.636,
    weekly_return_q10_raw: -0.009,
    weekly_return_q90_raw: 0.053,
    weekly_return_q10_calibrated: -0.008,
    weekly_return_q90_calibrated: 0.051,
    weekly_interval_calibrated: true,
    prediction_horizon_days: 5,
    reference_price: 4.525,
    reference_price_date: '2026-11-11',
    daily_forecasts: Array.from({ length: 5 }, (_, idx) => ({
      day: idx + 1,
      forecast_date: `2026-11-${13 + idx}`,
      daily_return: 0.0048,
      cumulative_return: (idx + 1) * 0.0049,
      price_median: 4.525 * (1 + (idx + 1) * 0.0049),
      price_q10: 4.49,
      price_q90: 4.61,
      price_q02: 4.45,
      price_q98: 4.66,
    })),
  },
  model_metadata: {
    symbol: 'HG=F',
    trained_at: '2026-11-10T03:00:00Z',
    checkpoint_path: 'models/tft_asro_hg_f.ckpt',
    config: { encoder_length: 60, prediction_length: 5 },
    metrics: {
      directional_accuracy: 0.57,
      weekly_directional_accuracy: 0.64,
      weekly_sample_count: 140,
      sharpe_ratio: 1.28,
      sortino_ratio: 1.74,
      mae: 0.012,
      rmse: 0.017,
    },
  },
  generated_at: '2026-11-12T10:00:00Z',
};

const EXPLAIN_FIXTURE: ExplainResponse = {
  schema_version: 1,
  symbol: 'HG=F',
  reference_price_date: '2026-11-11',
  generated_at: '2026-11-12T10:00:00Z',
  run_id: 'run-test-1',
  model: { checkpoint: 'models/tft_asro_hg_f.ckpt', trained_at: '2026-11-10T03:00:00Z' },
  horizon: { label: '5D', days: 5 },
  layout_hash: { encoder: 'enc-1', decoder: 'dec-1', static: 'stat-1' },
  variable_selection: {
    method: 'vsn_encoder',
    encoder: [
      {
        feature: 'comex_inventory_draw',
        weight: 0.34,
        label: 'COMEX Warehouse Draws',
        description: 'Prompt copper warehouse net withdrawals',
        category: 'Supply',
        time_horizon: '5D',
      },
      {
        feature: 'dxy_momentum',
        weight: 0.21,
        label: 'US Dollar Index Drift',
        description: 'Broad USD impulse affecting metals pricing',
        category: 'Macro',
        time_horizon: '5D',
      },
    ],
    decoder: [],
    static: [],
  },
  attention: {
    method: 'temporal_attention',
    step: { kind: 'horizon', horizon_index: 4 },
    window: { start: '2026-11-01', end: '2026-11-11', days: 10 },
    top_days: [{ date: '2026-11-10', weight: 0.28 }],
    series: [
      { date: '2026-11-08', weight: 0.11 },
      { date: '2026-11-09', weight: 0.16 },
      { date: '2026-11-10', weight: 0.28 },
      { date: '2026-11-11', weight: 0.22 },
    ],
  },
  notes: 'Raw encoder variable weights prior to quantile calibration.',
};

const COMMENTARY_FIXTURE: CommentaryResponse = {
  symbol: 'HG=F',
  commentary: 'RISKS (1 factor):\n• Dollar firmness.\nOPPORTUNITIES (1 factor):\n• Warehouse draws.\nSYNTHESIS:\nCopper 5D trend remains positive.',
  error: null,
  generated_at: '2026-11-12T10:00:00Z',
  ai_stance: 'BULLISH',
  generation_mode: 'llm',
};

const HEALTH_FIXTURE: HealthResponse = {
  status: 'healthy',
  db_type: 'postgres',
  models_found: 2,
  pipeline_locked: false,
  timestamp: '2026-11-12T10:00:00Z',
  last_pipeline_status: 'ok',
  last_pipeline_run_at: '2026-11-12T09:50:00Z',
  price_bar_latest_date: '2026-11-11',
  price_bar_staleness_days: 1,
};

const CONSENSUS_FIXTURE: ConsensusSignal = {
  consensus_direction: 'BULLISH',
  confidence: 'HIGH',
  position_scale: 0.85,
  blended_return: 0.018,
  xgb_return_raw: 0.0085,
  xgb_return_adjusted: 0.0079,
  tft_return: 0.0245,
  xgb_direction: 1,
  tft_direction: 1,
};

const TFT_SUMMARY_FIXTURE: TFTModelSummaryResponse = {
  symbol: 'HG=F',
  trained_at: '2026-11-10T03:00:00Z',
  checkpoint_path: 'models/tft_asro_hg_f.ckpt',
  config: { encoder_length: 60 },
  metrics: { weekly_directional_accuracy: 0.64, sharpe_ratio: 1.28, sortino_ratio: 1.74 },
  variable_importance: [],
  variable_importance_method: 'vsn_encoder',
  quality_gate: { passed: true, reasons: [], warnings: [], metrics: { weekly_raw_magnitude_ratio: 1.12 } },
};

const BACKTEST_FIXTURE: BacktestReportResponse = {
  report_date: '2026-11-10',
  summary_metrics: { directional_accuracy: 0.64, sharpe_ratio: 1.28 },
  window_metrics: [],
  theta_comparison: null,
  verdict: 'PASSED',
};

const SENTIMENT_SUMMARY_FIXTURE: api.SentimentSummary = {
  index: 0.34,
  label: 'Bullish',
  source: 'daily_v2',
  components: {
    llm_impact_avg: 0.31,
    finbert_pn_avg: 0.29,
    rule_sign_avg: 0.18,
    avg_confidence: 0.82,
    avg_relevance: 0.88,
    sample_size: 42,
  },
  trend: [{ date: '2026-11-11', index: 0.34, news_count: 14 }],
  recent_articles: [],
  data_freshness: {
    newest: '2026-11-12T09:30:00Z',
    oldest: '2026-11-05T09:30:00Z',
    age_hours: 1.5,
    article_count_24h: 11,
  },
  generated_at: '2026-11-12T10:00:00Z',
};

const NEWS_LIST_FIXTURE: NewsListResponse = {
  items: [
    {
      id: 101,
      raw_id: 501,
      title: 'COMEX copper stocks decline on tight smelter supply',
      description: 'Prompt spreads firm across North American hubs.',
      url: 'https://example.com/copper-stocks',
      channel: 'newsapi',
      publisher: 'Reuters',
      source_feed: 'Reuters Metals',
      published_at: '2026-11-12T08:00:00Z',
      fetched_at: '2026-11-12T08:05:00Z',
      language: 'en',
      sentiment: {
        label: 'BULLISH',
        final_score: 0.62,
        impact_score_llm: 0.58,
        confidence: 0.88,
        relevance: 0.94,
        event_type: 'supply_disruption',
        finbert: { pos: 0.72, neu: 0.21, neg: 0.07 },
        reasoning: 'Tight inventory draw supports near-term price floor.',
        scored_at: '2026-11-12T08:06:00Z',
        scoring_mode: 'llm',
        model_name: null,
        fallback_reason: null,
      },
    },
  ],
  total: 1,
  limit: 20,
  offset: 0,
  has_more: false,
  generated_at: '2026-11-12T10:00:00Z',
  as_of: '2026-11-12T10:00:00Z',
  data_as_of: '2026-11-12T08:00:00Z',
  filters: {},
};

const NEWS_STATS_FIXTURE: NewsStatsResponse = {
  window_hours: 168,
  total_articles: 42,
  scored_articles: 42,
  label_distribution: { BULLISH: 24, NEUTRAL: 12, BEARISH: 6 },
  event_type_distribution: { supply_disruption: 14, macro_policy: 10 },
  channel_distribution: { newsapi: 28, google_news: 14 },
  top_publishers: [{ publisher: 'Reuters', count: 14, avg_final_score: 0.45 }],
  avg_final_score: 0.34,
  avg_confidence: 0.82,
  avg_relevance: 0.88,
  generated_at: '2026-11-12T10:00:00Z',
};

const HEATMAP_FIXTURE = {
  id: 'metals',
  name: 'Metals Ecosystem',
  children: [
    {
      id: 'copper_miners',
      name: 'Copper Miners',
      children: [{ id: 'FCX', name: 'Freeport-McMoRan', symbol: 'FCX', value: 120, changePct: 2.35 }],
    },
  ],
  _meta: { payload_count: 1, refresh_in_progress: false },
};

const LONG_TERM_FIXTURE: LongTermOutlookResponse = {
  status: 'ok',
  symbol: 'HG=F',
  as_of_date: '2026-11-11',
  generated_at: '2026-11-12T10:00:00Z',
  model_version: 'lt_structural_v1',
  source: 'snapshot',
  current_price: 4.525,
  available_horizons: ['20D', '60D'],
  default_horizon: '20D',
  data_window: {
    start_date: '2011-11-18',
    end_date: '2026-11-11',
    weekly_bars: 780,
    years_covered: 15,
    fred_macro_enabled: true,
    symbols_used: ['HG=F', 'DX-Y.NYB', 'GXC'],
  },
  horizons: {
    '20D': {
      horizon_key: '20D',
      horizon_label: '1 Month (20D)',
      horizon_days: 20,
      horizon_weeks: 4,
      target_date: '2026-12-09',
      reference_price: 4.525,
      target_price_q50: 4.68,
      expected_return: 0.0343,
      expected_return_pct: 3.43,
      expected_log_return: 0.0337,
      price_q10: 4.24,
      price_q25: 4.46,
      price_q75: 4.91,
      price_q90: 5.12,
      return_q10: -0.063,
      return_q25: -0.014,
      return_q75: 0.085,
      return_q90: 0.131,
      stance: 'BULLISH',
      prob_bullish: 0.56,
      prob_neutral: 0.27,
      prob_bearish: 0.17,
      conviction: 0.64,
      conviction_label: 'HIGH',
      regime_threshold_pct: 2.5,
      components: {
        macro_ridge_return: 0.038,
        theta_damped_return: 0.026,
        macro_ridge_weight: 0.65,
        theta_weight: 0.35,
        realized_vol_annualized: 0.2,
        horizon_vol: 0.056,
        cqr_q10_width: 0.09,
        cqr_q90_width: 0.088,
      },
      key_drivers: [],
    },
    '60D': {
      horizon_key: '60D',
      horizon_label: '1 Quarter (60D)',
      horizon_days: 60,
      horizon_weeks: 12,
      target_date: '2027-02-05',
      reference_price: 4.525,
      target_price_q50: 4.82,
      expected_return: 0.0652,
      expected_return_pct: 6.52,
      expected_log_return: 0.0631,
      price_q10: 4.12,
      price_q25: 4.48,
      price_q75: 5.15,
      price_q90: 5.54,
      return_q10: -0.089,
      return_q25: -0.01,
      return_q75: 0.138,
      return_q90: 0.224,
      stance: 'BULLISH',
      prob_bullish: 0.6,
      prob_neutral: 0.24,
      prob_bearish: 0.16,
      conviction: 0.68,
      conviction_label: 'HIGH',
      regime_threshold_pct: 4.4,
      components: {
        macro_ridge_return: 0.072,
        theta_damped_return: 0.052,
        macro_ridge_weight: 0.65,
        theta_weight: 0.35,
        realized_vol_annualized: 0.2,
        horizon_vol: 0.097,
        cqr_q10_width: 0.14,
        cqr_q90_width: 0.145,
      },
      key_drivers: [],
    },
  },
  macro_scorecard: {
    composite_score: 0.36,
    composite_stance: 'BULLISH',
    pillars: {
      trend_structure: {
        key: 'trend_structure',
        label: 'Trend Structure',
        score: 0.45,
        score_pct: 72,
        stance: 'BULLISH',
        summary: 'Above 40W moving average.',
        metrics: [],
      },
      usd_macro: {
        key: 'usd_macro',
        label: 'USD & Macro',
        score: 0.2,
        score_pct: 60,
        stance: 'NEUTRAL',
        summary: 'USD neutral.',
        metrics: [],
      },
      china_industrial_demand: {
        key: 'china_industrial_demand',
        label: 'China & Industrial Demand',
        score: 0.35,
        score_pct: 68,
        stance: 'BULLISH',
        summary: 'China demand firm.',
        metrics: [],
      },
      cross_asset_confirmation: {
        key: 'cross_asset_confirmation',
        label: 'Cross-Asset Confirmation',
        score: 0.4,
        score_pct: 70,
        stance: 'BULLISH',
        summary: 'Miners confirming.',
        metrics: [],
      },
    },
  },
  validation: {
    '20D': {
      horizon_key: '20D',
      horizon_weeks: 4,
      purge_gap_weeks: 4,
      annualisation: 12,
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
      annualisation: 4,
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

function installMockApiDefaults() {
  vi.mocked(api.fetchAnalysis).mockResolvedValue(ANALYSIS_FIXTURE);
  vi.mocked(api.fetchHistory).mockResolvedValue(HISTORY_FIXTURE);
  vi.mocked(api.fetchHealth).mockResolvedValue(HEALTH_FIXTURE);
  vi.mocked(api.fetchCommentary).mockResolvedValue(COMMENTARY_FIXTURE);
  vi.mocked(api.fetchMarketPrices).mockResolvedValue({
    symbols: { 'HG=F': { price: 4.525, change: 0.82, date: '2026-11-11' } },
  });
  vi.mocked(api.fetchTFTAnalysis).mockResolvedValue(TFT_FIXTURE);
  vi.mocked(api.fetchExplain).mockResolvedValue(EXPLAIN_FIXTURE);
  vi.mocked(api.fetchLivePrice).mockResolvedValue({ symbol: 'HG=F', price: 4.525, error: null });
  vi.mocked(api.fetchConsensusSignal).mockResolvedValue(CONSENSUS_FIXTURE);
  vi.mocked(api.fetchTftModelSummary).mockResolvedValue(TFT_SUMMARY_FIXTURE);
  vi.mocked(api.fetchLatestBacktest).mockResolvedValue(BACKTEST_FIXTURE);
  vi.mocked(api.fetchLongTermOutlook).mockResolvedValue(LONG_TERM_FIXTURE);
  vi.mocked(api.fetchMarketHeatmap).mockResolvedValue(HEATMAP_FIXTURE);
  vi.mocked(api.fetchHeatmapCategoryContext).mockResolvedValue({
    categoryId: 'copper_miners',
    categoryName: 'Copper Miners',
    symbolCount: 1,
    news: null,
  });
  vi.mocked(api.fetchSentimentSummary).mockResolvedValue(SENTIMENT_SUMMARY_FIXTURE);
  vi.mocked(api.fetchNews).mockResolvedValue(NEWS_LIST_FIXTURE);
  vi.mocked(api.fetchNewsStats).mockResolvedValue(NEWS_STATS_FIXTURE);
}

function setupBrowserStubs() {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: false,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
  }));

  class MockResizeObserver {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
  }
  vi.stubGlobal('ResizeObserver', MockResizeObserver);

  class MockIntersectionObserver {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
  }
  vi.stubGlobal('IntersectionObserver', MockIntersectionObserver);

  HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    getImageData: vi.fn(),
    putImageData: vi.fn(),
    createImageData: vi.fn(),
    setTransform: vi.fn(),
    drawImage: vi.fn(),
    save: vi.fn(),
    fillText: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
    translate: vi.fn(),
    scale: vi.fn(),
    rotate: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    measureText: vi.fn().mockReturnValue({ width: 0 }),
    createRadialGradient: vi.fn().mockReturnValue({ addColorStop: vi.fn() }),
  });
}

describe('OverviewPage unboxed workspace streams, horizon separation & Cuprum integration', () => {
  beforeEach(() => {
    window.localStorage.clear();
    setupBrowserStubs();
    installMockApiDefaults();
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it('renders OverviewSkeleton while initial analysis data is pending', () => {
    vi.mocked(api.fetchAnalysis).mockImplementation(() => new Promise(() => {}));
    vi.mocked(api.fetchHistory).mockImplementation(() => new Promise(() => {}));
    vi.mocked(api.fetchTFTAnalysis).mockImplementation(() => new Promise(() => {}));

    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/dashboard']}>
          <OverviewPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    const skeleton = screen.getByRole('status', { name: 'Loading copper market overview' });
    expect(skeleton).toHaveAttribute('aria-busy', 'true');
    expect(skeleton).toHaveAttribute('data-cm-dashboard-ready', 'false');
  });

  it('enforces honest 5D (√52) vs T+1 (√252) separation, switches interactive SegmentedTabs lenses, and mounts global Cuprum', async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, staleTime: 60_000 } },
    });

    const { container } = render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/dashboard']}>
          <Routes>
            <Route element={<AppShell />}>
              <Route path="/dashboard" element={<OverviewPage />} />
            </Route>
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );

    // Wait for dashboard ready state + lazy panels
    await waitFor(() => {
      const root = container.querySelector('[data-cm-dashboard-ready="true"]');
      expect(root).not.toBeNull();
    });

    // Strict 5D vs T+1 separation + non-causal XAI focus disclosure + 1M/1Q Long-Term Outlook
    expect(screen.getByText('5D TFT Primary')).toBeInTheDocument();
    expect(screen.getByText('Expected 5D Performance')).toBeInTheDocument();
    expect(screen.getByText('T+1 TFT Step 1')).toBeInTheDocument();
    expect(screen.getByText(/Raw 5D Focus · Non-causal/i)).toBeInTheDocument();
    expect(screen.getByText('Quality Gate Passed')).toBeInTheDocument();
    expect(screen.getByText('T+1 Daily Drivers · XGBoost')).toBeInTheDocument();
    expect(screen.getByText(/Auxiliary single-day \(T\+1\) XGBoost feature weights/i)).toBeInTheDocument();
    expect(screen.getByTestId('news-sentiment-readout')).toHaveTextContent(/▲\s*Bullish/);
    expect(screen.getByRole('meter', { name: /7D News Sentiment/i })).toHaveAttribute('value', '0.34');

    // Long-Term Outlook (1M / 1Q) lazy surface resolves cleanly
    const ltPanel = await screen.findByTestId('long-term-outlook-panel', {}, { timeout: 12_000 });
    expect(ltPanel).toBeInTheDocument();
    expect(within(ltPanel).getByTestId('lt-target-price')).toHaveTextContent('$4.68');
    fireEvent.click(within(ltPanel).getByRole('tab', { name: /1 Quarter \(60D\)/i }));
    expect(within(ltPanel).getByTestId('lt-target-price')).toHaveTextContent('$4.82');

    // Global Cuprum dock mounted via AppShell with synced telemetry context
    const cuprumStage = await screen.findByTestId('cuprum-companion-stage', {}, { timeout: 12_000 });
    expect(cuprumStage).toBeInTheDocument();

    // Stream lens SegmentedTabs filtering across views
    const lensTablist = screen.getByRole('tablist', { name: /Filter open dashboard intelligence stream/i });
    expect(lensTablist).toBeInTheDocument();

    expect(container.querySelector('#price-forecast')).toBeInTheDocument();
    expect(container.querySelector('#weekly-outlook')).toBeInTheDocument();
    expect(container.querySelector('#model-reliability')).toBeInTheDocument();
    expect(container.querySelector('#long-term-outlook')).toBeInTheDocument();
    expect(container.querySelector('#market-drivers')).toBeInTheDocument();
    expect(container.querySelector('#neural-analysis')).toBeInTheDocument();
    expect(container.querySelector('#news-intelligence')).toBeInTheDocument();
    expect(container.querySelector('#market-map')).toBeInTheDocument();

    // Switch to 5D & T+1 Models
    fireEvent.click(within(lensTablist).getByRole('tab', { name: /5D & T\+1 Models/i }));
    expect(container.querySelector('#price-forecast')).toBeInTheDocument();
    expect(container.querySelector('#weekly-outlook')).toBeInTheDocument();
    expect(container.querySelector('#long-term-outlook')).toBeInTheDocument();
    expect(container.querySelector('#news-intelligence')).not.toBeInTheDocument();
    expect(container.querySelector('#market-map')).not.toBeInTheDocument();

    // Switch to 1M / 1Q Outlook lens
    fireEvent.click(within(lensTablist).getByRole('tab', { name: /1M \/ 1Q Outlook/i }));
    expect(container.querySelector('#long-term-outlook')).toBeInTheDocument();
    expect(container.querySelector('#price-forecast')).not.toBeInTheDocument();

    // Switch to XAI Focus & Drivers (5D TFT VSN vs T+1 XGBoost drivers)
    fireEvent.click(within(lensTablist).getByRole('tab', { name: /XAI Focus & Drivers/i }));
    const driversSection = container.querySelector('#market-drivers') as HTMLElement;
    expect(driversSection).toBeInTheDocument();
    expect(within(driversSection).getByText('+0.85%')).toBeInTheDocument();
    expect(container.querySelector('#neural-analysis')).toBeInTheDocument();
    expect(await within(container.querySelector('#neural-analysis') as HTMLElement).findByText('COMEX Warehouse Draws', {}, { timeout: 8_000 })).toBeInTheDocument();
    expect(container.querySelector('#price-forecast')).not.toBeInTheDocument();

    // Interactive driver category filtering inside T+1 Daily Drivers
    fireEvent.click(within(driversSection).getByRole('button', { name: /^Macro/i }));
    expect(within(driversSection).getByText('USD Dollar 5D Drift')).toBeInTheDocument();
    expect(within(driversSection).queryByText('Exchange Inventory Draw')).not.toBeInTheDocument();

    // Switch to Copper News
    fireEvent.click(within(lensTablist).getByRole('tab', { name: /Copper News/i }));
    expect(container.querySelector('#news-intelligence')).toBeInTheDocument();
    expect(container.querySelector('#market-drivers')).not.toBeInTheDocument();

    // Switch to Metals Map
    fireEvent.click(within(lensTablist).getByRole('tab', { name: /Metals Map/i }));
    expect(container.querySelector('#market-map')).toBeInTheDocument();
    expect(container.querySelector('#news-intelligence')).not.toBeInTheDocument();

    // Restore All Streams
    fireEvent.click(within(lensTablist).getByRole('tab', { name: /All Streams/i }));
    expect(container.querySelector('#price-forecast')).toBeInTheDocument();
    expect(container.querySelector('#long-term-outlook')).toBeInTheDocument();
    expect(container.querySelector('#news-intelligence')).toBeInTheDocument();
    expect(container.querySelector('#market-map')).toBeInTheDocument();
  }, 25_000);
});
