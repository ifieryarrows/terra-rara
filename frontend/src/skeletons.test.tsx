// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { OverviewPage } from './pages/OverviewPage';
import { ModelsPage } from './pages/ModelsPage';
import { ValidationPage } from './pages/ValidationPage';
import { SystemPage } from './pages/SystemPage';
import {
  OverviewSkeleton,
  ModelsSkeleton,
  ValidationSkeleton,
  SystemSkeleton,
} from './components/skeletons';

const hooks = vi.hoisted(() => ({
  model: vi.fn(),
  validation: vi.fn(),
  health: vi.fn(),
  sentiment: vi.fn(),
}));

const loadingResult = () => ({
  data: undefined,
  isLoading: true,
  isFetching: true,
  isError: false,
  refetch: vi.fn(),
});

vi.mock('./hooks/useQueries', () => ({
  useTftModelSummary: () => hooks.model(),
  useBacktestReport: () => hooks.validation(),
  useSystemStatus: () => hooks.health(),
  useSentimentSummary: () => hooks.sentiment(),
}));

vi.mock('./api', () => ({
  fetchAnalysis: vi.fn(() => new Promise(() => {})),
  fetchHistory: vi.fn(() => new Promise(() => {})),
  fetchCommentary: vi.fn(() => new Promise(() => {})),
  fetchTFTAnalysis: vi.fn(() => new Promise(() => {})),
  fetchLivePrice: vi.fn(() => new Promise(() => {})),
}));

beforeEach(() => {
  vi.clearAllMocks();
  hooks.model.mockReturnValue(loadingResult());
  hooks.validation.mockReturnValue(loadingResult());
  hooks.health.mockReturnValue(loadingResult());
  hooks.sentiment.mockReturnValue(loadingResult());
});

afterEach(cleanup);

describe('workspace skeleton loading views', () => {
  describe('OverviewPage skeleton loading state', () => {
    it('renders high-fidelity skeleton view instead of generic 3-line ViewState', () => {
      render(<OverviewPage />);
      const status = screen.getByRole('status', { name: /loading copper market overview/i });
      expect(status).toHaveAttribute('aria-busy', 'true');
      expect(status).toHaveAttribute('aria-live', 'polite');

      // Check quote strip skeleton
      expect(status.querySelector('.cm-dashboard-hero-data')).toBeInTheDocument();
      expect(status.querySelector('.cm-dashboard-price-row')).toBeInTheDocument();
      expect(status.querySelector('.cm-dashboard-sentiment')).toBeInTheDocument();
      expect(status.querySelector('.cm-dashboard-sentiment .cm-sentiment-badge')).toBeInTheDocument();
      expect(status.querySelector('.cm-dashboard-sentiment .cm-dashboard-sentiment-score')).toBeInTheDocument();
      expect(status.querySelector('.cm-overview-tools a[href="#price-forecast"]')).toHaveAttribute('tabIndex', '-1');
      expect(status.querySelector('.cm-overview-tools button.cm-refresh')).toBeDisabled();
      expect(status.querySelector('.cm-overview-tools button.cm-refresh')).toHaveAttribute('tabIndex', '-1');

      // Check price forecast chart container skeleton
      const chartSection = status.querySelector('#price-forecast');
      expect(chartSection).toBeInTheDocument();
      expect(chartSection?.querySelector('.cm-price-chart')).toBeInTheDocument();
      expect(chartSection?.querySelector('.cm-chart-toolbar')).toBeInTheDocument();
      expect(chartSection?.querySelector('.cm-chart-window-options')).toBeInTheDocument();
      expect(chartSection?.querySelector('.cm-chart-legend')).toBeInTheDocument();
      expect(chartSection?.querySelector('.cm-price-plot.cm-chart-skeleton-plot')).toBeInTheDocument();
      expect(chartSection?.querySelector('.cm-chart-note')).toBeInTheDocument();
      expect(chartSection?.querySelector('.cm-chart-table summary')).toHaveAttribute('tabIndex', '-1');

      // Check weekly outlook and bottom intelligence panels
      expect(status.querySelector('#weekly-outlook')).toBeInTheDocument();
      expect(status.querySelector('#weekly-outlook .cm-data-disclosure summary')).toHaveAttribute('tabIndex', '-1');
      expect(status.querySelector('#market-drivers')).toBeInTheDocument();
      expect(status.querySelector('#model-reliability')).toBeInTheDocument();
      expect(status.querySelector('#model-reliability .cm-reliability')).toBeInTheDocument();
      expect(status.querySelectorAll('#model-reliability .cm-reliability-row')).toHaveLength(2);
      expect(status.querySelectorAll('#model-reliability .cm-reliability-row small')).toHaveLength(2);
      expect(status.querySelector('#model-reliability .cm-text-link')).toBeInTheDocument();
      expect(status.querySelector('#model-reliability .pt-3')).not.toBeInTheDocument();
      expect(status.querySelector('#neural-analysis')).toBeInTheDocument();
      expect(status.querySelector('#neural-analysis .cm-commentary')).toBeInTheDocument();
      expect(status.querySelector('#news-intelligence')).toBeInTheDocument();
      expect(status.querySelector('#news-intelligence .cm-news-panel')).toBeInTheDocument();
      expect(status.querySelector('#news-intelligence .cm-news-controls')).toBeInTheDocument();
      expect(status.querySelector('#news-intelligence .cm-news-feed')).toBeInTheDocument();
      expect(status.querySelectorAll('#news-intelligence .cm-news-card')).toHaveLength(3);
      expect(status.querySelector('#market-map')).toBeInTheDocument();

      // Ensure no generic 3-line skeleton is rendered on OverviewPage
      expect(status.querySelector('.cm-view-state')).not.toBeInTheDocument();
    });

    it('ensures internal shimmer bones are hidden from screen readers', () => {
      render(<OverviewSkeleton />);
      const bones = document.querySelectorAll('.cm-skeleton-bone');
      expect(bones.length).toBeGreaterThan(10);
      bones.forEach((bone) => {
        expect(bone).toHaveAttribute('aria-hidden', 'true');
      });
    });
  });

  describe('ModelsPage skeleton loading state', () => {
    it('renders checkpoint info card, metrics grids, and feature importance skeleton', () => {
      render(<ModelsPage />);
      const status = screen.getByRole('status', { name: /loading model intelligence/i });
      expect(status).toHaveAttribute('aria-busy', 'true');
      expect(status).toHaveAttribute('aria-live', 'polite');

      // Checkpoint info card skeleton
      expect(screen.getByRole('heading', { name: 'TFT-ASRO Model' })).toBeInTheDocument();

      // Action chip skeleton
      expect(status.querySelector('.cm-skeleton-bone--chip, .rounded-full')).toBeInTheDocument();

      // Primary weekly strategy metrics grid (8 cards)
      expect(screen.getByText('Weekly strategy')).toBeInTheDocument();
      expect(screen.getByText('Weekly Directional Accuracy')).toBeInTheDocument();
      expect(screen.getByText('Weekly Sharpe Ratio')).toBeInTheDocument();
      expect(screen.getByText('Weekly Sortino Ratio')).toBeInTheDocument();
      expect(screen.getByText('Weekly Magnitude Ratio')).toBeInTheDocument();

      // Single-step daily diagnostics metrics grid (8 cards)
      expect(screen.getByText(/Daily diagnostics/)).toBeInTheDocument();
      expect(screen.getByText('Daily Directional Accuracy')).toBeInTheDocument();
      expect(screen.getByText('Daily Sharpe Ratio')).toBeInTheDocument();

      // Verify daily diagnostics disclosure is closed by default to avoid CLS on data arrival
      const diagnostics = status.querySelector('.cm-diagnostics');
      expect(diagnostics).toBeInTheDocument();
      expect(diagnostics).not.toHaveAttribute('open');

      // Feature inputs / variable importance section skeleton
      expect(screen.getByText('Model inputs')).toBeInTheDocument();

      // Verify skeleton metric elements
      const metrics = status.querySelectorAll('.cm-metric');
      expect(metrics.length).toBe(16); // 8 weekly + 8 daily

      // Verify no generic ViewState loading box
      expect(status.querySelector('.cm-view-state')).not.toBeInTheDocument();
    });

    it('ensures internal shimmer bones are hidden from screen readers', () => {
      render(<ModelsSkeleton />);
      const bones = document.querySelectorAll('.cm-skeleton-bone');
      expect(bones.length).toBeGreaterThan(5);
      bones.forEach((bone) => {
        expect(bone).toHaveAttribute('aria-hidden', 'true');
      });
    });
  });

  describe('ValidationPage skeleton loading state', () => {
    it('renders backtest summary metrics, baseline comparison, and table skeleton', () => {
      render(<ValidationPage />);
      const status = screen.getByRole('status', { name: /loading walk-forward validation/i });
      expect(status).toHaveAttribute('aria-busy', 'true');
      expect(status).toHaveAttribute('aria-live', 'polite');

      // Header and Out-of-sample summary metrics (5 cards)
      expect(screen.getByRole('heading', { name: 'Walk-Forward Validation' })).toBeInTheDocument();
      expect(screen.getByText('Validation at a glance')).toBeInTheDocument();
      expect(screen.getByText('Directional Accuracy')).toBeInTheDocument();
      expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
      expect(screen.getByText('Variance Ratio')).toBeInTheDocument();
      expect(screen.getAllByText('MAE').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('RMSE').length).toBeGreaterThanOrEqual(1);

      // Verify metric hints are rendered only for metrics that have hints (MAE and RMSE)
      const hints = status.querySelectorAll('.cm-metric-grid .cm-metric-hint');
      expect(hints).toHaveLength(2);

      // Baseline comparison placeholder and details disclosure
      expect(screen.getByText('Against the Theta baseline')).toBeInTheDocument();
      expect(status.querySelector('.cm-data-disclosure')).toBeInTheDocument();
      expect(status.querySelector('.cm-data-disclosure summary')).toHaveAttribute('tabIndex', '-1');

      // Window-by-window slice table skeleton with accessible captions
      expect(screen.getByText('Window-by-window results')).toBeInTheDocument();
      const tables = status.querySelectorAll('table.cm-table');
      expect(tables.length).toBe(2); // 1 comparison table + 1 window slice table
      expect(tables[0].querySelector('caption')).toBeInTheDocument();
      expect(tables[1].querySelector('caption')).toBeInTheDocument();

      // Verify no generic ViewState loading box
      expect(status.querySelector('.cm-view-state')).not.toBeInTheDocument();
    });

    it('ensures internal shimmer bones are hidden from screen readers', () => {
      render(<ValidationSkeleton />);
      const bones = document.querySelectorAll('.cm-skeleton-bone');
      expect(bones.length).toBeGreaterThan(5);
      bones.forEach((bone) => {
        expect(bone).toHaveAttribute('aria-hidden', 'true');
      });
    });
  });

  describe('SystemPage skeleton loading state', () => {
    it('renders core service cards and freshness status table rows', () => {
      render(<SystemPage />);
      const status = screen.getByRole('status', { name: /loading system status/i });
      expect(status).toHaveAttribute('aria-busy', 'true');
      expect(status).toHaveAttribute('aria-live', 'polite');

      // Header
      expect(screen.getByRole('heading', { name: 'System Status' })).toBeInTheDocument();

      // Core services & Snapshot panels
      expect(screen.getByText('Core services')).toBeInTheDocument();
      expect(screen.getByText('Database')).toBeInTheDocument();
      expect(screen.getByText('Redis queue')).toBeInTheDocument();
      expect(screen.getByText('Pipeline lock')).toBeInTheDocument();
      expect(screen.getByText('Trained models on disk')).toBeInTheDocument();

      expect(screen.getByText('Snapshot & data')).toBeInTheDocument();
      expect(screen.getByText('Latest snapshot age')).toBeInTheDocument();

      // Data freshness section
      expect(screen.getByText('Data freshness')).toBeInTheDocument();
      expect(screen.getByText('Pipeline run (worker) completed')).toBeInTheDocument();
      expect(screen.getByText('Pipeline status')).toBeInTheDocument();
      expect(screen.getByText('TFT model trained')).toBeInTheDocument();

      // Model artifact storage section
      expect(screen.getByText('Model artifact storage')).toBeInTheDocument();
      expect(screen.getByText(/HF Hub is used/)).toBeInTheDocument();

      // Verify status rows structure
      const rows = status.querySelectorAll('.cm-status-row');
      expect(rows.length).toBe(16); // 4 core + 4 snapshot + 8 freshness

      // Verify no generic ViewState loading box
      expect(status.querySelector('.cm-view-state')).not.toBeInTheDocument();
    });

    it('ensures internal shimmer bones are hidden from screen readers', () => {
      render(<SystemSkeleton />);
      const bones = document.querySelectorAll('.cm-skeleton-bone');
      expect(bones.length).toBeGreaterThan(5);
      bones.forEach((bone) => {
        expect(bone).toHaveAttribute('aria-hidden', 'true');
      });
    });
  });

  describe('preservation of error and empty states in ViewState', () => {
    it('preserves error ViewState on ModelsPage when query fails', () => {
      hooks.model.mockReturnValue({ data: undefined, isLoading: false, isError: true, isFetching: false, refetch: vi.fn() });
      render(<ModelsPage />);
      expect(screen.getByRole('alert')).toHaveTextContent('Model intelligence could not be loaded');
    });

    it('preserves empty ViewState on ModelsPage when data is absent', () => {
      hooks.model.mockReturnValue({ data: null, isLoading: false, isError: false, isFetching: false, refetch: vi.fn() });
      render(<ModelsPage />);
      expect(screen.getByRole('status')).toHaveTextContent('No model metadata available');
    });

    it('preserves error ViewState on ValidationPage when query fails', () => {
      hooks.validation.mockReturnValue({ data: undefined, isLoading: false, isError: true, isFetching: false, refetch: vi.fn() });
      render(<ValidationPage />);
      expect(screen.getByRole('alert')).toHaveTextContent('The validation report could not be loaded');
    });

    it('preserves empty ViewState on ValidationPage when report is unavailable', () => {
      hooks.validation.mockReturnValue({ data: { available: false }, isLoading: false, isError: false, isFetching: false, refetch: vi.fn() });
      render(<ValidationPage />);
      expect(screen.getByRole('status')).toHaveTextContent('No backtest report available yet');
    });

    it('preserves error ViewState on SystemPage when health check fails', () => {
      hooks.health.mockReturnValue({ data: undefined, isLoading: false, isError: true, isFetching: false, refetch: vi.fn() });
      render(<SystemPage />);
      expect(screen.getByRole('alert')).toHaveTextContent('System status is unavailable');
    });
  });

  describe('prefers-reduced-motion safeguards', () => {
    it('renders skeletons cleanly when prefers-reduced-motion is active', () => {
      vi.stubGlobal('matchMedia', (query: string) => ({
        matches: query.includes('prefers-reduced-motion'),
        media: query,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
      }));

      const { container } = render(<OverviewSkeleton />);
      const skeletonBones = container.querySelectorAll('.cm-skeleton-bone');
      expect(skeletonBones.length).toBeGreaterThan(0);
      expect(window.matchMedia('(prefers-reduced-motion: reduce)').matches).toBe(true);

      vi.unstubAllGlobals();
    });
  });
});
