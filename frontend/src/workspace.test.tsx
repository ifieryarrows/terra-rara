// @vitest-environment jsdom
import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ModelsPage } from './pages/ModelsPage';
import { ValidationPage } from './pages/ValidationPage';
import { SystemPage } from './pages/SystemPage';

const hooks = vi.hoisted(() => ({ model: vi.fn(), validation: vi.fn(), health: vi.fn() }));
vi.mock('./hooks/useQueries', () => ({ useTftModelSummary: hooks.model, useBacktestReport: hooks.validation, useSystemStatus: hooks.health }));
const result = (data: unknown) => ({ data, isLoading: false, isFetching: false, isError: false, refetch: vi.fn() });
beforeEach(() => { vi.clearAllMocks(); });
afterEach(cleanup);

describe('workspace evidence and disclosure', () => {
  it('keeps weekly evidence primary and exposes the daily diagnostics on request', async () => {
    hooks.model.mockReturnValue(result({ metrics: { weekly_directional_accuracy: .54, weekly_sample_count: 100, directional_accuracy: .49 } }));
    render(<ModelsPage/>);
    expect(screen.getByText('54.00%')).toBeVisible();
    expect(screen.getByText('Daily Directional Accuracy')).not.toBeVisible();
    await userEvent.click(screen.getByText('Daily diagnostics'));
    expect(screen.getByText('Daily Directional Accuracy')).toBeVisible();
    expect(screen.getByText('49.00%')).toBeVisible();
    expect(screen.getByText(/gate ≥ 53%/)).toBeVisible();
  });
  it('renders existing rolling-window report fields and preserves real zero values', () => {
    hooks.validation.mockReturnValue(result({ summary_metrics: { mean_da: 0, mean_sharpe: 0, mean_mae: .018, mean_vr: 1.1 }, window_metrics: [{ da: 0, sharpe: 0, mae: .02 }], theta_comparison: { tft_da: 0, theta_da: .5, tft_mae: .018 } }));
    render(<ValidationPage/>);
    expect(screen.getAllByText('0.00%')).toHaveLength(3);
    expect(screen.getByText('50.00%')).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Against the Theta baseline' })).toBeVisible();
    const mae = within(screen.getAllByRole('row').find(row => row.textContent?.startsWith('MAE'))!);
    expect(mae.getByText('0.0180')).toBeVisible();
    expect(mae.getByText('—')).toBeVisible();
  });
  it('does not label an unrelated comparison payload as Theta metrics', () => {
    hooks.validation.mockReturnValue(result({ summary_metrics: {}, theta_comparison: { tft_better_mae: false } }));
    render(<ValidationPage/>);
    expect(screen.queryByRole('heading', { name: 'Against the Theta baseline' })).not.toBeInTheDocument();
    expect(screen.getByText('Comparison report details')).toBeVisible();
  });
  it('retains retry and the validation error state', async () => {
    const retry = vi.fn();
    hooks.validation.mockReturnValue({ ...result(undefined), isError: true, refetch: retry });
    render(<ValidationPage/>);
    expect(screen.getByRole('alert')).toHaveTextContent('could not be loaded');
    await userEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(retry).toHaveBeenCalledOnce();
  });
  it('does not fabricate a free pipeline or zero models from absent health fields', () => {
    hooks.health.mockReturnValue(result({}));
    render(<SystemPage/>);
    const row = screen.getByText('Trained models on disk').closest('.cm-status-row')!;
    expect(row).toHaveTextContent('—');
    expect(screen.getByText('Pipeline lock').closest('.cm-status-row')).toHaveTextContent('unknown');
    expect(screen.queryByText('free')).not.toBeInTheDocument();
  });
});
