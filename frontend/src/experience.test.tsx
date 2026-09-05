// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

vi.mock('@vercel/speed-insights/react', () => ({ SpeedInsights: () => null }));
// Keep navigation tests independent of live financial services.
vi.mock('./pages/OverviewPage', () => ({ OverviewPage: () => <h1>Market overview</h1> }));
vi.mock('./pages/ModelsPage', () => ({ ModelsPage: () => <h1>Model intelligence</h1> }));
vi.mock('./pages/ValidationPage', () => ({ ValidationPage: () => <h1>Validation report</h1> }));
vi.mock('./pages/SystemPage', () => ({ SystemPage: () => <h1>System health</h1> }));

beforeEach(() => {
  vi.stubGlobal('matchMedia', (query: string) => ({ matches: query.includes('prefers-reduced-motion'), media: query, addEventListener: vi.fn(), removeEventListener: vi.fn(), addListener: vi.fn(), removeListener: vi.fn() }));
  window.scrollTo = vi.fn();
  window.history.replaceState({}, '', '/');
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('product introduction and workspace routes', () => {
  it('keeps the introduction useful in reduced motion and identifies sample data', () => {
    render(<App/>);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Read the market.');
    expect(document.querySelector('.cm-story--static')).not.toBeNull();
    expect(screen.getAllByText('Illustrative preview').length).toBeGreaterThan(0);
    expect(screen.getAllByRole('link', { name: 'Enter CopperMind' })).toHaveLength(2);
    expect(screen.queryByRole('heading', { name: 'Market overview' })).not.toBeInTheDocument();
  });
  it('enters the actual dashboard route without completing the story and restores a focus target', async () => {
    const user = userEvent.setup();
    render(<App/>);
    await user.click(screen.getByRole('link', { name: 'Open dashboard' }));
    expect(await screen.findByRole('heading', { name: 'Market overview' })).toBeInTheDocument();
    expect(window.location.pathname).toBe('/dashboard');
    await waitFor(() => expect(document.getElementById('main-content')).toHaveFocus());
    expect(screen.getByRole('navigation', { name: 'Workspace' })).toBeInTheDocument();
    expect(document.querySelector('.cm-story')).toBeNull();
  });
  it.each([['/models', 'Model intelligence'], ['/validation', 'Validation report'], ['/system', 'System health']])('preserves direct navigation to %s', async (path, heading) => {
    window.history.replaceState({}, '', path);
    render(<App/>);
    expect(await screen.findByRole('heading', { name: heading })).toBeInTheDocument();
    expect(window.location.pathname).toBe(path);
  });
  it('preserves a query-bearing legacy root link when moving to the dashboard', async () => {
    window.history.replaceState({}, '', '/?symbol=HG%3DF');
    render(<App/>);
    expect(await screen.findByRole('heading', { name: 'Market overview' })).toBeInTheDocument();
    expect(window.location.pathname).toBe('/dashboard');
    expect(window.location.search).toBe('?symbol=HG%3DF');
  });
  it('provides a way out of an unknown URL', () => {
    window.history.replaceState({}, '', '/unknown-view');
    render(<App/>);
    expect(screen.getByRole('heading', { name: 'That page is not here.' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open dashboard' })).toHaveAttribute('href', '/dashboard');
  });
});
