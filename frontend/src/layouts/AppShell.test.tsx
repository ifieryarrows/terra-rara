// @vitest-environment jsdom
import { StrictMode } from 'react';
import { cleanup, render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes, Link, useLocation } from 'react-router-dom';
import { AppShell, SCROLL_THRESHOLD } from './AppShell';

describe('AppShell auto-hiding sticky header', () => {
  beforeEach(() => {
    vi.stubGlobal('matchMedia', (query: string) => ({
      matches: false,
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
    }));
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
      arc: vi.fn(),
      fill: vi.fn(),
    });
    Object.defineProperty(window, 'scrollY', { configurable: true, writable: true, value: 0 });
    Object.defineProperty(document.documentElement, 'scrollTop', { configurable: true, writable: true, value: 0 });
    delete (document as any).scrollingElement;
  });

  afterEach(() => {
    cleanup();
    delete (document as any).scrollingElement;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  const simulateScroll = async (y: number) => {
    act(() => {
      window.scrollY = y;
      document.documentElement.scrollTop = y;
      window.dispatchEvent(new Event('scroll'));
    });
  };

  it('renders brand, navigation links, and the skip navigation link', () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    expect(screen.getByRole('link', { name: 'Skip to workspace' })).toHaveAttribute('href', '#main-content');
    expect(screen.getByRole('link', { name: 'CopperMind Terra Rara home' })).toBeVisible();
    expect(screen.getByRole('link', { name: 'Overview' })).toBeVisible();
    expect(screen.getByRole('link', { name: 'Models' })).toBeVisible();
    expect(screen.getByRole('link', { name: 'Validation' })).toBeVisible();
    expect(screen.getByRole('link', { name: 'System' })).toBeVisible();
  });

  it('remains visible within the initial scroll threshold', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');
    expect(header).not.toHaveClass('cm-workspace-header--hidden');
    expect(header).toHaveAttribute('data-state', 'visible');

    // Scroll down but below threshold
    await simulateScroll(SCROLL_THRESHOLD - 20);
    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });

    // Scroll to exact threshold
    await simulateScroll(SCROLL_THRESHOLD);
    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });
  });

  it('hides the header when scrolling downward past the threshold', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    // Scroll beyond threshold
    await simulateScroll(150);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'hidden');
    });

    // Continuing downward keeps it hidden
    await simulateScroll(300);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'hidden');
    });
  });

  it('reveals the header immediately when scrolling upward from any depth', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    // First scroll down to hide it
    await simulateScroll(400);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Scroll upward from depth (e.g. 400 -> 360)
    await simulateScroll(360);
    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });

    // Scrolling down again re-hides it
    await simulateScroll(390);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'hidden');
    });
  });

  it('restores header visibility when returning to the very top of the page', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    // Scroll down deep
    await simulateScroll(500);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Return to top
    await simulateScroll(0);
    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });
  });

  it('handles negative scrollY (elastic rubber-band overscroll) safely', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    // Simulate rubber-band bounce at top
    await simulateScroll(-30);
    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });
  });

  it('reveals the header when keyboard focus enters header elements and hides on blur when scrolled past threshold', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    // Scroll down to hide header
    await simulateScroll(250);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Focus into a nav link inside the header
    const modelsLink = screen.getByRole('link', { name: 'Models' });
    modelsLink.focus();

    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });

    // Focus out of header to main content
    const mainContent = screen.getByRole('main');
    mainContent.focus();

    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'hidden');
    });
  });

  it('allows skip navigation link to be focused and functional without interfering with header', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const skipLink = screen.getByRole('link', { name: 'Skip to workspace' });
    skipLink.focus();
    expect(skipLink).toHaveFocus();
    expect(skipLink).toHaveAttribute('href', '#main-content');
  });

  it('restores header visibility upon route change', async () => {
    function TestApp() {
      return (
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/dashboard" element={<div>Dashboard content</div>} />
            <Route path="/models" element={<div>Models content</div>} />
          </Route>
        </Routes>
      );
    }

    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <TestApp />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    // Scroll down to hide
    await simulateScroll(200);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Navigate to /models by clicking the nav link
    const modelsLink = screen.getByRole('link', { name: 'Models' });
    await user.click(modelsLink);

    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });
  });

  it('prevents header pop-open during bottom elastic rubber-band overscroll bounce-back', async () => {
    Object.defineProperty(document.documentElement, 'scrollHeight', { configurable: true, writable: true, value: 2000 });
    Object.defineProperty(window, 'innerHeight', { configurable: true, writable: true, value: 800 });

    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    // Scroll to bottom (1200)
    await simulateScroll(1200);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Inertial overscroll past bottom (1260)
    await simulateScroll(1260);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Rubber-band elastic bounce back from 1260 to 1220
    await simulateScroll(1220);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Settles at bottom (1200)
    await simulateScroll(1200);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Intentionally scrolls upward (1200 -> 1150)
    await simulateScroll(1150);
    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });
  });

  it('resets scroll direction baseline across route changes so subsequent downward scroll hides header', async () => {
    function TestApp() {
      return (
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/dashboard" element={<div>Dashboard content</div>} />
            <Route path="/models" element={<div>Models content</div>} />
          </Route>
        </Routes>
      );
    }

    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <TestApp />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    // Scroll down to 300 on /dashboard -> hides
    await simulateScroll(300);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Navigate to /models
    const modelsLink = screen.getByRole('link', { name: 'Models' });
    await user.click(modelsLink);

    // Visibility restored on new route
    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });

    // Emulate page scroll reset to top upon route change and unfocusing header
    act(() => {
      (document.activeElement as HTMLElement)?.blur();
      window.scrollY = 0;
      document.documentElement.scrollTop = 0;
    });

    // Now user scrolls downward from 0 to 120 (past SCROLL_THRESHOLD) on /models
    await simulateScroll(120);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'hidden');
    });
  });

  it('initializes header as hidden when mounted already scrolled past threshold', async () => {
    window.scrollY = 350;
    document.documentElement.scrollTop = 350;

    render(
      <StrictMode>
        <MemoryRouter initialEntries={['/dashboard']}>
          <AppShell />
        </MemoryRouter>
      </StrictMode>
    );

    const header = screen.getByRole('banner');
    expect(header).toHaveClass('cm-workspace-header--hidden');
    expect(header).toHaveAttribute('data-state', 'hidden');

    // Scrolling up reveals it
    await simulateScroll(300);
    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });
  });

  it('handles window resize events without breaking scroll tracking', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    await simulateScroll(250);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    act(() => {
      window.dispatchEvent(new Event('resize'));
    });

    expect(header).toHaveClass('cm-workspace-header--hidden');

    // Scroll up after resize reveals
    await simulateScroll(200);
    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });
  });

  it('reveals the header when backward keyboard navigation enters from below', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    // Scroll down to hide
    await simulateScroll(250);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Focus main content, then focus backwards into the last header link ('System')
    const mainContent = screen.getByRole('main');
    mainContent.focus();

    const systemLink = screen.getByRole('link', { name: 'System' });
    systemLink.focus();

    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });
  });

  it('remains visible while an inner element is focused even when scrolling down, and hides upon blurring', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');
    const modelsLink = screen.getByRole('link', { name: 'Models' });

    // Focus link while at top
    modelsLink.focus();
    expect(header).not.toHaveClass('cm-workspace-header--hidden');

    // Scroll down to 300 while link is still focused
    await simulateScroll(300);
    await waitFor(() => {
      // Must remain visible for accessibility/focus indicator
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });

    // Blur the focused element to main content
    const mainContent = screen.getByRole('main');
    mainContent.focus();

    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'hidden');
    });
  });

  it('does not reset visibility or baseline when query parameters change on the same pathname', async () => {
    function QueryTestApp() {
      const location = useLocation();
      return (
        <Routes>
          <Route element={<AppShell />}>
            <Route
              path="/dashboard"
              element={
                <div>
                  <span>Search: {location.search}</span>
                  <Link to="/dashboard?filter=active">Apply filter</Link>
                </div>
              }
            />
          </Route>
        </Routes>
      );
    }

    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <QueryTestApp />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    // Scroll down past threshold to hide header
    await simulateScroll(300);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Unfocus and click link that changes search param only
    act(() => {
      (document.activeElement as HTMLElement)?.blur();
    });

    const filterLink = screen.getByRole('link', { name: 'Apply filter' });
    await user.click(filterLink);

    // Header must remain hidden because pathname did not change
    expect(screen.getByText('Search: ?filter=active')).toBeInTheDocument();
    expect(header).toHaveClass('cm-workspace-header--hidden');
    expect(header).toHaveAttribute('data-state', 'hidden');
  });

  it('reveals immediately on micro upward scroll of 1px from any depth', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');

    // Scroll down deep to hide
    await simulateScroll(500);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Micro upward scroll of only 1px (500 -> 499)
    await simulateScroll(499);
    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });
  });

  it('falls back to document.scrollingElement when window.scrollY is absent', async () => {
    Object.defineProperty(window, 'scrollY', { configurable: true, writable: true, value: undefined });
    Object.defineProperty(document, 'scrollingElement', {
      configurable: true,
      value: { scrollTop: 250, scrollHeight: 2000, clientHeight: 800 },
    });

    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');
    expect(header).toHaveClass('cm-workspace-header--hidden');
    expect(header).toHaveAttribute('data-state', 'hidden');
  });

  it('restores header visibility upon route change in StrictMode and tracks subsequent downward scroll', async () => {
    function StrictTestApp() {
      return (
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/dashboard" element={<div>Dashboard content</div>} />
            <Route path="/models" element={<div>Models content</div>} />
          </Route>
        </Routes>
      );
    }

    const user = userEvent.setup();
    render(
      <StrictMode>
        <MemoryRouter initialEntries={['/dashboard']}>
          <StrictTestApp />
        </MemoryRouter>
      </StrictMode>
    );

    const header = screen.getByRole('banner');

    // Scroll down to 250 on /dashboard -> hides
    await simulateScroll(250);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
    });

    // Navigate to /models in StrictMode
    const modelsLink = screen.getByRole('link', { name: 'Models' });
    await user.click(modelsLink);

    // Visibility restored on new route
    await waitFor(() => {
      expect(header).not.toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'visible');
    });

    // Scroll reset to top upon route change and header unfocused
    act(() => {
      (document.activeElement as HTMLElement)?.blur();
      window.scrollY = 0;
      document.documentElement.scrollTop = 0;
    });

    // Downward scroll past threshold hides header again
    await simulateScroll(100);
    await waitFor(() => {
      expect(header).toHaveClass('cm-workspace-header--hidden');
      expect(header).toHaveAttribute('data-state', 'hidden');
    });
  });

  it('cancels pending requestAnimationFrame on component unmount without error', async () => {
    const cancelSpy = vi.spyOn(window, 'cancelAnimationFrame');
    const { unmount } = render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    // Trigger a scroll which schedules a RAF
    act(() => {
      window.scrollY = 200;
      window.dispatchEvent(new Event('scroll'));
    });

    // Unmount before RAF finishes
    unmount();

    expect(cancelSpy).toHaveBeenCalled();
    cancelSpy.mockRestore();
  });

  it('handles headless zero viewport and zero scrollHeight gracefully without error', () => {
    Object.defineProperty(window, 'innerHeight', { configurable: true, writable: true, value: 0 });
    Object.defineProperty(document.documentElement, 'clientHeight', { configurable: true, writable: true, value: 0 });
    Object.defineProperty(document.documentElement, 'scrollHeight', { configurable: true, writable: true, value: 0 });

    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AppShell />
      </MemoryRouter>
    );

    const header = screen.getByRole('banner');
    expect(header).not.toHaveClass('cm-workspace-header--hidden');
    expect(header).toHaveAttribute('data-state', 'visible');
  });
});
