// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { createRef } from 'react';
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CompanyLogo } from './CompanyLogo';
import HeatmapCategoryPanel, { type HeatmapCategoryPanelHandle } from './HeatmapCategoryPanel';
import { HeatmapPanel } from './HeatmapPanel';
import HeatmapTreemap from './HeatmapTreemap';
import { clampTooltipPosition, computePanelPosition, computePointerPanelPosition, getColorForChange } from './heatmap-utils';
import { resetFailedLogosForTests } from './logo-cache';
import type { HeatmapNode } from './heatmap-layout';

const heatmapQueryMocks = vi.hoisted(() => ({ context: null as any }));

vi.mock('../../hooks/useQueries', () => ({
  useHeatmapCategoryContext: () => ({ data: heatmapQueryMocks.context }),
  useMarketHeatmap: () => ({
    data: {
      id: 'root', name: 'Root', _meta: { payload_count: 2, is_stale: false, refresh_in_progress: false }, children: [{
        id: 'sector', name: 'Technology', children: [{
          id: 'industry', name: 'Semiconductors', children: [
            {
              id: 'nvda', name: 'NVDA', shortName: 'NVIDIA', weight: 100, price: 100, changePercent: 2,
              sector: 'Technology', industry: 'Semiconductors', sparkline: [1, 2, 3],
            },
            {
              id: 'amd', name: 'AMD', shortName: 'Advanced Micro Devices, Incorporated with a very long company name', weight: 70, price: 90, changePercent: -1,
              sector: 'Technology', industry: 'Semiconductors', sparkline: null,
            },
          ],
        }],
      }],
    },
    isError: false,
    isLoading: false,
    error: null,
  }),
}));

describe('heatmap interaction primitives', () => {
  beforeEach(() => {
    heatmapQueryMocks.context = null;
    vi.stubEnv('VITE_LOGO_DEV_PUBLISHABLE_KEY', 'pk_test');
    resetFailedLogosForTests();
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => window.setTimeout(() => callback(performance.now()), 0));
    vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id));
    class TestPointerEvent extends MouseEvent {
      pointerId: number;
      constructor(type: string, init: PointerEventInit = {}) {
        super(type, init);
        this.pointerId = init.pointerId ?? 1;
      }
    }
    vi.stubGlobal('PointerEvent', TestPointerEvent);
    class ImmediateResizeObserver {
      constructor(private callback: ResizeObserverCallback) {}
      observe(target: Element) {
        this.callback([{ target, contentRect: { width: 700, height: 560 } } as ResizeObserverEntry], this as unknown as ResizeObserver);
      }
      disconnect() {}
      unobserve() {}
    }
    vi.stubGlobal('ResizeObserver', ImmediateResizeObserver);
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it('lazy-loads eligible logos and replaces broken images with initials', () => {
    class ImmediateObserver {
      constructor(private callback: IntersectionObserverCallback) {}
      observe(element: Element) { this.callback([{ isIntersecting: true, target: element } as IntersectionObserverEntry], this as unknown as IntersectionObserver); }
      disconnect() {}
      unobserve() {}
      takeRecords() { return []; }
      root = null; rootMargin = ''; thresholds = [];
    }
    vi.stubGlobal('IntersectionObserver', ImmediateObserver);
    const { container } = render(<CompanyLogo ticker="BRK.B" label="Berkshire Hathaway" size={32} />);
    const image = container.querySelector('img') as HTMLImageElement;
    expect(image).toHaveAttribute('loading', 'lazy');
    expect(image).toHaveAttribute('decoding', 'async');
    expect(image.getAttribute('src')).toContain('/ticker/BRK-B');
    fireEvent.error(image);
    expect(screen.getByText('BR')).toBeTruthy();
  });

  it('uses the equity ticker when a stale snapshot has no logoTicker field', async () => {
    const data: HeatmapNode = {
      id: 'root', name: 'Root', children: [{
        id: 'sector', name: 'Technology', children: [{
          id: 'industry', name: 'Semiconductors', children: [{
            id: 'nvda', name: 'NVDA', shortName: 'NVIDIA', instrumentType: 'equity', weight: 100, price: 100, changePercent: 2,
          }],
        }],
      }],
    };
    const { container } = render(
      <HeatmapTreemap data={data} width={700} height={400} zoom={1} hoveredCategoryId={null} onCategoryHover={() => {}} />,
    );
    await waitFor(() => expect(container.querySelector('img[src*="/ticker/NVDA"]')).toBeTruthy());
  });

  it('clamps tooltips and category panels to visible bounds', () => {
    expect(clampTooltipPosition(990, 790, 1_000, 800)).toEqual({ left: 684, top: 622 });
    const anchor = { left: 900, top: 700, right: 980, bottom: 760, width: 80, height: 60 };
    const bounds = { left: 0, top: 0, right: 1_000, bottom: 800, width: 1_000, height: 800 };
    const position = computePanelPosition(anchor, bounds, 1_000, 800);
    expect(position.mode).toBe('float');
    expect(position.left).toBeGreaterThanOrEqual(10);
    expect(position.top + position.maxHeight).toBeLessThanOrEqual(790);
    const pointerPosition = computePointerPanelPosition(970, 760, bounds, 1_000, 800, 420, 400);
    expect(pointerPosition.mode).toBe('float');
    expect(pointerPosition.left).toBeLessThan(970);
    expect(pointerPosition.top).toBeLessThan(760);
    const wideCell = { left: 30, top: 120, right: 420, bottom: 520, width: 390, height: 400 };
    const firstCellPosition = computePointerPanelPosition(160, 300, bounds, 1_000, 800, 380, 480, wideCell);
    const movedCellPosition = computePointerPanelPosition(260, 300, bounds, 1_000, 800, 380, 480, wideCell);
    expect(firstCellPosition.left - 160).toBe(18);
    expect(movedCellPosition.left - firstCellPosition.left).toBe(100);
    expect(firstCellPosition.top).toBe(300 - 48);
    expect(movedCellPosition.top).toBe(firstCellPosition.top);
    expect(movedCellPosition.left + movedCellPosition.width).toBeLessThanOrEqual(bounds.right - 10);

    const moreRoomOnLeft = { left: 850, top: 120, right: 920, bottom: 320, width: 70, height: 200 };
    const leftLanePosition = computePointerPanelPosition(885, 200, { ...bounds, right: 1_400, width: 1_400 }, 1_400, 800, 380, 480, moreRoomOnLeft);
    expect(885 - (leftLanePosition.left + leftLanePosition.width)).toBe(18);
  });

  it('moves the stock panel synchronously and never intercepts unpinned hover', async () => {
    vi.useFakeTimers();
    const panelHandle = createRef<HeatmapCategoryPanelHandle>();
    const selected = {
      id: 'nvda', name: 'NVDA', shortName: 'NVIDIA', weight: 100, price: 100, changePercent: 2,
      sector: 'Technology', industry: 'Semiconductors', sparkline: [1, 2, 3],
    };
    const anchor = {
      id: 'industry', name: 'Semiconductors', depth: 2, pointer: { x: 300, y: 200 },
      rect: { left: 200, top: 100, right: 500, bottom: 400, width: 300, height: 300 },
      containerRect: { left: 0, top: 0, right: 1_400, bottom: 800, width: 1_400, height: 800 },
    };
    render(
      <HeatmapCategoryPanel
        ref={panelHandle}
        categoryId="industry"
        categoryName="Semiconductors"
        leaves={[selected]}
        activeLeaf={selected}
        anchor={anchor}
        view="market"
        pinned={false}
        onClose={() => {}}
        onPointerEnter={() => {}}
        onPointerLeave={() => {}}
      />,
    );
    const panel = screen.getByLabelText(/Semiconductors category details/i);
    const initialTransform = panel.style.transform;
    expect(panel.style.transition).toBe('');
    const widthRead = vi.fn(() => 380);
    const heightRead = vi.fn(() => 480);
    Object.defineProperty(panel, 'offsetWidth', { configurable: true, get: widthRead });
    Object.defineProperty(panel, 'offsetHeight', { configurable: true, get: heightRead });
    panelHandle.current?.move(400, 200);
    expect(panel.style.transform).not.toBe(initialTransform);
    expect(panel.style.pointerEvents).toBe('none');
    expect(widthRead).not.toHaveBeenCalled();
    expect(heightRead).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1_000);
    expect(panel.style.pointerEvents).toBe('none');
  });

  it('keeps duplicate provider IDs isolated during React reconciliation and hover', () => {
    const data: HeatmapNode = {
      id: 'root', name: 'Root', children: [{
        id: 'sector', name: 'EM & China', children: [{
          id: 'industry', name: 'Commodity EM', children: [
            { id: 'duplicate-ewz', name: 'EWZ', shortName: 'First EWZ', weight: 100 },
            { id: 'duplicate-ewz', name: 'EWZ', shortName: 'Second EWZ', weight: 90 },
          ],
        }],
      }],
    };
    const hover = vi.fn();
    const { container } = render(
      <HeatmapTreemap
        data={data}
        width={700}
        height={400}
        zoom={1}
        hoveredCategoryId={null}
        onCategoryHover={() => {}}
        onLeafHover={hover}
      />,
    );
    const cells = Array.from(container.querySelectorAll<HTMLElement>('[data-hm-leaf-id]'));
    const renderIds = cells.map((cell) => cell.dataset.hmLeafId);
    expect(cells).toHaveLength(2);
    expect(new Set(renderIds).size).toBe(2);
    fireEvent.pointerOver(cells[1], { clientX: 300, clientY: 200 });
    expect(hover).toHaveBeenLastCalledWith(expect.objectContaining({ shortName: 'Second EWZ' }));
  });

  it('uses a vivid continuous red-neutral-green market scale', () => {
    expect(getColorForChange(-5)).toBe('rgb(255,48,72)');
    expect(getColorForChange(0)).toBe('#414852');
    expect(getColorForChange(5)).toBe('rgb(18,196,91)');
  });

  it('supports category hover and Enter pinning through delegated handlers', () => {
    const data: HeatmapNode = {
      id: 'root', name: 'Root', children: [{
        id: 'sector', name: 'Technology', children: [{
          id: 'industry', name: 'Semiconductors', children: [{
            id: 'nvda', name: 'NVDA', shortName: 'NVIDIA', weight: 100, price: 100, changePercent: 2,
          }],
        }],
      }],
    };
    const hover = vi.fn();
    const pin = vi.fn();
    const open = vi.spyOn(window, 'open').mockImplementation(() => null);
    render(
      <HeatmapTreemap
        data={data}
        width={700}
        height={400}
        zoom={1}
        hoveredCategoryId={null}
        onCategoryHover={hover}
        onCategoryClick={pin}
      />,
    );
    const sector = screen.getByRole('button', { name: /Sector or asset class: Technology/i });
    fireEvent.pointerOver(sector, { clientX: 10, clientY: 10 });
    expect(hover).toHaveBeenCalledWith(expect.objectContaining({ id: 'sector' }));
    fireEvent.keyDown(sector, { key: 'Enter' });
    expect(pin).toHaveBeenCalledWith(expect.objectContaining({ id: 'sector' }));
    fireEvent.doubleClick(screen.getByRole('button', { name: /^NVDA,/i }));
    expect(open).toHaveBeenCalledWith('https://finance.yahoo.com/quote/NVDA', '_blank', 'noopener,noreferrer');
    expect(screen.getByRole('button', { name: /^NVDA,/i })).not.toHaveTextContent('$100');
  });

  it('zooms directly with the mouse wheel around the pointer', async () => {
    const data: HeatmapNode = {
      id: 'root', name: 'Root', children: [{
        id: 'sector', name: 'Technology', children: [{
          id: 'industry', name: 'Semiconductors', children: [{ id: 'nvda', name: 'NVDA', weight: 100 }],
        }],
      }],
    };
    const zoom = vi.fn();
    render(<HeatmapTreemap data={data} width={700} height={400} zoom={1} hoveredCategoryId={null} onCategoryHover={() => {}} onZoomDelta={zoom} />);
    const map = screen.getByLabelText(/Interactive market heatmap/i);
    const normal = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: -100 });
    map.dispatchEvent(normal);
    expect(normal.defaultPrevented).toBe(true);
    await new Promise((resolve) => window.setTimeout(resolve, 5));
    expect(zoom).toHaveBeenCalledTimes(1);
    expect(zoom.mock.calls[0][0]).toBeGreaterThan(0);
  });

  it('drags a zoomed map to pan without activating a category', () => {
    const data: HeatmapNode = {
      id: 'root', name: 'Root', children: [{
        id: 'sector', name: 'Technology', children: [{
          id: 'industry', name: 'Semiconductors', children: [{ id: 'nvda', name: 'NVDA', weight: 100 }],
        }],
      }],
    };
    const pin = vi.fn();
    render(<HeatmapTreemap data={data} width={700} height={400} zoom={2} hoveredCategoryId={null} onCategoryHover={() => {}} onCategoryClick={pin} />);
    const map = screen.getByLabelText(/Interactive market heatmap/i) as HTMLDivElement;
    Object.assign(map, {
      setPointerCapture: vi.fn(),
      hasPointerCapture: vi.fn(() => true),
      releasePointerCapture: vi.fn(),
    });
    const pointerDown = new PointerEvent('pointerdown', { pointerId: 7, clientX: 200, clientY: 180, button: 0, bubbles: true, cancelable: true });
    map.dispatchEvent(pointerDown);
    expect(pointerDown.defaultPrevented).toBe(true);
    fireEvent.pointerMove(map, { pointerId: 7, clientX: 140, clientY: 120 });
    fireEvent.pointerUp(map, { pointerId: 7, clientX: 140, clientY: 120, button: 0 });
    expect(map.scrollLeft).toBe(60);
    expect(map.scrollTop).toBe(60);
    expect(pin).not.toHaveBeenCalled();
  });

  it('merges hovered stock details into the peer panel without a second tooltip', async () => {
    vi.useFakeTimers();
    render(<HeatmapPanel />);
    await vi.advanceTimersByTimeAsync(20);
    const map = screen.getByLabelText(/Interactive market heatmap/i) as HTMLDivElement;
    map.getBoundingClientRect = () => ({
      x: 0, y: 0, left: 0, top: 0, right: 700, bottom: 560, width: 700, height: 560,
      toJSON: () => ({}),
    });
    fireEvent.pointerOver(screen.getByRole('button', { name: /^NVDA,/i }), { clientX: 160, clientY: 160 });
    fireEvent.pointerMove(map, { clientX: 220, clientY: 180 });
    await vi.advanceTimersByTimeAsync(90);
    const panel = screen.getByLabelText(/Semiconductors category details/i);
    expect(panel.style.width).toBe('380px');
    expect(panel.style.transition).toBe('');
    expect(panel.style.transform).toContain('translate3d(238px');
    expect(within(panel).getByText('Technology - Semiconductors')).toBeVisible();
    expect(within(panel).getAllByText('NVDA')).toHaveLength(1);
    expect(within(panel).getAllByTestId('peer-row')).toHaveLength(1);
    expect(within(panel).getByTestId('peer-row')).toHaveTextContent('AMD');
    expect(screen.queryByRole('tooltip')).toBeNull();

    fireEvent.pointerOver(screen.getByRole('button', { name: /^AMD,/i }), { clientX: 240, clientY: 180 });
    await vi.advanceTimersByTimeAsync(20);
    const selected = within(panel).getByTestId('selected-stock');
    expect(within(selected).getByText('AMD')).toBeVisible();
    expect(within(selected).getByTitle(/Advanced Micro Devices/i)).toBeVisible();
    expect(within(panel).getAllByText('AMD')).toHaveLength(1);
  });

  it('virtualizes a long peer list while keeping the selected stock fixed', () => {
    const selected = {
      id: 's0', name: 'S0', shortName: 'Selected company', weight: 100, price: 101, changePercent: 1,
      sector: 'Technology', industry: 'Semiconductors', sparkline: [1, 2, 3],
    };
    const leaves = [selected, ...Array.from({ length: 59 }, (_, index) => ({
      id: `s${index + 1}`, name: `S${index + 1}`, shortName: `Peer ${index + 1}`, weight: 10,
      price: 50 + index, changePercent: index % 2 ? -1 : 1, sector: 'Technology', industry: 'Semiconductors',
    }))];
    const anchor = {
      id: 'industry', name: 'Semiconductors', depth: 2, pointer: { x: 100, y: 100 },
      rect: { left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 },
      containerRect: { left: 0, top: 0, right: 700, bottom: 560, width: 700, height: 560 },
    };
    render(
      <HeatmapCategoryPanel
        categoryId="industry"
        categoryName="Semiconductors"
        leaves={leaves}
        activeLeaf={selected}
        anchor={anchor}
        view="market"
        pinned={false}
        onClose={() => {}}
        onPointerEnter={() => {}}
        onPointerLeave={() => {}}
      />,
    );
    const panel = screen.getByLabelText(/Semiconductors category details/i);
    expect(within(panel).getAllByText('S0')).toHaveLength(1);
    expect(within(panel).getByTestId('peer-list')).toHaveStyle({ height: '360px' });
    expect(within(panel).getAllByTestId('peer-row').length).toBeLessThan(leaves.length - 1);
    expect(within(panel).queryByText('Recent')).toBeNull();
  });

  it('uses the hovered category name instead of metadata from its first stock', () => {
    const leaf = {
      id: 'fcx', name: 'FCX', shortName: 'Freeport-McMoRan', weight: 100, price: 51, changePercent: 1,
      group: 'Copper Miners', subgroup: 'Major Producers', sparkline: [1, 2, 3],
    };
    const anchor = {
      id: 'other-equities', name: 'Other Equities', depth: 1, pointer: { x: 100, y: 100 },
      rect: { left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 },
      containerRect: { left: 0, top: 0, right: 700, bottom: 560, width: 700, height: 560 },
    };
    render(
      <HeatmapCategoryPanel
        categoryId="other-equities"
        categoryName="Other Equities"
        leaves={[leaf]}
        activeLeaf={null}
        anchor={anchor}
        view="market"
        pinned={false}
        onClose={() => {}}
        onPointerEnter={() => {}}
        onPointerLeave={() => {}}
      />,
    );
    const panel = screen.getByLabelText(/Other Equities category details/i);
    expect(within(panel).getByText('Other Equities')).toBeVisible();
    expect(within(panel).queryByText('Copper Miners - Major Producers')).toBeNull();
  });

  it('keeps the peer list scoped to the selected stock industry', () => {
    heatmapQueryMocks.context = {
      categoryId: 'broad-category', categoryName: 'Unclassified Equities', symbolCount: 3,
      news: { id: 1, title: 'Unrelated broad-category news', summary: null, url: null, publisher: null, publishedAt: null, sentiment: null },
      stockNews: {
        NVDA: {
          id: 2, title: 'NVIDIA raises its quarterly outlook', summary: 'Data-center demand supported the new outlook.',
          url: 'https://example.test/nvda', publisher: 'Market Wire', publishedAt: '2026-08-30T12:00:00Z', sentiment: 'POSITIVE',
        },
      },
    };
    const selected = {
      id: 'nvda', name: 'NVDA', shortName: 'NVIDIA', weight: 100, price: 100, changePercent: 2,
      sector: 'Technology', industry: 'Semiconductors', sparkline: [1, 2, 3],
    };
    const leaves = [
      selected,
      { id: 'amd', name: 'AMD', shortName: 'AMD', weight: 80, price: 90, changePercent: -1, sector: 'Technology', industry: 'Semiconductors' },
      { id: 'jpm', name: 'JPM', shortName: 'JPMorgan', weight: 90, price: 120, changePercent: 1, sector: 'Financial', industry: 'Banks' },
    ];
    const anchor = {
      id: 'broad-category', name: 'Unclassified Equities', depth: 2, pointer: { x: 100, y: 100 },
      rect: { left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 },
      containerRect: { left: 0, top: 0, right: 700, bottom: 560, width: 700, height: 560 },
    };
    render(
      <HeatmapCategoryPanel
        categoryId="broad-category"
        categoryName="Unclassified Equities"
        leaves={leaves}
        activeLeaf={selected}
        anchor={anchor}
        view="market"
        pinned={false}
        onClose={() => {}}
        onPointerEnter={() => {}}
        onPointerLeave={() => {}}
      />,
    );
    const panel = screen.getByLabelText(/Unclassified Equities category details/i);
    expect(within(panel).getByTestId('peer-row')).toHaveTextContent('AMD');
    expect(within(panel).queryByText('JPM')).toBeNull();
    expect(within(panel).queryByText('Unrelated broad-category news')).toBeNull();
    expect(within(panel).getByText('Data-center demand supported the new outlook.')).toBeVisible();
    expect(within(panel).getByText(/Market Wire/)).toBeVisible();
  });

  it('shows a compact unavailable message when the selected stock has no cached news', () => {
    heatmapQueryMocks.context = {
      categoryId: 'industry', categoryName: 'Semiconductors', symbolCount: 1, news: null, stockNews: {},
    };
    const selected = {
      id: 'amd', name: 'AMD', shortName: 'Advanced Micro Devices', weight: 100, price: 90, changePercent: -1,
      sector: 'Technology', industry: 'Semiconductors', sparkline: null,
    };
    const anchor = {
      id: 'industry', name: 'Semiconductors', depth: 2, pointer: { x: 100, y: 100 },
      rect: { left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 },
      containerRect: { left: 0, top: 0, right: 700, bottom: 560, width: 700, height: 560 },
    };
    render(
      <HeatmapCategoryPanel
        categoryId="industry"
        categoryName="Semiconductors"
        leaves={[selected]}
        activeLeaf={selected}
        anchor={anchor}
        view="market"
        pinned={false}
        onClose={() => {}}
        onPointerEnter={() => {}}
        onPointerLeave={() => {}}
      />,
    );
    expect(screen.getByText('No recent news is available for AMD.')).toBeVisible();
  });

  it('highlights the industry grid immediately while retaining panel hover intent', async () => {
    vi.useFakeTimers();
    render(<HeatmapPanel />);
    await vi.advanceTimersByTimeAsync(20);
    const map = screen.getByLabelText(/Interactive market heatmap/i) as HTMLDivElement;
    map.getBoundingClientRect = () => ({
      x: 0, y: 0, left: 0, top: 0, right: 700, bottom: 560, width: 700, height: 560,
      toJSON: () => ({}),
    });
    const stock = map.querySelector<HTMLElement>('[data-hm-leaf-id]')!;
    fireEvent.pointerOver(stock, { clientX: 100, clientY: 100 });
    const industryCells = Array.from(map.querySelectorAll<HTMLElement>('[data-hm-parent-id="industry"]'));
    const industry = map.querySelector<HTMLElement>('[data-hm-category-id="industry"]')!;
    expect(industryCells.length).toBeGreaterThan(1);
    industryCells.forEach((cell) => {
      expect(cell.classList.contains('border')).toBe(false);
    });
    expect(industry).toHaveStyle({ backgroundColor: '#d99a5b' });
    await vi.advanceTimersByTimeAsync(89);
    expect(screen.queryByLabelText(/Semiconductors category details/i)).toBeNull();
    await vi.advanceTimersByTimeAsync(1);
    const panel = screen.getByLabelText(/Semiconductors category details/i);
    const initialTransform = panel.style.transform;
    fireEvent.pointerMove(stock, { clientX: 620, clientY: 300 });
    await vi.advanceTimersByTimeAsync(20);
    expect(panel.style.transform).not.toBe(initialTransform);
    fireEvent.pointerOut(stock, { relatedTarget: document.body });
    expect(industry).toHaveStyle({ backgroundColor: '#020617' });
    await vi.advanceTimersByTimeAsync(100);
    fireEvent.pointerEnter(panel);
    await vi.advanceTimersByTimeAsync(200);
    expect(screen.getByLabelText(/Semiconductors category details/i)).toBeVisible();
    fireEvent.pointerLeave(panel);
    await vi.advanceTimersByTimeAsync(180);
    expect(screen.queryByLabelText(/Semiconductors category details/i)).toBeNull();
  });
});
