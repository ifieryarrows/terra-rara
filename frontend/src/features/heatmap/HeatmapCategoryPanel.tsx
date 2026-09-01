import React, { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { CategoryAnchor } from './HeatmapTreemap';
import type { HeatmapData } from './heatmap-layout';
import { Sparkline } from './Sparkline';
import { useHeatmapCategoryContext } from '../../hooks/useQueries';
import { computePointerPanelPosition } from './heatmap-utils';

const ROW_HEIGHT = 40;
const OVERSCAN = 6;
const MAX_VISIBLE_ROWS = 9;
const POINTER_INTERACTION_IDLE_MS = 120;

interface Props {
  categoryId: string;
  categoryName: string;
  leaves: HeatmapData[];
  activeLeaf?: HeatmapData | null;
  anchor: CategoryAnchor;
  view: 'market' | 'themes';
  pinned: boolean;
  onClose: () => void;
  onPointerEnter: () => void;
  onPointerLeave: () => void;
}

export interface HeatmapCategoryPanelHandle {
  move: (x: number, y: number) => void;
}

const HeatmapCategoryPanel = memo(forwardRef<HeatmapCategoryPanelHandle, Props>(function HeatmapCategoryPanel({
  categoryId,
  categoryName,
  leaves,
  activeLeaf,
  anchor,
  view,
  pinned,
  onClose,
  onPointerEnter,
  onPointerLeave,
}, ref) {
  const [scrollTop, setScrollTop] = useState(0);
  const scrollFrame = useRef<number | null>(null);
  const positionFrame = useRef<number | null>(null);
  const interactionTimer = useRef<number | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLElement>(null);
  const panelSizeRef = useRef({ width: 380, height: 480 });
  const pointerRef = useRef(anchor.pointer || { x: anchor.rect.right, y: anchor.rect.top });
  const scopedLeaves = useMemo(() => {
    if (!activeLeaf) return leaves;
    const activeIndustry = (activeLeaf.industry || activeLeaf.subgroup || '').trim().toLocaleLowerCase();
    const activeSector = (activeLeaf.sector || activeLeaf.group || '').trim().toLocaleLowerCase();
    if (!activeIndustry) return leaves;
    const matches = leaves.filter((item) => {
      const industry = (item.industry || item.subgroup || '').trim().toLocaleLowerCase();
      const sector = (item.sector || item.group || '').trim().toLocaleLowerCase();
      return industry === activeIndustry && (!activeSector || !sector || sector === activeSector);
    });
    return matches.length ? matches : leaves;
  }, [activeLeaf, leaves]);
  const sorted = useMemo(
    () => [...scopedLeaves].sort((a, b) => (b.changePercent || 0) - (a.changePercent || 0)),
    [scopedLeaves],
  );
  const peers = useMemo(() => sorted.filter((item) => {
    if (!activeLeaf) return true;
    if (activeLeaf.id && item.id) return activeLeaf.id !== item.id;
    return activeLeaf.name !== item.name;
  }), [activeLeaf, sorted]);
  const sectorName = activeLeaf?.sector || activeLeaf?.group;
  const industryName = activeLeaf?.industry || activeLeaf?.subgroup || categoryName;
  const heading = activeLeaf && sectorName && sectorName !== industryName
    ? `${sectorName} - ${industryName}`
    : industryName;
  const { data: context } = useHeatmapCategoryContext(categoryId, view, true);
  const contextMatchesIndustry = !activeLeaf
    || (context?.categoryName || '').trim().toLocaleLowerCase() === (industryName || '').trim().toLocaleLowerCase();
  const stockNews = activeLeaf
    ? context?.stockNews?.[activeLeaf.name] || context?.stockNews?.[activeLeaf.name.toUpperCase()]
    : undefined;
  const visibleNews = activeLeaf ? stockNews : (contextMatchesIndustry ? context?.news : undefined);
  const initialPosition = computePointerPanelPosition(
    pointerRef.current.x,
    pointerRef.current.y,
    anchor.containerRect,
    window.innerWidth,
    window.innerHeight,
    undefined,
    undefined,
    activeLeaf ? anchor.rect : undefined,
  );
  const virtualized = peers.length > 40;
  const listHeight = Math.min(ROW_HEIGHT * MAX_VISIBLE_ROWS, Math.max(ROW_HEIGHT, peers.length * ROW_HEIGHT));
  const start = virtualized ? Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN) : 0;
  const end = virtualized ? Math.min(peers.length, start + MAX_VISIBLE_ROWS + OVERSCAN * 2) : peers.length;
  const visible = peers.slice(start, end);
  const style: React.CSSProperties = initialPosition.mode === 'sheet'
    ? { position: 'fixed', left: 0, right: 0, bottom: 0, maxHeight: initialPosition.maxHeight, zIndex: 70 }
    : {
        position: 'fixed', left: 0, top: 0, width: initialPosition.width, maxHeight: initialPosition.maxHeight, zIndex: 70,
        transform: `translate3d(${initialPosition.left}px, ${initialPosition.top}px, 0)`,
        willChange: 'transform',
      };
  const applyPosition = useCallback(() => {
    positionFrame.current = null;
    const element = panelRef.current;
    if (!element || window.innerWidth <= 640) return;
    const position = computePointerPanelPosition(
      pointerRef.current.x,
      pointerRef.current.y,
      anchor.containerRect,
      window.innerWidth,
      window.innerHeight,
      panelSizeRef.current.width,
      panelSizeRef.current.height,
      activeLeaf ? anchor.rect : undefined,
    );
    if (position.mode === 'float') {
      element.style.transform = `translate3d(${position.left}px, ${position.top}px, 0)`;
    }
  }, [activeLeaf, anchor.containerRect, anchor.rect]);

  const schedulePosition = useCallback(() => {
    if (positionFrame.current === null) positionFrame.current = requestAnimationFrame(applyPosition);
  }, [applyPosition]);

  const pausePointerInteraction = useCallback(() => {
    const element = panelRef.current;
    if (!element) return;
    if (!activeLeaf || pinned || window.innerWidth <= 640) {
      if (interactionTimer.current !== null) window.clearTimeout(interactionTimer.current);
      interactionTimer.current = null;
      element.style.pointerEvents = 'auto';
      return;
    }
    element.style.pointerEvents = 'none';
    if (interactionTimer.current !== null) window.clearTimeout(interactionTimer.current);
    interactionTimer.current = window.setTimeout(() => {
      interactionTimer.current = null;
      if (panelRef.current) panelRef.current.style.pointerEvents = 'auto';
    }, POINTER_INTERACTION_IDLE_MS);
  }, [activeLeaf, pinned]);

  useImperativeHandle(ref, () => ({
    move(x, y) {
      pointerRef.current = { x, y };
      if (positionFrame.current !== null) {
        cancelAnimationFrame(positionFrame.current);
        positionFrame.current = null;
      }
      pausePointerInteraction();
      applyPosition();
    },
  }), [applyPosition, pausePointerInteraction]);

  useLayoutEffect(() => {
    pointerRef.current = anchor.pointer || { x: anchor.rect.right, y: anchor.rect.top };
    const element = panelRef.current;
    if (element && window.innerWidth > 640) {
      const bounds = element.getBoundingClientRect();
      if (bounds.width > 0 && bounds.height > 0) {
        panelSizeRef.current = { width: bounds.width, height: bounds.height };
      }
    }
    applyPosition();
    pausePointerInteraction();
  }, [anchor, applyPosition, pausePointerInteraction]);

  useEffect(() => {
    setScrollTop(0);
    if (listRef.current) listRef.current.scrollTop = 0;
  }, [categoryId]);

  useEffect(() => {
    const element = panelRef.current;
    if (!element || typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver(([entry]) => {
      const width = entry.contentRect.width;
      const height = entry.contentRect.height;
      if (width > 0 && height > 0) panelSizeRef.current = { width: Math.min(380, width), height };
      schedulePosition();
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [schedulePosition]);

  useEffect(() => () => {
    if (scrollFrame.current !== null) cancelAnimationFrame(scrollFrame.current);
    if (positionFrame.current !== null) cancelAnimationFrame(positionFrame.current);
    if (interactionTimer.current !== null) window.clearTimeout(interactionTimer.current);
  }, []);

  return (
    <aside
      ref={panelRef}
      style={style}
      className="flex flex-col overflow-hidden rounded-lg border border-copper-400/40 bg-slate-950 text-slate-100 shadow-2xl"
      onPointerEnter={onPointerEnter}
      onPointerLeave={onPointerLeave}
      aria-label={`${categoryName} category details`}
    >
      <header className="flex min-h-10 items-center justify-between gap-3 border-b border-white/10 px-3 py-2" title={pinned ? 'Pinned category' : undefined}>
        <div className="min-w-0">
          <h3 className="truncate text-xs font-semibold tracking-wide text-slate-200">{heading}</h3>
        </div>
        <button type="button" onClick={onClose} className="flex h-6 w-6 shrink-0 items-center justify-center rounded text-base leading-none text-slate-500 hover:bg-white/5 hover:text-white" aria-label="Close category panel">×</button>
      </header>

      {visibleNews && (
        <a
          href={visibleNews.url || undefined}
          target={visibleNews.url ? '_blank' : undefined}
          rel="noopener noreferrer"
          className="block border-b border-white/10 bg-white/[0.025] px-3 py-2 hover:bg-white/[0.045]"
        >
          <div className="flex items-center gap-2 text-[9px] text-slate-500">
            <span>{visibleNews.publishedAt ? new Date(visibleNews.publishedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : 'Recent'}</span>
            {visibleNews.publisher && <span className="truncate text-slate-600">· {visibleNews.publisher}</span>}
            {visibleNews.sentiment && <span className="ml-auto shrink-0 text-copper-300">{visibleNews.sentiment}</span>}
          </div>
          <p className="mt-1 line-clamp-3 text-[11px] leading-[1.35] text-slate-300">{visibleNews.summary || visibleNews.title}</p>
        </a>
      )}

      {activeLeaf && context && !visibleNews && (
        <div className="flex min-h-9 items-center gap-2 border-b border-white/10 bg-white/[0.018] px-3 py-2 text-[10px] text-slate-500" role="status">
          <span aria-hidden="true" className="text-copper-400">✧</span>
          <span>No recent news is available for {activeLeaf.name}.</span>
        </div>
      )}

      {activeLeaf && (
        <div className="grid grid-cols-[minmax(0,1fr)_82px_58px_58px] items-center gap-2 border-b border-white/10 bg-white/[0.045] px-3 py-2.5" data-testid="selected-stock">
          <div className="min-w-0">
            <strong className="block truncate text-lg font-bold leading-none tracking-wide text-white">{activeLeaf.name}</strong>
            <span className="mt-1 block truncate text-[9px] leading-none text-slate-400" title={activeLeaf.shortName || activeLeaf.instrumentType || 'Instrument'}>{activeLeaf.shortName || activeLeaf.instrumentType || 'Instrument'}</span>
          </div>
          <Sparkline values={activeLeaf.sparkline} positive={(activeLeaf.changePercent || 0) >= 0} width={82} height={30} />
          <div className="text-right font-mono text-sm tabular-nums text-slate-100">{activeLeaf.price == null ? '—' : activeLeaf.price.toFixed(activeLeaf.price < 10 ? 3 : 2)}</div>
          <div className={`text-right font-mono text-sm font-bold tabular-nums ${(activeLeaf.changePercent || 0) > 0 ? 'text-emerald-400' : (activeLeaf.changePercent || 0) < 0 ? 'text-rose-400' : 'text-slate-400'}`}>
            {(activeLeaf.changePercent || 0) > 0 ? '+' : ''}{(activeLeaf.changePercent || 0).toFixed(2)}%
          </div>
        </div>
      )}

      <div
        ref={listRef}
        className="custom-scrollbar relative overflow-y-auto overscroll-contain"
        data-testid="peer-list"
        style={{ height: listHeight, minHeight: Math.min(ROW_HEIGHT, listHeight) }}
        onScroll={(event) => {
          const next = event.currentTarget.scrollTop;
          if (scrollFrame.current !== null) cancelAnimationFrame(scrollFrame.current);
          scrollFrame.current = requestAnimationFrame(() => setScrollTop(next));
        }}
      >
        {peers.length === 0 ? (
          <p className="px-3 py-3 text-[10px] text-slate-500">No additional peers in this category.</p>
        ) : (
          <div className="relative" style={{ height: virtualized ? peers.length * ROW_HEIGHT : 'auto' }}>
            {visible.map((item, index) => {
              const rowIndex = start + index;
              const change = item.changePercent || 0;
              return (
                <div
                  key={item.id || item.name}
                  data-testid="peer-row"
                  aria-label={`${item.name}, ${item.shortName || 'peer'}, ${item.price ?? 'price unavailable'}, ${change >= 0 ? 'plus' : 'minus'} ${Math.abs(change).toFixed(2)} percent`}
                  className={`grid grid-cols-[minmax(54px,1fr)_82px_58px_58px] items-center gap-2 border-b border-white/[0.05] px-3 hover:bg-white/[0.055] ${rowIndex % 2 ? 'bg-white/[0.025]' : 'bg-transparent'}`}
                  style={{
                    height: ROW_HEIGHT,
                    ...(virtualized ? { position: 'absolute', left: 0, right: 0, transform: `translateY(${rowIndex * ROW_HEIGHT}px)` } : {}),
                  }}
                >
                  <strong className="truncate text-xs font-semibold text-slate-100">{item.name}</strong>
                  <Sparkline values={item.sparkline} positive={change >= 0} width={82} height={22} />
                  <span className="text-right font-mono text-[11px] tabular-nums text-slate-300">{item.price == null ? '—' : item.price.toFixed(item.price < 10 ? 3 : 2)}</span>
                  <span className={`text-right font-mono text-[11px] font-semibold tabular-nums ${change > 0 ? 'text-emerald-400' : change < 0 ? 'text-rose-400' : 'text-slate-500'}`}>{change > 0 ? '+' : ''}{change.toFixed(2)}%</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}));

export default HeatmapCategoryPanel;
