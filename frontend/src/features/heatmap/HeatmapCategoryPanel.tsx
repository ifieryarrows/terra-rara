import React, { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { CategoryAnchor } from './HeatmapTreemap';
import type { HeatmapData } from './heatmap-layout';
import { Sparkline } from './Sparkline';
import { useHeatmapCategoryContext } from '../../hooks/useQueries';
import { computePointerPanelPosition, HEATMAP_PANEL_DEFAULT_HEIGHT, HEATMAP_PANEL_WIDTH } from './heatmap-utils';

const ROW_HEIGHT = 25;
const OVERSCAN = 6;
const MAX_VISIBLE_ROWS = 8;
const VIRTUALIZATION_THRESHOLD = 8;

interface PeerRowProps {
  item: HeatmapData;
  rowIndex: number;
  rowHeight: number;
  virtualized: boolean;
}

const PeerRow = memo(function PeerRow({
  item,
  rowIndex,
  rowHeight,
  virtualized,
}: PeerRowProps) {
  const change = item.changePercent || 0;
  return (
    <div
      data-testid="peer-row"
      aria-label={`${item.name}, ${item.shortName || 'peer'}, ${item.price ?? 'price unavailable'}, ${change >= 0 ? 'plus' : 'minus'} ${Math.abs(change).toFixed(2)} percent`}
      className={`grid grid-cols-[minmax(0,1fr)_64px_52px_54px] items-center gap-1.5 border-b border-white/[0.05] px-2.5 hover:bg-white/[0.055] ${rowIndex % 2 ? 'bg-white/[0.025]' : 'bg-transparent'}`}
      style={{
        height: rowHeight,
        ...(virtualized ? { position: 'absolute', left: 0, right: 0, transform: `translateY(${rowIndex * rowHeight}px)` } : {}),
      }}
    >
      <strong className="truncate text-[11px] font-semibold text-slate-100">{item.name}</strong>
      <Sparkline values={item.sparkline} positive={change >= 0} width={64} height={15} />
      <span className="text-right font-mono text-[10.5px] tabular-nums text-slate-300">{item.price == null ? '—' : item.price.toFixed(item.price < 10 ? 3 : 2)}</span>
      <span className={`text-right font-mono text-[10.5px] font-semibold tabular-nums ${change > 0 ? 'text-emerald-400' : change < 0 ? 'text-rose-400' : 'text-slate-500'}`}>{change > 0 ? '+' : ''}{change.toFixed(2)}%</span>
    </div>
  );
});

interface Props {
  categoryId: string;
  categoryName: string;
  leaves: HeatmapData[];
  activeLeaf?: HeatmapData | null;
  anchor: CategoryAnchor;
  view: 'market' | 'themes';
  pinned: boolean;
  onClose: () => void;
  onDrillDown?: () => void;
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
  onDrillDown,
  onPointerEnter,
  onPointerLeave,
}, ref) {
  const [scrollTop, setScrollTop] = useState(0);
  const scrollFrame = useRef<number | null>(null);
  const positionFrame = useRef<number | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLElement>(null);
  const panelSizeRef = useRef({ width: HEATMAP_PANEL_WIDTH, height: HEATMAP_PANEL_DEFAULT_HEIGHT });
  const pointerRef = useRef(anchor.pointer || { x: anchor.rect.right, y: anchor.rect.top });
  const sorted = useMemo(
    () => [...leaves].sort((a, b) => (b.changePercent || 0) - (a.changePercent || 0)),
    [leaves],
  );
  const leavesByIndustry = useMemo(() => {
    const groups = new Map<string, HeatmapData[]>();
    sorted.forEach((item) => {
      const industry = (item.industry || item.subgroup || '').trim().toLocaleLowerCase();
      if (!industry) return;
      const group = groups.get(industry) || [];
      group.push(item);
      groups.set(industry, group);
    });
    return groups;
  }, [sorted]);
  const peers = useMemo(() => {
    if (!activeLeaf) return sorted;
    const industry = (activeLeaf.industry || activeLeaf.subgroup || '').trim().toLocaleLowerCase();
    const sector = (activeLeaf.sector || activeLeaf.group || '').trim().toLocaleLowerCase();
    const industryLeaves = industry ? leavesByIndustry.get(industry) : undefined;
    const scopedLeaves = industryLeaves && sector
      ? industryLeaves.filter((item) => {
        const itemSector = (item.sector || item.group || '').trim().toLocaleLowerCase();
        return !itemSector || itemSector === sector;
      })
      : industryLeaves;
    const candidates = scopedLeaves?.length ? scopedLeaves : sorted;
    return candidates.filter((item) => {
      if (activeLeaf.id && item.id) return activeLeaf.id !== item.id;
      return activeLeaf.name !== item.name;
    });
  }, [activeLeaf, leavesByIndustry, sorted]);
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
  const isSheet = initialPosition.mode === 'sheet';
  const isInteractive = pinned || isSheet;
  const maxRows = isSheet ? 2 : MAX_VISIBLE_ROWS;
  const rowHeight = isSheet ? 24 : ROW_HEIGHT;
  const virtualized = peers.length > VIRTUALIZATION_THRESHOLD;
  const listHeight = Math.min(rowHeight * maxRows, Math.max(rowHeight, peers.length * rowHeight));
  const start = virtualized ? Math.max(0, Math.floor(scrollTop / rowHeight) - OVERSCAN) : 0;
  const end = virtualized ? Math.min(peers.length, start + maxRows + OVERSCAN * 2) : peers.length;
  const visible = peers.slice(start, end);
  const style: React.CSSProperties = isSheet
    ? { position: 'absolute', left: 6, right: 6, bottom: 6, maxHeight: Math.min(144, Math.max(108, Math.round(anchor.containerRect.height * 0.34))), zIndex: 70 }
    : {
        position: 'fixed', left: 0, top: 0, width: initialPosition.width, maxHeight: initialPosition.maxHeight, zIndex: 70,
        transform: `translate3d(${initialPosition.left}px, ${initialPosition.top}px, 0)`,
        willChange: 'transform', pointerEvents: pinned ? 'auto' : 'none',
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
  const applyPositionRef = useRef(applyPosition);
  applyPositionRef.current = applyPosition;

  const schedulePosition = useCallback(() => {
    if (positionFrame.current === null) {
      positionFrame.current = requestAnimationFrame(() => applyPositionRef.current());
    }
  }, []);

  useImperativeHandle(ref, () => ({
    move(x, y) {
      pointerRef.current = { x, y };
      schedulePosition();
    },
  }), [schedulePosition]);

  useLayoutEffect(() => {
    if (typeof ResizeObserver !== 'undefined') return;
    const element = panelRef.current;
    if (element && window.innerWidth > 640) {
      const bounds = element.getBoundingClientRect();
      if (bounds.width > 0 && bounds.height > 0) {
        panelSizeRef.current = { width: Math.min(HEATMAP_PANEL_WIDTH, bounds.width), height: bounds.height };
      }
    }
  }, [categoryId, isSheet]);

  useLayoutEffect(() => {
    pointerRef.current = anchor.pointer || { x: anchor.rect.right, y: anchor.rect.top };
    applyPosition();
  }, [anchor, applyPosition]);

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
      if (width > 0 && height > 0) panelSizeRef.current = { width: Math.min(HEATMAP_PANEL_WIDTH, width), height };
      schedulePosition();
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [categoryId, isSheet, schedulePosition]);

  useEffect(() => () => {
    if (scrollFrame.current !== null) cancelAnimationFrame(scrollFrame.current);
    if (positionFrame.current !== null) cancelAnimationFrame(positionFrame.current);
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
      <header className={`flex items-center justify-between gap-2 border-b border-white/10 ${isSheet ? 'min-h-7 px-2.5 py-0.5' : 'min-h-7 px-2.5 py-1'}`} title={pinned ? 'Pinned category' : undefined}>
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-[11px] font-semibold tracking-wide text-slate-200">{heading}</h3>
        </div>
        {isInteractive && (
          <div className="flex shrink-0 items-center gap-1">
            {onDrillDown && (
              <button
                type="button"
                onClick={onDrillDown}
                aria-label={`Drill into ${categoryName}`}
                className="rounded border border-copper-400/35 bg-copper-500/10 px-1.5 py-0.5 text-[10px] font-medium text-copper-300 transition hover:border-copper-400/60 hover:bg-copper-500/20 hover:text-white"
              >
                Focus →
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              aria-label="Close category details"
              className={`inline-grid place-items-center rounded leading-none text-slate-400 transition hover:bg-white/10 hover:text-white ${isSheet ? 'h-7 w-7 text-sm' : 'h-5 w-5 text-xs'}`}
            >
              ×
            </button>
          </div>
        )}
      </header>

      {isSheet ? (
        visibleNews && (
          <a
            href={visibleNews.url || undefined}
            target={visibleNews.url ? '_blank' : undefined}
            rel="noopener noreferrer"
            className="block border-b border-white/10 bg-white/[0.025] px-2.5 py-1 hover:bg-white/[0.045]"
          >
            <div className="flex items-center gap-2 text-[9px] leading-none text-slate-500">
              <span>{visibleNews.publishedAt ? new Date(visibleNews.publishedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : 'Recent'}</span>
              {visibleNews.publisher && <span className="truncate text-slate-600">· {visibleNews.publisher}</span>}
              {visibleNews.sentiment && <span className="ml-auto shrink-0 text-copper-300">{visibleNews.sentiment}</span>}
            </div>
            <p className="mt-0.5 line-clamp-1 text-[10.5px] leading-[1.25] text-slate-300">{visibleNews.summary || visibleNews.title}</p>
          </a>
        )
      ) : (
        <div
          data-testid="category-news-slot"
          className={`shrink-0 overflow-hidden border-b border-white/10 ${visibleNews ? 'h-[44px]' : context ? 'h-[24px]' : 'h-[38px]'}`}
        >
          {visibleNews ? (
            <a
              href={visibleNews.url || undefined}
              target={visibleNews.url ? '_blank' : undefined}
              rel="noopener noreferrer"
              className="block h-full bg-white/[0.025] px-2.5 py-1 hover:bg-white/[0.045]"
            >
              <div className="flex items-center gap-1.5 text-[9px] leading-none text-slate-500">
                <span>{visibleNews.publishedAt ? new Date(visibleNews.publishedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : 'Recent'}</span>
                {visibleNews.publisher && <span className="truncate text-slate-600">· {visibleNews.publisher}</span>}
                {visibleNews.sentiment && <span className="ml-auto shrink-0 text-copper-300">{visibleNews.sentiment}</span>}
              </div>
              <p className="mt-0.5 line-clamp-2 text-[10.5px] leading-[1.25] text-slate-300">{visibleNews.summary || visibleNews.title}</p>
            </a>
          ) : context ? (
            <div className="flex h-full items-center gap-1.5 bg-white/[0.018] px-2.5 text-[10px] text-slate-500" role="status">
              <span aria-hidden="true" className="text-copper-400">✧</span>
              <span className="truncate">No recent news is available for {activeLeaf ? activeLeaf.name : categoryName}.</span>
            </div>
          ) : (
            <div aria-hidden="true" className="flex h-full flex-col justify-center gap-1.5 bg-white/[0.018] px-2.5 py-1">
              <span className="h-1.5 w-20 rounded bg-white/10" />
              <span className="h-2 w-4/5 rounded bg-white/[0.06]" />
            </div>
          )}
        </div>
      )}

      {activeLeaf && (
        <div className={`grid grid-cols-[minmax(0,1fr)_64px_52px_54px] items-center gap-1.5 border-b border-white/10 bg-white/[0.045] px-2.5 ${isSheet ? 'py-1' : 'py-1.5'}`} data-testid="selected-stock">
          <div className="min-w-0">
            <strong className={`block truncate font-bold leading-none tracking-wide text-white ${isSheet ? 'text-xs' : 'text-[13px]'}`}>{activeLeaf.name}</strong>
            <span className="mt-0.5 block truncate text-[9px] leading-none text-slate-400" title={activeLeaf.shortName || activeLeaf.instrumentType || 'Instrument'}>{activeLeaf.shortName || activeLeaf.instrumentType || 'Instrument'}</span>
          </div>
          <Sparkline values={activeLeaf.sparkline} positive={(activeLeaf.changePercent || 0) >= 0} width={64} height={isSheet ? 16 : 20} />
          <div className="text-right font-mono text-xs tabular-nums text-slate-100">{activeLeaf.price == null ? '—' : activeLeaf.price.toFixed(activeLeaf.price < 10 ? 3 : 2)}</div>
          <div className={`text-right font-mono text-xs font-bold tabular-nums ${(activeLeaf.changePercent || 0) > 0 ? 'text-emerald-400' : (activeLeaf.changePercent || 0) < 0 ? 'text-rose-400' : 'text-slate-400'}`}>
            {(activeLeaf.changePercent || 0) > 0 ? '+' : ''}{(activeLeaf.changePercent || 0).toFixed(2)}%
          </div>
        </div>
      )}

      <div
        ref={listRef}
        className="custom-scrollbar relative overflow-y-auto overscroll-contain"
        data-testid="peer-list"
        style={{ height: listHeight, minHeight: Math.min(rowHeight, listHeight) }}
        onScroll={(event) => {
          const next = event.currentTarget.scrollTop;
          if (scrollFrame.current !== null) cancelAnimationFrame(scrollFrame.current);
          scrollFrame.current = requestAnimationFrame(() => setScrollTop(next));
        }}
      >
        {peers.length === 0 ? (
          <p className="px-2.5 py-2 text-[10px] text-slate-500">No additional peers in this category.</p>
        ) : (
          <div className="relative" style={{ height: virtualized ? peers.length * rowHeight : 'auto' }}>
            {visible.map((item, index) => (
              <PeerRow
                key={item.id || item.name}
                item={item}
                rowIndex={start + index}
                rowHeight={rowHeight}
                virtualized={virtualized}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}));

export default HeatmapCategoryPanel;
