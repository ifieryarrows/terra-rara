import { useEffect, useState, useCallback, useMemo, useRef, useContext, Suspense, lazy, startTransition, type Dispatch, type SetStateAction } from 'react';
import { QueryClientContext } from '@tanstack/react-query';
import { FinancialPanel as GlassCard } from '../components/ui/FinancialPanel';
import { PriceForecastChart } from '../features/forecast/PriceForecastChart';
import { RefreshButton } from '../components/ui/RefreshButton';
import { ModelReliability } from '../features/forecast/ModelReliability';
import { MetricTile } from '../components/ui/MetricTile';
import { SegmentedTabs, type SegmentedTabOption } from '../components/ui/SegmentedTabs';
import { ViewState } from '../components/ui/ViewState';
import { MapSkeleton, OverviewSkeleton } from '../components/skeletons/OverviewSkeleton';
import { BrandMark } from '../components/ui/BrandMark';
import {
  Activity, AlertTriangle, BarChart3, Brain, Compass, Cpu,
  Crosshair, Globe, Newspaper,
} from 'lucide-react';
import clsx from 'clsx';
import { formatQuoteDelta, quoteComparison } from '../utils/quote';
import { driverCategoryLabel, driverCategoryTone } from '../utils/driverCategories';
import { useCuprumFeed } from '../contexts/CuprumContext';
import { playObservatoryChime } from '../utils/cosmicVision';

import {
  fetchAnalysis,
  fetchHistory,
  fetchCommentary,
  fetchTFTAnalysis,
  fetchExplain,
  fetchLivePrice,
} from '../api';
import { COPPER_INSTRUMENT, DEFAULT_COPPER_SYMBOL } from '../config/instruments';
import type {
  AnalysisReport, HistoryResponse, Influencer,
  CommentaryResponse, TFTAnalysisResponse, ExplainResponse
} from '../types';
import {
  useSentimentSummary,
  useTftModelSummary,
} from '../hooks/useQueries';
import '../App.css';

// Lazy load heavy intelligence panels so static dashboardJS stays compact
const HeatmapPanel = lazy(() => import('../features/heatmap/HeatmapPanel').then(m => ({ default: m.HeatmapPanel })));
const NewsIntelligencePanel = lazy(() =>
  import('../features/news/NewsIntelligencePanel').then(m => ({ default: m.NewsIntelligencePanel })),
);
const ModelFocusPanel = lazy(() =>
  import('../components/ModelFocusPanel').then(m => ({ default: m.ModelFocusPanel })),
);
const LongTermOutlookPanel = lazy(() =>
  import('../features/forecast/LongTermOutlookPanel').then(m => ({ default: m.ConnectedLongTermOutlookPanel })),
);
const VenusParticles = lazy(() =>
  import('../components/ui/VenusParticles').then(m => ({ default: m.VenusParticles })),
);

type OverviewStreamLens = 'all' | 'forecast' | 'long-term' | 'xai' | 'news' | 'map';

const STREAM_LENS_OPTIONS: readonly SegmentedTabOption<OverviewStreamLens>[] = [
  { value: 'all', label: 'All Streams', icon: Compass },
  { value: 'forecast', label: '5D & T+1 Models', icon: Activity },
  { value: 'long-term', label: '1M / 1Q Outlook', icon: Compass },
  { value: 'xai', label: 'XAI Focus & Drivers', icon: Brain },
  { value: 'news', label: 'Copper News', icon: Newspaper },
  { value: 'map', label: 'Metals Map', icon: Globe },
];

const DRIVER_CATEGORY_PALETTE: Record<string, string> = {
  Momentum: '#dda174',
  Trend: '#8eadde',
  Volatility: '#e5b98e',
  Sentiment: '#a5b8e3',
  Macro: '#74c7a9',
  Sector: '#db8296',
  Embedding: '#a3afbf',
  Other: '#6f7f96',
};

function formatSignedPct(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '—';
  const pct = value * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(digits)}%`;
}

function formatDriverLabel(inf: Influencer): string {
  const rawLabel = (inf.label || inf.description || inf.feature || '').trim();
  if (/^(\d+-day\s+(?:price\s+)?(?:change|return)|1\s*day\s*price)/i.test(rawLabel)) {
    return `Copper · ${rawLabel}`;
  }
  const smaMatch = rawLabel.match(/^SMA\s*[·-]\s*(\d+)$/i);
  if (smaMatch) {
    return `Copper · ${smaMatch[1] ?? '20'}-day moving average`;
  }
  return rawLabel;
}

// --- Main App ---

function settle<T>(
  res: PromiseSettledResult<T | null>,
  setter: Dispatch<SetStateAction<T | null>>,
  label: string,
  silent: boolean,
  errs: Record<string, string>,
) {
  if (res.status === 'fulfilled') setter((prev) => (isSamePayload(prev, res.value) ? prev : res.value));
  else {
    if (!silent) setter(null);
    errs[label] = String(res.reason);
  }
}

function dirGlyph(v?: string | null): [string, string] {
  return v === 'BULLISH' ? ['▲', 'text-[#74c7a9]'] : v === 'BEARISH' ? ['▼', 'text-[#db8296]'] : ['■', 'text-[#dda174]'];
}

// O(1) structural/timestamp/boundary equality to avoid JSON.stringify on polls.
function isSamePayload(previous: unknown, next: unknown): boolean {
  if (previous === next) return true;
  if (!previous || !next || typeof previous !== 'object' || typeof next !== 'object') return false;
  const a = previous as Record<string, any>;
  const b = next as Record<string, any>;
  if (
    a.generated_at !== b.generated_at ||
    a.reference_price_date !== b.reference_price_date ||
    a.current_price !== b.current_price ||
    a.predicted_return !== b.predicted_return ||
    a.primary_forecast_return !== b.primary_forecast_return ||
    a.quality_state !== b.quality_state ||
    a.model_state !== b.model_state ||
    a.notes !== b.notes
  ) return false;
  const al = Array.isArray(a.data) ? a.data : null;
  const bl = Array.isArray(b.data) ? b.data : null;
  if (al?.length !== bl?.length || (al?.length && bl && (al[0]?.date !== bl[0]?.date || al[al.length - 1]?.date !== bl[bl.length - 1]?.date || al[al.length - 1]?.price !== bl[bl.length - 1]?.price))) return false;
  if (a.top_influencers?.length !== b.top_influencers?.length || a.top_influencers?.[0]?.feature !== b.top_influencers?.[0]?.feature || a.top_influencers?.[0]?.importance !== b.top_influencers?.[0]?.importance) return false;
  if (a.prediction?.weekly_return !== b.prediction?.weekly_return || a.prediction?.daily_forecasts?.length !== b.prediction?.daily_forecasts?.length) return false;
  return a.attention?.series?.length === b.attention?.series?.length && a.variable_selection?.encoder?.length === b.variable_selection?.encoder?.length;
}

export type DeferredPanelKey = 'companion' | 'news' | 'map';
export type DeferredPanelsState = Record<DeferredPanelKey, boolean>;
export type DeferredPanelsEvent =
  | { type: 'reveal'; key: DeferredPanelKey }
  | { type: 'reveal-all' };

export const INITIAL_DEFERRED_PANELS_STATE: DeferredPanelsState = {
  companion: false,
  news: false,
  map: false,
};

export function advanceDeferredPanels(
  state: DeferredPanelsState,
  event: DeferredPanelsEvent,
): DeferredPanelsState {
  switch (event.type) {
    case 'reveal':
      if (state[event.key]) return state;
      return { ...state, [event.key]: true };
    case 'reveal-all':
      if (state.companion && state.news && state.map) return state;
      return { companion: true, news: true, map: true };
  }
}

export const OverviewPage = () => {
  const queryClient = useContext(QueryClientContext);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [streamLens, setStreamLens] = useState<OverviewStreamLens>('all');
  const [selectedDriverCategory, setSelectedDriverCategory] = useState<string>('ALL');
  const coreInFlight = useRef(false);
  const enrichInFlight = useRef(false);
  const busyCount = useRef(0);
  // Below-fold panels mount as the reader approaches them (or on the first Tab),
  // keeping entry work proportional to what is actually on screen.
  const [deferredPanels, setDeferredPanels] = useState<DeferredPanelsState>(INITIAL_DEFERRED_PANELS_STATE);
  const [commentaryLoading, setCommentaryLoading] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisReport | null>(null);
  const [tftAnalysis, setTftAnalysis] = useState<TFTAnalysisResponse | null>(null);
  const [explain, setExplain] = useState<ExplainResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [commentary, setCommentary] = useState<CommentaryResponse | null>(null);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [loadErrors, setLoadErrors] = useState<Record<string, string>>({});
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [lastLiveUpdateAt, setLastLiveUpdateAt] = useState<Date | null>(null);
  const sentimentSummary = useSentimentSummary(7, 6);
  const tftSummary = useTftModelSummary(DEFAULT_COPPER_SYMBOL);

  const setBusy = useCallback((delta: 1 | -1) => {
    busyCount.current = Math.max(0, busyCount.current + delta);
    setIsRefreshing(busyCount.current > 0);
  }, []);

  // Core snapshot (analysis + history) gates the skeleton and dashboard-ready.
  const loadCore = useCallback(async (silent = false) => {
    if (coreInFlight.current) return;
    coreInFlight.current = true;
    setBusy(1);
    try {
      const [a, h] = await Promise.allSettled([
        fetchAnalysis(DEFAULT_COPPER_SYMBOL),
        fetchHistory(DEFAULT_COPPER_SYMBOL, 180),
      ]);
      const errs: Record<string, string> = {};
      settle(a, setAnalysis, 'Price forecast', silent, errs);
      settle(h, setHistory, 'Price history', silent, errs);
      setLoadErrors(({ 'Price forecast': _a, 'Price history': _b, ...rest }) => ({ ...rest, ...errs }));
      setIsInitialLoad(false);
    } finally {
      setBusy(-1);
      coreInFlight.current = false;
    }
  }, [setBusy]);

  // Weekly-model enrichment (TFT + model focus) hydrates in the background so a
  // slow live-inference fallback can never hold the chart hostage.
  const loadEnrichment = useCallback(async (silent = false) => {
    if (enrichInFlight.current) return;
    enrichInFlight.current = true;
    setBusy(1);
    try {
      const [t, e] = await Promise.allSettled([
        fetchTFTAnalysis(DEFAULT_COPPER_SYMBOL),
        fetchExplain(DEFAULT_COPPER_SYMBOL),
      ]);
      const errs: Record<string, string> = {};
      settle(t, setTftAnalysis, 'Deep-learning forecast', silent, errs);
      settle(e, setExplain, 'Model focus', silent, errs);
      setLoadErrors(({ 'Deep-learning forecast': _t, 'Model focus': _e, ...rest }) => ({ ...rest, ...errs }));
    } finally {
      setBusy(-1);
      enrichInFlight.current = false;
    }
  }, [setBusy]);

  const loadCommentary = useCallback(async () => {
    setCommentaryLoading(true);
    try {
      const data = await fetchCommentary(DEFAULT_COPPER_SYMBOL);
      setCommentary(data);
      setLoadErrors((previous) => {
        const next = { ...previous };
        delete next['AI commentary'];
        return next;
      });
    } catch (err) {
      setCommentary(null);
      setLoadErrors((previous) => ({ ...previous, 'AI commentary': String(err) }));
    } finally {
      setCommentaryLoading(false);
    }
  }, []);

  // Initial load: the core snapshot gates the skeleton; enrichment hydrates in the background.
  useEffect(() => {
    void loadCore(false);
    void loadEnrichment(false);
  }, [loadCore, loadEnrichment]);

  // Keep the section rail in sync with the part of the workspace in view.
  // Silent refresh: core every minute; TFT/explain on the weekly-model cadence.
  useEffect(() => {
    const coreInterval = window.setInterval(() => {
      if (!document.hidden) void loadCore(true);
    }, 60000);
    const enrichmentInterval = window.setInterval(() => {
      if (!document.hidden) void loadEnrichment(true);
    }, 300000);
    return () => {
      window.clearInterval(coreInterval);
      window.clearInterval(enrichmentInterval);
    };
  }, [loadCore, loadEnrichment]);

  // Mount the below-fold panels once their section approaches the viewport.
  // Re-armed when the entry skeleton is replaced by the real layout.
  useEffect(() => {
    const panels: Array<[DeferredPanelKey, string]> = [
      ['companion', 'neural-analysis'],
      ['news', 'news-intelligence'],
      ['map', 'market-map'],
    ];
    const revealAll = () => startTransition(() => {
      setDeferredPanels((prev) => advanceDeferredPanels(prev, { type: 'reveal-all' }));
    });
    if (typeof IntersectionObserver === 'undefined') {
      revealAll();
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      startTransition(() => {
        setDeferredPanels((previous) => {
          let next = previous;
          for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            const key = panels.find(([, id]) => id === (entry.target as HTMLElement).id)?.[0];
            if (!key) continue;
            next = advanceDeferredPanels(next, { type: 'reveal', key });
            observer.unobserve(entry.target);
          }
          return next;
        });
      });
    }, { rootMargin: '600px 0px' });
    for (const [, id] of panels) {
      const node = document.getElementById(id);
      if (node) observer.observe(node);
    }
    // Keyboard users tab through the document without scrolling; mount everything
    // on the first Tab so focus never reaches an empty section.
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return;
      document.removeEventListener('keydown', onKeyDown);
      revealAll();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      observer.disconnect();
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [isInitialLoad]);

  // Live price snapshot polling (TradingView/websocket removed).
  useEffect(() => {
    const fetchSnapshot = async () => {
      try {
        const payload = await fetchLivePrice();
        if (typeof payload.price === 'number' && Number.isFinite(payload.price)) {
          setLivePrice(payload.price);
          setLastLiveUpdateAt(new Date());
        }
      } catch {
        // Best-effort polling.
      }
    };

    void fetchSnapshot();
    const id = window.setInterval(() => void fetchSnapshot(), 120000);
    return () => {
      window.clearInterval(id);
    };
  }, []);

  // Load commentary after analysis
  useEffect(() => {
    if (analysis) loadCommentary();
  }, [analysis, loadCommentary]);

  const latestHistoryPrice = useMemo(() => {
    const rows = history?.data;
    if (!rows?.length) return null;
    for (let index = rows.length - 1; index >= 0; index -= 1) {
      if (rows[index].price != null) return rows[index].price;
    }
    return null;
  }, [history]);

  const quotePrice = livePrice ?? latestHistoryPrice;

  const sessionRange = useMemo(() => {
    if (!history?.data || history.data.length === 0) return null;
    const recent = history.data.slice(-5);
    const prices = recent.map((p) => p.price).filter((p): p is number => typeof p === 'number' && Number.isFinite(p));
    if (prices.length === 0) return null;
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    return { min, max };
  }, [history?.data]);

  const quotePercentInRange = useMemo(() => {
    if (!quotePrice || !sessionRange || sessionRange.max <= sessionRange.min) return 50;
    const pct = ((quotePrice - sessionRange.min) / (sessionRange.max - sessionRange.min)) * 100;
    return Math.min(94, Math.max(6, pct));
  }, [quotePrice, sessionRange]);

  const topDrivers = useMemo(() => (analysis?.top_influencers ?? []).slice(0, 5), [analysis?.top_influencers]);

  const driverBreakdown = useMemo(() => {
    if (!topDrivers.length) return [];
    const grouped = new Map<string, number>();
    let total = 0;
    for (const inf of topDrivers) {
      const cat = inf.category || 'Other';
      const val = Math.max(0, inf.importance);
      grouped.set(cat, (grouped.get(cat) ?? 0) + val);
      total += val;
    }
    if (total <= 0) return [];
    return Array.from(grouped.entries())
      .map(([category, weight]) => ({
        category,
        label: driverCategoryLabel[category] || category,
        sharePct: (weight / total) * 100,
        color: DRIVER_CATEGORY_PALETTE[category] || DRIVER_CATEGORY_PALETTE.Other,
      }))
      .sort((a, b) => b.sharePct - a.sharePct);
  }, [topDrivers]);

  const visibleDrivers = useMemo(() => {
    if (selectedDriverCategory === 'ALL') return topDrivers;
    return topDrivers.filter((inf) => (inf.category || 'Other') === selectedDriverCategory);
  }, [topDrivers, selectedDriverCategory]);

  const focusFallback = (
    <div className="cm-commentary py-8 text-center text-xs text-slate-400" role="status">
      Loading 5D model focus…
    </div>
  );
  const longTermFallback = (
    <div className="cm-commentary py-8 text-center text-xs text-slate-400" role="status">
      Loading 1M / 1Q long-term outlook…
    </div>
  );
  const newsFallback = (
    <div className="cm-news-loading flex items-center justify-center gap-3" role="status">
      <span className="cm-inline-loader" aria-hidden="true" />
      <span className="text-xs text-slate-400 tracking-widest uppercase">
        Loading news…
      </span>
    </div>
  );

  const tftReturn = tftAnalysis?.primary_forecast_return
    ?? tftAnalysis?.weekly_forecast?.expected_return
    ?? tftAnalysis?.prediction?.weekly_return
    ?? null;
  const tftWeeklyQ10 = tftAnalysis?.primary_forecast_q10
    ?? tftAnalysis?.weekly_forecast?.q10_return
    ?? tftAnalysis?.prediction?.weekly_return_q10_calibrated
    ?? null;
  const tftWeeklyQ90 = tftAnalysis?.primary_forecast_q90
    ?? tftAnalysis?.weekly_forecast?.q90_return
    ?? tftAnalysis?.prediction?.weekly_return_q90_calibrated
    ?? null;
  const tftCalibrated = tftAnalysis?.weekly_forecast?.calibrated
    ?? tftAnalysis?.prediction?.weekly_interval_calibrated
    ?? false;
  const tftDegraded = !!tftAnalysis && (
    tftAnalysis.quality_state === 'degraded' ||
    tftAnalysis.model_state === 'retrain_required' ||
    tftAnalysis.is_forecast_healthy === false
  );
  const tftDegradedMessage = tftAnalysis?.message || 'TFT weekly forecast is unavailable until the weekly model artifacts are refreshed.';
  const tftBullish = tftReturn !== null ? tftReturn >= 0 : null;
  const tftMetrics = tftAnalysis?.model_metadata?.metrics;
  const mergedTftMetrics = useMemo(() => {
    if (!tftMetrics && !tftSummary.data?.metrics) return undefined;
    return {
      ...(tftMetrics ?? {}),
      ...(tftSummary.data?.metrics ?? {}),
    };
  }, [tftMetrics, tftSummary.data?.metrics]);
  const tftQualityGate = tftSummary.data?.quality_gate ?? tftAnalysis?.model_metadata?.quality_gate ?? null;
  const tftDirection = tftAnalysis?.direction;
  const tftImpulse = tftAnalysis?.t1_impulse ?? tftAnalysis?.weekly_forecast?.t1_impulse ?? 'NEUTRAL';
  const tftReferencePrice = tftAnalysis?.prediction?.reference_price;
  const tftReferenceDate = tftAnalysis?.prediction?.reference_price_date;
  const tftWeeklyPrice = tftAnalysis?.prediction?.weekly_price;
  const tftAnomaly = tftAnalysis?.prediction?.anomaly_detected;
  const tftInstrument = tftAnalysis?.prediction?.instrument;
  const tftStalenessDays = tftAnalysis?.prediction?.baseline_staleness_days ?? 0;
  // Anything >= 3 calendar days is flagged; 0-2 is considered fresh (weekend).
  const tftBaselineIsStale = tftStalenessDays >= 3;
  const tftWeeklyDa = tftMetrics?.weekly_directional_accuracy;
  const tftRiskLevel = tftAnalysis?.risk_level;
  const tftRiskPips = tftRiskLevel === 'HIGH' ? 3 : tftRiskLevel === 'MEDIUM' ? 2 : tftRiskLevel === 'LOW' ? 1 : 0;

  const newsSentimentIndex = sentimentSummary.data?.index;
  const rawSentimentLabel = sentimentSummary.data?.label?.trim().toUpperCase();
  const newsSampleCount = sentimentSummary.data?.data_freshness?.article_count_window
    ?? sentimentSummary.data?.components?.sample_size
    ?? null;
  const clampedSentiment =
    typeof newsSentimentIndex === 'number' && Number.isFinite(newsSentimentIndex)
      ? Math.max(-1, Math.min(1, newsSentimentIndex))
      : null;
  const sentimentStrength =
    clampedSentiment == null
      ? (sentimentSummary.isLoading ? 'LOADING' : 'OFFLINE')
      : Math.abs(clampedSentiment) >= 0.45
        ? 'STRONG'
        : Math.abs(clampedSentiment) >= 0.20
          ? 'MODERATE'
          : Math.abs(clampedSentiment) >= 0.06
            ? 'MILD'
            : 'BALANCED';
  const sentimentPosPct = clampedSentiment != null ? ((clampedSentiment + 1) / 2) * 100 : 50;
  const sentimentDir =
    rawSentimentLabel === 'BULLISH' || rawSentimentLabel === 'POSITIVE' || (!rawSentimentLabel && clampedSentiment != null && clampedSentiment > 0.1)
      ? 'Bullish'
      : rawSentimentLabel === 'BEARISH' || rawSentimentLabel === 'NEGATIVE' || (!rawSentimentLabel && clampedSentiment != null && clampedSentiment < -0.1)
        ? 'Bearish'
        : 'Neutral';
  const newsSentimentMeta =
    sentimentDir === 'Bullish'
      ? { glyph: '▲', text: 'Bullish', tone: 'good' as const }
      : sentimentDir === 'Bearish'
        ? { glyph: '▼', text: 'Bearish', tone: 'bad' as const }
        : { glyph: '■', text: 'Balanced', tone: 'neutral' as const };
  const quoteChange = quoteComparison(livePrice, latestHistoryPrice);
  const rawRangePct = quotePrice != null && sessionRange && sessionRange.max > sessionRange.min
    ? Math.round(Math.max(0, Math.min(100, ((quotePrice - sessionRange.min) / (sessionRange.max - sessionRange.min)) * 100)))
    : null;

  const qMedianMarkerPct =
    tftWeeklyQ10 != null && tftWeeklyQ90 != null && tftWeeklyQ90 > tftWeeklyQ10 && tftReturn != null
      ? Math.max(4, Math.min(96, ((tftReturn - tftWeeklyQ10) / (tftWeeklyQ90 - tftWeeklyQ10)) * 100))
      : 50;
  const qZeroMarkerPct =
    tftWeeklyQ10 != null && tftWeeklyQ90 != null && tftWeeklyQ10 < 0 && tftWeeklyQ90 > 0
      ? Math.max(3, Math.min(97, ((0 - tftWeeklyQ10) / (tftWeeklyQ90 - tftWeeklyQ10)) * 100))
      : null;

  const cuprumTelemetry = useMemo(
    () => ({
      activeSymbol: COPPER_INSTRUMENT.canonicalSymbol,
      trendPct: quoteChange?.percent ?? (tftReturn != null ? tftReturn * 100 : null),
      conviction: tftRiskLevel ?? null,
      confidenceScore: tftWeeklyDa ?? null,
      sentimentScore: newsSentimentIndex ?? null,
      regime: tftDegraded ? 'degraded' : tftDirection ?? null,
      isStale: tftBaselineIsStale,
      hasError: Object.keys(loadErrors).length > 0,
      quote: {
        symbol: COPPER_INSTRUMENT.canonicalSymbol,
        price: quotePrice,
        previous_close: latestHistoryPrice,
        change: quoteChange?.delta ?? null,
        change_pct: quoteChange?.percent ?? null,
        currency: 'USD / lb',
        as_of: tftReferenceDate ?? null,
        fetched_at: lastLiveUpdateAt?.toISOString() ?? null,
        market_status: livePrice != null ? 'LIVE' : 'SETTLED',
      },
      forecast: tftAnalysis,
      sentiment: sentimentSummary.data ?? null,
      commentary,
      explain,
      tftAnalysis,
      loading: commentaryLoading || isInitialLoad,
      error: loadErrors['AI commentary'] || loadErrors['Model focus'] || null,
    }),
    [
      commentary,
      commentaryLoading,
      explain,
      isInitialLoad,
      lastLiveUpdateAt,
      latestHistoryPrice,
      livePrice,
      loadErrors,
      newsSentimentIndex,
      quoteChange?.delta,
      quoteChange?.percent,
      quotePrice,
      sentimentSummary.data,
      tftAnalysis,
      tftBaselineIsStale,
      tftDegraded,
      tftDirection,
      tftReferenceDate,
      tftReturn,
      tftRiskLevel,
      tftWeeklyDa,
    ],
  );
  useCuprumFeed(cuprumTelemetry);

  // Only show full loading on initial load
  if (isInitialLoad && !analysis) {
    return <OverviewSkeleton />;
  }

  const handleStreamLensChange = (nextLens: OverviewStreamLens) => {
    void playObservatoryChime({ volume: 0.3 });
    startTransition(() => {
      setStreamLens(nextLens);
      if (nextLens !== 'all') {
        setDeferredPanels((prev) => advanceDeferredPanels(prev, { type: 'reveal-all' }));
      }
    });
  };

  const handleManualRefresh = () => {
    void playObservatoryChime({ volume: 0.35 });
    void loadCore(true);
    void loadEnrichment(true);
    if (queryClient) {
      void queryClient.invalidateQueries();
    }
  };

  const showForecast = streamLens === 'all' || streamLens === 'forecast';
  const showLongTerm = streamLens === 'all' || streamLens === 'forecast' || streamLens === 'long-term';
  const showXai = streamLens === 'all' || streamLens === 'xai';
  const showNews = streamLens === 'all' || streamLens === 'news';
  const showMap = streamLens === 'all' || streamLens === 'map';

  const spotTone: 'good' | 'bad' | 'copper' =
    quoteChange?.tone === 'positive'
      ? 'good'
      : quoteChange?.tone === 'negative'
        ? 'bad'
        : 'copper';
  const weeklyTone: 'good' | 'bad' | 'forecast' =
    tftDegraded || tftBullish == null ? 'forecast' : tftBullish ? 'good' : 'bad';
  const t1Tone: 'good' | 'bad' | 'neutral' =
    tftImpulse === 'BULLISH' ? 'good' : tftImpulse === 'BEARISH' ? 'bad' : 'neutral';

  return (
    <div className="cm-market-dashboard font-sans selection:bg-copper-500/30" data-cm-dashboard-ready={isInitialLoad ? 'false' : 'true'}>
      <div className="cm-dashboard-page relative z-10 grid min-w-0 grid-cols-1">

        <header className="cm-dashboard-hero">
          <Suspense fallback={null}>
            <VenusParticles />
          </Suspense>
          <div className="cm-dashboard-hero-copy">
            <p className="cm-eyebrow cm-dashboard-hero-eyebrow" data-cm-route-reveal="copy">
              <BrandMark size={20} variant="small" />
              <span>COPPER INTELLIGENCE // TERRA RARA</span>
              <span className="cm-telemetry-dot-live" aria-hidden="true" />
            </p>
            <h1 className="cm-dashboard-title" data-cm-route-reveal="copy">
              COMEX <span>COPPER</span>
            </h1>
            <p className="cm-dashboard-hero-subtitle" data-cm-route-reveal="copy">
              High Grade Copper Futures · 5D TFT-ASRO Quantiles &amp; Real-Time Price Action.
            </p>
          </div>

          <div className="cm-dashboard-hero-data" data-cm-route-reveal="surface" role="group" aria-label="Copper market snapshot">
            <div className="cm-dashboard-instrument">
              <div className="cm-instrument-meta">
                <p className="cm-dashboard-instrument-name">{COPPER_INSTRUMENT.displayName} · COMEX</p>
              </div>
              <span className={clsx('cm-dashboard-quote-status', livePrice != null ? 'cm-dashboard-quote-status--live' : 'cm-dashboard-quote-status--delayed')}>
                <i aria-hidden="true" className="cm-beacon-dot" />
                {livePrice != null ? 'LIVE COMEX' : latestHistoryPrice != null ? 'SETTLED CLOSE' : 'Awaiting quote'}
              </span>
            </div>

            <div className="cm-dashboard-price-row">
              <span className="cm-dashboard-price">{quotePrice != null ? quotePrice.toFixed(4) : '--'}</span>
              <span className="cm-dashboard-currency">USD / lb</span>
              <span
                aria-hidden={!quoteChange}
                className={clsx(
                  'cm-dashboard-change',
                  quoteChange
                    ? quoteChange.tone === 'neutral' ? 'text-slate-400' : quoteChange.tone === 'positive' ? 'text-[#74c7a9]' : 'text-[#db8296]'
                    : 'cm-dashboard-change--placeholder'
                )}
              >
                {quoteChange ? (
                  <span className="cm-delta-pill inline-flex items-center gap-1 font-medium text-sm tabular-nums">
                    {quoteChange.tone === 'positive' ? '▲' : quoteChange.tone === 'negative' ? '▼' : '■'} {formatQuoteDelta(quoteChange.delta)} <span>({formatQuoteDelta(quoteChange.percent)}%)</span>
                  </span>
                ) : null}
              </span>
            </div>

            <div className="cm-session-range-strip" aria-label="Session trading range">
              <div className="cm-session-range-labels">
                <div className="cm-session-range-bound">
                  <span className="cm-session-range-tag">5D LOW</span>
                  <span className="cm-session-range-val">{sessionRange?.min != null ? `$${sessionRange.min.toFixed(4)}` : quotePrice != null ? `$${(quotePrice * 0.985).toFixed(4)}` : '--'}</span>
                </div>
                <div className="cm-session-range-prev">
                  <span className="text-slate-400 text-xs">PREV CLOSE</span>
                  <strong className="text-slate-200">{latestHistoryPrice != null ? `$${latestHistoryPrice.toFixed(4)}` : '—'}</strong>
                  {rawRangePct != null && (
                    <span className="cm-range-percentile">{rawRangePct}% POS</span>
                  )}
                </div>
                <div className="cm-session-range-bound cm-session-range-bound--high">
                  <span className="cm-session-range-tag">5D HIGH</span>
                  <span className="cm-session-range-val">{sessionRange?.max != null ? `$${sessionRange.max.toFixed(4)}` : quotePrice != null ? `$${(quotePrice * 1.018).toFixed(4)}` : '--'}</span>
                </div>
              </div>
              <div className="cm-session-range-track" aria-hidden="true">
                <div className="cm-session-range-gradient" />
                <div className="cm-session-range-dot" style={{ left: `${quotePercentInRange}%` }} title="Quote in 5D range" />
              </div>
            </div>

            <div className="cm-dashboard-quote-meta">
              <span>{lastLiveUpdateAt ? `Checked ${lastLiveUpdateAt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })} ${lastLiveUpdateAt.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })}` : latestHistoryPrice != null ? 'Showing the last available historical close' : 'Waiting for latest quote'}</span>
              <span aria-hidden={!quoteChange} className={quoteChange ? undefined : 'cm-dashboard-meta--placeholder'}>{quoteChange ? 'Change vs last stored close' : null}</span>
            </div>
          </div>
        </header>

        {/* Open Horizon Telemetry Ribbon: strictly separates 5D Primary vs T+1 Diagnostic */}
        <section className="cm-telemetry-ribbon" aria-label="Copper horizon and pipeline telemetry" data-cm-route-reveal="surface">
          <MetricTile
            label="HG=F Spot"
            value={quotePrice != null ? `$${quotePrice.toFixed(4)}` : '--'}
            delta={quoteChange ? `${quoteChange.tone === 'positive' ? '▲' : quoteChange.tone === 'negative' ? '▼' : '■'} ${formatQuoteDelta(quoteChange.percent)}%` : undefined}
            tone={spotTone}
            badge={livePrice != null ? 'LIVE' : 'CLOSE'}
            hint={sessionRange ? `5D $${sessionRange.min.toFixed(2)}–$${sessionRange.max.toFixed(2)}` : 'COMEX Futures'}
          />
          <MetricTile
            label="5D TFT Primary"
            value={tftDegraded ? 'Degraded' : formatSignedPct(tftReturn)}
            delta={tftWeeklyPrice != null && !tftDegraded ? `${tftBullish ? '▲' : '▼'} $${tftWeeklyPrice.toFixed(2)}` : undefined}
            tone={weeklyTone}
            badge="5D"
            hint={
              tftDegraded
                ? 'Awaiting weekly refresh'
                : tftWeeklyQ10 != null && tftWeeklyQ90 != null
                  ? `Q10–Q90 ${formatSignedPct(tftWeeklyQ10, 1)}..${formatSignedPct(tftWeeklyQ90, 1)}`
                  : '5-day cumulative quantile cone'
            }
          />
          <MetricTile
            label="T+1 TFT Step 1"
            value={tftImpulse}
            delta={tftAnalysis?.t1_return != null ? `${dirGlyph(tftImpulse)[0]} ${formatSignedPct(tftAnalysis.t1_return)}` : undefined}
            tone={t1Tone}
            badge="T+1"
            hint={
              analysis?.predicted_return != null
                ? `TFT Step-1 · XGB T+1 ${formatSignedPct(analysis.predicted_return)}`
                : tftMetrics?.directional_accuracy != null
                  ? `Daily DA ${(tftMetrics.directional_accuracy * 100).toFixed(1)}%`
                  : 'Step-1 quantile median'
            }
          />
          <MetricTile
            label="7D News Sentiment"
            value={
              <span className="inline-flex items-center gap-1.5" data-testid="news-sentiment-readout">
                <span aria-hidden="true">{clampedSentiment == null ? '■' : newsSentimentMeta.glyph}</span>
                <span>{clampedSentiment == null ? '—' : newsSentimentMeta.text}</span>
              </span>
            }
            delta={clampedSentiment != null ? `${clampedSentiment >= 0 ? '+' : ''}${clampedSentiment.toFixed(2)} / ±1` : undefined}
            badge={sentimentStrength}
            tone={newsSentimentMeta.tone}
            hint={
              <div className="cm-sentiment-meter">
                <meter
                  className="sr-only"
                  min={-1}
                  max={1}
                  low={-0.15}
                  high={0.15}
                  optimum={0.5}
                  value={clampedSentiment ?? 0}
                  aria-label="7D News Sentiment scale from -1 bearish to +1 bullish"
                />
                <div className="cm-sentiment-meter__track" aria-hidden="true">
                  <span className="cm-sentiment-meter__zero" />
                  {clampedSentiment != null && (
                    <>
                      <span
                        className="cm-sentiment-meter__fill"
                        style={{
                          left: clampedSentiment < 0 ? `${sentimentPosPct}%` : '50%',
                          width: `${Math.abs(sentimentPosPct - 50)}%`,
                          color: clampedSentiment < 0 ? 'var(--cm-negative)' : 'var(--cm-positive)',
                        }}
                      />
                      <span
                        className="cm-sentiment-meter__pin"
                        style={{
                          left: `${sentimentPosPct}%`,
                          color: sentimentDir === 'Bullish' ? 'var(--cm-positive)' : sentimentDir === 'Bearish' ? 'var(--cm-negative)' : 'var(--cm-muted)',
                        }}
                      />
                    </>
                  )}
                </div>
                <div className="cm-sentiment-meter__scale">
                  <span>▼ -1</span>
                  <span>{newsSampleCount != null ? `${newsSampleCount} articles` : 'FinBERT tone'}</span>
                  <span>+1 ▲</span>
                </div>
              </div>
            }
          />
        </section>

        <div className="cm-overview-tools" data-cm-route-reveal="surface">
          <div className="cm-overview-tools-main">
            <SegmentedTabs
              value={streamLens}
              onChange={handleStreamLensChange}
              options={STREAM_LENS_OPTIONS}
              ariaLabel="Filter open dashboard intelligence stream"
              size="sm"
            />
          </div>
          <div className="cm-tools-sync-cluster">
            <span className="cm-tools-sync-tag">
              <i className="cm-sync-pulse" aria-hidden="true" />
              {lastLiveUpdateAt ? `SYNCED ${lastLiveUpdateAt.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })}` : 'TELEMETRY ACTIVE'}
            </span>
            <RefreshButton label="Refresh overview" busy={isRefreshing} onClick={handleManualRefresh} />
          </div>
        </div>
        {Object.keys(loadErrors).length > 0 && (
          <div className="cm-dashboard-alerts" role="status" aria-live="polite" data-cm-route-reveal="surface">
            <AlertTriangle size={15} aria-hidden="true" />
            <strong>Some data could not be refreshed</strong>
            <span>{Object.keys(loadErrors).join(' · ')} · Visible values may be from the previous response.</span>
          </div>
        )}

        <div className="cm-dashboard-content grid min-w-0 gap-4">
          <div className="cm-dashboard-primary-column grid grid-cols-12 gap-6" data-cm-route-reveal="surface">

            {showForecast && (
              <GlassCard
                id="price-forecast"
                eyebrow="01 // 5D PRIMARY TRAJECTORY & QUANTILE CONE"
                title={`Price Forecast · ${COPPER_INSTRUMENT.displayName}`}
                icon={Activity}
                colSpan={12}
                className="cm-price-action-section"
                headerExtra={
                  <span className="cm-panel-telemetry-meta">
                    <span>{tftAnalysis ? '5D TFT-ASRO Overlay' : 'Historical Closes'}</span>
                    {tftReferenceDate && <span>Ref {tftReferenceDate}</span>}
                  </span>
                }
              >
                <PriceForecastChart
                  history={history?.data ?? []}
                  forecast={tftAnalysis}
                  historyError={!!loadErrors['Price history']}
                  forecastError={!!loadErrors['Deep-learning forecast']}
                />
              </GlassCard>
            )}

            {showForecast && (
              <GlassCard
                id="weekly-outlook"
                eyebrow="02 // 5D WEEKLY PRIMARY · TFT-ASRO"
                title="Weekly model outlook"
                icon={Brain}
                colSpan={8}
                className={clsx('cm-weekly-forecast-panel relative', tftBullish === null ? '' : tftBullish ? 'cm-weekly--bullish' : 'cm-weekly--bearish')}
                headerExtra={
                  tftAnalysis && !tftDegraded ? (
                    <span className="cm-panel-telemetry-meta">
                      <span>5D Cumulative</span>
                      <span>{tftCalibrated ? 'Calibrated 80% Band' : 'Raw Quantiles'}</span>
                    </span>
                  ) : null
                }
              >
                {tftDegraded ? (
                  <div className="flex flex-col justify-center h-full py-10 gap-4">
                    <div className="flex items-center gap-2 text-[#dda174]">
                      <AlertTriangle size={18} />
                      <span className="text-xs font-bold uppercase tracking-widest">Forecast Degraded</span>
                    </div>
                    <p className="text-sm text-gray-400 leading-relaxed">
                      {tftDegradedMessage}
                    </p>
                    <div className="cm-degraded-note">
                      <span className="text-xs text-gray-300">Historical prices remain available while the forecast is unavailable.</span>
                    </div>
                  </div>
                ) : tftAnalysis?.prediction ? (() => {
                  const prediction = tftAnalysis.prediction;
                  return (
                    <div className="relative z-10 space-y-5">
                      {/* Weekly direction badge + target headline */}
                      <div className="cm-weekly-headline-row">
                        <div>
                          <div className="mb-1.5 flex items-center gap-2.5 flex-wrap">
                            <span className="text-xs text-slate-400 uppercase tracking-widest">
                              Expected 5D Performance
                            </span>
                            {tftBaselineIsStale && (
                              <span
                                title={`The forecast baseline close is ${tftStalenessDays} calendar days old.`}
                                className="cm-stale-flag text-[#dda174] text-xs tracking-wider"
                              >
                                Stale {tftStalenessDays}d
                              </span>
                            )}
                          </div>
                          <div className="flex flex-wrap items-baseline gap-2.5">
                            <span className={clsx('text-4xl font-light font-mono cm-tabular tracking-tight', tftBullish == null ? 'text-slate-400' : tftBullish ? 'text-[#74c7a9]' : 'text-[#db8296]')}>
                              {tftReturn == null ? '—' : formatSignedPct(tftReturn)}
                            </span>
                            <span className="text-base text-slate-200 font-mono cm-tabular">${prediction.weekly_price?.toFixed(2) ?? '--'}</span>
                            {tftReferencePrice != null && (
                              <span className="text-xs text-slate-400 font-mono cm-tabular">
                                (from ${tftReferencePrice.toFixed(2)})
                              </span>
                            )}
                          </div>
                        </div>

                        <div className={clsx('cm-consensus-readout', dirGlyph(tftDirection)[1])}>
                          <span aria-hidden="true">{dirGlyph(tftDirection)[0]}</span>
                          <span>{tftDirection ?? 'Unavailable'}</span>
                        </div>
                      </div>

                      <div className="cm-weekly-meta-grid">
                        <div>
                          <span>Instrument</span>
                          <strong className="font-mono">{tftInstrument?.symbol || COPPER_INSTRUMENT.canonicalSymbol}</strong>
                        </div>
                        <div>
                          <span>Reference Close Date</span>
                          <strong className="font-mono cm-tabular">{tftReferenceDate ?? '--'}</strong>
                        </div>
                        <div>
                          <span>Horizon</span>
                          <strong className="font-mono">5 Trading Days</strong>
                        </div>
                      </div>

                      {tftAnomaly && (
                        <p className="cm-anomaly-callout">
                          The model flagged an unusual output. Review model validation before interpreting this forecast.
                        </p>
                      )}

                      {/* Primary weekly quantile distribution bar */}
                      <div className="cm-weekly-range">
                        <div className="cm-weekly-range-head">
                          <span>5D Return Envelope {tftCalibrated ? '(Calibrated)' : '(Raw)'}</span>
                          <span>Q10 · Q50 · Q90</span>
                        </div>
                        <div className="cm-quantile-spectrum">
                          <div className="cm-quantile-bound">
                            <small>Q10 LOW</small>
                            <span className="font-mono cm-tabular text-[#db8296]">
                              {tftWeeklyQ10 != null ? formatSignedPct(tftWeeklyQ10) : '--'}
                            </span>
                          </div>
                          <div className="cm-quantile-bar-wrap" aria-hidden="true">
                            <div className="cm-quantile-bar-track">
                              <div className="cm-quantile-bar-fill" />
                              {qZeroMarkerPct != null && (
                                <span className="cm-quantile-zero-tick" style={{ left: `${qZeroMarkerPct}%` }} title="0.00% flat baseline" />
                              )}
                              <span className="cm-quantile-median-dot" style={{ left: `${qMedianMarkerPct}%` }} />
                            </div>
                          </div>
                          <div className="cm-quantile-bound cm-quantile-bound--high">
                            <small>Q90 HIGH</small>
                            <span className="font-mono cm-tabular text-[#74c7a9]">
                              {tftWeeklyQ90 != null ? formatSignedPct(tftWeeklyQ90) : '--'}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Dispersion / volatility classification */}
                      <div className="cm-dispersion-row">
                        <div className="cm-dispersion-head">
                          <div>
                            <span className="cm-dispersion-label">Model volatility classification</span>
                            <strong className="cm-dispersion-value">{tftAnalysis.risk_level ?? 'Unavailable'}</strong>
                          </div>
                          <div className="cm-dispersion-pips" aria-hidden="true">
                            {[1, 2, 3].map((step) => (
                              <span key={step} className={clsx('cm-dispersion-pip', step <= tftRiskPips && 'is-active')} />
                            ))}
                          </div>
                        </div>
                        <p className="cm-dispersion-note">
                          Based on forecast dispersion. This is separate from model quality and is not an investment safety rating.
                        </p>
                      </div>
                    </div>
                  );
                })() : (
                  <ViewState
                    kind={loadErrors['Deep-learning forecast'] ? 'error' : 'empty'}
                    title={loadErrors['Deep-learning forecast'] ? 'Weekly forecast could not be loaded' : 'No weekly forecast available'}
                    description="Use Refresh overview to check for the latest available forecast."
                    compact
                  />
                )}
              </GlassCard>
            )}

            {/* 5D Model Health + Quality Gate Snapshot */}
            {showForecast && (
              <GlassCard
                id="model-reliability"
                eyebrow="03 // 5D QUALITY GATE & CALIBRATION"
                title="Model reliability"
                icon={Crosshair}
                colSpan={4}
                headerExtra={
                  <span
                    className={clsx(
                      'cm-panel-telemetry-meta',
                      tftDegraded || tftQualityGate?.passed === false
                        ? 'text-[#db8296]'
                        : 'text-[#74c7a9]',
                    )}
                  >
                    <span>
                      {tftDegraded
                        ? 'Attention'
                        : tftQualityGate?.passed === false
                          ? 'Gate Breach'
                          : tftQualityGate?.passed === true || tftAnalysis
                            ? 'Gate Active'
                            : 'Pending'}
                    </span>
                  </span>
                }
              >
                <ModelReliability
                  metrics={mergedTftMetrics}
                  qualityGate={tftQualityGate}
                  unavailable={!!loadErrors['Deep-learning forecast'] && !tftSummary.data}
                />
              </GlassCard>
            )}

            {/* Long-Term Outlook (1M / 20D & 1Q / 60D Structural Ensemble) */}
            {showLongTerm && (
              <GlassCard
                id="long-term-outlook"
                eyebrow="04 // 1M & 1Q STRATEGIC OUTLOOK · STRUCTURAL ENSEMBLE"
                title="Long-Term Outlook · 1M / 1Q"
                icon={Compass}
                colSpan={12}
                headerExtra={
                  <span className="cm-panel-telemetry-meta">
                    <span>20D &amp; 60D Horizons</span>
                    <span>Macro Ridge + Damped Theta</span>
                  </span>
                }
              >
                <Suspense fallback={longTermFallback}>
                  <LongTermOutlookPanel symbol={DEFAULT_COPPER_SYMBOL} />
                </Suspense>
              </GlassCard>
            )}

            {/* Raw 5D Pre-Calibration TFT-ASRO Explainability (VSN + Attention) */}
            {showXai && (
              <GlassCard
                id="neural-analysis"
                eyebrow="05 // 5D MODEL FOCUS · PRE-CALIBRATION XAI"
                title="5D Model Focus · VSN & Attention"
                icon={Cpu}
                colSpan={7}
                headerExtra={
                  <span className="cm-panel-telemetry-meta">
                    <span>5D TFT-ASRO</span>
                    <span>{explain ? 'Raw 5D Focus · Non-causal' : 'Awaiting Explainability'}</span>
                  </span>
                }
              >
                <div id="tft-focus">
                  {deferredPanels.companion || streamLens === 'xai' ? (
                    <Suspense fallback={focusFallback}>
                      <ModelFocusPanel
                        data={explain}
                        error={loadErrors['Model focus']}
                      />
                    </Suspense>
                  ) : focusFallback}
                </div>
              </GlassCard>
            )}

            {/* Auxiliary T+1 Daily Diagnostic (XGBoost) — strictly separated from 5D TFT */}
            {showXai && (
              <GlassCard
                id="market-drivers"
                eyebrow="06 // T+1 DAILY DIAGNOSTIC · XGBOOST"
                title="T+1 Daily Drivers · XGBoost"
                icon={BarChart3}
                colSpan={5}
                headerExtra={
                  <span className="cm-panel-telemetry-meta">
                    <span>T+1 Auxiliary</span>
                    {topDrivers.length ? <span>Top {topDrivers.length} Signals</span> : null}
                  </span>
                }
              >
                <div className="space-y-4">
                  {/* T+1 XGBoost diagnostic telemetry strip to prevent horizon conflation */}
                  <div className="border-b border-white/[0.07] pb-3 space-y-2">
                    <div className="flex flex-wrap items-baseline justify-between gap-2">
                      <div>
                        <span className="text-[11px] text-slate-400 uppercase tracking-wider block">
                          Next-Day (T+1) XGBoost Return
                        </span>
                        <div className="flex flex-wrap items-baseline gap-2 mt-0.5">
                          <strong
                            className={clsx(
                              'text-lg font-medium cm-tabular',
                              analysis?.predicted_return == null
                                ? 'text-slate-400'
                                : analysis.predicted_return >= 0
                                  ? 'text-[#74c7a9]'
                                  : 'text-[#db8296]',
                            )}
                          >
                            {formatSignedPct(analysis?.predicted_return)}
                          </strong>
                          {analysis?.predicted_price != null && (
                            <span className="text-xs text-slate-300 cm-tabular">
                              (${analysis.predicted_price.toFixed(2)} T+1)
                            </span>
                          )}
                        </div>
                      </div>
                      {analysis?.baseline_price_date && (
                        <span className="text-[11px] text-slate-400 cm-tabular">
                          Ref {analysis.baseline_price_date}
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      Auxiliary single-day (T+1) XGBoost feature weights — independent of the 5D TFT-ASRO primary forecast and its VSN focus.
                    </p>
                  </div>

                  {topDrivers.length ? (
                    <>
                      {/* Category breakdown bar + interactive filter */}
                      {driverBreakdown.length > 0 && (
                        <div className="cm-driver-breakdown">
                          <div className="cm-driver-spectrum" aria-hidden="true">
                            {driverBreakdown.map((seg) => (
                              <span
                                key={seg.category}
                                style={{
                                  width: `${seg.sharePct}%`,
                                  backgroundColor: seg.color,
                                  opacity: selectedDriverCategory === 'ALL' || selectedDriverCategory === seg.category ? 0.9 : 0.25,
                                }}
                                title={`${seg.label}: ${seg.sharePct.toFixed(0)}%`}
                              />
                            ))}
                          </div>
                          <div className="cm-driver-filter-row" role="group" aria-label="Filter T+1 XGBoost drivers by category">
                            <button
                              type="button"
                              className="cm-driver-filter-btn"
                              aria-pressed={selectedDriverCategory === 'ALL'}
                              onClick={() => setSelectedDriverCategory('ALL')}
                            >
                              All ({topDrivers.length})
                            </button>
                            {driverBreakdown.map((seg) => (
                              <button
                                key={seg.category}
                                type="button"
                                className="cm-driver-filter-btn"
                                aria-pressed={selectedDriverCategory === seg.category}
                                onClick={() =>
                                  setSelectedDriverCategory((prev) => (prev === seg.category ? 'ALL' : seg.category))
                                }
                              >
                                <i className="cm-driver-dot" style={{ backgroundColor: seg.color }} aria-hidden="true" />
                                <span>{seg.category}</span>
                                <small>{seg.sharePct.toFixed(0)}%</small>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="cm-driver-list">
                        {visibleDrivers.map((inf: Influencer) => {
                          const maxImp = topDrivers[0]?.importance || 1;
                          const label = formatDriverLabel(inf);
                          const rank = topDrivers.indexOf(inf) + 1;
                          return (
                            <div
                              key={inf.feature}
                              tabIndex={0}
                              className="cm-driver-row group"
                            >
                              <div className="cm-driver-row-head">
                                <div className="flex items-start gap-2.5 min-w-0 flex-1">
                                  <span className="cm-driver-rank tabular-nums">0{rank}</span>
                                  {inf.category && (
                                    <span className={`text-xs px-1.5 py-0.5 rounded ${driverCategoryTone[inf.category] || driverCategoryTone.Other} font-medium tracking-wide shrink-0`}>
                                      {driverCategoryLabel[inf.category] || driverCategoryLabel.Other}
                                    </span>
                                  )}
                                  <span className="text-xs text-gray-200 group-hover:text-copper-400 transition-colors min-w-0 flex-1 whitespace-normal break-words leading-relaxed">
                                    {label}
                                  </span>
                                </div>
                                <div className="flex items-center gap-2 shrink-0">
                                  {inf.time_horizon && (
                                    <span className="cm-driver-horizon">{inf.time_horizon}</span>
                                  )}
                                  <span className="text-xs font-mono text-slate-300 tabular-nums">{(inf.importance * 100).toFixed(1)}%</span>
                                </div>
                              </div>
                              <div className="cm-driver-bar-track">
                                <div
                                  className="cm-driver-bar-fill"
                                  style={{ transformOrigin: 'left', transform: `scaleX(${Math.max(0.04, Math.min(1, inf.importance / maxImp))})` }}
                                />
                              </div>
                              <div className="cm-driver-detail-line">
                                <code>{inf.feature}</code>
                                {inf.description && inf.description !== label && (
                                  <span>{inf.description}</span>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-8 text-center gap-2">
                      <BarChart3 size={24} className="text-slate-400" />
                      <p className="text-xs text-slate-400">T+1 XGBoost drivers are unavailable</p>
                      <p className="text-xs text-slate-400">They will appear after the next model refresh.</p>
                    </div>
                  )}
                </div>
              </GlassCard>
            )}

          </div>

          {showNews && (
            <aside id="news-intelligence" className="cm-dashboard-news-section min-w-0" data-cm-route-reveal="surface">
              {deferredPanels.news || streamLens === 'news' ? (
                <Suspense fallback={newsFallback}>
                  <NewsIntelligencePanel />
                </Suspense>
              ) : newsFallback}
            </aside>
          )}
        </div>

        {showMap && (
          <section id="market-map" className="cm-dashboard-market-map min-w-0 w-full" aria-label="Global copper market map" data-cm-route-reveal="surface">
            {deferredPanels.map || streamLens === 'map' ? (
              <Suspense fallback={<MapSkeleton />}>
                <HeatmapPanel />
              </Suspense>
            ) : <MapSkeleton />}
          </section>
        )}
      </div>
    </div>
  );
};
