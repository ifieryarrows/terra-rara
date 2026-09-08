import { useEffect, useState, useCallback, useRef, Suspense, lazy, memo } from 'react';
import { motion } from 'framer-motion';
import { FinancialPanel as GlassCard } from '../components/ui/FinancialPanel';
import { PriceForecastChart } from '../features/forecast/PriceForecastChart';
import { RefreshButton } from '../components/ui/RefreshButton';
import { ModelReliability } from '../features/forecast/ModelReliability';
import { ViewState } from '../components/ui/ViewState';
import { PageHeader } from '../components/ui/PageHeader';
import {
  Activity, Globe, BarChart3, Cpu, TrendingUp, TrendingDown,
  Brain, Crosshair, AlertTriangle, Minus
} from 'lucide-react';
import clsx from 'clsx';
import { formatQuoteDelta, quoteComparison } from '../utils/quote';

import {
  fetchAnalysis,
  fetchHistory,
  fetchCommentary,
  fetchTFTAnalysis,
  fetchLivePrice,
} from '../api';
import { COPPER_INSTRUMENT, DEFAULT_COPPER_SYMBOL } from '../config/instruments';
import type {
  AnalysisReport, HistoryResponse, Influencer,
  CommentaryResponse, TFTAnalysisResponse
} from '../types';
import { useSentimentSummary } from '../hooks/useQueries';
import '../App.css';

// Lazy load heavy components
const HeatmapPanel = lazy(() => import('../features/heatmap/HeatmapPanel').then(m => ({ default: m.HeatmapPanel })));
const NewsIntelligencePanel = lazy(() =>
  import('../features/news/NewsIntelligencePanel').then(m => ({ default: m.NewsIntelligencePanel })),
);

// --- Skeleton Components for perceived performance ---
const MapSkeleton = () => (
  <div className="h-[400px] w-full flex items-center justify-center bg-midnight/50 rounded-xl">
    <div className="flex flex-col items-center gap-3">
      <Globe size={32} className="text-copper-500/50 animate-pulse" />
      <span className="text-slate-400 text-xs font-mono">Loading intelligence map...</span>
    </div>
  </div>
);

const driverCategoryTone: Record<string, string> = {
  Momentum:   'bg-copper-500/15 text-copper-300',
  Trend:      'bg-blue-500/15 text-blue-300',
  Volatility: 'bg-amber-500/15 text-amber-300',
  Sentiment:  'bg-violet-500/15 text-violet-300',
  Macro:      'bg-emerald-500/15 text-emerald-300',
  Sector:     'bg-rose-500/15 text-rose-300',
  Embedding:  'bg-slate-500/15 text-slate-300',
  Other:      'bg-white/5 text-gray-400',
};

const driverCategoryLabel: Record<string, string> = {
  Momentum: 'Momentum signal',
  Trend: 'Trend signal',
  Volatility: 'Price volatility',
  Sentiment: 'News sentiment',
  Macro: 'Macro market',
  Sector: 'Related market',
  Embedding: 'News pattern',
  Other: 'Other factor',
};

// --- Components ---

const NumberTicker = memo(({ value, format = (v: number) => v.toFixed(2), className = ''}: { value: number; format?: (v: number) => string; className?: string }) => (
  <span className={clsx("font-mono tabular-nums", className)}>{format(value)}</span>
));
NumberTicker.displayName = 'NumberTicker';

// --- Main App ---

export const OverviewPage = () => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const refreshInFlight = useRef(false);
  const [commentaryLoading, setCommentaryLoading] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisReport | null>(null);
  const [tftAnalysis, setTftAnalysis] = useState<TFTAnalysisResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [commentary, setCommentary] = useState<CommentaryResponse | null>(null);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [loadErrors, setLoadErrors] = useState<Record<string, string>>({});
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [lastLiveUpdateAt, setLastLiveUpdateAt] = useState<Date | null>(null);
  const sentimentSummary = useSentimentSummary(7, 6);

  // Silent refresh - no loading state flash after initial load
  const loadData = useCallback(async (silent = false) => {
    if (refreshInFlight.current) return;
    refreshInFlight.current = true;
    setIsRefreshing(true);
    const [analysisResult, historyResult, tftResult] = await Promise.allSettled([
        fetchAnalysis(DEFAULT_COPPER_SYMBOL),
        fetchHistory(DEFAULT_COPPER_SYMBOL, 180),
        fetchTFTAnalysis(DEFAULT_COPPER_SYMBOL),
    ]);
    const nextErrors: Record<string, string> = {};
    if (analysisResult.status === 'fulfilled') setAnalysis(analysisResult.value);
    else {
      if (!silent) setAnalysis(null);
      nextErrors['Price forecast'] = String(analysisResult.reason);
    }
    if (historyResult.status === 'fulfilled') setHistory(historyResult.value);
    else {
      if (!silent) setHistory(null);
      nextErrors['Price history'] = String(historyResult.reason);
    }
    if (tftResult.status === 'fulfilled') setTftAnalysis(tftResult.value);
    else {
      if (!silent) setTftAnalysis(null);
      nextErrors['Deep-learning forecast'] = String(tftResult.reason);
    }
    setLoadErrors((previous) => ({
      ...(previous['AI commentary'] ? { 'AI commentary': previous['AI commentary'] } : {}),
      ...nextErrors,
    }));
    setIsInitialLoad(false);
    setIsRefreshing(false);
    refreshInFlight.current = false;
  }, []);

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

  // Initial load
  useEffect(() => {
    loadData(false);
  }, [loadData]);

  // Silent refresh every 60s - no UI flash
  useEffect(() => {
    const interval = setInterval(() => loadData(true), 60000);
    return () => clearInterval(interval);
  }, [loadData]);

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

  // Only show full loading on initial load
  if (isInitialLoad && !analysis) {
    return <div className="space-y-6"><PageHeader eyebrow="01 / MARKET INTELLIGENCE" title="Market overview" description="Copper prices, context and quantitative forecasts."/><ViewState kind="loading" title="Opening your market view" description="Retrieving prices, market context and available forecasts."/></div>;
  }

  const tftReturn = tftAnalysis?.primary_forecast_return
    ?? tftAnalysis?.weekly_forecast?.expected_return
    ?? tftAnalysis?.prediction?.weekly_return
    ?? null;
  const tftDegraded = !!tftAnalysis && (
    tftAnalysis.quality_state === 'degraded' ||
    tftAnalysis.model_state === 'retrain_required' ||
    tftAnalysis.is_forecast_healthy === false
  );
  const tftDegradedMessage = tftAnalysis?.message || 'TFT weekly forecast is unavailable until the weekly model artifacts are refreshed.';
  const tftBullish = tftReturn !== null ? tftReturn >= 0 : null;
  const tftMetrics = tftAnalysis?.model_metadata?.metrics;
  const tftDirection = tftAnalysis?.direction;
  const tftImpulse = tftAnalysis?.t1_impulse ?? tftAnalysis?.weekly_forecast?.t1_impulse ?? 'NEUTRAL';
  const tftReferencePrice = tftAnalysis?.prediction?.reference_price;
  const tftReferenceDate = tftAnalysis?.prediction?.reference_price_date;
  const tftAnomaly = tftAnalysis?.prediction?.anomaly_detected;
  const tftInstrument = tftAnalysis?.prediction?.instrument;
  const tftStalenessDays = tftAnalysis?.prediction?.baseline_staleness_days ?? 0;
  // Anything >= 3 calendar days is flagged; 0-2 is considered fresh (weekend).
  const tftBaselineIsStale = tftStalenessDays >= 3;
  const newsSentimentIndex = sentimentSummary.data?.index;
  const newsSentimentLabel = sentimentSummary.data?.label ?? 'Neutral';
  const newsSentimentMeta =
    newsSentimentLabel === 'Bullish'
      ? { tone: 'text-emerald-300', chip: 'bg-emerald-500/15 text-emerald-300 border-emerald-400/30', icon: TrendingUp, label: 'Positive' }
      : newsSentimentLabel === 'Bearish'
      ? { tone: 'text-rose-300', chip: 'bg-rose-500/15 text-rose-300 border-rose-400/30', icon: TrendingDown, label: 'Negative' }
      : { tone: 'text-amber-300', chip: 'bg-amber-500/15 text-amber-300 border-amber-400/30', icon: Minus, label: 'Balanced' };
  const SentimentIcon = newsSentimentMeta.icon;
  const latestHistoryPrice = [...(history?.data || [])]
    .reverse()
    .find((p) => p.price != null)?.price ?? null;
  const quotePrice = livePrice ?? latestHistoryPrice;
  const quoteChange = quoteComparison(livePrice, latestHistoryPrice);


  return (
    <div className="font-sans selection:bg-copper-500/30">


      <div className="relative z-10 grid min-w-0 grid-cols-1 gap-8">

        {/* Header */}
        <header className="cm-overview-header">
          <div className="space-y-1">
            <p className="cm-eyebrow">01 / MARKET INTELLIGENCE</p>
            <h1
              className="text-3xl sm:text-4xl font-medium text-white tracking-tight"
            >
              Market overview
            </h1>
            <p className="text-slate-400 text-sm">Copper prices, context and quantitative forecasts.</p>
          </div>

          <div className="cm-quote-strip">
            <div className="cm-quote">
              <div className="flex items-center gap-3">
                <div className="w-11 h-11 rounded-md flex items-center justify-center shrink-0">
                  <span className="cm-brand-mark" aria-hidden="true">Cu</span>
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-xs text-slate-400 uppercase tracking-widest font-semibold">
                    Copper Futures
                  </p>
                  <div className="mt-1 flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded-md border border-slate-700 bg-slate-900 text-xs text-white font-semibold tracking-wide">
                      {COPPER_INSTRUMENT.canonicalSymbol}
                    </span>
                    <span className="text-xs text-slate-400">{livePrice != null ? 'Delayed quote' : latestHistoryPrice != null ? 'Last available close' : 'Awaiting quote'}</span>
                  </div>
                  <div className="mt-1 flex flex-wrap items-baseline gap-2 font-mono">
                    <span className="text-3xl text-white leading-none">
                      {quotePrice != null ? quotePrice.toFixed(4) : '--'}
                    </span>
                    <span className="text-sm text-slate-400">USD</span>
                    {quoteChange && (
                      <span className={clsx("text-xl leading-none", quoteChange.tone === "neutral" ? "text-slate-400" : quoteChange.tone === "positive" ? "text-emerald-400" : "text-rose-400")}>
                        {formatQuoteDelta(quoteChange.delta)} {formatQuoteDelta(quoteChange.percent)}%
                      </span>
                    )}
                  </div>
                  <p className="mt-0.5 text-xs text-slate-400">
                    {lastLiveUpdateAt
                      ? `Last checked ${lastLiveUpdateAt.toLocaleDateString()} ${lastLiveUpdateAt.toLocaleTimeString()}`
                      : latestHistoryPrice != null ? 'Showing the last available historical close' : 'Waiting for latest quote'}
                  </p>
                  {quoteChange && <p className="text-xs text-slate-400 mt-1">Change vs last stored close</p>}
                </div>
              </div>
            </div>
            <div className="px-4 py-2 rounded-xl bg-midnight/50 flex flex-col items-end min-w-[120px]">
              <span className="text-xs text-slate-400 font-bold uppercase tracking-wider">7D News Sentiment</span>
              <div className={clsx("mt-1 inline-flex items-center gap-1.5 px-2 py-1 rounded-md border text-xs font-semibold", newsSentimentMeta.chip)}>
                <SentimentIcon size={12} />
                <span>{newsSentimentIndex == null ? (sentimentSummary.isLoading ? 'Loading' : 'Unavailable') : newsSentimentMeta.label}</span>
              </div>
              <div className={clsx("mt-1 font-mono text-xs", newsSentimentMeta.tone)}>
                <NumberTicker value={newsSentimentIndex ?? NaN} format={(v: number) => Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${v.toFixed(3)}` : '—'} />
              </div>
            </div>
          </div>
        </header>

        <div className="cm-overview-tools">
          <nav aria-label="Overview sections"><a href="#price-forecast">Price chart</a><a href="#news-intelligence">News</a><a href="#market-map">Market map</a></nav>
          <RefreshButton label="Refresh overview" busy={isRefreshing} onClick={() => { void loadData(true); }}/>
        </div>
        {Object.keys(loadErrors).length > 0 && (
          <div className="flex flex-wrap gap-2" role="status">
            {Object.keys(loadErrors).map((endpoint) => (
              <span key={endpoint} className="inline-flex items-center gap-1.5 rounded-md border border-amber-400/30 bg-amber-500/10 px-2.5 py-1 text-xs text-amber-200">
                <AlertTriangle size={12} /> {endpoint} could not be refreshed; any visible values are from the previous response
              </span>
            ))}
          </div>
        )}

        {/* Dashboard Grid + persistent News sidebar (desktop).
            On mobile/tablet the news panel stacks under the dashboard.
            Width grows with the viewport so chips/filters have room to breathe. */}
        <div className="grid gap-4 lg:gap-6 lg:grid-cols-[minmax(0,1fr)_340px] xl:grid-cols-[minmax(0,1fr)_380px] 2xl:grid-cols-[minmax(0,1fr)_420px]">
        {/* Main dashboard column */}
        <div className="grid grid-cols-12 gap-6">

          {/* Primary weekly forecast; single-step diagnostics are grouped below. */}
          <GlassCard title="Deep Learning Weekly Forecast" icon={Brain} colSpan={4} className={clsx("relative overflow-hidden", tftBullish === null ? "" : tftBullish ? "border-emerald-500/30" : "border-rose-500/30")}>
            {tftDegraded ? (
              <div className="flex flex-col justify-center h-full py-10 gap-4">
                <div className="flex items-center gap-2 text-amber-300">
                  <AlertTriangle size={18} />
                  <span className="text-xs font-bold uppercase tracking-widest">Forecast Degraded</span>
                </div>
                <p className="text-sm text-gray-400 leading-relaxed">
                  {tftDegradedMessage}
                </p>
                <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2">
                  <span className="text-xs text-gray-300">Historical prices remain available while the forecast is unavailable.</span>
                </div>
              </div>
            ) : tftAnalysis?.prediction ? (() => {
              const prediction = tftAnalysis.prediction;
              const weeklyQ10 = tftAnalysis.primary_forecast_q10 ?? tftAnalysis.weekly_forecast?.q10_return ?? prediction.weekly_return_q10_calibrated;
              const weeklyQ90 = tftAnalysis.primary_forecast_q90 ?? tftAnalysis.weekly_forecast?.q90_return ?? prediction.weekly_return_q90_calibrated;
              const calibrated = tftAnalysis.weekly_forecast?.calibrated ?? prediction.weekly_interval_calibrated ?? false;
              return (
                <>
                  <div className="absolute top-0 right-0 p-4 opacity-5">
                    {tftBullish ? <TrendingUp size={100} /> : <TrendingDown size={100} />}
                  </div>
                  <div className="relative z-10 space-y-4">

                    {/* Weekly direction badge */}
                    <div className="flex items-center gap-2 flex-wrap">
                      <div className={clsx(
                        "inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-sm font-bold tracking-wide",
                        tftDirection === 'BULLISH' ? "bg-emerald-400/10 text-emerald-400 border border-emerald-400/20" :
                        tftDirection === 'BEARISH' ? "bg-rose-400/10 text-rose-400 border border-rose-400/20" :
                                                     "bg-amber-400/10 text-amber-400 border border-amber-400/20"
                      )}>
                        {tftDirection === 'BULLISH' ? <TrendingUp size={14} /> : tftDirection === 'BEARISH' ? <TrendingDown size={14} /> : <Activity size={14} />}
                        {tftDirection ?? 'Unavailable'}
                      </div>
                    </div>

                    {/* Primary 5-day headline from the published weekly forecast. */}
                    <div>
                      <div className="mb-1 flex items-center justify-between gap-3">
                        <span className="text-xs text-slate-400 uppercase tracking-widest">
                          Expected 5D Performance
                        </span>
                        {tftBaselineIsStale && (
                          <span
                            title={`The forecast baseline close is ${tftStalenessDays} calendar days old.`}
                            className="px-1.5 py-0.5 rounded border border-amber-500/40 bg-amber-500/10 text-amber-300 text-xs tracking-wider"
                          >
                            Stale {tftStalenessDays}d
                          </span>
                        )}
                      </div>
                      <div className="flex flex-wrap items-baseline gap-2">
                        <span className={clsx("text-3xl font-light font-mono", tftBullish == null ? "text-slate-400" : tftBullish ? "text-emerald-400" : "text-rose-400")}>
                          {tftReturn == null ? '—' : `${tftReturn >= 0 ? '+' : ''}${(tftReturn * 100).toFixed(2)}%`}
                        </span>
                        <span className="text-sm text-gray-400 font-mono">${prediction.weekly_price?.toFixed(2) ?? '--'}</span>
                        {tftReferencePrice != null && (
                          <span className="text-xs text-slate-400 font-mono">
                            (from ${tftReferencePrice.toFixed(2)})
                          </span>
                        )}
                      </div>
                      <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-400">
                        <div>
                          <span className="block uppercase tracking-wider">Instrument</span>
                          <span className="font-mono text-gray-300">{tftInstrument?.symbol || COPPER_INSTRUMENT.canonicalSymbol}</span>
                        </div>
                        <div className="text-right">
                          <span className="block uppercase tracking-wider">Close Date</span>
                          <span className="font-mono text-gray-300">{tftReferenceDate ?? '--'}</span>
                        </div>
                      </div>
                      {tftAnomaly && (
                        <p className="mt-1 text-xs text-amber-400">
                          The model flagged an unusual output. Review model validation before interpreting this forecast.
                        </p>
                      )}
                    </div>

                    {/* Primary weekly interval */}
                    <div className="rounded-lg bg-white/[0.02] border border-white/5 px-3 py-2">
                      <p className="text-xs text-slate-400 uppercase tracking-widest mb-1.5">
                        5D Range {calibrated ? '(calibrated)' : '(raw)'}
                      </p>
                      <div className="flex items-center justify-between">
                        <div className="text-center">
                          <p className="text-xs text-slate-400 mb-0.5">Low</p>
                          <span className="text-sm font-mono text-rose-400/80">
                            {weeklyQ10 != null ? `${(weeklyQ10 * 100).toFixed(2)}%` : '--'}
                          </span>
                        </div>
                        <div className="flex-1 mx-3 h-1.5 rounded-full bg-white/5 relative overflow-hidden">
                          <div className="absolute inset-0 bg-gradient-to-r from-rose-500/40 via-gray-500/20 to-emerald-500/40 rounded-full" />
                        </div>
                        <div className="text-center">
                          <p className="text-xs text-slate-400 mb-0.5">High</p>
                          <span className="text-sm font-mono text-emerald-400/80">
                            {weeklyQ90 != null ? `${weeklyQ90 >= 0 ? '+' : ''}${(weeklyQ90 * 100).toFixed(2)}%` : '--'}
                          </span>
                        </div>
                      </div>
                    </div>

                    <details className="cm-data-disclosure"><summary>T+1 diagnostics</summary>
                    <div className="flex items-center justify-between py-2 border-t border-white/5">
                      <span className="text-xs text-slate-400 uppercase tracking-wider">T+1 Impulse</span>
                      <div className="flex items-center gap-1.5">
                        {tftImpulse === 'BULLISH' ? <TrendingUp size={12} className="text-emerald-400" /> :
                         tftImpulse === 'BEARISH' ? <TrendingDown size={12} className="text-rose-400" /> :
                         <Activity size={12} className="text-amber-400" />}
                        <span className={clsx("text-xs font-bold tracking-wide",
                          tftImpulse === 'BULLISH' ? "text-emerald-400" :
                          tftImpulse === 'BEARISH' ? "text-rose-400" : "text-amber-400"
                        )}>
                          {tftImpulse}
                          {tftAnalysis.t1_return != null ? ` ${tftAnalysis.t1_return >= 0 ? '+' : ''}${(tftAnalysis.t1_return * 100).toFixed(2)}%` : ''}
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div className="rounded bg-white/[0.02] border border-white/5 px-2 py-1.5">
                        <span className="block text-xs text-slate-400 uppercase tracking-wider">Direction Score</span>
                        <span className="font-mono text-xs text-gray-200">
                          {tftMetrics?.directional_accuracy != null ? `${(tftMetrics.directional_accuracy * 100).toFixed(1)}%` : '--'}
                        </span>
                      </div>
                      <div className="rounded bg-white/[0.02] border border-white/5 px-2 py-1.5 text-right">
                        <span className="block text-xs text-slate-400 uppercase tracking-wider">Sharpe</span>
                        <span className="font-mono text-xs text-gray-200">
                          {tftMetrics?.sharpe_ratio != null ? tftMetrics.sharpe_ratio.toFixed(2) : '--'}
                        </span>
                      </div>
                    </div>

                    </details>

                    <div className="border-t border-white/5 pt-3">
                      <p className="text-xs text-slate-400">Model volatility classification</p>
                      <p className="text-sm text-slate-200 mt-1">{tftAnalysis.risk_level ?? 'Unavailable'}</p>
                      <p className="text-xs text-slate-400 mt-1">Based on forecast dispersion. This is separate from model quality and is not an investment safety rating.</p>
                    </div>
                  </div>
                </>
              );
            })() : (
              <ViewState kind={loadErrors['Deep-learning forecast'] ? 'error' : 'empty'} title={loadErrors['Deep-learning forecast'] ? 'Weekly forecast could not be loaded' : 'No weekly forecast available'} description="Use Refresh overview to check for the latest available forecast." compact/>
            )}
          </GlassCard>

          <GlassCard id="price-forecast" title={`Price Forecast (${COPPER_INSTRUMENT.canonicalSymbol})`} icon={Activity} colSpan={8} className="min-h-[400px]">
            <PriceForecastChart history={history?.data ?? []} forecast={tftAnalysis} historyError={!!loadErrors['Price history']} forecastError={!!loadErrors['Deep-learning forecast']}/>
          </GlassCard>

          {/* Influencers Card — shows human-readable labels, category chips and
              technical ids on hover. Backend contract: Influencer has
              `label`, `description`, `category`, `time_horizon`. */}
          <GlassCard title="Market Drivers" icon={BarChart3} colSpan={4}>
            <div className="space-y-4">
              {analysis?.top_influencers?.length ? (
                analysis.top_influencers.slice(0, 5).map((inf: Influencer, i: number) => {
                  const maxImp = analysis.top_influencers[0]?.importance || 1;
                  const label = inf.label || inf.description || inf.feature;
                  return (
                    <div key={inf.feature} className="group" title={inf.feature}>
                      <div className="flex justify-between items-start mb-1 gap-3">
                        <div className="flex items-start gap-2 min-w-0 flex-1">
                          {inf.category && (
                            <span className={`text-xs px-1.5 py-0.5 rounded ${driverCategoryTone[inf.category] || driverCategoryTone.Other} font-medium tracking-wide shrink-0`}>
                              {driverCategoryLabel[inf.category] || driverCategoryLabel.Other}
                            </span>
                          )}
                          <span className="text-xs text-gray-300 group-hover:text-copper-400 transition-colors min-w-0 flex-1 whitespace-normal break-words leading-relaxed">
                            {label}
                          </span>
                        </div>
                        <span className="text-xs font-mono text-slate-400 shrink-0">{(inf.importance * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-gradient-to-r from-copper-500 to-copper-400"
                          initial={false}
                          style={{ transformOrigin: "left", transform: "scaleX(" + Math.max(0, Math.min(1, inf.importance / maxImp)) + ")" }}
                          transition={{ delay: 0.2 + (i * 0.1), duration: 0.8 }}
                        />
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="flex flex-col items-center justify-center py-8 text-center gap-2">
                  <BarChart3 size={24} className="text-slate-400" />
                  <p className="text-xs text-slate-400">Market drivers are unavailable</p>
                  <p className="text-xs text-slate-400">They will appear after the next model refresh.</p>
                </div>
              )}
            </div>
          </GlassCard>

          {/* Model Health Card */}
          <GlassCard title="Model Reliability" icon={Crosshair} colSpan={4}>
            <ModelReliability metrics={tftMetrics} unavailable={!!loadErrors['Deep-learning forecast']}/>
          </GlassCard>

          {/* AI Commentary Card */}
          <GlassCard title="Neural Analysis" icon={Cpu} colSpan={4}>
            <div className="flex items-center justify-between mb-3">
              {commentary?.generated_at && (
                <span className="text-xs text-slate-400 font-mono">
                  {new Date(commentary.generated_at).toLocaleTimeString()}
                </span>
              )}
              {commentary?.generation_mode && (
                <span
                  className={clsx(
                    "text-xs font-mono px-1.5 py-0.5 rounded-full border",
                    commentary.generation_mode !== 'llm' && commentary.generation_mode !== 'llm_repaired'
                      ? "text-amber-300 border-amber-400/30 bg-amber-500/10"
                      : "text-emerald-300 border-emerald-400/30 bg-emerald-500/10",
                  )}
                  title={commentary.fallback_reason || commentary.model_name || undefined}
                >
                  {commentary.generation_mode === 'deterministic_fallback' ? 'Local fallback' : commentary.generation_mode === 'llm' || commentary.generation_mode === 'llm_repaired' ? 'AI generated' : 'Unavailable'}
                </span>
              )}
            </div>
            <div className="cm-commentary text-sm text-gray-300 leading-relaxed">
              {commentary?.commentary ? (
                <p className="font-light whitespace-pre-wrap">{commentary.commentary || ''}</p>
              ) : (
                <span className="text-slate-400" role="status">{commentaryLoading ? 'Loading available commentary…' : loadErrors['AI commentary'] || commentary?.error ? 'AI commentary is temporarily unavailable.' : 'No commentary is available for this snapshot.'}</span>
              )}
            </div>
          </GlassCard>

        </div>
        {/* Right sticky News Intelligence sidebar (desktop) / stacks under on mobile */}
        <aside id="news-intelligence" className="cm-news-sidebar min-w-0">
          <Suspense
            fallback={
              <div className="glass-panel h-full min-h-[480px] flex items-center justify-center">
                <span className="text-xs text-slate-400 font-mono tracking-widest uppercase">
                  Loading news…
                </span>
              </div>
            }
          >
            <NewsIntelligencePanel />
          </Suspense>
        </aside>
        </div>

        {/* The market map owns the full content width. News remains available
            above without consuming horizontal heatmap space. */}
        <div id="market-map" className="min-w-0 w-full">
          <Suspense fallback={<MapSkeleton />}>
            <HeatmapPanel />
          </Suspense>
        </div>
      </div>
    </div>
  );
}
