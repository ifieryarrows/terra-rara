import { useEffect, useState, useCallback, useRef, Suspense, lazy, memo } from 'react';
import { motion } from 'framer-motion';
import { FinancialPanel as GlassCard } from '../components/ui/FinancialPanel';
import { PriceForecastChart } from '../features/forecast/PriceForecastChart';
import { RefreshButton } from '../components/ui/RefreshButton';
import { ModelReliability } from '../features/forecast/ModelReliability';
import { ViewState } from '../components/ui/ViewState';
import { OverviewSkeleton } from '../components/skeletons';
import { BrandMark } from '../components/ui/BrandMark';
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
  <div className="cm-map-skeleton h-[400px] w-full flex items-center justify-center bg-midnight/50 rounded-xl" role="status" aria-label="Loading intelligence map">
    <div className="flex flex-col items-center gap-3">
      <div className="cm-map-skeleton__visual" aria-hidden="true">
        <span className="cm-map-skeleton__orbit" />
        <span className="cm-map-skeleton__orbit cm-map-skeleton__orbit--reverse" />
        <Globe size={32} className="cm-map-skeleton__globe" />
      </div>
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
  const [activeOverviewSection, setActiveOverviewSection] = useState('price-forecast');
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

  // Keep the section rail in sync with the part of the workspace in view.
  useEffect(() => {
    if (isInitialLoad) return;
    const ids = ['price-forecast', 'weekly-outlook', 'market-drivers', 'news-intelligence', 'market-map'];
    if (typeof IntersectionObserver === 'undefined') return;
    const observer = new IntersectionObserver((entries) => {
      const current = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (current?.target.id) {
        setActiveOverviewSection((prev) => (prev === current.target.id ? prev : current.target.id));
      }
    }, { rootMargin: '-18% 0px -68% 0px', threshold: [0, 0.2, 0.5, 1] });

    ids.forEach((id) => {
      const section = document.getElementById(id);
      if (section) observer.observe(section);
    });

    return () => {
      observer.disconnect();
    };
  }, [isInitialLoad]);

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
    return <OverviewSkeleton />;
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


  const handleSectionNav = (event: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    event.preventDefault();
    setActiveOverviewSection(id);
    const target = document.getElementById(id);
    if (target) {
      window.history.pushState(null, '', `#${id}`);
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div className="cm-market-dashboard font-sans selection:bg-copper-500/30" data-cm-dashboard-ready={isInitialLoad ? 'false' : 'true'}>
      <div className="cm-dashboard-page relative z-10 grid min-w-0 grid-cols-1">

        <header className="cm-dashboard-hero">
          <div className="cm-dashboard-hero-copy">
            <p className="cm-eyebrow cm-dashboard-hero-eyebrow" data-cm-route-reveal="copy"><BrandMark size={18} variant="small"/>COPPER INTELLIGENCE / TERRA RARA</p>
            <h1 className="cm-dashboard-title" data-cm-route-reveal="copy">Copper<br/><span>market</span></h1>
            <p className="cm-dashboard-hero-subtitle" data-cm-route-reveal="copy">COMEX futures · Price action, model outlook and market context.</p>
          </div>

          <div className="cm-dashboard-hero-data" data-cm-route-reveal="surface" role="group" aria-label="Copper market snapshot">
            <div className="cm-dashboard-instrument">
              <div>
                <p className="cm-dashboard-instrument-name">{COPPER_INSTRUMENT.displayName}</p>
                <span className="cm-dashboard-symbol">{COPPER_INSTRUMENT.canonicalSymbol}</span>
              </div>
              <span className={clsx("cm-dashboard-quote-status", livePrice != null ? "cm-dashboard-quote-status--delayed" : "")}>
                <i aria-hidden="true"/>{livePrice != null ? 'Delayed quote' : latestHistoryPrice != null ? 'Last available close' : 'Awaiting quote'}
              </span>
            </div>

            <div className="cm-dashboard-price-row">
              <span className="cm-dashboard-price">{quotePrice != null ? quotePrice.toFixed(4) : '--'}</span>
              <span className="cm-dashboard-currency">USD</span>
              {quoteChange && (
                <span className={clsx("cm-dashboard-change", quoteChange.tone === "neutral" ? "text-slate-400" : quoteChange.tone === "positive" ? "text-emerald-400" : "text-rose-400")}>
                  {formatQuoteDelta(quoteChange.delta)} <span>({formatQuoteDelta(quoteChange.percent)}%)</span>
                </span>
              )}
            </div>

            <div className="cm-dashboard-quote-meta">
              <span>{lastLiveUpdateAt ? `Checked ${lastLiveUpdateAt.toLocaleDateString()} ${lastLiveUpdateAt.toLocaleTimeString()}` : latestHistoryPrice != null ? 'Showing the last available historical close' : 'Waiting for latest quote'}</span>
              {quoteChange && <span>Change vs last stored close</span>}
            </div>

            <div className="cm-dashboard-sentiment">
              <div className="cm-dashboard-sentiment-copy"><span>NEWS SENTIMENT</span><small>Headline tone</small></div>
              <div className={clsx("cm-sentiment-badge", newsSentimentMeta.chip)}>
                <SentimentIcon size={14} aria-hidden="true" />
                <span>{newsSentimentIndex == null ? (sentimentSummary.isLoading ? 'Loading' : 'Unavailable') : newsSentimentMeta.label}</span>
              </div>
              <span className={clsx("cm-dashboard-sentiment-score", newsSentimentMeta.tone)}>
                <NumberTicker value={newsSentimentIndex ?? NaN} format={(v: number) => Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${v.toFixed(3)}` : '—'} />
              </span>
            </div>
          </div>
        </header>

        <div className="cm-overview-tools" data-cm-route-reveal="surface">
          <nav aria-label="Market overview sections">
            <a href="#price-forecast" onClick={(e) => handleSectionNav(e, 'price-forecast')} aria-current={activeOverviewSection === 'price-forecast' ? 'location' : undefined}>Price action</a>
            <a href="#weekly-outlook" onClick={(e) => handleSectionNav(e, 'weekly-outlook')} aria-current={activeOverviewSection === 'weekly-outlook' ? 'location' : undefined}>Weekly outlook</a>
            <a href="#market-drivers" onClick={(e) => handleSectionNav(e, 'market-drivers')} aria-current={activeOverviewSection === 'market-drivers' ? 'location' : undefined}>Drivers</a>
            <a href="#news-intelligence" onClick={(e) => handleSectionNav(e, 'news-intelligence')} aria-current={activeOverviewSection === 'news-intelligence' ? 'location' : undefined}>News flow</a>
            <a href="#market-map" onClick={(e) => handleSectionNav(e, 'market-map')} aria-current={activeOverviewSection === 'market-map' ? 'location' : undefined}>Market map</a>
          </nav>
          <RefreshButton label="Refresh overview" busy={isRefreshing} onClick={() => { void loadData(true); }}/>
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

          <GlassCard id="price-forecast" title={`Price action / ${COPPER_INSTRUMENT.canonicalSymbol}`} icon={Activity} colSpan={12} className="cm-price-action-section">
            <PriceForecastChart history={history?.data ?? []} forecast={tftAnalysis} historyError={!!loadErrors['Price history']} forecastError={!!loadErrors['Deep-learning forecast']}/>
          </GlassCard>

          <GlassCard id="weekly-outlook" title="Weekly model outlook" icon={Brain} colSpan={6} className={clsx("cm-weekly-forecast-panel relative", tftBullish === null ? "" : tftBullish ? "border-emerald-500/30" : "border-rose-500/30")}>
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
                    <div className="cm-weekly-range">
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

                    <div className="cm-diagnostic-metrics grid grid-cols-2 gap-2">
                      <div>
                        <span className="block text-xs text-slate-400 uppercase tracking-wider">Direction Score</span>
                        <span className="font-mono text-xs text-gray-200">
                          {tftMetrics?.directional_accuracy != null ? `${(tftMetrics.directional_accuracy * 100).toFixed(1)}%` : '--'}
                        </span>
                      </div>
                      <div className="text-right">
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

          {/* Influencers Card — shows human-readable labels, category chips and
              technical ids on hover. Backend contract: Influencer has
              `label`, `description`, `category`, `time_horizon`. */}
          <GlassCard id="market-drivers" title="Market drivers" icon={BarChart3} colSpan={6}>
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
          <GlassCard id="model-reliability" title="Model reliability" icon={Crosshair} colSpan={4}>
            <ModelReliability metrics={tftMetrics} unavailable={!!loadErrors['Deep-learning forecast']}/>
          </GlassCard>

          {/* AI Commentary Card */}
          <GlassCard id="neural-analysis" title="Neural analysis" icon={Cpu} colSpan={8}>
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
        <aside id="news-intelligence" className="cm-dashboard-news-section min-w-0" data-cm-route-reveal="surface">
          <Suspense
            fallback={
              <div className="cm-news-loading min-h-[280px] flex items-center justify-center gap-3" role="status">
                <span className="cm-inline-loader" aria-hidden="true" />
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

        <div id="market-map" className="cm-dashboard-market-map min-w-0 w-full" data-cm-route-reveal="surface">
          <Suspense fallback={<MapSkeleton />}>
            <HeatmapPanel />
          </Suspense>
        </div>
      </div>
    </div>
  );
}
