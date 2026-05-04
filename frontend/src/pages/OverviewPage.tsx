import { useEffect, useState, useCallback, useMemo, Suspense, lazy, memo } from 'react';
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { motion } from 'framer-motion';
import {
  Activity, Globe, BarChart3, Cpu, TrendingUp, TrendingDown,
  Brain, Crosshair, AlertTriangle, CheckCircle2, Clock, Minus
} from 'lucide-react';
import clsx from 'clsx';

import {
  fetchAnalysis,
  fetchHistory,
  fetchCommentary,
  fetchTFTAnalysis,
  fetchLivePrice,
} from '../api';
import { COPPER_INSTRUMENT, DEFAULT_COPPER_SYMBOL } from '../config/instruments';
import type {
  AnalysisReport, HistoryResponse,
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
const ChartSkeleton = () => (
  <div className="h-[350px] w-full flex items-center justify-center">
    <div className="flex flex-col items-center gap-3">
      <div className="w-8 h-8 border-2 border-copper-500/30 border-t-copper-500 rounded-full animate-spin" />
      <span className="text-gray-500 text-xs font-mono">Loading chart...</span>
    </div>
  </div>
);

const MapSkeleton = () => (
  <div className="h-[400px] w-full flex items-center justify-center bg-midnight/50 rounded-xl">
    <div className="flex flex-col items-center gap-3">
      <Globe size={32} className="text-copper-500/50 animate-pulse" />
      <span className="text-gray-500 text-xs font-mono">Loading intelligence map...</span>
    </div>
  </div>
);

// --- Components ---

const GlassCard = memo(({ title, icon: Icon, children, className = '', colSpan = 1 }: any) => (
  <motion.div
    className={clsx(
      "glass-panel p-6 shadow-lg hover:shadow-xl transition-shadow duration-500",
      className
    )}
    style={{ gridColumn: `span ${colSpan}` }}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, ease: "easeOut" }}
  >
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-2 text-gray-400">
        {Icon && <Icon size={18} className="text-copper-400" />}
        <span className="text-xs font-bold tracking-widest uppercase">{title}</span>
      </div>
    </div>
    <div className="relative">
      {children}
    </div>
  </motion.div>
));
GlassCard.displayName = 'GlassCard';

const NumberTicker = memo(({ value, format = (v: number) => v.toFixed(2), className = '' }: any) => {
  return (
    <motion.span
      key={value}
      initial={{ opacity: 0, filter: 'blur(4px)' }}
      animate={{ opacity: 1, filter: 'blur(0px)' }}
      transition={{ duration: 0.3 }}
      className={clsx("font-mono", className)}
    >
      {format(value)}
    </motion.span>
  );
});
NumberTicker.displayName = 'NumberTicker';

// Simple progress bar [0-100]
const ProgressBar = memo(({ value, max = 100, color = 'bg-emerald-500' }: { value: number; max?: number; color?: string }) => (
  <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
    <motion.div
      className={clsx("h-full rounded-full", color)}
      initial={{ width: 0 }}
      animate={{ width: `${Math.min(100, (value / max) * 100)}%` }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
    />
  </div>
));
ProgressBar.displayName = 'ProgressBar';

function addBusinessDays(start: Date, n: number): Date {
  const result = new Date(start);
  let added = 0;
  while (added < n) {
    result.setDate(result.getDate() + 1);
    if (result.getDay() !== 0 && result.getDay() !== 6) added++;
  }
  return result;
}

const ForecastTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div className="bg-midnight/90 backdrop-blur-md border border-white/10 rounded-xl px-3 py-2 text-xs font-mono">
      <p className="text-gray-400 mb-1 font-sans">
        {new Date(label).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
      </p>
      {d?.price != null && (
        <p className="text-copper-400">Price: ${d.price.toFixed(2)}</p>
      )}
      {d?.isForecast && d?.priceMedian != null && (
        <>
          <p className="text-violet-400">Forecast: ${d.priceMedian.toFixed(2)}</p>
          {d?.priceQ10 != null && (
            <p className="text-violet-400/60">
              80% Range: ${d.priceQ10.toFixed(2)} — ${d.priceQ90.toFixed(2)}
            </p>
          )}
        </>
      )}
    </div>
  );
};

// --- Main App ---

export const OverviewPage = () => {
  const [analysis, setAnalysis] = useState<AnalysisReport | null>(null);
  const [tftAnalysis, setTftAnalysis] = useState<TFTAnalysisResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [commentary, setCommentary] = useState<CommentaryResponse | null>(null);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [lastLiveUpdateAt, setLastLiveUpdateAt] = useState<Date | null>(null);
  const sentimentSummary = useSentimentSummary(7, 6);

  // Silent refresh - no loading state flash after initial load
  const loadData = useCallback(async (silent = false) => {
    try {
      const [analysisData, historyData, tftData] = await Promise.all([
        fetchAnalysis(DEFAULT_COPPER_SYMBOL),
        fetchHistory(DEFAULT_COPPER_SYMBOL, 180),
        fetchTFTAnalysis(DEFAULT_COPPER_SYMBOL),
      ]);
      setAnalysis(analysisData);
      setHistory(historyData);
      setTftAnalysis(tftData);
      if (!silent) setIsInitialLoad(false);
    } catch (err) {
      console.error('Data load failed:', err);
      if (!silent) setIsInitialLoad(false);
    }
  }, []);

  const loadCommentary = useCallback(async () => {
    try {
      const data = await fetchCommentary(DEFAULT_COPPER_SYMBOL);
      setCommentary(data);
    } catch (err) {
      console.error(err);
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
        if (typeof payload.price === 'number') {
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

  const { forecastChartData, yDomain, lastHistDate } = useMemo(() => {
    const all = history?.data || [];
    if (all.length === 0) return { forecastChartData: [] as any[], yDomain: [0, 10] as [number, number], lastHistDate: '' };

    // Filter out unclosed/invalid bars (e.g. today's incomplete bar with null price)
    // to prevent gaps in the chart and ensure the bridge point has a valid number.
    const validHistory = all.filter((p: any) => p.price != null && !isNaN(p.price));
    if (validHistory.length === 0) return { forecastChartData: [] as any[], yDomain: [0, 10] as [number, number], lastHistDate: '' };

    const recent = validHistory.slice(-30);
    const last = recent[recent.length - 1];

    const hist = recent.slice(0, -1).map((p: any) => ({ date: p.date, price: p.price }));

    const hasForecast = !!tftAnalysis?.prediction?.daily_forecasts?.length;

    const bridge: any = {
      date: last.date,
      price: last.price,
      ...(hasForecast && {
        priceMedian: last.price,
        priceQ10: last.price,
        priceQ90: last.price,
      }),
    };

    const forecasts = hasForecast
      ? tftAnalysis!.prediction.daily_forecasts.slice(0, 1).map((fc: any) => {
          const d = addBusinessDays(new Date(last.date), fc.day);
          return {
            date: d.toISOString().split('T')[0],
            priceMedian: fc.price_median,
            priceQ10: fc.price_q10,
            priceQ90: fc.price_q90,
            isForecast: true as const,
          };
        })
      : [];

    const data = [...hist, bridge, ...forecasts];

    let min = Infinity, max = -Infinity;
    for (const p of data) {
      if (p.price != null) { min = Math.min(min, p.price); max = Math.max(max, p.price); }
      if ('priceQ10' in p && p.priceQ10 != null) { min = Math.min(min, p.priceQ10); }
      if ('priceQ90' in p && p.priceQ90 != null) { max = Math.max(max, p.priceQ90); }
      if ('priceMedian' in p && p.priceMedian != null) {
        min = Math.min(min, p.priceMedian);
        max = Math.max(max, p.priceMedian);
      }
    }
    const pad = (max - min) * 0.05;

    return {
      forecastChartData: data,
      yDomain: [min - pad, max + pad] as [number, number],
      lastHistDate: last.date,
    };
  }, [history, tftAnalysis]);

  const theme = {
    bull: '#34D399',
    bear: '#FB7185',
    copper: '#F59E0B',
    grid: 'rgba(255,255,255,0.05)',
    text: '#9CA3AF'
  };

  // Only show full loading on initial load
  if (isInitialLoad && !analysis) {
    return (
      <div className="min-h-screen bg-midnight flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-copper-500/30 border-t-copper-500 rounded-full animate-spin" />
          <span className="text-copper-500 font-mono text-sm tracking-widest">INITIALIZING...</span>
        </div>
      </div>
    );
  }

  // CRITICAL: headline percentage MUST come from the same place as the T+1
  // price shown beside it. Previously we read `predicted_return_median`
  // while the price came from `daily_forecasts[0].price_median`, which
  // could diverge once the backend applied a display clamp. Sourcing both
  // from `daily_forecasts[0]` makes them guaranteed-consistent.
  const t1Forecast = tftAnalysis?.prediction?.daily_forecasts?.[0];
  const tftReturn = t1Forecast?.daily_return ?? tftAnalysis?.prediction?.predicted_return_median ?? null;
  const tftBullish = tftReturn !== null ? tftReturn >= 0 : null;
  const tftMetrics = tftAnalysis?.model_metadata?.metrics;
  const tftDirection = tftAnalysis?.direction;
  const tftWeeklyTrend = tftAnalysis?.weekly_trend;
  const tftReferencePrice = tftAnalysis?.prediction?.reference_price;
  const tftReferenceDate = tftAnalysis?.prediction?.reference_price_date;
  const tftAnomaly = tftAnalysis?.prediction?.anomaly_detected;
  const tftInstrument = tftAnalysis?.prediction?.instrument;
  const tftStalenessDays = tftAnalysis?.prediction?.baseline_staleness_days ?? 0;
  // Anything >= 3 calendar days is flagged; 0-2 is considered fresh (weekend).
  const tftBaselineIsStale = tftStalenessDays >= 3;
  const newsSentimentIndex = sentimentSummary.data?.index ?? 0;
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
  const quoteDelta = quotePrice != null && latestHistoryPrice != null
    ? quotePrice - latestHistoryPrice
    : null;
  const quoteDeltaPct = quoteDelta != null && latestHistoryPrice
    ? (quoteDelta / latestHistoryPrice) * 100
    : null;

  return (
    <div className="font-sans selection:bg-copper-500/30">

      {/* Background Gradient Mesh - kept for overview visual flavor */}
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-copper-500/10 via-transparent to-transparent pointer-events-none" />

      <div className="relative z-10 grid gap-8">

        {/* Header */}
        <header className="flex justify-between items-end pb-8 border-b border-white/5">
          <div className="space-y-1">
            <motion.h1
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              className="text-4xl font-bold text-white tracking-tight"
            >
              Terra Rara
            </motion.h1>
            <p className="text-gray-500 text-sm tracking-widest font-mono uppercase">Copper Intelligence Platform</p>
          </div>

          <div className="flex bg-slate-900 rounded-lg p-1.5 border border-slate-800 gap-1 shadow-sm">
            <div className="px-3 py-2 rounded-xl bg-black min-w-[360px] overflow-hidden border border-slate-800">
              <div className="flex items-center gap-3">
                <div className="w-11 h-11 rounded-md flex items-center justify-center shrink-0">
                  <img
                    src="https://s3-symbol-logo.tradingview.com/metal/copper--big.svg"
                    alt="Copper logo"
                    className="w-full h-full object-contain"
                    loading="lazy"
                  />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">
                    Copper Futures
                  </p>
                  <div className="mt-1 flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded-md border border-slate-700 bg-slate-900 text-[11px] text-white font-semibold tracking-wide">
                      {COPPER_INSTRUMENT.canonicalSymbol}
                    </span>
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" title="Live price feed active" />
                  </div>
                  <div className="mt-1 flex items-baseline gap-2 font-mono">
                    <span className="text-3xl text-white leading-none">
                      {quotePrice != null ? quotePrice.toFixed(4) : '--'}
                    </span>
                    <span className="text-sm text-slate-400">USD</span>
                    {quoteDelta != null && quoteDeltaPct != null && (
                      <span className={clsx("text-xl leading-none", quoteDelta >= 0 ? "text-emerald-400" : "text-rose-400")}>
                        {quoteDelta >= 0 ? '+' : ''}{quoteDelta.toFixed(2)} {quoteDelta >= 0 ? '+' : ''}{quoteDeltaPct.toFixed(2)}%
                      </span>
                    )}
                  </div>
                  <p className="mt-0.5 text-[11px] text-slate-500">
                    {lastLiveUpdateAt
                      ? `As of ${lastLiveUpdateAt.toLocaleDateString()} ${lastLiveUpdateAt.toLocaleTimeString()}`
                      : 'Waiting for latest quote'}
                  </p>
                </div>
              </div>
            </div>
            <div className="px-4 py-2 rounded-xl bg-midnight/50 flex flex-col items-end min-w-[120px]">
              <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">7D News Sentiment</span>
              <div className={clsx("mt-1 inline-flex items-center gap-1.5 px-2 py-1 rounded-md border text-[11px] font-semibold", newsSentimentMeta.chip)}>
                <SentimentIcon size={12} />
                <span>{newsSentimentMeta.label}</span>
              </div>
              <div className={clsx("mt-1 font-mono text-xs", newsSentimentMeta.tone)}>
                <NumberTicker value={newsSentimentIndex} format={(v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(3)}`} />
              </div>
            </div>
          </div>
        </header>

        {/* Dashboard Grid + persistent News sidebar (desktop).
            On mobile/tablet the news panel stacks under the dashboard.
            Width grows with the viewport so chips/filters have room to breathe. */}
        <div className="grid gap-4 lg:gap-6 lg:grid-cols-[minmax(0,1fr)_340px] xl:grid-cols-[minmax(0,1fr)_380px] 2xl:grid-cols-[minmax(0,1fr)_420px]">
        {/* Main dashboard column */}
        <div className="grid grid-cols-12 gap-6">

          {/* Deep Learning Forecast — primary T+1 forecast */}
          <GlassCard title="Deep Learning Forecast" icon={Brain} colSpan={4} className={clsx("relative overflow-hidden", tftBullish === null ? "" : tftBullish ? "border-emerald-500/30" : "border-rose-500/30")}>
            {tftAnalysis ? (() => {
              const t1 = tftAnalysis.prediction.daily_forecasts?.[0];
              return (
                <>
                  <div className="absolute top-0 right-0 p-4 opacity-5">
                    {tftBullish ? <TrendingUp size={100} /> : <TrendingDown size={100} />}
                  </div>
                  <div className="relative z-10 space-y-4">

                    {/* T+1 Direction badge */}
                    <div className="flex items-center gap-2 flex-wrap">
                      <div className={clsx(
                        "inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-sm font-bold tracking-wide",
                        tftDirection === 'BULLISH' ? "bg-emerald-400/10 text-emerald-400 border border-emerald-400/20" :
                        tftDirection === 'BEARISH' ? "bg-rose-400/10 text-rose-400 border border-rose-400/20" :
                                                     "bg-amber-400/10 text-amber-400 border border-amber-400/20"
                      )}>
                        {tftDirection === 'BULLISH' ? <TrendingUp size={14} /> : tftDirection === 'BEARISH' ? <TrendingDown size={14} /> : <Activity size={14} />}
                        {tftDirection}
                      </div>
                    </div>

                    {/* Next session headline — percent and price derive from
                        the same forecast entry (single source of truth). */}
                    <div>
                      <div className="mb-1 flex items-center justify-between gap-3">
                        <span className="text-[10px] text-gray-500 uppercase tracking-widest">
                          T+1 Forecast
                        </span>
                        {tftBaselineIsStale && (
                          <span
                            title={`Last PriceBar is ${tftStalenessDays}d old; lazy ingest attempted on next forecast request.`}
                            className="px-1.5 py-0.5 rounded border border-amber-500/40 bg-amber-500/10 text-amber-300 text-[9px] tracking-wider"
                          >
                            Stale {tftStalenessDays}d
                          </span>
                        )}
                      </div>
                      <div className="flex items-baseline gap-2">
                        <span className={clsx("text-3xl font-light font-mono", tftBullish ? "text-emerald-400" : "text-rose-400")}>
                          {tftBullish ? '+' : ''}{((tftReturn ?? 0) * 100).toFixed(2)}%
                        </span>
                        <span className="text-sm text-gray-400 font-mono">${t1?.price_median?.toFixed(2) ?? '--'}</span>
                        {tftReferencePrice && (
                          <span className="text-[10px] text-gray-600 font-mono">
                            (from ${tftReferencePrice.toFixed(2)})
                          </span>
                        )}
                      </div>
                      <div className="mt-2 grid grid-cols-2 gap-2 text-[10px] text-gray-500">
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
                        <p className="mt-1 text-[10px] text-amber-400">
                          Anomalous raw model output; value bounded to +/-12%. Check training logs.
                        </p>
                      )}
                    </div>

                    {/* T+1 expected range */}
                    <div className="rounded-lg bg-white/[0.02] border border-white/5 px-3 py-2">
                      <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-1.5">Expected Range (80% confidence)</p>
                      <div className="flex items-center justify-between">
                        <div className="text-center">
                          <p className="text-[10px] text-gray-600 mb-0.5">Low</p>
                          <span className="text-sm font-mono text-rose-400/80">${t1?.price_q10?.toFixed(2) ?? '--'}</span>
                        </div>
                        <div className="flex-1 mx-3 h-1.5 rounded-full bg-white/5 relative overflow-hidden">
                          <div className="absolute inset-0 bg-gradient-to-r from-rose-500/40 via-gray-500/20 to-emerald-500/40 rounded-full" />
                        </div>
                        <div className="text-center">
                          <p className="text-[10px] text-gray-600 mb-0.5">High</p>
                          <span className="text-sm font-mono text-emerald-400/80">${t1?.price_q90?.toFixed(2) ?? '--'}</span>
                        </div>
                      </div>
                    </div>

                    {/* Weekly trend — direction only, no price targets */}
                    <div className="flex items-center justify-between py-2 border-t border-white/5">
                      <span className="text-[10px] text-gray-500 uppercase tracking-wider">Week Trend</span>
                      <div className="flex items-center gap-1.5">
                        {tftWeeklyTrend === 'BULLISH' ? <TrendingUp size={12} className="text-emerald-400" /> :
                         tftWeeklyTrend === 'BEARISH' ? <TrendingDown size={12} className="text-rose-400" /> :
                         <Activity size={12} className="text-amber-400" />}
                        <span className={clsx("text-xs font-bold tracking-wide",
                          tftWeeklyTrend === 'BULLISH' ? "text-emerald-400" :
                          tftWeeklyTrend === 'BEARISH' ? "text-rose-400" : "text-amber-400"
                        )}>
                          {tftWeeklyTrend}
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div className="rounded bg-white/[0.02] border border-white/5 px-2 py-1.5">
                        <span className="block text-[9px] text-gray-600 uppercase tracking-wider">Direction Score</span>
                        <span className="font-mono text-xs text-gray-200">
                          {tftMetrics?.directional_accuracy != null ? `${(tftMetrics.directional_accuracy * 100).toFixed(1)}%` : '--'}
                        </span>
                      </div>
                      <div className="rounded bg-white/[0.02] border border-white/5 px-2 py-1.5 text-right">
                        <span className="block text-[9px] text-gray-600 uppercase tracking-wider">Sharpe</span>
                        <span className="font-mono text-xs text-gray-200">
                          {tftMetrics?.sharpe_ratio != null ? tftMetrics.sharpe_ratio.toFixed(2) : '--'}
                        </span>
                      </div>
                    </div>

                    {/* Risk */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        {tftAnalysis.risk_level === 'LOW'
                          ? <CheckCircle2 size={13} className="text-emerald-400" />
                          : tftAnalysis.risk_level === 'MEDIUM'
                          ? <AlertTriangle size={13} className="text-amber-400" />
                          : <AlertTriangle size={13} className="text-rose-400" />}
                        <span className={clsx("text-xs font-medium",
                          tftAnalysis.risk_level === 'LOW' ? "text-emerald-400" :
                          tftAnalysis.risk_level === 'MEDIUM' ? "text-amber-400" : "text-rose-400"
                        )}>
                          {tftAnalysis.risk_level} RISK
                        </span>
                      </div>
                    </div>
                  </div>
                </>
              );
            })() : (
              <div className="flex flex-col items-center justify-center h-full py-10 text-center gap-3">
                <Brain size={32} className="text-gray-700" />
                <div>
                  <p className="text-xs text-gray-500">Model not trained yet</p>
                  <p className="text-[10px] text-gray-700 mt-1">Run the TFT training workflow</p>
                </div>
              </div>
            )}
          </GlassCard>

          {/* Price Forecast Chart */}
          <GlassCard title={`Price Forecast (${COPPER_INSTRUMENT.canonicalSymbol})`} icon={Activity} colSpan={8} className="min-h-[400px]">
            {forecastChartData.length > 0 ? (
              <div className="h-[350px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={forecastChartData} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={theme.copper} stopOpacity={0.2} />
                        <stop offset="95%" stopColor={theme.copper} stopOpacity={0} />
                      </linearGradient>
                    </defs>

                    <CartesianGrid stroke={theme.grid} vertical={false} strokeDasharray="4 4" />
                    <XAxis
                      dataKey="date"
                      tick={{ fill: theme.text, fontSize: 10, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}
                      tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' })}
                      axisLine={false}
                      tickLine={false}
                      interval="preserveStartEnd"
                    />
                    <YAxis
                      orientation="right"
                      domain={yDomain}
                      tick={{ fill: theme.text, fontSize: 10, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(val) => `$${val.toFixed(2)}`}
                      width={60}
                    />
                    <Tooltip content={<ForecastTooltip />} />

                    {/* Historical price area */}
                    <Area
                      type="monotone"
                      dataKey="price"
                      stroke={theme.copper}
                      fill="url(#priceGradient)"
                      strokeWidth={2}
                      connectNulls={false}
                    />

                    {/* Q10-Q90 confidence band (80%) — lower edge */}
                    <Line type="linear" dataKey="priceQ10" stroke="#8B5CF6" strokeWidth={1} strokeDasharray="3 4" strokeOpacity={0.4} dot={false} connectNulls={false} />
                    {/* Q10-Q90 confidence band (80%) — upper edge */}
                    <Line type="linear" dataKey="priceQ90" stroke="#8B5CF6" strokeWidth={1} strokeDasharray="3 4" strokeOpacity={0.4} dot={false} connectNulls={false} />

                    {/* Forecast median line */}
                    <Line
                      type="linear"
                      dataKey="priceMedian"
                      stroke="#8B5CF6"
                      strokeWidth={2}
                      strokeDasharray="6 3"
                      dot={false}
                      connectNulls={false}
                    />

                    {/* "Today" divider */}
                    {lastHistDate && (
                      <ReferenceLine
                        x={lastHistDate}
                        stroke="rgba(255,255,255,0.15)"
                        strokeDasharray="3 3"
                        label={{ value: 'Today', position: 'top', fill: '#6B7280', fontSize: 9, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}
                      />
                    )}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <ChartSkeleton />
            )}
          </GlassCard>

          {/* Influencers Card — shows human-readable labels, category chips and
              technical ids on hover. Backend contract: Influencer has
              `label`, `description`, `category`, `time_horizon`. */}
          <GlassCard title="Market Drivers" icon={BarChart3} colSpan={4}>
            <div className="space-y-4">
              {analysis?.top_influencers.slice(0, 5).map((inf: any, i: number) => {
                const maxImp = analysis.top_influencers[0]?.importance || 1;
                const label = inf.label || inf.description || inf.feature;
                const categoryTone: Record<string, string> = {
                  Momentum:   'bg-copper-500/15 text-copper-300',
                  Trend:      'bg-blue-500/15 text-blue-300',
                  Volatility: 'bg-amber-500/15 text-amber-300',
                  Sentiment:  'bg-violet-500/15 text-violet-300',
                  Macro:      'bg-emerald-500/15 text-emerald-300',
                  Sector:     'bg-rose-500/15 text-rose-300',
                  Embedding:  'bg-slate-500/15 text-slate-300',
                  Other:      'bg-white/5 text-gray-400',
                };
                return (
                  <div key={inf.feature} className="group" title={inf.feature}>
                    <div className="flex justify-between items-start mb-1 gap-2">
                      <div className="flex items-start gap-2 min-w-0 flex-1">
                        {inf.category && (
                          <span className={`text-[9px] px-1.5 py-0.5 rounded ${categoryTone[inf.category] || categoryTone.Other} font-medium uppercase tracking-wider shrink-0`}>
                            {inf.category}
                          </span>
                        )}
                        <span className="text-xs text-gray-300 group-hover:text-copper-400 transition-colors whitespace-normal break-words leading-snug">
                          {label}
                        </span>
                        {inf.time_horizon && (
                          <span className="text-[9px] font-mono text-gray-500 shrink-0">{inf.time_horizon}</span>
                        )}
                      </div>
                      <span className="text-xs font-mono text-gray-500 shrink-0">{(inf.importance * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-copper-500 to-rose-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${(inf.importance / maxImp) * 100}%` }}
                        transition={{ delay: 0.2 + (i * 0.1), duration: 0.8 }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </GlassCard>

          {/* Model Health Card */}
          <GlassCard title="Model Reliability" icon={Crosshair} colSpan={4}>
            {tftMetrics ? (() => {
              const da = (tftMetrics.directional_accuracy ?? 0) * 100;
              const sharpe = tftMetrics.sharpe_ratio ?? 0;

              const daGood = da >= 52;
              const sharpeGood = sharpe >= 0;

              const overallGood = (daGood ? 1 : 0) + (sharpeGood ? 1 : 0);
              const overallLabel = overallGood === 2 ? 'HEALTHY' : overallGood === 1 ? 'FAIR' : 'CALIBRATING';
              const overallColor = overallGood === 2 ? 'text-emerald-400' : overallGood === 1 ? 'text-amber-400' : 'text-rose-400';

              return (
                <div className="space-y-4">
                  {/* Overall status */}
                  <div className="flex items-center justify-between pb-3 border-b border-white/5">
                    <span className="text-xs text-gray-500">Overall Status</span>
                    <span className={clsx("text-xs font-bold tracking-wider", overallColor)}>{overallLabel}</span>
                  </div>

                  {/* Direction accuracy */}
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {daGood ? <CheckCircle2 size={13} className="text-emerald-400" /> : <AlertTriangle size={13} className="text-rose-400" />}
                        <span className="text-xs text-gray-400">Direction Accuracy</span>
                      </div>
                      <span className={clsx("text-xs font-mono font-medium", daGood ? "text-emerald-400" : "text-rose-400")}>{da.toFixed(1)}%</span>
                    </div>
                    <ProgressBar value={da} max={100} color={daGood ? "bg-emerald-500" : "bg-rose-500"} />
                    <p className="text-[10px] text-gray-600">
                      {da >= 55 ? "Strong directional signal" : da >= 50 ? "Beats coin flip" : "Below random — still learning"}
                    </p>
                  </div>

                  {/* Strategy performance */}
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {sharpeGood ? <CheckCircle2 size={13} className="text-emerald-400" /> : <AlertTriangle size={13} className="text-rose-400" />}
                        <span className="text-xs text-gray-400">Strategy Performance</span>
                      </div>
                      <span className={clsx("text-xs font-mono font-medium", sharpeGood ? "text-emerald-400" : "text-rose-400")}>
                        Sharpe {sharpe >= 0 ? '+' : ''}{sharpe.toFixed(2)}
                      </span>
                    </div>
                    <ProgressBar value={Math.min(Math.abs(sharpe) * 50, 100)} max={100} color={sharpeGood ? "bg-emerald-500" : "bg-rose-500"} />
                    <p className="text-[10px] text-gray-600">
                      {sharpe > 1 ? "Strong risk-adjusted returns" : sharpe > 0 ? "Positive expected return" : "Negative — do not trade"}
                    </p>
                  </div>
                </div>
              );
            })() : (
              <div className="flex flex-col items-center justify-center py-10 text-center gap-3">
                <Clock size={28} className="text-gray-700" />
                <div>
                  <p className="text-xs text-gray-500">No training data yet</p>
                  <p className="text-[10px] text-gray-700 mt-1">Run pipeline with train_model=true</p>
                </div>
              </div>
            )}
          </GlassCard>

          {/* AI Commentary Card */}
          <GlassCard title="Neural Analysis" icon={Cpu} colSpan={4}>
            <div className="flex items-center justify-between mb-3">
              {commentary?.generated_at && (
                <span className="text-[10px] text-gray-600 font-mono">
                  {new Date(commentary.generated_at).toLocaleTimeString()}
                </span>
              )}
            </div>
            <div className="h-[140px] overflow-y-auto text-sm text-gray-300 leading-relaxed custom-scrollbar">
              {commentary ? (
                <p className="font-light whitespace-pre-wrap">{commentary.commentary || ''}</p>
              ) : (
                <span className="text-gray-500 animate-pulse">Processing market signals...</span>
              )}
            </div>
          </GlassCard>

          {/* CopperMind Universe Map - Lazy loaded */}
          <div className="col-span-12">
            <Suspense fallback={<MapSkeleton />}>
              <HeatmapPanel />
            </Suspense>
          </div>

        </div>
        {/* Right sticky News Intelligence sidebar (desktop) / stacks under on mobile */}
        <aside className="lg:sticky lg:top-24 lg:self-start lg:max-h-[calc(100vh-120px)] min-h-[480px]">
          <Suspense
            fallback={
              <div className="glass-panel h-full min-h-[480px] flex items-center justify-center">
                <span className="text-xs text-gray-500 font-mono tracking-widest uppercase">
                  Loading news…
                </span>
              </div>
            }
          >
            <NewsIntelligencePanel />
          </Suspense>
        </aside>
        </div>
      </div>
    </div>
  );
}

