import { useEffect, useState, useCallback, useMemo, Suspense, lazy, useRef } from 'react';
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { motion } from 'framer-motion';
import {
  Activity, Globe, Zap, BarChart3, Cpu, TrendingUp, TrendingDown,
  Brain, Crosshair, AlertTriangle, CheckCircle2, Clock
} from 'lucide-react';
import clsx from 'clsx';

import { fetchAnalysis, fetchHistory, fetchCommentary, fetchTFTAnalysis } from '../api';
import type {
  AnalysisReport, HistoryResponse,
  CommentaryResponse, TFTAnalysisResponse
} from '../types';
import '../App.css';

// Lazy load heavy components
const HeatmapPanel = lazy(() => import('../features/heatmap/HeatmapPanel').then(m => ({ default: m.HeatmapPanel })));

// --- Skeleton Components for perceived performance ---
const SkeletonPulse = ({ className = '' }: { className?: string }) => (
  <div className={clsx("animate-pulse bg-white/5 rounded", className)} />
);

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

const GlassCard = ({ title, icon: Icon, children, className = '', colSpan = 1 }: any) => (
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
);

const NumberTicker = ({ value, format = (v: number) => v.toFixed(2), className = '' }: any) => {
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
};

// Simple progress bar [0-100]
const ProgressBar = ({ value, max = 100, color = 'bg-emerald-500' }: { value: number; max?: number; color?: string }) => (
  <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
    <motion.div
      className={clsx("h-full rounded-full", color)}
      initial={{ width: 0 }}
      animate={{ width: `${Math.min(100, (value / max) * 100)}%` }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
    />
  </div>
);

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
  const tradingViewLoaded = useRef(false);

  // Silent refresh - no loading state flash after initial load
  const loadData = useCallback(async (silent = false) => {
    try {
      const [analysisData, historyData, tftData] = await Promise.all([
        fetchAnalysis('HG=F'),
        fetchHistory('HG=F', 180),
        fetchTFTAnalysis('HG=F'),
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
      const data = await fetchCommentary('HG=F');
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

  // Lazy load TradingView widget after initial render
  useEffect(() => {
    if (isInitialLoad || tradingViewLoaded.current) return;

    const timer = setTimeout(() => {
      const container = document.getElementById('tradingview-widget-container');
      if (container && !container.querySelector('script')) {
        tradingViewLoaded.current = true;

        const widgetDiv = document.createElement('div');
        widgetDiv.className = 'tradingview-widget-container__widget';
        container.appendChild(widgetDiv);

        const script = document.createElement('script');
        script.type = 'text/javascript';
        script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js';
        script.async = true;
        script.textContent = JSON.stringify({
          symbol: "EIGHTCAP:XCUUSD",
          width: "100%",
          isTransparent: true,
          colorTheme: "dark",
          locale: "en"
        });
        container.appendChild(script);
      }
    }, 100); // Small delay to prioritize main content

    return () => clearTimeout(timer);
  }, [isInitialLoad]);

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

  const isBullish = analysis && analysis.predicted_return >= 0;

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
  const tftInstrumentLabel = tftInstrument?.name || 'COMEX Copper Futures (HG=F)';
  const tftInstrumentKind = tftInstrument?.kind || 'futures';

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
            <div className="px-2 py-1 rounded-xl bg-midnight/50 min-w-[180px] overflow-hidden">
              <div id="tradingview-widget-container" className="tradingview-widget-container">
                {/* Skeleton while TradingView loads */}
                {!tradingViewLoaded.current && (
                  <div className="flex items-center gap-2 py-2">
                    <SkeletonPulse className="w-16 h-4" />
                    <SkeletonPulse className="w-12 h-6" />
                  </div>
                )}
              </div>
            </div>
            <div className="px-4 py-2 rounded-xl bg-midnight/50 flex flex-col items-end min-w-[120px]">
              <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Sentiment</span>
              <div className={clsx("font-mono text-lg font-light", (analysis?.sentiment_index || 0) >= 0 ? "text-emerald-400" : "text-rose-400")}>
                <NumberTicker value={analysis?.sentiment_index || 0} format={(v: number) => v.toFixed(3)} />
              </div>
            </div>
          </div>
        </header>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-12 gap-6">

          {/* XGBoost Prediction Card */}
          <GlassCard title="XGBoost (T+1)" icon={Zap} colSpan={3} className={clsx("relative overflow-hidden", isBullish ? "border-emerald-500/30" : "border-rose-500/30")}>
            <div className="absolute top-0 right-0 p-4 opacity-10">
              {isBullish ? <TrendingUp size={80} /> : <TrendingDown size={80} />}
            </div>
            <div className="relative z-10 flex flex-col h-full justify-between py-1">
              <div>
                <div className="flex items-baseline gap-2">
                  <span className={clsx("text-4xl font-light font-mono tracking-tighter", isBullish ? "text-emerald-400" : "text-rose-400")}>
                    {isBullish ? '+' : ''}<NumberTicker value={(analysis?.predicted_return || 0) * 100} />%
                  </span>
                </div>
                <div className="mt-3 space-y-1">
                  <div className="flex justify-between text-xs py-1.5 border-b border-white/5">
                    <span className="text-gray-500">Target</span>
                    <span className="font-mono text-gray-200">${analysis?.predicted_price?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between text-xs py-1.5 border-b border-white/5">
                    <span className="text-gray-500">Coverage</span>
                    <span className="font-mono text-gray-200">{(analysis?.data_quality.coverage_pct || 0)}%</span>
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-white/5">
                <div className="flex items-center gap-2 mb-2">
                  <div className={clsx(
                    "w-2 h-2 rounded-full animate-pulse",
                    commentary?.ai_stance === 'BULLISH' ? "bg-emerald-400" :
                      commentary?.ai_stance === 'BEARISH' ? "bg-rose-400" : "bg-amber-400"
                  )} />
                  <span className={clsx(
                    "text-[10px] font-bold tracking-wider",
                    commentary?.ai_stance === 'BULLISH' ? "text-emerald-400" :
                      commentary?.ai_stance === 'BEARISH' ? "text-rose-400" : "text-amber-400"
                  )}>
                    {commentary?.ai_stance || 'ANALYZING...'}
                  </span>
                </div>
                <div className="h-[80px] overflow-y-auto text-xs text-gray-400 leading-relaxed custom-scrollbar">
                  {commentary ? (
                    <p className="font-light">{(commentary.commentary || '').split('\n')[0]}</p>
                  ) : (
                    <span className="text-gray-500 animate-pulse">Processing...</span>
                  )}
                </div>
              </div>
            </div>
          </GlassCard>

          {/* TFT-ASRO Prediction Card — T+1 focused */}
          <GlassCard title="Deep Learning Forecast" icon={Brain} colSpan={3} className={clsx("relative overflow-hidden", tftBullish === null ? "" : tftBullish ? "border-emerald-500/30" : "border-rose-500/30")}>
            {tftAnalysis ? (() => {
              const t1 = tftAnalysis.prediction.daily_forecasts?.[0];
              return (
                <>
                  <div className="absolute top-0 right-0 p-4 opacity-5">
                    {tftBullish ? <TrendingUp size={100} /> : <TrendingDown size={100} />}
                  </div>
                  <div className="relative z-10 space-y-4">

                    {/* T+1 Direction badge */}
                    <div className={clsx(
                      "inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-sm font-bold tracking-wide",
                      tftDirection === 'BULLISH' ? "bg-emerald-400/10 text-emerald-400 border border-emerald-400/20" :
                      tftDirection === 'BEARISH' ? "bg-rose-400/10 text-rose-400 border border-rose-400/20" :
                                                   "bg-amber-400/10 text-amber-400 border border-amber-400/20"
                    )}>
                      {tftDirection === 'BULLISH' ? <TrendingUp size={14} /> : tftDirection === 'BEARISH' ? <TrendingDown size={14} /> : <Activity size={14} />}
                      {tftDirection}
                    </div>

                    {/* Next session headline — percent and price derive from
                        the same forecast entry (single source of truth). */}
                    <div>
                      <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-0.5 flex items-center gap-2 flex-wrap">
                        <span>Next Session (T+1)</span>
                        <span
                          title={tftInstrument?.note || ''}
                          className="px-1.5 py-0.5 rounded border border-slate-700 text-slate-300 text-[9px] normal-case tracking-wider"
                        >
                          {tftInstrumentKind === 'futures' ? 'Futures' : tftInstrumentKind === 'spot' ? 'Spot' : 'Instrument'}: {tftInstrument?.symbol || 'HG=F'}
                        </span>
                        {tftReferenceDate && (
                          <span className="text-gray-600 normal-case">
                            vs. {new Date(tftReferenceDate).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} close
                          </span>
                        )}
                        {tftBaselineIsStale && (
                          <span
                            title={`Last PriceBar is ${tftStalenessDays}d old — lazy ingest attempted on next forecast request.`}
                            className="px-1.5 py-0.5 rounded border border-amber-500/40 bg-amber-500/10 text-amber-300 text-[9px] normal-case tracking-wider"
                          >
                            STALE {tftStalenessDays}d
                          </span>
                        )}
                      </p>
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
                      <p className="mt-1 text-[10px] text-gray-500">
                        Baseline: <span className="text-gray-400">{tftInstrumentLabel}</span>.
                        LME spot (XCU/USD) can differ by 1–3 USD due to basis and roll.
                      </p>
                      {tftAnomaly && (
                        <p className="mt-1 text-[10px] text-amber-400">
                          ⚠ Anomalous raw model output — value bounded to ±12%. Check training logs.
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

                    {/* Risk + model tag */}
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
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-violet-500/10 text-violet-400 border border-violet-500/20">TFT-ASRO</span>
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
          <GlassCard title="Price Forecast" icon={Activity} colSpan={6} className="min-h-[400px]">
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
                    <div className="flex justify-between items-center mb-1 gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        {inf.category && (
                          <span className={`text-[9px] px-1.5 py-0.5 rounded ${categoryTone[inf.category] || categoryTone.Other} font-medium uppercase tracking-wider shrink-0`}>
                            {inf.category}
                          </span>
                        )}
                        <span className="text-xs text-gray-300 group-hover:text-copper-400 transition-colors truncate">
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
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-copper-500/10 text-copper-500 border border-copper-500/20">MIMO-V3</span>
                {commentary?.generated_at && (
                  <span className="text-[10px] text-gray-600 font-mono">
                    {new Date(commentary.generated_at).toLocaleTimeString()}
                  </span>
                )}
              </div>
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
      </div>
    </div>
  );
}

