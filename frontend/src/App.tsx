import { useEffect, useState, useCallback, useMemo, Suspense, lazy, useRef } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceDot, ReferenceLine, ReferenceArea
} from 'recharts';
import { motion } from 'framer-motion';
import {
  Activity, Globe, Zap, BarChart3, Cpu, TrendingUp, TrendingDown,
  Brain, Target, Shield, BarChart2
} from 'lucide-react';
import clsx from 'clsx';
import { SpeedInsights } from '@vercel/speed-insights/react';

import { fetchAnalysis, fetchHistory, fetchCommentary, fetchTFTAnalysis } from './api';
import type {
  AnalysisReport, HistoryResponse, HistoryDataPoint,
  CommentaryResponse, TFTAnalysisResponse
} from './types';
import './App.css';

// Lazy load heavy components
const MarketMap = lazy(() => import('./components/MarketMap').then(m => ({ default: m.MarketMap })));

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

// --- Main App ---

function App() {
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

  const chartData: HistoryDataPoint[] = useMemo(() => history?.data || [], [history]);
  const lastPoint = chartData[chartData.length - 1];

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

  const tftReturn = tftAnalysis?.prediction?.predicted_return_median ?? null;  // T+1
  const tftWeeklyReturn = tftAnalysis?.prediction?.weekly_return ?? null;     // T+5
  const tftBullish = tftReturn !== null ? tftReturn >= 0 : null;
  const tftMetrics = tftAnalysis?.model_metadata?.metrics;
  const tftDirection = tftAnalysis?.direction;           // T+1 based
  const tftWeeklyTrend = tftAnalysis?.weekly_trend;      // T+5 based

  return (
    <div className="min-h-screen bg-midnight text-gray-100 p-8 font-sans selection:bg-copper-500/30">

      {/* Background Gradient Mesh */}
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-copper-500/10 via-midnight to-midnight pointer-events-none" />

      <div className="max-w-7xl mx-auto relative z-10 grid gap-8">

        {/* Header */}
        <header className="flex justify-between items-end pb-8 border-b border-white/5">
          <div className="space-y-1">
            <motion.h1
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              className="text-4xl font-bold bg-gradient-to-r from-gray-100 to-gray-400 bg-clip-text text-transparent"
            >
              Terra Rara
            </motion.h1>
            <p className="text-gray-500 text-sm tracking-widest font-mono uppercase">Copper Intelligence Platform</p>
          </div>

          <div className="flex bg-white/5 backdrop-blur-md rounded-2xl p-1.5 border border-white/5 gap-1">
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
          <GlassCard title="XGBoost (T+1)" icon={Zap} colSpan={3} className={clsx("relative overflow-hidden", isBullish ? "shadow-glow-emerald" : "shadow-glow-rose")}>
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

          {/* TFT-ASRO Prediction Card */}
          <GlassCard title="TFT-ASRO" icon={Brain} colSpan={3} className={clsx("relative overflow-hidden", tftBullish === null ? "" : tftBullish ? "shadow-glow-emerald" : "shadow-glow-rose")}>
            {tftAnalysis ? (
              <>
                <div className="absolute top-0 right-0 p-4 opacity-10">
                  {tftBullish ? <TrendingUp size={80} /> : <TrendingDown size={80} />}
                </div>
                <div className="relative z-10 flex flex-col h-full justify-between py-1">
                  <div>
                    {/* T+1 headline (most reliable signal) */}
                    <div className="flex items-baseline gap-2">
                      <span className={clsx("text-4xl font-light font-mono tracking-tighter", tftBullish ? "text-emerald-400" : "text-rose-400")}>
                        {tftBullish ? '+' : ''}<NumberTicker value={(tftReturn || 0) * 100} />%
                      </span>
                      <span className="text-xs text-gray-500 font-mono">T+1</span>
                    </div>

                    {/* Weekly trend summary */}
                    <div className="mt-2 flex items-center gap-2">
                      <span className="text-[10px] text-gray-500 uppercase tracking-wider">5D Trend</span>
                      <span className={clsx("font-mono text-xs font-medium",
                        tftWeeklyTrend === 'BULLISH' ? "text-emerald-400" :
                        tftWeeklyTrend === 'BEARISH' ? "text-rose-400" : "text-amber-400"
                      )}>
                        {tftWeeklyReturn !== null ? `${tftWeeklyReturn >= 0 ? '+' : ''}${(tftWeeklyReturn * 100).toFixed(2)}%` : '—'}
                      </span>
                      <span className={clsx("text-[10px] font-bold",
                        tftWeeklyTrend === 'BULLISH' ? "text-emerald-400" :
                        tftWeeklyTrend === 'BEARISH' ? "text-rose-400" : "text-amber-400"
                      )}>
                        {tftWeeklyTrend}
                      </span>
                    </div>

                    {/* Daily forecast mini-table */}
                    <div className="mt-3 border border-white/5 rounded-lg overflow-hidden">
                      <div className="grid grid-cols-4 text-[10px] font-bold text-gray-500 uppercase tracking-wider bg-white/[0.02] px-2 py-1.5">
                        <span>Day</span>
                        <span className="text-right">Return</span>
                        <span className="text-right">Price</span>
                        <span className="text-right">Band</span>
                      </div>
                      {tftAnalysis.prediction.daily_forecasts?.map((fc) => {
                        const dayBull = fc.return_median >= 0;
                        return (
                          <div key={fc.day} className={clsx(
                            "grid grid-cols-4 text-[11px] font-mono px-2 py-1 border-t border-white/5 transition-colors",
                            fc.day === 1 ? "bg-white/[0.03]" : "hover:bg-white/[0.02]"
                          )}>
                            <span className={clsx("text-gray-400", fc.day === 1 && "text-gray-200 font-medium")}>T+{fc.day}</span>
                            <span className={clsx("text-right", dayBull ? "text-emerald-400" : "text-rose-400")}>
                              {dayBull ? '+' : ''}{(fc.return_median * 100).toFixed(2)}%
                            </span>
                            <span className="text-right text-gray-300">${fc.price_median.toFixed(2)}</span>
                            <span className="text-right text-gray-500 text-[10px]">
                              {fc.price_q10.toFixed(2)}–{fc.price_q90.toFixed(2)}
                            </span>
                          </div>
                        );
                      })}
                    </div>

                    <div className="mt-2 flex justify-between text-xs py-1.5">
                      <span className="text-gray-500">Risk</span>
                      <span className={clsx("font-mono text-xs font-bold",
                        tftAnalysis.risk_level === 'LOW' ? "text-emerald-400" :
                        tftAnalysis.risk_level === 'MEDIUM' ? "text-amber-400" : "text-rose-400"
                      )}>
                        {tftAnalysis.risk_level}
                      </span>
                    </div>
                  </div>

                  <div className="mt-3 pt-3 border-t border-white/5">
                    <div className="flex items-center gap-2">
                      <div className={clsx(
                        "w-2 h-2 rounded-full animate-pulse",
                        tftDirection === 'BULLISH' ? "bg-emerald-400" :
                          tftDirection === 'BEARISH' ? "bg-rose-400" : "bg-amber-400"
                      )} />
                      <span className={clsx(
                        "text-[10px] font-bold tracking-wider",
                        tftDirection === 'BULLISH' ? "text-emerald-400" :
                          tftDirection === 'BEARISH' ? "text-rose-400" : "text-amber-400"
                      )}>
                        T+1: {tftDirection}
                      </span>
                      <span className="ml-auto px-2 py-0.5 rounded text-[10px] font-bold bg-violet-500/10 text-violet-400 border border-violet-500/20">TFT</span>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-full py-8 text-center">
                <Brain size={32} className="text-gray-600 mb-3" />
                <span className="text-xs text-gray-500">TFT model not available yet</span>
              </div>
            )}
          </GlassCard>

          {/* Chart Card */}
          <GlassCard title="Market Flow" icon={Activity} colSpan={6} className="min-h-[400px]">
            {chartData.length > 0 ? (
              <div className="h-[350px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={theme.copper} stopOpacity={0.2} />
                        <stop offset="95%" stopColor={theme.copper} stopOpacity={0} />
                      </linearGradient>
                    </defs>

                    <CartesianGrid stroke={theme.grid} vertical={false} strokeDasharray="4 4" />
                    <XAxis
                      dataKey="date"
                      tick={{ fill: theme.text, fontSize: 10, fontFamily: 'JetBrains Mono' }}
                      tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' })}
                      axisLine={false}
                      tickLine={false}
                      interval="preserveStartEnd"
                    />
                    <YAxis
                      orientation="right"
                      domain={['auto', 'auto']}
                      tick={{ fill: theme.text, fontSize: 10, fontFamily: 'JetBrains Mono' }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(val) => `$${val.toFixed(2)}`}
                      width={60}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(11, 17, 32, 0.8)',
                        backdropFilter: 'blur(8px)',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderRadius: '12px',
                        color: '#F3F4F6'
                      }}
                      itemStyle={{ fontFamily: 'JetBrains Mono', fontSize: '12px' }}
                      labelStyle={{ fontFamily: 'Plus Jakarta Sans', color: '#9CA3AF', marginBottom: '8px' }}
                    />
                    <Area
                      type="monotone"
                      dataKey="price"
                      stroke={theme.copper}
                      fill="url(#priceGradient)"
                      strokeWidth={2}
                    />
                    {/* TFT confidence band (Q10–Q90) */}
                    {tftAnalysis && lastPoint && (
                      <ReferenceArea
                        x1={lastPoint.date}
                        x2={lastPoint.date}
                        y1={tftAnalysis.prediction.predicted_price_q10}
                        y2={tftAnalysis.prediction.predicted_price_q90}
                        fill="#8B5CF6"
                        fillOpacity={0.15}
                        strokeOpacity={0}
                      />
                    )}
                    {tftAnalysis && lastPoint && (
                      <ReferenceLine
                        y={tftAnalysis.prediction.predicted_price_q10}
                        stroke="#8B5CF6"
                        strokeDasharray="4 4"
                        strokeOpacity={0.3}
                      />
                    )}
                    {tftAnalysis && lastPoint && (
                      <ReferenceLine
                        y={tftAnalysis.prediction.predicted_price_q90}
                        stroke="#8B5CF6"
                        strokeDasharray="4 4"
                        strokeOpacity={0.3}
                      />
                    )}
                    {/* XGBoost prediction dot */}
                    {analysis && lastPoint && (
                      <ReferenceDot
                        x={lastPoint.date}
                        y={analysis.predicted_price}
                        r={4}
                        fill={isBullish ? theme.bull : theme.bear}
                        stroke="#fff"
                        strokeWidth={2}
                      />
                    )}
                    {/* TFT median prediction dot */}
                    {tftAnalysis && lastPoint && (
                      <ReferenceDot
                        x={lastPoint.date}
                        y={tftAnalysis.prediction.predicted_price_median}
                        r={4}
                        fill="#8B5CF6"
                        stroke="#fff"
                        strokeWidth={2}
                      />
                    )}
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <ChartSkeleton />
            )}
          </GlassCard>

          {/* Influencers Card */}
          <GlassCard title="Market Drivers" icon={BarChart3} colSpan={4}>
            <div className="space-y-4">
              {analysis?.top_influencers.slice(0, 5).map((inf, i) => (
                <div key={inf.feature} className="group">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-gray-400 group-hover:text-copper-400 transition-colors uppercase tracking-wide">
                      {inf.feature.replace(/_/g, ' ')}
                    </span>
                    <span className="text-xs font-mono text-gray-500">{(inf.importance * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-copper-500 to-rose-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${(inf.importance / analysis.top_influencers[0].importance) * 100}%` }}
                      transition={{ delay: 0.2 + (i * 0.1), duration: 0.8 }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </GlassCard>

          {/* TFT Model Metrics Card */}
          <GlassCard title="Deep Learning Metrics" icon={BarChart2} colSpan={4}>
            {tftMetrics ? (
              <div className="space-y-3">
                {[
                  { label: 'Directional Accuracy', value: tftMetrics.directional_accuracy, fmt: (v: number) => `${(v * 100).toFixed(1)}%`, threshold: 0.5, icon: Target },
                  { label: 'Sharpe Ratio', value: tftMetrics.sharpe_ratio, fmt: (v: number) => v.toFixed(2), threshold: 0, icon: Zap },
                  { label: 'Variance Ratio', value: tftMetrics.variance_ratio, fmt: (v: number) => v.toFixed(2), threshold: 0.5, icon: Activity },
                  { label: 'Tail Capture', value: tftMetrics.tail_capture_rate, fmt: (v: number) => `${(v * 100).toFixed(1)}%`, threshold: 0.5, icon: Shield },
                ].map(({ label, value, fmt, threshold, icon: MetricIcon }) => {
                  const v = value ?? 0;
                  const good = v >= threshold;
                  return (
                    <div key={label} className="flex items-center justify-between py-2 border-b border-white/5 last:border-0">
                      <div className="flex items-center gap-2">
                        <MetricIcon size={14} className="text-gray-500" />
                        <span className="text-xs text-gray-400">{label}</span>
                      </div>
                      <span className={clsx("font-mono text-sm font-medium", good ? "text-emerald-400" : "text-rose-400")}>
                        {fmt(v)}
                      </span>
                    </div>
                  );
                })}
                <div className="pt-2 flex items-center justify-between text-[10px] text-gray-600">
                  <span>pred_std: {tftMetrics.pred_std?.toFixed(4)}</span>
                  <span>actual_std: {tftMetrics.actual_std?.toFixed(4)}</span>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <BarChart2 size={28} className="text-gray-600 mb-2" />
                <span className="text-xs text-gray-500">Metrics unavailable</span>
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
                <p className="font-light">{commentary.commentary || ''}</p>
              ) : (
                <span className="text-gray-500 animate-pulse">Processing market signals...</span>
              )}
            </div>
          </GlassCard>

          {/* Market Map - Lazy loaded */}
          <div className="col-span-12">
            <GlassCard title="Global Intelligence Map" icon={Globe} colSpan={12}>
              <Suspense fallback={<MapSkeleton />}>
                <MarketMap />
              </Suspense>
            </GlassCard>
          </div>

        </div>
      </div>
      <SpeedInsights />
    </div>
  );
}

export default App;
