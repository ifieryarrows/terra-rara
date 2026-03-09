import { useEffect, useState, useCallback, useMemo, Suspense, lazy, useRef } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceDot, ReferenceLine, ComposedChart, Line
} from 'recharts';
import { motion } from 'framer-motion';
import {
  Activity, Globe, Zap, BarChart3, Cpu, TrendingUp, TrendingDown,
  Brain, Crosshair, AlertTriangle, CheckCircle2, Clock
} from 'lucide-react';
import clsx from 'clsx';
import { SpeedInsights } from '@vercel/speed-insights/react';

import { fetchAnalysis, fetchHistory, fetchCommentary, fetchTFTAnalysis } from './api';
import type {
  AnalysisReport, HistoryResponse, HistoryDataPoint,
  CommentaryResponse, TFTAnalysisResponse, TFTDailyForecast
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

// 5-day forecast sparkline with uncertainty band
const TFTSparkline = ({ forecasts, currentPrice }: { forecasts: TFTDailyForecast[], currentPrice: number }) => {
  if (!forecasts?.length) return null;
  const lastMedian = forecasts[forecasts.length - 1].price_median;
  const isBull = lastMedian >= currentPrice;
  const color = isBull ? '#34D399' : '#FB7185';

  const raw = [
    { label: 'Now', med: currentPrice, base: currentPrice, spread: 0 },
    ...forecasts.map(fc => ({
      label: `T+${fc.day}`,
      med: fc.price_median,
      base: fc.price_q10,
      spread: fc.price_q90 - fc.price_q10,
    })),
  ];

  const allVals = [currentPrice, ...forecasts.flatMap(fc => [fc.price_q10, fc.price_q90])];
  const lo = Math.min(...allVals);
  const hi = Math.max(...allVals);
  const pad = (hi - lo) * 0.6 || 0.05;

  return (
    <ResponsiveContainer width="100%" height={72}>
      <ComposedChart data={raw} margin={{ top: 6, right: 2, bottom: 0, left: 2 }}>
        <defs>
          <linearGradient id="tftBand" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.3} />
            <stop offset="100%" stopColor={color} stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <YAxis domain={[lo - pad, hi + pad]} hide />
        {/* Transparent base positions band at Q10 */}
        <Area type="monotone" dataKey="base" stackId="b" fill="transparent" stroke="none" isAnimationActive={false} legendType="none" />
        {/* Colored spread from Q10 to Q90 */}
        <Area type="monotone" dataKey="spread" stackId="b" fill="url(#tftBand)" stroke="none" legendType="none" />
        {/* Median price line */}
        <Line type="monotone" dataKey="med" stroke={color} strokeWidth={2} dot={false} legendType="none" />
      </ComposedChart>
    </ResponsiveContainer>
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
          <GlassCard title="Deep Learning Forecast" icon={Brain} colSpan={3} className={clsx("relative overflow-hidden", tftBullish === null ? "" : tftBullish ? "shadow-glow-emerald" : "shadow-glow-rose")}>
            {tftAnalysis ? (() => {
              const t1 = tftAnalysis.prediction.daily_forecasts?.[0];
              const t5 = tftAnalysis.prediction.daily_forecasts?.[4];
              const currentPrice = tftAnalysis.prediction.predicted_price_median / (1 + (tftReturn ?? 0));
              return (
                <>
                  <div className="absolute top-0 right-0 p-4 opacity-5">
                    {tftBullish ? <TrendingUp size={100} /> : <TrendingDown size={100} />}
                  </div>
                  <div className="relative z-10 space-y-3">

                    {/* Direction badge */}
                    <div className={clsx(
                      "inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-sm font-bold tracking-wide",
                      tftDirection === 'BULLISH' ? "bg-emerald-400/10 text-emerald-400 border border-emerald-400/20" :
                      tftDirection === 'BEARISH' ? "bg-rose-400/10 text-rose-400 border border-rose-400/20" :
                                                   "bg-amber-400/10 text-amber-400 border border-amber-400/20"
                    )}>
                      {tftDirection === 'BULLISH' ? <TrendingUp size={14} /> : tftDirection === 'BEARISH' ? <TrendingDown size={14} /> : <Activity size={14} />}
                      {tftDirection}
                    </div>

                    {/* Tomorrow headline */}
                    <div>
                      <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-0.5">Tomorrow</p>
                      <div className="flex items-baseline gap-2">
                        <span className={clsx("text-3xl font-light font-mono", tftBullish ? "text-emerald-400" : "text-rose-400")}>
                          {tftBullish ? '+' : ''}{((tftReturn ?? 0) * 100).toFixed(2)}%
                        </span>
                        <span className="text-sm text-gray-400 font-mono">${t1?.price_median.toFixed(2)}</span>
                      </div>
                      <p className="text-[10px] text-gray-600 mt-0.5 font-mono">
                        range ${t1?.price_q10.toFixed(2)} – ${t1?.price_q90.toFixed(2)}
                      </p>
                    </div>

                    {/* 5-day sparkline */}
                    <div className="rounded-xl bg-white/[0.02] border border-white/5 px-2 pt-2 pb-1">
                      <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-1 px-1">5-Day Outlook</p>
                      <TFTSparkline forecasts={tftAnalysis.prediction.daily_forecasts} currentPrice={currentPrice} />
                      <div className="flex justify-between text-[10px] text-gray-600 font-mono px-1 mt-1">
                        <span>Now ${currentPrice.toFixed(2)}</span>
                        <span className={clsx(tftWeeklyTrend === 'BULLISH' ? "text-emerald-400/70" : "text-rose-400/70")}>
                          Fri {tftWeeklyReturn !== null ? `${tftWeeklyReturn >= 0 ? '+' : ''}${(tftWeeklyReturn * 100).toFixed(1)}%` : ''} ${t5?.price_median.toFixed(2)}
                        </span>
                      </div>
                    </div>

                    {/* Risk + model tag */}
                    <div className="flex items-center justify-between pt-1">
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
                          {tftAnalysis.risk_level} VOLATILITY
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
                    {/* TFT T+1 confidence band lines */}
                    {tftAnalysis && lastPoint && (
                      <ReferenceLine
                        y={tftAnalysis.prediction.predicted_price_q10}
                        stroke="#8B5CF6"
                        strokeDasharray="3 4"
                        strokeOpacity={0.45}
                        label={{ value: `Q10 $${tftAnalysis.prediction.predicted_price_q10.toFixed(2)}`, position: 'left', fill: '#8B5CF6', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                      />
                    )}
                    {tftAnalysis && lastPoint && (
                      <ReferenceLine
                        y={tftAnalysis.prediction.predicted_price_q90}
                        stroke="#8B5CF6"
                        strokeDasharray="3 4"
                        strokeOpacity={0.45}
                        label={{ value: `Q90 $${tftAnalysis.prediction.predicted_price_q90.toFixed(2)}`, position: 'left', fill: '#8B5CF6', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                      />
                    )}
                    {/* XGBoost T+1 prediction dot */}
                    {analysis && lastPoint && (
                      <ReferenceDot
                        x={lastPoint.date}
                        y={analysis.predicted_price}
                        r={5}
                        fill={isBullish ? theme.bull : theme.bear}
                        stroke="#0b1120"
                        strokeWidth={2}
                        label={{ value: `XGB $${analysis.predicted_price.toFixed(2)}`, position: 'right', fill: isBullish ? theme.bull : theme.bear, fontSize: 9, fontFamily: 'JetBrains Mono' }}
                      />
                    )}
                    {/* TFT median prediction dot */}
                    {tftAnalysis && lastPoint && (
                      <ReferenceDot
                        x={lastPoint.date}
                        y={tftAnalysis.prediction.predicted_price_median}
                        r={5}
                        fill="#8B5CF6"
                        stroke="#0b1120"
                        strokeWidth={2}
                        label={{ value: `TFT $${tftAnalysis.prediction.predicted_price_median.toFixed(2)}`, position: 'right', fill: '#8B5CF6', fontSize: 9, fontFamily: 'JetBrains Mono' }}
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

          {/* Model Health Card */}
          <GlassCard title="Model Reliability" icon={Crosshair} colSpan={4}>
            {tftMetrics ? (() => {
              const da = (tftMetrics.directional_accuracy ?? 0) * 100;
              const sharpe = tftMetrics.sharpe_ratio ?? 0;
              const vr = (tftMetrics.variance_ratio ?? 0) * 100;
              const predStd = (tftMetrics.pred_std ?? 0) * 100;
              const actualStd = (tftMetrics.actual_std ?? 0) * 100;

              const daGood = da >= 52;
              const sharpeGood = sharpe >= 0;
              const vrGood = vr >= 50;

              const overallGood = (daGood ? 1 : 0) + (sharpeGood ? 1 : 0) + (vrGood ? 1 : 0);
              const overallLabel = overallGood === 3 ? 'HEALTHY' : overallGood === 2 ? 'FAIR' : overallGood === 1 ? 'CALIBRATING' : 'TRAINING';
              const overallColor = overallGood === 3 ? 'text-emerald-400' : overallGood >= 2 ? 'text-amber-400' : 'text-rose-400';

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

                  {/* Forecast range */}
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {vrGood ? <CheckCircle2 size={13} className="text-emerald-400" /> : <AlertTriangle size={13} className="text-amber-400" />}
                        <span className="text-xs text-gray-400">Forecast Range</span>
                      </div>
                      <span className={clsx("text-xs font-mono font-medium", vrGood ? "text-emerald-400" : "text-amber-400")}>
                        ±{predStd.toFixed(2)}% vs ±{actualStd.toFixed(2)}%
                      </span>
                    </div>
                    <ProgressBar value={Math.min(vr, 100)} max={100} color={vrGood ? "bg-emerald-500" : "bg-amber-500"} />
                    <p className="text-[10px] text-gray-600">
                      {vr >= 80 ? "Moves match real market" : vr >= 50 ? "Reasonable amplitude" : "Predictions too conservative"}
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
