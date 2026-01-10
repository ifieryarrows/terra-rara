import { useEffect, useState, useCallback, useMemo } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot
} from 'recharts';
import { motion } from 'framer-motion';
import { Activity, Globe, Zap, BarChart3, RefreshCw, Cpu, TrendingUp, TrendingDown } from 'lucide-react';
import clsx from 'clsx'; // Utility for conditional classes

import { fetchAnalysis, fetchHistory, fetchCommentary } from './api';
import { MarketMap } from './components/MarketMap';
import type {
  AnalysisReport, HistoryResponse, HistoryDataPoint,
  LoadingState, CommentaryResponse
} from './types';
import './App.css';

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
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [commentary, setCommentary] = useState<CommentaryResponse | null>(null);
  const [loadingState, setLoadingState] = useState<LoadingState>('idle');

  const loadData = useCallback(async () => {
    setLoadingState('loading');
    try {
      const [analysisData, historyData] = await Promise.all([
        fetchAnalysis('HG=F'),
        fetchHistory('HG=F', 180),
      ]);
      setAnalysis(analysisData);
      setHistory(historyData);
      setLoadingState('success');
    } catch {
      setLoadingState('error');
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

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 60000);
    return () => clearInterval(interval);
  }, [loadData]);

  useEffect(() => {
    if (loadingState === 'success') loadCommentary();
  }, [loadingState, loadCommentary]);

  const chartData: HistoryDataPoint[] = useMemo(() => history?.data || [], [history]);
  const lastPoint = chartData[chartData.length - 1];

  const theme = {
    bull: '#34D399', // Emerald 400
    bear: '#FB7185', // Rose 400
    copper: '#F59E0B', // Amber 500
    grid: 'rgba(255,255,255,0.05)',
    text: '#9CA3AF'
  };

  if (loadingState === 'loading' || loadingState === 'idle') {
    return (
      <div className="min-h-screen bg-midnight flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-copper-500/30 border-t-copper-500 rounded-full animate-spin" />
          <span className="text-copper-500 font-mono text-sm tracking-widest animate-pulse">SYNCHRONIZING...</span>
        </div>
      </div>
    );
  }

  const isBullish = analysis && analysis.predicted_return >= 0;

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
            <div className="px-4 py-2 rounded-xl bg-midnight/50 flex flex-col items-end min-w-[120px]">
              <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">HG=F Price</span>
              <div className="text-copper-400 font-mono text-lg font-light">
                $<NumberTicker value={analysis?.current_price || 0} />
              </div>
            </div>
            <div className="px-4 py-2 rounded-xl bg-midnight/50 flex flex-col items-end min-w-[120px]">
              <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Sentiment</span>
              <div className={clsx("font-mono text-lg font-light", isBullish ? "text-emerald-400" : "text-rose-400")}>
                <NumberTicker value={analysis?.sentiment_index || 0} format={(v: number) => v.toFixed(3)} />
              </div>
            </div>
          </div>
        </header>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-12 gap-6">

          {/* Prediction Card */}
          <GlassCard title="Model Forecast (T+1)" icon={Zap} colSpan={4} className={clsx("relative overflow-hidden", isBullish ? "shadow-glow-emerald" : "shadow-glow-rose")}>
            <div className="absolute top-0 right-0 p-6 opacity-10">
              {isBullish ? <TrendingUp size={120} /> : <TrendingDown size={120} />}
            </div>

            <div className="relative z-10 flex flex-col h-full justify-between py-2">
              <div>
                <div className="flex items-baseline gap-2">
                  <span className={clsx("text-5xl font-light font-mono tracking-tighter", isBullish ? "text-emerald-400" : "text-rose-400")}>
                    {isBullish ? '+' : ''}<NumberTicker value={(analysis?.predicted_return || 0) * 100} />%
                  </span>
                </div>
                <div className="mt-4 space-y-2">
                  <div className="flex justify-between text-sm py-2 border-b border-white/5">
                    <span className="text-gray-500">Target Close</span>
                    <span className="font-mono text-gray-200">${analysis?.predicted_price.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between text-sm py-2 border-b border-white/5">
                    <span className="text-gray-500">Confidence</span>
                    <span className="font-mono text-gray-200">{(analysis?.data_quality.coverage_pct || 0)}%</span>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-white/5">
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <div className={clsx("w-2 h-2 rounded-full", isBullish ? "bg-emerald-400 animate-pulse" : "bg-rose-400 animate-pulse")} />
                  AI CONVICTION: {isBullish ? 'BULLISH' : 'BEARISH'}
                </div>
              </div>
            </div>
          </GlassCard>

          {/* Chart Card */}
          <GlassCard title="Market Flow" icon={Activity} colSpan={8} className="min-h-[400px]">
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
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </GlassCard>

          {/* Influencers Card */}
          <GlassCard title="Market Drivers" icon={BarChart3} colSpan={6}>
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

          {/* AI Commentary Card */}
          <GlassCard title="Neural Analysis" icon={Cpu} colSpan={6} className="text-sm leading-relaxed text-gray-300">
            <div className="h-[240px] overflow-y-auto pr-2 custom-scrollbar">
              {commentary ? (
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-4">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-copper-500/10 text-copper-500 border border-copper-500/20">MIMO-V3</span>
                    <span className="text-xs text-gray-600 font-mono">
                      {commentary.generated_at ? new Date(commentary.generated_at).toLocaleTimeString() : ''}
                    </span>
                  </div>
                  {(commentary.commentary || '').split('\n').map((p, i) => (
                    <p key={i} className="font-light">{p}</p>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-500 gap-2">
                  <RefreshCw className="animate-spin" size={16} /> Processing signals...
                </div>
              )}
            </div>
          </GlassCard>

          {/* Market Map */}
          <div className="col-span-12">
            <GlassCard title="Global Intelligence Map" icon={Globe} colSpan={12}>
              <MarketMap />
            </GlassCard>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
