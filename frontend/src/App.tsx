import { useEffect, useState, useCallback, useMemo } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot
} from 'recharts';
import { motion } from 'framer-motion';
import { Activity, Globe, Zap, BarChart3, RefreshCw, Cpu } from 'lucide-react';

import { fetchAnalysis, fetchHistory, fetchCommentary } from './api';
import { MarketMap } from './components/MarketMap';
import type {
  AnalysisReport, HistoryResponse, HistoryDataPoint,
  LoadingState, CommentaryResponse
} from './types';
import './App.css';

// --- HUD Components ---

const HudCard = ({ title, icon: Icon, children, className = '', colSpan = 1 }: any) => (
  <motion.div
    className={`ind-card ${className}`}
    style={{ gridColumn: `span ${colSpan}` }}
    initial={{ opacity: 0, scale: 0.95 }}
    animate={{ opacity: 1, scale: 1 }}
    transition={{ duration: 0.4 }}
  >
    <div className="card-header">
      <div className="card-title">
        {Icon && <Icon size={14} />}
        {title}
      </div>
      <div className="corner-bracket" />
    </div>
    <div className="card-content">
      {children}
    </div>
  </motion.div>
);

const NumberTicker = ({ value, format = (v: number) => v.toFixed(2), className = '' }: any) => {
  return (
    <motion.span
      key={value}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={className}
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
    const interval = setInterval(loadData, 60000); // Live refresh every minute
    return () => clearInterval(interval);
  }, [loadData]);

  useEffect(() => {
    if (loadingState === 'success') loadCommentary();
  }, [loadingState, loadCommentary]);

  const chartData: HistoryDataPoint[] = useMemo(() => history?.data || [], [history]);
  const lastPoint = chartData[chartData.length - 1];

  // Colors
  const colors = {
    price: '#B87333',
    bull: '#43B3AE',
    bear: '#C04000',
    grid: 'rgba(67, 179, 174, 0.08)',
    text: '#A1A1AA'
  };

  if (loadingState === 'loading' || loadingState === 'idle') {
    return (
      <div className="app-container">
        <div className="loading-container" style={{ minHeight: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <div className="scan-line" />
          <div style={{ color: colors.bull, fontFamily: 'var(--font-mono)' }}>INITIALIZING SYSTEM...</div>
        </div>
      </div>
    );
  }

  // Derived stats
  const isBullish = analysis && analysis.predicted_return >= 0;
  const sentimentColor = isBullish ? 'text-bull' : 'text-bear';
  const predictionColor = isBullish ? colors.bull : colors.bear;

  return (
    <div className="app-main">
      <div className="scan-line" />

      <div className="app-container">
        {/* HEADER */}
        <header className="hud-header">
          <div className="brand-section">
            <motion.h1 initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }}>
              TERRA RARA
            </motion.h1>
            <span>INTELLIGENCE PLATFORM v2.1</span>
          </div>

          <div className="market-ticker">
            <div className="ticker-item">
              <div className="ticker-label">HG=F PRICE</div>
              <div className="ticker-value text-copper">
                $<NumberTicker value={analysis?.current_price || 0} />
              </div>
            </div>
            <div className="ticker-item">
              <div className="ticker-label">SENTIMENT</div>
              <div className={`ticker-value ${sentimentColor}`}>
                <NumberTicker value={analysis?.sentiment_index || 0} format={(v: number) => v.toFixed(3)} />
              </div>
            </div>
            <div className="ticker-item">
              <div className="ticker-label">STATUS</div>
              <div className="ticker-value text-bull" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div className="dot" style={{ background: colors.bull }} /> LIVE
              </div>
            </div>
          </div>
        </header>

        {/* BENTO GRID */}
        <div className="bento-grid">

          {/* PREDICTION CARD */}
          <HudCard title="MODEL PREDICTION (T+1)" icon={Zap} colSpan={3} className={`prediction-card ${isBullish ? 'bg-bull-glow' : ''}`}>
            <div className="trend-indicator" style={{ color: predictionColor, display: 'flex', alignItems: 'center' }}>
              <span className="trend-arrow">{isBullish ? '▲' : '▼'}</span>
              <div className="big-stat">
                <NumberTicker value={Math.abs((analysis?.predicted_return || 0) * 100)} />%
              </div>
            </div>
            <div className="stat-label">
              ESTIMATED CLOSE: <span className="text-platinum" style={{ fontFamily: 'var(--font-mono)' }}>${analysis?.predicted_price.toFixed(4)}</span>
            </div>
            <div className="stat-label" style={{ marginTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.5rem' }}>
              CONFIDENCE: <span style={{ color: colors.text }}>{(analysis?.data_quality.coverage_pct || 0)}%</span>
            </div>
          </HudCard>

          {/* CHART MODULE */}
          <HudCard title="MARKET TRAJECTORY" icon={Activity} colSpan={9} className="chart-card">
            <div style={{ width: '100%', height: '100%' }}>
              <ResponsiveContainer>
                <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={colors.price} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={colors.price} stopOpacity={0} />
                    </linearGradient>
                    <mask id="gridMask">
                      <rect x="0" y="0" width="100%" height="100%" fill="white" />
                      <rect x="0" y="0" width="100%" height="100%" fill="url(#gridPattern)" />
                    </mask>
                  </defs>

                  <CartesianGrid stroke={colors.grid} vertical={false} strokeDasharray="2 2" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: colors.text, fontSize: 10, fontFamily: 'var(--font-mono)' }}
                    tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' })}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    orientation="right"
                    domain={['auto', 'auto']}
                    tick={{ fill: colors.text, fontSize: 10, fontFamily: 'var(--font-mono)' }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={(val) => `$${val.toFixed(2)}`}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#12181f', borderColor: '#333' }}
                    itemStyle={{ fontFamily: 'var(--font-mono)' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="price"
                    stroke={colors.price}
                    fill="url(#priceFill)"
                    strokeWidth={2}
                    animationDuration={2000}
                  />
                  {analysis && lastPoint && (
                    <ReferenceDot x={lastPoint.date} y={analysis.predicted_price} r={6} fill={isBullish ? colors.bull : colors.bear} stroke="#fff" />
                  )}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </HudCard>

          {/* INFLUENCERS */}
          <HudCard title="FEATURE IMPORTANCE" icon={BarChart3} colSpan={6} className="influencers-card">
            <div className="influencers-list">
              {analysis?.top_influencers.slice(0, 5).map((inf, i) => (
                <div key={inf.feature} className="bar-row">
                  <div className="bar-label">{inf.feature.replace(/_/g, ' ')}</div>
                  <div className="bar-track">
                    <motion.div
                      className="bar-fill"
                      initial={{ width: 0 }}
                      animate={{ width: `${(inf.importance / analysis.top_influencers[0].importance) * 100}%` }}
                      transition={{ delay: 0.5 + (i * 0.1), duration: 0.8 }}
                    />
                  </div>
                  <div className="bar-value">{(inf.importance * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </HudCard>

          {/* AI COMMENTARY */}
          <HudCard title="SYSTEM ANALYSIS" icon={Cpu} colSpan={6} className="commentary-card">
            <div className="commentary-scroll" style={{ maxHeight: '200px', overflowY: 'auto' }}>
              {commentary ? (
                <div>
                  <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between' }}>
                    <span className="ai-badge">MIMO-V2</span>
                    <span className="text-titanium" style={{ fontSize: '0.7rem' }}>
                      {commentary.generated_at ? new Date(commentary.generated_at).toLocaleTimeString() : '--'}
                    </span>
                  </div>
                  {(commentary.commentary || '').split('\n').map((p, i) => (
                    <p key={i} style={{ marginBottom: '0.8em' }}>{p}</p>
                  ))}
                </div>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', color: colors.text }}>
                  <RefreshCw className="spin" size={16} /> ANALYZING MARKET DATA...
                </div>
              )}
            </div>
          </HudCard>

          {/* MARKET MAP */}
          <div style={{ gridColumn: 'span 12' }}>
            <HudCard title="GLOBAL INTELLIGENCE MAP" icon={Globe} colSpan={12}>
              <MarketMap />
            </HudCard>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
