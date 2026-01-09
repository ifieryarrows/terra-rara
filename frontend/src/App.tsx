import { useEffect, useState, useCallback } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from 'recharts';
import { fetchAnalysis, fetchHistory, fetchCommentary } from './api';
import { MarketMap } from './components/MarketMap';
import type {
  AnalysisReport,
  HistoryResponse,
  HistoryDataPoint,
  LoadingState,
  Influencer,
  CommentaryResponse,
} from './types';

// Custom tooltip component
interface TooltipProps {
  active?: boolean;
  payload?: Array<{ value: number; dataKey: string }>;
  label?: string;
  analysis?: AnalysisReport | null;
  isLastPoint?: boolean;
}

function CustomTooltip({ active, payload, label, analysis, isLastPoint }: TooltipProps) {
  if (!active || !payload || !payload.length) return null;

  const priceVal = payload.find((p) => p.dataKey === 'price')?.value;
  const sentimentVal = payload.find((p) => p.dataKey === 'sentiment_index')?.value;

  return (
    <div className="custom-tooltip">
      <div className="tooltip-date">{label}</div>
      {priceVal !== undefined && (
        <div className="tooltip-row">
          <span className="tooltip-label">Price</span>
          <span className="tooltip-value price">${priceVal.toFixed(4)}</span>
        </div>
      )}
      {sentimentVal !== undefined && sentimentVal !== null && (
        <div className="tooltip-row">
          <span className="tooltip-label">Sentiment</span>
          <span className="tooltip-value sentiment">{sentimentVal.toFixed(3)}</span>
        </div>
      )}
      {isLastPoint && analysis && (
        <>
          <div className="tooltip-divider" />
          <div className="tooltip-row">
            <span className="tooltip-label">🎯 Tomorrow</span>
            <span className="tooltip-value prediction">${analysis.predicted_price.toFixed(4)}</span>
          </div>
          <div className="tooltip-row">
            <span className="tooltip-label">Change</span>
            <span className={`tooltip-value ${analysis.predicted_return >= 0 ? 'positive' : 'negative'}`}>
              {(analysis.predicted_return * 100).toFixed(2)}%
            </span>
          </div>
        </>
      )}
    </div>
  );
}

// Feature name to human-readable description
function getFeatureDescription(feature: string): string {
  const descriptions: Record<string, string> = {
    // Sentiment features
    'sentiment__index': 'Market Sentiment Index',
    'sentiment__news_count': 'News Volume',

    // US Dollar Index (DX-Y.NYB)
    'DX_Y_NYB_price_sma_ratio': '📈 US Dollar Strength',
    'DX_Y_NYB_vol_10': '📊 USD Volatility (10d)',
    'DX_Y_NYB_ret1': '💵 US Dollar Daily Return',
    'DX_Y_NYB_lag_ret1_1': 'USD Return (1d ago)',
    'DX_Y_NYB_lag_ret1_2': 'USD Return (2d ago)',
    'DX_Y_NYB_lag_ret1_3': 'USD Return (3d ago)',
    'DX_Y_NYB_lag_ret1_5': 'USD Return (5d ago)',
    'DX_Y_NYB_SMA_5': 'USD SMA (5d)',
    'DX_Y_NYB_SMA_10': 'USD SMA (10d)',
    'DX_Y_NYB_SMA_20': 'USD SMA (20d)',
    'DX_Y_NYB_EMA_5': 'USD EMA (5d)',
    'DX_Y_NYB_EMA_10': 'USD EMA (10d)',
    'DX_Y_NYB_EMA_20': 'USD EMA (20d)',
    'DX_Y_NYB_RSI_14': 'USD Momentum (RSI)',

    // Copper (HG=F)
    'HG_F_price_sma_ratio': '🔶 Copper Trend Strength',
    'HG_F_vol_10': '📊 Copper Volatility',
    'HG_F_ret1': '🔶 Copper Daily Return',
    'HG_F_lag_ret1_1': 'Copper Return (1d ago)',
    'HG_F_lag_ret1_2': 'Copper Return (2d ago)',
    'HG_F_lag_ret1_3': 'Copper Return (3d ago)',
    'HG_F_lag_ret1_5': 'Copper Return (5d ago)',
    'HG_F_SMA_5': '🔶 Copper Short-term Trend',
    'HG_F_SMA_10': '🔶 Copper Medium Trend',
    'HG_F_SMA_20': 'Copper Long-term Trend',
    'HG_F_EMA_5': 'Copper EMA (5d)',
    'HG_F_EMA_10': '🔶 Copper EMA (10d)',
    'HG_F_EMA_20': 'Copper EMA (20d)',
    'HG_F_RSI_14': 'Copper Momentum (RSI)',

    // Crude Oil (CL=F)
    'CL_F_price_sma_ratio': '🛢️ Oil Trend Strength',
    'CL_F_vol_10': '🛢️ Oil Volatility',
    'CL_F_ret1': '🛢️ Oil Daily Return',
    'CL_F_lag_ret1_1': 'Oil Return (1d ago)',
    'CL_F_lag_ret1_2': 'Oil Return (2d ago)',
    'CL_F_lag_ret1_3': 'Oil Return (3d ago)',
    'CL_F_lag_ret1_5': 'Oil Return (5d ago)',
    'CL_F_SMA_5': 'Oil SMA (5d)',
    'CL_F_SMA_10': 'Oil SMA (10d)',
    'CL_F_SMA_20': 'Oil SMA (20d)',
    'CL_F_EMA_5': 'Oil EMA (5d)',
    'CL_F_EMA_10': 'Oil EMA (10d)',
    'CL_F_EMA_20': 'Oil EMA (20d)',
    'CL_F_RSI_14': 'Oil Momentum (RSI)',

    // China ETF (FXI)
    'FXI_price_sma_ratio': '🇨🇳 China Market Strength',
    'FXI_vol_10': '🇨🇳 China Volatility',
    'FXI_ret1': '🇨🇳 China Daily Return',
    'FXI_lag_ret1_1': 'China Return (1d ago)',
    'FXI_lag_ret1_2': 'China Return (2d ago)',
    'FXI_lag_ret1_3': 'China Return (3d ago)',
    'FXI_lag_ret1_5': 'China Return (5d ago)',
    'FXI_SMA_5': 'China SMA (5d)',
    'FXI_SMA_10': 'China SMA (10d)',
    'FXI_SMA_20': 'China SMA (20d)',
    'FXI_EMA_5': 'China EMA (5d)',
    'FXI_EMA_10': 'China EMA (10d)',
    'FXI_EMA_20': 'China EMA (20d)',
    'FXI_RSI_14': 'China Momentum (RSI)',
  };

  // Human-readable names for display
  const displayNames: Record<string, string> = {
    'DX_Y_NYB_price_sma_ratio': 'US Dollar Index',
    'DX_Y_NYB_vol_10': 'USD Volatility',
    'HG_F_SMA_5': 'Copper Trend (5d)',
    'HG_F_EMA_10': 'Copper Trend (10d)',
    'HG_F_SMA_10': 'Copper SMA (10d)',
    'HG_F_EMA_20': 'Copper EMA (20d)',
    'HG_F_lag_ret1_2': 'Copper Momentum',
    'DX_Y_NYB_lag_ret1_3': 'USD Momentum',
    'FXI_SMA_5': 'China Market',
    'DX_Y_NYB_lag_ret1_2': 'USD Movement',
  };

  return displayNames[feature] || descriptions[feature] || feature.replace(/_/g, ' ');
}

function App() {
  const [analysis, setAnalysis] = useState<AnalysisReport | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [commentary, setCommentary] = useState<CommentaryResponse | null>(null);
  const [loadingState, setLoadingState] = useState<LoadingState>('idle');
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setLoadingState('loading');
    setError(null);

    try {
      const [analysisData, historyData] = await Promise.all([
        fetchAnalysis('HG=F'),
        fetchHistory('HG=F', 180),
      ]);

      setAnalysis(analysisData);
      setHistory(historyData);
      setLoadingState('success');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load data';
      setError(message);
      setLoadingState('error');
    }
  }, []);

  // Load commentary separately (lazy, non-blocking)
  const loadCommentary = useCallback(async () => {
    try {
      const commentaryData = await fetchCommentary('HG=F');
      setCommentary(commentaryData);
    } catch (err) {
      console.error('Failed to load commentary:', err);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Load commentary after main data loads
  useEffect(() => {
    if (loadingState === 'success') {
      loadCommentary();
    }
  }, [loadingState, loadCommentary]);

  // Prepare chart data
  const chartData: HistoryDataPoint[] = history?.data || [];

  // Get the last price and prediction point
  const lastDataPoint = chartData.length > 0 ? chartData[chartData.length - 1] : null;

  // Format large numbers
  const formatPrice = (price: number) => {
    return price >= 100 ? price.toFixed(2) : price.toFixed(4);
  };

  const formatPercent = (value: number) => {
    const pct = value * 100;
    const sign = pct >= 0 ? '+' : '';
    return `${sign}${pct.toFixed(2)}%`;
  };

  // Render loading state
  if (loadingState === 'loading') {
    return (
      <div className="app">
        <header className="header">
          <div className="header-content">
            <div className="logo">
              <div className="logo-icon">Cu</div>
              <span className="logo-text">CopperMind</span>
            </div>
          </div>
        </header>
        <main className="main">
          <div className="loading-container">
            <div className="loading-spinner" />
            <p className="loading-text">Loading market intelligence...</p>
          </div>
        </main>
      </div>
    );
  }

  // Render error state
  if (loadingState === 'error') {
    return (
      <div className="app">
        <header className="header">
          <div className="header-content">
            <div className="logo">
              <div className="logo-icon">Cu</div>
              <span className="logo-text">CopperMind</span>
            </div>
          </div>
        </header>
        <main className="main">
          <div className="error-container">
            <div className="error-icon">⚠️</div>
            <h2 className="error-title">Unable to Load Data</h2>
            <p className="error-message">{error}</p>
            <button className="error-retry" onClick={loadData}>
              Try Again
            </button>
          </div>
        </main>
      </div>
    );
  }

  // Main dashboard render
  const sentimentClass = analysis?.sentiment_label?.toLowerCase() || 'neutral';
  const isPredictionPositive = (analysis?.predicted_return || 0) >= 0;

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">Cu</div>
            <span className="logo-text">CopperMind</span>
          </div>
          {analysis && (
            <div className="header-stats">
              <div className="header-stat">
                <div className="header-stat-label">Current Price</div>
                <div className="header-stat-value">
                  ${formatPrice(analysis.current_price)}
                </div>
              </div>
              <div className="header-stat">
                <div className="header-stat-label">Predicted Return</div>
                <div
                  className={`header-stat-value ${isPredictionPositive ? 'positive' : 'negative'
                    }`}
                >
                  {formatPercent(analysis.predicted_return)}
                </div>
              </div>
            </div>
          )}
        </div>
      </header>

      <main className="main">
        {/* Cards Grid */}
        <div className="cards-grid">
          {/* Predicted Price Card */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Tomorrow's Prediction</span>
              <div className="card-icon" style={{ background: 'var(--success-bg)', color: 'var(--success)' }}>
                🎯
              </div>
            </div>
            <div className="card-value">
              ${analysis ? formatPrice(analysis.predicted_price) : '—'}
            </div>
            <div className="card-subtitle">
              {analysis && (
                <>
                  <span className={isPredictionPositive ? 'positive' : 'negative'}>
                    {formatPercent(analysis.predicted_return)} from current
                  </span>
                  <br />
                  <span style={{ fontSize: '0.75rem', opacity: 0.7 }}>
                    Range: ${formatPrice(analysis.confidence_lower)} - ${formatPrice(analysis.confidence_upper)}
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Sentiment Card */}
          <div className={`card ${sentimentClass}`}>
            <div className="card-header">
              <span className="card-title">Market Sentiment</span>
              <div className="card-icon">
                {sentimentClass === 'bullish' ? '🐂' : sentimentClass === 'bearish' ? '🐻' : '➖'}
              </div>
            </div>
            <div className="card-value">
              {analysis?.sentiment_label || 'Neutral'}
            </div>
            <div className="card-subtitle">
              Index: {analysis ? analysis.sentiment_index.toFixed(3) : '—'}
            </div>
          </div>

          {/* Data Quality Card */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Data Quality</span>
              <div className="card-icon" style={{ background: 'var(--warning-bg)', color: 'var(--warning)' }}>
                📊
              </div>
            </div>
            <div className="card-value">
              {analysis?.data_quality.coverage_pct || 0}%
            </div>
            <div className="card-subtitle">
              {analysis?.data_quality.news_count_7d || 0} news articles (7d)
            </div>
          </div>

          {/* Generated At Card */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Last Updated</span>
              <div className="card-icon" style={{ background: 'var(--neutral-bg)', color: 'var(--neutral)' }}>
                🕐
              </div>
            </div>
            <div className="card-value" style={{ fontSize: '1.25rem' }}>
              {analysis
                ? new Date(analysis.generated_at).toLocaleTimeString()
                : '—'}
            </div>
            <div className="card-subtitle">
              {analysis
                ? new Date(analysis.generated_at).toLocaleDateString()
                : 'Loading...'}
            </div>
          </div>
        </div>

        {/* Chart Section */}
        <div className="chart-section">
          <div className="chart-header">
            <h2 className="chart-title">Price & Sentiment History</h2>
            <div className="chart-legend">
              <div className="legend-item">
                <span className="legend-dot price" />
                <span>Price</span>
              </div>
              <div className="legend-item">
                <span className="legend-dot sentiment" />
                <span>Sentiment</span>
              </div>
              {analysis && (
                <div className="legend-item">
                  <span className="legend-dot prediction" />
                  <span>Prediction</span>
                </div>
              )}
            </div>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartData}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                  tickLine={{ stroke: 'var(--chart-grid)' }}
                  axisLine={{ stroke: 'var(--chart-grid)' }}
                  tickFormatter={(value: string) => {
                    const date = new Date(value);
                    return `${date.getMonth() + 1}/${date.getDate()}`;
                  }}
                  interval="preserveStartEnd"
                  minTickGap={50}
                />
                <YAxis
                  yAxisId="price"
                  orientation="left"
                  tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                  tickLine={{ stroke: 'var(--chart-grid)' }}
                  axisLine={{ stroke: 'var(--chart-grid)' }}
                  tickFormatter={(value: number) => `$${value.toFixed(2)}`}
                  domain={['auto', 'auto']}
                />
                <YAxis
                  yAxisId="sentiment"
                  orientation="right"
                  tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                  tickLine={{ stroke: 'var(--chart-grid)' }}
                  axisLine={{ stroke: 'var(--chart-grid)' }}
                  tickFormatter={(value: number) => value.toFixed(2)}
                  domain={[-1, 1]}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  yAxisId="price"
                  type="monotone"
                  dataKey="price"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  fill="url(#priceGradient)"
                  dot={false}
                  activeDot={{ r: 4, fill: '#f59e0b' }}
                />
                <Area
                  yAxisId="sentiment"
                  type="monotone"
                  dataKey="sentiment_index"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  fill="url(#sentimentGradient)"
                  dot={false}
                  activeDot={{ r: 4, fill: '#3b82f6' }}
                  connectNulls
                />
                {/* Prediction point */}
                {analysis && lastDataPoint && (
                  <ReferenceDot
                    yAxisId="price"
                    x={lastDataPoint.date}
                    y={analysis.predicted_price}
                    r={8}
                    fill="#22c55e"
                    stroke="#fff"
                    strokeWidth={2}
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Prediction Summary under chart */}
          {analysis && (
            <div className="prediction-summary">
              <div className="prediction-item">
                <span className="prediction-label">🎯 Tomorrow's Prediction</span>
                <span className="prediction-value">${formatPrice(analysis.predicted_price)}</span>
              </div>
              <div className="prediction-item">
                <span className="prediction-label">Expected Change</span>
                <span className={`prediction-value ${isPredictionPositive ? 'positive' : 'negative'}`}>
                  {formatPercent(analysis.predicted_return)}
                </span>
              </div>
              <div className="prediction-item">
                <span className="prediction-label">Confidence Range</span>
                <span className="prediction-value">
                  ${formatPrice(analysis.confidence_lower)} - ${formatPrice(analysis.confidence_upper)}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Influencers Section */}
        {analysis && analysis.top_influencers.length > 0 && (
          <div className="influencers-section">
            <h2 className="influencers-title">Top Price Influencers</h2>
            <div className="influencers-list">
              {analysis.top_influencers.slice(0, 5).map((inf: Influencer, idx: number) => {
                const maxImportance = Math.max(
                  ...analysis.top_influencers.map((i: Influencer) => i.importance)
                );
                const barWidth = (inf.importance / maxImportance) * 100;

                return (
                  <div key={inf.feature} className="influencer-item">
                    <div className="influencer-rank">{idx + 1}</div>
                    <div className="influencer-info">
                      <div className="influencer-name">{getFeatureDescription(inf.feature)}</div>
                      <div className="influencer-desc">
                        {inf.description || inf.feature.replace(/_/g, ' ')}
                      </div>
                    </div>
                    <div className="influencer-bar-container">
                      <div
                        className="influencer-bar"
                        style={{ width: `${barWidth}%` }}
                      />
                    </div>
                    <div className="influencer-value">
                      {(inf.importance * 100).toFixed(1)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* AI Commentary Section */}
        <div className="commentary-section">
          <div className="commentary-header">
            <h2 className="commentary-title">🤖 AI Market Analysis</h2>
            {commentary?.generated_at && (
              <span className="commentary-timestamp">
                {new Date(commentary.generated_at).toLocaleString()}
              </span>
            )}
          </div>
          <div className="commentary-content">
            {commentary?.commentary ? (
              <div className="commentary-text">
                {commentary.commentary.split('\n').map((paragraph, idx) => (
                  <p key={idx}>{paragraph}</p>
                ))}
              </div>
            ) : commentary?.error ? (
              <div className="commentary-placeholder">
                <span className="commentary-icon">⚙️</span>
                <p>AI Commentary not available</p>
                <p className="commentary-hint">{commentary.error}</p>
              </div>
            ) : (
              <div className="commentary-loading">
                <div className="commentary-spinner" />
                <p>Generating AI analysis...</p>
              </div>
            )}
          </div>
        </div>

        {/* Market Intelligence Map */}
        <MarketMap />
      </main>

      {/* Footer */}
      <footer className="footer">
        <p className="footer-text">
          CopperMind v1.0.0 • Built with FinBERT + XGBoost •{' '}
          <a href="/api/docs" target="_blank" rel="noopener noreferrer">
            API Docs
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;

