import axios from 'axios';
import type {
  AnalysisReport,
  HistoryResponse,
  HealthResponse,
  CommentaryResponse,
  TFTAnalysisResponse,
  NewsFeedFilters,
  NewsItem,
  NewsListResponse,
  NewsStatsResponse,
} from './types';
import { DEFAULT_COPPER_SYMBOL } from './config/instruments';

// Base URL for API calls
// In production (Vercel), use VITE_API_URL env var pointing to Hugging Face backend
// In development, uses Vite proxy to localhost:8000
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 90000, // 90 seconds - HF Space cold start can take a while
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status;
    const message = error.response?.data?.detail || error.message;

    console.error(`[API Error] ${status}: ${message}`);

    // Create a more user-friendly error
    const userError = new Error(
      status === 503
        ? 'Pipeline is currently running. Please try again in a few minutes.'
        : status === 404
          ? 'Data not available. Please run the pipeline first (make seed && make train).'
          : `API Error: ${message}`
    );

    return Promise.reject(userError);
  }
);

/**
 * Fetch current analysis report
 */
export async function fetchAnalysis(symbol: string = DEFAULT_COPPER_SYMBOL): Promise<AnalysisReport> {
  const response = await api.get<AnalysisReport>('/analysis', {
    params: { symbol },
  });
  return response.data;
}

/**
 * Fetch historical price and sentiment data
 */
export async function fetchHistory(
  symbol: string = DEFAULT_COPPER_SYMBOL,
  days: number = 180
): Promise<HistoryResponse> {
  const response = await api.get<HistoryResponse>('/history', {
    params: { symbol, days },
  });
  return response.data;
}

/**
 * Health check
 */
export async function fetchHealth(): Promise<HealthResponse> {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
}

/**
 * Fetch AI-generated market commentary
 */
export async function fetchCommentary(symbol: string = DEFAULT_COPPER_SYMBOL): Promise<CommentaryResponse> {
  const response = await api.get<CommentaryResponse>('/commentary', {
    params: { symbol },
  });
  return response.data;
}

/**
 * Market prices response type
 */
export interface MarketPricesResponse {
  symbols: Record<string, {
    price: number | null;
    change: number | null;
    date: string | null;
  }>;
}

/**
 * Fetch market prices for all tracked symbols
 */
export async function fetchMarketPrices(): Promise<MarketPricesResponse> {
  const response = await api.get<MarketPricesResponse>('/market-prices');
  return response.data;
}

/**
 * Fetch TFT-ASRO deep learning analysis
 */
export async function fetchTFTAnalysis(symbol: string = DEFAULT_COPPER_SYMBOL): Promise<TFTAnalysisResponse | null> {
  try {
    const response = await api.get<TFTAnalysisResponse>(`/analysis/tft/${symbol}`);
    return response.data;
  } catch {
    // TFT model may not be available yet — return null instead of throwing
    return null;
  }
}

export default api;

/**
 * Live price response type (canonical Yahoo/COMEX futures)
 */
export interface LivePriceResponse {
  symbol: string;
  price: number | null;
  error: string | null;
}

/**
 * Fetch current canonical copper futures price.
 */
export async function fetchLivePrice(): Promise<LivePriceResponse> {
  const response = await api.get<LivePriceResponse>('/live-price');
  return response.data;
}

/**
 * Fetch Consensus Signal
 */
export async function fetchConsensusSignal(symbol: string = DEFAULT_COPPER_SYMBOL): Promise<any> {
  const response = await api.get('/analysis/consensus', { params: { symbol } });
  return response.data;
}

/**
 * Fetch TFT Model Summary
 */
export async function fetchTftModelSummary(symbol: string = DEFAULT_COPPER_SYMBOL): Promise<any> {
  const response = await api.get('/models/tft/summary', { params: { symbol } });
  return response.data;
}

/**
 * Fetch Latest Backtest Report
 */
export async function fetchLatestBacktest(): Promise<any> {
  const response = await api.get('/models/tft/backtest/latest');
  return response.data;
}

/**
 * Fetch Market Heatmap
 */
export async function fetchMarketHeatmap(): Promise<any> {
  const response = await api.get('/market-heatmap');
  return response.data;
}

/**
 * Sentiment summary types — match the `/api/sentiment/summary` contract.
 */
export interface SentimentSummaryComponents {
  llm_impact_avg: number | null;
  finbert_pn_avg: number | null;
  rule_sign_avg: number | null;
  avg_confidence: number | null;
  avg_relevance: number | null;
  sample_size: number;
}

export interface SentimentTrendPoint {
  date: string | null;
  index: number;
  news_count: number;
}

export interface SentimentRecentArticle {
  title: string;
  source: string | null;
  url: string | null;
  published_at: string | null;
  sentiment: {
    label: string | null;
    final_score: number | null;
    relevance: number | null;
    confidence: number | null;
    event_type: string | null;
  } | null;
}

export interface SentimentSummary {
  index: number;
  label: 'Bullish' | 'Bearish' | 'Neutral';
  source: 'daily_v2' | 'rolling_v2' | 'legacy_v1' | 'none';
  components: SentimentSummaryComponents;
  trend: SentimentTrendPoint[];
  recent_articles: SentimentRecentArticle[];
  data_freshness: {
    newest: string | null;
    oldest: string | null;
    age_hours: number | null;
    article_count_24h: number;
    window_start?: string | null;
    window_days?: number;
    article_count_window?: number;
  };
  generated_at: string;
}

/**
 * Fetch stable sentiment summary (no LLM on hot path).
 */
export async function fetchSentimentSummary(days = 7, recentLimit = 6): Promise<SentimentSummary> {
  const response = await api.get<SentimentSummary>('/sentiment/summary', {
    params: { days, recent_limit: recentLimit },
  });
  return response.data;
}

// =============================================================================
// News Intelligence feed (mirrors /api/news endpoints)
// =============================================================================

function toNewsParams(filters: NewsFeedFilters = {}): Record<string, string | number> {
  const params: Record<string, string | number> = {
    limit: filters.limit ?? 20,
    offset: filters.offset ?? 0,
    since_hours: filters.since_hours ?? 168,
    label: filters.label ?? 'all',
    event_type: filters.event_type ?? 'all',
    min_relevance: filters.min_relevance ?? 0,
    channel: filters.channel ?? 'all',
  };
  if (filters.publisher && filters.publisher.trim()) {
    params.publisher = filters.publisher.trim();
  }
  if (filters.search && filters.search.trim()) {
    params.search = filters.search.trim();
  }
  return params;
}

/**
 * Fetch the paginated news feed. Backend caches at 60s TTL.
 */
export async function fetchNews(filters: NewsFeedFilters = {}): Promise<NewsListResponse> {
  const response = await api.get<NewsListResponse>('/news', { params: toNewsParams(filters) });
  return response.data;
}

/**
 * Fetch a single article with full sentiment detail.
 */
export async function fetchNewsById(processedId: number): Promise<NewsItem> {
  const response = await api.get<NewsItem>(`/news/${processedId}`);
  return response.data;
}

/**
 * Fetch aggregate stats used by the NewsIntelligencePanel header.
 */
export async function fetchNewsStats(sinceHours = 168): Promise<NewsStatsResponse> {
  const response = await api.get<NewsStatsResponse>('/news/stats', {
    params: { since_hours: sinceHours },
  });
  return response.data;
}

