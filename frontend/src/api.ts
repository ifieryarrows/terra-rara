import axios from 'axios';
import type { AnalysisReport, HistoryResponse, HealthResponse, CommentaryResponse, TFTAnalysisResponse } from './types';

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
export async function fetchAnalysis(symbol: string = 'HG=F'): Promise<AnalysisReport> {
  const response = await api.get<AnalysisReport>('/analysis', {
    params: { symbol },
  });
  return response.data;
}

/**
 * Fetch historical price and sentiment data
 */
export async function fetchHistory(
  symbol: string = 'HG=F',
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
export async function fetchCommentary(symbol: string = 'HG=F'): Promise<CommentaryResponse> {
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
export async function fetchTFTAnalysis(symbol: string = 'HG=F'): Promise<TFTAnalysisResponse | null> {
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
 * Live price response type (Twelve Data)
 */
export interface LivePriceResponse {
  symbol: string;
  price: number | null;
  error: string | null;
}

/**
 * Fetch real-time copper price from Twelve Data
 */
export async function fetchLivePrice(): Promise<LivePriceResponse> {
  const response = await api.get<LivePriceResponse>('/live-price');
  return response.data;
}

/**
 * Fetch Consensus Signal
 */
export async function fetchConsensusSignal(symbol: string = 'HG=F'): Promise<any> {
  const response = await api.get('/analysis/consensus', { params: { symbol } });
  return response.data;
}

/**
 * Fetch TFT Model Summary
 */
export async function fetchTftModelSummary(symbol: string = 'HG=F'): Promise<any> {
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

