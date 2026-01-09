import axios from 'axios';
import type { AnalysisReport, HistoryResponse, HealthResponse, CommentaryResponse } from './types';

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

export default api;
