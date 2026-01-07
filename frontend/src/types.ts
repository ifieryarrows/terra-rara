/**
 * TypeScript type definitions matching backend Pydantic schemas
 */

export interface Influencer {
  feature: string;
  importance: number;
  description?: string;
}

export interface DataQuality {
  news_count_7d: number;
  missing_days: number;
  coverage_pct: number;
  language_filtered?: number;
}

export interface AnalysisReport {
  symbol: string;
  current_price: number;
  predicted_return: number;
  predicted_price: number;
  confidence_lower: number;
  confidence_upper: number;
  sentiment_index: number;
  sentiment_label: 'Bullish' | 'Bearish' | 'Neutral';
  top_influencers: Influencer[];
  data_quality: DataQuality;
  generated_at: string;
}

export interface HistoryDataPoint {
  date: string;
  price: number;
  sentiment_index: number | null;
  sentiment_news_count: number | null;
}

export interface HistoryResponse {
  symbol: string;
  data: HistoryDataPoint[];
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  db_type: string;
  models_found: number;
  pipeline_locked: boolean;
  timestamp: string;
  news_count?: number;
  price_bars_count?: number;
}

export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface CommentaryResponse {
  symbol: string;
  commentary: string | null;
  error: string | null;
  generated_at: string | null;
}
