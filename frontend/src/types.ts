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
  ai_stance?: 'BULLISH' | 'NEUTRAL' | 'BEARISH';
}

export interface TFTDailyForecast {
  day: number;
  daily_return: number;
  cumulative_return: number;
  price_median: number;
  price_q10: number;
  price_q90: number;
  price_q02: number;
  price_q98: number;
}

export interface TFTPrediction {
  predicted_return_median: number;
  predicted_return_q10: number;
  predicted_return_q90: number;
  predicted_price_median: number;
  predicted_price_q10: number;
  predicted_price_q90: number;
  confidence_band_96: [number, number];
  volatility_estimate: number;
  quantiles: Record<string, number>;
  weekly_return: number;
  weekly_price: number;
  prediction_horizon_days: number;
  daily_forecasts: TFTDailyForecast[];
}

export interface TFTModelMetadata {
  symbol: string;
  trained_at: string | null;
  checkpoint_path: string | null;
  config: Record<string, any>;
  metrics: {
    mae?: number;
    rmse?: number;
    directional_accuracy?: number;
    sharpe_ratio?: number;
    sortino_ratio?: number;
    variance_ratio?: number;
    pred_std?: number;
    actual_std?: number;
    tail_capture_rate?: number;
  };
}

export interface TFTAnalysisResponse {
  symbol: string;
  model_type: string;
  direction: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  weekly_trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW';
  prediction: TFTPrediction;
  model_metadata: TFTModelMetadata | null;
  generated_at: string;
}
