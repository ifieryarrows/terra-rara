/**
 * TypeScript type definitions matching backend Pydantic schemas
 */

export interface Influencer {
  feature: string;
  importance: number;
  label?: string;
  description?: string;
  category?: string;
  time_horizon?: string;
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
  redis_ok?: boolean | null;
  last_snapshot_age_seconds?: number | null;
  /** Worker pipeline run timestamps (see HealthResponse docstring). */
  last_pipeline_run_at?: string | null;
  last_pipeline_status?: 'ok' | 'running' | 'failed' | 'stale' | string | null;
  last_snapshot_generated_at?: string | null;
  last_tft_prediction_at?: string | null;
  tft_model_trained_at?: string | null;
  tft_reference_price_date?: string | null;
  price_bar_latest_date?: string | null;
  price_bar_staleness_days?: number | null;
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
  raw_daily_return?: number;
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
  /** Explicit contract: the close used as the basis for all returns/prices */
  reference_price?: number;
  reference_price_date?: string | null;
  return_basis?: string;
  raw_predicted_return_median?: number;
  anomaly_detected?: boolean;
  /** Freshness metadata (added 2026-04). */
  baseline_staleness_days?: number;
  lazy_ingest_triggered?: boolean;
  instrument?: {
    symbol: string;
    /** "futures" | "spot" | "unknown" */
    kind: string;
    name: string;
    note?: string;
  };
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
  /** Where the payload came from. Added 2026-04 alongside daily TFT snapshot persistence. */
  source?: 'snapshot' | 'live';
  /** Only populated when `source === 'snapshot'` — when the worker persisted the row. */
  snapshot_generated_at?: string | null;
}

export interface ConsensusSignal {
  consensus_direction: string;
  confidence: string;
  position_scale: number;
  blended_return: number;
  xgb_return_raw: number;
  xgb_return_adjusted: number;
  tft_return: number;
  xgb_direction: number;
  tft_direction: number;
}

export interface QualityGateResponse {
  passed: boolean;
  reasons: string[];
  metrics: Record<string, number>;
}

export interface TFTModelSummaryResponse {
  symbol: string;
  trained_at: string | null;
  checkpoint_path: string;
  config: Record<string, any>;
  metrics: Record<string, number>;
  variable_importance: Influencer[];
  quality_gate: QualityGateResponse | null;
}

export interface BacktestReportResponse {
  report_date: string;
  summary_metrics: Record<string, any>;
  window_metrics: Array<Record<string, any>>;
  theta_comparison: Record<string, any> | null;
  verdict: string | null;
}
