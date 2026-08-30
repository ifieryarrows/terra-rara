import { useQuery } from '@tanstack/react-query';
import { 
  fetchHealth, 
  fetchConsensusSignal, 
  fetchTftModelSummary, 
  fetchLatestBacktest, 
  fetchMarketHeatmap,
  fetchHeatmapCategoryContext,
  fetchAnalysis,
  fetchHistory,
  fetchTFTAnalysis,
  fetchCommentary,
  fetchSentimentSummary,
} from '../api';
import { DEFAULT_COPPER_SYMBOL } from '../config/instruments';

export function useSystemStatus() {
  return useQuery({
    queryKey: ['system-status'],
    queryFn: fetchHealth,
    refetchInterval: 60000, // 1 minute
  });
}

export function useConsensusSignal(symbol: string = DEFAULT_COPPER_SYMBOL) {
  return useQuery({
    queryKey: ['consensus-signal', symbol],
    queryFn: () => fetchConsensusSignal(symbol),
    refetchInterval: 300000, // 5 minutes
  });
}

export function useTftModelSummary(symbol: string = DEFAULT_COPPER_SYMBOL) {
  return useQuery({
    queryKey: ['tft-summary', symbol],
    queryFn: () => fetchTftModelSummary(symbol),
    // Models don't change often, 10 min refresh or manual
    refetchInterval: 600000, 
  });
}

export function useBacktestReport() {
  return useQuery({
    queryKey: ['backtest-report'],
    queryFn: fetchLatestBacktest,
    refetchInterval: false, // Manual refresh or very long TTL
  });
}

export function useMarketHeatmap(view: 'market' | 'themes' = 'market') {
  return useQuery({
    queryKey: ['market-heatmap', view],
    queryFn: () => fetchMarketHeatmap(view),
    refetchInterval: (query) => {
      const data: any = query.state.data;
      const meta = data?._meta;
      const count = meta?.payload_count ?? (data?.children?.length ? -1 : 0);
      if (meta?.refresh_in_progress) return 5_000;
      const nextDelay = meta?.next_refresh_at
        ? Math.max(1_000, new Date(meta.next_refresh_at).getTime() - Date.now())
        : null;
      if (count <= 0) return nextDelay ?? 3_000;
      return nextDelay ?? 900_000;
    },
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    staleTime: 900_000,
    gcTime: 1_800_000,
  });
}

export function useHeatmapCategoryContext(
  categoryId: string,
  view: 'market' | 'themes',
  enabled: boolean,
) {
  return useQuery({
    queryKey: ['market-heatmap-context', view, categoryId],
    queryFn: () => fetchHeatmapCategoryContext(categoryId, view),
    enabled: enabled && !!categoryId,
    staleTime: 300_000,
    gcTime: 900_000,
    retry: 1,
    refetchOnWindowFocus: false,
  });
}

/**
 * Stable hybrid sentiment summary (DB-backed, no LLM on the hot path).
 * Refreshes every 2 minutes to match pipeline cadence without spamming.
 */
export function useSentimentSummary(days: number = 7, recentLimit: number = 6) {
  return useQuery({
    queryKey: ['sentiment-summary', days, recentLimit],
    queryFn: () => fetchSentimentSummary(days, recentLimit),
    refetchInterval: 120_000,
    refetchOnWindowFocus: false,
    staleTime: 60_000,
  });
}

// Additional wrappers for Overview
export function useOverviewData(symbol: string = DEFAULT_COPPER_SYMBOL) {
  const analysisQuery = useQuery({
    queryKey: ['analysis', symbol],
    queryFn: () => fetchAnalysis(symbol),
    refetchInterval: 60000,
  });

  const historyQuery = useQuery({
    queryKey: ['history', symbol],
    queryFn: () => fetchHistory(symbol, 180),
    refetchInterval: 300000,
  });

  const tftAnalysisQuery = useQuery({
    queryKey: ['tft-analysis', symbol],
    queryFn: () => fetchTFTAnalysis(symbol),
    refetchInterval: 60000,
  });
  
  const commentaryQuery = useQuery({
    queryKey: ['commentary', symbol],
    queryFn: () => fetchCommentary(symbol),
    enabled: !!analysisQuery.data,
    refetchInterval: 60000,
  });

  return {
    analysis: analysisQuery,
    history: historyQuery,
    tftAnalysis: tftAnalysisQuery,
    commentary: commentaryQuery,
  };
}
