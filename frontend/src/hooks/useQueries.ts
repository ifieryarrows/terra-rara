import { useQuery } from '@tanstack/react-query';
import { 
  fetchHealth, 
  fetchConsensusSignal, 
  fetchTftModelSummary, 
  fetchLatestBacktest, 
  fetchMarketHeatmap,
  fetchAnalysis,
  fetchHistory,
  fetchTFTAnalysis,
  fetchCommentary
} from '../api';

export function useSystemStatus() {
  return useQuery({
    queryKey: ['system-status'],
    queryFn: fetchHealth,
    refetchInterval: 60000, // 1 minute
  });
}

export function useConsensusSignal(symbol: string = 'HG=F') {
  return useQuery({
    queryKey: ['consensus-signal', symbol],
    queryFn: () => fetchConsensusSignal(symbol),
    refetchInterval: 300000, // 5 minutes
  });
}

export function useTftModelSummary(symbol: string = 'HG=F') {
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

export function useMarketHeatmap() {
  return useQuery({
    queryKey: ['market-heatmap'],
    queryFn: fetchMarketHeatmap,
    // Polling handled by React Query instead of setInterval
    refetchInterval: 900000, // 15 minutes exact
    refetchOnWindowFocus: false,
  });
}

// Additional wrappers for Overview
export function useOverviewData(symbol: string = 'HG=F') {
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
