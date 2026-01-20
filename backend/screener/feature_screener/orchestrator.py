"""
Feature Screener orchestrator.

Coordinates price fetching, evaluation, scoring, and output generation.
"""

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from screener.core.config import ScreenerConfig, load_config, compute_config_hash
from screener.core.fingerprint import compute_fingerprint
from screener.core.run_context import RunContext, DATA_PROVIDER_NOTES, get_run_date

from screener.contracts.universe import UniverseOutput
from screener.contracts.screener import (
    ScreenerOutput,
    ScreenerMeta,
    TargetInfo,
    AnalysisParameters,
    ScreenerCandidate,
    PeriodMetrics as PeriodMetricsContract,
    CandidateDecision,
    ExcludedCandidate,
    ArtifactReference,
)

from screener.feature_screener.fetcher import PriceFetcher
from screener.feature_screener.evaluator import Evaluator, CandidateEvaluation

logger = logging.getLogger(__name__)


class FeatureScreener:
    """
    Main orchestrator for correlation screening.
    """
    
    def __init__(
        self,
        config: ScreenerConfig,
        universe: UniverseOutput
    ):
        """
        Initialize screener.
        
        Args:
            config: Screener configuration
            universe: Universe to screen
        """
        self.config = config
        self.universe = universe
        self.context = RunContext(prefix="scr")
        
        # Determine OOS end date (run_date if not specified)
        self.oos_end = self.config.analysis.oos_end
        if self.oos_end is None:
            self.oos_end = date.fromisoformat(get_run_date())
        
        # Initialize fetcher
        cache_dir = Path(config.artifact_base_dir) / "raw" / "prices"
        self.fetcher = PriceFetcher(config.fetcher, cache_dir)
        
        # Storage
        self.prices: dict[str, pd.DataFrame] = {}
        self.evaluations: list[CandidateEvaluation] = []
        self.output: Optional[ScreenerOutput] = None
    
    def fetch_prices(self) -> int:
        """
        Fetch prices for target and all universe symbols.
        
        Returns:
            Number of symbols successfully fetched
        """
        logger.info("Fetching prices...")
        
        # Get list of symbols to fetch
        symbols = [self.config.target]
        symbols.extend(self.universe.get_included_tickers())
        
        # Add control symbols
        for ctrl in self.config.analysis.controls:
            if ctrl not in symbols:
                symbols.append(ctrl)
        
        # Deduplicate
        symbols = list(dict.fromkeys(symbols))
        
        # Calculate date range
        start_date = str(self.config.analysis.is_start)
        end_date = str(self.oos_end)
        
        logger.info(f"Fetching {len(symbols)} symbols from {start_date} to {end_date}")
        
        def progress_callback(current, total, symbol):
            if current % 20 == 0 or current == total:
                logger.info(f"Fetch progress: {current}/{total} ({symbol})")
        
        self.prices = self.fetcher.fetch_batch(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=self.config.filter.frequency,
            progress_callback=progress_callback
        )
        
        logger.info(f"Fetched {len(self.prices)} symbols successfully")
        
        return len(self.prices)
    
    def run_evaluation(self) -> int:
        """
        Run IS/OOS evaluation on all candidates.
        
        Returns:
            Number of candidates that passed IS retention
        """
        logger.info("Running evaluation...")
        
        # Check target exists
        target_ticker = self.config.target
        if target_ticker not in self.prices:
            raise ValueError(f"Target {target_ticker} not in fetched prices")
        
        target_df = self.prices[target_ticker]
        target_returns = target_df["returns"].dropna()
        
        # Prepare control series
        controls = {}
        for ctrl in self.config.analysis.controls:
            if ctrl in self.prices:
                controls[ctrl] = self.prices[ctrl]["returns"].dropna()
            else:
                logger.warning(f"Control {ctrl} not available")
        
        # Initialize evaluator
        evaluator = Evaluator(
            target_returns=target_returns,
            is_start=self.config.analysis.is_start,
            is_end=self.config.analysis.is_end,
            oos_start=self.config.analysis.oos_start,
            oos_end=self.oos_end,
            rolling_window=self.config.analysis.rolling_window_weeks,
            lead_lag_max=self.config.analysis.lead_lag_max_periods,
            sign_flip_epsilon=self.config.analysis.sign_flip_epsilon,
            min_is_corr_threshold=self.config.analysis.min_is_corr_threshold,
            min_obs=self.config.analysis.min_obs,
            controls=controls,
            enable_partial_corr=self.config.analysis.enable_partial_corr
        )
        
        # Prepare candidates (excluding target and controls)
        candidates = {}
        skip_tickers = {target_ticker} | set(self.config.analysis.controls)
        
        for symbol in self.universe.universe:
            if symbol.status != "included":
                continue
            
            ticker = symbol.canonical_ticker
            if ticker in skip_tickers:
                continue
            
            if ticker not in self.prices:
                continue
            
            returns = self.prices[ticker]["returns"].dropna()
            candidates[ticker] = (returns, symbol.category)
        
        logger.info(f"Evaluating {len(candidates)} candidates...")
        
        def progress_callback(current, total, ticker):
            if current % 25 == 0 or current == total:
                logger.info(f"Eval progress: {current}/{total} ({ticker})")
        
        self.evaluations = evaluator.evaluate_batch(
            candidates=candidates,
            progress_callback=progress_callback
        )
        
        passed = sum(1 for e in self.evaluations if e.passed_is_retention)
        logger.info(f"Evaluation complete: {passed}/{len(self.evaluations)} passed retention")
        
        return passed
    
    def _has_valid_oos(self, eval_result: CandidateEvaluation) -> bool:
        """Check if candidate has valid OOS data."""
        if eval_result.oos_metrics is None:
            return False
        if eval_result.oos_metrics.n_obs is None or eval_result.oos_metrics.n_obs < self.config.analysis.min_obs:
            return False
        if eval_result.oos_metrics.pearson is None:
            return False
        return True
    
    def score_and_rank(self) -> list[CandidateEvaluation]:
        """
        Score and rank candidates.
        
        Returns:
            Sorted list of evaluations (best first)
        """
        # Filter to those that passed IS retention AND have valid OOS
        passed = [
            e for e in self.evaluations 
            if e.passed_is_retention and self._has_valid_oos(e)
        ]
        
        # Score each candidate
        for eval_result in passed:
            eval_result.composite_score = self._compute_score(eval_result)
        
        # Sort by score (descending)
        passed.sort(key=lambda x: getattr(x, 'composite_score', 0), reverse=True)
        
        # Assign ranks
        for rank, eval_result in enumerate(passed, start=1):
            eval_result.rank = rank
        
        return passed
    
    def _compute_score(self, eval_result: CandidateEvaluation) -> float:
        """
        Compute composite score for ranking.
        
        Scoring formula:
        - Base: abs(IS pearson) * 0.4
        - OOS consistency: abs(OOS pearson) * 0.3 (if available)
        - Stability: (1 - rolling_std/0.3) * 0.2 (penalize high variance)
        - Sign consistency: (1 - sign_flips/5) * 0.1
        """
        score = 0.0
        
        # IS correlation (40%)
        if eval_result.is_metrics.pearson is not None:
            score += abs(eval_result.is_metrics.pearson) * 0.4
        
        # OOS correlation (30%)
        if eval_result.oos_metrics and eval_result.oos_metrics.pearson is not None:
            score += abs(eval_result.oos_metrics.pearson) * 0.3
        elif eval_result.is_metrics.pearson is not None:
            # No OOS data, use IS with penalty
            score += abs(eval_result.is_metrics.pearson) * 0.15
        
        # Stability (20%)
        if eval_result.is_metrics.rolling_corr_std is not None:
            stability = max(0, 1 - eval_result.is_metrics.rolling_corr_std / 0.3)
            score += stability * 0.2
        
        # Sign consistency (10%)
        if eval_result.is_metrics.sign_flip_count is not None:
            consistency = max(0, 1 - eval_result.is_metrics.sign_flip_count / 5)
            score += consistency * 0.1
        
        return round(score, 4)
    
    def build_output(self) -> ScreenerOutput:
        """
        Build the final output contract.
        
        Returns:
            ScreenerOutput object
        """
        # Get ranked candidates
        ranked = self.score_and_rank()
        
        # Build candidate contracts
        candidates = []
        excluded = []
        
        for eval_result in self.evaluations:
            has_valid_oos = self._has_valid_oos(eval_result)
            
            if eval_result.passed_is_retention and has_valid_oos:
                # Passed both IS and OOS - include as candidate
                candidate = self._eval_to_candidate(eval_result, include_in_model=True)
                candidates.append(candidate)
            elif eval_result.passed_is_retention and not has_valid_oos:
                # Passed IS but no valid OOS data
                if eval_result.oos_metrics is None:
                    reason = "oos_missing_data"
                elif eval_result.oos_metrics.n_obs is None or eval_result.oos_metrics.n_obs < self.config.analysis.min_obs:
                    reason = "oos_insufficient_observations"
                else:
                    reason = "oos_correlation_null"
                excluded.append(ExcludedCandidate(
                    ticker=eval_result.ticker,
                    reason=reason,
                    is_pearson=eval_result.is_metrics.pearson
                ))
            else:
                # Did not pass IS retention
                excluded.append(ExcludedCandidate(
                    ticker=eval_result.ticker,
                    reason=eval_result.error or "is_corr_below_threshold",
                    is_pearson=eval_result.is_metrics.pearson
                ))
        
        # Sort candidates by rank
        candidates.sort(key=lambda c: c.decision.rank or 999)
        
        # Build target info
        target_ticker = self.config.target
        target_df = self.prices.get(target_ticker)
        
        target_info = TargetInfo(
            ticker=target_ticker,
            first_date=str(target_df.index.min()) if target_df is not None else None,
            last_date=str(target_df.index.max()) if target_df is not None else None,
            total_weeks=len(target_df) if target_df is not None else None
        )
        
        # Build analysis parameters
        analysis_params = AnalysisParameters(
            is_start=str(self.config.analysis.is_start),
            is_end=str(self.config.analysis.is_end),
            oos_start=str(self.config.analysis.oos_start),
            oos_end=str(self.oos_end),
            rolling_window_weeks=self.config.analysis.rolling_window_weeks,
            lead_lag_max_periods=self.config.analysis.lead_lag_max_periods,
            sign_flip_epsilon=self.config.analysis.sign_flip_epsilon,
            min_is_corr_threshold=self.config.analysis.min_is_corr_threshold,
            min_obs=self.config.analysis.min_obs,
            controls=self.config.analysis.controls
        )
        
        # Build metadata
        config_hash = compute_config_hash(self.config)
        
        # Get universe fingerprint (prefer content_fingerprint if available)
        universe_fp = getattr(self.universe, 'content_fingerprint', None) or self.universe.fingerprint
        
        meta = ScreenerMeta(
            generated_at=self.context.generated_at,
            run_id=self.context.run_id,
            git_commit=self.context.git_commit,
            config_hash=config_hash,
            lib_versions=self.context.lib_versions,
            universe_version=self.universe.meta.run_id,
            universe_fingerprint=universe_fp
        )
        
        # Build content dict for DETERMINISTIC fingerprint (excludes meta)
        from screener.core.canonical import build_screener_content_dict
        
        content_dict = build_screener_content_dict(
            target=target_info.model_dump(),
            analysis_parameters=analysis_params.model_dump(),
            universe_content_fingerprint=universe_fp,
            candidates=[c.model_dump(by_alias=True) for c in candidates],
            excluded=[e.model_dump() for e in excluded]
        )
        content_fingerprint = compute_fingerprint(content_dict)
        
        # Build output dict for FULL fingerprint (includes meta)
        output_dict = {
            "meta": meta.model_dump(),
            "target": target_info.model_dump(),
            "analysis_parameters": analysis_params.model_dump(),
            "candidates": [c.model_dump(by_alias=True) for c in candidates],
            "excluded": [e.model_dump() for e in excluded]
        }
        output_fingerprint = compute_fingerprint(output_dict)
        
        self.output = ScreenerOutput(
            meta=meta,
            target=target_info,
            analysis_parameters=analysis_params,
            candidates=candidates,
            excluded=excluded,
            artifacts=[],
            content_fingerprint=content_fingerprint,
            output_fingerprint=output_fingerprint,
            fingerprint=content_fingerprint  # Deprecated, for backward compat
        )
        
        return self.output
    
    def _eval_to_candidate(
        self, 
        eval_result: CandidateEvaluation,
        include_in_model: bool = False
    ) -> ScreenerCandidate:
        """Convert CandidateEvaluation to ScreenerCandidate contract."""
        
        # Build IS metrics
        is_metrics = PeriodMetricsContract(
            pearson=eval_result.is_metrics.pearson,
            spearman=eval_result.is_metrics.spearman,
            rolling_corr_mean=eval_result.is_metrics.rolling_corr_mean,
            rolling_corr_std=eval_result.is_metrics.rolling_corr_std,
            sign_flip_count=eval_result.is_metrics.sign_flip_count,
            best_lead_lag=eval_result.is_metrics.best_lead_lag,
            best_lead_lag_corr=eval_result.is_metrics.best_lead_lag_corr,
            partial_corr=eval_result.is_metrics.partial_corr,
            passed_retention=eval_result.is_metrics.passed_retention,
            n_obs=eval_result.is_metrics.n_obs,
            first_date=eval_result.is_metrics.first_date,
            last_date=eval_result.is_metrics.last_date
        )
        
        # Build OOS metrics if available
        oos_metrics = None
        if eval_result.oos_metrics:
            oos_metrics = PeriodMetricsContract(
                pearson=eval_result.oos_metrics.pearson,
                spearman=eval_result.oos_metrics.spearman,
                rolling_corr_mean=eval_result.oos_metrics.rolling_corr_mean,
                rolling_corr_std=eval_result.oos_metrics.rolling_corr_std,
                sign_flip_count=eval_result.oos_metrics.sign_flip_count,
                frozen_lag=eval_result.oos_metrics.frozen_lag,
                lag_corr_at_frozen=eval_result.oos_metrics.lag_corr_at_frozen,
                partial_corr=eval_result.oos_metrics.partial_corr,
                n_obs=eval_result.oos_metrics.n_obs,
                first_date=eval_result.oos_metrics.first_date,
                last_date=eval_result.oos_metrics.last_date
            )
        
        # Build decision - include_in_model now passed as argument
        decision = CandidateDecision(
            rank=getattr(eval_result, 'rank', None),
            score_composite=getattr(eval_result, 'composite_score', None),
            include_in_model=include_in_model,
            notes=[]
        )
        
        return ScreenerCandidate(
            ticker=eval_result.ticker,
            category=eval_result.category,
            pairwise_obs=eval_result.pairwise_obs,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            overrides_applied=[],
            decision=decision
        )
    
    def save(self, output_dir: str | Path) -> Path:
        """
        Save output to directory.
        
        Args:
            output_dir: Base output directory
            
        Returns:
            Path to screener_output.json file
        """
        if self.output is None:
            raise ValueError("Must call build_output() first")
        
        # Create run directory
        run_dir = Path(output_dir) / self.context.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save screener output
        output_path = run_dir / "screener_output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.output.model_dump(by_alias=True), f, indent=2, default=str)
        
        # Save manifest
        manifest = {
            "run_id": self.context.run_id,
            "generated_at": self.context.generated_at,
            "universe_version": self.universe.meta.run_id,
            "artifacts": [
                {
                    "name": "screener_output.json",
                    "sha256": self.output.fingerprint
                }
            ],
            "fetch_stats": self.fetcher.get_stats()
        }
        
        manifest_path = run_dir / "run_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        
        # Create 'latest' link
        latest_dir = Path(output_dir) / "latest"
        if latest_dir.exists():
            import shutil
            if latest_dir.is_symlink():
                latest_dir.unlink()
            elif latest_dir.is_dir():
                shutil.rmtree(latest_dir)
        
        try:
            latest_dir.symlink_to(run_dir.name, target_is_directory=True)
        except OSError:
            import shutil
            shutil.copytree(run_dir, latest_dir)
        
        logger.info(f"Saved results to {output_path}")
        
        return output_path


def load_universe(universe_path: str | Path) -> UniverseOutput:
    """Load universe from JSON file."""
    with open(universe_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return UniverseOutput(**data)


def run_screener(
    universe_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path = "artifacts/runs"
) -> ScreenerOutput:
    """
    Run full screening pipeline.
    
    Args:
        universe_path: Path to universe.json
        config_path: Path to YAML config
        output_dir: Output directory for results
        
    Returns:
        ScreenerOutput object
    """
    config = load_config(config_path)
    universe = load_universe(universe_path)
    
    screener = FeatureScreener(config, universe)
    screener.fetch_prices()
    screener.run_evaluation()
    output = screener.build_output()
    screener.save(output_dir)
    
    return output


def main(
    universe_path: str,
    config_path: str,
    overrides_path: Optional[str] = None,
    output_dir: str = "artifacts/runs"
):
    """
    CLI entry point for feature screener.
    
    Args:
        universe_path: Path to universe.json
        config_path: Path to YAML config
        overrides_path: Optional path to overrides.json
        output_dir: Output directory
    """
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        output = run_screener(universe_path, config_path, output_dir)
        
        # Summary
        passed = len([c for c in output.candidates if c.decision.include_in_model])
        
        print(f"\n{'='*60}")
        print(f"Screening Complete")
        print(f"{'='*60}")
        print(f"Run ID:       {output.meta.run_id}")
        print(f"Universe:     {output.meta.universe_version}")
        print(f"Target:       {output.target.ticker}")
        print(f"Candidates:   {len(output.candidates)} passed / {len(output.excluded)} excluded")
        print(f"Fingerprint:  {output.fingerprint[:30]}...")
        print(f"{'='*60}")
        
        # Top 10
        print(f"\nTop 10 Candidates:")
        print(f"{'-'*60}")
        print(f"{'Rank':<5} {'Ticker':<12} {'Category':<18} {'IS Corr':<10} {'OOS Corr':<10}")
        print(f"{'-'*60}")
        
        for c in output.get_top_candidates(10):
            oos_corr = c.oos_metrics.pearson if c.oos_metrics else None
            oos_str = f"{oos_corr:.3f}" if oos_corr is not None else "N/A"
            is_str = f"{c.is_metrics.pearson:.3f}" if c.is_metrics.pearson else "N/A"
            
            print(f"{c.decision.rank:<5} {c.ticker:<12} {(c.category or ''):<18} {is_str:<10} {oos_str:<10}")
        
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Screening failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
