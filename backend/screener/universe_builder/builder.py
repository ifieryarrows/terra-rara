"""
Universe Builder orchestrator.

Coordinates source loading, probing, filtering, and output generation.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from screener.core.config import ScreenerConfig, load_config, compute_config_hash
from screener.core.fingerprint import compute_fingerprint
from screener.core.run_context import RunContext, DATA_PROVIDER_NOTES

from screener.universe_builder.sources import (
    load_source,
    merge_sources,
    SourceLoadResult,
)
from screener.universe_builder.prober import probe_batch, ProbeResult
from screener.universe_builder.categorize import categorize_batch, get_category_priority
from screener.contracts.universe import (
    UniverseOutput,
    UniverseMeta,
    LibVersions,
    SourceInfo,
    FilterParameters,
    UniverseSymbol,
    UniverseSummary,
)

logger = logging.getLogger(__name__)


class UniverseBuilder:
    """
    Builds and validates a universe of symbols for screening.
    """
    
    def __init__(self, config: ScreenerConfig):
        """
        Initialize builder.
        
        Args:
            config: Screener configuration
        """
        self.config = config
        self.context = RunContext(prefix="univ")
        self.source_results: list[SourceLoadResult] = []
        self.merged_tickers: list[dict] = []
        self.probe_results: list[ProbeResult] = []
        self.output: Optional[UniverseOutput] = None
    
    def load_sources(self) -> int:
        """
        Load tickers from all configured sources.
        
        Returns:
            Number of unique tickers loaded
        """
        logger.info("Loading sources...")
        
        for source_config in self.config.seed_sources:
            source_dict = source_config.model_dump()
            result = load_source(source_dict)
            self.source_results.append(result)
            
            if result.errors:
                for error in result.errors:
                    logger.warning(f"Source {source_config.type}: {error}")
        
        # Merge and deduplicate
        self.merged_tickers = merge_sources(self.source_results)
        
        # Categorize
        self.merged_tickers = categorize_batch(self.merged_tickers)
        
        logger.info(f"Loaded {len(self.merged_tickers)} unique tickers from {len(self.source_results)} sources")
        
        return len(self.merged_tickers)
    
    def probe_tickers(self, skip_probing: bool = False) -> int:
        """
        Probe all tickers to validate and get metadata.
        
        Args:
            skip_probing: If True, skip yfinance probing (for testing)
            
        Returns:
            Number of valid tickers
        """
        if skip_probing:
            # Create mock results for testing
            logger.info("Skipping probing (test mode)")
            for ticker_info in self.merged_tickers:
                self.probe_results.append(ProbeResult(
                    ticker=ticker_info["ticker"],
                    canonical_ticker=ticker_info["canonical_ticker"],
                    valid=True,
                    first_date="2015-01-02",
                    last_date=self.context.run_date,
                    total_weeks=400,
                    coverage_pct=99.0
                ))
            return len(self.probe_results)
        
        logger.info(f"Probing {len(self.merged_tickers)} tickers...")
        
        def progress_callback(current, total, ticker):
            if current % 20 == 0 or current == total:
                logger.info(f"Progress: {current}/{total} ({ticker})")
        
        self.probe_results = probe_batch(
            tickers=self.merged_tickers,
            frequency=self.config.filter.frequency,
            min_history_days=self.config.filter.min_history_days,
            progress_callback=progress_callback
        )
        
        valid_count = sum(1 for r in self.probe_results if r.valid)
        logger.info(f"Probing complete: {valid_count}/{len(self.probe_results)} valid")
        
        return valid_count
    
    def build_output(self) -> UniverseOutput:
        """
        Build the final output contract.
        
        Returns:
            UniverseOutput object
        """
        from screener.core.canonical import build_universe_content_dict
        
        # Map probe results by ticker
        probe_map = {r.canonical_ticker: r for r in self.probe_results}
        
        # Build universe symbols
        universe_symbols = []
        included_count = 0
        excluded_count = 0
        
        for ticker_info in self.merged_tickers:
            canonical = ticker_info["canonical_ticker"]
            probe = probe_map.get(canonical)
            
            if probe and probe.valid:
                # Check coverage threshold
                if probe.coverage_pct >= self.config.filter.min_coverage_pct:
                    status = "included"
                    exclusion_reason = None
                    included_count += 1
                else:
                    status = "excluded"
                    exclusion_reason = f"coverage_below_threshold_{probe.coverage_pct:.1f}pct"
                    excluded_count += 1
                
                symbol = UniverseSymbol(
                    ticker=ticker_info["ticker"],
                    canonical_ticker=canonical,
                    category=ticker_info.get("category"),
                    first_date=probe.first_date,
                    last_date=probe.last_date,
                    total_weeks=probe.total_weeks,
                    coverage_pct=probe.coverage_pct,
                    status=status,
                    exclusion_reason=exclusion_reason,
                    sources=ticker_info.get("sources", []),
                    source_tag=ticker_info.get("source_tag")  # Deprecated
                )
            else:
                # Failed probe
                exclusion_reason = probe.error if probe else "no_probe_result"
                excluded_count += 1
                
                symbol = UniverseSymbol(
                    ticker=ticker_info["ticker"],
                    canonical_ticker=canonical,
                    category=ticker_info.get("category"),
                    status="excluded",
                    exclusion_reason=exclusion_reason,
                    sources=ticker_info.get("sources", []),
                    source_tag=ticker_info.get("source_tag")  # Deprecated
                )
            
            universe_symbols.append(symbol)
        
        # Sort by category priority, then by ticker
        universe_symbols.sort(
            key=lambda s: (
                get_category_priority(s.category or "unknown"),
                s.canonical_ticker
            )
        )
        
        # Build source info
        sources = [result.to_source_info() for result in self.source_results]
        
        # Build metadata
        config_hash = compute_config_hash(self.config)
        
        meta = UniverseMeta(
            generated_at=self.context.generated_at,
            run_id=self.context.run_id,
            git_commit=self.context.git_commit,
            config_hash=config_hash,
            lib_versions=LibVersions(**self.context.lib_versions),
            data_provider_notes=DATA_PROVIDER_NOTES
        )
        
        filter_params = FilterParameters(
            min_history_days=self.config.filter.min_history_days,
            min_coverage_pct=self.config.filter.min_coverage_pct,
            frequency=self.config.filter.frequency
        )
        
        summary = UniverseSummary(
            total_candidates=len(universe_symbols),
            included=included_count,
            excluded=excluded_count
        )
        
        # Compute CONTENT fingerprint (deterministic - excludes meta)
        content_dict = build_universe_content_dict(
            sources=sources,
            filter_parameters=filter_params.model_dump(),
            universe=[s.model_dump() for s in universe_symbols],
            summary=summary.model_dump()
        )
        content_fingerprint = compute_fingerprint(content_dict)
        
        # Compute OUTPUT fingerprint (full envelope - includes meta)
        output_dict = {
            "meta": meta.model_dump(),
            "sources": sources,
            "filter_parameters": filter_params.model_dump(),
            "universe": [s.model_dump() for s in universe_symbols],
            "summary": summary.model_dump()
        }
        output_fingerprint = compute_fingerprint(output_dict)
        
        self.output = UniverseOutput(
            meta=meta,
            sources=[SourceInfo(**s) for s in sources],
            filter_parameters=filter_params,
            universe=universe_symbols,
            summary=summary,
            content_fingerprint=content_fingerprint,
            output_fingerprint=output_fingerprint,
            fingerprint=content_fingerprint  # Deprecated, for backward compat
        )
        
        return self.output
    
    def save(self, output_dir: str | Path) -> Path:
        """
        Save output to directory.
        
        Args:
            output_dir: Base output directory
            
        Returns:
            Path to universe.json file
        """
        if self.output is None:
            raise ValueError("Must call build_output() first")
        
        # Create run directory
        run_dir = Path(output_dir) / self.context.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save universe.json
        universe_path = run_dir / "universe.json"
        with open(universe_path, "w", encoding="utf-8") as f:
            json.dump(self.output.model_dump(), f, indent=2, default=str)
        
        # Save manifest
        manifest = {
            "run_id": self.context.run_id,
            "generated_at": self.context.generated_at,
            "artifacts": [
                {
                    "name": "universe.json",
                    "sha256": self.output.fingerprint
                }
            ]
        }
        
        manifest_path = run_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        
        # Create 'latest' symlink (or copy on Windows)
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
            # Symlinks may not work on Windows without admin privileges
            import shutil
            shutil.copytree(run_dir, latest_dir)
        
        logger.info(f"Saved universe to {universe_path}")
        
        return universe_path


def build_universe(
    config_path: str | Path,
    output_dir: str | Path = "artifacts/universes",
    skip_probing: bool = False
) -> UniverseOutput:
    """
    Build universe from config file.
    
    Args:
        config_path: Path to YAML config file
        output_dir: Output directory for artifacts
        skip_probing: Skip yfinance probing (for testing)
        
    Returns:
        UniverseOutput object
    """
    config = load_config(config_path)
    builder = UniverseBuilder(config)
    
    builder.load_sources()
    builder.probe_tickers(skip_probing=skip_probing)
    output = builder.build_output()
    builder.save(output_dir)
    
    return output


def main(config_path: str, output_dir: str = "artifacts/universes"):
    """
    CLI entry point for universe builder.
    
    Args:
        config_path: Path to YAML config file
        output_dir: Output directory for artifacts
    """
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        output = build_universe(config_path, output_dir)
        
        print(f"\n{'='*60}")
        print(f"Universe Build Complete")
        print(f"{'='*60}")
        print(f"Run ID:     {output.meta.run_id}")
        print(f"Total:      {output.summary.total_candidates}")
        print(f"Included:   {output.summary.included}")
        print(f"Excluded:   {output.summary.excluded}")
        print(f"Fingerprint: {output.fingerprint[:30]}...")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)
