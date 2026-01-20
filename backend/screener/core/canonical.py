"""
Canonical content preparation for deterministic fingerprinting.

Provides helpers to build content dicts that produce stable fingerprints:
- Sorted lists
- Excluded non-deterministic fields
- Consistent serialization
"""

from typing import Any


def build_universe_content_dict(
    sources: list[dict],
    filter_parameters: dict,
    universe: list[dict],
    summary: dict
) -> dict:
    """
    Build canonical content dict for Universe fingerprint.
    
    Excludes non-deterministic fields and ensures stable ordering.
    
    Fields EXCLUDED (non-deterministic):
        - meta.* (run_id, generated_at, git_commit, lib_versions)
        - envelope fields
    
    Fields INCLUDED (deterministic):
        - sources (sorted by type+path)
        - filter_parameters (all)
        - universe (sorted by canonical_ticker)
        - summary (all)
    
    Args:
        sources: List of source info dicts
        filter_parameters: Filter params dict
        universe: List of universe symbol dicts
        summary: Summary dict
        
    Returns:
        Canonical content dict for fingerprinting
    """
    # Sort sources by type, then by path
    sorted_sources = sorted(
        sources,
        key=lambda s: (s.get("type", ""), s.get("path") or "", s.get("sha256") or "")
    )
    
    # Sort universe by canonical_ticker for stable order
    sorted_universe = sorted(
        universe,
        key=lambda s: s.get("canonical_ticker", "")
    )
    
    # Clean universe entries (remove deprecated source_tag, keep sources list)
    cleaned_universe = []
    for sym in sorted_universe:
        clean_sym = {
            "ticker": sym.get("ticker"),
            "canonical_ticker": sym.get("canonical_ticker"),
            "category": sym.get("category"),
            "first_date": sym.get("first_date"),
            "last_date": sym.get("last_date"),
            "total_weeks": sym.get("total_weeks"),
            "coverage_pct": sym.get("coverage_pct"),
            "status": sym.get("status"),
            "exclusion_reason": sym.get("exclusion_reason"),
            "sources": sorted(sym.get("sources", [])),  # Sort sources list
        }
        cleaned_universe.append(clean_sym)
    
    return {
        "sources": sorted_sources,
        "filter_parameters": filter_parameters,
        "universe": cleaned_universe,
        "summary": summary
    }


def build_screener_content_dict(
    target: dict,
    analysis_parameters: dict,
    universe_content_fingerprint: str,
    candidates: list[dict],
    excluded: list[dict]
) -> dict:
    """
    Build canonical content dict for Screener fingerprint.
    
    Excludes non-deterministic fields and ensures stable ordering.
    
    Fields EXCLUDED:
        - meta.* (run_id, generated_at, lib_versions)
        - artifacts[].created_at, size_bytes, path
        - universe_fingerprint in meta (use universe_content_fingerprint param)
    
    Fields INCLUDED:
        - target info
        - analysis_parameters (all)
        - universe_content_fingerprint (links to input universe)
        - candidates (sorted by ticker)
        - excluded (sorted by ticker)
    
    Args:
        target: Target info dict
        analysis_parameters: Analysis params dict
        universe_content_fingerprint: Content fingerprint of input universe
        candidates: List of candidate dicts
        excluded: List of excluded candidate dicts
        
    Returns:
        Canonical content dict for fingerprinting
    """
    # Sort candidates by ticker for stable order
    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.get("ticker", "")
    )
    
    # Sort excluded by ticker
    sorted_excluded = sorted(
        excluded,
        key=lambda e: e.get("ticker", "")
    )
    
    # Clean candidates (remove any runtime fields)
    cleaned_candidates = []
    for cand in sorted_candidates:
        # Include all analysis results but ensure stable structure
        cleaned_candidates.append(cand)
    
    return {
        "target": target,
        "analysis_parameters": analysis_parameters,
        "universe_content_fingerprint": universe_content_fingerprint,
        "candidates": cleaned_candidates,
        "excluded": sorted_excluded
    }
