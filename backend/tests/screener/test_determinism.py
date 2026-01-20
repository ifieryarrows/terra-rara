"""
Unit tests for fingerprint determinism.

Tests that same inputs produce same content_fingerprint,
even when run_id and generated_at change.
"""

import pytest
from datetime import datetime

from screener.core.fingerprint import compute_fingerprint
from screener.core.canonical import (
    build_universe_content_dict,
    build_screener_content_dict,
)


class TestContentFingerprintDeterminism:
    """Tests that content_fingerprint is deterministic."""
    
    def test_same_inputs_same_content_fingerprint(self):
        """Same inputs should produce identical content fingerprint."""
        sources = [
            {"type": "seed_csv", "path": "test.csv", "sha256": "sha256:abc123"}
        ]
        filter_params = {
            "min_history_days": 730,
            "min_coverage_pct": 80.0,
            "frequency": "W-FRI"
        }
        universe = [
            {
                "ticker": "FCX",
                "canonical_ticker": "FCX",
                "category": "miner_major",
                "status": "included",
                "sources": ["seed_csv:test.csv"]
            },
            {
                "ticker": "BHP",
                "canonical_ticker": "BHP",
                "category": "miner_major",
                "status": "included",
                "sources": ["seed_csv:test.csv"]
            }
        ]
        summary = {"total_candidates": 2, "included": 2, "excluded": 0}
        
        # Build content dict twice
        content1 = build_universe_content_dict(sources, filter_params, universe, summary)
        content2 = build_universe_content_dict(sources, filter_params, universe, summary)
        
        # Fingerprints should be identical
        fp1 = compute_fingerprint(content1)
        fp2 = compute_fingerprint(content2)
        
        assert fp1 == fp2
    
    def test_different_order_same_fingerprint(self):
        """Different order of universe entries should produce same fingerprint."""
        sources = [{"type": "seed_csv", "path": "test.csv"}]
        filter_params = {"min_history_days": 730}
        summary = {"total_candidates": 2}
        
        # Order 1: FCX first
        universe1 = [
            {"ticker": "FCX", "canonical_ticker": "FCX", "sources": []},
            {"ticker": "BHP", "canonical_ticker": "BHP", "sources": []}
        ]
        
        # Order 2: BHP first
        universe2 = [
            {"ticker": "BHP", "canonical_ticker": "BHP", "sources": []},
            {"ticker": "FCX", "canonical_ticker": "FCX", "sources": []}
        ]
        
        content1 = build_universe_content_dict(sources, filter_params, universe1, summary)
        content2 = build_universe_content_dict(sources, filter_params, universe2, summary)
        
        fp1 = compute_fingerprint(content1)
        fp2 = compute_fingerprint(content2)
        
        # Should be same because canonical sorts by canonical_ticker
        assert fp1 == fp2
    
    def test_sources_list_order_independent(self):
        """Different order of sources list should produce same fingerprint."""
        filter_params = {}
        summary = {}
        universe = [{"ticker": "FCX", "canonical_ticker": "FCX", "sources": ["a", "b"]}]
        
        # Sources in different order
        sources1 = [
            {"type": "seed_csv", "path": "a.csv"},
            {"type": "macro_peers"}
        ]
        sources2 = [
            {"type": "macro_peers"},
            {"type": "seed_csv", "path": "a.csv"}
        ]
        
        content1 = build_universe_content_dict(sources1, filter_params, universe, summary)
        content2 = build_universe_content_dict(sources2, filter_params, universe, summary)
        
        fp1 = compute_fingerprint(content1)
        fp2 = compute_fingerprint(content2)
        
        assert fp1 == fp2


class TestOutputFingerprintChanges:
    """Tests that output_fingerprint changes with meta fields."""
    
    def test_different_run_id_different_output_fingerprint(self):
        """Different run_id should produce different output fingerprint."""
        base_content = {"universe": []}
        
        # Two different metas
        output1 = {
            "meta": {"run_id": "univ-20260119-abc123", "generated_at": "2026-01-19T10:00:00Z"},
            **base_content
        }
        output2 = {
            "meta": {"run_id": "univ-20260119-xyz789", "generated_at": "2026-01-19T10:00:00Z"},
            **base_content
        }
        
        fp1 = compute_fingerprint(output1)
        fp2 = compute_fingerprint(output2)
        
        # Output fingerprints should differ
        assert fp1 != fp2
    
    def test_different_timestamp_different_output_fingerprint(self):
        """Different generated_at should produce different output fingerprint."""
        output1 = {
            "meta": {"run_id": "univ-20260119-abc123", "generated_at": "2026-01-19T10:00:00Z"},
            "universe": []
        }
        output2 = {
            "meta": {"run_id": "univ-20260119-abc123", "generated_at": "2026-01-19T10:01:00Z"},
            "universe": []
        }
        
        fp1 = compute_fingerprint(output1)
        fp2 = compute_fingerprint(output2)
        
        assert fp1 != fp2


class TestScreenerContentFingerprint:
    """Tests for screener content fingerprint."""
    
    def test_screener_content_deterministic(self):
        """Screener content fingerprint should be deterministic."""
        target = {"ticker": "HG=F"}
        analysis_params = {"is_start": "2018-01-01", "is_end": "2023-12-31"}
        universe_fp = "sha256:abc123"
        candidates = [
            {"ticker": "FCX", "is": {"pearson": 0.75}},
            {"ticker": "BHP", "is": {"pearson": 0.68}}
        ]
        excluded = [{"ticker": "BAD", "reason": "no_data"}]
        
        content1 = build_screener_content_dict(
            target, analysis_params, universe_fp, candidates, excluded
        )
        content2 = build_screener_content_dict(
            target, analysis_params, universe_fp, candidates, excluded
        )
        
        fp1 = compute_fingerprint(content1)
        fp2 = compute_fingerprint(content2)
        
        assert fp1 == fp2
    
    def test_screener_candidates_sorted(self):
        """Candidates in different order should produce same fingerprint."""
        target = {"ticker": "HG=F"}
        analysis_params = {}
        universe_fp = "sha256:abc"
        excluded = []
        
        # Different order
        candidates1 = [
            {"ticker": "FCX"},
            {"ticker": "BHP"}
        ]
        candidates2 = [
            {"ticker": "BHP"},
            {"ticker": "FCX"}
        ]
        
        content1 = build_screener_content_dict(
            target, analysis_params, universe_fp, candidates1, excluded
        )
        content2 = build_screener_content_dict(
            target, analysis_params, universe_fp, candidates2, excluded
        )
        
        fp1 = compute_fingerprint(content1)
        fp2 = compute_fingerprint(content2)
        
        assert fp1 == fp2
