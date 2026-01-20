"""
Unit tests for fingerprinting.

Tests determinism and hash consistency.
"""

import pytest
import tempfile
from pathlib import Path

from screener.core.fingerprint import (
    compute_fingerprint,
    compute_file_fingerprint,
    compute_dataframe_fingerprint,
    verify_fingerprint,
)


class TestComputeFingerprint:
    """Tests for compute_fingerprint function."""
    
    def test_deterministic(self):
        """Same data should produce same hash."""
        data = {"key": "value", "number": 42}
        
        hash1 = compute_fingerprint(data)
        hash2 = compute_fingerprint(data)
        
        assert hash1 == hash2
    
    def test_order_invariant(self):
        """Dict key order should not affect hash."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}
        data3 = {"b": 2, "c": 3, "a": 1}
        
        hash1 = compute_fingerprint(data1)
        hash2 = compute_fingerprint(data2)
        hash3 = compute_fingerprint(data3)
        
        assert hash1 == hash2 == hash3
    
    def test_nested_deterministic(self):
        """Nested structures should be deterministic."""
        data = {
            "outer": {
                "inner1": [1, 2, 3],
                "inner2": {"deep": "value"}
            }
        }
        
        hash1 = compute_fingerprint(data)
        hash2 = compute_fingerprint(data)
        
        assert hash1 == hash2
    
    def test_format(self):
        """Hash should have correct format."""
        data = {"test": "data"}
        hash_val = compute_fingerprint(data)
        
        assert hash_val.startswith("sha256:")
        assert len(hash_val) == 71  # "sha256:" + 64 hex chars
    
    def test_different_data_different_hash(self):
        """Different data should produce different hashes."""
        hash1 = compute_fingerprint({"a": 1})
        hash2 = compute_fingerprint({"a": 2})
        
        assert hash1 != hash2
    
    def test_handles_dates(self):
        """Date objects should be handled via default=str."""
        from datetime import date, datetime
        
        data = {
            "date": date(2024, 1, 15),
            "datetime": datetime(2024, 1, 15, 10, 30, 0)
        }
        
        # Should not raise
        hash_val = compute_fingerprint(data)
        assert hash_val.startswith("sha256:")


class TestComputeFileFingerprint:
    """Tests for compute_file_fingerprint function."""
    
    def test_same_content_same_hash(self):
        """Same file content should produce same hash."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            path = f.name
        
        try:
            hash1 = compute_file_fingerprint(path)
            hash2 = compute_file_fingerprint(path)
            assert hash1 == hash2
        finally:
            Path(path).unlink()
    
    def test_different_content_different_hash(self):
        """Different content should produce different hashes."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("content 1")
            path1 = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("content 2")
            path2 = f.name
        
        try:
            hash1 = compute_file_fingerprint(path1)
            hash2 = compute_file_fingerprint(path2)
            assert hash1 != hash2
        finally:
            Path(path1).unlink()
            Path(path2).unlink()
    
    def test_file_not_found(self):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_file_fingerprint("/nonexistent/path/file.txt")


class TestVerifyFingerprint:
    """Tests for verify_fingerprint function."""
    
    def test_valid_fingerprint(self):
        """Correct fingerprint should verify."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            path = f.name
        
        try:
            expected = compute_file_fingerprint(path)
            assert verify_fingerprint(path, expected) == True
        finally:
            Path(path).unlink()
    
    def test_invalid_fingerprint(self):
        """Wrong fingerprint should not verify."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            path = f.name
        
        try:
            assert verify_fingerprint(path, "sha256:wronghash") == False
        finally:
            Path(path).unlink()


class TestDataFrameFingerprint:
    """Tests for compute_dataframe_fingerprint function."""
    
    def test_deterministic(self):
        """Same DataFrame should produce same hash."""
        import pandas as pd
        
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0]
        })
        
        hash1 = compute_dataframe_fingerprint(df)
        hash2 = compute_dataframe_fingerprint(df)
        
        assert hash1 == hash2
    
    def test_different_data_different_hash(self):
        """Different DataFrames should produce different hashes."""
        import pandas as pd
        
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})
        
        hash1 = compute_dataframe_fingerprint(df1)
        hash2 = compute_dataframe_fingerprint(df2)
        
        assert hash1 != hash2
