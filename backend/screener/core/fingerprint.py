"""
SHA256 fingerprinting for artifacts and data.

Provides deterministic hashing for:
- JSON-serializable data structures
- Files (binary content)
- Pandas DataFrames
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def compute_fingerprint(data: Any) -> str:
    """
    Compute deterministic SHA256 hash of JSON-serializable data.
    
    Uses sorted keys and consistent serialization for determinism.
    Handles date/datetime objects via default=str.
    
    Args:
        data: JSON-serializable data (dict, list, primitives)
        
    Returns:
        String in format "sha256:<64-char-hex>"
    """
    canonical = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=True,
        default=str,
        separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def compute_file_fingerprint(path: str | Path) -> str:
    """
    Compute SHA256 hash of file contents.
    
    Reads file in binary mode in chunks for memory efficiency.
    
    Args:
        path: Path to file
        
    Returns:
        String in format "sha256:<64-char-hex>"
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    h = hashlib.sha256()
    
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    
    return f"sha256:{h.hexdigest()}"


def compute_dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Compute deterministic hash of DataFrame contents.
    
    Uses CSV serialization for consistency.
    Index is included in the hash.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        String in format "sha256:<64-char-hex>"
    """
    # Serialize to CSV string for hashing
    csv_content = df.to_csv(index=True, date_format="%Y-%m-%d")
    digest = hashlib.sha256(csv_content.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def verify_fingerprint(path: str | Path, expected: str) -> bool:
    """
    Verify that file matches expected fingerprint.
    
    Args:
        path: Path to file
        expected: Expected fingerprint (sha256:...)
        
    Returns:
        True if fingerprints match, False otherwise
    """
    actual = compute_file_fingerprint(path)
    return actual == expected
