"""
Run context management for screener.

Provides:
- Unique run ID generation
- Git commit detection
- Library version capture
- Timestamp management
"""

import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional
import secrets
import importlib.metadata


def generate_run_id(prefix: str = "run") -> str:
    """
    Generate unique run ID.
    
    Format: {prefix}-{YYYYMMDD}-{random6}
    Example: run-20260119-abc123
    
    Args:
        prefix: Prefix for run ID (e.g., "univ", "scr")
        
    Returns:
        Unique run ID string
    """
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    random_suffix = secrets.token_hex(3)  # 6 hex chars
    return f"{prefix}-{date_str}-{random_suffix}"


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash.
    
    Returns:
        Short commit hash (7 chars) or None if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return None


def get_lib_versions() -> dict[str, str]:
    """
    Get versions of key libraries.
    
    Returns:
        Dict mapping library name to version string
    """
    libs = ["python", "yfinance", "pandas", "numpy", "pydantic"]
    versions = {}
    
    # Python version
    versions["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Package versions
    for lib in libs[1:]:  # Skip python
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "not_installed"
    
    return versions


def get_current_timestamp() -> str:
    """
    Get current UTC timestamp in ISO format.
    
    Returns:
        ISO 8601 formatted timestamp with Z suffix
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_run_date() -> str:
    """
    Get current date in YYYY-MM-DD format.
    
    Returns:
        Date string
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class RunContext:
    """
    Encapsulates all metadata for a single run.
    
    Provides immutable snapshot of run context at creation time.
    """
    
    def __init__(self, prefix: str = "run"):
        self.run_id = generate_run_id(prefix)
        self.generated_at = get_current_timestamp()
        self.run_date = get_run_date()
        self.git_commit = get_git_commit()
        self.lib_versions = get_lib_versions()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "git_commit": self.git_commit,
            "lib_versions": self.lib_versions
        }


# Data provider notes (static, documented limitations)
DATA_PROVIDER_NOTES = [
    "yfinance: HG=F close is provider-reported, not official COMEX settlement",
    "yfinance: Adj Close may be missing for some symbols; falls back to Close",
    "yfinance: Rate limit ~2000 requests/hour; exponential backoff applied",
    "yfinance: Historical depth varies by symbol; some limited to recent years"
]
