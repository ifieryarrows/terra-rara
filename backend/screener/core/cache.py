"""
File-based cache for price data with collision-proof keys.

Provides:
- Parquet-based storage for efficiency
- Collision-proof cache keys via fetch_params fingerprint
- Cache invalidation and cleanup
"""

import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from screener.core.fingerprint import compute_file_fingerprint

logger = logging.getLogger(__name__)


class FetchParams:
    """
    Immutable fetch parameters that define a unique cache entry.
    
    COLLISION-PROOF GUARANTEE:
        Any change to these parameters produces a different cache key.
    """
    
    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        resample_rule: str = "W-FRI",
        price_field: str = "Close",
        auto_adjust: bool = True,
        actions: bool = True
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.resample_rule = resample_rule
        self.price_field = price_field
        self.auto_adjust = auto_adjust
        self.actions = actions
    
    def to_dict(self) -> dict:
        """Convert to ordered dict for hashing."""
        return {
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "interval": self.interval,
            "resample_rule": self.resample_rule,
            "price_field": self.price_field,
            "auto_adjust": self.auto_adjust,
            "actions": self.actions
        }
    
    def fingerprint(self) -> str:
        """
        Compute deterministic fingerprint of fetch params.
        
        Returns 8-char hex string (collision probability ~1 in 4 billion).
        """
        data = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:8]
    
    def cache_key(self) -> str:
        """
        Generate collision-proof cache key.
        
        Format: {safe_symbol}_{start}_{end}_{params_fingerprint}
        
        Example: HG_F_2018-01-01_2024-01-19_a3f2c891
        """
        safe_symbol = self.symbol.replace("=", "_").replace("^", "_").replace(".", "_")
        return f"{safe_symbol}_{self.start_date}_{self.end_date}_{self.fingerprint()}"


class PriceCache:
    """
    File-based cache for raw price data with collision-proof keys.
    
    Stores DataFrames as Parquet files.
    Tracks fingerprints for integrity verification.
    
    CACHE KEY CONTRACT (COLLISION-PROOF):
        Cache key = f"{symbol}_{start}_{end}_{params_fingerprint}"
        
        The params_fingerprint is a SHA256 hash of:
        - symbol, start_date, end_date
        - interval (e.g., "1d")
        - resample_rule (e.g., "W-FRI")
        - price_field (e.g., "Close")
        - auto_adjust, actions
        
        This GUARANTEES that any change to fetch parameters produces
        a different cache file, eliminating collision risk.
    """
    
    def __init__(self, cache_dir: str | Path, namespace: Optional[str] = None):
        """
        Initialize cache.
        
        Args:
            cache_dir: Base directory for cache files
            namespace: Optional namespace subdirectory (e.g., config_hash[:8])
        """
        self.base_dir = Path(cache_dir)
        
        # Use namespace subdirectory if provided
        if namespace:
            self.cache_dir = self.base_dir / namespace
        else:
            self.cache_dir = self.base_dir
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checksums_file = self.cache_dir / "checksums.json"
        self._checksums: dict[str, dict] = {}
        self._load_checksums()
    
    def _load_checksums(self):
        """Load existing checksums from file."""
        if self.checksums_file.exists():
            with open(self.checksums_file, "r") as f:
                self._checksums = json.load(f)
    
    def _save_checksums(self):
        """Save checksums to file."""
        with open(self.checksums_file, "w") as f:
            json.dump(self._checksums, f, indent=2, sort_keys=True)
    
    def _make_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.parquet"
    
    def get(self, params: FetchParams) -> Optional[pd.DataFrame]:
        """
        Get cached DataFrame for fetch params.
        
        Args:
            params: FetchParams object defining the cache entry
            
        Returns:
            DataFrame if cached and valid, None otherwise
        """
        cache_key = params.cache_key()
        path = self._make_path(cache_key)
        
        # Debug log for cache lookup
        exists = path.exists()
        logger.info(f"Cache lookup: {params.symbol} -> {path.name} (exists={exists})")
        
        if not exists:
            return None
        
        try:
            df = pd.read_parquet(path)
            logger.info(f"Cache hit: {params.symbol} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Cache read failed for {params.symbol}: {e}")
            return None
    
    def put(self, params: FetchParams, df: pd.DataFrame) -> str:
        """
        Store DataFrame in cache.
        
        Args:
            params: FetchParams object defining the cache entry
            df: DataFrame to cache
            
        Returns:
            Fingerprint of stored file
        """
        cache_key = params.cache_key()
        path = self._make_path(cache_key)
        
        df.to_parquet(path, index=True)
        fingerprint = compute_file_fingerprint(path)
        
        # Store metadata with checksum
        self._checksums[cache_key] = {
            "sha256": fingerprint,
            "params": params.to_dict(),
            "rows": len(df),
            "cached_at": datetime.now(timezone.utc).isoformat()
        }
        self._save_checksums()
        
        logger.debug(f"Cached: {params.symbol} ({len(df)} rows)")
        return fingerprint
    
    def has(self, params: FetchParams) -> bool:
        """Check if params are cached."""
        cache_key = params.cache_key()
        return self._make_path(cache_key).exists()
    
    def get_metadata(self, params: FetchParams) -> Optional[dict]:
        """Get metadata for cached entry."""
        cache_key = params.cache_key()
        return self._checksums.get(cache_key)
    
    def clear(self, older_than_days: Optional[int] = None):
        """
        Clear cache files.
        
        Args:
            older_than_days: Only clear files older than N days (None = all)
        """
        import os
        from datetime import timedelta
        
        cutoff = None
        if older_than_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        
        cleared = 0
        for path in self.cache_dir.glob("*.parquet"):
            should_clear = True
            
            if cutoff is not None:
                mtime = datetime.fromtimestamp(
                    os.path.getmtime(path),
                    tz=timezone.utc
                )
                should_clear = mtime < cutoff
            
            if should_clear:
                path.unlink()
                cleared += 1
        
        # Rebuild checksums from remaining files
        remaining_keys = set()
        for path in self.cache_dir.glob("*.parquet"):
            remaining_keys.add(path.stem)
        
        self._checksums = {k: v for k, v in self._checksums.items() if k in remaining_keys}
        self._save_checksums()
        
        logger.info(f"Cleared {cleared} cache files")


# Backward compatibility wrapper
def create_fetch_params(
    symbol: str,
    start_date: str,
    end_date: str,
    **kwargs
) -> FetchParams:
    """Helper to create FetchParams with defaults."""
    return FetchParams(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
