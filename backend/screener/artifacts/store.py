"""
Artifact store management.

Handles artifact directory structure and manifest generation.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from screener.core.fingerprint import compute_file_fingerprint

logger = logging.getLogger(__name__)


class ArtifactStore:
    """
    Manages artifact storage with directory structure and manifests.
    
    Directory layout:
        artifacts/
        ├── runs/
        │   ├── scr-20260119-xyz789/
        │   │   ├── run_manifest.json
        │   │   ├── screener_output.json
        │   │   └── config_snapshot.yaml
        │   └── latest/ -> scr-20260119-xyz789
        ├── universes/
        │   ├── univ-20260119-abc123/
        │   │   ├── universe.json
        │   │   └── manifest.json
        │   └── latest/ -> univ-20260119-abc123
        ├── raw/
        │   ├── prices/
        │   │   ├── HG_F_20260119.parquet
        │   │   └── checksums.json
        │   └── checksums.json
        └── derived/
            └── checksums.json
    """
    
    def __init__(self, base_dir: str | Path):
        """
        Initialize artifact store.
        
        Args:
            base_dir: Base directory for all artifacts
        """
        self.base_dir = Path(base_dir)
        
        # Create directory structure
        self.runs_dir = self.base_dir / "runs"
        self.universes_dir = self.base_dir / "universes"
        self.raw_dir = self.base_dir / "raw"
        self.derived_dir = self.base_dir / "derived"
        
        for dir_path in [self.runs_dir, self.universes_dir, self.raw_dir, self.derived_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_run_dir(self, run_id: str) -> Path:
        """Get directory for a specific run."""
        return self.runs_dir / run_id
    
    def get_universe_dir(self, universe_id: str) -> Path:
        """Get directory for a specific universe."""
        return self.universes_dir / universe_id
    
    def get_latest_universe(self) -> Optional[Path]:
        """Get path to latest universe.json."""
        latest = self.universes_dir / "latest" / "universe.json"
        if latest.exists():
            return latest
        
        # Fallback: find most recent
        universe_dirs = sorted(
            [d for d in self.universes_dir.iterdir() if d.is_dir() and d.name.startswith("univ-")],
            reverse=True
        )
        
        if universe_dirs:
            return universe_dirs[0] / "universe.json"
        
        return None
    
    def get_latest_screener_output(self) -> Optional[Path]:
        """Get path to latest screener_output.json."""
        latest = self.runs_dir / "latest" / "screener_output.json"
        if latest.exists():
            return latest
        
        # Fallback: find most recent
        run_dirs = sorted(
            [d for d in self.runs_dir.iterdir() if d.is_dir() and d.name.startswith("scr-")],
            reverse=True
        )
        
        if run_dirs:
            return run_dirs[0] / "screener_output.json"
        
        return None
    
    def list_runs(self, limit: int = 10) -> list[dict]:
        """
        List recent screening runs.
        
        Args:
            limit: Maximum number of runs to return
            
        Returns:
            List of run metadata dicts
        """
        runs = []
        
        run_dirs = sorted(
            [d for d in self.runs_dir.iterdir() if d.is_dir() and d.name.startswith("scr-")],
            reverse=True
        )
        
        for run_dir in run_dirs[:limit]:
            manifest_path = run_dir / "run_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                runs.append({
                    "run_id": run_dir.name,
                    "path": str(run_dir),
                    "generated_at": manifest.get("generated_at"),
                    "universe_version": manifest.get("universe_version")
                })
            else:
                runs.append({
                    "run_id": run_dir.name,
                    "path": str(run_dir)
                })
        
        return runs
    
    def list_universes(self, limit: int = 10) -> list[dict]:
        """
        List recent universes.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of universe metadata dicts
        """
        universes = []
        
        universe_dirs = sorted(
            [d for d in self.universes_dir.iterdir() if d.is_dir() and d.name.startswith("univ-")],
            reverse=True
        )
        
        for univ_dir in universe_dirs[:limit]:
            manifest_path = univ_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                universes.append({
                    "universe_id": univ_dir.name,
                    "path": str(univ_dir),
                    "generated_at": manifest.get("generated_at")
                })
            else:
                universes.append({
                    "universe_id": univ_dir.name,
                    "path": str(univ_dir)
                })
        
        return universes


def create_run_manifest(
    run_id: str,
    generated_at: str,
    universe_version: str,
    artifacts: list[dict],
    extra_metadata: Optional[dict] = None
) -> dict:
    """
    Create a run manifest document.
    
    Args:
        run_id: Unique run identifier
        generated_at: ISO timestamp
        universe_version: Universe used for this run
        artifacts: List of artifact references with name and sha256
        extra_metadata: Additional metadata to include
        
    Returns:
        Manifest dict
    """
    manifest = {
        "run_id": run_id,
        "generated_at": generated_at,
        "universe_version": universe_version,
        "artifacts": artifacts
    }
    
    if extra_metadata:
        manifest.update(extra_metadata)
    
    return manifest
