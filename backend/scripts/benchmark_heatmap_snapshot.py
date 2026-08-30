"""Read-only heatmap DB/hierarchy/serialization microbenchmark."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import time

from app.db import SessionLocal
from app.heatmap import flatten_heatmap_leaves, hierarchy_for_view
from app.models import HeatmapCache


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentile_value) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")

    db_ms: list[float] = []
    hierarchy_ms: list[float] = []
    serialize_ms: list[float] = []
    body = b""
    payload: dict = {}
    for _ in range(args.runs):
        started = time.perf_counter()
        with SessionLocal() as session:
            cache = session.query(HeatmapCache).first()
            if cache is None or not isinstance(cache.payload_json, dict):
                raise SystemExit("No heatmap snapshot is available")
            payload = cache.payload_json
        db_ms.append((time.perf_counter() - started) * 1_000)

        started = time.perf_counter()
        market = hierarchy_for_view(payload, "market")
        hierarchy_ms.append((time.perf_counter() - started) * 1_000)

        started = time.perf_counter()
        body = json.dumps(market, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        serialize_ms.append((time.perf_counter() - started) * 1_000)

    print(json.dumps({
        "runs": args.runs,
        "symbols": len(flatten_heatmap_leaves(payload)),
        "db_p50_ms": round(statistics.median(db_ms), 3),
        "db_p95_ms": round(percentile(db_ms, 0.95), 3),
        "hierarchy_p95_ms": round(percentile(hierarchy_ms, 0.95), 3),
        "serialize_p95_ms": round(percentile(serialize_ms, 0.95), 3),
        "raw_bytes": len(body),
        "gzip_bytes": len(gzip.compress(body, compresslevel=6)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
