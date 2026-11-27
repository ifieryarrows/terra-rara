"""Exogenous Catalyst & Shock Radar (0-72h) aggregated from NewsSentimentV2.

Clusters recent high-relevance, high-impact articles by ``event_type`` and
requires multi-source corroboration before triggering ``CATALYST_ACTIVE`` so
single-source headlines do not fire false alarms. Runs purely on existing DB
sentiment rows with zero extra LLM calls.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models import NewsProcessed, NewsRaw, NewsSentimentV2
from app.settings import get_settings

logger = logging.getLogger(__name__)

_IGNORED_EVENT_TYPES: frozenset[str] = frozenset({"non_copper", "mixed_unclear", ""})

EVENT_TYPE_LABELS: dict[str, str] = {
    "supply_disruption": "Supply Disruption",
    "supply_expansion": "Supply Expansion",
    "demand_increase": "Demand Surge",
    "demand_decrease": "Demand Contraction",
    "inventory_draw": "Inventory Draw",
    "inventory_build": "Inventory Build",
    "policy_support": "Policy / Stimulus Support",
    "policy_drag": "Policy / Tariff Drag",
    "macro_usd_up": "USD Tightening",
    "macro_usd_down": "USD Easing",
    "cost_push": "Smelter / Cost Push",
}


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_source_name(raw: NewsRaw) -> str:
    """Extract canonical lowercase publisher/source string for distinct-source counting."""
    pub = getattr(raw, "publisher", None)
    if isinstance(pub, str) and pub.strip():
        return pub.strip()
    payload = getattr(raw, "raw_payload", None)
    if isinstance(payload, dict):
        src = payload.get("source")
        if isinstance(src, dict):
            name = src.get("name") or src.get("id")
            if isinstance(name, str) and name.strip():
                return name.strip()
        elif isinstance(src, str) and src.strip():
            return src.strip()
    feed_src = getattr(raw, "source", None)
    if isinstance(feed_src, str) and feed_src.strip():
        return feed_src.strip()
    return "unknown"


def _quiet_radar_payload(
    *,
    now: datetime,
    window_hours: int,
    active_window_hours: int,
) -> dict[str, Any]:
    return {
        "status": "QUIET",
        "cone_override_warning": False,
        "cone_warning_message": None,
        "dominant_cluster": None,
        "active_clusters": [],
        "window_hours": int(window_hours),
        "active_window_hours": int(active_window_hours),
        "total_high_impact_articles": 0,
        "generated_at": now.isoformat(),
    }


def build_catalyst_radar(
    session: Session,
    *,
    now: Optional[datetime] = None,
    window_hours: int = 72,
    active_window_hours: int = 24,
    horizon_days: Optional[int] = None,
    min_relevance: float = 0.70,
    min_abs_impact: float = 0.55,
    min_confidence: float = 0.50,
) -> dict[str, Any]:
    """Build deterministic 0-72h exogenous catalyst summary from NewsSentimentV2."""
    now_utc = _as_utc(now) or datetime.now(timezone.utc)
    if session is None:
        return _quiet_radar_payload(
            now=now_utc, window_hours=window_hours, active_window_hours=active_window_hours
        )

    if horizon_days is None:
        try:
            resolved_horizon = max(1, int(get_settings().sentiment_horizon_days))
        except Exception:
            resolved_horizon = 5
    else:
        resolved_horizon = max(1, int(horizon_days))

    window_start = now_utc - timedelta(hours=int(window_hours))
    active_cutoff = now_utc - timedelta(hours=int(active_window_hours))

    try:
        rows = (
            session.query(NewsSentimentV2, NewsProcessed, NewsRaw)
            .join(NewsProcessed, NewsProcessed.id == NewsSentimentV2.news_processed_id)
            .join(NewsRaw, NewsRaw.id == NewsProcessed.raw_id)
            .filter(
                NewsRaw.published_at >= window_start,
                NewsProcessed.duplicate_of_id.is_(None),
                NewsSentimentV2.horizon_days == resolved_horizon,
                NewsSentimentV2.relevance_score >= float(min_relevance),
                NewsSentimentV2.confidence_calibrated >= float(min_confidence),
            )
            .all()
        )
    except Exception as exc:
        logger.warning("Catalyst radar DB query failed (fail-open to QUIET): %s", exc)
        return _quiet_radar_payload(
            now=now_utc, window_hours=window_hours, active_window_hours=active_window_hours
        )

    buckets: dict[str, list[dict[str, Any]]] = {}
    total_high_impact = 0

    for score, proc, raw in rows:
        ev = str(getattr(score, "event_type", "") or "").strip().lower()
        if ev in _IGNORED_EVENT_TYPES or ev not in EVENT_TYPE_LABELS:
            continue

        llm_imp = getattr(score, "impact_score_llm", 0.0) or 0.0
        final_sc = getattr(score, "final_score", 0.0) or 0.0
        if not (math.isfinite(llm_imp) and math.isfinite(final_sc)):
            continue
        effective_imp = float(llm_imp if abs(llm_imp) >= abs(final_sc) else final_sc)
        if max(abs(float(llm_imp)), abs(float(final_sc))) < float(min_abs_impact):
            continue

        pub_dt = _as_utc(getattr(raw, "published_at", None)) or now_utc
        source_display = _normalize_source_name(raw)
        rel = float(getattr(score, "relevance_score", 0.0) or 0.0)
        conf = float(getattr(score, "confidence_calibrated", 0.0) or 0.0)
        title = getattr(raw, "title", None) or getattr(proc, "canonical_title", None) or ""
        url = getattr(raw, "url", None)

        total_high_impact += 1
        buckets.setdefault(ev, []).append(
            {
                "title": str(title),
                "publisher": source_display,
                "source_key": source_display.lower(),
                "published_at": pub_dt,
                "impact_score": effective_imp,
                "abs_impact": abs(effective_imp),
                "relevance": rel,
                "confidence": conf,
                "url": url,
                "is_24h": pub_dt >= active_cutoff,
            }
        )

    clusters: list[dict[str, Any]] = []
    for ev, items in buckets.items():
        n72 = len(items)
        n24 = sum(1 for x in items if x["is_24h"])
        distinct_sources = len({x["source_key"] for x in items})
        distinct_sources_24h = len({x["source_key"] for x in items if x["is_24h"]})

        weights = [max(x["relevance"] * x["confidence"], 0.1) for x in items]
        w_sum = sum(weights)
        avg_imp = sum(x["impact_score"] * w for x, w in zip(items, weights)) / max(w_sum, 1e-6)
        max_abs_imp = max(x["abs_impact"] for x in items)
        avg_rel = sum(x["relevance"] for x in items) / n72
        avg_conf = sum(x["confidence"] for x in items) / n72

        severity = min(
            1.0,
            0.45 * max_abs_imp
            + 0.25 * avg_rel
            + 0.15 * min(distinct_sources / 3.0, 1.0)
            + 0.15 * (1.0 if n24 > 0 else 0.4),
        )
        direction = "BULLISH" if avg_imp >= 0.0 else "BEARISH"
        label = EVENT_TYPE_LABELS.get(ev, ev.replace("_", " ").title())
        latest_dt = max(x["published_at"] for x in items)

        corroborated = (
            (n24 >= 2 and distinct_sources_24h >= 2 and abs(avg_imp) >= 0.60)
            or (n72 >= 3 and distinct_sources >= 2 and n24 >= 1 and abs(avg_imp) >= 0.65)
        )

        sorted_items = sorted(
            items, key=lambda x: (x["abs_impact"], x["published_at"]), reverse=True
        )
        top_headlines = [
            {
                "title": x["title"],
                "publisher": x["publisher"],
                "published_at": x["published_at"].isoformat(),
                "impact_score": round(float(x["impact_score"]), 3),
                "url": x["url"],
            }
            for x in sorted_items[:3]
        ]

        clusters.append(
            {
                "event_type": ev,
                "label": label,
                "direction": direction,
                "article_count_72h": n72,
                "article_count_24h": n24,
                "distinct_sources": distinct_sources,
                "avg_impact": round(float(avg_imp), 3),
                "max_abs_impact": round(float(max_abs_imp), 3),
                "avg_relevance": round(float(avg_rel), 3),
                "avg_confidence": round(float(avg_conf), 3),
                "severity_score": round(float(severity), 3),
                "corroborated": bool(corroborated),
                "latest_published_at": latest_dt.isoformat(),
                "top_headlines": top_headlines,
            }
        )

    clusters.sort(
        key=lambda c: (
            1 if c["corroborated"] else 0,
            c["severity_score"],
            c["article_count_24h"],
            c["article_count_72h"],
        ),
        reverse=True,
    )

    if not clusters:
        return _quiet_radar_payload(
            now=now_utc, window_hours=window_hours, active_window_hours=active_window_hours
        )

    top_clusters = clusters[:4]
    dominant = top_clusters[0]
    any_corroborated = any(c["corroborated"] for c in top_clusters)
    if any_corroborated:
        status = "CATALYST_ACTIVE"
        cone_override_warning = True
        cone_warning_message = (
            f"Active {dominant['label'].lower()} catalyst corroborated across "
            f"{dominant['distinct_sources']} sources — statistical 5D price cone may understate near-term gap risk."
        )
    else:
        status = "WATCH"
        cone_override_warning = False
        cone_warning_message = None

    return {
        "status": status,
        "cone_override_warning": cone_override_warning,
        "cone_warning_message": cone_warning_message,
        "dominant_cluster": dominant,
        "active_clusters": top_clusters,
        "window_hours": int(window_hours),
        "active_window_hours": int(active_window_hours),
        "total_high_impact_articles": int(total_high_impact),
        "generated_at": now_utc.isoformat(),
    }
