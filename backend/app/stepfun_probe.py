"""
StepFun / OpenRouter probe utility for sentiment V2 diagnostics.

Usage:
    py -m app.stepfun_probe
    py -m app.stepfun_probe --sample-size 24 --db-limit 24
    py -m app.stepfun_probe --skip-db
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from sqlalchemy.orm import Session

from app.ai_engine import (
    LLM_SCORING_PROVIDER_OPTIONS,
    LLM_SCORING_RESPONSE_FORMAT_V2,
    _build_llm_v2_user_prompt,
    score_batch_with_llm_v2,
)
from app.db import SessionLocal, init_db
from app.models import NewsProcessed, NewsRaw
from app.openrouter_client import OpenRouterError, create_chat_completion
from app.settings import get_settings


def _build_handcrafted_articles() -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "title": "Major copper mine outage in Chile removes 180k tonnes from expected supply",
            "description": "Analysts expect lower exchange inventories and tighter concentrates market.",
            "text": (
                "Major copper mine outage in Chile removes 180k tonnes from expected supply. "
                "Analysts expect lower exchange inventories and tighter concentrates market."
            ),
        },
        {
            "id": 2,
            "title": "China property slowdown deepens and cable demand weakens",
            "description": "Fabricators report softer orders and lower cathode premiums.",
            "text": (
                "China property slowdown deepens and cable demand weakens. "
                "Fabricators report softer orders and lower cathode premiums."
            ),
        },
        {
            "id": 3,
            "title": "Semiconductor patent lawsuit update",
            "description": "No direct discussion of copper demand or supply.",
            "text": "Semiconductor patent lawsuit update with no direct copper market linkage.",
        },
    ]


def _load_db_articles(session: Session, limit: int) -> list[dict[str, Any]]:
    rows = (
        session.query(
            NewsProcessed.id.label("processed_id"),
            NewsRaw.title,
            NewsRaw.description,
            NewsProcessed.cleaned_text,
            NewsRaw.published_at,
        )
        .join(NewsRaw, NewsProcessed.raw_id == NewsRaw.id)
        .order_by(NewsRaw.published_at.desc(), NewsProcessed.id.desc())
        .limit(max(1, int(limit)))
        .all()
    )

    articles: list[dict[str, Any]] = []
    for row in rows:
        title = str(row.title or "")[:500]
        description = str(row.description or "")[:800]
        text = str(row.cleaned_text or f"{title} {description}")[:1800]
        articles.append(
            {
                "id": int(row.processed_id),
                "title": title,
                "description": description,
                "text": text,
            }
        )
    return articles


async def _run_strict_probe(
    *,
    model: str,
    articles: list[dict[str, Any]],
) -> dict[str, Any]:
    settings = get_settings()
    user_prompt = _build_llm_v2_user_prompt(articles, horizon_days=5)
    started = time.perf_counter()
    try:
        data = await create_chat_completion(
            api_key=settings.openrouter_api_key or "",
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Senior Copper Futures Analyst. Return only valid JSON array with keys: "
                        "id,label,impact_score,confidence,relevance,event_type,reasoning."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1800,
            temperature=0.0,
            timeout_seconds=60.0,
            max_retries=settings.openrouter_max_retries,
            rpm=settings.openrouter_rpm,
            fallback_models=settings.openrouter_fallback_models_list,
            response_format=LLM_SCORING_RESPONSE_FORMAT_V2,
            provider=LLM_SCORING_PROVIDER_OPTIONS,
            extra_payload={"reasoning": {"exclude": True}},
        )
        elapsed = time.perf_counter() - started
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content")
        return {
            "ok": True,
            "elapsed_sec": round(elapsed, 2),
            "finish_reason": choice.get("finish_reason"),
            "content_len": len(content or ""),
            "preview": str(content)[:240],
        }
    except OpenRouterError as exc:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "elapsed_sec": round(elapsed, 2),
            "error_type": "OpenRouterError",
            "status_code": exc.status_code,
            "message": str(exc)[:280],
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "elapsed_sec": round(elapsed, 2),
            "error_type": type(exc).__name__,
            "message": str(exc)[:280],
        }


async def _run_v2_probe(
    *,
    articles: list[dict[str, Any]],
    horizon_days: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    bundle = await score_batch_with_llm_v2(articles, horizon_days=horizon_days)
    elapsed = time.perf_counter() - started

    sample_results = []
    for item in bundle.get("results", [])[:3]:
        sample_results.append(
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "impact_score": item.get("impact_score"),
                "confidence": item.get("confidence"),
                "relevance": item.get("relevance"),
                "event_type": item.get("event_type"),
            }
        )

    return {
        "elapsed_sec": round(elapsed, 2),
        "result_count": len(bundle.get("results", [])),
        "fallback_count": int(bundle.get("fallback_count", 0)),
        "parse_fail_count": int(bundle.get("parse_fail_count", 0)),
        "escalation_count": int(bundle.get("escalation_count", 0)),
        "failed_ids": bundle.get("failed_ids", []),
        "model_fast": bundle.get("model_fast"),
        "model_reliable": bundle.get("model_reliable"),
        "sample_results": sample_results,
    }


async def _run_probe(sample_size: int, db_limit: int, skip_db: bool) -> None:
    settings = get_settings()
    fast_model = settings.resolved_scoring_fast_model
    reliable_model = settings.resolved_scoring_reliable_model

    print("=== StepFun Probe ===")
    print(f"fast_model={fast_model}")
    print(f"reliable_model={reliable_model}")
    print(f"openrouter_rpm={settings.openrouter_rpm} max_retries={settings.openrouter_max_retries}")

    handcrafted = _build_handcrafted_articles()

    strict_summary = await _run_strict_probe(model=fast_model, articles=handcrafted)
    print("\n[1] strict_schema_probe")
    print(json.dumps(strict_summary, ensure_ascii=True, indent=2))

    v2_smoke = await _run_v2_probe(articles=handcrafted, horizon_days=5)
    print("\n[2] v2_smoke_probe_handcrafted")
    print(json.dumps(v2_smoke, ensure_ascii=True, indent=2))

    if skip_db:
        return

    with SessionLocal() as session:
        db_articles = _load_db_articles(session, limit=db_limit)
    if not db_articles:
        print("\n[3] v2_db_probe: no articles found in news_processed")
        return

    db_probe_articles = db_articles[: max(1, min(sample_size, len(db_articles)))]
    v2_db = await _run_v2_probe(articles=db_probe_articles, horizon_days=5)
    print("\n[3] v2_db_probe")
    print(f"sampled_articles={len(db_probe_articles)}")
    print(json.dumps(v2_db, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe StepFun/OpenRouter behavior for sentiment V2.")
    parser.add_argument("--sample-size", type=int, default=12, help="How many DB articles to score in DB probe.")
    parser.add_argument("--db-limit", type=int, default=24, help="How many latest DB articles to fetch before sampling.")
    parser.add_argument("--skip-db", action="store_true", help="Skip DB probe; run only strict + handcrafted tests.")
    args = parser.parse_args()

    init_db()
    asyncio.run(_run_probe(sample_size=args.sample_size, db_limit=args.db_limit, skip_db=args.skip_db))


if __name__ == "__main__":
    main()

