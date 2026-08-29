"""Live, non-mocked probe for the exact sentiment and commentary contracts."""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from types import SimpleNamespace

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import ai_engine, commentary
from app.settings import get_settings


async def run_probe(*, fast: str, reliable: str, commentary_model: str) -> dict:
    runtime = get_settings()
    if not runtime.openrouter_api_key:
        return {"ok": False, "error": "OpenRouter credential is not configured"}

    probe_settings = SimpleNamespace(
        openrouter_api_key=runtime.openrouter_api_key,
        openrouter_max_retries=1,
        openrouter_rpm=runtime.openrouter_rpm,
        openrouter_timeout_seconds=runtime.openrouter_timeout_seconds,
        openrouter_chain_deadline_seconds=runtime.openrouter_chain_deadline_seconds,
        openrouter_fallback_models_list=[],
        resolved_scoring_fast_model=fast,
        resolved_scoring_reliable_model=reliable,
        resolved_commentary_model=commentary_model,
    )
    original_ai_settings = ai_engine.get_settings
    original_commentary_settings = commentary.get_settings
    ai_engine.get_settings = lambda: probe_settings
    commentary.get_settings = lambda: probe_settings
    try:
        article = [{
            "id": 1,
            "title": "Copper mine disruption tightens near-term concentrate supply",
            "description": "A temporary outage is expected to reduce shipments into the refined market.",
        }]
        scoring = {}
        for role, model, repair in (
            ("fast", fast, reliable),
            ("reliable", reliable, fast),
        ):
            valid, failed, metrics, rate_limited = await ai_engine._score_subset_with_model_v2(
                settings=probe_settings,
                model_name=model,
                repair_model_name=repair,
                articles=article,
                horizon_days=5,
            )
            result = valid.get(1)
            scoring[role] = {
                "requested_model": model,
                "actual_model": result.get("llm_model") if result else None,
                "ok_without_repair": bool(result and not failed and metrics.get("repair_success_count", 0) == 0),
                "failure_category": metrics.get("failure_category"),
                "rate_limited": bool(rate_limited),
            }

        commentary_result = await commentary._generate_commentary_and_stance(
            current_price=6.6,
            predicted_price=6.62,
            predicted_return=0.003,
            sentiment_index=0.2,
            sentiment_label="Bullish",
            top_influencers=[{"feature": "sentiment__index", "importance": 0.2}],
            news_count=8,
        )
        commentary_probe = {
            "requested_model": commentary_model,
            "actual_model": commentary_result.model_name,
            "generation_mode": commentary_result.generation_mode,
            "fallback_reason": commentary_result.fallback_reason,
            "ok_without_repair": commentary_result.generation_mode == "llm",
        }
        ok = all(item["ok_without_repair"] for item in scoring.values()) and commentary_probe["ok_without_repair"]
        return {"ok": ok, "scoring": scoring, "commentary": commentary_probe}
    finally:
        ai_engine.get_settings = original_ai_settings
        commentary.get_settings = original_commentary_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe exact OpenRouter production contracts")
    parser.add_argument("--fast", default="minimax/minimax-m2.7:free")
    parser.add_argument("--reliable", default="minimax/minimax-m3:free")
    parser.add_argument("--commentary", default="minimax/minimax-m3:free")
    args = parser.parse_args()
    result = asyncio.run(run_probe(fast=args.fast, reliable=args.reliable, commentary_model=args.commentary))
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
