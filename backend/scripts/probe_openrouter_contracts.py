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


async def run_probe(*, fast: str, reliable: str, commentary_model: str, batch_size: int = 12) -> dict:
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
        # Match the worker's 12-article V2 scoring chunk, using synthetic
        # scenarios so the probe never depends on mutable market data.
        scenarios = [
            ("Copper mine outage", "A temporary outage reduces concentrate shipments."),
            ("Refinery restart", "A smelter resumes production after scheduled maintenance."),
            ("Grid demand rises", "New grid projects increase near-term copper orders."),
            ("Dollar strengthens", "The US dollar rises against major currencies."),
            ("Warehouse inventories fall", "Exchange-registered copper stocks decline."),
            ("Scrap supply expands", "More secondary copper enters the market."),
            ("Housing starts slow", "Construction demand weakens this quarter."),
            ("Electronics orders improve", "Manufacturers report stronger copper demand."),
            ("Port disruption ends", "Concentrate exports resume after a short delay."),
            ("Oil price changes", "Crude oil moves with no stated copper link."),
            ("New mine commissioning", "Additional copper capacity begins operating."),
            ("Mixed industrial data", "Orders rise in one region and fall in another."),
        ]
        articles = [
            {"id": index + 1, "title": f"Synthetic contract probe: {title}", "description": description}
            for index, (title, description) in enumerate(scenarios[:max(1, min(batch_size, 12))])
        ]
        scoring = {}
        for role, model, repair in (
            ("fast", fast, reliable),
            ("reliable", reliable, fast),
        ):
            valid, failed, metrics, rate_limited = await ai_engine._score_subset_with_model_v2(
                settings=probe_settings,
                model_name=model,
                repair_model_name=repair,
                articles=articles,
                horizon_days=5,
            )
            result = valid.get(1)
            scoring[role] = {
                "requested_model": model,
                "actual_model": result.get("llm_model") if result else None,
                "item_count": len(articles),
                "valid_count": len(valid),
                "ok_without_repair": bool(len(valid) == len(articles) and not failed and metrics.get("repair_success_count", 0) == 0),
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
    parser.add_argument("--fast", default="nvidia/nemotron-3-super-120b-a12b:free")
    parser.add_argument("--reliable", default="liquid/lfm-2.5-2.6b:free")
    parser.add_argument("--commentary", default="nvidia/nemotron-3-super-120b-a12b:free")
    parser.add_argument("--batch-size", type=int, default=12, help="Synthetic scoring items, up to the worker's 12-item chunk")
    args = parser.parse_args()
    result = asyncio.run(run_probe(fast=args.fast, reliable=args.reliable, commentary_model=args.commentary, batch_size=args.batch_size))
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
