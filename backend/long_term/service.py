"""Worker orchestration and snapshot persistence for the Long-Term Outlook Engine."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.instruments import canonicalize_instrument_symbol
from app.long_term_models import LongTermOutlookSnapshot
from long_term.schemas import LongTermOutlookResponse

logger = logging.getLogger(__name__)

MODEL_VERSION = "lt_committee_v2"


def generate_long_term_outlook_report(
    session: Optional[Session] = None,
    symbol: str = "HG=F",
    *,
    allow_network: bool = True,
    lookback_years: int = 15,
) -> LongTermOutlookResponse:
    """Compute a complete 20D/60D outlook snapshot and purged walk-forward validation."""
    from long_term.backtest import run_purged_walk_forward_suite
    from long_term.features import load_long_term_weekly_panel
    from long_term.model import fit_and_predict_long_term_outlook

    canonical_symbol = canonicalize_instrument_symbol(symbol)
    panel_bundle = load_long_term_weekly_panel(
        session=session,
        target_symbol=canonical_symbol,
        allow_network=allow_network,
        lookback_years=lookback_years,
    )
    validation = run_purged_walk_forward_suite(panel_bundle.panel)
    return fit_and_predict_long_term_outlook(
        panel_bundle=panel_bundle,
        validation=validation,
        model_version=MODEL_VERSION,
        symbol=canonical_symbol,
    )


def run_long_term_outlook_stage(
    session: Session,
    symbol: str = "HG=F",
    run_id: Optional[str] = None,
    *,
    allow_network: bool = True,
    commit: bool = True,
) -> dict[str, Any]:
    """Fail-open worker entrypoint that computes and persists the 20D/60D snapshot."""
    canonical_symbol = canonicalize_instrument_symbol(symbol)
    try:
        response = generate_long_term_outlook_report(
            session=session,
            symbol=canonical_symbol,
            allow_network=allow_network,
        )
        if response.status != "ok" or not response.as_of_date:
            return {
                "ok": False,
                "generated": False,
                "symbol": canonical_symbol,
                "reason": response.message or "insufficient_history",
            }

        payload_dict = response.model_dump(mode="json")
        validation_dict = {
            k: v.model_dump(mode="json") for k, v in response.validation.items()
        }
        snapshot_row = LongTermOutlookSnapshot(
            symbol=canonical_symbol,
            as_of_date=response.as_of_date,
            payload_json=payload_dict,
            validation_json=validation_dict,
            model_version=response.model_version or MODEL_VERSION,
            generated_at=datetime.now(timezone.utc),
            run_id=run_id,
        )
        session.add(snapshot_row)
        session.flush()
        if commit:
            session.commit()

        return {
            "ok": True,
            "generated": True,
            "symbol": canonical_symbol,
            "as_of_date": response.as_of_date,
            "model_version": snapshot_row.model_version,
            "horizons": {
                k: {
                    "stance": v.stance,
                    "institutional_rating": v.institutional_rating,
                    "expected_return_pct": v.expected_return_pct,
                    "target_price_q50": v.target_price_q50,
                    "conviction": v.conviction,
                }
                for k, v in response.horizons.items()
            },
        }
    except Exception as exc:
        logger.warning(
            "[run_id=%s] Long-term outlook stage failed (fail-open) for %s: %s",
            run_id,
            canonical_symbol,
            exc,
            exc_info=True,
        )
        try:
            session.rollback()
        except Exception:
            pass
        return {
            "ok": False,
            "generated": False,
            "symbol": canonical_symbol,
            "error": str(exc),
        }
