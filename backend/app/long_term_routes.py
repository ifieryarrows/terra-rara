"""Snapshot-only FastAPI router for Long-Term Copper Outlook (20D / 60D)."""

from __future__ import annotations

import logging
import sys
from datetime import timezone
from typing import Any, Optional

from fastapi import APIRouter

from app import db as app_db
from app.instruments import canonicalize_instrument_symbol
from app.long_term_models import LongTermOutlookSnapshot
from long_term.schemas import LongTermOutlookResponse

logger = logging.getLogger(__name__)

# Patchable alias for unit tests mirroring app.main.SessionLocal overrides.
SessionLocal = app_db.SessionLocal
_ORIGINAL_SESSION_LOCAL = SessionLocal

long_term_router = APIRouter(tags=["Long-Term Outlook"])


def _open_session():
    if SessionLocal is not _ORIGINAL_SESSION_LOCAL:
        return SessionLocal()
    if app_db.SessionLocal is not _ORIGINAL_SESSION_LOCAL:
        return app_db.SessionLocal()
    main_mod = sys.modules.get("app.main")
    if main_mod is not None:
        main_session = getattr(main_mod, "SessionLocal", _ORIGINAL_SESSION_LOCAL)
        if main_session is not _ORIGINAL_SESSION_LOCAL:
            return main_session()
    return app_db.SessionLocal()


def _unavailable_response(
    symbol: str,
    message: str = "No long-term outlook snapshot available yet. Run the pipeline worker to generate 20D/60D forecasts.",
) -> LongTermOutlookResponse:
    return LongTermOutlookResponse(
        status="unavailable",
        symbol=symbol,
        source="unavailable",
        available_horizons=["20D", "60D"],
        default_horizon="20D",
        horizons={},
        validation={},
        message=message,
    )


def _as_iso_utc(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


@long_term_router.get(
    "/api/outlook/long-term/{symbol}",
    response_model=LongTermOutlookResponse,
    summary="Get snapshot-only Long-Term Copper Outlook (20D & 60D)",
    description=(
        "Serves the latest persisted 1-Month (20D) and 1-Quarter (60D) structural "
        "outlook snapshot from the database. Snapshot-only: never triggers inline "
        "training or network downloads in an HTTP handler."
    ),
)
def get_long_term_outlook(symbol: str = "HG=F") -> LongTermOutlookResponse:
    """Return the latest persisted long-term outlook snapshot or a clean unavailable payload."""
    canonical_symbol = canonicalize_instrument_symbol(symbol)
    try:
        with _open_session() as session:
            latest = (
                session.query(LongTermOutlookSnapshot)
                .filter(LongTermOutlookSnapshot.symbol == canonical_symbol)
                .order_by(
                    LongTermOutlookSnapshot.generated_at.desc(),
                    LongTermOutlookSnapshot.id.desc(),
                )
                .first()
            )
            if latest is None or not isinstance(latest.payload_json, dict):
                return _unavailable_response(canonical_symbol)

            payload = dict(latest.payload_json)
            payload["status"] = "ok"
            payload["symbol"] = canonical_symbol
            payload["source"] = "snapshot"
            payload.setdefault("as_of_date", latest.as_of_date)
            payload.setdefault("model_version", latest.model_version)
            payload.setdefault("generated_at", _as_iso_utc(latest.generated_at))
            if isinstance(latest.validation_json, dict) and latest.validation_json:
                payload["validation"] = latest.validation_json
            elif "validation" not in payload or not isinstance(payload.get("validation"), dict):
                payload["validation"] = {}

            return LongTermOutlookResponse.model_validate(payload)
    except Exception as exc:
        logger.warning(
            "Long-term outlook snapshot lookup failed for %s: %s",
            canonical_symbol,
            exc,
        )
        return _unavailable_response(
            canonical_symbol,
            message="Long-term outlook snapshot is temporarily unavailable.",
        )
