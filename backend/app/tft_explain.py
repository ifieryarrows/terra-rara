"""Build the per-forecast TFT-ASRO explanation payload stored as ``explain_json``.

The payload enriches the raw interpretation produced by
:func:`deep_learning.models.tft_copper.interpret_batch` (VSN variable-selection
weights plus encoder attention) with human-readable feature labels, five-session
recent deltas and window dates. Everything returned is plain JSON data
(floats/strings/``None``), safe for SQLAlchemy ``JSON`` columns and for the
``/api/explain/{symbol}`` endpoint.

Building is best-effort: hot-path callers must use
:func:`try_build_explain_payload`, which never raises. An explanation is
optional and must never fail a pipeline run.
"""

import hashlib
import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Optional, Sequence

import pandas as pd

from app.features import describe_feature
from app.price_utils import finite_number

logger = logging.getLogger(__name__)

EXPLAIN_SCHEMA_VERSION = 1
TOP_RECENT = 5
RECENT_BASIS_SESSIONS = 5
# Hard cap on serialized encoder entries. A full VSN list (mrmr_top_k=80 real
# features with label+description+category metadata) serializes to roughly
# 20-40 KB per snapshot row per day — stored in Postgres, returned by every
# /api/explain poll, and injected into the DOM via the "Raw JSON" details.
# The panel renders only the top five rows, so 25 caps the stored artifact
# while keeping meaningful context.
TOP_ENCODER = 25

_HORIZON_LABEL = "5-day cumulative log return (primary)"
_HORIZON_DAYS = 5
_TOP_DAYS = 5

# Features whose five-session change reads naturally as a percentage.
_LEVEL_TOKENS = frozenset(
    {
        "price",
        "sma",
        "ema",
        "close",
        "open",
        "high",
        "low",
        "stock",
        "tonnes",
        "inventory",
        "warehouse",
        "volume",
    }
)

_NOTES = (
    "Model focus only: variable-selection (VSN) weights and attention weights show where the "
    "TFT-ASRO model looked inside the latest encoder window. They are raw, pre-calibration "
    "signals and are not causal: do not read them as market drivers or investment advice."
)


def _json_date(value: Any) -> Optional[str]:
    """Normalize a date-like value to an ISO string; other values pass through ``str``."""
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _json_text(value: Any) -> Optional[str]:
    """Normalize an identifier value to ``str`` (JSON-safe)."""
    if value is None:
        return None
    return str(value)


def _sanitize_weight(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float rounded to 6 dp, or ``None`` when unusable."""
    weight = finite_number(value)
    if weight is None:
        return None
    return round(weight, 6)


def _layout_hash(feature_names: Sequence[str]) -> str:
    """12-char sha1 fingerprint of a model group's feature layout."""
    joined = "\n".join(str(name) for name in feature_names)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def _recent_kind(feature: str, category: str) -> Optional[str]:
    """Return ``"pct"``/``"abs"`` for a recent-delta block, or ``None`` when none applies.

    Embedding features have no meaningful signed delta; level-like features
    (prices, averages, physical stock levels, volume) read best as percentage
    changes; everything else uses absolute changes.
    """
    lowered = (feature or "").lower()
    if category == "Embedding" or "emb_" in lowered:
        return None
    tokens = re.split(r"[^a-z0-9]+", lowered)
    if any(token in _LEVEL_TOKENS for token in tokens):
        return "pct"
    return "abs"


def _recent_block(
    feature: str,
    category: str,
    feature_frame: pd.DataFrame,
    window_dates: Sequence[Optional[str]],
) -> Optional[dict]:
    """Five-session delta block for an encoder feature, or ``None`` when it cannot be built.

    Compares the first and last finite values within the tail ``RECENT_BASIS_SESSIONS``
    rows of ``feature_frame``; the block is skipped for embedding features, for
    features missing from the frame, and whenever fewer than two finite values are
    available. ``change`` is always the absolute first→last delta; ``change_pct``
    holds the percentage delta for ``pct`` features and is ``None`` otherwise.
    """
    kind = _recent_kind(feature, category)
    if kind is None or not feature or feature not in feature_frame.columns:
        return None

    tail = feature_frame[feature].tail(RECENT_BASIS_SESSIONS)
    offset = len(feature_frame) - len(tail)
    finite_values: list[tuple[int, float]] = []
    for position, raw in enumerate(tail.tolist()):
        value = finite_number(raw)
        if value is not None:
            finite_values.append((offset + position, value))
    if len(finite_values) < 2:
        return None

    last_position, last_value = finite_values[-1]
    first_position, first_value = finite_values[0]
    change = finite_number(last_value - first_value)
    if change is None:
        return None

    change_pct: Optional[float] = None
    if kind == "pct" and first_value != 0.0:
        raw_pct = finite_number(change / abs(first_value) * 100.0)
        if raw_pct is None:
            return None
        change_pct = round(raw_pct, 6)

    # Label the actual observed span, not the nominal window: when the tail rows
    # contain missing values the comparison may span fewer sessions, and the UI
    # prints this label verbatim.
    basis = f"{last_position - first_position + 1} sessions"

    return {
        "basis": basis,
        "kind": kind,
        "last_value": round(last_value, 6),
        "change": round(change, 6),
        "change_pct": change_pct,
        "as_of": window_dates[last_position],
    }


def _entry(feature: str, weight: float) -> dict:
    """One variable-selection entry enriched with ``describe_feature`` metadata."""
    description = describe_feature(feature) or {}
    return {
        "feature": feature,
        "weight": weight,
        "label": description.get("label", feature),
        "description": description.get("description", ""),
        "category": description.get("category", "Other"),
        "time_horizon": description.get("time_horizon", ""),
    }


def _group_entries(group: Sequence[dict]) -> list[dict]:
    """Convert one interpretation group, dropping non-finite weights."""
    entries: list[dict] = []
    for item in group:
        feature = _json_text(item.get("feature")) or ""
        weight = _sanitize_weight(item.get("weight"))
        if weight is None:
            continue
        entries.append(_entry(feature, weight))
    return entries


def _layout_names(group: Sequence[dict]) -> list[str]:
    """Feature names of one interpretation group, in model order."""
    return [_json_text(item.get("feature")) or "" for item in group]


def build_explain_payload(
    *,
    interpretation: dict,
    window_dates: Sequence[Any],
    feature_frame: pd.DataFrame,
    reference_price_date: Any,
    symbol: str,
    checkpoint: Any,
    trained_at: Any = None,
    run_id: Any = None,
    generated_at: Any = None,
    attention_pred_horizon: int = 0,
) -> dict:
    """Assemble the ``explain_json`` payload for one TFT-ASRO forecast.

    ``interpretation`` is an :func:`interpret_batch` result; ``window_dates`` must
    carry one date label per attention value and ``feature_frame`` must have the
    same number of rows (positional alignment). Encoder entries are sorted by
    descending weight and capped at ``TOP_ENCODER``; decoder/static entries keep
    model order. Only the top ``TOP_RECENT`` encoder entries receive a
    five-session ``recent`` block.

    Raises ``ValueError`` on length mismatches; any other malformed input raises
    the underlying error. Callers on the prediction path should prefer
    :func:`try_build_explain_payload`.
    """
    attention_raw = list(interpretation["attention"])
    dates = [_json_date(value) for value in window_dates]

    if len(dates) != len(attention_raw):
        raise ValueError(
            f"window_dates length ({len(dates)}) must match "
            f"attention length ({len(attention_raw)})"
        )
    if len(feature_frame) != len(dates):
        raise ValueError(
            f"feature_frame length ({len(feature_frame)}) must match "
            f"window_dates length ({len(dates)})"
        )

    encoder_entries = _group_entries(interpretation["encoder"])
    encoder_entries.sort(key=lambda entry: entry["weight"], reverse=True)
    for entry in encoder_entries[:TOP_RECENT]:
        recent = _recent_block(entry["feature"], entry["category"], feature_frame, dates)
        if recent is not None:
            entry["recent"] = recent
    # Cap the emitted list only after the recent blocks are attached: they belong
    # to the top five, which always survives the cap.
    encoder_entries = encoder_entries[:TOP_ENCODER]
    decoder_entries = _group_entries(interpretation["decoder"])
    static_entries = _group_entries(interpretation["static"])

    series = []
    for date_value, raw_weight in zip(dates, attention_raw):
        weight = _sanitize_weight(raw_weight)
        series.append({"date": date_value, "weight": 0.0 if weight is None else weight})
    top_days = [
        {"date": entry["date"], "weight": entry["weight"]}
        for entry in sorted(series, key=lambda entry: entry["weight"], reverse=True)[:_TOP_DAYS]
    ]

    if generated_at is None:
        generated_at = datetime.now(timezone.utc)

    return {
        "schema_version": EXPLAIN_SCHEMA_VERSION,
        "symbol": _json_text(symbol),
        "reference_price_date": _json_date(reference_price_date),
        "generated_at": _json_date(generated_at),
        "run_id": _json_text(run_id),
        "model": {
            "checkpoint": _json_text(checkpoint),
            "trained_at": _json_date(trained_at),
        },
        "horizon": {"label": _HORIZON_LABEL, "days": _HORIZON_DAYS},
        "layout_hash": {
            "encoder": _layout_hash(_layout_names(interpretation["encoder"])),
            "decoder": _layout_hash(_layout_names(interpretation["decoder"])),
            "static": _layout_hash(_layout_names(interpretation["static"])),
        },
        "variable_selection": {
            "method": "vsn",
            "encoder": encoder_entries,
            "decoder": decoder_entries,
            "static": static_entries,
        },
        "attention": {
            "method": "tft_attention",
            "step": {
                "kind": "first_forecast_day",
                "horizon_index": int(attention_pred_horizon),
            },
            "window": {
                "start": dates[0] if dates else None,
                "end": dates[-1] if dates else None,
                "days": len(dates),
            },
            "top_days": top_days,
            "series": series,
        },
        "notes": _NOTES,
    }


def try_build_explain_payload(**kwargs: Any) -> Optional[dict]:
    """Best-effort wrapper around :func:`build_explain_payload`.

    Never raises: any failure is logged as a warning and yields ``None`` so an
    explanation can never fail the prediction pipeline.
    """
    try:
        return build_explain_payload(**kwargs)
    except Exception as exc:
        logger.warning("TFT explain payload build failed; skipping explanation: %s", exc)
        return None
